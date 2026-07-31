"""Subtracting the polyolefin matrix to expose the trace fractions.

This is the centrepiece of Milestone 2. In a mixed-polyolefin recyclate the
alkane / alkene / diene comb accounts for the overwhelming majority of the
signal, and it buries everything that matters: a 0.5 % PET contamination sits
three orders of magnitude below it. Removing the comb is what makes the rest of
the analysis possible — measured on the benchmark, it drops the number of
components per co-eluting window from a median of 7 (maximum 12) to a median of
3, and empties a third of the windows entirely.

Why the model is built from the data rather than from a spectral library
------------------------------------------------------------------------
A template-based subtraction would need tabulated alkane and alkene spectra. That
would be circular here — the benchmark's own spectra come from such a table — and
brittle in the field, where the real spectra depend on tuning, source condition
and column. So nothing is assumed about *what* the comb looks like. What is
assumed is its **structure**, which is physics rather than identification:

* it appears as a regular sequence of clusters, one per carbon number;
* the spectra of neighbouring clusters change only gradually, because adding one
  CH₂ barely changes the fragmentation pattern.

The second property does double duty. It is how a *contaminated* cluster is
recognised: where styrene co-elutes with the C8 cluster, that cluster's measured
spectrum stops resembling its neighbours'. Those positions get a spectrum
interpolated from their clean neighbours instead, so the fit models the comb only
and leaves the styrene in the residual. Without that step, subtraction would
remove the very analytes it exists to reveal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls
from scipy.signal import find_peaks

from data_schemas.pyrogram import PreprocessingStep
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import PyRecycleError

__all__ = [
    "MATRIX_INDICATOR_IONS",
    "CombDetection",
    "PolyolefinMatrixModel",
    "MatrixSubtractionResult",
    "MatrixSubtractionError",
    "detect_homologue_comb",
    "build_matrix_model",
    "subtract_polymer_matrix",
]


class MatrixSubtractionError(PyRecycleError):
    """The polyolefin matrix could not be modelled."""


MATRIX_INDICATOR_IONS: dict[str, tuple[float, ...]] = {
    "PE_PP_Backbone": (41.0, 43.0, 55.0, 57.0, 69.0, 71.0, 83.0, 85.0),
    "PE_Backbone": (43.0, 57.0, 71.0, 85.0, 41.0, 55.0),
    "PP_Backbone": (41.0, 55.0, 56.0, 69.0, 70.0, 83.0, 84.0),
}
"""Ion sets whose summed chromatogram tracks a given polyolefin backbone.

These are the alkyl (``14k+1``) and alkenyl (``14k-1``) fragment series that every
saturated and unsaturated aliphatic chain produces. They are used only to *locate*
the comb — the spectra that get subtracted are measured from the data.
"""


@dataclass(frozen=True, slots=True)
class CombDetection:
    """Where the homologous series sits in the run.

    Attributes:
        apex_indices: Scan index of each detected cluster apex.
        apex_times_s: Retention time of each apex.
        boundaries: Scan indices splitting the run between clusters, length
            ``n_clusters + 1``.
        series_index: Position within the series, ``0`` for the first detected
            cluster. This is a relative index; converting it to an absolute carbon
            number needs an external anchor, which Milestone 3 supplies.
        spacing_s: Median retention gap between neighbouring clusters.
        contaminated: True where the cluster's measured spectrum departs from its
            neighbours', indicating a co-eluting foreign compound.
    """

    apex_indices: np.ndarray
    apex_times_s: np.ndarray
    boundaries: np.ndarray
    series_index: np.ndarray
    spacing_s: float
    contaminated: np.ndarray

    @property
    def n_clusters(self) -> int:
        return int(self.apex_indices.size)

    @property
    def n_contaminated(self) -> int:
        return int(np.count_nonzero(self.contaminated))


@dataclass(slots=True)
class PolyolefinMatrixModel:
    """A fitted model of the polyolefin comb.

    Attributes:
        profiles: Elution profile of each cluster, shape ``(n_scans, n_clusters)``,
            each normalised to unit area.
        spectra: Spectrum of each cluster, shape ``(n_clusters, n_mz)``, rows
            summing to one, with contaminated positions repaired.
        amplitudes: Fitted area of each cluster, shape ``(n_clusters,)``.
        detection: The underlying comb detection.
    """

    profiles: np.ndarray
    spectra: np.ndarray
    amplitudes: np.ndarray
    detection: CombDetection

    @property
    def n_clusters(self) -> int:
        return int(self.amplitudes.size)

    def reconstruct(self) -> np.ndarray:
        """The modelled matrix contribution, shape ``(n_scans, n_mz)``."""
        return (self.profiles * self.amplitudes[None, :]) @ self.spectra


@dataclass(slots=True)
class MatrixSubtractionResult:
    """What is left after the polyolefin backbone has been removed.

    Attributes:
        residual: The cube with the matrix subtracted — the input to curve
            resolution.
        matrix: The subtracted contribution, kept for inspection and for the
            chain-length statistics Milestone 3 builds on.
        model: The fitted comb model.
        explained_fraction: Share of the original total signal attributed to the
            matrix.
        diagnostics: Cluster count, contaminated positions, residual structure.
    """

    residual: PyrogramDataCube
    matrix: PyrogramDataCube
    model: PolyolefinMatrixModel
    explained_fraction: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _indicator_chromatogram(
    cube: PyrogramDataCube, ions: tuple[float, ...]
) -> np.ndarray:
    """Summed chromatogram of the ions that track the aliphatic backbone."""
    available = [ion for ion in ions if np.min(np.abs(cube.mz_axis - ion)) <= 0.5]
    if not available:
        raise MatrixSubtractionError(
            f"none of the indicator ions {ions} lie inside the acquired m/z range "
            f"[{cube.mz_axis[0]:.0f}, {cube.mz_axis[-1]:.0f}]"
        )
    return cube.eic_sum(available)


def detect_homologue_comb(
    cube: PyrogramDataCube,
    *,
    matrix_type: str = "PE_PP_Backbone",
    min_prominence_fraction: float = 0.02,
    spacing_tolerance: float = 0.45,
    max_spacing_variation: float = 0.12,
    min_clusters: int = 5,
    sub_clusters: int = 3,
    contamination_excess_sigmas: float = 4.0,
) -> CombDetection:
    """Locate the homologous-series clusters and flag the contaminated ones.

    Args:
        cube: Baseline-corrected pyrogram.
        matrix_type: Key into :data:`MATRIX_INDICATOR_IONS`.
        min_prominence_fraction: Peak prominence threshold on the indicator
            chromatogram, relative to its maximum.
        spacing_tolerance: How far a gap may deviate from the median spacing, as a
            fraction, before it is treated as a break in the series rather than a
            member of it.
        max_spacing_variation: Maximum coefficient of variation of the retention
            gaps. A real homologous series is highly regular; anything less
            regular is not one, and modelling it as one would subtract genuine
            analyte signal.
        min_clusters: Minimum number of members. A polyolefin comb spans many
            carbon numbers, so a handful of peaks is not enough evidence.
        sub_clusters: Segments per cluster, each modelled with its own spectrum.
            A cluster is a fused diene/alkene/alkane triplet whose spectrum
            changes across it; one averaged spectrum leaves a remnant that buries
            small markers.
        contamination_excess_sigmas: How far above the neighbouring clusters a
            channel must stand to be treated as a co-eluting foreign compound.

    Returns:
        The detected comb.

    Raises:
        MatrixSubtractionError: If the indicator ions are outside the acquired
            range, or fewer than three clusters are found — too few to establish
            that a regular series exists at all.
    """
    if matrix_type not in MATRIX_INDICATOR_IONS:
        raise MatrixSubtractionError(
            f"unknown matrix_type {matrix_type!r}; available: "
            f"{', '.join(sorted(MATRIX_INDICATOR_IONS))}"
        )

    indicator = _indicator_chromatogram(cube, MATRIX_INDICATOR_IONS[matrix_type])
    if indicator.max() <= 0.0:
        raise MatrixSubtractionError("indicator chromatogram is empty")

    apexes, _ = find_peaks(
        indicator, prominence=min_prominence_fraction * float(indicator.max())
    )
    if apexes.size < min_clusters:
        raise MatrixSubtractionError(
            f"found only {apexes.size} candidate cluster(s); at least "
            f"{min_clusters} are needed to establish a homologous series"
        )

    apex_times = cube.retention_times[apexes]
    gaps = np.diff(apex_times)
    spacing = float(np.median(gaps))

    # Keep the longest run of clusters whose spacing is consistent. A stray peak
    # from a large non-matrix compound would otherwise be adopted into the series
    # and its spectrum subtracted from the data.
    consistent = np.abs(gaps - spacing) <= spacing_tolerance * spacing
    runs: list[list[int]] = [[0]]
    for position, is_consistent in enumerate(consistent, start=1):
        if is_consistent:
            runs[-1].append(position)
        else:
            runs.append([position])
    longest = max(runs, key=len)

    apexes = apexes[longest]
    apex_times = apex_times[longest]

    # A homologous series is regular by construction: each additional CH2 adds
    # almost the same retention increment. Measured on the benchmark, a genuine
    # comb has a gap coefficient of variation around 0.05, while the peak sequence
    # a pure-polystyrene pyrogram offers up sits at 0.15-0.24. Without this guard
    # the detector happily models any three roughly-spaced peaks as a comb and
    # subtracts real analyte signal from a sample that has no matrix at all.
    final_gaps = np.diff(apex_times)
    variation = (
        float(np.std(final_gaps) / np.mean(final_gaps)) if final_gaps.size else 1.0
    )
    if variation > max_spacing_variation:
        raise MatrixSubtractionError(
            f"cluster spacing varies by {variation:.0%} (limit "
            f"{max_spacing_variation:.0%}); the peak sequence is not a homologous "
            "series, so this sample is not polyolefin-dominated"
        )
    if len(longest) < min_clusters:
        raise MatrixSubtractionError(
            f"only {len(longest)} regularly spaced clusters found, {min_clusters} "
            "required; the sample does not look polyolefin-dominated"
        )

    midpoints = ((apexes[:-1] + apexes[1:]) // 2).astype(int)
    half_span = int(np.median(np.diff(apexes)) // 2) if apexes.size > 1 else 1
    boundaries = np.concatenate(
        (
            [max(int(apexes[0]) - half_span, 0)],
            midpoints,
            [min(int(apexes[-1]) + half_span, cube.n_scans)],
        )
    ).astype(int)

    segment_edges = _segment_edges(boundaries, sub_clusters)
    segment_spectra = _cluster_spectra(cube, segment_edges)
    _, segment_flags = _repair_spectra(
        segment_spectra, stride=sub_clusters, excess_sigmas=contamination_excess_sigmas
    )
    # Report contamination per cluster: a cluster counts as contaminated when any
    # of its segments does.
    contaminated = np.zeros(boundaries.size - 1, dtype=bool)
    for position, flagged in enumerate(segment_flags):
        cluster = min(position // sub_clusters, contaminated.size - 1)
        contaminated[cluster] |= bool(flagged)

    return CombDetection(
        apex_indices=apexes.astype(int),
        apex_times_s=apex_times,
        boundaries=boundaries,
        series_index=np.arange(apexes.size),
        spacing_s=spacing,
        contaminated=contaminated,
    )


def _segment_edges(boundaries: np.ndarray, sub_clusters: int) -> np.ndarray:
    """Split every cluster into ``sub_clusters`` equal slices along the scan axis.

    One spectrum per cluster is not enough. A cluster is a fused triplet — the
    alkadiene elutes first, then the 1-alkene, then the n-alkane — and those three
    have visibly different spectra. Describing the whole cluster with a single
    averaged spectrum leaves systematic structure behind, and that remnant is what
    swamps the small markers: measured before this change, the recovered
    caprolactam spectrum carried m/z 43 and 57 from leftover comb and reached a
    similarity of only 0.71, while naphthalene disappeared into the remnant
    entirely.
    """
    edges: list[int] = []
    for index in range(boundaries.size - 1):
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        if stop - start < sub_clusters:
            edges.append(start)
            continue
        edges.extend(
            int(round(position))
            for position in np.linspace(start, stop, sub_clusters + 1)[:-1]
        )
    edges.append(int(boundaries[-1]))
    return np.unique(np.asarray(edges, dtype=int))


def _cluster_spectra(cube: PyrogramDataCube, boundaries: np.ndarray) -> np.ndarray:
    """Area-summed spectrum of each segment, rows normalised to sum one."""
    n_segments = boundaries.size - 1
    spectra = np.zeros((n_segments, cube.n_mz), dtype=np.float64)
    for index in range(n_segments):
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        if stop <= start:
            continue
        summed = cube.intensities[start:stop, :].sum(axis=0)
        total = summed.sum()
        if total > 0.0:
            spectra[index, :] = summed / total
    return spectra


def _repair_spectra(
    spectra: np.ndarray,
    *,
    stride: int = 1,
    neighbourhood: int = 2,
    excess_sigmas: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove foreign contributions from the measured cluster spectra, channel by channel.

    Detection is **per channel**, not per spectrum, and the difference is not a
    detail. Comparing whole spectra by cosine similarity was tried first and
    measured: it flagged only one contaminated cluster in the mixed-polyolefin
    benchmark and let the subtraction destroy 88 % of the caprolactam signal. The
    reason is a matter of scale. Caprolactam co-elutes with the C14 cluster at
    1.5 % PA6 content, so it barely moves the cluster's overall spectral shape —
    the cosine stays well above any sensible threshold. But the cluster's fitted
    amplitude is huge, so even a tiny relative contribution at m/z 113 becomes a
    large absolute one, and subtracting it cancels the marker almost exactly.

    Working per channel catches precisely that: m/z 113 stands out sharply against
    the neighbouring clusters' values for the same channel, even though the
    spectra as a whole look alike. Only channels in *excess* of the neighbourhood
    are repaired — a deficit is ordinary variation along the series.

    Args:
        spectra: Measured segment spectra, shape ``(n_segments, n_mz)``.
        stride: Number of segments per cluster. Comparison runs along the carbon
            axis at constant position within the cluster — segment ``j`` of one
            cluster against segment ``j`` of its neighbours. Comparing adjacent
            *segments* would be wrong: within one cluster they are the diene, the
            alkene and the alkane, which genuinely differ, and every one of them
            would be flagged as contaminated.
        neighbourhood: How many clusters on each side form the reference.
        excess_sigmas: How far above the neighbourhood's robust spread a channel
            must lie to count as foreign.

    Returns:
        ``(repaired_spectra, contaminated_mask)``.
    """
    repaired = spectra.copy()
    n_segments = spectra.shape[0]
    contaminated = np.zeros(n_segments, dtype=bool)
    stride = max(int(stride), 1)
    if n_segments < 3 * stride:
        return repaired, contaminated

    for index in range(n_segments):
        neighbours = [
            index + offset * stride
            for offset in range(-neighbourhood, neighbourhood + 1)
            if offset != 0 and 0 <= index + offset * stride < n_segments
        ]
        if len(neighbours) < 2:
            continue

        block = spectra[neighbours]
        expected = np.median(block, axis=0)
        spread = 1.4826 * np.median(np.abs(block - expected), axis=0)
        # A channel that is constant across the neighbourhood has zero spread; a
        # floor keeps it from flagging every rounding difference as foreign.
        floor = np.maximum(spread, 1e-4 * float(expected.max()))

        foreign = (spectra[index] - expected) > excess_sigmas * floor
        if np.any(foreign):
            contaminated[index] = True
            repaired[index, foreign] = expected[foreign]

        total = repaired[index].sum()
        if total > 0.0:
            repaired[index, :] /= total

    return repaired, contaminated


def _cluster_profiles(
    indicator: np.ndarray, boundaries: np.ndarray, retention_times: np.ndarray
) -> np.ndarray:
    """Empirical elution profile of each cluster, normalised to unit area.

    Taken from the indicator chromatogram rather than fitted to a peak shape: the
    cluster is a fused alkane/alkene/diene triplet whose shape no single
    exponentially modified Gaussian describes well, and its true shape is right
    there in the data.
    """
    n_clusters = boundaries.size - 1
    profiles = np.zeros((retention_times.size, n_clusters), dtype=np.float64)
    for index in range(n_clusters):
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        if stop <= start:
            continue
        profiles[start:stop, index] = np.clip(indicator[start:stop], 0.0, None)
        # Über die volle Zeitachse normieren, nicht nur über das Segment: das
        # Profil ist außerhalb null, und die Trapezregel zählt an den
        # Segmenträndern halbe Trapeze gegen null mit. Bei schmalen Segmenten war
        # der Unterschied bis zu 20 %.
        area = float(np.trapezoid(profiles[:, index], retention_times))
        if area > 0.0:
            profiles[:, index] /= area
        else:
            profiles[:, index] = 0.0
    return profiles


def _fit_amplitudes(
    data: np.ndarray,
    profiles: np.ndarray,
    spectra: np.ndarray,
    *,
    ridge: float = 1e-9,
) -> np.ndarray:
    """Non-negative least squares for the per-cluster amplitudes.

    The design matrix of the full problem would be ``(n_scans * n_mz) × n_clusters``
    — for a production run that is 1.3 million rows, several hundred megabytes.
    It is never formed. The normal equations have a separable structure,

        ``(MᵀM)[j,k] = (cⱼ·c_k)(sⱼ·s_k)``   and   ``(Mᵀd)[k] = c_kᵀ D s_k``,

    so both sides collapse to products of small matrices. A Cholesky factor of the
    resulting Gram matrix turns the normal equations back into a least-squares
    problem that :func:`scipy.optimize.nnls` can solve exactly.
    """
    profile_gram = profiles.T @ profiles
    spectral_gram = spectra @ spectra.T
    gram = profile_gram * spectral_gram
    target = np.einsum("sk,sm,km->k", profiles, data, spectra, optimize=True)

    scale = float(np.trace(gram)) / max(gram.shape[0], 1)
    gram = gram + ridge * max(scale, 1.0) * np.eye(gram.shape[0])

    try:
        factor = cho_factor(gram, lower=True)
    except np.linalg.LinAlgError as error:  # pragma: no cover - defensive
        raise MatrixSubtractionError(
            "the cluster model is numerically singular; clusters may be duplicated"
        ) from error

    lower = np.tril(factor[0]) if factor[1] else np.triu(factor[0]).T
    whitened = cho_solve(factor, target)
    amplitudes, _ = nnls(lower.T, lower.T @ whitened)
    return amplitudes


def build_matrix_model(
    cube: PyrogramDataCube,
    detection: CombDetection,
    *,
    matrix_type: str = "PE_PP_Backbone",
    sub_clusters: int = 3,
    repair_contaminated: bool = True,
) -> PolyolefinMatrixModel:
    """Fit the comb model to a pyrogram.

    Args:
        cube: Baseline-corrected pyrogram.
        detection: Output of :func:`detect_homologue_comb`.
        matrix_type: Indicator ion set, used for the profile shapes.
        sub_clusters: Segments per cluster; must match the value used for detection.
        repair_contaminated: Replace contaminated cluster spectra with
            interpolated ones. Switching this off subtracts the co-eluting
            analytes along with the matrix and is only useful for demonstrating
            that failure.

    Returns:
        The fitted model.
    """
    indicator = _indicator_chromatogram(cube, MATRIX_INDICATOR_IONS[matrix_type])
    edges = _segment_edges(detection.boundaries, sub_clusters)
    profiles = _cluster_profiles(indicator, edges, cube.retention_times)
    spectra = _cluster_spectra(cube, edges)
    if repair_contaminated:
        spectra, _ = _repair_spectra(spectra, stride=sub_clusters)

    amplitudes = _fit_amplitudes(cube.intensities, profiles, spectra)
    return PolyolefinMatrixModel(
        profiles=profiles,
        spectra=spectra,
        amplitudes=amplitudes,
        detection=detection,
    )


def subtract_polymer_matrix(
    cube: PyrogramDataCube,
    matrix_type: str = "PE_PP_Backbone",
    *,
    min_prominence_fraction: float = 0.02,
    max_spacing_variation: float = 0.12,
    min_clusters: int = 5,
    sub_clusters: int = 3,
    contamination_excess_sigmas: float = 4.0,
    repair_contaminated: bool = True,
    clip_negative: bool = True,
) -> MatrixSubtractionResult:
    """Remove the polyolefin backbone, leaving the trace fractions behind.

    Args:
        cube: Baseline-corrected pyrogram. Running this on uncorrected data will
            fold the column bleed into the matrix model.
        matrix_type: Which backbone to model — ``"PE_PP_Backbone"`` (default),
            ``"PE_Backbone"`` or ``"PP_Backbone"``.
        min_prominence_fraction: Cluster detection threshold on the indicator
            chromatogram.
        max_spacing_variation: Regularity requirement on the series; see
            :func:`detect_homologue_comb`.
        min_clusters: Minimum number of series members required.
        sub_clusters: Segments per cluster, each with its own measured spectrum.
        contamination_excess_sigmas: How far above the neighbouring clusters a
            channel must stand to be treated as a co-eluting foreign compound.
        repair_contaminated: Interpolate contaminated cluster spectra from clean
            neighbours, so that co-eluting analytes stay in the residual.
        clip_negative: Clamp negative residuals to zero. Recommended, since ion
            counts cannot be negative and MCR-ALS imposes non-negativity anyway.

    Returns:
        The residual, the subtracted matrix, the fitted model and diagnostics.

    Raises:
        MatrixSubtractionError: If no regular homologous series can be found, or
            the requested ``matrix_type`` is unknown.
    """
    detection = detect_homologue_comb(
        cube,
        matrix_type=matrix_type,
        min_prominence_fraction=min_prominence_fraction,
        max_spacing_variation=max_spacing_variation,
        min_clusters=min_clusters,
        sub_clusters=sub_clusters,
        contamination_excess_sigmas=contamination_excess_sigmas,
    )
    model = build_matrix_model(
        cube,
        detection,
        matrix_type=matrix_type,
        sub_clusters=sub_clusters,
        repair_contaminated=repair_contaminated,
    )

    modelled = model.reconstruct()
    residual_intensities = cube.intensities - modelled
    if clip_negative:
        residual_intensities = np.clip(residual_intensities, 0.0, None)

    total = cube.total_signal
    explained = float(modelled.sum() / total) if total > 0.0 else 0.0

    step = PreprocessingStep(
        name=f"matrix_subtraction:{matrix_type}",
        parameters={
            "n_clusters": detection.n_clusters,
            "n_contaminated": detection.n_contaminated,
            "explained_fraction": round(explained, 6),
            "repair_contaminated": repair_contaminated,
        },
        note="data-driven polyolefin comb model; spectra measured, not tabulated",
    )

    residual = cube.with_intensities(residual_intensities, step)
    matrix_cube = cube.with_intensities(
        np.clip(modelled, 0.0, None),
        PreprocessingStep(name=f"matrix_model:{matrix_type}"),
    )

    return MatrixSubtractionResult(
        residual=residual,
        matrix=matrix_cube,
        model=model,
        explained_fraction=explained,
        diagnostics={
            "n_clusters": detection.n_clusters,
            "n_contaminated": detection.n_contaminated,
            "cluster_spacing_s": detection.spacing_s,
            "contaminated_times_s": detection.apex_times_s[
                detection.contaminated
            ].tolist(),
            "residual_fraction": (
                float(residual.total_signal / total) if total > 0.0 else 0.0
            ),
        },
    )
