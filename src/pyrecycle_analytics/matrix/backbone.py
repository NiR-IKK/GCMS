"""Splitting the subtracted polyolefin comb between PE and PP.

Why this exists
---------------
Polyethylene and polypropylene produce *the same kind* of comb: a regular series
of aliphatic clusters, one per carbon number. The matrix model removes that comb
as one object, which is what makes the trace analysis possible — but it also
means a mixed polyolefin recyclate loses both of its main fractions in a single
step, and the passport then has to report the comb as one polymer. Measured on
the benchmark, that put polypropylene at 5.5 % where the truth was 31 %.

What actually separates them
----------------------------
Not retention: the two combs interleave. What separates them is the spectrum.
Propylene units put a methyl branch on every third carbon, so PP pyrolysate
fragments into the ``14k`` series (m/z 56, 70, 84) far more than any polyethylene,
which favours the alkyl ``14k+1`` and alkenyl ``14k-1`` series. Amplitude-weighted
over the whole comb, that difference is large and stable.

Why it needs measured references
--------------------------------
Two approaches were measured against the benchmark's known blends:

* **Fragment-series masks** — synthetic endmembers built from the ``14k`` and
  ``14k±1`` series alone, assuming nothing about real spectra. Measured: 10 % PP
  in virgin HDPE (truth 0 %) and 35 % PP in virgin polypropylene (truth 100 %).
  Unusable. Real aliphatic spectra spread their intensity over far more channels
  than the idealised series, and a mask captures only part of each.
* **Comb spectra measured from virgin references** — 0.0 %, 0.0 %, 100.0 % on the
  three virgin recipes and 35.0 % against a true 34.4 % on the mixed one.

So the split is a *calibration*, not an algorithm parameter: without virgin
reference runs of the same materials under the same method it is not attempted at
all, and the passport says the polyolefin fraction is unsplit rather than
inventing a division. That is the honest trade, and it is why
:meth:`BackboneEndmembers.from_references` takes measured cubes rather than a
table of numbers.

A per-cluster variant was also measured and **discarded**: unmixing each cluster
separately and averaging gave 13.8 % PP in virgin HDPE against 0.0 % for the
global fit. A single cluster spectrum carries too little signal, and non-negative
least squares on a noisy vector biases towards mixing. The amplitude-weighted
comb spectrum is both simpler and markedly more accurate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from data_schemas.enums import PolymerClass
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import PyRecycleError
from pyrecycle_analytics.matrix.polyolefin import (
    PolyolefinMatrixModel,
    build_matrix_model,
    detect_homologue_comb,
)

__all__ = [
    "BackboneEndmembers",
    "BackboneSplit",
    "BackboneSplitError",
    "comb_spectrum",
    "split_backbone",
]

COLLINEARITY_LIMIT = 0.995
"""Cosine above which two endmembers are treated as not independently resolvable.

Measured on the benchmark: HDPE against LDPE is 0.9990, PP against either is far
lower. The HD/LD direction therefore carries almost no independent information,
and a split along it is reported as a hint rather than as a figure.
"""

MAX_RESIDUAL_FRACTION = 0.35
"""Unexplained share of the comb spectrum above which the split is refused.

A comb the references cannot reconstruct is not a comb of those materials — a
different pyrolysis temperature, a different column, or a polyolefin grade that
is not in the reference set. Fitting anyway would produce a number with no
meaning behind it.
"""


class BackboneSplitError(PyRecycleError):
    """The polyolefin comb could not be split between the reference materials."""


def comb_spectrum(model: PolyolefinMatrixModel) -> np.ndarray:
    """Amplitude-weighted spectrum of a fitted comb, normalised to unit sum.

    Weighting by amplitude rather than averaging the cluster spectra is what makes
    this usable: the clusters differ by orders of magnitude in size, and an
    unweighted mean would let the smallest, noisiest cluster count as much as the
    apex of the distribution.

    Args:
        model: A fitted comb model.

    Returns:
        The spectrum, shape ``(n_mz,)``.

    Raises:
        BackboneSplitError: If the model carries no signal.
    """
    spectrum = (model.spectra * model.amplitudes[:, None]).sum(axis=0)
    total = float(spectrum.sum())
    if total <= 0.0:
        raise BackboneSplitError("the fitted comb carries no signal")
    return np.asarray(spectrum / total, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BackboneEndmembers:
    """Comb spectra of known polyolefins, measured under the sample's method.

    Attributes:
        spectra: One L1-normalised comb spectrum per material, shape
            ``(n_materials, n_mz)``.
        polymers: The material each row stands for, in the same order.
        mz_axis: Nominal masses the columns refer to.
        source: Where the references came from, carried into the passport so a
            reader can tell a calibrated split from a borrowed one.
    """

    spectra: np.ndarray
    polymers: tuple[PolymerClass, ...]
    mz_axis: np.ndarray
    source: str = "measured reference runs"

    def __post_init__(self) -> None:
        if self.spectra.ndim != 2:
            raise BackboneSplitError("endmember spectra must be a 2-D array")
        if self.spectra.shape[0] != len(self.polymers):
            raise BackboneSplitError(
                f"{self.spectra.shape[0]} spectra for {len(self.polymers)} polymers"
            )
        if self.spectra.shape[1] != self.mz_axis.size:
            raise BackboneSplitError("spectra and m/z axis disagree in length")
        if len(set(self.polymers)) != len(self.polymers):
            raise BackboneSplitError("each polymer may appear only once")
        if len(self.polymers) < 2:
            raise BackboneSplitError(
                "splitting needs at least two reference materials; with one there "
                "is nothing to split between"
            )

    @classmethod
    def from_references(
        cls,
        references: Mapping[PolymerClass, PyrogramDataCube],
        *,
        matrix_type: str = "PE_PP_Backbone",
        source: str = "measured reference runs",
    ) -> BackboneEndmembers:
        """Build endmembers from virgin reference pyrograms.

        The references must be measured under the same method as the samples they
        will be applied to. Pyrolysis temperature especially: the alkene-to-alkane
        ratio of a polyolefin comb shifts with it, and a reference run 50 °C away
        describes a different comb.

        Args:
            references: One virgin pyrogram per material, baseline-corrected.
            matrix_type: Indicator ion set used to locate each comb.
            source: Provenance note carried into the passport.

        Returns:
            The endmember set.

        Raises:
            BackboneSplitError: If a reference has no detectable comb, or if the
                references were acquired on different m/z axes.
        """
        polymers = tuple(references)
        axis: np.ndarray | None = None
        rows: list[np.ndarray] = []
        for polymer in polymers:
            cube = references[polymer]
            if axis is None:
                axis = np.asarray(cube.mz_axis, dtype=np.float64)
            elif not np.array_equal(axis, cube.mz_axis):
                raise BackboneSplitError(
                    f"reference {polymer} was acquired on a different m/z axis; "
                    "re-import the references with a common mz_range so the "
                    "spectra are comparable channel by channel"
                )
            try:
                model = build_matrix_model(
                    cube, detect_homologue_comb(cube, matrix_type=matrix_type)
                )
            except PyRecycleError as error:
                raise BackboneSplitError(
                    f"no polyolefin comb found in the {polymer} reference: {error}"
                ) from error
            rows.append(comb_spectrum(model))

        assert axis is not None  # guaranteed: at least one reference, checked below
        return cls(
            spectra=np.vstack(rows), polymers=polymers, mz_axis=axis, source=source
        )

    @property
    def n_materials(self) -> int:
        return len(self.polymers)

    def collinear_pairs(self) -> list[tuple[PolymerClass, PolymerClass, float]]:
        """Endmember pairs too similar to be told apart, with their cosine."""
        norms = np.linalg.norm(self.spectra, axis=1)
        pairs: list[tuple[PolymerClass, PolymerClass, float]] = []
        for i in range(self.n_materials):
            for j in range(i + 1, self.n_materials):
                denominator = float(norms[i] * norms[j])
                if denominator <= 0.0:
                    continue
                cosine = float(self.spectra[i] @ self.spectra[j] / denominator)
                if cosine >= COLLINEARITY_LIMIT:
                    pairs.append((self.polymers[i], self.polymers[j], cosine))
        return pairs


@dataclass(slots=True)
class BackboneSplit:
    """How the polyolefin comb divides between the reference materials.

    Attributes:
        shares: Signal share per material, summing to one.
        residual_fraction: Share of the comb spectrum the references failed to
            explain. Small values mean the sample's comb really is made of these
            materials; large ones mean it is not, and the split is refused.
        endmember_source: Provenance of the references.
        merged: Groups whose members could not be resolved from one another, each
            reported as its combined share and a grade hint.
        notes: Everything a reader needs to weigh the numbers.
    """

    shares: dict[PolymerClass, float]
    residual_fraction: float
    endmember_source: str
    merged: tuple[tuple[tuple[PolymerClass, ...], float], ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dominant(self) -> PolymerClass:
        """Material holding the largest share."""
        return max(self.shares, key=lambda polymer: self.shares[polymer])

    def share_of(self, polymer: PolymerClass) -> float:
        """Share of one material, zero when it is not in the reference set."""
        return float(self.shares.get(polymer, 0.0))


def _align(spectrum: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Move a spectrum onto another nominal-mass axis.

    Channels the target does not cover are dropped rather than folded into a
    neighbour: a nominal mass is a distinct fragment, and adding m/z 56 into m/z 57
    would destroy exactly the branched-versus-linear contrast the split rests on.
    """
    if np.array_equal(source, target):
        return spectrum
    aligned = np.zeros(target.size, dtype=np.float64)
    lookup = {int(round(float(mz))): index for index, mz in enumerate(target)}
    for value, mz in zip(spectrum, source, strict=True):
        index = lookup.get(int(round(float(mz))))
        if index is not None:
            aligned[index] += float(value)
    return aligned


def split_backbone(
    model: PolyolefinMatrixModel,
    endmembers: BackboneEndmembers,
    mz_axis: np.ndarray,
    *,
    max_residual_fraction: float = MAX_RESIDUAL_FRACTION,
) -> BackboneSplit:
    """Divide a fitted comb between the reference materials.

    Solves ``min ‖s − Eᵀw‖`` for ``w ≥ 0``, where ``s`` is the sample's
    amplitude-weighted comb spectrum and ``E`` the endmember spectra. Because both
    are normalised to unit sum, the weights *are* signal shares — no response
    factor enters here, and none should: this divides the comb's signal, and the
    conversion to mass shares happens later, once, in the passport.

    Args:
        model: The fitted comb of the sample.
        endmembers: Reference comb spectra.
        mz_axis: Nominal masses of the sample's spectra.
        max_residual_fraction: Refuse the split above this unexplained share.

    Returns:
        The split, with its unexplained share and any unresolvable groups.

    Raises:
        BackboneSplitError: If the references explain too little of the comb, or
            if their m/z axes do not overlap the sample's.
    """
    sample = comb_spectrum(model)
    sample_axis = np.asarray(mz_axis, dtype=np.float64)
    basis = np.column_stack(
        [
            _align(row, endmembers.mz_axis, sample_axis)
            for row in endmembers.spectra
        ]
    )

    covered = basis.sum(axis=0)
    if np.any(covered <= 0.0):
        missing = [
            str(polymer)
            for polymer, total in zip(endmembers.polymers, covered, strict=True)
            if total <= 0.0
        ]
        raise BackboneSplitError(
            f"the reference spectra for {', '.join(missing)} share no m/z channel "
            "with the sample; the runs were acquired over different mass ranges"
        )

    weights, _ = nnls(basis, sample)
    total = float(weights.sum())
    if total <= 0.0:
        raise BackboneSplitError(
            "no non-negative combination of the references describes this comb"
        )

    residual = float(np.abs(sample - basis @ weights).sum())
    if residual > max_residual_fraction:
        raise BackboneSplitError(
            f"the references explain only {1.0 - residual:.0%} of the comb "
            f"(limit {1.0 - max_residual_fraction:.0%}); they were probably "
            "measured under a different method, or the sample contains a "
            "polyolefin that is not among them"
        )

    shares = {
        polymer: float(weight / total)
        for polymer, weight in zip(endmembers.polymers, weights, strict=True)
    }

    notes: list[str] = [
        f"polyolefin split against {endmembers.n_materials} reference materials "
        f"({endmembers.source}); {residual:.1%} of the comb spectrum unexplained"
    ]
    merged: list[tuple[tuple[PolymerClass, ...], float]] = []
    for first, second, cosine in endmembers.collinear_pairs():
        merged.append(
            ((first, second), shares.get(first, 0.0) + shares.get(second, 0.0))
        )
        notes.append(
            f"{first} and {second} have a spectral cosine of {cosine:.4f} and "
            "cannot be resolved from one another; their combined share is the "
            "firm figure and the division between them is a grade hint"
        )

    return BackboneSplit(
        shares=shares,
        residual_fraction=residual,
        endmember_source=endmembers.source,
        merged=tuple(merged),
        notes=tuple(notes),
    )
