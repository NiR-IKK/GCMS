"""The end-to-end deconvolution chain.

Ties Milestone 2 together in the order the benchmark measurement showed to be
necessary: preprocess, **subtract the polyolefin matrix**, cut into windows,
estimate each window's rank, resolve it, and stitch the per-window
factorisations back into one run-wide result.

The matrix subtraction comes before the curve resolution, not after. Measured on
the benchmark, it drops the number of components per co-eluting window from a
median of 7 (maximum 12) to a median of 3 and empties a third of the windows
outright. Running MCR-ALS on the unsubtracted data means asking it to separate a
dozen components whose spectra have cosine similarities above 0.98 — a problem
that constrained curve resolution cannot solve and should not be handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.deconvolution.mcrals import McrAlsOptions, mcr_als
from pyrecycle_analytics.deconvolution.rank import estimate_rank
from pyrecycle_analytics.deconvolution.result import ResolutionError, ResolutionResult
from pyrecycle_analytics.deconvolution.windows import RetentionWindow, detect_windows
from pyrecycle_analytics.matrix import MatrixSubtractionError, subtract_polymer_matrix
from pyrecycle_analytics.preprocessing import PreprocessingConfig, preprocess
from pyrecycle_analytics.validation.metrics import explained_variance, lack_of_fit

__all__ = ["DeconvolutionConfig", "DeconvolutionReport", "resolve_pyrogram"]


class DeconvolutionConfig(BaseModel):
    """Reproducible description of the whole chain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    subtract_matrix: bool = Field(
        True,
        description="Remove the polyolefin backbone before resolving. Switching "
        "this off is only useful for demonstrating why it is on.",
    )
    matrix_type: str = "PE_PP_Backbone"
    require_matrix: bool = Field(
        False,
        description="Fail when no homologous series is found. Off by default so "
        "that a PS- or PET-dominated sample, which legitimately has no polyolefin "
        "matrix, still resolves.",
    )
    max_rank: int = Field(8, ge=1, le=30)
    rank_margin: int = Field(
        2,
        ge=0,
        le=10,
        description="Extra components granted beyond the estimated rank. Rank "
        "estimation is conservative on windows that still carry matrix remnant, "
        "and a trace marker sharing a window with a large peak only gets its own "
        "component if one is available. Measured on the benchmark, raising the "
        "rank from 2 to 8 in the caprolactam window lifted its spectral recovery "
        "from 0.74 to 0.87. The extra components are pruned afterwards by "
        "min_component_area_fraction, so the cost of being generous is bounded.",
    )
    min_prominence_fraction: float = Field(0.005, gt=0.0, lt=1.0)
    merge_resolution: float = Field(1.0, gt=0.0)
    max_iterations: int = Field(200, ge=1, le=5000)
    min_window_scans: int = Field(
        8,
        ge=3,
        description="Windows shorter than this carry too few points to resolve "
        "anything and are skipped.",
    )
    min_window_signal_fraction: float = Field(
        1e-4,
        ge=0.0,
        lt=1.0,
        description="Windows carrying less than this share of the run's total "
        "signal are skipped. After matrix subtraction the residual is mostly "
        "empty, and without this the peak finder turns noise ripples into windows "
        "that each yield a spurious component.",
    )
    min_component_area_fraction: float = Field(
        2e-4,
        ge=0.0,
        lt=1.0,
        description="Components below this share of the run's total resolved area "
        "are dropped. Applied globally rather than per window: a component that "
        "dominates an otherwise empty window is still noise if the window itself "
        "holds nothing. Set low enough to keep genuine trace markers — the "
        "smallest real marker in the benchmark sits near 2e-4 of the total.",
    )


@dataclass(slots=True)
class DeconvolutionReport:
    """Result of resolving a whole run.

    Attributes:
        result: The stitched run-wide factorisation.
        processed: The cube the resolution actually ran on, after preprocessing
            and matrix subtraction.
        n_windows: Windows examined.
        n_resolved: Windows that produced components.
        matrix_explained_fraction: Share of signal attributed to the polyolefin
            backbone; 0.0 when subtraction was skipped or found no series.
        warnings: Per-window problems worth surfacing — rank disagreement,
            non-convergence, skipped windows.
        diagnostics: Free-form detail for the benchmark report.
    """

    result: ResolutionResult
    processed: PyrogramDataCube
    n_windows: int
    n_resolved: int
    matrix_explained_fraction: float
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _stitch(
    windows: list[tuple[RetentionWindow, ResolutionResult]],
    cube: PyrogramDataCube,
) -> ResolutionResult:
    """Place per-window factorisations into one run-wide pair of factors."""
    profile_columns: list[np.ndarray] = []
    spectrum_rows: list[np.ndarray] = []

    for window, resolved in windows:
        for component in range(resolved.n_components):
            column = np.zeros(cube.n_scans, dtype=np.float64)
            column[window.start_index : window.stop_index] = resolved.C[:, component]
            profile_columns.append(column)
            spectrum_rows.append(resolved.S[component, :])

    if not profile_columns:
        return ResolutionResult(
            retention_times=cube.retention_times,
            mz_axis=cube.mz_axis,
            C=np.zeros((cube.n_scans, 1)),
            S=np.zeros((1, cube.n_mz)),
            method="mcr-als/windowed",
            diagnostics={"note": "no window produced a component"},
        )

    profiles = np.column_stack(profile_columns)
    spectra = np.vstack(spectrum_rows)
    reconstruction = profiles @ spectra
    return ResolutionResult(
        retention_times=cube.retention_times,
        mz_axis=cube.mz_axis,
        C=profiles,
        S=spectra,
        lack_of_fit=lack_of_fit(cube.intensities, reconstruction),
        explained_variance=explained_variance(cube.intensities, reconstruction),
        method="mcr-als/windowed",
    ).sorted_by_retention()


def resolve_pyrogram(
    cube: PyrogramDataCube,
    config: DeconvolutionConfig | None = None,
) -> DeconvolutionReport:
    """Run the full Milestone 2 chain on a pyrogram.

    Args:
        cube: Ingested pyrogram, raw or already preprocessed.
        config: Chain description; defaults are used when omitted.

    Returns:
        The stitched factorisation with its diagnostics.

    Raises:
        MatrixSubtractionError: If ``require_matrix`` is set and no homologous
            series can be found.
    """
    config = config or DeconvolutionConfig()
    warnings: list[str] = []

    processed = preprocess(cube, config.preprocessing)

    matrix_fraction = 0.0
    if config.subtract_matrix:
        try:
            subtraction = subtract_polymer_matrix(
                processed, config.matrix_type
            )
            processed = subtraction.residual
            matrix_fraction = subtraction.explained_fraction
        except MatrixSubtractionError as error:
            if config.require_matrix:
                raise
            warnings.append(
                f"no polyolefin matrix subtracted ({error}); resolving the full signal"
            )

    windows = detect_windows(
        processed,
        min_prominence_fraction=config.min_prominence_fraction,
        merge_resolution=config.merge_resolution,
    )

    resolved: list[tuple[RetentionWindow, ResolutionResult]] = []
    rank_disagreements = 0
    non_converged = 0
    skipped_quiet = 0
    total_signal = processed.total_signal

    for window in windows:
        if window.n_scans < config.min_window_scans:
            continue
        block = processed.intensities[window.slice()]
        if not np.any(block):
            continue
        if (
            total_signal > 0.0
            and float(block.sum()) / total_signal < config.min_window_signal_fraction
        ):
            skipped_quiet += 1
            continue

        ceiling = min(config.max_rank, window.n_scans - 1, processed.n_mz)
        estimate = estimate_rank(block, max_rank=max(ceiling, 1))
        if not estimate.agreement:
            rank_disagreements += 1
            warnings.append(
                f"window {window.start_s:.0f}-{window.end_s:.0f}s: {estimate.warning}"
            )

        # A window must be allowed at least as many components as it has resolved
        # maxima; fewer would merge peaks that are already visibly separate.
        n_components = int(
            np.clip(max(estimate.rank, window.n_peaks) + config.rank_margin, 1, ceiling)
        )
        try:
            outcome = mcr_als(
                block,
                n_components,
                retention_times=processed.retention_times[window.slice()],
                mz_axis=processed.mz_axis,
                options=McrAlsOptions(max_iterations=config.max_iterations),
            )
        except ResolutionError as error:
            warnings.append(
                f"window {window.start_s:.0f}-{window.end_s:.0f}s not resolved: {error}"
            )
            continue

        if not outcome.converged:
            non_converged += 1
        resolved.append((window, outcome))

    stitched = _stitch(resolved, processed)
    before_filter = stitched.n_components
    stitched = stitched.drop_negligible_of_total(config.min_component_area_fraction)

    return DeconvolutionReport(
        result=stitched,
        processed=processed,
        n_windows=len(windows),
        n_resolved=len(resolved),
        matrix_explained_fraction=matrix_fraction,
        warnings=tuple(warnings),
        diagnostics={
            "rank_disagreements": rank_disagreements,
            "non_converged_windows": non_converged,
            "skipped_quiet_windows": skipped_quiet,
            "components_before_filter": before_filter,
            "components_after_filter": stitched.n_components,
            "matrix_subtracted": config.subtract_matrix and matrix_fraction > 0.0,
        },
    )
