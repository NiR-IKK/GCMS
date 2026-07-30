"""Baseline correction for pyrogram channels.

Py-GC/MS baselines are not flat. Two effects dominate and both must be removed
before any ratio-based marker logic runs:

* **Column bleed.** The stationary phase degrades as the oven ramps, producing a
  monotonically rising background concentrated on the siloxane ions
  (m/z 73, 147, 207, 221, 281). A trace PET or PA marker sitting on that ramp is
  quantified wrongly by tens of percent if the ramp is not subtracted.
* **Unresolved polymer hump.** The polyolefin homologous series is so dense that
  its unresolved envelope acts as a slowly varying pedestal under everything else.

Two complementary estimators are provided. AsLS (asymmetric least squares,
Eilers & Boelens) is the general-purpose smooth-baseline estimator and is what
the pipeline uses by default. SNIP (statistics-sensitive non-linear iterative
peak clipping) is a morphological alternative that is more robust when peaks are
very dense, because it never fits through them.

Correction is applied **per m/z channel**, not to the TIC. Subtracting a TIC
baseline scaled across channels would distort mass spectra; each channel has its
own background chemistry, so each gets its own estimate.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from scipy.linalg import solveh_banded
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve

from data_schemas.pyrogram import PreprocessingStep
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import PreprocessingError

__all__ = [
    "asls_baseline",
    "snip_baseline",
    "asls_baseline_matrix",
    "snip_baseline_matrix",
    "correct_baseline",
]


def _second_difference_operator(n: int) -> csc_matrix:
    """Sparse second-difference matrix ``D2`` with ``D2 @ y ~ y''``."""
    diagonals = [
        np.ones(n - 2),
        -2.0 * np.ones(n - 2),
        np.ones(n - 2),
    ]
    return diags(diagonals, offsets=[0, 1, 2], shape=(n - 2, n), format="csc")


def asls_baseline(
    signal: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    n_iter: int = 20,
    tol: float = 1e-6,
) -> np.ndarray:
    """Asymmetric least squares baseline of a single chromatogram.

    Fits a smooth curve that is penalised much more strongly for lying *above*
    the signal than below it, so peaks are ignored while the background is
    followed. Smoothness is controlled by the second-derivative penalty ``lam``.

    Args:
        signal: Chromatogram, shape ``(n_scans,)``.
        lam: Smoothness penalty. Larger values give stiffer baselines; 1e5-1e8 is
            the useful range for GC/MS at 5-20 Hz.
        p: Asymmetry in ``(0, 1)``. Points above the current baseline get weight
            ``p``, points below get ``1 - p``. Use 0.001-0.05 for peak-rich data.
        n_iter: Maximum reweighting iterations.
        tol: Stop when the mean absolute change of the baseline falls below
            ``tol`` times the signal's mean absolute value.

    Returns:
        Estimated baseline, shape ``(n_scans,)``.

    Raises:
        PreprocessingError: If parameters are out of range or the signal is too
            short (fewer than 4 points) for a second-derivative penalty.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = signal.size
    if n < 4:
        raise PreprocessingError(f"asls_baseline needs at least 4 points, got {n}")
    if not 0.0 < p < 1.0:
        raise PreprocessingError(f"p must lie in (0, 1), got {p}")
    if lam <= 0.0:
        raise PreprocessingError(f"lam must be > 0, got {lam}")
    if n_iter < 1:
        raise PreprocessingError(f"n_iter must be >= 1, got {n_iter}")

    difference_operator = _second_difference_operator(n)
    penalty = lam * (difference_operator.T @ difference_operator)
    weights = np.ones(n)
    baseline = np.zeros(n)
    scale = max(float(np.mean(np.abs(signal))), np.finfo(float).tiny)

    for _ in range(n_iter):
        weight_matrix = diags(weights, 0, shape=(n, n), format="csc")
        baseline_new = spsolve(weight_matrix + penalty, weights * signal)
        change = float(np.mean(np.abs(baseline_new - baseline))) / scale
        baseline = baseline_new
        weights = np.where(signal > baseline, p, 1.0 - p)
        if change < tol:
            break

    return baseline


def snip_baseline(
    signal: np.ndarray,
    n_iter: int = 40,
    *,
    decreasing: bool = True,
    log_transform: bool = True,
) -> np.ndarray:
    """SNIP baseline of a single chromatogram.

    Iteratively clips each point to the mean of its two neighbours at distance
    ``w`` whenever that mean is lower, for ``w`` running from ``n_iter`` down to 1.
    Because it only ever pushes the estimate *down*, the result never cuts into
    the background, which makes it the safer choice in dense homologous-series
    regions where AsLS can be pulled up by fused peaks.

    Args:
        signal: Chromatogram, shape ``(n_scans,)``.
        n_iter: Largest clipping half-window in scans. Should exceed the widest
            peak's half-width but stay well below the background's curvature scale.
        decreasing: Run windows from wide to narrow (the standard "SNIP with
            decreasing clipping window", which preserves narrow peaks better).
        log_transform: Apply the classical double-log-square-root transform, which
            makes the clipping behave consistently across four decades of intensity.

    Returns:
        Estimated baseline, shape ``(n_scans,)``.

    Raises:
        PreprocessingError: If ``n_iter`` is not positive or exceeds half the
            signal length.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = signal.size
    if n_iter < 1:
        raise PreprocessingError(f"n_iter must be >= 1, got {n_iter}")
    if n < 3:
        raise PreprocessingError(f"snip_baseline needs at least 3 points, got {n}")
    max_window = (n - 1) // 2
    if n_iter > max_window:
        n_iter = max(max_window, 1)

    offset = 0.0
    working = signal.astype(np.float64).copy()
    if log_transform:
        offset = float(min(working.min(), 0.0))
        working = np.log(np.log(np.sqrt(np.clip(working - offset, 0.0, None) + 1.0) + 1.0) + 1.0)

    windows = range(n_iter, 0, -1) if decreasing else range(1, n_iter + 1)
    for window in windows:
        shifted_left = np.roll(working, window)
        shifted_right = np.roll(working, -window)
        neighbour_mean = 0.5 * (shifted_left + shifted_right)
        # Edges have no valid neighbour pair; leave them untouched.
        candidate = working.copy()
        candidate[window : n - window] = np.minimum(
            working[window : n - window], neighbour_mean[window : n - window]
        )
        working = candidate

    if log_transform:
        working = (np.exp(np.exp(working) - 1.0) - 1.0) ** 2 - 1.0 + offset
        working = np.clip(working, None, signal)

    return working


def _asls_banded(
    signal: np.ndarray,
    penalty_bands: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Solve ``(W + lam D'D) z = W y`` given the penalty in upper-banded form."""
    bands = penalty_bands.copy()
    bands[-1, :] += weights  # last row of the banded form is the main diagonal
    return solveh_banded(bands, weights * signal, lower=False, check_finite=False)


def _penalty_bands(n: int, lam: float) -> np.ndarray:
    """Upper-banded representation of ``lam * D2' D2`` for :func:`solveh_banded`.

    ``D2' D2`` is symmetric pentadiagonal, so the dense solve of the sparse
    formulation can be replaced by a banded Cholesky. That matters here because
    the baseline is estimated for every one of ~500 m/z channels.
    """
    main = np.full(n, 6.0)
    main[0] = main[-1] = 1.0
    main[1] = main[-2] = 5.0
    first = np.full(n - 1, -4.0)
    first[0] = first[-1] = -2.0
    second = np.ones(n - 2)

    bands = np.zeros((3, n))
    bands[0, 2:] = lam * second
    bands[1, 1:] = lam * first
    bands[2, :] = lam * main
    return bands


def asls_baseline_matrix(
    intensities: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    n_iter: int = 20,
    *,
    min_channel_signal: float = 0.0,
) -> np.ndarray:
    """AsLS baselines for every m/z channel of an intensity matrix.

    Args:
        intensities: Matrix ``(n_scans, n_mz)``.
        lam: Smoothness penalty, see :func:`asls_baseline`.
        p: Asymmetry parameter, see :func:`asls_baseline`.
        n_iter: Reweighting iterations.
        min_channel_signal: Channels whose maximum is at or below this value are
            treated as empty and get a zero baseline, which avoids fitting noise.

    Returns:
        Baseline matrix of the same shape as ``intensities``.

    Raises:
        PreprocessingError: If the matrix is not 2-D or too short in the scan axis.
    """
    intensities = np.asarray(intensities, dtype=np.float64)
    if intensities.ndim != 2:
        raise PreprocessingError(f"expected a 2-D matrix, got {intensities.ndim}-D")
    n_scans, n_mz = intensities.shape
    if n_scans < 4:
        raise PreprocessingError(f"need at least 4 scans for AsLS, got {n_scans}")
    if not 0.0 < p < 1.0:
        raise PreprocessingError(f"p must lie in (0, 1), got {p}")
    if lam <= 0.0:
        raise PreprocessingError(f"lam must be > 0, got {lam}")

    bands = _penalty_bands(n_scans, lam)
    baselines = np.zeros_like(intensities)

    for channel in range(n_mz):
        column = intensities[:, channel]
        if column.max() <= min_channel_signal:
            continue
        weights = np.ones(n_scans)
        baseline = np.zeros(n_scans)
        for _ in range(n_iter):
            baseline = _asls_banded(column, bands, weights)
            weights = np.where(column > baseline, p, 1.0 - p)
        baselines[:, channel] = baseline

    return baselines


def snip_baseline_matrix(
    intensities: np.ndarray,
    n_iter: int = 40,
    *,
    min_channel_signal: float = 0.0,
) -> np.ndarray:
    """SNIP baselines for every m/z channel of an intensity matrix.

    Args:
        intensities: Matrix ``(n_scans, n_mz)``.
        n_iter: Largest clipping half-window in scans.
        min_channel_signal: Channels at or below this maximum get a zero baseline.

    Returns:
        Baseline matrix of the same shape as ``intensities``.

    Raises:
        PreprocessingError: If the matrix is not 2-D.
    """
    intensities = np.asarray(intensities, dtype=np.float64)
    if intensities.ndim != 2:
        raise PreprocessingError(f"expected a 2-D matrix, got {intensities.ndim}-D")
    baselines = np.zeros_like(intensities)
    for channel in range(intensities.shape[1]):
        column = intensities[:, channel]
        if column.max() <= min_channel_signal:
            continue
        baselines[:, channel] = snip_baseline(column, n_iter=n_iter)
    return baselines


@overload
def correct_baseline(
    cube: PyrogramDataCube,
    method: str = ...,
    *,
    clip_negative: bool = ...,
    return_baseline: Literal[False] = ...,
    **parameters: float,
) -> PyrogramDataCube: ...


@overload
def correct_baseline(
    cube: PyrogramDataCube,
    method: str = ...,
    *,
    clip_negative: bool = ...,
    return_baseline: Literal[True],
    **parameters: float,
) -> tuple[PyrogramDataCube, np.ndarray]: ...


def correct_baseline(
    cube: PyrogramDataCube,
    method: str = "asls",
    *,
    clip_negative: bool = True,
    return_baseline: bool = False,
    **parameters: float,
) -> PyrogramDataCube | tuple[PyrogramDataCube, np.ndarray]:
    """Remove the per-channel baseline from a data cube.

    Args:
        cube: Pyrogram to correct.
        method: ``"asls"`` (default, smooth penalised fit) or ``"snip"``
            (morphological clipping).
        clip_negative: Clamp negative residuals to zero. Recommended: ion counts
            cannot be negative, and negative channels break non-negativity
            constraints in MCR-ALS downstream.
        return_baseline: Also return the estimated baseline matrix.
        **parameters: Forwarded to the chosen estimator (``lam``, ``p``,
            ``n_iter``, ``min_channel_signal``).

    Returns:
        The corrected cube, or ``(cube, baseline)`` when ``return_baseline`` is set.

    Raises:
        PreprocessingError: If ``method`` is unknown or parameters are invalid.
    """
    if method == "asls":
        baseline = asls_baseline_matrix(cube.intensities, **parameters)  # type: ignore[arg-type]
    elif method == "snip":
        baseline = snip_baseline_matrix(cube.intensities, **parameters)  # type: ignore[arg-type]
    else:
        raise PreprocessingError(f"unknown baseline method {method!r}; use 'asls' or 'snip'")

    corrected = cube.intensities - baseline
    if clip_negative:
        corrected = np.clip(corrected, 0.0, None)

    step = PreprocessingStep(
        name=f"baseline:{method}",
        parameters={
            "clip_negative": clip_negative,
            **{key: value for key, value in parameters.items()},
        },
        note="per-m/z-channel baseline subtraction",
    )
    result = cube.with_intensities(corrected, step)
    return (result, baseline) if return_baseline else result
