"""Noise suppression along the retention-time axis.

Curve resolution is a least-squares procedure, so shot noise on the raw channels
propagates straight into the resolved profiles. Mild smoothing along the *time*
axis before deconvolution measurably improves conditioning.

Smoothing is never applied along the m/z axis: adjacent nominal masses are
chemically unrelated (m/z 56 and 57 come from different fragmentation routes), and
mixing them destroys exactly the fine structure the marker logic depends on.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from data_schemas.pyrogram import PreprocessingStep
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import PreprocessingError

__all__ = ["savgol_smooth", "gaussian_smooth", "smooth_cube", "estimate_noise_sigma"]


def savgol_smooth(
    intensities: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Savitzky-Golay smoothing along the scan axis.

    Fits a local polynomial in a sliding window, which preserves peak height and
    position far better than a moving average — important because peak area
    ratios are the quantitative output of this platform.

    Args:
        intensities: Signal, 1-D or 2-D ``(n_scans, n_mz)``.
        window_length: Window size in scans; must be odd and exceed ``polyorder``.
        polyorder: Polynomial degree, typically 2 or 3.
        axis: Axis to filter along; 0 is the retention-time axis.

    Returns:
        Smoothed array of the same shape, clipped at zero.

    Raises:
        PreprocessingError: If the window is even, too small for the polynomial
            order, or longer than the data along ``axis``.
    """
    intensities = np.asarray(intensities, dtype=np.float64)
    if window_length % 2 == 0:
        raise PreprocessingError(f"window_length must be odd, got {window_length}")
    if window_length <= polyorder:
        raise PreprocessingError(
            f"window_length ({window_length}) must exceed polyorder ({polyorder})"
        )
    if window_length > intensities.shape[axis]:
        raise PreprocessingError(
            f"window_length ({window_length}) exceeds the {intensities.shape[axis]} "
            f"points available along axis {axis}"
        )
    smoothed = savgol_filter(
        intensities, window_length=window_length, polyorder=polyorder, axis=axis, mode="interp"
    )
    # The polynomial fit can undershoot into negative values on steep peak flanks.
    return np.clip(smoothed, 0.0, None)


def gaussian_smooth(
    intensities: np.ndarray,
    sigma_scans: float = 1.0,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Gaussian smoothing along the scan axis.

    Args:
        intensities: Signal, 1-D or 2-D.
        sigma_scans: Kernel width in scans. Keep well below the peak sigma
            (typically <= 0.5 x peak sigma) to avoid broadening.
        axis: Axis to filter along.

    Returns:
        Smoothed array of the same shape.

    Raises:
        PreprocessingError: If ``sigma_scans`` is not positive.
    """
    if sigma_scans <= 0.0:
        raise PreprocessingError(f"sigma_scans must be > 0, got {sigma_scans}")
    intensities = np.asarray(intensities, dtype=np.float64)
    return gaussian_filter1d(intensities, sigma=sigma_scans, axis=axis, mode="nearest")


def estimate_noise_sigma(signal: np.ndarray) -> float:
    """Robust noise estimate from the median absolute second difference.

    Uses the second difference so that a smooth baseline and broad peaks cancel
    out, leaving point-to-point noise. The median makes the estimate insensitive
    to the peaks themselves, so it works on a full chromatogram without needing a
    hand-picked blank region.

    Args:
        signal: Chromatogram or channel, shape ``(n,)`` with ``n >= 3``.

    Returns:
        Estimated standard deviation of the additive noise.

    Raises:
        PreprocessingError: If fewer than three points are supplied.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size < 3:
        raise PreprocessingError(f"need at least 3 points, got {signal.size}")
    second_difference = np.diff(signal, n=2)
    # For white noise, Var(y[i-1] - 2 y[i] + y[i+1]) = 6 sigma^2; 1.4826 converts
    # a median absolute deviation into a Gaussian-equivalent sigma.
    mad = float(np.median(np.abs(second_difference - np.median(second_difference))))
    return 1.4826 * mad / np.sqrt(6.0)


def smooth_cube(
    cube: PyrogramDataCube,
    method: str = "savgol",
    **parameters: float | int,
) -> PyrogramDataCube:
    """Smooth every m/z channel of a cube along retention time.

    Args:
        cube: Pyrogram to smooth.
        method: ``"savgol"`` (default) or ``"gaussian"``.
        **parameters: Forwarded to the chosen filter (``window_length``,
            ``polyorder`` for Savitzky-Golay; ``sigma_scans`` for Gaussian).

    Returns:
        A new cube with the smoothing recorded in its audit trail.

    Raises:
        PreprocessingError: If ``method`` is unknown or parameters are invalid.
    """
    if method == "savgol":
        smoothed = savgol_smooth(cube.intensities, **parameters)  # type: ignore[arg-type]
    elif method == "gaussian":
        smoothed = gaussian_smooth(cube.intensities, **parameters)  # type: ignore[arg-type]
    else:
        raise PreprocessingError(f"unknown smoothing method {method!r}; use 'savgol' or 'gaussian'")

    step = PreprocessingStep(
        name=f"smooth:{method}",
        parameters=dict(parameters),
        note="applied along the retention-time axis only",
    )
    return cube.with_intensities(smoothed, step)
