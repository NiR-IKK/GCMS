"""Chromatographic peak-shape models.

Every peak in a pyrogram — a synthetic one we generate, or a resolved profile a
curve-resolution algorithm returns — is described by the same small family of
functions. Keeping them in one place means the simulator and the deconvolution
engine cannot drift apart in their idea of what a peak looks like.

The workhorse is the exponentially modified Gaussian (EMG): a Gaussian
convolved with a one-sided exponential decay. It is the standard empirical model
for the asymmetric tailing produced by slow mass transfer in a GC column, and it
reduces to a plain Gaussian as the tailing constant goes to zero.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcx

__all__ = [
    "gaussian",
    "emg",
    "emg_apex_time",
    "fwhm_gaussian",
    "profile_matrix",
    "chromatographic_resolution",
]

# Below this tau/sigma ratio the EMG is numerically indistinguishable from a
# Gaussian, and the closed form loses precision, so we switch models.
_MIN_TAU_RATIO = 1e-6


def gaussian(
    t: np.ndarray,
    center: float,
    sigma: float,
    area: float = 1.0,
) -> np.ndarray:
    """Area-normalised Gaussian elution profile.

    Args:
        t: Retention-time grid in seconds.
        center: Apex position in seconds.
        sigma: Standard deviation in seconds.
        area: Integrated area of the profile (in intensity x seconds).

    Returns:
        Profile evaluated on ``t``, integrating to ``area``.

    Raises:
        ValueError: If ``sigma`` is not strictly positive.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    t = np.asarray(t, dtype=np.float64)
    z = (t - center) / sigma
    return (area / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def emg(
    t: np.ndarray,
    center: float,
    sigma: float,
    tau: float,
    area: float = 1.0,
) -> np.ndarray:
    """Area-normalised exponentially modified Gaussian.

    Models the tailing of a real chromatographic peak as a Gaussian of width
    ``sigma`` convolved with an exponential decay of time constant ``tau``.
    ``center`` is the Gaussian's centre, not the apex of the resulting
    asymmetric peak; use :func:`emg_apex_time` for the observed apex.

    The implementation uses the scaled complementary error function so that the
    profile stays accurate for strongly tailing peaks (large ``tau/sigma``),
    where the naive ``exp(...) * erfc(...)`` form overflows and underflows into
    ``inf * 0``.

    Args:
        t: Retention-time grid in seconds.
        center: Gaussian centre in seconds.
        sigma: Gaussian standard deviation in seconds.
        tau: Exponential decay constant in seconds; 0 yields a pure Gaussian.
        area: Integrated area of the profile.

    Returns:
        Profile evaluated on ``t``, integrating to ``area`` over the real line.

    Raises:
        ValueError: If ``sigma <= 0`` or ``tau < 0``.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if tau < 0.0:
        raise ValueError(f"tau must be >= 0, got {tau}")

    t = np.asarray(t, dtype=np.float64)
    if tau <= _MIN_TAU_RATIO * sigma:
        return gaussian(t, center, sigma, area)

    dt = t - center
    # z >= 0 on the rising side and in the core; z < 0 far into the tail.
    z = (sigma / tau - dt / sigma) / np.sqrt(2.0)
    scale = area / (2.0 * tau)

    result = np.empty_like(dt)
    positive = z >= 0.0

    # Stable core/front: exp(u) * erfc(z) == exp(-dt^2 / 2 sigma^2) * erfcx(z),
    # an algebraic identity that removes the cancelling large exponents.
    if np.any(positive):
        dt_pos = dt[positive]
        result[positive] = (
            scale * np.exp(-0.5 * (dt_pos / sigma) ** 2) * erfcx(z[positive])
        )

    # Deep tail: here u = sigma^2/(2 tau^2) - dt/tau is guaranteed negative,
    # so the direct form is well behaved and erfc(z) -> 2.
    negative = ~positive
    if np.any(negative):
        u = sigma * sigma / (2.0 * tau * tau) - dt[negative] / tau
        result[negative] = scale * np.exp(u) * erfc(z[negative])

    return result


def emg_apex_time(center: float, sigma: float, tau: float) -> float:
    """Approximate observed apex of an EMG peak.

    The exponential modifier pushes the maximum to later times. There is no
    closed form; this uses the standard second-order approximation, which is
    accurate to well below one scan period for the ``tau/sigma <= 3`` range that
    covers usable chromatography.

    Args:
        center: Gaussian centre in seconds.
        sigma: Gaussian standard deviation in seconds.
        tau: Exponential decay constant in seconds.

    Returns:
        Apex retention time in seconds.
    """
    if tau <= _MIN_TAU_RATIO * sigma:
        return float(center)
    ratio = sigma / tau
    # Solving d/dt [ln f] = 0 to second order in the tail expansion.
    shift = sigma * (1.0 / (ratio + np.sqrt(ratio * ratio + 4.0) / 2.0)) * 0.5
    return float(center + min(shift, tau))


def fwhm_gaussian(sigma: float) -> float:
    """Full width at half maximum of a Gaussian of width ``sigma``."""
    return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)


def profile_matrix(
    t: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    taus: np.ndarray | None = None,
    areas: np.ndarray | None = None,
) -> np.ndarray:
    """Build the elution-profile matrix ``C`` of a bilinear GC/MS model.

    In the bilinear model ``D = C @ S`` the columns of ``C`` are the pure
    concentration profiles of each component over retention time, and the rows of
    ``S`` are their mass spectra. This helper assembles ``C``.

    Args:
        t: Retention-time grid, shape ``(n_scans,)``.
        centers: Apex/centre positions, shape ``(n_components,)``.
        sigmas: Peak widths, shape ``(n_components,)``.
        taus: Tailing constants, shape ``(n_components,)``. ``None`` means all zero.
        areas: Integrated areas, shape ``(n_components,)``. ``None`` means unit area.

    Returns:
        Array of shape ``(n_scans, n_components)``.

    Raises:
        ValueError: If the parameter arrays have inconsistent lengths.
    """
    t = np.asarray(t, dtype=np.float64)
    centers = np.atleast_1d(np.asarray(centers, dtype=np.float64))
    sigmas = np.atleast_1d(np.asarray(sigmas, dtype=np.float64))
    n_components = centers.size

    if sigmas.size != n_components:
        raise ValueError(f"sigmas has {sigmas.size} entries, expected {n_components}")
    taus_arr = (
        np.zeros(n_components)
        if taus is None
        else np.atleast_1d(np.asarray(taus, dtype=np.float64))
    )
    areas_arr = (
        np.ones(n_components)
        if areas is None
        else np.atleast_1d(np.asarray(areas, dtype=np.float64))
    )
    if taus_arr.size != n_components:
        raise ValueError(f"taus has {taus_arr.size} entries, expected {n_components}")
    if areas_arr.size != n_components:
        raise ValueError(f"areas has {areas_arr.size} entries, expected {n_components}")

    profiles = np.empty((t.size, n_components), dtype=np.float64)
    for index in range(n_components):
        profiles[:, index] = emg(
            t,
            center=float(centers[index]),
            sigma=float(sigmas[index]),
            tau=float(taus_arr[index]),
            area=float(areas_arr[index]),
        )
    return profiles


def chromatographic_resolution(
    rt_a: float,
    sigma_a: float,
    rt_b: float,
    sigma_b: float,
) -> float:
    """Resolution ``R`` between two peaks.

    Uses the width-sum definition ``R = |Δt| / (2 (σ_a + σ_b))``, which is the
    Gaussian-equivalent of the classical ``2Δt / (w_a + w_b)``. ``R >= 1.5``
    is baseline resolution; ``R < 1.0`` is the co-elution regime that requires
    curve resolution rather than integration.

    Args:
        rt_a: Apex of the first peak in seconds.
        sigma_a: Width of the first peak in seconds.
        rt_b: Apex of the second peak in seconds.
        sigma_b: Width of the second peak in seconds.

    Returns:
        Dimensionless resolution; ``inf`` if both widths are zero.
    """
    width_sum = sigma_a + sigma_b
    if width_sum <= 0.0:
        return float("inf")
    return float(abs(rt_b - rt_a) / (2.0 * width_sum))
