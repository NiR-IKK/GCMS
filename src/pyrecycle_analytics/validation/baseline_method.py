"""The naive method that curve resolution has to beat.

What an analyst does by hand: find the peaks in the total ion current, take the
mass spectrum at each apex, subtract a neighbouring blank, and integrate the TIC
between the peak's feet. It is fast, it needs no chemometrics, and on
well-separated peaks it is perfectly adequate.

Implementing it is not busywork. Without a reference point, "MCR-ALS reached a
spectral similarity of 0.93" is an uninterpretable number — it could be worse
than doing nothing. This module turns every claim about the deconvolution engine
into a comparison, and it is expected to fail exactly where the interesting
cases are: fused clusters, where one TIC peak hides four compounds and the apex
spectrum is a mixture of all of them.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths

from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.deconvolution.result import ResolutionResult
from pyrecycle_analytics.validation.metrics import explained_variance, lack_of_fit

__all__ = ["naive_peak_integration"]


def naive_peak_integration(
    cube: PyrogramDataCube,
    *,
    min_prominence_fraction: float = 0.01,
    min_distance_scans: int = 3,
    background_offset_widths: float = 2.5,
) -> ResolutionResult:
    """Resolve a pyrogram by plain TIC peak picking and apex spectra.

    Args:
        cube: Pyrogram, ideally already baseline-corrected.
        min_prominence_fraction: Peak prominence threshold, relative to the
            largest TIC value. Lower values find more (and more spurious) peaks.
        min_distance_scans: Minimum separation between accepted peaks.
        background_offset_widths: How far outside a peak, in units of its own
            width, the blank spectrum is taken for background subtraction.

    Returns:
        A :class:`ResolutionResult` in the same convention as any other method:
        rows of ``S`` sum to one, columns of ``C`` carry the area.

    Raises:
        ValueError: If ``min_prominence_fraction`` is not positive.
    """
    if min_prominence_fraction <= 0.0:
        raise ValueError(
            f"min_prominence_fraction must be > 0, got {min_prominence_fraction}"
        )

    tic = cube.tic
    if tic.max() <= 0.0:
        empty_profiles = np.zeros((cube.n_scans, 1))
        empty_spectra = np.zeros((1, cube.n_mz))
        empty_spectra[0, 0] = 1.0
        return ResolutionResult(
            retention_times=cube.retention_times,
            mz_axis=cube.mz_axis,
            C=empty_profiles,
            S=empty_spectra,
            method="naive-peak-integration",
            diagnostics={"n_peaks": 0, "note": "empty chromatogram"},
        )

    peaks, _ = find_peaks(
        tic,
        prominence=min_prominence_fraction * float(tic.max()),
        distance=max(min_distance_scans, 1),
    )
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(tic))])

    # rel_height=0.9 approximates the peak's feet rather than its half width,
    # which is what "integrate between the feet" means in practice.
    _, _, left_bases, right_bases = peak_widths(tic, peaks, rel_height=0.9)

    profiles = np.zeros((cube.n_scans, peaks.size), dtype=np.float64)
    spectra = np.zeros((peaks.size, cube.n_mz), dtype=np.float64)

    for column, apex in enumerate(peaks):
        left = int(np.floor(left_bases[column]))
        right = int(np.ceil(right_bases[column])) + 1
        left = max(left, 0)
        right = min(right, cube.n_scans)
        if right <= left:
            left, right = int(apex), int(apex) + 1

        # The profile is simply the TIC inside the peak's extent — no attempt to
        # unmix, which is exactly the limitation being demonstrated.
        profiles[left:right, column] = tic[left:right]

        half_width = max((right - left) // 2, 1)
        offset = int(background_offset_widths * half_width)
        blank_start = min(right + offset, cube.n_scans - 1)
        blank_stop = min(blank_start + half_width, cube.n_scans)
        background = (
            cube.intensities[blank_start:blank_stop, :].mean(axis=0)
            if blank_stop > blank_start
            else np.zeros(cube.n_mz)
        )

        spectrum = np.clip(cube.intensities[int(apex), :] - background, 0.0, None)
        total = spectrum.sum()
        if total <= 0.0:
            spectrum = np.clip(cube.intensities[int(apex), :], 0.0, None)
            total = spectrum.sum()
        spectra[column, :] = spectrum / total if total > 0.0 else 0.0

    reconstruction = profiles @ spectra
    return ResolutionResult(
        retention_times=cube.retention_times,
        mz_axis=cube.mz_axis,
        C=profiles,
        S=spectra,
        lack_of_fit=lack_of_fit(cube.intensities, reconstruction),
        explained_variance=explained_variance(cube.intensities, reconstruction),
        n_iterations=1,
        converged=True,
        method="naive-peak-integration",
        diagnostics={"n_peaks": int(peaks.size)},
    )
