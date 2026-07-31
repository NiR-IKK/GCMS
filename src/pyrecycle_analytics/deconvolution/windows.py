"""Cutting a run into windows of co-eluting peaks.

Curve resolution is never applied to a whole 30-minute run: the bilinear model
would need ~190 components, and the least-squares problem becomes both
intractable and hopelessly under-determined. It is applied to a window around one
cluster of overlapping peaks, where three to five components suffice.

This module finds those clusters without knowing the answer. It mirrors what
``PyrogramTruth.coeluting_groups()`` does on the ground truth — group peaks whose
chromatographic resolution falls below a threshold — but works from the data:
peaks are detected on the total ion current, their widths estimated from their
own shape, and neighbours merged when they are not resolved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths

from pyrecycle_analytics.core.datacube import PyrogramDataCube

__all__ = ["RetentionWindow", "detect_windows"]

# Converts a full width at half maximum into a Gaussian standard deviation.
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True, slots=True)
class RetentionWindow:
    """A contiguous stretch of the run treated as one resolution problem.

    Attributes:
        start_index: First scan index, inclusive.
        stop_index: Last scan index, exclusive.
        start_s: Retention time of the first scan.
        end_s: Retention time of the last scan.
        apex_indices: Scan indices of the detected peaks inside the window. Their
            count is the first, cheapest guess at the window's rank.
    """

    start_index: int
    stop_index: int
    start_s: float
    end_s: float
    apex_indices: tuple[int, ...]

    @property
    def n_scans(self) -> int:
        return self.stop_index - self.start_index

    @property
    def n_peaks(self) -> int:
        """Number of resolved maxima — a lower bound on the component count."""
        return len(self.apex_indices)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def slice(self) -> slice:
        """Scan-axis slice for indexing an intensity matrix."""
        return slice(self.start_index, self.stop_index)

    def extract(self, cube: PyrogramDataCube) -> PyrogramDataCube:
        """Cut this window out of a cube."""
        return cube.window(rt_range_s=(self.start_s, self.end_s))


def _peak_sigmas(
    signal: np.ndarray,
    peaks: np.ndarray,
    *,
    max_median_multiple: float = 2.0,
) -> np.ndarray:
    """Gaussian-equivalent width of each detected peak, in scans.

    The widths are clamped to a multiple of their own median, and that clamp is
    not cosmetic. ``peak_widths`` measures where the signal falls to half the
    apex — but inside a fused homologue cluster it never does, so the routine
    walks on until it leaves the whole cluster and reports a width several times
    too large. Measured on the benchmark: median 19 scans, maximum 33, where the
    true peak width is about 12. Feeding those maxima into the window padding
    bridged 44-second gaps and merged an entire comb region into one 240-second
    window holding 42 components.
    """
    if peaks.size == 0:
        return np.zeros(0)
    widths, _, _, _ = peak_widths(signal, peaks, rel_height=0.5)
    sigmas = np.maximum(widths * _FWHM_TO_SIGMA, 1e-6)
    median = float(np.median(sigmas))
    if median > 0.0:
        sigmas = np.minimum(sigmas, max_median_multiple * median)
    return sigmas


def detect_windows(
    cube: PyrogramDataCube,
    *,
    min_prominence_fraction: float = 0.005,
    min_distance_scans: int = 2,
    merge_resolution: float = 1.0,
    padding_sigmas: float = 3.0,
    max_window_scans: int | None = 240,
) -> tuple[RetentionWindow, ...]:
    """Partition a run into windows of unresolved peaks.

    Args:
        cube: Pyrogram, ideally baseline-corrected — a rising background shifts
            peak prominences and makes the threshold behave inconsistently.
        min_prominence_fraction: Minimum peak prominence relative to the largest
            TIC value. Lowering it finds more, and more spurious, peaks.
        min_distance_scans: Minimum separation between accepted maxima.
        merge_resolution: Two neighbouring peaks are put in the same window when
            their resolution ``Δt / (2(σ₁+σ₂))`` falls below this. 1.0 marks the
            conventional boundary between fused and resolved.
        padding_sigmas: How far beyond the outermost apex a window extends, in
            units of that peak's width, so the tails are included.
        max_window_scans: Optional hard cap. A window wider than this is split at
            its deepest interior minimum, which keeps a pathological merge from
            producing one window covering the whole run.

    Returns:
        Windows ordered by retention time. Empty if the chromatogram carries no
        detectable peak.

    Raises:
        ValueError: If ``min_prominence_fraction`` or ``merge_resolution`` is not
            positive.
    """
    if min_prominence_fraction <= 0.0:
        raise ValueError(
            f"min_prominence_fraction must be > 0, got {min_prominence_fraction}"
        )
    if merge_resolution <= 0.0:
        raise ValueError(f"merge_resolution must be > 0, got {merge_resolution}")

    tic = cube.tic
    if tic.max() <= 0.0:
        return ()

    peaks, _ = find_peaks(
        tic,
        prominence=min_prominence_fraction * float(tic.max()),
        distance=max(min_distance_scans, 1),
    )
    if peaks.size == 0:
        return ()

    sigmas_scans = _peak_sigmas(tic, peaks)
    scan_period = cube.mean_scan_period_s or 1.0
    sigmas_s = sigmas_scans * scan_period
    apex_times = cube.retention_times[peaks]

    # Group neighbouring peaks that are not chromatographically resolved.
    groups: list[list[int]] = [[0]]
    for position in range(1, peaks.size):
        delta = apex_times[position] - apex_times[position - 1]
        width_sum = sigmas_s[position] + sigmas_s[position - 1]
        resolution = delta / (2.0 * width_sum) if width_sum > 0 else np.inf
        if resolution < merge_resolution:
            groups[-1].append(position)
        else:
            groups.append([position])

    windows: list[RetentionWindow] = []
    for group in groups:
        first, last = group[0], group[-1]
        pad_left = padding_sigmas * sigmas_scans[first]
        pad_right = padding_sigmas * sigmas_scans[last]
        start = max(int(np.floor(peaks[first] - pad_left)), 0)
        stop = min(int(np.ceil(peaks[last] + pad_right)) + 1, cube.n_scans)
        if stop <= start:
            start, stop = int(peaks[first]), int(peaks[first]) + 1
        windows.append(
            RetentionWindow(
                start_index=start,
                stop_index=stop,
                start_s=float(cube.retention_times[start]),
                end_s=float(cube.retention_times[stop - 1]),
                apex_indices=tuple(int(peaks[index]) for index in group),
            )
        )

    windows = _merge_overlapping(windows, cube)
    if max_window_scans is not None:
        windows = _split_oversized(windows, cube, max_window_scans)
    return tuple(windows)


def _merge_overlapping(
    windows: list[RetentionWindow], cube: PyrogramDataCube
) -> list[RetentionWindow]:
    """Fuse windows whose padded extents overlap.

    Padding can push two nominally separate groups into each other. Leaving them
    overlapping would resolve the same peak twice and double-count its area.
    """
    if not windows:
        return windows
    merged: list[RetentionWindow] = [windows[0]]
    for window in windows[1:]:
        previous = merged[-1]
        if window.start_index < previous.stop_index:
            merged[-1] = RetentionWindow(
                start_index=previous.start_index,
                stop_index=max(previous.stop_index, window.stop_index),
                start_s=previous.start_s,
                end_s=float(
                    cube.retention_times[
                        max(previous.stop_index, window.stop_index) - 1
                    ]
                ),
                apex_indices=previous.apex_indices + window.apex_indices,
            )
        else:
            merged.append(window)
    return merged


def _split_oversized(
    windows: list[RetentionWindow], cube: PyrogramDataCube, max_scans: int
) -> list[RetentionWindow]:
    """Split windows wider than ``max_scans`` at their deepest interior minimum."""
    tic = cube.tic
    result: list[RetentionWindow] = []
    queue = list(windows)

    while queue:
        window = queue.pop(0)
        if window.n_scans <= max_scans or window.n_peaks < 2:
            result.append(window)
            continue

        interior = tic[window.start_index : window.stop_index]
        # Only split between two apexes, never through one.
        candidates = [
            index
            for index in range(1, len(window.apex_indices))
            if window.apex_indices[index] > window.apex_indices[index - 1] + 1
        ]
        if not candidates:
            result.append(window)
            continue

        best_split = None
        best_value = np.inf
        for index in candidates:
            left_apex = window.apex_indices[index - 1]
            right_apex = window.apex_indices[index]
            local = tic[left_apex:right_apex]
            if local.size == 0:
                continue
            minimum_at = left_apex + int(np.argmin(local))
            if float(tic[minimum_at]) < best_value:
                best_value = float(tic[minimum_at])
                best_split = minimum_at

        if best_split is None:
            result.append(window)
            continue

        del interior
        left_apexes = tuple(a for a in window.apex_indices if a <= best_split)
        right_apexes = tuple(a for a in window.apex_indices if a > best_split)
        if not left_apexes or not right_apexes:
            result.append(window)
            continue

        queue.insert(
            0,
            RetentionWindow(
                start_index=best_split + 1,
                stop_index=window.stop_index,
                start_s=float(cube.retention_times[best_split + 1]),
                end_s=window.end_s,
                apex_indices=right_apexes,
            ),
        )
        queue.insert(
            0,
            RetentionWindow(
                start_index=window.start_index,
                stop_index=best_split + 1,
                start_s=window.start_s,
                end_s=float(cube.retention_times[best_split]),
                apex_indices=left_apexes,
            ),
        )

    return sorted(result, key=lambda window: window.start_index)
