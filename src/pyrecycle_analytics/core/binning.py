"""Mapping ragged scan data onto a rectangular m/z grid.

Raw GC/MS files store each scan as a variable-length list of (m/z, intensity)
pairs. Every chemometric method used downstream — MCR-ALS, PARAFAC2, matrix
subtraction — needs a *rectangular* matrix instead: rows are scans, columns are
fixed m/z channels. This module performs that conversion once, at ingestion
time, so no algorithm has to deal with ragged input.

Unit-resolution quadrupole data is binned onto integer m/z centres, which is the
lossless representation for that instrument class: a nominal-mass EI spectrum
carries no information between integer masses.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from data_schemas.pyrogram import MzAxisSpec
from pyrecycle_analytics.exceptions import CorruptRawDataError

__all__ = [
    "make_nominal_mz_axis",
    "make_uniform_mz_axis",
    "axis_to_spec",
    "bin_scan",
    "bin_ragged_scans",
]


def make_nominal_mz_axis(mz_low: float, mz_high: float) -> np.ndarray:
    """Integer m/z centres covering ``[mz_low, mz_high]``.

    Args:
        mz_low: Lowest m/z to keep (inclusive after rounding down).
        mz_high: Highest m/z to keep (inclusive after rounding up).

    Returns:
        Float array of integer-valued centres, shape ``(n_bins,)``.

    Raises:
        ValueError: If the window is empty or non-positive.
    """
    if mz_low <= 0.0:
        raise ValueError(f"mz_low must be > 0, got {mz_low}")
    low = int(np.floor(mz_low))
    high = int(np.ceil(mz_high))
    if high < low:
        raise ValueError(f"empty m/z window: [{mz_low}, {mz_high}]")
    return np.arange(low, high + 1, dtype=np.float64)


def make_uniform_mz_axis(mz_low: float, mz_high: float, bin_width: float) -> np.ndarray:
    """Uniformly spaced m/z centres for high-resolution data.

    Args:
        mz_low: Lower edge of the first bin.
        mz_high: Upper bound of the covered range.
        bin_width: Bin spacing in Th.

    Returns:
        Bin centres, shape ``(n_bins,)``.

    Raises:
        ValueError: If ``bin_width`` is not positive or the window is empty.
    """
    if bin_width <= 0.0:
        raise ValueError(f"bin_width must be > 0, got {bin_width}")
    if mz_high <= mz_low:
        raise ValueError(f"empty m/z window: [{mz_low}, {mz_high}]")
    n_bins = int(np.ceil((mz_high - mz_low) / bin_width))
    return mz_low + (np.arange(n_bins, dtype=np.float64) + 0.5) * bin_width


def axis_to_spec(mz_axis: np.ndarray) -> MzAxisSpec:
    """Describe an m/z axis as a serialisable :class:`MzAxisSpec`.

    Args:
        mz_axis: Monotonically increasing bin centres.

    Returns:
        Spec capturing range, bin count, spacing and whether the grid is nominal.

    Raises:
        ValueError: If the axis is empty or not strictly increasing.
    """
    mz_axis = np.asarray(mz_axis, dtype=np.float64)
    if mz_axis.size == 0:
        raise ValueError("mz_axis must not be empty")
    if mz_axis.size > 1 and np.any(np.diff(mz_axis) <= 0.0):
        raise ValueError("mz_axis must be strictly increasing")

    if mz_axis.size > 1:
        spacings = np.diff(mz_axis)
        bin_width = float(np.median(spacings))
    else:
        bin_width = 1.0

    is_nominal = bool(
        np.allclose(mz_axis, np.round(mz_axis), atol=1e-9) and abs(bin_width - 1.0) < 1e-9
    )
    half = bin_width / 2.0
    return MzAxisSpec(
        mz_low=float(mz_axis[0] - half),
        mz_high=float(mz_axis[-1] + half),
        n_bins=int(mz_axis.size),
        bin_width=bin_width,
        is_nominal=is_nominal,
    )


def _bin_edges(mz_axis: np.ndarray) -> np.ndarray:
    """Edges midway between consecutive centres, extrapolated at both ends."""
    centers = np.asarray(mz_axis, dtype=np.float64)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    inner = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (inner[0] - centers[0])
    last = centers[-1] + (centers[-1] - inner[-1])
    return np.concatenate(([first], inner, [last]))


def bin_scan(
    mz_values: np.ndarray,
    intensities: np.ndarray,
    mz_axis: np.ndarray,
) -> np.ndarray:
    """Accumulate one ragged scan onto a fixed m/z grid.

    Intensities are *summed* into their bin rather than interpolated: a mass
    spectrum is a set of discrete ion counts, and summing conserves total ion
    current, which every downstream normalisation relies on.

    Args:
        mz_values: Measured m/z positions of this scan, shape ``(n_peaks,)``.
        intensities: Matching intensities, shape ``(n_peaks,)``.
        mz_axis: Target bin centres, shape ``(n_bins,)``.

    Returns:
        Binned spectrum, shape ``(n_bins,)``. Ions outside the grid are dropped.

    Raises:
        CorruptRawDataError: If m/z and intensity arrays have different lengths.
    """
    mz_values = np.asarray(mz_values, dtype=np.float64).ravel()
    intensities = np.asarray(intensities, dtype=np.float64).ravel()
    if mz_values.size != intensities.size:
        raise CorruptRawDataError(
            f"scan has {mz_values.size} m/z values but {intensities.size} intensities"
        )

    edges = _bin_edges(mz_axis)
    n_bins = len(mz_axis)
    spectrum = np.zeros(n_bins, dtype=np.float64)
    if mz_values.size == 0:
        return spectrum

    # searchsorted with 'right' puts a value exactly on an edge into the upper bin,
    # matching the half-open [edge_i, edge_i+1) convention.
    indices = np.searchsorted(edges, mz_values, side="right") - 1
    inside = (indices >= 0) & (indices < n_bins)
    if not np.all(inside):
        indices = indices[inside]
        intensities = intensities[inside]
    if indices.size:
        np.add.at(spectrum, indices, intensities)
    return spectrum


def bin_ragged_scans(
    scans: Sequence[tuple[np.ndarray, np.ndarray]],
    mz_axis: np.ndarray,
) -> np.ndarray:
    """Bin a sequence of ragged scans into a dense intensity matrix.

    Args:
        scans: One ``(mz_values, intensities)`` pair per scan, in acquisition order.
        mz_axis: Target bin centres, shape ``(n_mz,)``.

    Returns:
        Intensity matrix of shape ``(n_scans, n_mz)``, dtype float64.

    Raises:
        CorruptRawDataError: If any scan has mismatched array lengths.
    """
    mz_axis = np.asarray(mz_axis, dtype=np.float64)
    matrix = np.zeros((len(scans), mz_axis.size), dtype=np.float64)
    for scan_index, (mz_values, intensities) in enumerate(scans):
        matrix[scan_index, :] = bin_scan(mz_values, intensities, mz_axis)
    return matrix
