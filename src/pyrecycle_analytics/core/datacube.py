"""The central numeric container: a retention-time x m/z x intensity cube.

Everything in the platform operates on :class:`PyrogramDataCube`. It is a thin,
validated wrapper around a dense ``(n_scans, n_mz)`` matrix plus its two axes and
its metadata record. Deliberately thin, because the whole point of a bilinear
chemometric model is that the data *is* a matrix ``D`` — the wrapper only makes
sure the axes, units and provenance travel with it.

Conventions used throughout:

* retention time is always in **seconds**, monotonically increasing;
* the intensity matrix is ``D[scan, mz_channel]``, float64;
* transformations are **functional**: they return a new cube and append a
  :class:`PreprocessingStep` to the audit trail rather than mutating in place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_schemas.pyrogram import PreprocessingStep, PyrogramMetadata, round_trip_json
from pyrecycle_analytics.core.binning import axis_to_spec
from pyrecycle_analytics.exceptions import DataCubeError

__all__ = ["PyrogramDataCube"]


@dataclass
class PyrogramDataCube:
    """A single pyrogram as a dense retention-time x m/z intensity matrix.

    Attributes:
        retention_times: Scan times in seconds, shape ``(n_scans,)``, increasing.
        mz_axis: m/z bin centres, shape ``(n_mz,)``, strictly increasing.
        intensities: Signal matrix ``D``, shape ``(n_scans, n_mz)``, float64.
        metadata: Provenance, acquisition conditions and preprocessing history.
    """

    retention_times: np.ndarray
    mz_axis: np.ndarray
    intensities: np.ndarray
    metadata: PyrogramMetadata
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.retention_times = np.ascontiguousarray(self.retention_times, dtype=np.float64).ravel()
        self.mz_axis = np.ascontiguousarray(self.mz_axis, dtype=np.float64).ravel()
        self.intensities = np.ascontiguousarray(self.intensities, dtype=np.float64)
        self.validate()

    # ------------------------------------------------------------------ shape

    @property
    def n_scans(self) -> int:
        return int(self.retention_times.size)

    @property
    def n_mz(self) -> int:
        return int(self.mz_axis.size)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the intensity matrix ``(n_scans, n_mz)``."""
        return (self.n_scans, self.n_mz)

    @property
    def rt_range_s(self) -> tuple[float, float]:
        return (float(self.retention_times[0]), float(self.retention_times[-1]))

    @property
    def mean_scan_period_s(self) -> float:
        """Average spacing between scans in seconds."""
        if self.n_scans < 2:
            return 0.0
        return float(np.mean(np.diff(self.retention_times)))

    def validate(self) -> None:
        """Check internal consistency of axes and matrix.

        Raises:
            DataCubeError: On any shape mismatch, non-monotonic axis, empty axis
                or non-finite value.
        """
        if self.retention_times.size == 0:
            raise DataCubeError("retention_times must not be empty")
        if self.mz_axis.size == 0:
            raise DataCubeError("mz_axis must not be empty")
        if self.intensities.ndim != 2:
            raise DataCubeError(
                f"intensities must be 2-D (n_scans, n_mz), got {self.intensities.ndim}-D"
            )
        expected = (self.retention_times.size, self.mz_axis.size)
        if self.intensities.shape != expected:
            raise DataCubeError(
                f"intensities shape {self.intensities.shape} does not match axes {expected}"
            )
        if self.retention_times.size > 1 and np.any(np.diff(self.retention_times) <= 0.0):
            raise DataCubeError("retention_times must be strictly increasing")
        if self.mz_axis.size > 1 and np.any(np.diff(self.mz_axis) <= 0.0):
            raise DataCubeError("mz_axis must be strictly increasing")
        if not np.all(np.isfinite(self.intensities)):
            n_bad = int(np.count_nonzero(~np.isfinite(self.intensities)))
            raise DataCubeError(f"intensities contain {n_bad} non-finite value(s)")

    # ------------------------------------------------------- derived signals

    @property
    def tic(self) -> np.ndarray:
        """Total ion current: intensity summed over all m/z channels per scan."""
        if "tic" not in self._cache:
            self._cache["tic"] = self.intensities.sum(axis=1)
        return self._cache["tic"]

    @property
    def base_peak_chromatogram(self) -> np.ndarray:
        """Highest single-channel intensity per scan."""
        if "bpc" not in self._cache:
            self._cache["bpc"] = self.intensities.max(axis=1)
        return self._cache["bpc"]

    @property
    def base_peak_mz(self) -> np.ndarray:
        """m/z carrying the maximum intensity in each scan."""
        return self.mz_axis[self.intensities.argmax(axis=1)]

    @property
    def mean_spectrum(self) -> np.ndarray:
        """Intensity averaged over all scans — the run's average mass spectrum."""
        return self.intensities.mean(axis=0)

    @property
    def total_signal(self) -> float:
        """Sum of the whole matrix; the normalisation reference for yields."""
        return float(self.intensities.sum())

    # ------------------------------------------------------------ addressing

    def mz_index(self, mz: float) -> int:
        """Index of the m/z channel closest to ``mz``."""
        return int(np.argmin(np.abs(self.mz_axis - float(mz))))

    def nearest_scan(self, retention_time_s: float) -> int:
        """Index of the scan closest to ``retention_time_s``."""
        return int(np.argmin(np.abs(self.retention_times - float(retention_time_s))))

    def scan_range(self, rt_start_s: float, rt_end_s: float) -> tuple[int, int]:
        """Half-open scan index range covering ``[rt_start_s, rt_end_s]``.

        Args:
            rt_start_s: Window start in seconds.
            rt_end_s: Window end in seconds.

        Returns:
            ``(start, stop)`` suitable for slicing; always at least one scan wide
            when the window intersects the run.

        Raises:
            ValueError: If ``rt_end_s`` precedes ``rt_start_s``.
        """
        if rt_end_s < rt_start_s:
            raise ValueError(f"rt_end_s ({rt_end_s}) precedes rt_start_s ({rt_start_s})")
        start = int(np.searchsorted(self.retention_times, rt_start_s, side="left"))
        stop = int(np.searchsorted(self.retention_times, rt_end_s, side="right"))
        start = min(start, max(self.n_scans - 1, 0))
        stop = max(stop, start + 1)
        return start, min(stop, self.n_scans)

    # ------------------------------------------------------------ extraction

    def eic(self, mz: float, tolerance: float = 0.5) -> np.ndarray:
        """Extracted-ion chromatogram for a single m/z.

        Args:
            mz: Target m/z.
            tolerance: Half-width of the extraction window in Th. The default of
                0.5 selects exactly one channel on a nominal-mass grid.

        Returns:
            Chromatogram of shape ``(n_scans,)``; all-zero if no channel matches.
        """
        selected = np.abs(self.mz_axis - float(mz)) <= tolerance
        if not np.any(selected):
            return np.zeros(self.n_scans, dtype=np.float64)
        return self.intensities[:, selected].sum(axis=1)

    def eic_sum(self, mz_values: list[float] | np.ndarray, tolerance: float = 0.5) -> np.ndarray:
        """Summed extracted-ion chromatogram over several m/z values.

        Summing a compound's diagnostic ions (rather than using one) is the
        standard way to raise selectivity for trace foreign-polymer markers that
        sit on a large polyolefin background.

        Args:
            mz_values: Target m/z values.
            tolerance: Half-width of each extraction window in Th.

        Returns:
            Chromatogram of shape ``(n_scans,)``.
        """
        mz_values = np.atleast_1d(np.asarray(mz_values, dtype=np.float64))
        selected = np.zeros(self.n_mz, dtype=bool)
        for mz in mz_values:
            selected |= np.abs(self.mz_axis - mz) <= tolerance
        if not np.any(selected):
            return np.zeros(self.n_scans, dtype=np.float64)
        return self.intensities[:, selected].sum(axis=1)

    def mass_spectrum(self, scan_index: int) -> np.ndarray:
        """Mass spectrum of one scan.

        Args:
            scan_index: Index into the scan axis; negative indices count from the end.

        Returns:
            Copy of the spectrum, shape ``(n_mz,)``.

        Raises:
            IndexError: If the index is out of range.
        """
        return np.array(self.intensities[scan_index, :], dtype=np.float64)

    def mass_spectrum_at(self, retention_time_s: float) -> np.ndarray:
        """Mass spectrum of the scan closest to ``retention_time_s``."""
        return self.mass_spectrum(self.nearest_scan(retention_time_s))

    def averaged_spectrum(
        self,
        rt_start_s: float,
        rt_end_s: float,
        background_window_s: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Mean spectrum over an RT window, optionally background-subtracted.

        Averaging across a peak and subtracting a neighbouring blank window is the
        classical manual way to get a clean spectrum out of a rising baseline; it
        is kept here as the reference the automatic deconvolution is compared to.

        Args:
            rt_start_s: Start of the peak window in seconds.
            rt_end_s: End of the peak window in seconds.
            background_window_s: Optional ``(start, end)`` blank window whose mean
                spectrum is subtracted. Negative results are clipped to zero.

        Returns:
            Spectrum of shape ``(n_mz,)``.
        """
        start, stop = self.scan_range(rt_start_s, rt_end_s)
        spectrum = self.intensities[start:stop, :].mean(axis=0)
        if background_window_s is not None:
            bg_start, bg_stop = self.scan_range(*background_window_s)
            spectrum = spectrum - self.intensities[bg_start:bg_stop, :].mean(axis=0)
            spectrum = np.clip(spectrum, 0.0, None)
        return spectrum

    def window(
        self,
        rt_start_s: float,
        rt_end_s: float,
        *,
        mz_range: tuple[float, float] | None = None,
    ) -> PyrogramDataCube:
        """Cut a sub-cube out of the run.

        Local curve resolution is always applied to a window around a cluster of
        co-eluting peaks, never to the whole 60-minute run, so this is one of the
        most-used operations in the deconvolution engine.

        Args:
            rt_start_s: Window start in seconds.
            rt_end_s: Window end in seconds.
            mz_range: Optional inclusive ``(mz_low, mz_high)`` restriction.

        Returns:
            New cube sharing this cube's metadata with an added audit step.

        Raises:
            ValueError: If the window does not intersect the run.
        """
        start, stop = self.scan_range(rt_start_s, rt_end_s)
        if start >= stop:
            raise ValueError(f"window [{rt_start_s}, {rt_end_s}] s does not intersect the run")

        mz_mask = slice(None)
        if mz_range is not None:
            mz_low, mz_high = mz_range
            selected = np.flatnonzero((self.mz_axis >= mz_low) & (self.mz_axis <= mz_high))
            if selected.size == 0:
                raise ValueError(f"m/z window [{mz_low}, {mz_high}] selects no channel")
            mz_mask = slice(int(selected[0]), int(selected[-1]) + 1)

        sub_rt = self.retention_times[start:stop]
        sub_mz = self.mz_axis[mz_mask]
        sub_d = self.intensities[start:stop, mz_mask]

        step = PreprocessingStep(
            name="window",
            parameters={
                "rt_start_s": float(rt_start_s),
                "rt_end_s": float(rt_end_s),
                "mz_low": None if mz_range is None else float(mz_range[0]),
                "mz_high": None if mz_range is None else float(mz_range[1]),
            },
        )
        metadata = self.metadata.model_copy(
            update={
                "n_scans": int(sub_rt.size),
                "rt_start_s": float(sub_rt[0]),
                "rt_end_s": float(sub_rt[-1]),
                "mz_axis": axis_to_spec(sub_mz),
                "preprocessing": (*self.metadata.preprocessing, step),
            }
        )
        return PyrogramDataCube(
            retention_times=sub_rt.copy(),
            mz_axis=sub_mz.copy(),
            # An explicit copy, not ascontiguousarray: slicing a contiguous matrix
            # along the first axis yields a *view*, which ascontiguousarray hands
            # back unchanged. Writing to the window would then corrupt the parent.
            intensities=np.array(sub_d, dtype=np.float64, order="C", copy=True),
            metadata=metadata,
        )

    # ------------------------------------------------------- transformations

    def with_intensities(
        self,
        intensities: np.ndarray,
        step: PreprocessingStep | None = None,
    ) -> PyrogramDataCube:
        """Return a copy carrying a new intensity matrix.

        Args:
            intensities: Replacement matrix, shape ``(n_scans, n_mz)``.
            step: Audit record describing the transformation.

        Returns:
            New cube with the same axes and an extended preprocessing trail.

        Raises:
            DataCubeError: If the replacement matrix has the wrong shape.
        """
        intensities = np.ascontiguousarray(intensities, dtype=np.float64)
        if intensities.shape != self.shape:
            raise DataCubeError(
                f"replacement matrix shape {intensities.shape} != cube shape {self.shape}"
            )
        metadata = self.metadata if step is None else self.metadata.with_step(step)
        return PyrogramDataCube(
            retention_times=self.retention_times.copy(),
            mz_axis=self.mz_axis.copy(),
            intensities=intensities,
            metadata=metadata,
        )

    def copy(self) -> PyrogramDataCube:
        """Deep copy of axes, matrix and metadata."""
        return PyrogramDataCube(
            retention_times=self.retention_times.copy(),
            mz_axis=self.mz_axis.copy(),
            intensities=self.intensities.copy(),
            metadata=self.metadata.model_copy(deep=True),
        )

    # ------------------------------------------------------------ persistence

    def save_npz(self, path: str | Path, *, compress: bool = True) -> Path:
        """Persist the cube as a single ``.npz`` archive.

        Metadata is stored as a JSON string inside the archive, so a benchmark
        dataset is one self-describing file.

        Args:
            path: Destination path; ``.npz`` is appended if missing.
            compress: Use deflate compression (much smaller for sparse GC/MS data).

        Returns:
            The path actually written.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        write = np.savez_compressed if compress else np.savez
        with path.open("wb") as handle:
            write(
                handle,
                retention_times=self.retention_times,
                mz_axis=self.mz_axis,
                intensities=self.intensities,
                metadata_json=np.array(round_trip_json(self.metadata), dtype=object),
            )
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> PyrogramDataCube:
        """Load a cube previously written by :meth:`save_npz`.

        Args:
            path: Archive location.

        Returns:
            Reconstructed cube with validated metadata.
        """
        with np.load(Path(path), allow_pickle=True) as archive:
            metadata = PyrogramMetadata.model_validate_json(str(archive["metadata_json"].item()))
            return cls(
                retention_times=archive["retention_times"],
                mz_axis=archive["mz_axis"],
                intensities=archive["intensities"],
                metadata=metadata,
            )

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-serialisable description for logs and API responses."""
        rt_start, rt_end = self.rt_range_s
        return {
            "sample_id": self.metadata.sample.sample_id,
            "source_format": str(self.metadata.source_format),
            "n_scans": self.n_scans,
            "n_mz": self.n_mz,
            "rt_start_s": round(rt_start, 3),
            "rt_end_s": round(rt_end, 3),
            "mz_low": float(self.mz_axis[0]),
            "mz_high": float(self.mz_axis[-1]),
            "scan_period_s": round(self.mean_scan_period_s, 4),
            "total_signal": self.total_signal,
            "preprocessing": [step.name for step in self.metadata.preprocessing],
            "warnings": list(self.metadata.reader_warnings),
        }

    def __repr__(self) -> str:
        rt_start, rt_end = self.rt_range_s
        return (
            f"PyrogramDataCube(sample={self.metadata.sample.sample_id!r}, "
            f"scans={self.n_scans}, mz_channels={self.n_mz}, "
            f"rt={rt_start:.1f}-{rt_end:.1f}s, "
            f"format={self.metadata.source_format})"
        )

    def to_json_summary(self) -> str:
        """``summary()`` rendered as a JSON string."""
        return json.dumps(self.summary(), indent=2, sort_keys=True)
