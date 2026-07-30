"""Shared plumbing for all raw-data readers.

Every reader ends up doing the same three things: collect ragged scans, decide on
an m/z grid, and assemble a validated :class:`PyrogramDataCube` with full
provenance. That common tail lives here so the format-specific modules only
contain format-specific parsing.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from data_schemas.acquisition import (
    AcquisitionConditions,
    GcConditions,
    MsConditions,
    PyrolysisConditions,
    SampleMetadata,
)
from data_schemas.enums import SourceFormat
from data_schemas.pyrogram import PyrogramMetadata
from pyrecycle_analytics.core.binning import (
    axis_to_spec,
    bin_ragged_scans,
    make_nominal_mz_axis,
    make_uniform_mz_axis,
)
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import CorruptRawDataError

__all__ = ["IngestOptions", "sha256_of_file", "assemble_cube", "default_sample_metadata"]

# A GC/MS run is minutes to an hour long. If the largest retention value is below
# this threshold the axis is almost certainly in minutes, not seconds.
_MINUTES_SUSPICION_THRESHOLD_S = 180.0


@dataclass(slots=True)
class IngestOptions:
    """Reader configuration shared by all formats.

    Attributes:
        sample: Identity of the analysed material. Defaults to a record derived
            from the file name, so ingestion never blocks on missing LIMS data.
        pyrolysis: Pyrolyser settings; raw MS files do not carry them.
        gc: GC method description; raw MS files rarely carry a usable version.
        mz_range: Inclusive m/z window to keep. ``None`` uses the observed range.
        rt_range_s: Inclusive retention-time window to keep, in seconds. Useful to
            drop the solvent/gas peak at the start and the bleed ramp at the end.
        ms_level: MS level to ingest. Py-GC/MS is MS1 only.
        bin_width: m/z grid spacing. ``None`` selects nominal (unit-mass) binning,
            which is the correct choice for quadrupole EI data.
        compute_checksum: Hash the source file so results are traceable to bytes.
        max_scans: Optional hard cap, for smoke-testing very large files.
    """

    sample: SampleMetadata | None = None
    pyrolysis: PyrolysisConditions | None = None
    gc: GcConditions | None = None
    mz_range: tuple[float, float] | None = None
    rt_range_s: tuple[float, float] | None = None
    ms_level: int = 1
    bin_width: float | None = None
    compute_checksum: bool = True
    max_scans: int | None = None
    extra: dict[str, str] = field(default_factory=dict)


def sha256_of_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 digest of a file, read in chunks.

    Args:
        path: File to hash.
        chunk_size: Read block size in bytes.

    Returns:
        Lower-case hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def default_sample_metadata(path: Path | None, fallback: str = "unnamed-sample") -> SampleMetadata:
    """Minimal sample record derived from a file name.

    Args:
        path: Source file, or ``None`` for in-memory data.
        fallback: Sample id to use when no path is available.

    Returns:
        A :class:`SampleMetadata` whose ``sample_id`` is the file stem.
    """
    if path is None:
        return SampleMetadata(sample_id=fallback)
    stem = Path(path).stem or fallback
    return SampleMetadata(sample_id=stem[:64], description=f"Ingested from {Path(path).name}")


def _resolve_mz_axis(
    observed_low: float,
    observed_high: float,
    options: IngestOptions,
) -> np.ndarray:
    """Choose the m/z grid from the observed range and the reader options."""
    low, high = observed_low, observed_high
    if options.mz_range is not None:
        requested_low, requested_high = options.mz_range
        if requested_high <= requested_low:
            raise ValueError(f"mz_range {options.mz_range} is empty")
        low, high = requested_low, requested_high

    if options.bin_width is None or abs(options.bin_width - 1.0) < 1e-12:
        return make_nominal_mz_axis(low, high)
    return make_uniform_mz_axis(low, high, options.bin_width)


def assemble_cube(
    *,
    retention_times: np.ndarray,
    scans: Sequence[tuple[np.ndarray, np.ndarray]],
    source_format: SourceFormat,
    source_path: Path | None,
    options: IngestOptions,
    ms_conditions: MsConditions | None = None,
    warnings: Sequence[str] = (),
    extra: dict[str, str] | None = None,
) -> PyrogramDataCube:
    """Turn parsed ragged scans into a validated data cube.

    Handles the concerns every reader shares: retention-time sanity checks and
    sorting, unit heuristics, RT windowing, m/z grid selection, binning, and
    provenance assembly.

    Args:
        retention_times: Scan times in seconds, shape ``(n_scans,)``.
        scans: One ``(mz_values, intensities)`` pair per scan, same order as
            ``retention_times``.
        source_format: Container the data came from.
        source_path: Original file, or ``None`` for synthetic data.
        options: Reader configuration.
        ms_conditions: MS settings recovered from the file. Missing fields are
            filled from the observed data.
        warnings: Non-fatal parser complaints to record in the metadata.
        extra: Vendor attributes to preserve verbatim.

    Returns:
        A validated :class:`PyrogramDataCube`.

    Raises:
        CorruptRawDataError: If the file contains no usable scans, or the scan
            count does not match the retention-time axis.
    """
    collected_warnings = list(warnings)
    retention_times = np.asarray(retention_times, dtype=np.float64).ravel()

    if retention_times.size == 0 or len(scans) == 0:
        raise CorruptRawDataError("no scans of the requested MS level were found")
    if retention_times.size != len(scans):
        raise CorruptRawDataError(
            f"{retention_times.size} retention times but {len(scans)} scans"
        )

    # Some converters emit -1 as a "no retention time recorded" placeholder. A
    # pyrogram without a usable time axis cannot be interpreted at all — every
    # marker is identified by where it elutes — so this is fatal, and it is caught
    # here rather than surfacing later as a confusing schema validation error.
    if not np.all(np.isfinite(retention_times)):
        raise CorruptRawDataError("retention times contain non-finite values")
    if np.any(retention_times < 0.0):
        n_negative = int(np.count_nonzero(retention_times < 0.0))
        raise CorruptRawDataError(
            f"{n_negative} of {retention_times.size} scans carry a negative retention "
            "time (commonly a '-1' placeholder for 'not recorded'). The file does not "
            "contain a usable time axis; re-export it as mzML or ANDI-MS netCDF."
        )

    # Retention axis must be increasing; some vendor exports interleave scan types.
    if retention_times.size > 1 and np.any(np.diff(retention_times) <= 0.0):
        order = np.argsort(retention_times, kind="stable")
        retention_times = retention_times[order]
        scans = [scans[int(index)] for index in order]
        collected_warnings.append("retention times were not sorted; scans were reordered")
        duplicates = int(np.count_nonzero(np.diff(retention_times) == 0.0))
        if duplicates:
            keep = np.concatenate(([True], np.diff(retention_times) > 0.0))
            retention_times = retention_times[keep]
            scans = [scan for scan, keep_it in zip(scans, keep, strict=True) if keep_it]
            collected_warnings.append(
                f"dropped {duplicates} scan(s) with duplicate retention times"
            )

    if float(retention_times[-1]) < _MINUTES_SUSPICION_THRESHOLD_S and retention_times.size > 50:
        collected_warnings.append(
            f"retention axis ends at {retention_times[-1]:.2f}; unusually short for a "
            "GC run in seconds — verify the file's time unit"
        )

    if options.rt_range_s is not None:
        rt_low, rt_high = options.rt_range_s
        keep = (retention_times >= rt_low) & (retention_times <= rt_high)
        if not np.any(keep):
            raise CorruptRawDataError(
                f"rt_range_s {options.rt_range_s} selects none of the "
                f"{retention_times.size} scans"
            )
        retention_times = retention_times[keep]
        scans = [scan for scan, keep_it in zip(scans, keep, strict=True) if keep_it]

    if options.max_scans is not None and retention_times.size > options.max_scans:
        collected_warnings.append(
            f"truncated to the first {options.max_scans} of {retention_times.size} scans"
        )
        retention_times = retention_times[: options.max_scans]
        scans = list(scans[: options.max_scans])

    non_empty = [mz for mz, _ in scans if mz.size]
    if not non_empty:
        raise CorruptRawDataError("all scans are empty; nothing to bin")
    observed_low = float(min(float(np.min(mz)) for mz in non_empty))
    observed_high = float(max(float(np.max(mz)) for mz in non_empty))

    mz_axis = _resolve_mz_axis(observed_low, observed_high, options)
    intensities = bin_ragged_scans(scans, mz_axis)

    if not np.any(intensities):
        collected_warnings.append(
            "binned matrix is all zeros; check that mz_range overlaps the acquired range"
        )

    scan_period = (
        float(np.mean(np.diff(retention_times))) if retention_times.size > 1 else 0.0
    )
    observed_ms = MsConditions(
        mz_low=float(mz_axis[0]),
        mz_high=float(mz_axis[-1]),
        scan_rate_hz=1.0 / scan_period if scan_period > 0.0 else 1.0,
    )
    if ms_conditions is not None:
        observed_ms = ms_conditions.model_copy(
            update={
                "mz_low": float(mz_axis[0]),
                "mz_high": float(mz_axis[-1]),
                "scan_rate_hz": observed_ms.scan_rate_hz,
            }
        )

    acquisition = AcquisitionConditions(
        pyrolysis=options.pyrolysis or PyrolysisConditions(),
        gc=options.gc or GcConditions(),
        ms=observed_ms,
    )

    checksum: str | None = None
    if source_path is not None and options.compute_checksum:
        checksum = sha256_of_file(source_path)

    merged_extra = {**(extra or {}), **options.extra}

    metadata = PyrogramMetadata(
        sample=options.sample or default_sample_metadata(source_path),
        acquisition=acquisition,
        source_format=source_format,
        source_path=source_path,
        source_checksum=checksum,
        n_scans=int(retention_times.size),
        mz_axis=axis_to_spec(mz_axis),
        rt_start_s=float(retention_times[0]),
        rt_end_s=float(retention_times[-1]),
        reader_warnings=tuple(collected_warnings),
        extra=merged_extra,
    )

    return PyrogramDataCube(
        retention_times=retention_times,
        mz_axis=mz_axis,
        intensities=intensities,
        metadata=metadata,
    )
