"""Reader for ANDI-MS / AIA netCDF chromatography files (``.CDF``).

ANDI-MS (ASTM E1947, "Analytical Data Interchange Protocol for Chromatographic
Data") is the vendor-neutral GC/MS exchange format that every instrument in a
polymer lab can export: Agilent ChemStation/MassHunter, Shimadzu GCMSsolution,
Frontier Lab pyrolyser front-ends, Thermo Xcalibur.

Implementation note — why not pyopenms here: pyopenms covers the proteomics
XML family (mzML, mzXML, mzData) and has no ANDI/netCDF loader. The ANDI
container is netCDF classic (netCDF-3), which ``scipy.io.netcdf_file`` reads
directly, so the reader needs no additional dependency and no vendor SDK.

Data model of the format
------------------------
Scans are stored flattened. ``mass_values`` and ``intensity_values`` are one long
point list; ``scan_index`` gives each scan's offset into that list and
``point_count`` its length::

    scan k  ->  points[scan_index[k] : scan_index[k] + point_count[k]]

``scan_acquisition_time`` holds the retention time per scan, normally in seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file

from data_schemas.acquisition import MsConditions
from data_schemas.enums import AcquisitionMode, IonisationMode, Polarity, SourceFormat
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import CorruptRawDataError
from pyrecycle_analytics.ingestion.common import IngestOptions, assemble_cube

__all__ = ["read_andi_cdf", "ANDI_REQUIRED_VARIABLES"]

ANDI_REQUIRED_VARIABLES = ("scan_acquisition_time", "mass_values", "intensity_values")
"""Variables without which the file cannot be interpreted as ANDI-MS."""

# Global attributes worth preserving; ANDI files carry them as netCDF char arrays.
_INTERESTING_ATTRIBUTES = (
    "dataset_completeness",
    "ms_template_revision",
    "netcdf_revision",
    "languages",
    "administrative_comments",
    "dataset_origin",
    "dataset_owner",
    "dataset_date_time_stamp",
    "injection_date_time_stamp",
    "experiment_title",
    "operator_name",
    "source_file_reference",
    "source_file_format",
    "experiment_type",
    "sample_name",
    "sample_id",
    "detector_unit",
    "detector_name",
    "test_ionization_mode",
    "test_ionization_polarity",
    "test_detector_type",
    "test_ms_scan_mode",
    "test_separation_type",
    "instrument_name",
    "instrument_model",
    "instrument_mfr",
)

_POLARITY_KEYWORDS = {
    "positive": Polarity.POSITIVE,
    "pos": Polarity.POSITIVE,
    "negative": Polarity.NEGATIVE,
    "neg": Polarity.NEGATIVE,
}

_IONISATION_KEYWORDS = {
    "electron impact": IonisationMode.EI,
    "electron ionization": IonisationMode.EI,
    "ei": IonisationMode.EI,
    "chemical ionization": IonisationMode.CI,
    "ci": IonisationMode.CI,
}


def _decode(value: Any) -> str:
    """Best-effort conversion of a netCDF attribute to a clean string."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00 \t\r\n")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return value.tobytes().decode("utf-8", errors="replace").strip("\x00 \t\r\n")
        if value.size == 1:
            return str(value.item())
        return ", ".join(str(item) for item in value.tolist())
    return str(value).strip()


def _variable_data(dataset: Any, name: str) -> np.ndarray | None:
    """Read a netCDF variable, applying ANDI ``scale_factor`` / ``add_offset``.

    ANDI writers are allowed to store intensities as scaled integers. Ignoring the
    scaling silently distorts every ratio the platform computes, so it is applied
    here at the earliest possible point.
    """
    variable = dataset.variables.get(name)
    if variable is None:
        return None
    data = np.array(variable[:], dtype=np.float64)
    attributes = getattr(variable, "_attributes", {}) or {}
    scale = attributes.get("scale_factor")
    offset = attributes.get("add_offset")
    if scale is not None:
        data = data * float(np.asarray(scale).ravel()[0])
    if offset is not None:
        data = data + float(np.asarray(offset).ravel()[0])
    return data


def _retention_times_in_seconds(dataset: Any, warnings: list[str]) -> np.ndarray:
    """Retention axis, converted to seconds if the file declares minutes."""
    variable = dataset.variables.get("scan_acquisition_time")
    if variable is None:
        raise CorruptRawDataError("ANDI file has no 'scan_acquisition_time' variable")

    times = np.array(variable[:], dtype=np.float64).ravel()
    units = _decode(getattr(variable, "_attributes", {}).get("units", b"")).lower()
    if units.startswith("min"):
        times = times * 60.0
        warnings.append("scan_acquisition_time was in minutes; converted to seconds")
    elif units and not units.startswith("sec"):
        warnings.append(f"unrecognised time unit {units!r}; values assumed to be seconds")
    return times


def _scan_offsets(
    dataset: Any,
    n_scans: int,
    n_points: int,
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve ``scan_index`` / ``point_count`` into validated offsets and lengths."""
    scan_index_raw = dataset.variables.get("scan_index")
    point_count_raw = dataset.variables.get("point_count")

    if scan_index_raw is not None:
        offsets = np.array(scan_index_raw[:], dtype=np.int64).ravel()
    elif point_count_raw is not None:
        counts_tmp = np.array(point_count_raw[:], dtype=np.int64).ravel()
        offsets = np.concatenate(([0], np.cumsum(counts_tmp)[:-1]))
        warnings.append("no 'scan_index'; offsets derived from 'point_count'")
    else:
        # Last resort: a rectangular file with equal-length scans.
        if n_points % n_scans != 0:
            raise CorruptRawDataError(
                "ANDI file lacks both 'scan_index' and 'point_count', and the point "
                f"list ({n_points}) is not divisible by the scan count ({n_scans})"
            )
        per_scan = n_points // n_scans
        offsets = np.arange(n_scans, dtype=np.int64) * per_scan
        warnings.append(
            f"neither 'scan_index' nor 'point_count' present; assumed {per_scan} points per scan"
        )

    if offsets.size != n_scans:
        raise CorruptRawDataError(
            f"'scan_index' has {offsets.size} entries but there are {n_scans} scans"
        )

    if point_count_raw is not None:
        counts = np.array(point_count_raw[:], dtype=np.int64).ravel()
        if counts.size != n_scans:
            raise CorruptRawDataError(
                f"'point_count' has {counts.size} entries but there are {n_scans} scans"
            )
    else:
        counts = np.diff(np.concatenate((offsets, [n_points])))

    if np.any(offsets < 0) or np.any(counts < 0):
        raise CorruptRawDataError("'scan_index'/'point_count' contain negative values")

    ends = offsets + counts
    if np.any(ends > n_points):
        overflow = int(np.count_nonzero(ends > n_points))
        warnings.append(
            f"{overflow} scan(s) index past the end of the point list; they were clipped"
        )
        counts = np.minimum(counts, np.maximum(n_points - offsets, 0))

    return offsets, counts


def _ms_conditions_from_attributes(attributes: dict[str, str]) -> MsConditions:
    """Recover what the ANDI global attributes say about the MS settings."""
    polarity = Polarity.UNKNOWN
    polarity_text = attributes.get("test_ionization_polarity", "").lower()
    for keyword, polarity_value in _POLARITY_KEYWORDS.items():
        if keyword in polarity_text:
            polarity = polarity_value
            break

    ionisation = IonisationMode.UNKNOWN
    ionisation_text = attributes.get("test_ionization_mode", "").lower()
    for keyword, ionisation_value in _IONISATION_KEYWORDS.items():
        if keyword in ionisation_text:
            ionisation = ionisation_value
            break

    scan_mode = attributes.get("test_ms_scan_mode", "").lower()
    if "mass scan" in scan_mode or "full" in scan_mode:
        acquisition_mode = AcquisitionMode.FULL_SCAN
    elif "selected ion" in scan_mode or "sim" in scan_mode:
        acquisition_mode = AcquisitionMode.SIM
    else:
        acquisition_mode = AcquisitionMode.UNKNOWN

    # mz_low/mz_high/scan_rate_hz are overwritten from the actual data downstream.
    return MsConditions(
        ionisation=ionisation,
        polarity=polarity,
        acquisition_mode=acquisition_mode,
    )


def read_andi_cdf(
    path: str | Path,
    options: IngestOptions | None = None,
) -> PyrogramDataCube:
    """Read an ANDI-MS / AIA netCDF file into a data cube.

    Args:
        path: Path to the ``.cdf`` / ``.CDF`` file.
        options: Reader configuration; defaults are used when omitted.

    Returns:
        A validated :class:`PyrogramDataCube` with ANDI global attributes
        preserved in ``metadata.extra``.

    Raises:
        CorruptRawDataError: If required variables are missing, the point list is
            inconsistent with the scan index, or no usable scan remains.
    """
    path = Path(path)
    options = options or IngestOptions()
    warnings: list[str] = []

    with netcdf_file(str(path), "r", mmap=False) as dataset:
        available = set(dataset.variables)
        missing = [name for name in ANDI_REQUIRED_VARIABLES if name not in available]
        if missing:
            raise CorruptRawDataError(
                f"{path.name} is not a readable ANDI-MS file; missing variable(s): "
                f"{', '.join(missing)}"
            )

        attributes = {
            name: _decode(value)
            for name, value in (getattr(dataset, "_attributes", {}) or {}).items()
            if name in _INTERESTING_ATTRIBUTES
        }

        retention_times = _retention_times_in_seconds(dataset, warnings)
        mass_values = _variable_data(dataset, "mass_values")
        intensity_values = _variable_data(dataset, "intensity_values")
        assert mass_values is not None and intensity_values is not None  # guarded above

        if mass_values.size != intensity_values.size:
            raise CorruptRawDataError(
                f"'mass_values' has {mass_values.size} entries but 'intensity_values' "
                f"has {intensity_values.size}"
            )

        offsets, counts = _scan_offsets(
            dataset, retention_times.size, mass_values.size, warnings
        )
        total_intensity = _variable_data(dataset, "total_intensity")

    scans: list[tuple[np.ndarray, np.ndarray]] = []
    for offset, count in zip(offsets.tolist(), counts.tolist(), strict=True):
        stop = offset + count
        scans.append((mass_values[offset:stop], intensity_values[offset:stop]))

    empty = sum(1 for mz, _ in scans if mz.size == 0)
    if empty:
        warnings.append(f"{empty} of {len(scans)} scans contain no ions")

    cube = assemble_cube(
        retention_times=retention_times,
        scans=scans,
        source_format=SourceFormat.ANDI_CDF,
        source_path=path,
        options=options,
        ms_conditions=_ms_conditions_from_attributes(attributes),
        warnings=warnings,
        extra=attributes,
    )

    # The file's own TIC is an independent check on our binning: a mismatch means
    # ions fell outside the m/z grid or the scan index was misaligned.
    if total_intensity is not None and total_intensity.size == cube.n_scans:
        declared = float(np.sum(total_intensity))
        binned = cube.total_signal
        if declared > 0.0 and abs(binned - declared) / declared > 0.01:
            cube.metadata.reader_warnings = (
                *cube.metadata.reader_warnings,
                f"binned total ion current deviates from the file's 'total_intensity' by "
                f"{100.0 * (binned - declared) / declared:+.2f}% — ions may lie outside "
                "the selected m/z range",
            )

    return cube
