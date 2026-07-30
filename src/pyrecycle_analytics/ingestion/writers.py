"""Writers for the two supported exchange formats.

Two reasons this exists rather than being read-only:

1. **Testability.** The synthetic pyrogram generator produces cubes in memory.
   Writing them to real mzML / ANDI-CDF files and reading them back is the only
   way to test the ingestion layer against the actual parsers instead of a mock.
2. **Interoperability.** Deconvolved or matrix-subtracted pyrograms need to leave
   the platform so an analyst can open them in the vendor software or NIST MS
   Search they already trust.

Both writers emit centroid data: only channels with non-zero intensity are
stored, which is how quadrupole GC/MS data is represented natively and which
keeps files small (a binned cube is >95 % zeros).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file

from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.ingestion.mzml import _import_pyopenms

__all__ = ["write_mzml", "write_andi_cdf"]

_ANDI_TEMPLATE_REVISION = "1.0.1"
_ANDI_NETCDF_REVISION = "2.3.2"


def _sparse_scan(spectrum: np.ndarray, mz_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop zero channels from one binned spectrum."""
    non_zero = np.flatnonzero(spectrum > 0.0)
    return mz_axis[non_zero], spectrum[non_zero]


def write_mzml(
    cube: PyrogramDataCube,
    path: str | Path,
    *,
    keep_zeros: bool = False,
) -> Path:
    """Write a data cube as mzML.

    Args:
        cube: Pyrogram to serialise.
        path: Destination path; ``.mzML`` is appended when missing.
        keep_zeros: Store every m/z channel including empty ones. Off by default
            because it inflates files by more than an order of magnitude.

    Returns:
        The path actually written.

    Raises:
        MissingDependencyError: If pyopenms is not installed.
    """
    oms = _import_pyopenms()
    path = Path(path)
    if path.suffix.lower() != ".mzml":
        path = path.with_suffix(".mzML")
    path.parent.mkdir(parents=True, exist_ok=True)

    experiment = oms.MSExperiment()
    settings = oms.InstrumentSettings()
    settings.setPolarity(oms.IonSource.Polarity.POSITIVE)
    settings.setScanMode(oms.InstrumentSettings.ScanMode.MASSSPECTRUM)

    for scan_index in range(cube.n_scans):
        spectrum = oms.MSSpectrum()
        spectrum.setRT(float(cube.retention_times[scan_index]))
        spectrum.setMSLevel(1)
        spectrum.setInstrumentSettings(settings)
        spectrum.setNativeID(f"scan={scan_index + 1}")
        row = cube.intensities[scan_index, :]
        if keep_zeros:
            mz_values, intensities = cube.mz_axis, row
        else:
            mz_values, intensities = _sparse_scan(row, cube.mz_axis)
        spectrum.set_peaks((mz_values.astype(np.float64), intensities.astype(np.float64)))
        experiment.addSpectrum(spectrum)

    instrument = oms.Instrument()
    instrument.setName(
        (cube.metadata.acquisition.instrument_model or "PyRecycle-Analytics synthetic")[:200]
    )
    experiment.setInstrument(instrument)

    handler = oms.MzMLFile()
    setter = getattr(handler, "setLogType", None)
    if setter is not None:
        setter(oms.LogType.NONE)
    handler.store(str(path), experiment)
    return path


def _set_text_attribute(dataset: Any, name: str, value: str) -> None:
    """Attach a netCDF-3 character global attribute."""
    setattr(dataset, name, value.encode("utf-8") if value else b"")


def write_andi_cdf(
    cube: PyrogramDataCube,
    path: str | Path,
    *,
    keep_zeros: bool = False,
) -> Path:
    """Write a data cube as an ANDI-MS / AIA netCDF file.

    Produces the variable set that :func:`~pyrecycle_analytics.ingestion.andi_cdf.read_andi_cdf`
    and vendor software expect: a flattened point list plus per-scan offsets,
    counts, retention times and total ion current.

    Args:
        cube: Pyrogram to serialise.
        path: Destination path; ``.cdf`` is appended when missing.
        keep_zeros: Store every m/z channel including empty ones.

    Returns:
        The path actually written.
    """
    path = Path(path)
    if path.suffix.lower() not in {".cdf", ".nc"}:
        path = path.with_suffix(".cdf")
    path.parent.mkdir(parents=True, exist_ok=True)

    mass_chunks: list[np.ndarray] = []
    intensity_chunks: list[np.ndarray] = []
    point_counts = np.empty(cube.n_scans, dtype=np.int32)

    for scan_index in range(cube.n_scans):
        row = cube.intensities[scan_index, :]
        if keep_zeros:
            mz_values, intensities = cube.mz_axis, row
        else:
            mz_values, intensities = _sparse_scan(row, cube.mz_axis)
        mass_chunks.append(np.asarray(mz_values, dtype=np.float32))
        intensity_chunks.append(np.asarray(intensities, dtype=np.float32))
        point_counts[scan_index] = mz_values.size

    mass_values = (
        np.concatenate(mass_chunks) if mass_chunks else np.zeros(0, dtype=np.float32)
    )
    intensity_values = (
        np.concatenate(intensity_chunks) if intensity_chunks else np.zeros(0, dtype=np.float32)
    )
    scan_index_offsets = np.concatenate(([0], np.cumsum(point_counts)[:-1])).astype(np.int32)

    # netCDF-3 forbids zero-length dimensions, so an all-empty cube gets one dummy point.
    if mass_values.size == 0:
        mass_values = np.zeros(1, dtype=np.float32)
        intensity_values = np.zeros(1, dtype=np.float32)

    sample = cube.metadata.sample
    with netcdf_file(str(path), "w") as dataset:
        _set_text_attribute(dataset, "dataset_completeness", "C1+C2")
        _set_text_attribute(dataset, "ms_template_revision", _ANDI_TEMPLATE_REVISION)
        _set_text_attribute(dataset, "netcdf_revision", _ANDI_NETCDF_REVISION)
        _set_text_attribute(dataset, "languages", "English")
        _set_text_attribute(dataset, "experiment_type", "Centroided Mass Spectrum")
        _set_text_attribute(dataset, "test_separation_type", "Gas-Solid Chromatography")
        _set_text_attribute(dataset, "test_ms_scan_mode", "Mass Scan")
        _set_text_attribute(dataset, "test_ionization_mode", "Electron Impact")
        _set_text_attribute(dataset, "test_ionization_polarity", "Positive Polarity")
        _set_text_attribute(dataset, "test_detector_type", "Electron Multiplier")
        _set_text_attribute(dataset, "detector_unit", "Arbitrary Intensity Units")
        _set_text_attribute(dataset, "sample_name", sample.sample_id)
        _set_text_attribute(dataset, "experiment_title", sample.description or sample.sample_id)
        _set_text_attribute(dataset, "operator_name", sample.operator or "")
        _set_text_attribute(dataset, "dataset_origin", "PyRecycle-Analytics")
        _set_text_attribute(
            dataset,
            "dataset_date_time_stamp",
            datetime.now(UTC).strftime("%Y%m%d%H%M%S%z"),
        )

        dataset.createDimension("scan_number", cube.n_scans)
        dataset.createDimension("point_number", int(mass_values.size))

        acquisition_time = dataset.createVariable(
            "scan_acquisition_time", "d", ("scan_number",)
        )
        acquisition_time[:] = cube.retention_times.astype(np.float64)
        acquisition_time.units = b"Seconds"

        offsets_variable = dataset.createVariable("scan_index", "i", ("scan_number",))
        offsets_variable[:] = scan_index_offsets

        counts_variable = dataset.createVariable("point_count", "i", ("scan_number",))
        counts_variable[:] = point_counts

        total_intensity = dataset.createVariable("total_intensity", "d", ("scan_number",))
        total_intensity[:] = cube.tic.astype(np.float64)
        total_intensity.units = b"Arbitrary Intensity Units"

        mass_variable = dataset.createVariable("mass_values", "f", ("point_number",))
        mass_variable[:] = mass_values
        mass_variable.units = b"M/Z"

        intensity_variable = dataset.createVariable("intensity_values", "f", ("point_number",))
        intensity_variable[:] = intensity_values
        intensity_variable.units = b"Arbitrary Intensity Units"

    return path
