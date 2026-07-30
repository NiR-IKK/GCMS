"""Readers for the open MS XML formats (mzML, mzXML, mzData) via pyopenms.

pyopenms is used strictly as a parser here — file I/O, nothing else. All
chromatographic and chemometric logic in this platform works on the resulting
:class:`PyrogramDataCube`, never on OpenMS objects, so the dependency stays
replaceable and confined to this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from data_schemas.acquisition import MsConditions
from data_schemas.enums import AcquisitionMode, IonisationMode, Polarity, SourceFormat
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import CorruptRawDataError, MissingDependencyError
from pyrecycle_analytics.ingestion.common import IngestOptions, assemble_cube

__all__ = ["read_mzml", "read_mzxml", "read_mzdata", "pyopenms_available"]


@lru_cache(maxsize=1)
def _import_pyopenms() -> Any:
    """Import and configure pyopenms on first use.

    Ingestion of the XML formats is the only feature that needs it, so importing
    lazily keeps the chemometric core usable in a minimal environment (and keeps
    test collection fast — the OpenMS shared library is ~100 MB).

    Returns:
        The imported ``pyopenms`` module.

    Raises:
        MissingDependencyError: If the package is not installed.
    """
    try:
        import pyopenms  # noqa: PLC0415 - deliberate lazy import
    except ImportError as error:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            package="pyopenms",
            purpose="reading mzML / mzXML / mzData raw data",
            extra="io",
        ) from error

    _silence_openms_info_stream(pyopenms)
    return pyopenms


def _silence_openms_info_stream(oms: Any) -> None:
    """Detach the OpenMS INFO log from stdout.

    OpenMS writes progress lines ("N spectra ... stored.") straight to the C++
    ``cout``, which bypasses Python's stdout and surfaces at unpredictable points —
    including in the middle of CLI output or a JSON response. Problems reach the
    caller as exceptions and as ``metadata.reader_warnings`` instead, so the info
    stream carries nothing we need. Warnings and errors are left connected.
    """
    try:
        handler = oms.LogConfigHandler()
        handler.configure(handler.parse([b"INFO remove cout"]))
    except (AttributeError, RuntimeError):  # pragma: no cover - pyopenms version drift
        pass


def pyopenms_available() -> bool:
    """True when the optional pyopenms dependency can be imported."""
    try:
        _import_pyopenms()
    except MissingDependencyError:
        return False
    return True


def _polarity(spectrum: Any, oms: Any) -> Polarity:
    """Map an OpenMS polarity code onto our vocabulary."""
    try:
        code = spectrum.getInstrumentSettings().getPolarity()
    except (AttributeError, RuntimeError):
        return Polarity.UNKNOWN
    codes = oms.IonSource.Polarity
    if code == codes.POSITIVE:
        return Polarity.POSITIVE
    if code == codes.NEGATIVE:
        return Polarity.NEGATIVE
    return Polarity.UNKNOWN


def _acquisition_mode(spectrum: Any, oms: Any) -> AcquisitionMode:
    """Map an OpenMS scan mode onto our vocabulary."""
    try:
        code = spectrum.getInstrumentSettings().getScanMode()
    except (AttributeError, RuntimeError):
        return AcquisitionMode.UNKNOWN
    modes = oms.InstrumentSettings.ScanMode
    if code == modes.MASSSPECTRUM:
        return AcquisitionMode.FULL_SCAN
    if code == modes.SIM:
        return AcquisitionMode.SIM
    return AcquisitionMode.UNKNOWN


def _instrument_extra(experiment: Any) -> dict[str, str]:
    """Pull whatever instrument description the file carries, defensively.

    Raw files from GC/MS vendors are frequently written by converters that omit
    large parts of the mzML schema, so every accessor here is optional.
    """
    extra: dict[str, str] = {}
    try:
        instrument = experiment.getInstrument()
        for key, getter in (
            ("instrument_name", "getName"),
            ("instrument_vendor", "getVendor"),
            ("instrument_model", "getModel"),
            ("instrument_customisations", "getCustomizations"),
        ):
            accessor = getattr(instrument, getter, None)
            if accessor is None:
                continue
            value = accessor()
            text = value.decode() if isinstance(value, bytes) else str(value)
            if text.strip():
                extra[key] = text.strip()[:200]
    except (AttributeError, RuntimeError):  # pragma: no cover - vendor dependent
        pass

    try:
        source_files = experiment.getSourceFiles()
        names = []
        for source_file in source_files:
            name = source_file.getNameOfFile()
            names.append(name.decode() if isinstance(name, bytes) else str(name))
        if names:
            extra["original_source_files"] = "; ".join(names)[:400]
    except (AttributeError, RuntimeError):  # pragma: no cover - vendor dependent
        pass

    return extra


def _read_xml_experiment(
    path: Path,
    handler_name: str,
    source_format: SourceFormat,
    options: IngestOptions,
) -> PyrogramDataCube:
    """Shared body of the three XML readers.

    Args:
        path: File to read.
        handler_name: pyopenms file-handler class, e.g. ``"MzMLFile"``.
        source_format: Format tag recorded in the metadata.
        options: Reader configuration.

    Returns:
        A validated data cube.

    Raises:
        CorruptRawDataError: If the parser fails or no spectrum of the requested
            MS level is present.
        MissingDependencyError: If pyopenms is not installed.
    """
    oms = _import_pyopenms()
    handler = getattr(oms, handler_name)()

    # OpenMS otherwise prints progress bars to stdout, which corrupts CLI output.
    with_log = getattr(handler, "setLogType", None)
    if with_log is not None:
        with_log(oms.LogType.NONE)

    # Push the MS-level filter into the parser so MSn scans of a mixed file are
    # never materialised. The m/z restriction is *not* pushed down: pyopenms 3.x
    # exposes DRange1 without setters, so the range is applied when the ragged
    # scans are binned onto the target grid instead (ions outside it are dropped).
    peak_options = handler.getOptions()
    peak_options.setMSLevels([int(options.ms_level)])
    handler.setOptions(peak_options)

    experiment = oms.MSExperiment()
    try:
        handler.load(str(path), experiment)
    except Exception as error:  # pyopenms raises bare RuntimeError subclasses
        raise CorruptRawDataError(f"{handler_name} could not parse {path.name}: {error}") from error

    warnings: list[str] = []
    retention_times: list[float] = []
    scans: list[tuple[np.ndarray, np.ndarray]] = []
    skipped_levels: set[int] = set()
    profile_like = 0

    for spectrum in experiment:
        level = spectrum.getMSLevel()
        if level != options.ms_level:
            skipped_levels.add(int(level))
            continue
        mz_values, intensities = spectrum.get_peaks()
        mz_values = np.asarray(mz_values, dtype=np.float64)
        intensities = np.asarray(intensities, dtype=np.float64)
        # A nominal-mass EI scan has tens of ions; thousands means profile data,
        # which must be centroided before ratios mean anything.
        if mz_values.size > 2000:
            profile_like += 1
        retention_times.append(float(spectrum.getRT()))
        scans.append((mz_values, intensities))

    if not scans:
        raise CorruptRawDataError(
            f"{path.name} contains no MS{options.ms_level} spectra"
            + (f" (found levels: {sorted(skipped_levels)})" if skipped_levels else "")
        )
    if skipped_levels:
        warnings.append(
            f"ignored spectra at MS level(s) {sorted(skipped_levels)}; "
            f"only MS{options.ms_level} was requested"
        )
    if profile_like:
        warnings.append(
            f"{profile_like} scan(s) carry >2000 points and look like profile data; "
            "centroiding is recommended before quantitative marker work"
        )

    first = experiment.getSpectrum(0) if experiment.getNrSpectra() else None
    ms_conditions = MsConditions(
        ionisation=IonisationMode.EI,  # Py-GC/MS is EI; mzML rarely states it explicitly.
        polarity=_polarity(first, oms) if first is not None else Polarity.UNKNOWN,
        acquisition_mode=(
            _acquisition_mode(first, oms) if first is not None else AcquisitionMode.UNKNOWN
        ),
    )

    return assemble_cube(
        retention_times=np.asarray(retention_times, dtype=np.float64),
        scans=scans,
        source_format=source_format,
        source_path=path,
        options=options,
        ms_conditions=ms_conditions,
        warnings=warnings,
        extra=_instrument_extra(experiment),
    )


def read_mzml(path: str | Path, options: IngestOptions | None = None) -> PyrogramDataCube:
    """Read an mzML file (HUPO-PSI standard, the preferred exchange format).

    Args:
        path: Path to the ``.mzML`` file.
        options: Reader configuration; defaults are used when omitted.

    Returns:
        A validated :class:`PyrogramDataCube`.

    Raises:
        CorruptRawDataError: If parsing fails or no MS1 spectra are present.
        MissingDependencyError: If pyopenms is not installed.
    """
    return _read_xml_experiment(
        Path(path), "MzMLFile", SourceFormat.MZML, options or IngestOptions()
    )


def read_mzxml(path: str | Path, options: IngestOptions | None = None) -> PyrogramDataCube:
    """Read an mzXML file (legacy ISB format still emitted by some converters).

    Args:
        path: Path to the ``.mzXML`` file.
        options: Reader configuration; defaults are used when omitted.

    Returns:
        A validated :class:`PyrogramDataCube`.
    """
    return _read_xml_experiment(
        Path(path), "MzXMLFile", SourceFormat.MZXML, options or IngestOptions()
    )


def read_mzdata(path: str | Path, options: IngestOptions | None = None) -> PyrogramDataCube:
    """Read an mzData file (superseded by mzML, occasionally found in archives).

    Support is best-effort. mzData is a dead format and OpenMS's handler does not
    round-trip retention time, so files that omit the ``TimeInMinutes`` parameter
    are rejected with a clear error rather than yielding a cube with a meaningless
    time axis. Prefer mzML or ANDI-MS for anything quantitative.

    Args:
        path: Path to the ``.mzData`` file.
        options: Reader configuration; defaults are used when omitted.

    Returns:
        A validated :class:`PyrogramDataCube`.

    Raises:
        CorruptRawDataError: If the file carries no usable retention times.
    """
    return _read_xml_experiment(
        Path(path), "MzDataFile", SourceFormat.MZDATA, options or IngestOptions()
    )
