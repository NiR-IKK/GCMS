"""Format detection and the single public entry point for raw-data import.

Analysts hand the platform whatever their instrument exported. Trusting the file
extension alone is not good enough — ``.cdf`` files in the wild are sometimes
netCDF-4/HDF5, and ``.xml`` files are sometimes mzML — so detection inspects the
leading bytes first and falls back to the extension only when the content is
inconclusive.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from data_schemas.enums import SourceFormat
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import UnsupportedFormatError
from pyrecycle_analytics.ingestion.andi_cdf import read_andi_cdf
from pyrecycle_analytics.ingestion.common import IngestOptions
from pyrecycle_analytics.ingestion.mzml import read_mzdata, read_mzml, read_mzxml

__all__ = ["detect_format", "read_pyrogram", "READERS", "SUPPORTED_EXTENSIONS"]

READERS: dict[SourceFormat, Callable[[Path, IngestOptions], PyrogramDataCube]] = {
    SourceFormat.MZML: read_mzml,
    SourceFormat.MZXML: read_mzxml,
    SourceFormat.MZDATA: read_mzdata,
    SourceFormat.ANDI_CDF: read_andi_cdf,
}

SUPPORTED_EXTENSIONS: dict[str, SourceFormat] = {
    ".mzml": SourceFormat.MZML,
    ".mzxml": SourceFormat.MZXML,
    ".mzdata": SourceFormat.MZDATA,
    ".cdf": SourceFormat.ANDI_CDF,
    ".nc": SourceFormat.ANDI_CDF,
}

# netCDF classic ("CDF\x01") and 64-bit offset ("CDF\x02") are what ANDI-MS uses.
_NETCDF3_MAGICS = (b"CDF\x01", b"CDF\x02", b"CDF\x05")
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_HEADER_BYTES = 2048


def detect_format(path: str | Path) -> SourceFormat:
    """Identify the raw-data container of a file.

    Detection order is magic bytes, then XML root element, then file extension.

    Args:
        path: File to inspect.

    Returns:
        The detected :class:`SourceFormat`.

    Raises:
        FileNotFoundError: If the path does not exist or is not a file.
        UnsupportedFormatError: If the content matches no supported format —
            including the common case of a netCDF-4/HDF5 file wearing a ``.cdf``
            extension, which ANDI readers cannot parse.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")

    with path.open("rb") as handle:
        header = handle.read(_HEADER_BYTES)

    if header.startswith(_NETCDF3_MAGICS):
        return SourceFormat.ANDI_CDF

    if header.startswith(_HDF5_MAGIC):
        raise UnsupportedFormatError(
            f"{path.name} is an HDF5 container (netCDF-4 or vendor format), not ANDI-MS "
            "netCDF-3. Re-export it as AIA/ANDI (netCDF classic) or as mzML."
        )

    # mzML, mzXML and mzData are all XML; distinguish by root element. The check
    # is case-insensitive because converters disagree on capitalisation.
    text = header.decode("utf-8", errors="replace").lower()
    if "<?xml" in text or text.lstrip().startswith("<"):
        for needle, source_format in (
            ("<mzml", SourceFormat.MZML),
            ("<indexedmzml", SourceFormat.MZML),
            ("<mzxml", SourceFormat.MZXML),
            ("<msrun", SourceFormat.MZXML),  # mzXML root in some 2.x writers
            ("<mzdata", SourceFormat.MZDATA),
        ):
            if needle in text:
                return source_format

    extension_match = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if extension_match is not None:
        return extension_match

    raise UnsupportedFormatError(
        f"cannot identify the format of {path.name}. Supported: "
        f"{', '.join(sorted(SUPPORTED_EXTENSIONS))} "
        "(mzML/mzXML/mzData via pyopenms, ANDI-MS netCDF-3 via scipy)."
    )


def read_pyrogram(
    path: str | Path,
    options: IngestOptions | None = None,
    **option_overrides: object,
) -> PyrogramDataCube:
    """Read any supported raw-data file into a :class:`PyrogramDataCube`.

    This is the function the rest of the platform calls; nothing outside the
    ingestion package should need to know which parser handled a file.

    Args:
        path: Raw-data file (mzML, mzXML, mzData or ANDI-MS netCDF).
        options: Full reader configuration. When omitted, one is built from
            ``option_overrides``.
        **option_overrides: Individual :class:`IngestOptions` fields, e.g.
            ``mz_range=(35, 550)`` or ``rt_range_s=(180.0, 3600.0)``. Ignored when
            ``options`` is given explicitly.

    Returns:
        A validated data cube with provenance and reader warnings attached.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnsupportedFormatError: If the format cannot be identified.
        CorruptRawDataError: If the file is structurally invalid or empty.
        MissingDependencyError: If an XML format is requested without pyopenms.
        TypeError: If an unknown option name is passed.
    """
    path = Path(path)
    if options is None:
        unknown = set(option_overrides) - set(IngestOptions.__dataclass_fields__)
        if unknown:
            raise TypeError(
                f"unknown ingest option(s): {', '.join(sorted(unknown))}. "
                f"Valid options: {', '.join(sorted(IngestOptions.__dataclass_fields__))}"
            )
        options = IngestOptions(**option_overrides)  # type: ignore[arg-type]

    source_format = detect_format(path)
    reader = READERS.get(source_format)
    if reader is None:  # pragma: no cover - defensive; READERS covers the enum
        raise UnsupportedFormatError(f"no reader registered for {source_format}")
    return reader(path, options)
