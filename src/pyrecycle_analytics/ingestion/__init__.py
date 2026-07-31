"""Raw-data ingestion layer.

Turns instrument files into validated :class:`PyrogramDataCube` objects:

* **mzML / mzXML / mzData** — parsed with ``pyopenms``.
* **ANDI-MS / AIA netCDF (.CDF)** — parsed with ``scipy.io.netcdf_file``;
  ``pyopenms`` has no netCDF loader, and ANDI is netCDF-3, so no extra
  dependency is needed.

Typical use::

    from pyrecycle_analytics.ingestion import read_pyrogram

    cube = read_pyrogram("PCR_LDPE_batch7.CDF", mz_range=(35.0, 550.0))
"""

from __future__ import annotations

from pyrecycle_analytics.ingestion.andi_cdf import read_andi_cdf
from pyrecycle_analytics.ingestion.common import (
    IngestOptions,
    default_sample_metadata,
    sha256_of_file,
)
from pyrecycle_analytics.ingestion.mzml import (
    pyopenms_available,
    read_mzdata,
    read_mzml,
    read_mzxml,
)
from pyrecycle_analytics.ingestion.registry import (
    READERS,
    SUPPORTED_EXTENSIONS,
    detect_format,
    read_pyrogram,
    read_pyrogram_bytes,
)
from pyrecycle_analytics.ingestion.writers import write_andi_cdf, write_mzml

__all__ = [
    "READERS",
    "SUPPORTED_EXTENSIONS",
    "IngestOptions",
    "default_sample_metadata",
    "detect_format",
    "pyopenms_available",
    "read_andi_cdf",
    "read_mzdata",
    "read_mzml",
    "read_mzxml",
    "read_pyrogram",
    "read_pyrogram_bytes",
    "sha256_of_file",
    "write_andi_cdf",
    "write_mzml",
]
