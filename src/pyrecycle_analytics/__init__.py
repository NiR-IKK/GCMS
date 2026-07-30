"""PyRecycle-Analytics — inverse polymer analytics from Py-GC/MS data.

The platform reconstructs the composition of post-consumer recyclates from their
pyrolysis chromatograms: which polymers are present and in what proportion, which
additives and regulated substances came along, and how far the material has
already degraded.

Milestone layout (see README):

* ``ingestion`` — mzML / mzXML / mzData (pyopenms) and ANDI-MS netCDF (scipy)
  readers producing a validated :class:`~pyrecycle_analytics.core.PyrogramDataCube`.
* ``preprocessing`` — per-channel baseline correction and retention-axis smoothing.
* ``core`` — the data cube, chromatographic peak-shape physics and m/z binning.

Later milestones add curve resolution and polymer-matrix subtraction, the marker
library with degradation indices, and the recyclate-passport API.
"""

from __future__ import annotations

__version__ = "0.1.0"

from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import (
    CorruptRawDataError,
    DataCubeError,
    IngestionError,
    MissingDependencyError,
    PreprocessingError,
    PyRecycleError,
    UnsupportedFormatError,
)

__all__ = [
    "CorruptRawDataError",
    "DataCubeError",
    "IngestionError",
    "MissingDependencyError",
    "PreprocessingError",
    "PyRecycleError",
    "PyrogramDataCube",
    "UnsupportedFormatError",
    "__version__",
]
