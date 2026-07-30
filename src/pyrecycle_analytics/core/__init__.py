"""Core numeric containers and shared chromatographic physics."""

from __future__ import annotations

from pyrecycle_analytics.core.binning import (
    axis_to_spec,
    bin_ragged_scans,
    bin_scan,
    make_nominal_mz_axis,
    make_uniform_mz_axis,
)
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.core.peakshapes import (
    chromatographic_resolution,
    emg,
    emg_apex_time,
    fwhm_gaussian,
    gaussian,
    profile_matrix,
)

__all__ = [
    "PyrogramDataCube",
    "axis_to_spec",
    "bin_ragged_scans",
    "bin_scan",
    "chromatographic_resolution",
    "emg",
    "emg_apex_time",
    "fwhm_gaussian",
    "gaussian",
    "make_nominal_mz_axis",
    "make_uniform_mz_axis",
    "profile_matrix",
]
