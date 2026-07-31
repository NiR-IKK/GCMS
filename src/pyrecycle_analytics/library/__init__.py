"""Polymer marker library: reference spectra, retention indices, marker patterns."""

from __future__ import annotations

from pyrecycle_analytics.library.reference_data import (
    MARKER_PATTERNS,
    REFERENCE_COMPOUNDS,
    RESPONSE_FACTORS,
    STATIONARY_PHASE,
    LibraryCompound,
    LibraryPattern,
    LibraryPatternMember,
)
from pyrecycle_analytics.library.repository import (
    MarkerLibrary,
    create_library,
    seed_library,
)
from pyrecycle_analytics.library.retention_index import (
    RetentionIndexCalibration,
    RetentionIndexError,
    calibrate_from_comb,
    estimate_first_carbon_number,
)
from pyrecycle_analytics.library.schema import (
    Base,
    Compound,
    MarkerPattern,
    MarkerPatternMember,
    Polymer,
    ResponseFactor,
    RetentionIndexEntry,
    SpectrumPeak,
)

__all__ = [
    "MARKER_PATTERNS",
    "REFERENCE_COMPOUNDS",
    "RESPONSE_FACTORS",
    "STATIONARY_PHASE",
    "Base",
    "Compound",
    "LibraryCompound",
    "LibraryPattern",
    "LibraryPatternMember",
    "MarkerLibrary",
    "MarkerPattern",
    "MarkerPatternMember",
    "Polymer",
    "ResponseFactor",
    "RetentionIndexCalibration",
    "RetentionIndexEntry",
    "RetentionIndexError",
    "SpectrumPeak",
    "calibrate_from_comb",
    "create_library",
    "estimate_first_carbon_number",
    "seed_library",
]
