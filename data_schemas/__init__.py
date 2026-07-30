"""Validated data model of the PyRecycle-Analytics platform.

The package is intentionally separate from the algorithmic code in
``pyrecycle_analytics``: schemas are shared by the ingestion layer, the
chemometric engines, the persistence layer and the reporting API, and they must
be importable without pulling in heavy scientific dependencies.
"""

from __future__ import annotations

from data_schemas.acquisition import (
    AcquisitionConditions,
    GcConditions,
    MsConditions,
    OvenRamp,
    PyrolysisConditions,
    SampleMetadata,
)
from data_schemas.enums import (
    AcquisitionMode,
    IonisationMode,
    MarkerRole,
    Polarity,
    PolymerClass,
    PyrolysisMode,
    RecyclateStream,
    SourceFormat,
)
from data_schemas.pyrogram import MzAxisSpec, PreprocessingStep, PyrogramMetadata
from data_schemas.truth import ComponentTruth, DriftTruth, PyrogramTruth

__all__ = [
    "AcquisitionConditions",
    "AcquisitionMode",
    "ComponentTruth",
    "DriftTruth",
    "GcConditions",
    "IonisationMode",
    "MarkerRole",
    "MsConditions",
    "MzAxisSpec",
    "OvenRamp",
    "Polarity",
    "PolymerClass",
    "PreprocessingStep",
    "PyrogramMetadata",
    "PyrogramTruth",
    "PyrolysisConditions",
    "PyrolysisMode",
    "RecyclateStream",
    "SampleMetadata",
    "SourceFormat",
]
