"""Exception hierarchy for PyRecycle-Analytics.

A single root exception lets callers (API handlers, Streamlit UI, batch scripts)
distinguish "the pipeline rejected this input" from a genuine bug.
"""

from __future__ import annotations

__all__ = [
    "PyRecycleError",
    "IngestionError",
    "UnsupportedFormatError",
    "CorruptRawDataError",
    "MissingDependencyError",
    "DataCubeError",
    "PreprocessingError",
]


class PyRecycleError(Exception):
    """Base class for all errors raised by this platform."""


class IngestionError(PyRecycleError):
    """Raw-data import failed."""


class UnsupportedFormatError(IngestionError):
    """The file extension / magic bytes do not map to a known reader."""


class CorruptRawDataError(IngestionError):
    """The file parsed structurally but violates the format's data model."""


class MissingDependencyError(PyRecycleError):
    """An optional dependency required for the requested operation is absent."""

    def __init__(self, package: str, purpose: str, extra: str | None = None) -> None:
        install_hint = (
            f"pip install 'pyrecycle-analytics[{extra}]'" if extra else f"pip install {package}"
        )
        super().__init__(
            f"'{package}' is required for {purpose} but is not installed. "
            f"Install it with: {install_hint}"
        )
        self.package = package
        self.purpose = purpose


class DataCubeError(PyRecycleError):
    """The retention-time x m/z x intensity cube is internally inconsistent."""


class PreprocessingError(PyRecycleError):
    """A baseline / smoothing / binning operation received invalid parameters."""
