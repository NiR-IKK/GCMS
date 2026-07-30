"""Schema for an ingested pyrogram: metadata plus the axis description.

The numeric payload (the retention-time x m/z x intensity cube) lives in a
NumPy-backed container in ``pyrecycle_analytics.core.datacube``; this module
describes everything *about* that payload so it can be serialised, stored in a
relational table and returned from the API without dragging arrays along.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from data_schemas.acquisition import AcquisitionConditions, SampleMetadata
from data_schemas.enums import SourceFormat

__all__ = ["MzAxisSpec", "PreprocessingStep", "PyrogramMetadata", "round_trip_json"]


def _computed_field_exclusions(value: object) -> dict[str, object] | None:
    """Build a Pydantic ``exclude`` spec covering computed fields at any depth.

    The spec is derived from the *instance*, so nested models, tuples of models and
    dictionaries of models are all covered without the caller naming any paths.
    Returns ``None`` when the value contains nothing to exclude.
    """
    if isinstance(value, BaseModel):
        spec: dict[str, object] = dict.fromkeys(type(value).model_computed_fields, True)
        for name in type(value).model_fields:
            nested = _computed_field_exclusions(getattr(value, name, None))
            if nested:
                spec[name] = nested
        return spec or None

    if isinstance(value, list | tuple | set | frozenset):
        merged: dict[str, object] = {}
        for item in value:
            nested = _computed_field_exclusions(item)
            if nested:
                merged.update(nested)
        return {"__all__": merged} if merged else None

    if isinstance(value, dict):
        merged = {}
        for item in value.values():
            nested = _computed_field_exclusions(item)
            if nested:
                merged.update(nested)
        return {"__all__": merged} if merged else None

    return None


def round_trip_json(model: BaseModel) -> str:
    """Serialise a model to JSON that ``model_validate_json`` can read back.

    Pydantic includes ``@computed_field`` values in ``model_dump_json`` because they
    are useful in an API response, but they are not constructor arguments — feeding
    them back into a model with ``extra="forbid"`` fails validation. Anything that
    persists a model and later reloads it (the ``.npz`` cube archive, a benchmark
    truth file on disk) has to drop them first, at every level of nesting: a
    ``PyrogramTruth`` has computed fields of its own *and* a tuple of components
    that each have one.

    Args:
        model: Any Pydantic model in this package.

    Returns:
        JSON text containing only declared fields, recursively.
    """
    exclusions = _computed_field_exclusions(model)
    if not exclusions:
        return model.model_dump_json()
    return model.model_dump_json(exclude=exclusions)  # type: ignore[arg-type]


class MzAxisSpec(BaseModel):
    """Description of the m/z grid the intensity matrix is defined on.

    Quadrupole EI GC/MS is acquired at unit mass resolution, so the natural
    representation is a dense integer m/z grid. Keeping the grid explicit (rather
    than implied by array position) makes it possible to mix nominal-mass and
    high-resolution data in the same pipeline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mz_low: float = Field(..., gt=0.0)
    mz_high: float = Field(..., gt=0.0)
    n_bins: int = Field(..., gt=0)
    bin_width: float = Field(
        1.0, gt=0.0, description="Grid spacing in Th; 1.0 for nominal-mass binning."
    )
    is_nominal: bool = Field(
        True, description="True when bins are centred on integer masses (unit resolution)."
    )

    @field_validator("mz_high")
    @classmethod
    def _ordered(cls, mz_high: float, info) -> float:  # noqa: ANN001 - pydantic ValidationInfo
        mz_low = info.data.get("mz_low")
        if mz_low is not None and mz_high <= mz_low:
            raise ValueError(f"mz_high ({mz_high}) must exceed mz_low ({mz_low})")
        return mz_high


class PreprocessingStep(BaseModel):
    """Audit record of one transformation applied to the cube.

    Chemometric results are only defensible if the preprocessing chain is
    reproducible, so every operation appends an entry here instead of mutating
    data silently.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=64)
    parameters: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    applied_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    note: str | None = Field(None, max_length=300)


class PyrogramMetadata(BaseModel):
    """Everything needed to interpret and trace one pyrogram data cube."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sample: SampleMetadata
    acquisition: AcquisitionConditions = Field(default_factory=AcquisitionConditions)
    source_format: SourceFormat
    source_path: Path | None = Field(
        None, description="Original file location; None for synthetic or in-memory pyrograms."
    )
    source_checksum: str | None = Field(
        None,
        max_length=128,
        description="SHA-256 of the raw file, so a processed result can be traced back "
        "to exact bytes.",
    )
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    n_scans: int = Field(..., gt=0)
    mz_axis: MzAxisSpec
    rt_start_s: float = Field(..., ge=0.0)
    rt_end_s: float = Field(..., gt=0.0)

    reader_warnings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Non-fatal problems found while parsing (missing attributes, "
        "ragged scans, unsorted spectra).",
    )
    preprocessing: tuple[PreprocessingStep, ...] = Field(default_factory=tuple)
    extra: dict[str, str] = Field(
        default_factory=dict,
        description="Vendor-specific attributes preserved verbatim from the raw file.",
    )

    @field_validator("rt_end_s")
    @classmethod
    def _rt_ordered(cls, rt_end_s: float, info) -> float:  # noqa: ANN001
        rt_start_s = info.data.get("rt_start_s")
        if rt_start_s is not None and rt_end_s < rt_start_s:
            raise ValueError(f"rt_end_s ({rt_end_s}) must not precede rt_start_s ({rt_start_s})")
        return rt_end_s

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rt_span_s(self) -> float:
        return self.rt_end_s - self.rt_start_s

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_scan_period_s(self) -> float:
        """Average time between scans — the practical sampling density."""
        if self.n_scans < 2:
            return 0.0
        return self.rt_span_s / (self.n_scans - 1)

    def with_step(self, step: PreprocessingStep) -> PyrogramMetadata:
        """Return a copy with ``step`` appended to the preprocessing audit trail."""
        return self.model_copy(update={"preprocessing": (*self.preprocessing, step)})

    def with_warnings(self, *warnings: str) -> PyrogramMetadata:
        """Return a copy with additional reader warnings recorded."""
        return self.model_copy(update={"reader_warnings": (*self.reader_warnings, *warnings)})
