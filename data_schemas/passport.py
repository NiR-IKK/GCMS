"""Schema of the digital recyclate passport.

The passport is the platform's deliverable and the only artefact that leaves the
laboratory, so its schema carries two obligations that the rest of the code does
not.

**Every number states how far it can be trusted.** A composition percentage
derived without gravimetric calibration standards is semi-quantitative, and the
schema forces that to be declared rather than leaving a bare "18.4 %" that reads
like a measurement. :class:`CalibrationStatus` is a required field, not an
optional annotation.

**Regulatory findings separate detection from quantification.** REACH limits are
mass fractions. Py-GC/MS without calibration standards establishes that a
substance is *present*; it does not establish that it is below 0.1 %. The
:class:`RegulatoryFinding` model therefore has a detection flag and a separate,
nullable quantification, and a compliance statement can only be made when the
latter exists.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from data_schemas.acquisition import AcquisitionConditions, SampleMetadata
from data_schemas.enums import PolymerClass

__all__ = [
    "PASSPORT_SCHEMA_VERSION",
    "CalibrationStatus",
    "ConfidenceLevel",
    "PolymerFraction",
    "RegulatoryFinding",
    "DegradationAssessment",
    "AnalysisProvenance",
    "RecyclatePassport",
]

PASSPORT_SCHEMA_VERSION = "1.0"
"""Version of this schema, stored in every passport so old ones stay readable."""


class CalibrationStatus(StrEnum):
    """How far a quantitative statement can be trusted."""

    CALIBRATED = "calibrated"
    """Gravimetric reference blends were measured under the same method."""
    RESPONSE_CORRECTED = "response-corrected"
    """Corrected for polymer-specific pyrolysate yield, but not calibrated
    against weighed standards. Percentages are semi-quantitative."""
    UNCALIBRATED = "uncalibrated"
    """Raw signal shares. Not percentages of material in any defensible sense."""

    @property
    def is_quantitative(self) -> bool:
        """True only for statements that may be reported as measured values."""
        return self is CalibrationStatus.CALIBRATED

    @property
    def disclaimer(self) -> str:
        """Text that must accompany a figure with this status."""
        match self:
            case CalibrationStatus.CALIBRATED:
                return "Quantified against gravimetric reference blends."
            case CalibrationStatus.RESPONSE_CORRECTED:
                return (
                    "Semi-quantitative: corrected for polymer-specific pyrolysate "
                    "yield, but not calibrated against weighed reference blends. "
                    "Treat as an estimate, not a measured mass fraction."
                )
            case _:
                return (
                    "Not quantitative: raw signal shares, uncorrected for the fact "
                    "that polymers differ several-fold in GC-amenable pyrolysate "
                    "yield. Not comparable to a mass fraction."
                )


class ConfidenceLevel(StrEnum):
    """Qualitative strength of an identification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INDICATIVE = "indicative"
    """Rests on a single marker. A polymer is not established by one peak."""

    @classmethod
    def from_score(cls, score: float, *, single_marker: bool = False) -> ConfidenceLevel:
        """Map a numeric confidence onto a reportable level."""
        if single_marker:
            return cls.INDICATIVE
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.5:
            return cls.MEDIUM
        return cls.LOW


class PolymerFraction(BaseModel):
    """One polymer's share of the material."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    polymer: PolymerClass
    share_percent: float = Field(..., ge=0.0, le=100.0)
    calibration_status: CalibrationStatus
    confidence: ConfidenceLevel
    marker_pattern: str = Field(
        ..., description="Marker pattern the assignment rests on."
    )
    markers_found: tuple[str, ...] = Field(default_factory=tuple)
    markers_expected: int = Field(..., ge=1)
    uncertainty_percent: float | None = Field(
        None,
        ge=0.0,
        description="Absolute uncertainty on the share, when it can be estimated. "
        "None for uncalibrated results, where no honest interval exists.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pattern_coverage(self) -> float:
        """Fraction of the marker pattern that was actually found."""
        return len(self.markers_found) / self.markers_expected

    @model_validator(mode="after")
    def _uncertainty_requires_calibration(self) -> PolymerFraction:
        if (
            self.uncertainty_percent is not None
            and self.calibration_status is CalibrationStatus.UNCALIBRATED
        ):
            raise ValueError(
                "an uncalibrated share cannot carry an uncertainty interval; "
                "there is nothing to base it on"
            )
        return self


class RegulatoryFinding(BaseModel):
    """A regulated substance looked for in the sample.

    Detection and quantification are separate fields on purpose. Py-GC/MS
    reliably answers "is this substance present"; answering "is it below the
    0.1 % limit" additionally requires calibration standards, and conflating the
    two would put an unsupportable compliance claim into a document that looks
    official.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    substance: str
    regulation: str = Field(
        ..., description="Legal basis, e.g. 'REACH Annex XVII' or 'REACH SVHC list'."
    )
    cas_number: str | None = None
    detected: bool
    detection_confidence: ConfidenceLevel | None = None
    limit_percent: float | None = Field(
        None, ge=0.0, description="Regulatory threshold as a mass fraction, if one exists."
    )
    quantified_percent: float | None = Field(
        None,
        ge=0.0,
        description="Measured mass fraction. None unless calibration standards "
        "were run — which is the normal case.",
    )
    marker_compound: str | None = None
    note: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def compliance_assessable(self) -> bool:
        """Whether a statement about the limit can be made at all.

        False whenever the substance was not quantified, which is the usual
        outcome. The passport then reports presence and explicitly declines to
        assess compliance.
        """
        return self.quantified_percent is not None and self.limit_percent is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exceeds_limit(self) -> bool | None:
        """Whether the limit is exceeded, or ``None`` when that cannot be judged."""
        if not self.compliance_assessable:
            return None
        assert self.quantified_percent is not None and self.limit_percent is not None
        return self.quantified_percent > self.limit_percent

    @model_validator(mode="after")
    def _detection_needs_confidence(self) -> RegulatoryFinding:
        if self.detected and self.detection_confidence is None:
            raise ValueError(
                f"{self.substance}: a detection must carry a confidence level"
            )
        return self


class DegradationAssessment(BaseModel):
    """Ageing state of the material."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    carbonyl_index: float = Field(..., ge=0.0)
    acid_share: float = Field(..., ge=0.0, le=1.0)
    alkene_to_alkane: float = Field(..., ge=0.0)
    branching_index: float = Field(..., ge=0.0)
    mean_chain_length: float | None = Field(None, ge=0.0)
    relative_to_virgin: dict[str, float] = Field(
        default_factory=dict,
        description="Indices as multiples of a virgin reference. Empty when no "
        "reference was supplied, in which case the absolute values below are not "
        "interpretable on their own.",
    )
    reference_sample_id: str | None = None
    notes: tuple[str, ...] = Field(default_factory=tuple)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_interpretable(self) -> bool:
        """Whether the figures mean anything without further context.

        Degradation indices are ratios whose absolute level depends on the
        polymer and on the pyrolysis temperature. Only a comparison against a
        virgin reference of the same material makes them readable.
        """
        return bool(self.relative_to_virgin)


class AnalysisProvenance(BaseModel):
    """Everything needed to reproduce or audit the result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_file: str | None = None
    source_checksum: str | None = None
    preprocessing_steps: tuple[str, ...] = Field(default_factory=tuple)
    matrix_subtracted: bool = False
    matrix_explained_fraction: float = Field(0.0, ge=0.0, le=1.0)
    n_components_resolved: int = Field(0, ge=0)
    n_compounds_identified: int = Field(0, ge=0)
    library_version: str = "builtin"
    retention_index_anchor: str | None = Field(
        None, description="'anchored' or 'estimated'; None when no ladder was used."
    )
    software_version: str = "0.1.0"
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class RecyclatePassport(BaseModel):
    """The digital recyclate passport."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PASSPORT_SCHEMA_VERSION
    issued_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sample: SampleMetadata
    acquisition: AcquisitionConditions

    polymer_fractions: tuple[PolymerFraction, ...] = Field(default_factory=tuple)
    unassigned_share_percent: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Signal that could not be attributed to any identified polymer. "
        "Reported rather than distributed over the identified ones, which would "
        "silently inflate them.",
    )
    regulatory_findings: tuple[RegulatoryFinding, ...] = Field(default_factory=tuple)
    degradation: DegradationAssessment | None = None
    provenance: AnalysisProvenance = Field(default_factory=AnalysisProvenance)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def calibration_status(self) -> CalibrationStatus:
        """Weakest calibration status among the reported fractions.

        A passport is only as quantitative as its least supported figure.
        """
        if not self.polymer_fractions:
            return CalibrationStatus.UNCALIBRATED
        order = [
            CalibrationStatus.UNCALIBRATED,
            CalibrationStatus.RESPONSE_CORRECTED,
            CalibrationStatus.CALIBRATED,
        ]
        return min(
            (fraction.calibration_status for fraction in self.polymer_fractions),
            key=order.index,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def detected_substances(self) -> tuple[str, ...]:
        return tuple(
            finding.substance for finding in self.regulatory_findings if finding.detected
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unassessable_findings(self) -> tuple[str, ...]:
        """Detected substances whose compliance could not be judged.

        Surfaced at the top level because it is the passport's most easily
        misread section: a substance listed as detected but not quantified says
        nothing about whether a limit is met.
        """
        return tuple(
            finding.substance
            for finding in self.regulatory_findings
            if finding.detected and not finding.compliance_assessable
        )

    @model_validator(mode="after")
    def _shares_must_not_exceed_one_hundred(self) -> RecyclatePassport:
        total = sum(fraction.share_percent for fraction in self.polymer_fractions)
        total += self.unassigned_share_percent
        if total > 100.5:
            raise ValueError(
                f"polymer shares plus unassigned signal sum to {total:.2f} %, "
                "which exceeds 100 %"
            )
        return self

    def headline(self) -> str:
        """One-line summary for a listing or a log."""
        if not self.polymer_fractions:
            return f"{self.sample.sample_id}: no polymer identified"
        main = max(self.polymer_fractions, key=lambda fraction: fraction.share_percent)
        return (
            f"{self.sample.sample_id}: {main.polymer} "
            f"{main.share_percent:.1f} % ({self.calibration_status})"
        )
