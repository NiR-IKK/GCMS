"""Pydantic models describing *how* a pyrogram was produced.

Py-GC/MS results are only comparable if the pyrolysis and separation conditions
are comparable: a 600 °C single shot on a 30 m DB-5 does not yield the same
alkene/alkane ratio as a 700 °C shot on a polar column. Every derived quantity
in this platform (degradation index, marker triad ratios, matrix subtraction
templates) is therefore bound to an explicit acquisition description.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data_schemas.enums import (
    AcquisitionMode,
    IonisationMode,
    Polarity,
    PyrolysisMode,
    RecyclateStream,
)

__all__ = [
    "OvenRamp",
    "PyrolysisConditions",
    "GcConditions",
    "MsConditions",
    "SampleMetadata",
    "AcquisitionConditions",
]


class _Frozen(BaseModel):
    """Base for immutable, strictly validated metadata records."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


class OvenRamp(BaseModel):
    """One segment of the GC oven temperature program."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rate_c_per_min: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Heating rate; 0.0 marks an isothermal hold segment.",
    )
    target_c: float = Field(..., ge=-100.0, le=500.0, description="Set-point at segment end.")
    hold_min: float = Field(0.0, ge=0.0, le=300.0, description="Isothermal hold at target_c.")


class PyrolysisConditions(_Frozen):
    """Micro-furnace / filament pyrolyser settings."""

    temperature_c: float = Field(
        600.0,
        ge=100.0,
        le=1200.0,
        description="Pyrolysis set-point. 500-600 °C is the analytical standard for "
        "polyolefins; >700 °C promotes secondary aromatisation and distorts marker ratios.",
    )
    duration_s: float = Field(12.0, gt=0.0, le=600.0)
    mode: PyrolysisMode = PyrolysisMode.SINGLE_SHOT
    interface_temp_c: float = Field(320.0, ge=50.0, le=500.0)
    sample_mass_ug: float | None = Field(
        None, gt=0.0, le=5000.0, description="Weighed sample mass; required for absolute yields."
    )
    derivatisation_reagent: str | None = Field(
        None, description="e.g. 'TMAH 25% in methanol' for reactive pyrolysis of PET/PA."
    )

    @model_validator(mode="after")
    def _check_derivatisation(self) -> PyrolysisConditions:
        if self.mode is PyrolysisMode.REACTIVE_THM and not self.derivatisation_reagent:
            raise ValueError("reactive-pyrolysis-THM requires 'derivatisation_reagent'")
        return self


class GcConditions(_Frozen):
    """Capillary GC separation conditions."""

    column_name: str = Field("Ultra ALLOY-5", max_length=120)
    stationary_phase: str = Field(
        "5% diphenyl / 95% dimethylpolysiloxane",
        max_length=200,
        description="Phase chemistry — governs the elution order model used for "
        "retention-index anchoring.",
    )
    column_length_m: float = Field(30.0, gt=0.0, le=200.0)
    column_id_mm: float = Field(0.25, gt=0.0, le=1.0)
    film_thickness_um: float = Field(0.25, gt=0.0, le=10.0)
    carrier_gas: str = Field("helium", max_length=40)
    flow_ml_per_min: float = Field(1.0, gt=0.0, le=50.0)
    split_ratio: float | None = Field(
        50.0, ge=0.0, description="None marks a splitless injection."
    )
    inlet_temp_c: float = Field(320.0, ge=50.0, le=500.0)
    initial_temp_c: float = Field(40.0, ge=-100.0, le=500.0)
    initial_hold_min: float = Field(2.0, ge=0.0, le=300.0)
    oven_program: tuple[OvenRamp, ...] = Field(
        default=(OvenRamp(rate_c_per_min=10.0, target_c=320.0, hold_min=10.0),),
        description="Ordered ramp segments following the initial hold.",
    )

    @field_validator("oven_program")
    @classmethod
    def _monotonic_program(cls, ramps: tuple[OvenRamp, ...]) -> tuple[OvenRamp, ...]:
        if not ramps:
            raise ValueError("oven_program must contain at least one segment")
        for previous, current in zip(ramps, ramps[1:], strict=False):
            if current.target_c < previous.target_c:
                raise ValueError(
                    "oven_program target temperatures must be non-decreasing "
                    f"(got {previous.target_c} -> {current.target_c})"
                )
        return ramps

    @property
    def final_temp_c(self) -> float:
        return self.oven_program[-1].target_c

    @property
    def program_duration_min(self) -> float:
        """Total method time including the initial hold."""
        total = self.initial_hold_min
        current = self.initial_temp_c
        for ramp in self.oven_program:
            if ramp.rate_c_per_min > 0.0:
                total += (ramp.target_c - current) / ramp.rate_c_per_min
            total += ramp.hold_min
            current = ramp.target_c
        return total


class MsConditions(_Frozen):
    """Mass spectrometer acquisition settings."""

    ionisation: IonisationMode = IonisationMode.EI
    polarity: Polarity = Polarity.POSITIVE
    acquisition_mode: AcquisitionMode = AcquisitionMode.FULL_SCAN
    electron_energy_ev: float = Field(
        70.0,
        gt=0.0,
        le=200.0,
        description="70 eV is required for library-comparable fragment ratios.",
    )
    source_temp_c: float = Field(230.0, ge=50.0, le=400.0)
    transfer_line_temp_c: float = Field(300.0, ge=50.0, le=500.0)
    mz_low: float = Field(29.0, gt=0.0, le=5000.0)
    mz_high: float = Field(600.0, gt=0.0, le=10000.0)
    scan_rate_hz: float = Field(
        5.0,
        gt=0.0,
        le=500.0,
        description="Scans per second. Needs >=8-10 points across a peak for reliable "
        "curve resolution.",
    )
    detector_gain: float | None = Field(None, gt=0.0)

    @model_validator(mode="after")
    def _check_mz_window(self) -> MsConditions:
        if self.mz_high <= self.mz_low:
            raise ValueError(f"mz_high ({self.mz_high}) must exceed mz_low ({self.mz_low})")
        return self

    @property
    def scan_period_s(self) -> float:
        return 1.0 / self.scan_rate_hz


class SampleMetadata(_Frozen):
    """Identity and provenance of the analysed recyclate."""

    sample_id: str = Field(..., min_length=1, max_length=64)
    description: str = Field("", max_length=500)
    stream: RecyclateStream = RecyclateStream.UNKNOWN
    batch: str | None = Field(None, max_length=64)
    supplier: str | None = Field(None, max_length=120)
    collection_region: str | None = Field(
        None, max_length=120, description="Relevant for legislative marker expectations (REACH)."
    )
    replicate: int = Field(1, ge=1, le=1000)
    nominal_composition: dict[str, float] = Field(
        default_factory=dict,
        description="Declared or gravimetric mass fractions keyed by PolymerClass value. "
        "Populated for lab blends; empty for unknown PCR material.",
    )
    operator: str | None = Field(None, max_length=120)
    sampled_at: datetime | None = None

    @field_validator("nominal_composition")
    @classmethod
    def _check_fractions(cls, fractions: dict[str, float]) -> dict[str, float]:
        for key, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"nominal_composition['{key}']={value} outside [0, 1]")
        total = sum(fractions.values())
        if fractions and total > 1.0 + 1e-6:
            raise ValueError(f"nominal_composition sums to {total:.4f}, must not exceed 1.0")
        return fractions


class AcquisitionConditions(_Frozen):
    """Full instrumental context of a single run."""

    pyrolysis: PyrolysisConditions = Field(default_factory=PyrolysisConditions)
    gc: GcConditions = Field(default_factory=GcConditions)
    ms: MsConditions = Field(default_factory=MsConditions)
    instrument_model: str | None = Field(None, max_length=160)
    method_name: str | None = Field(None, max_length=160)
    acquired_at: datetime | None = None
