"""Ground-truth schema for simulated and gravimetrically prepared benchmarks.

Deconvolution cannot be regression-tested against real PCR samples: nobody knows
the true composition of a post-consumer flake blend. The platform therefore
treats *ground truth* as a first-class, validated artefact — produced either by
the ``SyntheticPyrogramGenerator`` or by weighing a lab blend — and every
chemometric claim (resolved profile, recovered spectrum, subtracted matrix,
degradation index) is scored against it.

These models are deliberately free of NumPy: they describe scalar truth
(positions, areas, ratios) that can be serialised next to a benchmark dataset.
The paired ``C`` / ``S`` matrices stay in memory alongside the cube.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from data_schemas.enums import MarkerRole, PolymerClass

__all__ = ["ComponentTruth", "DriftTruth", "PyrogramTruth"]


class ComponentTruth(BaseModel):
    """One chemically distinct species present in a simulated pyrogram."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(
        ..., ge=0, description="Column index in the pure-profile matrix C / row index in S."
    )
    name: str = Field(..., min_length=1, max_length=120)
    polymer_class: PolymerClass
    role: MarkerRole
    carbon_number: int | None = Field(
        None, ge=1, le=100, description="Set for members of a homologous series."
    )
    molecular_weight: float | None = Field(None, gt=0.0)

    source_areas: dict[str, float] = Field(
        default_factory=dict,
        description="Area contributed by each polymer, keyed by PolymerClass value. "
        "A compound is frequently produced by more than one polymer in a blend — "
        "n-nonane comes from both PE and PP, benzene from both PS and PVC — and the "
        "instrument sees a single peak. This field keeps the attribution that the "
        "measurement itself cannot resolve, so quantification can be scored against it. "
        "``polymer_class`` above names the dominant contributor.",
    )

    retention_time_s: float = Field(
        ..., gt=0.0, description="Apex position *after* the run's retention drift was applied."
    )
    nominal_retention_time_s: float = Field(
        ..., gt=0.0, description="Drift-free apex position of the underlying method."
    )
    peak_width_s: float = Field(..., gt=0.0, description="Gaussian sigma of the elution profile.")
    tailing_s: float = Field(
        0.0, ge=0.0, description="Exponential-modifier time constant (0 = symmetric Gaussian)."
    )
    area: float = Field(..., ge=0.0, description="True integrated TIC area of this component.")
    base_peak_mz: int = Field(..., gt=0, description="m/z carrying the highest intensity.")
    quantifier_mz: int = Field(
        ...,
        gt=0,
        description="Ion recommended for quantification; selective against the matrix, "
        "and therefore not necessarily the base peak.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def drift_s(self) -> float:
        """Signed retention shift this run applied to the component."""
        return self.retention_time_s - self.nominal_retention_time_s


class DriftTruth(BaseModel):
    """Parameters of the smooth retention-time warp applied to a simulated run.

    Real inter-run drift is not a constant offset: column ageing shifts late
    peaks more than early ones, and pressure fluctuation adds a slow oscillation.
    Reproducing that is what makes alignment / PARAFAC2 tests meaningful.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    offset_s: float = Field(0.0, description="Constant shift applied across the whole run.")
    linear_factor: float = Field(
        0.0, description="Fractional stretch of the time axis (1e-3 = 0.1% longer run)."
    )
    quadratic_factor: float = Field(
        0.0, description="Curvature term in s per (normalised time)^2; models column ageing."
    )
    oscillation_amplitude_s: float = Field(0.0, ge=0.0)
    oscillation_period_s: float = Field(600.0, gt=0.0)

    @property
    def is_identity(self) -> bool:
        return (
            self.offset_s == 0.0
            and self.linear_factor == 0.0
            and self.quadratic_factor == 0.0
            and self.oscillation_amplitude_s == 0.0
        )


class PyrogramTruth(BaseModel):
    """Complete ground truth of one simulated pyrogram."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recipe_name: str = Field(..., min_length=1, max_length=120)
    seed: int = Field(..., ge=0)
    blend_fractions: dict[str, float] = Field(
        ...,
        description="True mass fractions keyed by PolymerClass value; sums to ~1.0.",
    )
    degradation_levels: dict[str, float] = Field(
        default_factory=dict,
        description="Per-polymer thermo-oxidative ageing level in [0, 1] used to skew "
        "oxidised-fragment and iso-alkene abundances.",
    )
    components: tuple[ComponentTruth, ...] = Field(..., min_length=1)
    drift: DriftTruth = Field(default_factory=DriftTruth)
    noise_sigma: float = Field(0.0, ge=0.0, description="Additive detector noise level used.")
    baseline_included: bool = Field(
        True, description="Whether column bleed / rising baseline was added to the cube."
    )

    @field_validator("blend_fractions")
    @classmethod
    def _check_fractions(cls, fractions: dict[str, float]) -> dict[str, float]:
        if not fractions:
            raise ValueError("blend_fractions must not be empty")
        for key, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"blend_fractions['{key}']={value} outside [0, 1]")
        total = sum(fractions.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"blend_fractions must sum to 1.0, got {total:.6f}")
        return fractions

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_components(self) -> int:
        return len(self.components)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_area(self) -> float:
        return sum(component.area for component in self.components)

    def area_by_polymer(self) -> dict[str, float]:
        """Total simulated signal area attributable to each polymer.

        Built from :attr:`ComponentTruth.source_areas`, so shared compounds are
        split between their producers instead of being credited to one. This is the
        reference a composition estimate is scored against — note that it is a
        *signal* split, not the mass split in :attr:`blend_fractions`, because
        polymers differ in GC-amenable pyrolysate yield.

        Returns:
            Mapping from PolymerClass value to summed area.
        """
        totals: dict[str, float] = {}
        for component in self.components:
            contributions = component.source_areas or {
                str(component.polymer_class): component.area
            }
            for polymer, area in contributions.items():
                totals[polymer] = totals.get(polymer, 0.0) + area
        return totals

    def component_by_name(self, name: str) -> ComponentTruth:
        """Look up a component by exact name.

        Raises:
            KeyError: if no component carries that name.
        """
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(f"no synthetic component named {name!r}")

    def select(
        self,
        *,
        polymer_class: PolymerClass | None = None,
        role: MarkerRole | None = None,
        carbon_number: int | None = None,
        min_area: float = 0.0,
    ) -> tuple[ComponentTruth, ...]:
        """Filter components — the standard entry point for assertions in tests."""
        return tuple(
            component
            for component in self.components
            if (polymer_class is None or component.polymer_class is polymer_class)
            and (role is None or component.role is role)
            and (carbon_number is None or component.carbon_number == carbon_number)
            and component.area >= min_area
        )

    def coeluting_groups(self, *, resolution_threshold: float = 1.0) -> tuple[tuple[int, ...], ...]:
        """Group component indices that are not chromatographically resolved.

        Two adjacent peaks are treated as co-eluting when their chromatographic
        resolution ``R = Δt / (2 (σ₁ + σ₂))`` falls below ``resolution_threshold``
        (R < 1.0 means visibly fused peaks; R >= 1.5 is conventionally "resolved").
        The returned clusters are the units an MCR-ALS pipeline has to pull apart,
        so tests use them to pick genuinely hard windows instead of easy ones.

        Args:
            resolution_threshold: Resolution below which two peaks count as fused.

        Returns:
            Tuple of index tuples, ordered by retention time. Singletons are
            included so the grouping is a full partition of all components.
        """
        ordered = sorted(self.components, key=lambda component: component.retention_time_s)
        groups: list[list[int]] = []
        current: list[int] = []
        previous: ComponentTruth | None = None

        for component in ordered:
            if previous is not None:
                delta = component.retention_time_s - previous.retention_time_s
                width_sum = previous.peak_width_s + component.peak_width_s
                resolution = delta / (2.0 * width_sum) if width_sum > 0 else float("inf")
                if resolution >= resolution_threshold:
                    groups.append(current)
                    current = []
            current.append(component.index)
            previous = component

        if current:
            groups.append(current)
        return tuple(tuple(group) for group in groups)
