"""Synthetic Py-GC/MS pyrogram generator — the TDD foundation of the platform.

Nobody knows the true composition of a post-consumer flake blend, so there is no
real sample against which a deconvolution result can be scored. This module
supplies that missing reference: it builds pyrograms from a *declared* polymer
blend and returns both the data cube and the exact bilinear factors that produced
it. Every chemometric claim the platform makes is regression-tested against these.

What is simulated, and why each part matters
--------------------------------------------
* **Bilinear structure.** The clean signal is exactly ``D = C @ S``, the model
  MCR-ALS and PARAFAC2 assume. Ground truth is therefore not an approximation —
  it is the factorisation the algorithm is supposed to recover.
* **Homologous series.** A polyolefin pyrogram is a comb of alkane / 1-alkene /
  alkadiene triplets. Consecutive members are only a few seconds apart and their
  spectra are nearly identical, which makes the matrix rank-deficient. Handing an
  algorithm well-separated, distinct components would test nothing.
* **Deliberate co-elution.** Styrene sits on the C8 cluster, caprolactam on C14,
  benzoic acid on C13, and the PP trimer marker between C8 and C9. These are the
  real overlaps that make trace foreign-polymer detection hard, and they are placed
  on purpose.
* **Retention drift.** Each run gets a smooth, non-linear warp of its time axis,
  so peak positions are not reproducible between runs — the condition alignment
  and PARAFAC2 have to cope with.
* **Column bleed.** The rising background is placed on the siloxane ions
  (m/z 73, 147, 207, ...) rather than spread evenly, so baseline correction is
  tested on the interference pattern it actually has to handle.
* **Degradation.** A per-polymer ageing level shifts the iso-alkene / n-alkane
  ratio, adds ketones, aldehydes and acids, and skews the oligomer triad — the
  quantities the degradation index will later be built on.

Typical use::

    generator = SyntheticPyrogramGenerator(seed=42)
    sample = generator.generate(RECIPES["pcr_mixed_polyolefin"])

    sample.cube          # PyrogramDataCube, ready for the pipeline
    sample.truth         # PyrogramTruth: every component, area and RT
    sample.C, sample.S   # the factorisation to score a result against
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from data_schemas.acquisition import (
    AcquisitionConditions,
    GcConditions,
    MsConditions,
    PyrolysisConditions,
    SampleMetadata,
)
from data_schemas.enums import (
    MarkerRole,
    PolymerClass,
    RecyclateStream,
    SourceFormat,
)
from data_schemas.pyrogram import PyrogramMetadata
from data_schemas.truth import ComponentTruth, DriftTruth, PyrogramTruth
from pyrecycle_analytics.core.binning import axis_to_spec, make_nominal_mz_axis
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.core.peakshapes import profile_matrix
from tests.reference_spectra import (
    REFERENCE_COMPOUNDS,
    REFERENCE_METHOD_WINDOW_S,
    SILOXANE_BLEED_SPECTRUM,
    homologue_molecular_weight,
    homologue_quantifier_mz,
    homologue_role,
    homologue_spectrum,
)

__all__ = [
    "SyntheticComponent",
    "SyntheticPyrogram",
    "SyntheticPyrogramGenerator",
    "AlkaneRetentionModel",
    "NoiseModel",
    "BaselineModel",
    "PolymerFraction",
    "AdditiveSpike",
    "PyrogramRecipe",
    "RECIPES",
    "POLYMER_RESPONSE_FACTORS",
    "apply_retention_drift",
]

HomologueKind = Literal["alkane", "alkene", "diene", "isoalkene", "alkanone", "alkanal", "acid"]


# ---------------------------------------------------------------------------
# Instrument / method models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AlkaneRetentionModel:
    """Retention times of the n-alkane comb under a linear oven ramp.

    Under a linear temperature program n-alkane retention is very nearly linear in
    carbon number, with mild compression at the high-boiling end as the oven
    approaches its final temperature. That is exactly what this two-term model
    gives, and it is what makes the simulated comb spacing realistic: ~48 s
    between neighbours early in the run, ~30 s late.

    Attributes:
        reference_carbon: Carbon number the anchor retention time refers to.
        reference_rt_s: Retention time of ``reference_carbon`` in seconds.
        spacing_s: Retention increment per carbon at the anchor.
        compression_s: Quadratic term; larger values compress the late comb more.
    """

    reference_carbon: int = 8
    reference_rt_s: float = 420.0
    spacing_s: float = 48.0
    compression_s: float = 0.34

    def retention_time_s(self, carbon_number: int) -> float:
        """Retention time of the n-alkane with ``carbon_number`` carbons."""
        offset = carbon_number - self.reference_carbon
        return float(
            self.reference_rt_s
            + self.spacing_s * offset
            - self.compression_s * offset * abs(offset)
        )

    def local_spacing_s(self, carbon_number: int) -> float:
        """Retention gap to the next homologue at ``carbon_number``."""
        return abs(
            self.retention_time_s(carbon_number + 1) - self.retention_time_s(carbon_number)
        )


# Fractional offsets of the PE triplet members, expressed in units of the local
# comb spacing. On a non-polar phase the alkadiene elutes first, then the
# 1-alkene, then the n-alkane, all within a few seconds — a genuinely fused triplet.
_TRIPLET_OFFSETS: Mapping[str, float] = {
    "alkane": 0.0,
    "alkene": -0.10,
    "diene": -0.185,
    "isoalkene": -0.235,
    "alkanone": 0.30,
    "alkanal": 0.16,
    "acid": 0.62,
}


@dataclass(frozen=True, slots=True)
class NoiseModel:
    """Detector noise of the simulated MS.

    Attributes:
        detector_sigma: Standard deviation of the electronic noise added to every
            channel of every scan, in intensity units.
        shot_factor: Multiplier on ``sqrt(signal)`` shot noise. 1.0 corresponds to
            ideal Poisson counting statistics.
        spike_probability: Per-cell probability of a single-scan spike, modelling
            the ion-source discharges that corrupt real data.
        spike_magnitude: Mean magnitude of such a spike.
        quantise: Round to integers, as a real ADC does.
    """

    detector_sigma: float = 45.0
    shot_factor: float = 1.0
    spike_probability: float = 0.0
    spike_magnitude: float = 6000.0
    quantise: bool = True

    @classmethod
    def silent(cls) -> NoiseModel:
        """A noise model that adds nothing — for exact bilinearity tests."""
        return cls(detector_sigma=0.0, shot_factor=0.0, spike_probability=0.0, quantise=False)


@dataclass(frozen=True, slots=True)
class BaselineModel:
    """Background of the simulated run.

    Attributes:
        offset: Flat electronic offset present on every channel.
        bleed_amplitude: Peak intensity of the siloxane column bleed at the end
            of the run.
        bleed_onset_fraction: Fraction of the run after which bleed becomes
            visible, i.e. where the oven gets hot enough to degrade the phase.
        bleed_exponent: Steepness of the bleed ramp; 3 gives the characteristic
            late upswing.
        hump_amplitude: Broad unresolved-envelope pedestal spread over the
            hydrocarbon channels, modelling the fused polyolefin comb background.
        hump_center_fraction: Position of that pedestal's maximum in the run.
        hump_width_fraction: Width of the pedestal as a fraction of the run.
    """

    offset: float = 280.0
    bleed_amplitude: float = 3.2e4
    bleed_onset_fraction: float = 0.45
    bleed_exponent: float = 3.0
    hump_amplitude: float = 0.0
    hump_center_fraction: float = 0.45
    hump_width_fraction: float = 0.22

    @classmethod
    def none(cls) -> BaselineModel:
        """A baseline model that adds nothing — for exact bilinearity tests."""
        return cls(offset=0.0, bleed_amplitude=0.0, hump_amplitude=0.0)


# ---------------------------------------------------------------------------
# Recipe description
# ---------------------------------------------------------------------------

POLYMER_RESPONSE_FACTORS: Mapping[PolymerClass, float] = {
    PolymerClass.PE: 1.0,
    PolymerClass.PE_LD: 1.0,
    PolymerClass.PE_HD: 1.0,
    PolymerClass.PP: 0.95,
    PolymerClass.PS: 1.35,
    PolymerClass.ABS: 1.05,
    PolymerClass.SAN: 1.1,
    PolymerClass.PET: 0.42,
    PolymerClass.PA6: 0.68,
    PolymerClass.PA66: 0.35,
    PolymerClass.PVC: 0.30,
    PolymerClass.PC: 0.55,
    PolymerClass.PMMA: 0.90,
}
"""GC-amenable pyrolysate yield per unit mass, relative to polyethylene.

Mass fractions are *not* signal fractions. PVC loses most of its mass as HCl,
which barely reaches the detector; PET and the polyamides form polar, partly
non-eluting fragments. Ignoring this is one of the standard ways to get recyclate
quantification wrong by a factor of two, so the simulator builds it in and the
platform will have to correct for it.
"""


@dataclass(frozen=True, slots=True)
class PolymerFraction:
    """One polymer in a simulated blend.

    Attributes:
        polymer: Polymer class to simulate.
        mass_fraction: Gravimetric fraction of the blend, in ``[0, 1]``.
        degradation_level: Thermo-oxidative ageing in ``[0, 1]``. 0 is virgin
            material; 1 is heavily degraded multi-cycle recyclate.
    """

    polymer: PolymerClass
    mass_fraction: float
    degradation_level: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.mass_fraction <= 1.0:
            raise ValueError(f"mass_fraction {self.mass_fraction} outside [0, 1]")
        if not 0.0 <= self.degradation_level <= 1.0:
            raise ValueError(f"degradation_level {self.degradation_level} outside [0, 1]")


@dataclass(frozen=True, slots=True)
class AdditiveSpike:
    """An additive or contaminant added on top of the polymer blend.

    Attributes:
        compound: Key into :data:`~tests.reference_spectra.REFERENCE_COMPOUNDS`.
        relative_amount: Area relative to the blend's total polymer area. Realistic
            additive levels are 1e-4 to 1e-2, i.e. genuine trace analysis.
    """

    compound: str
    relative_amount: float

    def __post_init__(self) -> None:
        if self.relative_amount <= 0.0:
            raise ValueError(f"relative_amount must be > 0, got {self.relative_amount}")
        if self.compound not in REFERENCE_COMPOUNDS:
            raise KeyError(
                f"unknown reference compound {self.compound!r}; available: "
                f"{', '.join(sorted(REFERENCE_COMPOUNDS))}"
            )


@dataclass(frozen=True, slots=True)
class PyrogramRecipe:
    """Declarative description of a pyrogram to simulate.

    Attributes:
        name: Identifier used as the sample id and in the truth record.
        fractions: Polymer blend; mass fractions must sum to 1.
        additives: Additives and contaminants spiked on top.
        total_area: Total TIC area of the polymer signal, in intensity x seconds.
        stream: Provenance recorded in the sample metadata.
        drift: Retention warp for this run. ``None`` means no drift.
        noise: Detector noise model. ``None`` uses the generator's default.
        baseline: Background model. ``None`` uses the generator's default.
        description: Free-text note carried into the metadata.
    """

    name: str
    fractions: tuple[PolymerFraction, ...]
    additives: tuple[AdditiveSpike, ...] = ()
    total_area: float = 4.0e7
    stream: RecyclateStream = RecyclateStream.LAB_BLEND
    drift: DriftTruth | None = None
    noise: NoiseModel | None = None
    baseline: BaselineModel | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.fractions:
            raise ValueError("a recipe needs at least one polymer fraction")
        total = sum(fraction.mass_fraction for fraction in self.fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mass fractions must sum to 1.0, got {total:.6f}")
        if self.total_area <= 0.0:
            raise ValueError(f"total_area must be > 0, got {self.total_area}")
        seen: set[PolymerClass] = set()
        for fraction in self.fractions:
            if fraction.polymer in seen:
                raise ValueError(f"polymer {fraction.polymer} appears twice in the recipe")
            seen.add(fraction.polymer)

    def with_drift(self, drift: DriftTruth | None) -> PyrogramRecipe:
        """Return a copy of this recipe with a different retention warp."""
        return replace(self, drift=drift)


# ---------------------------------------------------------------------------
# Components and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticComponent:
    """A single species placed into a simulated pyrogram.

    Attributes:
        name: Unique compound name.
        polymer_class: Material it is attributed to.
        role: Function in its marker pattern.
        nominal_rt_s: Apex on the drift-free reference method.
        sigma_s: Gaussian width of the elution profile.
        tau_s: Exponential tailing constant.
        area: TIC area, in intensity x seconds.
        spectrum: m/z to relative intensity (base peak = 100).
        molecular_weight: Nominal molecular mass.
        quantifier_mz: Selective quantifier ion.
        carbon_number: Chain length for homologous-series members.
        source_areas: Area contributed by each producing polymer, keyed by
            ``PolymerClass`` value. Populated when duplicate compounds from
            different polymers are merged into the single peak the instrument sees.
    """

    name: str
    polymer_class: PolymerClass
    role: MarkerRole
    nominal_rt_s: float
    sigma_s: float
    tau_s: float
    area: float
    spectrum: Mapping[int, float]
    molecular_weight: float
    quantifier_mz: int
    carbon_number: int | None = None
    source_areas: Mapping[str, float] = field(default_factory=dict)

    @property
    def base_peak_mz(self) -> int:
        """m/z of the most intense ion in the tabulated spectrum."""
        return max(self.spectrum, key=lambda mz: self.spectrum[mz])

    @property
    def attributed_areas(self) -> dict[str, float]:
        """Per-polymer areas, falling back to sole attribution when unmerged."""
        return dict(self.source_areas) or {str(self.polymer_class): self.area}


@dataclass(slots=True)
class SyntheticPyrogram:
    """A simulated pyrogram together with its exact bilinear factorisation.

    Attributes:
        cube: The pyrogram as the pipeline sees it, with baseline and noise.
        truth: Scalar ground truth for every component.
        C: Pure elution profiles, shape ``(n_scans, n_components)``, scaled so
            each column integrates to that component's TIC area.
        S: Pure mass spectra, shape ``(n_components, n_mz)``, each row summing to
            1 over the acquired m/z window. With that normalisation ``C @ S`` has
            column sums equal to the TIC contributions, so ``C`` is directly the
            per-component TIC.
        baseline: Background matrix that was added, same shape as the cube.
        clean: The noise-free, baseline-free signal ``C @ S``.
        component_names: Component names in column order of ``C``.
    """

    cube: PyrogramDataCube
    truth: PyrogramTruth
    C: np.ndarray
    S: np.ndarray
    baseline: np.ndarray
    clean: np.ndarray
    component_names: tuple[str, ...]

    def index_of(self, name: str) -> int:
        """Column index in ``C`` of the component called ``name``.

        Args:
            name: Component name.

        Returns:
            Index into the component axis.

        Raises:
            KeyError: If no component carries that name.
        """
        try:
            return self.component_names.index(name)
        except ValueError as error:
            raise KeyError(f"no synthetic component named {name!r}") from error

    def profile(self, name: str) -> np.ndarray:
        """Pure elution profile of one component, shape ``(n_scans,)``."""
        return self.C[:, self.index_of(name)]

    def spectrum(self, name: str) -> np.ndarray:
        """Pure mass spectrum of one component, shape ``(n_mz,)``."""
        return self.S[self.index_of(name), :]

    @property
    def clean_cube(self) -> PyrogramDataCube:
        """The noise-free, baseline-free signal as a data cube.

        Useful as the target of a preprocessing test: correcting the noisy cube
        should approach this.
        """
        return self.cube.with_intensities(self.clean)

    def component_tic(self) -> np.ndarray:
        """Per-component TIC contributions, shape ``(n_scans, n_components)``.

        Equal to ``C`` by construction, since the spectra rows sum to one.
        """
        return self.C

    def resolution_of(self, name_a: str, name_b: str) -> float:
        """Chromatographic resolution between two named components."""
        first = self.truth.component_by_name(name_a)
        second = self.truth.component_by_name(name_b)
        width_sum = first.peak_width_s + second.peak_width_s
        if width_sum <= 0.0:
            return float("inf")
        return abs(second.retention_time_s - first.retention_time_s) / (2.0 * width_sum)


def apply_retention_drift(
    retention_times: np.ndarray | float,
    drift: DriftTruth,
    rt_start_s: float,
    rt_end_s: float,
) -> np.ndarray:
    """Warp retention times with a smooth, non-linear drift.

    Combines the four effects seen between real runs: a constant injection offset,
    a proportional stretch from carrier-gas flow change, a quadratic term from
    column ageing (which moves late peaks more than early ones), and a slow
    oscillation from pressure regulation.

    Args:
        retention_times: Time or times to warp, in seconds.
        drift: Warp parameters.
        rt_start_s: Run start, used to normalise the quadratic term.
        rt_end_s: Run end, used to normalise the quadratic term.

    Returns:
        Warped times, same shape as the input.

    Raises:
        ValueError: If the run window is empty.
    """
    span = rt_end_s - rt_start_s
    if span <= 0.0:
        raise ValueError(f"empty run window [{rt_start_s}, {rt_end_s}]")

    times = np.asarray(retention_times, dtype=np.float64)
    normalised = (times - rt_start_s) / span
    warped = (
        times
        + drift.offset_s
        + drift.linear_factor * (times - rt_start_s)
        + drift.quadratic_factor * normalised**2
    )
    if drift.oscillation_amplitude_s > 0.0:
        warped = warped + drift.oscillation_amplitude_s * np.sin(
            2.0 * np.pi * times / drift.oscillation_period_s
        )
    return warped


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SyntheticPyrogramGenerator:
    """Builds simulated pyrograms with exact ground truth.

    Attributes:
        rt_start_s: First scan time.
        rt_end_s: Last scan time.
        scan_rate_hz: Scans per second. 5 Hz over a 2 s peak sigma gives ~10-12
            points across a peak, the practical minimum for curve resolution.
        mz_low: Lowest acquired m/z (integer grid).
        mz_high: Highest acquired m/z.
        seed: Master random seed. Derived per-run seeds keep runs reproducible.
        retention: n-alkane comb retention model.
        base_sigma_s: Peak width at the start of the run.
        sigma_growth: Fractional width increase across the run — real peaks
            broaden with retention.
        base_tau_s: Tailing constant at the start of the run.
        noise: Default noise model.
        baseline: Default background model.
        carbon_range: Inclusive carbon-number window of the simulated homologous
            series.
    """

    rt_start_s: float = 60.0
    rt_end_s: float = 1800.0
    scan_rate_hz: float = 5.0
    mz_low: int = 29
    mz_high: int = 400
    seed: int = 0
    retention: AlkaneRetentionModel = field(default_factory=AlkaneRetentionModel)
    base_sigma_s: float = 1.9
    sigma_growth: float = 0.9
    base_tau_s: float = 0.8
    noise: NoiseModel = field(default_factory=NoiseModel)
    baseline: BaselineModel = field(default_factory=BaselineModel)
    carbon_range: tuple[int, int] = (6, 34)

    def __post_init__(self) -> None:
        if self.rt_end_s <= self.rt_start_s:
            raise ValueError(f"empty run window [{self.rt_start_s}, {self.rt_end_s}]")
        if self.scan_rate_hz <= 0.0:
            raise ValueError(f"scan_rate_hz must be > 0, got {self.scan_rate_hz}")
        if self.mz_high <= self.mz_low:
            raise ValueError(f"empty m/z window [{self.mz_low}, {self.mz_high}]")
        low, high = self.carbon_range
        if low < 4 or high <= low:
            raise ValueError(f"invalid carbon_range {self.carbon_range}")

    # ------------------------------------------------------------ axes / method

    @property
    def retention_times(self) -> np.ndarray:
        """Scan time axis in seconds."""
        n_scans = int(np.floor((self.rt_end_s - self.rt_start_s) * self.scan_rate_hz)) + 1
        return self.rt_start_s + np.arange(n_scans, dtype=np.float64) / self.scan_rate_hz

    @property
    def mz_axis(self) -> np.ndarray:
        """Integer m/z grid of the simulated acquisition."""
        return make_nominal_mz_axis(float(self.mz_low), float(self.mz_high))

    def _map_reference_rt(self, retention_time_s: float) -> float:
        """Map a tabulated retention time onto this generator's run window.

        Tabulated markers and the alkane comb are both defined on the reference
        method. If the generator is configured with a shorter or longer run, both
        are stretched by the same linear map so their relative positions — and
        therefore the intended co-elutions — survive.
        """
        ref_start, ref_end = REFERENCE_METHOD_WINDOW_S
        if abs(ref_start - self.rt_start_s) < 1e-9 and abs(ref_end - self.rt_end_s) < 1e-9:
            return retention_time_s
        scale = (self.rt_end_s - self.rt_start_s) / (ref_end - ref_start)
        return self.rt_start_s + (retention_time_s - ref_start) * scale

    def _width_at(self, retention_time_s: float) -> tuple[float, float]:
        """Peak sigma and tailing tau at a given retention time."""
        progress = (retention_time_s - self.rt_start_s) / (self.rt_end_s - self.rt_start_s)
        progress = float(np.clip(progress, 0.0, 1.0))
        sigma = self.base_sigma_s * (1.0 + self.sigma_growth * progress)
        tau = self.base_tau_s * (1.0 + 0.6 * progress)
        return sigma, tau

    # -------------------------------------------------------- series builders

    def _homologue_component(
        self,
        kind: HomologueKind,
        carbon_number: int,
        area: float,
        polymer_class: PolymerClass,
        *,
        name_prefix: str = "",
    ) -> SyntheticComponent:
        """Build one member of a homologous series."""
        anchor = self.retention.retention_time_s(carbon_number)
        spacing = self.retention.local_spacing_s(carbon_number)
        nominal = self._map_reference_rt(anchor + _TRIPLET_OFFSETS[kind] * spacing)
        sigma, tau = self._width_at(nominal)
        # Oxygenated homologues are polar and tail noticeably on a non-polar phase.
        if kind in {"alkanone", "alkanal", "acid"}:
            tau *= 2.2 if kind == "acid" else 1.5
        readable = {
            "alkane": "n-alkane",
            "alkene": "1-alkene",
            "diene": "alkadiene",
            "isoalkene": "iso-alkene",
            "alkanone": "2-alkanone",
            "alkanal": "n-alkanal",
            "acid": "n-alkanoic acid",
        }[kind]
        # The name identifies the *molecule*, not its origin: n-nonane from PE and
        # from PP is one compound giving one peak, and the merge step relies on that.
        return SyntheticComponent(
            name=f"{name_prefix}{readable} C{carbon_number}",
            polymer_class=polymer_class,
            role=homologue_role(kind),
            nominal_rt_s=nominal,
            sigma_s=sigma,
            tau_s=tau,
            area=area,
            spectrum=homologue_spectrum(kind, carbon_number),
            molecular_weight=float(homologue_molecular_weight(kind, carbon_number)),
            quantifier_mz=homologue_quantifier_mz(kind),
            carbon_number=carbon_number,
        )

    @staticmethod
    def _chain_length_weights(
        carbon_numbers: Sequence[int],
        center: float,
        sigma_low: float,
        sigma_high: float,
    ) -> np.ndarray:
        """Skewed chain-length distribution of a pyrolysate.

        Flash pyrolysis of a polyolefin yields a broad distribution peaking in the
        C11-C16 range with a long tail toward high carbon numbers, so the two sides
        get different widths.
        """
        numbers = np.asarray(carbon_numbers, dtype=np.float64)
        width = np.where(numbers < center, sigma_low, sigma_high)
        weights = np.exp(-0.5 * ((numbers - center) / width) ** 2)
        total = weights.sum()
        return weights / total if total > 0 else weights

    def polyolefin_series(
        self,
        polymer: PolymerClass,
        total_area: float,
        *,
        degradation_level: float = 0.0,
    ) -> list[SyntheticComponent]:
        """Homologous-series components of a polyolefin pyrolysate.

        Produces the alkane / 1-alkene / alkadiene triplets plus the iso-alkene
        that reports on branching, and — as ageing rises — the ketone, aldehyde
        and acid series that report on oxidation.

        Args:
            polymer: ``PE``, ``PE_LD``, ``PE_HD`` or ``PP``.
            total_area: Total TIC area to distribute over the series.
            degradation_level: Ageing in ``[0, 1]``.

        Returns:
            Components in no particular order.

        Raises:
            ValueError: If ``polymer`` is not a polyolefin.
        """
        if not polymer.is_polyolefin:
            raise ValueError(f"{polymer} is not a polyolefin")

        low, high = self.carbon_range
        carbons = list(range(low, high + 1))
        aged = float(np.clip(degradation_level, 0.0, 1.0))

        if polymer is PolymerClass.PP:
            # PP pyrolysis gives propene oligomers: strongly branched, concentrated
            # at multiples of three carbons, and decaying with oligomer order — the
            # C9 trimer is the dominant peak of a PP pyrogram, not the mid-chain
            # region that dominates a PE pyrogram.
            center, sigma_low, sigma_high = 9.5 - 1.5 * aged, 2.5, 5.5
            ratios = {"isoalkene": 1.0, "alkene": 0.30, "alkane": 0.22, "diene": 0.10}
            oligomer_boost = 3.2
        else:
            center = 13.5 - 2.5 * aged
            sigma_low, sigma_high = 4.0, 6.5
            branch = {
                PolymerClass.PE_LD: 0.145,
                PolymerClass.PE_HD: 0.035,
                PolymerClass.EVA: 0.11,
            }.get(polymer, 0.075)
            ratios = {
                "alkane": 1.0,
                # Chain scission during ageing shifts the balance toward alkenes.
                "alkene": 0.62 * (1.0 + 0.40 * aged),
                "diene": 0.18 * (1.0 + 0.25 * aged),
                # Secondary radical attack creates additional branch points.
                "isoalkene": branch * (1.0 + 3.0 * aged),
            }
            oligomer_boost = 1.0

        weights = self._chain_length_weights(carbons, center, sigma_low, sigma_high)
        if polymer is PolymerClass.PP:
            # Propene trimer/tetramer/pentamer stand out from the background.
            multiples = np.array(
                [oligomer_boost if carbon % 3 == 0 else 1.0 for carbon in carbons]
            )
            weights = weights * multiples
            weights = weights / weights.sum()

        # Oxidation products take a share of the total that grows with ageing.
        oxidised_share = 0.0015 + 0.075 * aged
        hydrocarbon_area = total_area * (1.0 - oxidised_share)

        components: list[SyntheticComponent] = []
        ratio_sum = sum(ratios.values())
        for carbon, weight in zip(carbons, weights, strict=True):
            for kind, ratio in ratios.items():
                if kind == "diene" and carbon < 4:
                    continue
                if kind == "isoalkene" and carbon < 6:
                    continue
                if kind == "alkene" and carbon < 3:
                    continue
                area = hydrocarbon_area * float(weight) * ratio / ratio_sum
                if area <= 0.0:
                    continue
                # The C9 branched alkene of a PP pyrolysate is one specific molecule:
                # 2,4-dimethyl-1-heptene, the propene trimer. It is *the* PP
                # identification marker, so it gets its tabulated spectrum and its
                # oligomer role rather than the generic branched-alkene pattern.
                # Substituting rather than adding matters — emitting both would
                # double-count the same compound.
                if polymer is PolymerClass.PP and kind == "isoalkene" and carbon == 9:
                    components.append(
                        self.discrete_marker(
                            "2,4-dimethyl-1-heptene",
                            area,
                            polymer_class=polymer,
                            carbon_number=carbon,
                        )
                    )
                    continue
                components.append(
                    self._homologue_component(kind, carbon, area, polymer)  # type: ignore[arg-type]
                )

        if oxidised_share > 0.0:
            components.extend(
                self._oxidation_series(
                    polymer, total_area * oxidised_share, center=center, aged=aged
                )
            )
        return components

    def _oxidation_series(
        self,
        polymer: PolymerClass,
        total_area: float,
        *,
        center: float,
        aged: float,
    ) -> list[SyntheticComponent]:
        """Ketone / aldehyde / acid series produced by thermo-oxidative ageing.

        The relative weight of the three families shifts with ageing: aldehydes are
        early intermediates, ketones accumulate, and acids dominate late. That
        ordering is what lets a degradation index distinguish "processed twice"
        from "processed six times".
        """
        low, high = self.carbon_range
        # Oxidation products are shorter than the parent hydrocarbon fragments.
        carbons = [carbon for carbon in range(max(low, 5), min(high, 22) + 1)]
        weights = self._chain_length_weights(carbons, max(center - 2.0, 7.0), 3.0, 4.5)

        family_weights = {
            "alkanal": 0.42 - 0.22 * aged,
            "alkanone": 0.38 + 0.05 * aged,
            "acid": 0.20 + 0.17 * aged,
        }
        family_sum = sum(family_weights.values())

        components: list[SyntheticComponent] = []
        for carbon, weight in zip(carbons, weights, strict=True):
            for kind, family_weight in family_weights.items():
                area = total_area * float(weight) * family_weight / family_sum
                if area <= 0.0:
                    continue
                components.append(
                    self._homologue_component(
                        kind,  # type: ignore[arg-type]
                        carbon,
                        area,
                        polymer,
                    )
                )
        return components

    def discrete_marker(
        self,
        compound_name: str,
        area: float,
        *,
        polymer_class: PolymerClass | None = None,
        carbon_number: int | None = None,
    ) -> SyntheticComponent:
        """Build a component from a tabulated reference compound.

        Args:
            compound_name: Key into :data:`REFERENCE_COMPOUNDS`.
            area: TIC area to assign.
            polymer_class: Override the tabulated attribution — needed because
                styrene is a PS marker in one blend and an ABS marker in another.
            carbon_number: Chain length, for tabulated compounds that are also
                members of a homologous series.

        Returns:
            The component.

        Raises:
            KeyError: If the compound is not tabulated.
        """
        try:
            reference = REFERENCE_COMPOUNDS[compound_name]
        except KeyError as error:
            raise KeyError(
                f"unknown reference compound {compound_name!r}; available: "
                f"{', '.join(sorted(REFERENCE_COMPOUNDS))}"
            ) from error

        nominal = self._map_reference_rt(reference.retention_time_s)
        sigma, tau = self._width_at(nominal)
        return SyntheticComponent(
            name=reference.name,
            polymer_class=polymer_class or reference.polymer_class,
            role=reference.role,
            nominal_rt_s=nominal,
            sigma_s=sigma * reference.peak_width_scale,
            tau_s=tau * reference.tailing_scale,
            area=area,
            spectrum=reference.spectrum,
            molecular_weight=reference.molecular_weight,
            quantifier_mz=reference.quantifier_mz,
            carbon_number=carbon_number,
        )

    def styrenic_series(
        self,
        polymer: PolymerClass,
        total_area: float,
        *,
        degradation_level: float = 0.0,
    ) -> list[SyntheticComponent]:
        """Marker pattern of a styrenic polymer.

        The quantitative content is the monomer : dimer : trimer triad. Ageing
        favours the monomer, because a shortened chain yields relatively more
        single-unit fragments — which is precisely why the triad *ratio*, not the
        styrene peak alone, is the identification criterion.

        Args:
            polymer: ``PS``, ``ABS`` or ``SAN``.
            total_area: Total TIC area for this polymer.
            degradation_level: Ageing in ``[0, 1]``.

        Returns:
            Components of the styrenic pattern.

        Raises:
            ValueError: If ``polymer`` is not styrenic.
        """
        if not polymer.is_styrenic:
            raise ValueError(f"{polymer} is not a styrenic polymer")
        aged = float(np.clip(degradation_level, 0.0, 1.0))

        pattern: dict[str, float] = {
            "styrene": 100.0 * (1.0 + 0.80 * aged),
            "2,4-diphenyl-1-butene": 12.0,
            "2,4,6-triphenyl-1-hexene": 6.0 * (1.0 - 0.50 * aged),
            "toluene": 6.5,
            "alpha-methylstyrene": 4.0,
            "benzene": 2.2,
            "benzaldehyde": 0.6 + 9.0 * aged,
            "acetophenone": 0.4 + 7.0 * aged,
        }
        if polymer is PolymerClass.SAN:
            pattern["acrylonitrile"] = 42.0
        if polymer is PolymerClass.ABS:
            pattern["acrylonitrile"] = 26.0
            pattern["4-vinylcyclohexene"] = 18.0

        total_weight = sum(pattern.values())
        return [
            self.discrete_marker(
                name, total_area * weight / total_weight, polymer_class=polymer
            )
            for name, weight in pattern.items()
        ]

    def condensation_polymer_series(
        self,
        polymer: PolymerClass,
        total_area: float,
        *,
        degradation_level: float = 0.0,
    ) -> list[SyntheticComponent]:
        """Marker pattern of a polyester, polyamide, polycarbonate or acrylic.

        These are the trace foreign fractions that matrix subtraction has to expose:
        in a 2 % PET contamination the benzoic-acid marker is three orders of
        magnitude below the polyolefin comb.

        Args:
            polymer: ``PET``, ``PA6``, ``PA66``, ``PC``, ``PMMA`` or ``PVC``.
            total_area: Total TIC area for this polymer.
            degradation_level: Ageing in ``[0, 1]``; hydrolytic ageing of PET
                raises free benzoic acid.

        Returns:
            Components of the marker pattern.

        Raises:
            ValueError: If no pattern is defined for ``polymer``.
        """
        aged = float(np.clip(degradation_level, 0.0, 1.0))
        patterns: dict[PolymerClass, dict[str, float]] = {
            PolymerClass.PET: {
                "benzoic acid": 100.0 * (1.0 + 1.2 * aged),
                "vinyl benzoate": 46.0,
                "divinyl terephthalate": 28.0 * (1.0 - 0.4 * aged),
                "biphenyl": 9.0,
                "acetophenone": 5.0,
            },
            PolymerClass.PA6: {
                "epsilon-caprolactam": 100.0,
                "hexanenitrile": 7.5,
                "cyclopentanone": 2.5,
            },
            PolymerClass.PA66: {
                "cyclopentanone": 100.0,
                "hexanenitrile": 34.0,
                "benzene": 4.0,
            },
            PolymerClass.PVC: {
                "hydrogen chloride": 100.0,
                "benzene": 34.0,
                "toluene": 12.0,
                "naphthalene": 9.0,
                "indene": 6.0,
                "chlorobenzene": 3.5,
            },
            PolymerClass.PC: {
                "bisphenol A": 100.0,
                "phenol": 32.0,
                "4-isopropenylphenol": 24.0,
            },
            PolymerClass.PMMA: {
                "methyl methacrylate": 100.0,
                "methyl methacrylate dimer": 8.0,
            },
        }
        pattern = patterns.get(polymer)
        if pattern is None:
            raise ValueError(
                f"no marker pattern defined for {polymer}; available: "
                f"{', '.join(sorted(str(key) for key in patterns))}"
            )
        total_weight = sum(pattern.values())
        return [
            self.discrete_marker(
                name, total_area * weight / total_weight, polymer_class=polymer
            )
            for name, weight in pattern.items()
        ]

    def components_for(self, fraction: PolymerFraction, area: float) -> list[SyntheticComponent]:
        """Dispatch to the right series builder for one blend fraction.

        Args:
            fraction: Polymer and its ageing level.
            area: TIC area allotted to this polymer.

        Returns:
            The polymer's components.

        Raises:
            ValueError: If the polymer class has no simulation rule.
        """
        polymer = fraction.polymer
        if polymer.is_polyolefin:
            return self.polyolefin_series(
                polymer, area, degradation_level=fraction.degradation_level
            )
        if polymer.is_styrenic:
            return self.styrenic_series(
                polymer, area, degradation_level=fraction.degradation_level
            )
        return self.condensation_polymer_series(
            polymer, area, degradation_level=fraction.degradation_level
        )

    # ------------------------------------------------------------- assembling

    def _spectra_matrix(
        self,
        components: Sequence[SyntheticComponent],
        mz_axis: np.ndarray,
    ) -> np.ndarray:
        """Build ``S`` with rows normalised to sum 1 over the acquired window.

        Ions outside the scan range are simply not seen by the instrument, so they
        are dropped and the remaining pattern is renormalised. That keeps the
        invariant "column of ``C`` equals this component's TIC" exact, which is
        what makes area comparisons in tests meaningful.
        """
        mz_index = {int(round(mz)): position for position, mz in enumerate(mz_axis)}
        spectra = np.zeros((len(components), mz_axis.size), dtype=np.float64)
        for row, component in enumerate(components):
            for mz, intensity in component.spectrum.items():
                position = mz_index.get(int(mz))
                if position is not None:
                    spectra[row, position] += intensity
            total = spectra[row].sum()
            if total <= 0.0:
                raise ValueError(
                    f"component {component.name!r} has no ion inside the acquired window "
                    f"[{mz_axis[0]:.0f}, {mz_axis[-1]:.0f}]"
                )
            spectra[row] /= total
        return spectra

    def _baseline_matrix(
        self,
        retention_times: np.ndarray,
        mz_axis: np.ndarray,
        model: BaselineModel,
    ) -> np.ndarray:
        """Background matrix: flat offset + siloxane bleed ramp + broad hump."""
        baseline = np.full((retention_times.size, mz_axis.size), model.offset, dtype=np.float64)

        if model.bleed_amplitude > 0.0:
            span = self.rt_end_s - self.rt_start_s
            onset = self.rt_start_s + model.bleed_onset_fraction * span
            progress = np.clip(
                (retention_times - onset) / max(self.rt_end_s - onset, 1e-9), 0.0, 1.0
            )
            ramp = model.bleed_amplitude * progress**model.bleed_exponent

            bleed_spectrum = np.zeros(mz_axis.size, dtype=np.float64)
            mz_index = {int(round(mz)): position for position, mz in enumerate(mz_axis)}
            for mz, intensity in SILOXANE_BLEED_SPECTRUM.items():
                position = mz_index.get(int(mz))
                if position is not None:
                    bleed_spectrum[position] += intensity
            peak = bleed_spectrum.max()
            if peak > 0.0:
                baseline += np.outer(ramp, bleed_spectrum / peak)

        if model.hump_amplitude > 0.0:
            span = self.rt_end_s - self.rt_start_s
            center = self.rt_start_s + model.hump_center_fraction * span
            width = max(model.hump_width_fraction * span, 1e-9)
            envelope = model.hump_amplitude * np.exp(
                -0.5 * ((retention_times - center) / width) ** 2
            )
            # The unresolved polyolefin envelope lives on the alkyl/alkenyl ions.
            hump_spectrum = np.zeros(mz_axis.size, dtype=np.float64)
            mz_index = {int(round(mz)): position for position, mz in enumerate(mz_axis)}
            for mz, intensity in ((41, 0.7), (43, 1.0), (55, 0.65), (57, 0.9), (71, 0.4)):
                position = mz_index.get(mz)
                if position is not None:
                    hump_spectrum[position] += intensity
            baseline += np.outer(envelope, hump_spectrum)

        return baseline

    def _add_noise(
        self,
        signal: np.ndarray,
        model: NoiseModel,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add shot noise, electronic noise, spikes and ADC quantisation."""
        noisy = signal
        if model.shot_factor > 0.0:
            noisy = noisy + rng.normal(
                0.0, model.shot_factor * np.sqrt(np.clip(signal, 0.0, None)), signal.shape
            )
        if model.detector_sigma > 0.0:
            noisy = noisy + rng.normal(0.0, model.detector_sigma, signal.shape)
        if model.spike_probability > 0.0:
            spikes = rng.random(signal.shape) < model.spike_probability
            if np.any(spikes):
                noisy = noisy + spikes * rng.exponential(model.spike_magnitude, signal.shape)
        # Ion counts cannot be negative; a real detector floors at zero.
        noisy = np.clip(noisy, 0.0, None)
        return np.round(noisy) if model.quantise else noisy

    def _metadata(
        self,
        recipe: PyrogramRecipe,
        retention_times: np.ndarray,
        mz_axis: np.ndarray,
        replicate: int,
    ) -> PyrogramMetadata:
        """Assemble metadata describing the simulated acquisition."""
        nominal = {
            str(fraction.polymer): fraction.mass_fraction for fraction in recipe.fractions
        }
        sample = SampleMetadata(
            sample_id=f"{recipe.name}-r{replicate}"[:64],
            description=recipe.description or f"Synthetic pyrogram from recipe {recipe.name!r}",
            stream=recipe.stream,
            replicate=replicate,
            nominal_composition=nominal,
            operator="SyntheticPyrogramGenerator",
        )
        acquisition = AcquisitionConditions(
            pyrolysis=PyrolysisConditions(temperature_c=600.0, duration_s=12.0),
            gc=GcConditions(),
            ms=MsConditions(
                mz_low=float(mz_axis[0]),
                mz_high=float(mz_axis[-1]),
                scan_rate_hz=self.scan_rate_hz,
            ),
            instrument_model="PyRecycle-Analytics synthetic pyrolyser-GC/MS",
            method_name="reference 10 C/min ramp",
        )
        return PyrogramMetadata(
            sample=sample,
            acquisition=acquisition,
            source_format=SourceFormat.SYNTHETIC,
            source_path=None,
            n_scans=int(retention_times.size),
            mz_axis=axis_to_spec(mz_axis),
            rt_start_s=float(retention_times[0]),
            rt_end_s=float(retention_times[-1]),
            extra={"recipe": recipe.name, "generator_seed": str(self.seed)},
        )

    @staticmethod
    def _merge_duplicate_compounds(
        components: Sequence[SyntheticComponent],
    ) -> list[SyntheticComponent]:
        """Collapse repeated compounds into the single peak the instrument sees.

        A blend routinely produces the same molecule from several polymers:
        n-nonane from both PE and PP, benzene from both PS and PVC, acetophenone
        from both PS and PET. Leaving those as separate components would make the
        ground truth unidentifiable — two entries with identical spectrum and
        identical retention time cannot be told apart by any algorithm, so a
        recovery score against them would be meaningless.

        Areas are therefore summed into one component, its class and role are taken
        from the dominant contributor, and the per-polymer split is preserved in
        ``source_areas`` for quantification tests.

        Args:
            components: Possibly duplicated components.

        Returns:
            One component per distinct compound, in first-seen order.
        """
        merged: dict[str, SyntheticComponent] = {}
        for component in components:
            contribution = component.attributed_areas
            existing = merged.get(component.name)
            if existing is None:
                merged[component.name] = replace(component, source_areas=contribution)
                continue

            combined = dict(existing.source_areas)
            for polymer, area in contribution.items():
                combined[polymer] = combined.get(polymer, 0.0) + area
            dominant = component if component.area > existing.area else existing
            merged[component.name] = replace(
                dominant,
                area=existing.area + component.area,
                source_areas=combined,
            )
        return list(merged.values())

    def build_components(self, recipe: PyrogramRecipe) -> list[SyntheticComponent]:
        """Expand a recipe into its component list, areas included.

        Mass fractions are converted to signal areas through
        :data:`POLYMER_RESPONSE_FACTORS`, so a 2 % PVC contamination contributes far
        less than 2 % of the chromatogram — the discrepancy any honest recyclate
        quantification has to undo.

        Compounds produced by more than one polymer in the blend are merged into a
        single peak, with the per-polymer split retained in ``source_areas``.

        Args:
            recipe: Blend description.

        Returns:
            One component per distinct compound in the blend and its additives.
        """
        response = {
            fraction.polymer: POLYMER_RESPONSE_FACTORS.get(fraction.polymer, 1.0)
            * fraction.mass_fraction
            for fraction in recipe.fractions
        }
        response_total = sum(response.values())
        if response_total <= 0.0:
            raise ValueError("recipe has zero total response; check mass fractions")

        components: list[SyntheticComponent] = []
        for fraction in recipe.fractions:
            share = response[fraction.polymer] / response_total
            area = recipe.total_area * share
            if area <= 0.0:
                continue
            components.extend(self.components_for(fraction, area))

        for spike in recipe.additives:
            components.append(
                self.discrete_marker(spike.compound, recipe.total_area * spike.relative_amount)
            )
        return self._merge_duplicate_compounds(components)

    def generate(
        self,
        recipe: PyrogramRecipe,
        *,
        replicate: int = 1,
        noiseless: bool = False,
    ) -> SyntheticPyrogram:
        """Simulate one pyrogram from a recipe.

        Args:
            recipe: Blend and instrument-condition description.
            replicate: Replicate number. Also perturbs the derived random seed, so
                replicates differ in noise while sharing composition.
            noiseless: Produce an exactly bilinear cube with no baseline and no
                noise. Use this for tests that need ``D == C @ S`` to hold exactly.

        Returns:
            The simulated pyrogram with its ground truth and factorisation.

        Raises:
            ValueError: If every component falls outside the run window, or a
                component has no ion inside the acquired m/z range.
        """
        # crc32, not hash(): Python randomises string hashing per process, which
        # would make "same seed, same recipe" reproducible only within one run.
        recipe_key = zlib.crc32(recipe.name.encode("utf-8"))
        rng = np.random.default_rng((self.seed, replicate, recipe_key))
        retention_times = self.retention_times
        mz_axis = self.mz_axis

        components = self.build_components(recipe)
        drift = recipe.drift or DriftTruth()

        drifted_rt = [
            float(
                apply_retention_drift(
                    component.nominal_rt_s, drift, self.rt_start_s, self.rt_end_s
                )
            )
            for component in components
        ]

        # Peaks whose apex leaves the acquisition window are simply not observed.
        # They must not appear in the ground truth either, or every recovery metric
        # would be penalised for components that were never measurable.
        keep = [
            index
            for index, retention_time in enumerate(drifted_rt)
            if self.rt_start_s <= retention_time <= self.rt_end_s
        ]
        if not keep:
            raise ValueError(
                f"recipe {recipe.name!r}: no component elutes inside "
                f"[{self.rt_start_s}, {self.rt_end_s}] s after drift"
            )
        components = [components[index] for index in keep]
        drifted_rt = [drifted_rt[index] for index in keep]

        # Sorting by retention time makes C banded, which is how a chromatogram is
        # naturally structured and what the windowing in the deconvolution expects.
        order = np.argsort(np.asarray(drifted_rt), kind="stable")
        components = [components[int(index)] for index in order]
        drifted_rt = [drifted_rt[int(index)] for index in order]

        spectra = self._spectra_matrix(components, mz_axis)
        profiles = profile_matrix(
            retention_times,
            centers=np.asarray(drifted_rt),
            sigmas=np.asarray([component.sigma_s for component in components]),
            taus=np.asarray([component.tau_s for component in components]),
            areas=np.asarray([component.area for component in components]),
        )
        clean = profiles @ spectra

        noise_model = NoiseModel.silent() if noiseless else (recipe.noise or self.noise)
        baseline_model = BaselineModel.none() if noiseless else (recipe.baseline or self.baseline)

        baseline = self._baseline_matrix(retention_times, mz_axis, baseline_model)
        observed = self._add_noise(clean + baseline, noise_model, rng)

        metadata = self._metadata(recipe, retention_times, mz_axis, replicate)
        cube = PyrogramDataCube(
            retention_times=retention_times,
            mz_axis=mz_axis,
            intensities=observed,
            metadata=metadata,
        )

        truth = PyrogramTruth(
            recipe_name=recipe.name,
            seed=self.seed,
            blend_fractions={
                str(fraction.polymer): fraction.mass_fraction for fraction in recipe.fractions
            },
            degradation_levels={
                str(fraction.polymer): fraction.degradation_level
                for fraction in recipe.fractions
            },
            components=tuple(
                ComponentTruth(
                    index=index,
                    name=component.name,
                    polymer_class=component.polymer_class,
                    role=component.role,
                    carbon_number=component.carbon_number,
                    molecular_weight=component.molecular_weight,
                    retention_time_s=retention_time,
                    nominal_retention_time_s=component.nominal_rt_s,
                    peak_width_s=component.sigma_s,
                    tailing_s=component.tau_s,
                    area=component.area,
                    base_peak_mz=component.base_peak_mz,
                    quantifier_mz=component.quantifier_mz,
                    source_areas=component.attributed_areas,
                )
                for index, (component, retention_time) in enumerate(
                    zip(components, drifted_rt, strict=True)
                )
            ),
            drift=drift,
            noise_sigma=noise_model.detector_sigma,
            baseline_included=bool(np.any(baseline)),
        )

        return SyntheticPyrogram(
            cube=cube,
            truth=truth,
            C=profiles,
            S=spectra,
            baseline=baseline,
            clean=clean,
            component_names=tuple(component.name for component in components),
        )

    def generate_series(
        self,
        recipe: PyrogramRecipe,
        n_runs: int,
        *,
        drift_scale: float = 1.0,
    ) -> list[SyntheticPyrogram]:
        """Simulate several runs of the same blend with different drift.

        This is the input a trilinear method needs: same chemistry, same
        composition, but retention axes that do not line up. PARAFAC2 exists
        precisely to handle that, and alignment methods have to be scored on it.

        Args:
            recipe: Blend description; its own ``drift`` field is ignored.
            n_runs: Number of runs to simulate, at least 1.
            drift_scale: Multiplier on the sampled drift magnitude. 0 gives
                perfectly aligned runs.

        Returns:
            One :class:`SyntheticPyrogram` per run.

        Raises:
            ValueError: If ``n_runs`` is below 1.
        """
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")

        drift_rng = np.random.default_rng((self.seed, 0xD21F7, n_runs))
        runs: list[SyntheticPyrogram] = []
        for replicate in range(1, n_runs + 1):
            drift = DriftTruth(
                offset_s=float(drift_rng.normal(0.0, 1.8)) * drift_scale,
                linear_factor=float(drift_rng.normal(0.0, 1.5e-3)) * drift_scale,
                quadratic_factor=float(drift_rng.normal(0.0, 9.0)) * drift_scale,
                oscillation_amplitude_s=abs(float(drift_rng.normal(0.0, 0.7))) * drift_scale,
                oscillation_period_s=float(drift_rng.uniform(400.0, 900.0)),
            )
            runs.append(self.generate(recipe.with_drift(drift), replicate=replicate))
        return runs


# ---------------------------------------------------------------------------
# Preset recipes
# ---------------------------------------------------------------------------


def _recipe(
    name: str,
    fractions: Iterable[tuple[PolymerClass, float] | tuple[PolymerClass, float, float]],
    *,
    additives: Iterable[tuple[str, float]] = (),
    stream: RecyclateStream = RecyclateStream.LAB_BLEND,
    total_area: float = 4.0e7,
    description: str = "",
    drift: DriftTruth | None = None,
) -> PyrogramRecipe:
    """Terser constructor for the preset table."""
    parsed: list[PolymerFraction] = []
    for entry in fractions:
        polymer, mass_fraction = entry[0], entry[1]
        degradation = entry[2] if len(entry) > 2 else 0.0  # type: ignore[misc]
        parsed.append(
            PolymerFraction(
                polymer=polymer, mass_fraction=mass_fraction, degradation_level=degradation
            )
        )
    return PyrogramRecipe(
        name=name,
        fractions=tuple(parsed),
        additives=tuple(
            AdditiveSpike(compound=compound, relative_amount=amount)
            for compound, amount in additives
        ),
        total_area=total_area,
        stream=stream,
        description=description,
        drift=drift,
    )


RECIPES: Mapping[str, PyrogramRecipe] = {
    "virgin_hdpe": _recipe(
        "virgin_hdpe",
        [(PolymerClass.PE_HD, 1.0)],
        additives=[("2,6-di-tert-butyl-4-methylphenol (BHT)", 8e-4)],
        stream=RecyclateStream.VIRGIN,
        description="Clean HDPE reference: pure alkane/alkene comb, minimal branching.",
    ),
    "virgin_ldpe": _recipe(
        "virgin_ldpe",
        [(PolymerClass.PE_LD, 1.0)],
        additives=[("2,6-di-tert-butyl-4-methylphenol (BHT)", 6e-4)],
        stream=RecyclateStream.VIRGIN,
        description="LDPE reference: same comb as HDPE but four times the iso-alkene "
        "content — the branching-index contrast case.",
    ),
    "virgin_pp": _recipe(
        "virgin_pp",
        [(PolymerClass.PP, 1.0)],
        additives=[("2,4-di-tert-butylphenol", 5e-4)],
        stream=RecyclateStream.VIRGIN,
        description="PP reference dominated by branched propene oligomers.",
    ),
    "virgin_ps": _recipe(
        "virgin_ps",
        [(PolymerClass.PS, 1.0)],
        stream=RecyclateStream.VIRGIN,
        description="PS reference for the monomer : dimer : trimer triad.",
    ),
    "aged_hdpe": _recipe(
        "aged_hdpe",
        [(PolymerClass.PE_HD, 1.0, 0.75)],
        additives=[("octadecanoic acid (stearic acid)", 2e-3)],
        stream=RecyclateStream.PCR_HDPE,
        description="Heavily reprocessed HDPE: elevated iso-alkenes and a full "
        "ketone/aldehyde/acid series.",
    ),
    "pcr_mixed_polyolefin": _recipe(
        "pcr_mixed_polyolefin",
        [
            (PolymerClass.PE_LD, 0.58, 0.35),
            (PolymerClass.PP, 0.32, 0.30),
            (PolymerClass.PS, 0.045, 0.20),
            (PolymerClass.PET, 0.030),
            (PolymerClass.PA6, 0.015),
            (PolymerClass.PVC, 0.010),
        ],
        additives=[
            ("bis(2-ethylhexyl) phthalate (DEHP)", 1.5e-3),
            ("erucamide", 9e-4),
            ("2,6-di-tert-butyl-4-methylphenol (BHT)", 4e-4),
            ("2-(2H-benzotriazol-2-yl)-p-cresol fragment", 2e-4),
        ],
        stream=RecyclateStream.PCR_MIXED_POLYOLEFIN,
        description="Realistic mixed-polyolefin sorting fraction: PE/PP matrix with "
        "percent-level PS/PET/PA6/PVC traces that only appear after matrix subtraction.",
    ),
    "pcr_ps_with_traces": _recipe(
        "pcr_ps_with_traces",
        [
            (PolymerClass.PS, 0.90, 0.40),
            (PolymerClass.PE_HD, 0.06),
            (PolymerClass.PP, 0.03),
            (PolymerClass.PET, 0.01),
        ],
        additives=[("dibutyl phthalate (DBP)", 8e-4)],
        stream=RecyclateStream.PCR_PS,
        description="PS-dominated recyclate; tests marker-triad quantification against "
        "a polyolefin trace background.",
    ),
    "weee_flame_retarded": _recipe(
        "weee_flame_retarded",
        [
            (PolymerClass.ABS, 0.55, 0.45),
            (PolymerClass.PC, 0.25),
            (PolymerClass.PS, 0.12),
            (PolymerClass.PP, 0.08),
        ],
        additives=[
            ("tetrabromobisphenol A fragment", 4e-3),
            ("2,4-di-tert-butylphenol", 6e-4),
        ],
        stream=RecyclateStream.PCR_WEEE,
        description="WEEE fraction with a brominated flame retardant — the "
        "legislative-marker detection case.",
    ),
    "coelution_stress": _recipe(
        "coelution_stress",
        [
            (PolymerClass.PE_HD, 0.70),
            (PolymerClass.PS, 0.10),
            (PolymerClass.PA6, 0.10),
            (PolymerClass.PP, 0.10),
        ],
        total_area=2.0e7,
        description="Worst case for curve resolution: styrene lands on the C8 cluster, "
        "caprolactam on C14, and the PP trimer marker between C8 and C9.",
    ),
    "trace_pet_in_polyolefin": _recipe(
        "trace_pet_in_polyolefin",
        [
            (PolymerClass.PE_LD, 0.945),
            (PolymerClass.PP, 0.050),
            (PolymerClass.PET, 0.005),
        ],
        description="0.5 % PET in a polyolefin matrix — after response correction the "
        "PET markers sit near 0.2 % of the total signal. The matrix-subtraction target.",
    ),
}
"""Named benchmark recipes, from clean references to the hard trace cases."""
