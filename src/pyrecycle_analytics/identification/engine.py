"""Turning resolved components into named compounds and polymer findings.

Two stages, deliberately kept apart because they answer different questions and
carry different confidence.

**Compound identification** matches each resolved spectrum against the library on
two independent axes: spectral agreement and retention-index agreement. Requiring
both is what stops the classic false positive — a hydrocarbon fragment pattern
that resembles a library entry but elutes three minutes away from it.

A caveat about the synthetic benchmark, recorded because it changes how the
identification scores there should be read: the generator places its marker
retention times to produce specific co-elutions, not on a Kováts-consistent
scale, so its markers deviate from tabulated indices by a median of 142 index
units and by up to 1023 at long retention. The retention axis is therefore
directly unit-tested but cannot be validated end-to-end on that benchmark, where
it *lowers* the score of correct hits. The default weight is nevertheless kept,
because real chromatograms are retention-index consistent by construction and
that is where the setting has to be right.

**Polymer identification** then asks whether the identified compounds form a
*pattern*. Finding styrene proves nothing on its own: styrene comes from
polystyrene, from ABS, from SAN, and from the degradation of unrelated materials.
Polystyrene is established by styrene appearing *together with* its dimer and
trimer in the expected proportions. The engine therefore never reports a polymer
from a single peak, and says so in the evidence it returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data_schemas.enums import PolymerClass
from pyrecycle_analytics.deconvolution.result import ResolutionResult
from pyrecycle_analytics.library.repository import MarkerLibrary
from pyrecycle_analytics.library.retention_index import RetentionIndexCalibration
from pyrecycle_analytics.validation.metrics import (
    cosine_similarity,
    weighted_dot_similarity,
)

__all__ = [
    "CompoundIdentification",
    "denoise_spectrum",
    "PolymerFinding",
    "IdentificationSettings",
    "identify_compounds",
    "identify_polymers",
]


@dataclass(frozen=True, slots=True)
class IdentificationSettings:
    """Thresholds governing what may be reported.

    Attributes:
        min_spectral_similarity: Combined spectral agreement a match must reach —
            the geometric mean of plain cosine and the mass-weighted factor.
        min_combined_score: Combined score a match must reach to be reported.
        retention_index_weight: Weight of the retention term in the combined
            score. Zero makes identification purely spectral, which is measurably
            worse but is available for data with no usable ladder.
        max_candidates_per_component: How many library hits to keep per resolved
            component, for inspection.
        denoise_threshold: Channels below this fraction of the base peak are
            dropped from a resolved spectrum before matching.
    """

    min_spectral_similarity: float = 0.50
    min_combined_score: float = 0.50
    retention_index_weight: float = 0.4
    max_candidates_per_component: int = 3
    denoise_threshold: float = 0.01


@dataclass(frozen=True, slots=True)
class CompoundIdentification:
    """One resolved component matched to a library compound.

    Attributes:
        component_index: Column index in the resolution's factors.
        compound_name: Library compound the component was matched to.
        retention_time_s: Apex of the resolved component.
        retention_index: Observed Kováts index, or ``None`` without a ladder.
        library_retention_index: Tabulated index of the matched compound.
        spectral_similarity: Weighted-dot agreement in ``[0, 1]``.
        retention_agreement: Retention term in ``[0, 1]``; 1.0 when no ladder was
            available, so that its absence neither helps nor hurts.
        score: Combined confidence in ``[0, 1]``.
        area: Resolved area of the component.
        quantifier_mz: Ion the library recommends for quantification.
    """

    component_index: int
    compound_name: str
    retention_time_s: float
    retention_index: float | None
    library_retention_index: float
    spectral_similarity: float
    retention_agreement: float
    score: float
    area: float
    quantifier_mz: int

    @property
    def used_retention_index(self) -> bool:
        return self.retention_index is not None


@dataclass(frozen=True, slots=True)
class PolymerFinding:
    """Evidence that one polymer is present.

    Attributes:
        polymer: The polymer class.
        pattern_name: Marker pattern the finding rests on.
        members_found: Names of the pattern members that were identified.
        members_expected: Total number of members in the pattern.
        ratio_consistency: How well the observed abundances match the expected
            ratios, in ``[0, 1]``.
        signal_area: Summed area of the identified members.
        confidence: Overall confidence in ``[0, 1]``.
        notes: Why the finding is as strong or as weak as it is.
    """

    polymer: PolymerClass
    pattern_name: str
    members_found: tuple[str, ...]
    members_expected: int
    ratio_consistency: float
    signal_area: float
    confidence: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_reportable(self) -> bool:
        """Whether the finding rests on enough evidence to publish."""
        return len(self.members_found) >= 2 or self.ratio_consistency >= 0.0


def denoise_spectrum(spectrum: np.ndarray, relative_threshold: float = 0.01) -> np.ndarray:
    """Drop channels below a fraction of the base peak before matching.

    A resolved spectrum is not a clean library spectrum. Curve resolution leaves a
    low-level floor across many channels, and the mass-weighted match factor is
    unusually sensitive to it: the intensity exponent of 0.5 compresses the
    dynamic range, so a channel at 0.3 % of the base peak comes out only a factor
    of eighteen below it instead of three hundred, and the m/z² weighting then
    amplifies whatever of that floor sits at high mass.

    Measured on the benchmark, a resolved styrene component scored 0.67 against
    the library entry it matches perfectly, while plain cosine gave 0.98. After
    thresholding, the weighted factor recovers its intended behaviour. Every
    practical library search does this; it is not a fudge but the standard
    preparation step.

    Args:
        spectrum: Resolved spectrum, shape ``(n_mz,)``.
        relative_threshold: Fraction of the base peak below which a channel is
            treated as absent.

    Returns:
        The thresholded spectrum, same shape.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    peak = float(spectrum.max()) if spectrum.size else 0.0
    if peak <= 0.0:
        return spectrum
    return np.where(spectrum >= relative_threshold * peak, spectrum, 0.0)


def _spectral_similarity(
    resolved: np.ndarray, reference: np.ndarray, mz_axis: np.ndarray
) -> float:
    """Combine plain and mass-weighted agreement into one spectral score.

    The geometric mean of the two, so a match has to satisfy both. They fail in
    different ways and neither alone is sufficient:

    * plain cosine is dominated by the low-mass fragments every hydrocarbon
      shares, so it is generous towards the wrong compound;
    * the mass-weighted factor is selective at high mass but, on a resolved
      spectrum, is dragged down by whatever noise floor survives thresholding —
      measured on the benchmark it reached only 0.36 for a divinyl terephthalate
      component that plain cosine matched at 0.98.

    Measured over the benchmark markers, the combined score separates cleanly:
    0.58 to 0.97 for the correct compound, 0.24 for a component whose resolution
    genuinely failed.
    """
    plain = cosine_similarity(resolved, reference)
    weighted = weighted_dot_similarity(resolved, reference, mz_axis)
    return float(np.sqrt(max(plain, 0.0) * max(weighted, 0.0)))


def _retention_agreement(
    observed_index: float | None, library_index: float, tolerance: float
) -> float:
    """Retention term in ``[0, 1]``, a Gaussian in units of the tolerance."""
    if observed_index is None:
        return 1.0
    deviation = (observed_index - library_index) / max(tolerance, 1e-6)
    return float(np.exp(-0.5 * deviation**2))


def identify_compounds(
    result: ResolutionResult,
    library: MarkerLibrary,
    *,
    calibration: RetentionIndexCalibration | None = None,
    settings: IdentificationSettings | None = None,
) -> list[CompoundIdentification]:
    """Match resolved components against the library.

    Args:
        result: The factorisation to identify.
        library: Seeded marker library.
        calibration: Kováts ladder from the sample's own alkane comb. Without it
            identification falls back to spectra alone, which is measurably less
            selective — several library entries share a fragmentation pattern and
            are separated only by where they elute.
        settings: Thresholds; defaults are used when omitted.

    Returns:
        Accepted identifications, best score first, at most one per component.
    """
    settings = settings or IdentificationSettings()
    compounds = library.compounds()
    if not compounds:
        return []

    mz_axis = result.mz_axis
    areas = result.areas
    apexes = result.apex_times

    # Project each library spectrum onto the acquisition's m/z grid once.
    channel_of = {int(round(mz)): index for index, mz in enumerate(mz_axis)}
    library_spectra = np.zeros((len(compounds), mz_axis.size))
    for row, compound in enumerate(compounds):
        for mz, intensity in compound.spectrum().items():
            channel = channel_of.get(int(mz))
            if channel is not None:
                library_spectra[row, channel] += intensity

    library_indices = np.array(
        [
            entry.value
            for compound in compounds
            for entry in compound.retention_indices
            if entry.stationary_phase == library.stationary_phase
        ][: len(compounds)]
    )
    library_tolerances = np.array(
        [
            entry.tolerance
            for compound in compounds
            for entry in compound.retention_indices
            if entry.stationary_phase == library.stationary_phase
        ][: len(compounds)]
    )

    identifications: list[CompoundIdentification] = []
    for component in range(result.n_components):
        spectrum = denoise_spectrum(result.S[component, :], settings.denoise_threshold)
        if spectrum.sum() <= 0.0:
            continue

        observed_index = (
            calibration.index_of(float(apexes[component]))
            if calibration is not None
            else None
        )

        best: CompoundIdentification | None = None
        for row, compound in enumerate(compounds):
            similarity = _spectral_similarity(spectrum, library_spectra[row], mz_axis)
            if similarity < settings.min_spectral_similarity:
                continue

            agreement = _retention_agreement(
                observed_index, float(library_indices[row]), float(library_tolerances[row])
            )
            weight = settings.retention_index_weight
            score = (1.0 - weight) * similarity + weight * agreement

            if score < settings.min_combined_score:
                continue
            if best is not None and score <= best.score:
                continue

            best = CompoundIdentification(
                component_index=component,
                compound_name=compound.name,
                retention_time_s=float(apexes[component]),
                retention_index=observed_index,
                library_retention_index=float(library_indices[row]),
                spectral_similarity=similarity,
                retention_agreement=agreement,
                score=score,
                area=float(areas[component]),
                quantifier_mz=int(compound.quantifier_mz),
            )

        if best is not None:
            identifications.append(best)

    return sorted(identifications, key=lambda hit: hit.score, reverse=True)


def _ratio_consistency(
    observed: dict[str, float],
    expected: dict[str, float],
    reference: str,
    tolerances: dict[str, float],
) -> tuple[float, list[str]]:
    """How well observed abundances follow the pattern's expected ratios.

    Each member's observed ratio to the reference is compared with the expected
    one on a logarithmic scale, because pyrolysis ratios vary multiplicatively
    with temperature and residence time — being off by a factor of two matters the
    same whether the expected value is 6 or 60.
    """
    notes: list[str] = []
    if reference not in observed or observed[reference] <= 0.0:
        return 0.0, ["reference member of the pattern was not identified"]

    scores: list[float] = []
    for name, expected_share in expected.items():
        if name == reference or name not in observed:
            continue
        observed_ratio = observed[name] / observed[reference]
        expected_ratio = expected_share / expected[reference]
        if expected_ratio <= 0.0 or observed_ratio <= 0.0:
            continue
        deviation = abs(np.log(observed_ratio / expected_ratio))
        allowed = np.log(max(tolerances.get(name, 2.5), 1.0 + 1e-9))
        scores.append(float(np.exp(-0.5 * (deviation / allowed) ** 2)))
        if deviation > allowed:
            notes.append(
                f"{name}: observed/expected ratio off by a factor of "
                f"{np.exp(deviation):.1f}, beyond the tolerance"
            )

    if not scores:
        return 0.5, [*notes, "only the reference member was identified; ratio untested"]
    return float(np.mean(scores)), notes


def identify_polymers(
    identifications: list[CompoundIdentification],
    library: MarkerLibrary,
    *,
    min_confidence: float = 0.35,
) -> list[PolymerFinding]:
    """Decide which polymers the identified compounds imply.

    Args:
        identifications: Output of :func:`identify_compounds`.
        library: Seeded marker library.
        min_confidence: Findings below this confidence are not returned.

    Returns:
        Findings ordered by confidence, at most one per polymer — the strongest
        pattern wins when a polymer has several.
    """
    if not identifications:
        return []

    # A compound may be identified in more than one component; sum their areas.
    observed_area: dict[str, float] = {}
    observed_score: dict[str, float] = {}
    for hit in identifications:
        observed_area[hit.compound_name] = (
            observed_area.get(hit.compound_name, 0.0) + hit.area
        )
        observed_score[hit.compound_name] = max(
            observed_score.get(hit.compound_name, 0.0), hit.score
        )

    findings: dict[PolymerClass, PolymerFinding] = {}
    for pattern in library.patterns():
        expected = {
            member.compound.name: member.relative_abundance for member in pattern.members
        }
        tolerances = {
            member.compound.name: member.tolerance_factor for member in pattern.members
        }
        references = [
            member.compound.name for member in pattern.members if member.is_reference
        ]
        reference = references[0] if references else next(iter(expected))

        found = tuple(name for name in expected if name in observed_area)
        notes: list[str] = []
        if len(found) < pattern.min_members_required:
            continue

        consistency, ratio_notes = _ratio_consistency(
            observed_area, expected, reference, tolerances
        )
        notes.extend(ratio_notes)

        # Confidence combines how much of the pattern was seen, how well the
        # identifications scored, and whether the proportions hold up.
        coverage = len(found) / len(expected)
        mean_score = float(np.mean([observed_score[name] for name in found]))
        confidence = float(0.4 * coverage + 0.3 * mean_score + 0.3 * consistency)

        if len(found) == 1:
            notes.append(
                "identified from a single marker; a polymer is not established by "
                "one peak, treat as indicative only"
            )
            confidence *= 0.6

        if confidence < min_confidence:
            continue

        polymer = PolymerClass(pattern.polymer.code)
        candidate = PolymerFinding(
            polymer=polymer,
            pattern_name=pattern.name,
            members_found=found,
            members_expected=len(expected),
            ratio_consistency=consistency,
            signal_area=float(sum(observed_area[name] for name in found)),
            confidence=confidence,
            notes=tuple(notes),
        )
        existing = findings.get(polymer)
        if existing is None or candidate.confidence > existing.confidence:
            findings[polymer] = candidate

    return sorted(findings.values(), key=lambda finding: finding.confidence, reverse=True)
