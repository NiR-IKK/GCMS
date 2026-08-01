"""Assembling the digital recyclate passport from an analysis.

This is where signal shares become reported percentages, and it is the step with
the most opportunity to mislead. Two conversions happen here and both are made
explicit in the output rather than folded silently into a number:

**Signal share to mass share.** Polymers differ several-fold in how much
GC-amenable pyrolysate they give per unit mass — PVC loses most of its mass as
HCl, PET and the polyamides form partly non-eluting polar fragments, polystyrene
over-responds. Dividing each polymer's signal by its response factor corrects the
systematic part of that. What it cannot do is establish the absolute scale, which
needs gravimetric reference blends. The result is therefore labelled
``RESPONSE_CORRECTED``, not ``CALIBRATED``.

**Unassigned signal stays unassigned.** Whatever could not be attributed to an
identified polymer is reported as its own share instead of being distributed over
the identified ones. Normalising it away would inflate every reported percentage
in exact proportion to how much of the sample the method failed to explain.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_schemas.acquisition import AcquisitionConditions, SampleMetadata
from data_schemas.enums import PolymerClass
from data_schemas.passport import (
    AnalysisProvenance,
    CalibrationStatus,
    ConfidenceLevel,
    DegradationAssessment,
    PolymerFraction,
    RecyclatePassport,
    RegulatoryFinding,
)
from pyrecycle_analytics import __version__
from pyrecycle_analytics.degradation.engine import DegradationIndices
from pyrecycle_analytics.identification.engine import (
    CompoundIdentification,
    PolymerFinding,
)
from pyrecycle_analytics.library.repository import MarkerLibrary
from pyrecycle_analytics.matrix.backbone import BackboneSplit
from pyrecycle_analytics.reporting.reach import NON_GC_AMENABLE_NOTE, REACH_WATCHLIST

__all__ = ["MatrixPolyolefinEvidence", "PassportInputs", "build_passport"]


@dataclass(slots=True)
class MatrixPolyolefinEvidence:
    """The polyolefin content, taken from the subtracted matrix.

    Necessary because of the order the pipeline runs in. The polyolefin backbone
    is removed *before* curve resolution, so by the time identification runs, the
    dominant fraction of a mixed-polyolefin recyclate has no markers left to find.
    Left at that, a passport reports polypropylene at 2 % where the truth is 32 %,
    with the missing signal parked in "unassigned" — technically honest, but
    useless.

    The matrix model already measured what was removed, so that measurement is
    carried forward as evidence in its own right. The grade comes from the
    branching index: propylene units put a methyl branch on every third carbon, so
    a PP-dominated matrix produces far more branched-alkene signal than any
    polyethylene.

    Where two polyolefins share the comb, the branching index alone cannot divide
    them: it is one number for the whole comb, and a PE/PP blend produces an
    intermediate value that reads as a single intermediate polymer. Measured, that
    put polypropylene at 5.5 % against a true 31 %. Supplying ``split`` — an
    unmixing against virgin reference runs — replaces the single assignment with
    the actual division. Without references the old behaviour stands, and the
    passport says the polyolefin fraction is unsplit.

    Attributes:
        signal_fraction: Share of the run's total signal attributed to the comb.
        branching_index: Iso-alkene over n-alkane signal, deciding the grade when
            no reference split is available.
        n_clusters: Homologues found, reported as evidence strength.
        split: Division of the comb between reference materials, when virgin
            references were supplied.
    """

    signal_fraction: float
    branching_index: float
    n_clusters: int
    split: BackboneSplit | None = None

    # Measured on the benchmark: HDPE 0.137, LDPE 0.186, PP 1.239. The gap between
    # any polyethylene and polypropylene is nearly an order of magnitude, so the
    # split is robust; the LDPE/HDPE boundary is much tighter and is reported as a
    # grade hint rather than a firm assignment.
    PP_BRANCHING_THRESHOLD = 0.5
    LDPE_BRANCHING_THRESHOLD = 0.16

    @property
    def polymer(self) -> PolymerClass:
        """Grade of the comb, or its dominant material when split."""
        if self.split is not None:
            return self.split.dominant
        if self.branching_index >= self.PP_BRANCHING_THRESHOLD:
            return PolymerClass.PP
        if self.branching_index >= self.LDPE_BRANCHING_THRESHOLD:
            return PolymerClass.PE_LD
        return PolymerClass.PE_HD

    # A reference material that takes less than this share of the comb is below
    # what the unmixing resolves; reporting it would put a polymer on the passport
    # on the strength of fitting noise. Dropped shares are folded back into the
    # remaining ones so the polyolefin total stays intact.
    MIN_REPORTED_SHARE = 0.01

    def contributions(self) -> list[tuple[PolymerClass, float, bool]]:
        """One entry per polyolefin the comb is attributed to.

        Collinear reference materials are combined rather than reported side by
        side. Two spectra a cosine of 0.999 apart carry almost no independent
        information, and printing them as separate percentages would dress a
        rounding difference up as a measurement.

        Returns:
            ``(polymer, share of the comb, split into more than one material)``.
        """
        if self.split is None:
            return [(self.polymer, 1.0, False)]

        shares = dict(self.split.shares)
        for group, combined in self.split.merged:
            present = [polymer for polymer in group if polymer in shares]
            if len(present) < 2:
                continue
            label = max(present, key=lambda polymer: shares[polymer])
            for polymer in present:
                shares.pop(polymer)
            shares[label] = combined

        kept = {
            polymer: share
            for polymer, share in shares.items()
            if share >= self.MIN_REPORTED_SHARE
        }
        if not kept:  # pragma: no cover - only when the comb is pure noise
            return [(self.polymer, 1.0, True)]

        total = sum(kept.values())
        return [
            (polymer, share / total, True)
            for polymer, share in sorted(kept.items(), key=lambda item: -item[1])
        ]

    @property
    def grade_note(self) -> str:
        """How firm the assignment is."""
        if self.split is not None:
            listed = ", ".join(
                f"{polymer} {100.0 * share:.1f} %"
                for polymer, share, _ in self.contributions()
            )
            return (
                f"the polyolefin comb was divided against virgin references "
                f"({self.split.endmember_source}): {listed}. "
                f"{self.split.residual_fraction:.1%} of the comb spectrum is not "
                "explained by those references."
            )
        if self.branching_index >= self.PP_BRANCHING_THRESHOLD:
            return (
                f"branching index {self.branching_index:.2f} is far above any "
                "polyethylene; assigned to polypropylene. No virgin references "
                "were supplied, so the comb is reported as one polymer — a PE/PP "
                "blend would read as a single intermediate grade"
            )
        return (
            f"branching index {self.branching_index:.2f} indicates polyethylene; "
            "the LD/HD boundary is narrow, so treat the grade as a hint and the "
            "polyolefin total as the firm figure. No virgin references were "
            "supplied, so a PE/PP blend would read as one intermediate grade"
        )


@dataclass(slots=True)
class PassportInputs:
    """Everything the passport builder needs.

    Attributes:
        sample: Identity of the analysed material.
        acquisition: Instrumental conditions.
        polymer_findings: Output of the identification engine.
        compound_identifications: Individual compound hits, for the regulatory
            section.
        degradation: Ageing indices, if computed.
        degradation_reference: Virgin-reference indices, without which the ageing
            figures are not interpretable.
        reference_sample_id: Identity of that reference.
        total_signal_area: Total resolved area, the denominator of every share.
        matrix_polyolefin: Polyolefin content taken from the subtracted matrix.
            Without it the dominant fraction of a polyolefin recyclate goes
            unreported, because its markers were removed before identification.
        provenance: Traceability record.
    """

    sample: SampleMetadata
    acquisition: AcquisitionConditions
    polymer_findings: list[PolymerFinding]
    compound_identifications: list[CompoundIdentification]
    degradation: DegradationIndices | None = None
    degradation_reference: DegradationIndices | None = None
    reference_sample_id: str | None = None
    total_signal_area: float = 0.0
    matrix_polyolefin: MatrixPolyolefinEvidence | None = None
    provenance: AnalysisProvenance | None = None


def _response_corrected_shares(
    findings: list[PolymerFinding],
    library: MarkerLibrary,
    *,
    pyrolysis_temperature_c: float,
) -> dict[PolymerClass, float]:
    """Convert per-polymer signal areas into response-corrected mass shares.

    Divides each polymer's signal by its GC-amenable yield relative to
    polyethylene, then normalises. Without this a 1 % PVC contamination reads as
    0.3 % and a 4.5 % polystyrene fraction reads as 6 %.
    """
    corrected: dict[PolymerClass, float] = {}
    for finding in findings:
        factor = library.response_factor(
            str(finding.polymer), pyrolysis_temperature_c=pyrolysis_temperature_c
        )
        if factor <= 0.0:  # pragma: no cover - defensive
            continue
        corrected[finding.polymer] = finding.signal_area / factor

    total = sum(corrected.values())
    if total <= 0.0:
        return {}
    return {polymer: value / total for polymer, value in corrected.items()}


def _regulatory_findings(
    identifications: list[CompoundIdentification],
) -> list[RegulatoryFinding]:
    """Check the watchlist against what was identified."""
    by_compound = {hit.compound_name: hit for hit in identifications}
    findings: list[RegulatoryFinding] = []

    for substance in REACH_WATCHLIST:
        hit = by_compound.get(substance.marker_compound)
        detected = hit is not None
        findings.append(
            RegulatoryFinding(
                substance=substance.name,
                regulation=substance.regulation,
                cas_number=substance.cas_number,
                detected=detected,
                detection_confidence=(
                    ConfidenceLevel.from_score(hit.score) if hit is not None else None
                ),
                limit_percent=substance.limit_percent,
                # Deliberately left unset: presence is established, concentration
                # is not. Filling this in would require calibration standards.
                quantified_percent=None,
                marker_compound=substance.marker_compound,
                note=substance.note,
            )
        )
    return findings


def _degradation_assessment(
    indices: DegradationIndices | None,
    reference: DegradationIndices | None,
    reference_sample_id: str | None,
) -> DegradationAssessment | None:
    """Package the ageing indices, with the virgin comparison where available."""
    if indices is None:
        return None

    relative: dict[str, float] = {}
    notes = list(indices.notes)
    if reference is not None:
        try:
            relative = {
                key: value
                for key, value in indices.relative_to(reference).items()
                if value != float("inf")
            }
        except ValueError as error:  # pragma: no cover - defensive
            notes.append(f"virgin comparison unavailable: {error}")
    else:
        notes.append(
            "no virgin reference was supplied; the absolute indices below are not "
            "interpretable on their own and must not be compared across polymers"
        )

    return DegradationAssessment(
        carbonyl_index=indices.carbonyl_index,
        acid_share=indices.acid_share,
        alkene_to_alkane=indices.alkene_to_alkane,
        branching_index=indices.branching_index,
        mean_chain_length=indices.mean_carbon_number,
        relative_to_virgin=relative,
        reference_sample_id=reference_sample_id,
        notes=tuple(notes),
    )


def build_passport(
    inputs: PassportInputs,
    library: MarkerLibrary,
    *,
    calibrated: bool = False,
) -> RecyclatePassport:
    """Assemble a recyclate passport.

    Args:
        inputs: Analysis results and sample context.
        library: Seeded marker library, for the response factors.
        calibrated: Set only when gravimetric reference blends were measured under
            the same method. Off by default, because they usually were not, and a
            passport that claims calibration it does not have is worse than one
            that admits the limitation.

    Returns:
        The passport, with every figure carrying its calibration status.
    """
    status = (
        CalibrationStatus.CALIBRATED if calibrated else CalibrationStatus.RESPONSE_CORRECTED
    )
    shares = _response_corrected_shares(
        inputs.polymer_findings,
        library,
        pyrolysis_temperature_c=inputs.acquisition.pyrolysis.temperature_c,
    )

    # Only the identified polymers' signal is attributable. The rest is reported
    # as unassigned rather than being normalised away.
    identified_area = sum(finding.signal_area for finding in inputs.polymer_findings)
    attributable = (
        min(identified_area / inputs.total_signal_area, 1.0)
        if inputs.total_signal_area > 0.0
        else 0.0
    )

    # The polyolefin matrix was removed before identification, so its share comes
    # from the matrix model rather than from marker evidence. It is claimed first,
    # and what the resolved components can account for is scaled into the rest.
    matrix = inputs.matrix_polyolefin
    matrix_share = 0.0
    matrix_polymers: set[PolymerClass] = set()
    fractions: list[PolymerFraction] = []
    if matrix is not None and matrix.signal_fraction > 0.0:
        # Response-correct each polyolefin separately. With the comb reported as
        # one polymer this made no difference; split, it does — polypropylene and
        # polyethylene do not give the same GC-amenable yield per unit mass.
        corrected: list[tuple[PolymerClass, float, bool]] = []
        for polymer, share, was_split in matrix.contributions():
            factor = library.response_factor(
                str(polymer),
                pyrolysis_temperature_c=inputs.acquisition.pyrolysis.temperature_c,
            )
            corrected.append(
                (polymer, matrix.signal_fraction * share / max(factor, 1e-9), was_split)
            )

        # Cap the polyolefin block as a whole, not each part: capping the parts
        # would silently change their ratio to one another.
        block = sum(value for _, value, _ in corrected)
        scale = min(1.0, 1.0 / block) if block > 1.0 else 1.0
        matrix_share = 100.0 * block * scale

        for polymer, value, was_split in corrected:
            share = 100.0 * value * scale
            if share <= 0.0:
                continue
            matrix_polymers.add(polymer)
            fractions.append(
                PolymerFraction(
                    polymer=polymer,
                    share_percent=share,
                    calibration_status=status,
                    confidence=(
                        ConfidenceLevel.HIGH
                        if matrix.n_clusters >= 8
                        else ConfidenceLevel.MEDIUM
                    ),
                    marker_pattern=(
                        "polyolefin homologous series, split against virgin references"
                        if was_split
                        else "polyolefin homologous series"
                    ),
                    markers_found=(f"n-alkane comb, {matrix.n_clusters} homologues",),
                    markers_expected=1,
                    uncertainty_percent=(
                        round(share * 0.25, 4) if calibrated else None
                    ),
                )
            )

    remaining = max(0.0, 100.0 - matrix_share)
    for finding in inputs.polymer_findings:
        if finding.polymer in matrix_polymers:
            continue
        share = shares.get(finding.polymer, 0.0) * attributable * remaining
        single_marker = len(finding.members_found) == 1
        fractions.append(
            PolymerFraction(
                polymer=finding.polymer,
                share_percent=share,
                calibration_status=status,
                confidence=ConfidenceLevel.from_score(
                    finding.confidence, single_marker=single_marker
                ),
                marker_pattern=finding.pattern_name,
                markers_found=finding.members_found,
                markers_expected=finding.members_expected,
                uncertainty_percent=(
                    round(share * 0.25, 4) if calibrated else None
                ),
            )
        )

    unassigned = max(0.0, 100.0 - sum(fraction.share_percent for fraction in fractions))

    provenance = inputs.provenance or AnalysisProvenance()
    extra_warnings: list[str] = [NON_GC_AMENABLE_NOTE]
    if matrix is not None:
        extra_warnings.append(matrix.grade_note)
    provenance = provenance.model_copy(
        update={
            "n_compounds_identified": len(inputs.compound_identifications),
            "software_version": __version__,
            "warnings": (*provenance.warnings, *extra_warnings),
        }
    )

    return RecyclatePassport(
        sample=inputs.sample,
        acquisition=inputs.acquisition,
        polymer_fractions=tuple(
            sorted(fractions, key=lambda fraction: fraction.share_percent, reverse=True)
        ),
        unassigned_share_percent=unassigned,
        regulatory_findings=tuple(_regulatory_findings(inputs.compound_identifications)),
        degradation=_degradation_assessment(
            inputs.degradation, inputs.degradation_reference, inputs.reference_sample_id
        ),
        provenance=provenance,
    )
