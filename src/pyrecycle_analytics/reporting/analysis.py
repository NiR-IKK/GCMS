"""One call from a raw pyrogram to a finished recyclate passport.

Chains everything the platform does: preprocess, subtract the polyolefin matrix,
resolve the co-eluting windows, calibrate the retention-index ladder from the
sample's own alkane comb, identify compounds and polymers, compute the ageing
indices, and assemble the passport.

Kept separate from the FastAPI layer so that the whole analysis is usable — and
testable — without a web server.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_schemas.passport import AnalysisProvenance, RecyclatePassport
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.deconvolution.pipeline import (
    DeconvolutionConfig,
    resolve_pyrogram,
)
from pyrecycle_analytics.deconvolution.result import ResolutionResult
from pyrecycle_analytics.degradation.engine import (
    DegradationIndices,
    compute_degradation_indices,
)
from pyrecycle_analytics.identification.engine import (
    CompoundIdentification,
    IdentificationSettings,
    PolymerFinding,
    identify_compounds,
    identify_polymers,
)
from pyrecycle_analytics.library.repository import MarkerLibrary
from pyrecycle_analytics.library.retention_index import (
    RetentionIndexCalibration,
    RetentionIndexError,
    calibrate_from_comb,
)
from pyrecycle_analytics.matrix.polyolefin import (
    MatrixSubtractionError,
    MatrixSubtractionResult,
    detect_homologue_comb,
    subtract_polymer_matrix,
)
from pyrecycle_analytics.preprocessing.pipeline import preprocess
from pyrecycle_analytics.reporting.passport import (
    MatrixPolyolefinEvidence,
    PassportInputs,
    build_passport,
)

__all__ = ["AnalysisResult", "analyse_pyrogram"]


@dataclass(slots=True)
class AnalysisResult:
    """Everything one analysis produced.

    The intermediate results are kept alongside the passport rather than thrown
    away. A passport states conclusions; a reader who wants to check one needs the
    evidence it rests on — which spectrum matched, how well, at which retention
    index — and recomputing the chain to see it would take minutes.

    Attributes:
        passport: The deliverable.
        degradation: Ageing indices, kept separately so a caller can reuse them
            as the virgin reference for a later sample.
        resolution: The resolved profiles and spectra behind the passport.
        identifications: Every library match, including those too weak to reach
            the passport.
        polymer_findings: Marker patterns that fired, with their consistency.
        matrix: The fitted polyolefin matrix, or ``None`` when no comb was found.
        n_components: Components the resolution returned.
        n_identified: Compounds the library matched.
        warnings: Everything the chain wanted to flag.
    """

    passport: RecyclatePassport
    degradation: DegradationIndices | None
    resolution: ResolutionResult
    identifications: tuple[CompoundIdentification, ...]
    polymer_findings: tuple[PolymerFinding, ...]
    matrix: MatrixSubtractionResult | None
    n_components: int
    n_identified: int
    warnings: tuple[str, ...]


def analyse_pyrogram(
    cube: PyrogramDataCube,
    library: MarkerLibrary,
    *,
    config: DeconvolutionConfig | None = None,
    identification: IdentificationSettings | None = None,
    first_carbon_number: int | None = None,
    degradation_reference: DegradationIndices | None = None,
    reference_sample_id: str | None = None,
    calibrated: bool = False,
) -> AnalysisResult:
    """Run the full chain and produce a recyclate passport.

    Args:
        cube: Ingested pyrogram.
        library: Seeded marker library.
        config: Deconvolution settings; defaults are used when omitted.
        identification: Identification thresholds.
        first_carbon_number: Carbon number of the first alkane-comb member, if
            known. Anchors the retention-index ladder; without it the ladder is
            estimated and labelled as such.
        degradation_reference: Ageing indices of virgin material of the same
            polymer. Without them the degradation figures are reported but marked
            as not interpretable.
        reference_sample_id: Identity of that reference, for the passport.
        calibrated: Whether gravimetric reference blends back the percentages.

    Returns:
        The passport plus the intermediate results worth keeping.
    """
    config = config or DeconvolutionConfig()
    warnings: list[str] = []

    report = resolve_pyrogram(cube, config)
    warnings.extend(report.warnings)

    # The ageing indices need the alkane comb intact, so they are computed on the
    # preprocessed cube before subtraction rather than on the residual.
    preprocessed = preprocess(cube, config.preprocessing)

    calibration: RetentionIndexCalibration | None = None
    matrix_model = None
    subtraction: MatrixSubtractionResult | None = None
    matrix_evidence: MatrixPolyolefinEvidence | None = None
    try:
        detection = detect_homologue_comb(preprocessed, matrix_type=config.matrix_type)
        subtraction = subtract_polymer_matrix(preprocessed, config.matrix_type)
        matrix_model = subtraction.model
        calibration = calibrate_from_comb(
            detection.apex_times_s,
            first_carbon_number=first_carbon_number,
            dead_time_s=None if first_carbon_number is not None else 36.0,
        )
    except (MatrixSubtractionError, RetentionIndexError) as error:
        warnings.append(
            f"no retention-index ladder available ({error}); identification falls "
            "back to spectral evidence alone, which is less selective"
        )

    identifications = identify_compounds(
        report.result, library, calibration=calibration, settings=identification
    )
    findings = identify_polymers(identifications, library)

    degradation: DegradationIndices | None = None
    try:
        degradation = compute_degradation_indices(
            preprocessed, matrix_model=matrix_model
        )
    except ValueError as error:
        warnings.append(f"degradation indices not computed: {error}")

    if subtraction is not None and matrix_model is not None and degradation is not None:
        matrix_evidence = MatrixPolyolefinEvidence(
            signal_fraction=subtraction.explained_fraction,
            branching_index=degradation.branching_index,
            n_clusters=matrix_model.detection.n_clusters,
        )

    provenance = AnalysisProvenance(
        source_file=(
            str(cube.metadata.source_path) if cube.metadata.source_path else None
        ),
        source_checksum=cube.metadata.source_checksum,
        preprocessing_steps=tuple(
            step.name for step in report.processed.metadata.preprocessing
        ),
        matrix_subtracted=bool(report.diagnostics.get("matrix_subtracted")),
        matrix_explained_fraction=report.matrix_explained_fraction,
        n_components_resolved=report.result.n_components,
        retention_index_anchor=(
            calibration.anchor_confidence if calibration is not None else None
        ),
        warnings=tuple(warnings),
    )

    passport = build_passport(
        PassportInputs(
            sample=cube.metadata.sample,
            acquisition=cube.metadata.acquisition,
            polymer_findings=findings,
            compound_identifications=identifications,
            degradation=degradation,
            degradation_reference=degradation_reference,
            reference_sample_id=reference_sample_id,
            total_signal_area=float(report.result.areas.sum()),
            matrix_polyolefin=matrix_evidence,
            provenance=provenance,
        ),
        library,
        calibrated=calibrated,
    )

    return AnalysisResult(
        passport=passport,
        degradation=degradation,
        resolution=report.result,
        identifications=tuple(identifications),
        polymer_findings=tuple(findings),
        matrix=subtraction,
        n_components=report.result.n_components,
        n_identified=len(identifications),
        warnings=tuple(warnings),
    )
