"""Tests für Bibliothek, Retentionsindex und Identifikation.

Der wichtigste Test in dieser Datei prüft keine Chemie, sondern die
Methodik: dass die Bibliothek **nicht** aus derselben Spektrentabelle stammt wie
der synthetische Benchmark. Täte sie es, würde sich die Identifikation gegen ihre
eigene Quelle validieren und jede Trefferquote wäre wertlos.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_schemas.enums import PolymerClass
from pyrecycle_analytics.deconvolution import DeconvolutionConfig, resolve_pyrogram
from pyrecycle_analytics.identification import (
    IdentificationSettings,
    denoise_spectrum,
    identify_compounds,
    identify_polymers,
)
from pyrecycle_analytics.library import (
    MARKER_PATTERNS,
    REFERENCE_COMPOUNDS,
    MarkerLibrary,
    RetentionIndexError,
    calibrate_from_comb,
    create_library,
    estimate_first_carbon_number,
    seed_library,
)
from pyrecycle_analytics.matrix import detect_homologue_comb
from pyrecycle_analytics.preprocessing import PreprocessingConfig, preprocess
from tests.synthetic_data import SyntheticPyrogram


@pytest.fixture(scope="module")
def library() -> MarkerLibrary:
    return MarkerLibrary.in_memory()


class TestLibraryIndependence:
    """Die methodische Kernanforderung von MS3."""

    def test_library_is_not_a_copy_of_the_benchmark_spectra(self) -> None:
        """Bibliothek und Simulator-Tabelle müssen sich unterscheiden.

        Beide beschreiben dieselben Verbindungen, aber aus unabhängigen Quellen —
        so wie ein Labor seine Bibliothek aus der Literatur baut und nicht aus dem
        Gerät, mit dem es später misst. Wären die Zahlen identisch, würde die
        Identifikation gegen ihre eigene Quelle geprüft.
        """
        from tests.reference_spectra import REFERENCE_COMPOUNDS as BENCHMARK

        shared = set(REFERENCE_COMPOUNDS) & set(BENCHMARK)
        assert len(shared) >= 15, "die Tabellen sollen dieselben Verbindungen abdecken"

        identical = [
            name
            for name in shared
            if dict(REFERENCE_COMPOUNDS[name].spectrum) == dict(BENCHMARK[name].spectrum)
        ]
        assert not identical, (
            f"identische Spektren in Bibliothek und Benchmark: {identical}. "
            "Die Identifikation würde sich gegen ihre eigene Quelle validieren."
        )

    def test_library_adds_retention_indices_the_benchmark_does_not_have(self) -> None:
        """Der RI ist eine echte zusätzliche Evidenzachse, nicht abgeleitet."""
        from tests.reference_spectra import ReferenceCompound

        assert not hasattr(ReferenceCompound, "retention_index")
        assert all(entry.retention_index > 0 for entry in REFERENCE_COMPOUNDS.values())


class TestLibrarySchema:
    def test_seeding_populates_every_table(self, library: MarkerLibrary) -> None:
        assert len(library.compounds()) == len(REFERENCE_COMPOUNDS)
        assert len(library.patterns()) == len(MARKER_PATTERNS)
        assert len(library.polymer_codes()) >= 10

    def test_seeding_is_idempotent(self) -> None:
        engine = create_library()
        seed_library(engine)
        first = len(MarkerLibrary(engine).compounds())
        seed_library(engine)
        assert len(MarkerLibrary(engine).compounds()) == first

    def test_spectra_and_indices_round_trip(self, library: MarkerLibrary) -> None:
        styrene = library.compound("styrene")
        assert styrene.spectrum()[104] == pytest.approx(100.0)
        assert library.retention_index("styrene") == pytest.approx(891.0)

    def test_unknown_compound_is_reported(self, library: MarkerLibrary) -> None:
        with pytest.raises(KeyError, match="not in the marker library"):
            library.compound("unobtainium stearate")

    def test_response_factors_are_persisted(self, library: MarkerLibrary) -> None:
        """Ohne sie ist jede Prozentangabe im Rezyklat-Pass falsch."""
        assert library.response_factor("PET") == pytest.approx(0.42)
        assert library.response_factor("PVC") == pytest.approx(0.30)
        assert library.response_factor("PE") == pytest.approx(1.0)

    def test_unknown_polymer_gets_a_neutral_response_factor(
        self, library: MarkerLibrary
    ) -> None:
        assert library.response_factor("PTFE") == 1.0

    def test_every_pattern_has_a_reference_member(self, library: MarkerLibrary) -> None:
        for pattern in library.patterns():
            assert any(member.is_reference for member in pattern.members)

    def test_no_pattern_identifies_a_polymer_from_a_single_peak_alone(self) -> None:
        """Muster mit nur einem Mitglied müssen als schwach gekennzeichnet sein."""
        single_member = [
            pattern for pattern in MARKER_PATTERNS if len(pattern.members) == 1
        ]
        for pattern in single_member:
            assert pattern.min_members_required == 1
            assert "diagnostic" in pattern.description.lower()


class TestRetentionIndex:
    def test_alkane_anchors_land_on_hundreds(self) -> None:
        """Definitionsgemäß: der n-Alkan mit n Kohlenstoffen hat RI = 100n."""
        times = np.array([420.0, 468.0, 515.0, 561.0, 607.0])
        calibration = calibrate_from_comb(times, first_carbon_number=8)
        for position, time in enumerate(times):
            assert calibration.index_of(float(time)) == pytest.approx(
                100.0 * (8 + position)
            )

    def test_interpolates_between_anchors(self) -> None:
        calibration = calibrate_from_comb(
            np.array([420.0, 470.0, 520.0]), first_carbon_number=8
        )
        assert calibration.index_of(445.0) == pytest.approx(850.0, abs=1.0)

    def test_extrapolates_outside_the_ladder(self) -> None:
        calibration = calibrate_from_comb(
            np.array([420.0, 470.0, 520.0]), first_carbon_number=8
        )
        assert calibration.index_of(370.0) < 800.0
        assert calibration.index_of(570.0) > 1000.0

    def test_anchor_estimation_from_the_dead_time(self) -> None:
        times = 420.0 + 48.0 * np.arange(8)
        # Bei 48 s je Kohlenstoff und C8 bei 420 s liegt die Totzeit bei ~36 s.
        assert estimate_first_carbon_number(times, dead_time_s=36.0) == 8

    def test_estimated_calibration_is_labelled_as_such(self) -> None:
        """Ein geschätzter Anker darf nicht wie ein gemessener aussehen."""
        times = 420.0 + 48.0 * np.arange(6)
        calibration = calibrate_from_comb(times, dead_time_s=36.0)
        assert calibration.anchor_confidence == "estimated"
        assert "estimated" in calibration.note
        anchored = calibrate_from_comb(times, first_carbon_number=8)
        assert anchored.anchor_confidence == "anchored"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({}, "supply either first_carbon_number"),
            ({"dead_time_s": 500.0}, "must be positive and precede"),
        ],
    )
    def test_invalid_anchoring_is_rejected(self, kwargs: dict, message: str) -> None:  # noqa: ANN001
        with pytest.raises(RetentionIndexError, match=message):
            calibrate_from_comb(np.array([420.0, 470.0, 520.0]), **kwargs)

    def test_too_few_rungs_are_rejected(self) -> None:
        with pytest.raises(RetentionIndexError, match="at least two rungs"):
            calibrate_from_comb(np.array([420.0]), first_carbon_number=8)

    def test_ladder_comes_from_the_sample_itself(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Der Kamm, der alles andere erschwert, liefert die Kováts-Leiter gratis."""
        processed = preprocess(
            hdpe_sample.cube, PreprocessingConfig(asls_iterations=8)
        )
        comb = detect_homologue_comb(processed)
        calibration = calibrate_from_comb(comb.apex_times_s, first_carbon_number=8)
        assert calibration.n_anchors >= 8
        styrene_region = calibration.index_of(414.0)
        assert 700.0 < styrene_region < 1000.0


class TestSpectrumDenoising:
    def test_low_channels_are_removed(self) -> None:
        spectrum = np.array([100.0, 0.5, 20.0, 0.2])
        cleaned = denoise_spectrum(spectrum, relative_threshold=0.01)
        assert cleaned[0] == 100.0
        assert cleaned[2] == 20.0
        assert cleaned[1] == 0.0
        assert cleaned[3] == 0.0

    def test_empty_spectrum_is_handled(self) -> None:
        assert not denoise_spectrum(np.zeros(5)).any()


class TestCompoundIdentification:
    @pytest.fixture(scope="class")
    @staticmethod
    def outcome(mixed_sample: SyntheticPyrogram, library: MarkerLibrary):  # noqa: ANN205
        config = DeconvolutionConfig(
            preprocessing=PreprocessingConfig(asls_iterations=8)
        )
        report = resolve_pyrogram(mixed_sample.cube, config)
        comb = detect_homologue_comb(
            preprocess(mixed_sample.cube, config.preprocessing)
        )
        calibration = calibrate_from_comb(comb.apex_times_s, first_carbon_number=8)
        return identify_compounds(report.result, library, calibration=calibration)

    def test_finds_the_foreign_polymer_markers(self, outcome) -> None:  # noqa: ANN001
        names = {hit.compound_name for hit in outcome}
        assert "styrene" in names
        assert "benzoic acid" in names
        assert "epsilon-caprolactam" in names

    def test_scores_are_bounded_and_ordered(self, outcome) -> None:  # noqa: ANN001
        assert outcome
        scores = [hit.score for hit in outcome]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= score <= 1.0 for score in scores)

    def test_retention_index_is_recorded(self, outcome) -> None:  # noqa: ANN001
        for hit in outcome:
            assert hit.used_retention_index
            assert hit.retention_index > 0.0

    def test_benchmark_retention_scale_is_not_kovats_consistent(self) -> None:
        """Ein in MS3 gefundener Mangel des synthetischen Generators aus MS1.

        Die Marker-Retentionszeiten im Generator wurden gesetzt, um bestimmte
        Koelutionen zu erzwingen — Styrol auf dem C8-Cluster, Caprolactam auf C14.
        Sie liegen dadurch nicht auf einer Kováts-konsistenten Skala relativ zum
        Alkankamm: gemessen weichen sie im Median um 142 Indexeinheiten ab, bei
        den spät eluierenden Markern um bis zu 1023.

        Folge: die Retentionsindex-Achse der Identifikation ist direkt getestet
        (siehe TestRetentionIndex), lässt sich auf diesem Benchmark aber **nicht**
        end-to-end validieren und zieht dort korrekte Treffer sogar herunter. Für
        echte Messdaten, deren Retentionsskala per Definition konsistent ist, ist
        sie richtig — deshalb bleibt die Voreinstellung, wie sie ist.

        Beides zugleich ist im Generator nicht zu haben: die Marker so zu
        verschieben, dass ihre Indizes stimmen, würde genau die Koelutionen
        auflösen, für die der Benchmark gebaut wurde. Der Test hält den Konflikt
        fest, damit er nicht unbemerkt verschwindet.
        """
        from tests.reference_spectra import REFERENCE_COMPOUNDS as BENCHMARK
        from tests.synthetic_data import AlkaneRetentionModel

        model = AlkaneRetentionModel()
        carbons = np.arange(6, 35)
        times = np.array([model.retention_time_s(int(n)) for n in carbons])

        def comb_index(retention_time: float) -> float:
            position = int(np.clip(np.searchsorted(times, retention_time) - 1, 0, times.size - 2))
            span = times[position + 1] - times[position]
            fraction = (retention_time - times[position]) / span
            return float(100.0 * (carbons[position] + fraction))

        deviations = [
            abs(
                comb_index(BENCHMARK[name].retention_time_s)
                - REFERENCE_COMPOUNDS[name].retention_index
            )
            for name in sorted(set(BENCHMARK) & set(REFERENCE_COMPOUNDS))
        ]
        assert np.median(deviations) > 50.0, (
            "Wenn der Generator inzwischen RI-konsistent ist, kann die "
            "Retentionsachse end-to-end validiert werden — dieser Test und der "
            "Hinweis in identification/engine.py sind dann zu entfernen."
        )

    def test_identification_without_a_ladder_is_less_selective(
        self, mixed_sample: SyntheticPyrogram, library: MarkerLibrary
    ) -> None:
        """Belegt, warum die RI-Verankerung gebaut wurde.

        Ohne Retentionsevidenz akzeptiert die Zuordnung Verbindungen, die spektral
        ähnlich sind, aber Minuten entfernt eluieren.
        """
        report = resolve_pyrogram(
            mixed_sample.cube,
            DeconvolutionConfig(preprocessing=PreprocessingConfig(asls_iterations=8)),
        )
        comb = detect_homologue_comb(
            preprocess(mixed_sample.cube, PreprocessingConfig(asls_iterations=8))
        )
        calibration = calibrate_from_comb(comb.apex_times_s, first_carbon_number=8)

        with_ladder = identify_compounds(report.result, library, calibration=calibration)
        without_ladder = identify_compounds(report.result, library, calibration=None)

        assert len(without_ladder) >= len(with_ladder)
        assert all(hit.retention_agreement == 1.0 for hit in without_ladder)

    def test_strict_settings_reduce_the_number_of_hits(
        self, mixed_sample: SyntheticPyrogram, library: MarkerLibrary
    ) -> None:
        report = resolve_pyrogram(
            mixed_sample.cube,
            DeconvolutionConfig(preprocessing=PreprocessingConfig(asls_iterations=8)),
        )
        lenient = identify_compounds(report.result, library)
        strict = identify_compounds(
            report.result,
            library,
            settings=IdentificationSettings(
                min_spectral_similarity=0.9, min_combined_score=0.9
            ),
        )
        assert len(strict) <= len(lenient)

    def test_an_empty_resolution_identifies_nothing(
        self, library: MarkerLibrary
    ) -> None:
        from pyrecycle_analytics.deconvolution.result import ResolutionResult

        empty = ResolutionResult(
            retention_times=np.linspace(0.0, 10.0, 11),
            mz_axis=np.arange(50.0, 60.0),
            C=np.zeros((11, 1)),
            S=np.zeros((1, 10)),
        )
        assert identify_compounds(empty, library) == []


class TestPolymerIdentification:
    def test_pure_polystyrene_is_identified_with_high_confidence(
        self, ps_sample: SyntheticPyrogram, library: MarkerLibrary
    ) -> None:
        report = resolve_pyrogram(
            ps_sample.cube,
            DeconvolutionConfig(preprocessing=PreprocessingConfig(asls_iterations=6)),
        )
        findings = identify_polymers(
            identify_compounds(report.result, library), library
        )
        polystyrene = [f for f in findings if f.polymer is PolymerClass.PS]
        assert polystyrene, "PS muss in einer reinen PS-Probe gefunden werden"
        assert polystyrene[0].confidence > 0.7
        assert len(polystyrene[0].members_found) >= 3

    def test_foreign_polymers_are_found_in_a_pcr_blend(
        self, mixed_sample: SyntheticPyrogram, library: MarkerLibrary
    ) -> None:
        config = DeconvolutionConfig(
            preprocessing=PreprocessingConfig(asls_iterations=8)
        )
        report = resolve_pyrogram(mixed_sample.cube, config)
        comb = detect_homologue_comb(
            preprocess(mixed_sample.cube, config.preprocessing)
        )
        calibration = calibrate_from_comb(comb.apex_times_s, first_carbon_number=8)
        findings = identify_polymers(
            identify_compounds(report.result, library, calibration=calibration), library
        )
        found = {finding.polymer for finding in findings}
        assert PolymerClass.PS in found
        assert PolymerClass.PET in found

    def test_a_single_marker_finding_is_flagged_as_indicative(
        self, mixed_sample: SyntheticPyrogram, library: MarkerLibrary
    ) -> None:
        """Ein Polymer aus einem Peak ist ein Hinweis, kein Befund."""
        report = resolve_pyrogram(
            mixed_sample.cube,
            DeconvolutionConfig(preprocessing=PreprocessingConfig(asls_iterations=8)),
        )
        findings = identify_polymers(
            identify_compounds(report.result, library), library
        )
        for finding in findings:
            if len(finding.members_found) == 1:
                assert any("single marker" in note for note in finding.notes)

    def test_nothing_is_reported_without_identifications(
        self, library: MarkerLibrary
    ) -> None:
        assert identify_polymers([], library) == []

    def test_confidence_is_bounded(
        self, ps_sample: SyntheticPyrogram, library: MarkerLibrary
    ) -> None:
        report = resolve_pyrogram(
            ps_sample.cube,
            DeconvolutionConfig(preprocessing=PreprocessingConfig(asls_iterations=6)),
        )
        findings = identify_polymers(
            identify_compounds(report.result, library), library
        )
        assert all(0.0 <= finding.confidence <= 1.0 for finding in findings)

    def test_at_most_one_finding_per_polymer(
        self, ps_sample: SyntheticPyrogram, library: MarkerLibrary
    ) -> None:
        report = resolve_pyrogram(
            ps_sample.cube,
            DeconvolutionConfig(preprocessing=PreprocessingConfig(asls_iterations=6)),
        )
        findings = identify_polymers(
            identify_compounds(report.result, library), library
        )
        polymers = [finding.polymer for finding in findings]
        assert len(polymers) == len(set(polymers))
