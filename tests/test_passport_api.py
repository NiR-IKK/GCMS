"""Tests für Rezyklat-Pass, Rendering und API.

Der Pass ist das einzige Dokument, das das Labor verlässt. Die meisten Tests hier
prüfen deshalb nicht, ob eine Zahl stimmt, sondern ob sie **richtig
gekennzeichnet** ist: dass eine unkalibrierte Prozentangabe nicht wie eine
gemessene aussieht, und dass ein detektierter Schadstoff nicht wie eine
Konformitätsaussage gelesen werden kann.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from data_schemas.enums import PolymerClass, RecyclateStream
from data_schemas.passport import (
    CalibrationStatus,
    ConfidenceLevel,
    PolymerFraction,
    RecyclatePassport,
    RegulatoryFinding,
)
from pyrecycle_analytics.api import create_app
from pyrecycle_analytics.deconvolution import DeconvolutionConfig
from pyrecycle_analytics.ingestion import write_andi_cdf
from pyrecycle_analytics.library import MarkerLibrary
from pyrecycle_analytics.preprocessing import PreprocessingConfig
from pyrecycle_analytics.reporting import analyse_pyrogram, render_html, render_pdf
from pyrecycle_analytics.reporting.passport import MatrixPolyolefinEvidence
from tests.synthetic_data import RECIPES, SyntheticPyrogram, SyntheticPyrogramGenerator


@pytest.fixture(scope="module")
def library() -> MarkerLibrary:
    return MarkerLibrary.in_memory()


@pytest.fixture(scope="module")
def small_sample() -> SyntheticPyrogram:
    """A reduced pyrogram, so the API tests do not run for minutes."""
    generator = SyntheticPyrogramGenerator(
        seed=7, rt_end_s=900.0, scan_rate_hz=2.0,
        mz_low=29, mz_high=250, carbon_range=(6, 14),
    )
    return generator.generate(RECIPES["pcr_mixed_polyolefin"])


@pytest.fixture(scope="module")
def analysis(small_sample: SyntheticPyrogram, library: MarkerLibrary):  # noqa: ANN201
    return analyse_pyrogram(
        small_sample.cube,
        library,
        config=DeconvolutionConfig(
            preprocessing=PreprocessingConfig(asls_iterations=6)
        ),
        first_carbon_number=8,
    )


class TestCalibrationHonesty:
    def test_uncalibrated_analysis_is_not_labelled_calibrated(self, analysis) -> None:  # noqa: ANN001
        """Der wichtigste Test des Passes.

        Eine Prozentangabe ohne gravimetrische Referenzmischungen ist
        semiquantitativ. Sie als Messwert auszugeben wäre in einem Dokument mit
        „Pass" im Namen ein echtes Problem.
        """
        passport = analysis.passport
        assert passport.calibration_status is CalibrationStatus.RESPONSE_CORRECTED
        assert not passport.calibration_status.is_quantitative
        assert "not calibrated" in passport.calibration_status.disclaimer

    def test_uncalibrated_fractions_carry_no_uncertainty_interval(
        self, analysis
    ) -> None:  # noqa: ANN001
        """Ein Intervall ohne Kalibrierung hätte keine Grundlage."""
        for fraction in analysis.passport.polymer_fractions:
            assert fraction.uncertainty_percent is None

    def test_an_uncalibrated_share_may_not_claim_an_interval(self) -> None:
        with pytest.raises(ValueError, match="cannot carry an uncertainty"):
            PolymerFraction(
                polymer=PolymerClass.PP,
                share_percent=10.0,
                calibration_status=CalibrationStatus.UNCALIBRATED,
                confidence=ConfidenceLevel.MEDIUM,
                marker_pattern="x",
                markers_expected=2,
                uncertainty_percent=1.0,
            )

    def test_passport_status_is_the_weakest_of_its_fractions(self) -> None:
        """Ein Pass ist nur so quantitativ wie seine schwächste Angabe."""
        passport = RecyclatePassport(
            sample=_sample(),
            acquisition=_acquisition(),
            polymer_fractions=(
                _fraction(PolymerClass.PP, 40.0, CalibrationStatus.CALIBRATED),
                _fraction(PolymerClass.PS, 5.0, CalibrationStatus.RESPONSE_CORRECTED),
            ),
        )
        assert passport.calibration_status is CalibrationStatus.RESPONSE_CORRECTED

    def test_calibrated_run_is_labelled_and_carries_intervals(
        self, small_sample: SyntheticPyrogram, library: MarkerLibrary
    ) -> None:
        result = analyse_pyrogram(
            small_sample.cube,
            library,
            config=DeconvolutionConfig(
                preprocessing=PreprocessingConfig(asls_iterations=6)
            ),
            calibrated=True,
        )
        assert result.passport.calibration_status is CalibrationStatus.CALIBRATED
        assert all(
            fraction.uncertainty_percent is not None
            for fraction in result.passport.polymer_fractions
        )


class TestUnassignedSignal:
    def test_unattributed_signal_is_reported_not_redistributed(
        self, analysis
    ) -> None:  # noqa: ANN001
        """Wegnormieren würde jede ausgewiesene Zahl aufblähen."""
        passport = analysis.passport
        total = sum(f.share_percent for f in passport.polymer_fractions)
        assert passport.unassigned_share_percent > 0.0
        assert total + passport.unassigned_share_percent == pytest.approx(100.0, abs=0.5)

    def test_shares_may_not_exceed_one_hundred(self) -> None:
        with pytest.raises(ValueError, match="exceeds 100"):
            RecyclatePassport(
                sample=_sample(),
                acquisition=_acquisition(),
                polymer_fractions=(
                    _fraction(PolymerClass.PP, 80.0),
                    _fraction(PolymerClass.PS, 40.0),
                ),
            )


class TestPolyolefinFromTheMatrix:
    def test_the_dominant_polyolefin_is_reported(self, analysis) -> None:  # noqa: ANN001
        """Ohne diese Evidenz fehlte der Hauptbestandteil im Pass.

        Die Polyolefin-Matrix wird vor der Kurvenauflösung abgezogen, also hat sie
        zum Identifikationszeitpunkt keine Marker mehr. Die abgezogene Menge ist
        die Evidenz und wird als solche weitergereicht.
        """
        polyolefins = [
            fraction
            for fraction in analysis.passport.polymer_fractions
            if fraction.polymer.is_polyolefin
            and fraction.marker_pattern == "polyolefin homologous series"
        ]
        assert polyolefins, "die Polyolefin-Fraktion fehlt im Pass"
        assert polyolefins[0].share_percent > 30.0

    @pytest.mark.parametrize(
        ("branching", "expected"),
        [
            (1.24, PolymerClass.PP),
            (0.19, PolymerClass.PE_LD),
            (0.14, PolymerClass.PE_HD),
        ],
    )
    def test_grade_follows_the_branching_index(
        self, branching: float, expected: PolymerClass
    ) -> None:
        """Gemessen: HDPE 0,137, LDPE 0,186, PP 1,239."""
        evidence = MatrixPolyolefinEvidence(
            signal_fraction=0.6, branching_index=branching, n_clusters=10
        )
        assert evidence.polymer is expected

    def test_polyethylene_grade_is_flagged_as_a_hint(self) -> None:
        """Die LD/HD-Grenze ist eng; das darf nicht wie eine feste Zuordnung wirken."""
        evidence = MatrixPolyolefinEvidence(
            signal_fraction=0.6, branching_index=0.18, n_clusters=10
        )
        assert "hint" in evidence.grade_note

    def test_correct_grade_on_pure_references(self, library: MarkerLibrary) -> None:
        generator = SyntheticPyrogramGenerator(
            seed=7, rt_end_s=900.0, scan_rate_hz=2.0,
            mz_low=29, mz_high=250, carbon_range=(6, 14),
        )
        config = DeconvolutionConfig(
            preprocessing=PreprocessingConfig(asls_iterations=6)
        )
        for recipe, expected in (
            ("virgin_hdpe", PolymerClass.PE_HD),
            ("virgin_pp", PolymerClass.PP),
        ):
            passport = analyse_pyrogram(
                generator.generate(RECIPES[recipe]).cube, library, config=config
            ).passport
            comb = [
                fraction
                for fraction in passport.polymer_fractions
                if fraction.marker_pattern == "polyolefin homologous series"
            ]
            assert comb and comb[0].polymer is expected


class TestRegulatoryHonesty:
    def test_detection_does_not_imply_a_compliance_statement(self, analysis) -> None:  # noqa: ANN001
        """REACH-Grenzwerte sind Massenanteile; ohne Standards nicht prüfbar."""
        passport = analysis.passport
        for finding in passport.regulatory_findings:
            if finding.detected:
                assert not finding.compliance_assessable
                assert finding.exceeds_limit is None

    def test_unassessable_findings_are_surfaced_at_the_top_level(
        self, analysis
    ) -> None:  # noqa: ANN001
        passport = analysis.passport
        if passport.detected_substances:
            assert passport.unassessable_findings

    def test_a_detection_must_carry_a_confidence(self) -> None:
        with pytest.raises(ValueError, match="must carry a confidence"):
            RegulatoryFinding(
                substance="DEHP", regulation="REACH", detected=True
            )

    def test_quantified_finding_can_be_assessed(self) -> None:
        finding = RegulatoryFinding(
            substance="DEHP",
            regulation="REACH Annex XVII",
            detected=True,
            detection_confidence=ConfidenceLevel.HIGH,
            limit_percent=0.1,
            quantified_percent=0.25,
        )
        assert finding.compliance_assessable
        assert finding.exceeds_limit is True

    def test_out_of_scope_substances_are_declared(self, analysis) -> None:  # noqa: ANN001
        """Schweigen zu Schwermetallen darf nicht als Unbedenklichkeit gelten."""
        warnings = " ".join(analysis.passport.provenance.warnings)
        assert "do not elute" in warnings
        assert "XRF" in warnings


class TestDegradationReporting:
    def test_without_a_reference_the_indices_are_marked_uninterpretable(
        self, analysis
    ) -> None:  # noqa: ANN001
        degradation = analysis.passport.degradation
        assert degradation is not None
        assert not degradation.is_interpretable
        assert any("no virgin reference" in note for note in degradation.notes)

    def test_with_a_reference_the_comparison_appears(
        self, small_sample: SyntheticPyrogram, library: MarkerLibrary
    ) -> None:
        config = DeconvolutionConfig(
            preprocessing=PreprocessingConfig(asls_iterations=6)
        )
        generator = SyntheticPyrogramGenerator(
            seed=7, rt_end_s=900.0, scan_rate_hz=2.0,
            mz_low=29, mz_high=250, carbon_range=(6, 14),
        )
        virgin = analyse_pyrogram(
            generator.generate(RECIPES["virgin_hdpe"]).cube, library, config=config
        )
        aged = analyse_pyrogram(
            small_sample.cube,
            library,
            config=config,
            degradation_reference=virgin.degradation,
            reference_sample_id="virgin_hdpe",
        )
        degradation = aged.passport.degradation
        assert degradation is not None
        assert degradation.is_interpretable
        assert degradation.reference_sample_id == "virgin_hdpe"


class TestRendering:
    def test_html_contains_the_calibration_caveat_before_the_numbers(
        self, analysis
    ) -> None:  # noqa: ANN001
        html = render_html(analysis.passport)
        assert html.index("Calibration status") < html.index("Polymer composition")
        assert "Semi-quantitative" in html

    def test_html_declares_unassessable_findings(self, analysis) -> None:  # noqa: ANN001
        html = render_html(analysis.passport)
        if analysis.passport.unassessable_findings:
            assert "compliance not" in html
            assert "No statement about compliance is made" in html

    def test_html_is_deterministic(self, analysis) -> None:  # noqa: ANN001
        """Zwei Läufe derselben Probe müssen diffbar sein."""
        assert render_html(analysis.passport) == render_html(analysis.passport)

    def test_html_escapes_sample_identifiers(self) -> None:
        passport = RecyclatePassport(
            sample=_sample(sample_id="<script>alert(1)</script>"),
            acquisition=_acquisition(),
        )
        html = render_html(passport)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_pdf_is_written(self, analysis, tmp_path: Path) -> None:  # noqa: ANN001
        path = render_pdf(analysis.passport, tmp_path / "passport")
        assert path.suffix == ".pdf"
        assert path.read_bytes().startswith(b"%PDF-")
        assert path.stat().st_size > 5_000


class TestApi:
    @pytest.fixture(scope="class")
    @staticmethod
    def client(library: MarkerLibrary) -> TestClient:
        return TestClient(create_app(library))

    @pytest.fixture(scope="class")
    @staticmethod
    def uploaded(client: TestClient, small_sample: SyntheticPyrogram, tmp_path_factory) -> str:  # noqa: ANN001
        directory = tmp_path_factory.mktemp("api")
        path = write_andi_cdf(small_sample.cube, directory / "Probe.CDF")
        response = client.post(
            "/samples",
            files={"file": ("Probe.CDF", path.read_bytes(), "application/octet-stream")},
        )
        assert response.status_code == 201
        return response.json()["sample_id"]

    def test_health_reports_the_library_size(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["library_compounds"] > 20

    def test_library_is_queryable_across_threads(self, client: TestClient) -> None:
        """Regressionstest für einen nur unter dem Server sichtbaren Fehler.

        SQLite mit ``:memory:`` gibt per Vorgabe jedem Thread eine eigene, leere
        Datenbank. Direkt aufgerufen funktionierte die Bibliothek, im
        FastAPI-Threadpool meldete jede Abfrage „no such table".
        """
        for _ in range(3):
            assert client.get("/health").json()["library_compounds"] > 20
        assert client.get("/library/polymers").status_code == 200

    def test_upload_returns_cube_dimensions(
        self, client: TestClient, uploaded: str
    ) -> None:
        body = client.get(f"/samples/{uploaded}").json()
        assert body["n_scans"] > 100
        assert body["n_mz"] > 50

    def test_analysis_produces_a_passport(
        self, client: TestClient, uploaded: str
    ) -> None:
        assert client.post(
            f"/samples/{uploaded}/analyse", json={"first_carbon_number": 8}
        ).status_code == 202
        assert client.get(f"/samples/{uploaded}").json()["status"] == "complete"

        response = client.get(f"/samples/{uploaded}/passport")
        assert response.status_code == 200
        passport = response.json()
        assert passport["calibration_status"] == "response-corrected"
        assert passport["polymer_fractions"]
        json.dumps(passport)

    def test_passport_renders_as_html_and_pdf(
        self, client: TestClient, uploaded: str
    ) -> None:
        html = client.get(f"/samples/{uploaded}/passport.html")
        assert html.status_code == 200
        assert "Digital recyclate passport" in html.text

        pdf = client.get(f"/samples/{uploaded}/passport.pdf")
        assert pdf.status_code == 200
        assert pdf.content.startswith(b"%PDF-")

    def test_unknown_sample_is_a_404(self, client: TestClient) -> None:
        assert client.get("/samples/deadbeef").status_code == 404
        assert client.get("/samples/deadbeef/passport").status_code == 404

    def test_passport_before_analysis_is_a_409(
        self, client: TestClient, small_sample: SyntheticPyrogram, tmp_path: Path
    ) -> None:
        path = write_andi_cdf(small_sample.cube, tmp_path / "second.CDF")
        sample_id = client.post(
            "/samples",
            files={"file": ("second.CDF", path.read_bytes(), "application/octet-stream")},
        ).json()["sample_id"]
        assert client.get(f"/samples/{sample_id}/passport").status_code == 409

    def test_unreadable_upload_is_a_422(self, client: TestClient) -> None:
        response = client.post(
            "/samples",
            files={"file": ("notes.csv", b"retention,area\n1,2\n", "text/csv")},
        )
        assert response.status_code == 422

    def test_regulated_substance_list_declares_what_is_out_of_scope(
        self, client: TestClient
    ) -> None:
        body = client.get("/library/regulated-substances").json()
        assert body["substances"]
        assert "do not elute" in body["out_of_scope"]


# --------------------------------------------------------------------- helpers


def _sample(sample_id: str = "unit-sample"):  # noqa: ANN202
    from data_schemas.acquisition import SampleMetadata

    return SampleMetadata(sample_id=sample_id, stream=RecyclateStream.PCR_MIXED_POLYOLEFIN)


def _acquisition():  # noqa: ANN202
    from data_schemas.acquisition import AcquisitionConditions

    return AcquisitionConditions()


def _fraction(
    polymer: PolymerClass,
    share: float,
    status: CalibrationStatus = CalibrationStatus.RESPONSE_CORRECTED,
) -> PolymerFraction:
    return PolymerFraction(
        polymer=polymer,
        share_percent=share,
        calibration_status=status,
        confidence=ConfidenceLevel.MEDIUM,
        marker_pattern="unit",
        markers_expected=2,
    )
