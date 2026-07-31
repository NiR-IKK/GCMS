"""Tests für die Logik hinter der Streamlit-Oberfläche.

Die Oberfläche selbst ist dünner Klebstoff um getestete Bibliotheksfunktionen;
geprüft wird hier das, was eigene Logik enthält — vor allem das Ausdünnen langer
Chromatogramme für die Anzeige. Ein 60-Minuten-Lauf bei 20 Hz hat 72 000 Punkte,
und naives Slicing würde genau die schmalen Peaks verschlucken, wegen derer man
überhaupt hinschaut.
"""

from __future__ import annotations

import numpy as np
import pytest

streamlit_app = pytest.importorskip(
    "streamlit_app",
    reason="Streamlit-Oberfläche; installieren mit pip install 'pyrecycle-analytics[ui]'",
)


class TestThinning:
    def test_short_traces_are_returned_untouched(self) -> None:
        x = np.linspace(0.0, 10.0, 50)
        y = np.sin(x)
        thin_x, thin_y = streamlit_app._thin(x, y)
        assert np.array_equal(thin_x, x)
        assert np.array_equal(thin_y, y)

    def test_long_traces_are_reduced_below_the_limit(self) -> None:
        x = np.linspace(0.0, 3600.0, 72_000)
        y = np.random.default_rng(0).random(x.size)
        thin_x, thin_y = streamlit_app._thin(x, y)
        assert thin_x.size <= streamlit_app.MAX_PLOT_POINTS
        assert thin_x.size == thin_y.size

    def test_narrow_peaks_survive_thinning(self) -> None:
        """The property that makes this worth writing rather than slicing.

        A single-scan spike in a 72 000-point trace must still be visible after
        reduction; ``y[::block]`` would drop it with 99 % probability.
        """
        x = np.linspace(0.0, 3600.0, 72_000)
        y = np.zeros_like(x)
        y[12_345] = 5_000.0
        y[60_001] = 3_000.0

        thin_x, thin_y = streamlit_app._thin(x, y)
        assert thin_y.max() == pytest.approx(5_000.0)
        assert 3_000.0 in thin_y

    def test_thinned_positions_stay_paired_with_their_values(self) -> None:
        """A peak must keep its retention time, or the plot lies about where it eluted."""
        x = np.linspace(0.0, 1000.0, 50_000)
        y = np.zeros_like(x)
        y[31_415] = 999.0

        thin_x, thin_y = streamlit_app._thin(x, y)
        peak_index = int(np.argmax(thin_y))
        assert thin_x[peak_index] == pytest.approx(x[31_415])

    def test_retention_axis_stays_monotonic(self) -> None:
        x = np.linspace(60.0, 1800.0, 40_000)
        y = np.random.default_rng(1).random(x.size)
        thin_x, _ = streamlit_app._thin(x, y)
        assert np.all(np.diff(thin_x) > 0.0)


class TestIonPresets:
    def test_presets_are_non_empty_and_numeric(self) -> None:
        assert streamlit_app.ION_PRESETS
        for label, ions in streamlit_app.ION_PRESETS.items():
            assert ions, f"{label} has no ions"
            assert all(isinstance(mz, float) and mz > 0.0 for mz in ions)

    def test_presets_cover_the_polymers_the_platform_targets(self) -> None:
        labels = " ".join(streamlit_app.ION_PRESETS).upper()
        for polymer in ("PS", "PET", "PA6", "PVC", "PP"):
            assert polymer in labels

    def test_carbonyl_preset_uses_the_oxidation_diagnostic_ions(self) -> None:
        """m/z 44/58/60/73 are the aldehyde, ketone and acid McLafferty ions."""
        carbonyl = next(
            ions for label, ions in streamlit_app.ION_PRESETS.items() if "Oxidation" in label
        )
        assert set(carbonyl) == {44.0, 58.0, 60.0, 73.0}


class TestAnalysisTab:
    """Die Auswertung hinter dem Reiter *Auswertung*.

    ``_run_analysis`` ist die einzige Stelle der Oberfläche mit eigener Logik, die
    über Darstellung hinausgeht: sie führt die Kette aus und übersetzt das
    Ergebnis in Tabellen. Geprüft wird, dass diese Übersetzung nichts verliert und
    nichts beschönigt — die Einschränkungen des Passes müssen auch in der
    Oberfläche ankommen, sonst nützen sie im Dokument wenig.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def outcome(mixed_sample):  # noqa: ANN001, ANN205
        # .__wrapped__ umgeht Streamlits Cache: ohne laufende Session hätte er
        # keinen Kontext, und für einen Test ist das Zwischenspeichern ohnehin
        # nur Störung.
        return streamlit_app._run_analysis.__wrapped__(mixed_sample.cube, 8)

    def test_analysis_succeeds(self, outcome) -> None:  # noqa: ANN001
        assert not isinstance(outcome, str), outcome

    def test_polymer_table_reports_the_unassigned_share(self, outcome) -> None:  # noqa: ANN001
        """Nicht zugeordnetes Signal muss sichtbar sein, nicht verteilt."""
        _, tables, _ = outcome
        fractions = tables["Polymerfraktionen"]
        assert "nicht zugeordnet" in set(fractions["Polymer"])
        assert fractions["Anteil [%]"].sum() == pytest.approx(100.0, abs=0.5)

    def test_calibration_status_and_disclaimer_are_carried_through(
        self, outcome
    ) -> None:  # noqa: ANN001
        summary, _, _ = outcome
        assert summary["calibration_status"] != "calibrated"
        assert summary["disclaimer"]

    def test_identification_table_shows_both_retention_indices(
        self, outcome
    ) -> None:  # noqa: ANN001
        """Ohne den Bibliothekswert daneben ist der Probenwert nicht bewertbar."""
        _, tables, _ = outcome
        columns = set(tables["Identifikationen"].columns)
        assert {"RI (Probe)", "RI (Bibliothek)", "Spektrum", "Score"} <= columns

    def test_regulatory_table_separates_detection_from_compliance(
        self, outcome
    ) -> None:  # noqa: ANN001
        _, tables, _ = outcome
        regulatory = tables["Regulatorik"]
        detected = regulatory[regulatory["Befund"] == "detektiert"]
        assert set(regulatory["Bewertung"]) <= {
            "nicht bewertbar", "—", "ÜBER GRENZWERT", "eingehalten"
        }
        for verdict in detected["Bewertung"]:
            assert verdict != "eingehalten", (
                "ein Nachweis ohne Kalibrierstandard darf nicht als eingehalten "
                "erscheinen"
            )

    def test_html_and_json_are_offered_for_download(self, outcome) -> None:  # noqa: ANN001
        summary, _, html = outcome
        assert html.startswith("<!DOCTYPE html>")
        assert summary["sample_id"] in html
        assert '"schema_version"' in summary["json"]

    def test_a_sample_without_a_comb_fails_softly(self, tiny_cube) -> None:  # noqa: ANN001
        """Ein Fehler darf die Oberfläche nicht abbrechen, sondern wird Text."""
        result = streamlit_app._run_analysis.__wrapped__(tiny_cube, None)
        assert isinstance(result, str)
        assert "fehlgeschlagen" in result


class TestPageRenders:
    """Die Seite einmal wirklich rendern, nicht nur ihre Hilfsfunktionen aufrufen.

    Ein Aufruf der Tab-Funktionen belegt nicht, dass die Seite zusammen läuft:
    Reiterreihenfolge, Seitenleisten-Verdrahtung und der Zustand zwischen zwei
    Läufen entstehen erst im Skriptdurchlauf.
    """

    @staticmethod
    def _app():  # noqa: ANN205
        apptest = pytest.importorskip("streamlit.testing.v1")
        app = apptest.AppTest.from_file("streamlit_app.py", default_timeout=300)
        app.run()
        return app

    def test_page_renders_without_an_exception(self) -> None:
        app = self._app()
        assert not app.exception, [str(e) for e in app.exception]

    def test_every_tab_is_present(self) -> None:
        app = self._app()
        app.sidebar.radio[0].set_value("Demo (simuliert)").run()
        assert [tab.label for tab in app.tabs] == [
            "Übersicht",
            "Ionenspuren",
            "Massenspektrum",
            "Vorverarbeitung",
            "Auswertung",
            "Export",
        ]

    def test_demo_source_renders_without_errors(self) -> None:
        app = self._app()
        app.sidebar.radio[0].set_value("Demo (simuliert)").run()
        assert not app.exception, [str(e) for e in app.exception]
        assert not app.error, [e.value for e in app.error]
        rendered = {tuple(frame.value.columns) for frame in app.dataframe}
        assert ("Kennwert", "Wert") in rendered, "Übersichtstabelle fehlt"


class TestArrowSerialisation:
    """Jede angezeigte Tabelle muss sich nach Arrow konvertieren lassen.

    Streamlit reicht jeden DataFrame an Arrow weiter. Eine Spalte mit gemischten
    Typen wird dort abgelehnt; Streamlit fängt das ab und wandelt still nach Text
    um, sodass die Seite weiterläuft und nur eine Meldung ins Log schreibt. Genau
    deshalb fällt es beim Rendern **nicht** auf — die Übersichtstabelle mischte
    Zahlen und Text, ohne dass ein Test oder die Oberfläche es gezeigt hätte.
    Geprüft wird darum die Konvertierung selbst, nicht das Rendern.
    """

    @staticmethod
    def _assert_convertible(frame, label: str) -> None:  # noqa: ANN001
        pa = pytest.importorskip("pyarrow")
        try:
            pa.Table.from_pandas(frame)
        except pa.ArrowTypeError as error:  # pragma: no cover - nur im Fehlerfall
            pytest.fail(f"{label} ist nicht Arrow-konvertierbar: {error}")

    def test_overview_table_converts(self, mixed_sample) -> None:  # noqa: ANN001
        frame = streamlit_app._metadata_table(mixed_sample.cube)
        self._assert_convertible(frame, "Übersichtstabelle")

    def test_analysis_tables_convert(self, mixed_sample) -> None:  # noqa: ANN001
        outcome = streamlit_app._run_analysis.__wrapped__(mixed_sample.cube, 8)
        assert not isinstance(outcome, str), outcome
        _, tables, _ = outcome
        for label, frame in tables.items():
            self._assert_convertible(frame, label)
