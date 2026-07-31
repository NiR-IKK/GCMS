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
