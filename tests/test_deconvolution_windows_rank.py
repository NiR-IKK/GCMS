"""Tests für Fensterbildung und Rangschätzung.

Beide Module entscheiden, welches Problem der Löser überhaupt gestellt bekommt.
Ein zu weites Fenster macht die Auflösung unmöglich, ein zu hoch geschätzter Rang
lässt den Löser Struktur aus Rauschen erfinden.

Ein Teil der Tests hält bewusst **Grenzen** fest statt Erfolge: dass Malinowskis
Indikatorfunktion auf diesen Matrixformen versagt, und dass die Rangschätzung auf
der unsubtrahierten Polyolefin-Matrix nicht zuverlässig ist. Beides ist gemessen,
und beides ist der Grund, warum die Matrix-Subtraktion vor der Kurvenauflösung
steht.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_schemas.acquisition import SampleMetadata
from data_schemas.enums import SourceFormat
from data_schemas.pyrogram import PyrogramMetadata
from pyrecycle_analytics.core.binning import axis_to_spec
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.core.peakshapes import gaussian
from pyrecycle_analytics.deconvolution.rank import (
    efa_rank,
    estimate_rank,
    evolving_factor_analysis,
    malinowski_indicator_rank,
    parallel_analysis_rank,
    variance_stabilise,
)
from pyrecycle_analytics.deconvolution.windows import detect_windows
from pyrecycle_analytics.preprocessing import PreprocessingConfig, preprocess
from tests.synthetic_data import SyntheticPyrogram


def _make_cube(
    retention_times: np.ndarray, mz_axis: np.ndarray, intensities: np.ndarray
) -> PyrogramDataCube:
    return PyrogramDataCube(
        retention_times=retention_times,
        mz_axis=mz_axis,
        intensities=intensities,
        metadata=PyrogramMetadata(
            sample=SampleMetadata(sample_id="unit"),
            source_format=SourceFormat.SYNTHETIC,
            n_scans=retention_times.size,
            mz_axis=axis_to_spec(mz_axis),
            rt_start_s=float(retention_times[0]),
            rt_end_s=float(retention_times[-1]),
        ),
    )


def _synthetic_window(
    n_components: int,
    *,
    n_scans: int = 120,
    n_mz: int = 60,
    noise: float = 0.0,
    separation_s: float = 8.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ein Fenster mit exakt bekannter Komponentenzahl und getrennten Spektren."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 120.0, n_scans)
    centers = 30.0 + separation_s * np.arange(n_components)
    C = np.column_stack([gaussian(t, center, 3.0, 1e5) for center in centers])
    S = np.zeros((n_components, n_mz))
    for row in range(n_components):
        channels = rng.choice(n_mz, size=6, replace=False)
        S[row, channels] = rng.uniform(0.5, 1.0, size=6)
        S[row] /= S[row].sum()
    D = C @ S
    if noise > 0.0:
        D = D + rng.normal(0.0, noise * np.sqrt(np.clip(D, 1.0, None)), D.shape)
        D = np.clip(D, 0.0, None)
    return t, D, C


class TestWindowDetection:
    def test_separated_peaks_give_separate_windows(self) -> None:
        t = np.linspace(0.0, 600.0, 1201)
        mz = np.arange(50.0, 55.0)
        tic = gaussian(t, 150.0, 3.0, 1e6) + gaussian(t, 450.0, 3.0, 1e6)
        D = np.outer(tic, np.array([0.4, 0.3, 0.1, 0.1, 0.1]))

        windows = detect_windows(_make_cube(t, mz, D))
        assert len(windows) == 2
        assert windows[0].end_s < windows[1].start_s

    def test_fused_peaks_are_merged_into_one_window(self) -> None:
        """Zwei Peaks mit R < 1 gehören in dasselbe Auflösungsproblem.

        Der Abstand ist mit 2,7 σ so gewählt, dass zwei Maxima überhaupt noch
        sichtbar sind — unterhalb von 2 σ verschmelzen zwei gleich große Gaußpeaks
        zu einem einzigen Buckel und kein Peakfinder der Welt sieht dort zwei.
        """
        t = np.linspace(0.0, 600.0, 1201)
        mz = np.arange(50.0, 55.0)
        tic = gaussian(t, 300.0, 3.0, 1e6) + gaussian(t, 308.0, 3.0, 1e6)
        D = np.outer(tic, np.array([0.4, 0.3, 0.1, 0.1, 0.1]))

        windows = detect_windows(_make_cube(t, mz, D))
        assert len(windows) == 1
        assert windows[0].n_peaks == 2

    def test_completely_fused_peaks_look_like_one_peak(self) -> None:
        """Unterhalb von 2 σ ist die Information im TIC schlicht nicht mehr da.

        Das ist keine Schwäche des Fensterfinders, sondern der Grund, warum es
        Kurvenauflösung gibt: die Trennung muss aus der m/z-Dimension kommen.
        """
        t = np.linspace(0.0, 600.0, 1201)
        mz = np.arange(50.0, 55.0)
        tic = gaussian(t, 300.0, 3.0, 1e6) + gaussian(t, 304.0, 3.0, 1e6)
        D = np.outer(tic, np.array([0.4, 0.3, 0.1, 0.1, 0.1]))

        windows = detect_windows(_make_cube(t, mz, D))
        assert len(windows) == 1
        assert windows[0].n_peaks == 1

    def test_windows_are_ordered_and_disjoint(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        processed = preprocess(
            mixed_sample.cube, PreprocessingConfig(asls_iterations=6)
        )
        windows = detect_windows(processed)
        assert windows
        for previous, current in zip(windows, windows[1:], strict=False):
            assert previous.stop_index <= current.start_index

    def test_windows_cover_most_of_the_true_components(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        processed = preprocess(
            mixed_sample.cube, PreprocessingConfig(asls_iterations=6)
        )
        windows = detect_windows(processed)
        covered = sum(
            1
            for component in mixed_sample.truth.components
            if any(
                window.start_s <= component.retention_time_s <= window.end_s
                for window in windows
            )
        )
        assert covered / mixed_sample.truth.n_components > 0.85

    def test_fused_comb_does_not_produce_a_runaway_window(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """Regressionstest für einen gemessenen Fehler.

        ``peak_widths`` misst die Breite, bei der das Signal auf die halbe
        Apexhöhe fällt. Innerhalb eines verschmolzenen Homologen-Clusters tut es
        das nie, also läuft die Routine weiter bis ans Clusterende und meldet eine
        um ein Vielfaches zu große Breite. Ungebremst überbrückte die daraus
        berechnete Fensterpolsterung 44-Sekunden-Lücken und verschmolz eine ganze
        Kammregion zu einem 240-Sekunden-Fenster mit 42 Komponenten — mehr, als
        jedes Auflösungsverfahren bewältigen kann.
        """
        processed = preprocess(
            mixed_sample.cube, PreprocessingConfig(asls_iterations=6)
        )
        windows = detect_windows(processed)
        worst = max(
            sum(
                1
                for component in mixed_sample.truth.components
                if window.start_s <= component.retention_time_s <= window.end_s
            )
            for window in windows
        )
        assert worst <= 15, f"ein Fenster enthält {worst} Komponenten"

    def test_max_window_scans_splits_an_oversized_window(self) -> None:
        t = np.linspace(0.0, 600.0, 1201)
        mz = np.arange(50.0, 55.0)
        tic = sum(gaussian(t, center, 3.0, 1e6) for center in (280.0, 292.0, 304.0))
        D = np.outer(tic, np.array([0.4, 0.3, 0.1, 0.1, 0.1]))
        cube = _make_cube(t, mz, D)

        wide = detect_windows(cube, max_window_scans=10_000)
        narrow = detect_windows(cube, max_window_scans=40)
        assert len(narrow) > len(wide)
        assert all(window.n_scans <= 60 for window in narrow)

    def test_window_extracts_a_matching_sub_cube(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        processed = preprocess(
            mixed_sample.cube, PreprocessingConfig(asls_iterations=6)
        )
        window = detect_windows(processed)[0]
        sub = window.extract(processed)
        assert sub.n_scans == window.n_scans
        assert sub.retention_times[0] == pytest.approx(window.start_s)

    def test_empty_chromatogram_gives_no_windows(self, tiny_cube) -> None:  # noqa: ANN001
        assert detect_windows(tiny_cube.with_intensities(np.zeros(tiny_cube.shape))) == ()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"min_prominence_fraction": 0.0}, "min_prominence_fraction must be > 0"),
            ({"merge_resolution": 0.0}, "merge_resolution must be > 0"),
        ],
    )
    def test_invalid_parameters_are_rejected(self, kwargs: dict, message: str) -> None:  # noqa: ANN001
        t = np.linspace(0.0, 100.0, 201)
        mz = np.arange(50.0, 52.0)
        cube = _make_cube(t, mz, np.ones((201, 2)))
        with pytest.raises(ValueError, match=message):
            detect_windows(cube, **kwargs)


class TestVarianceStabilisation:
    def test_anscombe_flattens_poisson_variance(self) -> None:
        """Der Grund für den Transform: Rauschen wird signalunabhängig.

        Auf Rohdaten skaliert die Streuung mit der Wurzel des Signals, sodass
        helle Kanäle jede rauschbasierte Schwelle sprengen. Nach der Transformation
        ist sie über Größenordnungen hinweg konstant.
        """
        rng = np.random.default_rng(0)
        raw_spread, stabilised_spread = [], []
        for level in (10.0, 1_000.0, 100_000.0):
            sample = rng.poisson(level, 20_000).astype(float)
            raw_spread.append(float(np.std(sample)))
            stabilised_spread.append(float(np.std(variance_stabilise(sample))))

        assert max(raw_spread) / min(raw_spread) > 50.0
        assert max(stabilised_spread) / min(stabilised_spread) < 1.2

    def test_negative_values_are_clipped(self) -> None:
        assert np.all(np.isfinite(variance_stabilise(np.array([-5.0, 0.0, 5.0]))))


class TestRankEstimators:
    @pytest.mark.parametrize("n_components", [2, 3, 5])
    def test_parallel_analysis_recovers_a_clean_low_rank_window(
        self, n_components: int
    ) -> None:
        _, D, _ = _synthetic_window(n_components, noise=0.0, seed=n_components)
        assert parallel_analysis_rank(D, max_rank=10) == n_components

    @pytest.mark.parametrize("n_components", [2, 3, 5])
    def test_parallel_analysis_survives_realistic_noise(
        self, n_components: int
    ) -> None:
        _, D, _ = _synthetic_window(n_components, noise=1.0, seed=n_components)
        estimate = parallel_analysis_rank(variance_stabilise(D), max_rank=10)
        assert abs(estimate - n_components) <= 1

    def test_efa_curves_have_the_expected_shape(self) -> None:
        """Vorwärts wachsend, rückwärts fallend — daher der Name."""
        _, D, _ = _synthetic_window(3, noise=0.0, seed=1)
        forward, backward = evolving_factor_analysis(D, max_rank=5)
        assert forward.shape[1] == 5
        assert forward[-1, 0] >= forward[0, 0]
        assert backward[0, 0] >= backward[-1, 0]

    @pytest.mark.parametrize("n_components", [2, 3, 4])
    def test_efa_recovers_a_clean_window(self, n_components: int) -> None:
        _, D, _ = _synthetic_window(n_components, noise=0.0, seed=n_components)
        assert abs(efa_rank(D, max_rank=10) - n_components) <= 1

    def test_efa_is_the_less_reliable_estimator_under_noise(self) -> None:
        """Begründet, warum EFA nur bestätigt und nicht entscheidet.

        Über Rangzahlen 2-6 und vier Rauschniveaus lag die Parallelanalyse in 18
        von 20 Fällen exakt richtig und nie um mehr als eins daneben; EFA lief
        mehrfach in seine Obergrenze. Deshalb entscheidet die Parallelanalyse.
        """
        parallel_errors, efa_errors = 0, 0
        for noise in (0.0, 1.0, 2.0, 4.0):
            for n_components in (2, 3, 4, 5, 6):
                _, D, _ = _synthetic_window(
                    n_components, noise=noise, seed=n_components * 10 + int(noise)
                )
                parallel_errors += abs(
                    parallel_analysis_rank(D, max_rank=12) - n_components
                )
                efa_errors += abs(efa_rank(D, max_rank=12) - n_components)
        assert parallel_errors < efa_errors
        assert parallel_errors <= 2

    def test_efa_rejects_degenerate_input(self) -> None:
        with pytest.raises(ValueError, match="at least 2 scans"):
            evolving_factor_analysis(np.ones((1, 5)))
        with pytest.raises(ValueError, match="expected a 2-D matrix"):
            evolving_factor_analysis(np.ones(5))

    def test_malinowski_indicator_is_unusable_on_short_wide_windows(self) -> None:
        """Dokumentiert eine gemessene Grenze, kein Implementierungsfehler.

        ``IND(k) = RE(k)/(p-k)²`` braucht den ``(p-k)²``-Term, um die Kurve
        umzubiegen. Bei einem 120×60-Fenster ändert er sich über den plausiblen
        Rangbereich kaum, während ``RE(k)`` stetig fällt — die Kurve fällt monoton
        und ihr Minimum liegt immer am größten geprüften ``k``. Deshalb ist der
        Schätzer standardmäßig nicht im Konsens.
        """
        _, D, _ = _synthetic_window(3, noise=0.5, seed=2)
        singular_values = np.linalg.svd(D, compute_uv=False)
        estimate = malinowski_indicator_rank(singular_values, *D.shape)
        assert estimate > 10, "IND liefert hier erwartungsgemäß keinen kleinen Rang"

    def test_malinowski_is_excluded_from_the_consensus_by_default(self) -> None:
        _, D, _ = _synthetic_window(3, noise=0.5, seed=2)
        assert "malinowski" not in estimate_rank(D).estimates
        assert "malinowski" in estimate_rank(D, include_malinowski=True).estimates


class TestEstimateRank:
    @pytest.mark.parametrize("n_components", [2, 3, 4, 5])
    def test_consensus_recovers_a_clean_window(self, n_components: int) -> None:
        _, D, _ = _synthetic_window(n_components, noise=0.0, seed=n_components)
        estimate = estimate_rank(D, max_rank=10)
        assert abs(estimate.rank - n_components) <= 1
        assert estimate.agreement

    @pytest.mark.parametrize("n_components", [2, 3, 4])
    def test_consensus_holds_up_under_shot_noise(self, n_components: int) -> None:
        """Der Regelfall nach der Matrix-Subtraktion: wenige, verrauschte Komponenten."""
        _, D, _ = _synthetic_window(n_components, noise=1.0, seed=10 + n_components)
        assert abs(estimate_rank(D, max_rank=10).rank - n_components) <= 1

    def test_stabilisation_is_off_by_default_because_it_measured_worse(self) -> None:
        """Hält das Ergebnis fest, das die Voreinstellung begründet.

        Die Anscombe-Transformation sollte die Rangschätzung robuster machen. Die
        Messung zeigt das Gegenteil: die Wurzel ist nichtlinear und zerstört die
        Niedrigrangstruktur, die gezählt werden soll. Bei rauschfreien Fenstern
        ist der Effekt drastisch — genau dort, wo die Antwort trivial sein sollte.
        """
        errors_on, errors_off = [], []
        for n_components in (2, 3, 4, 5):
            _, D, _ = _synthetic_window(n_components, noise=0.0, seed=n_components)
            errors_on.append(
                abs(estimate_rank(D, max_rank=12, stabilise=True).rank - n_components)
            )
            errors_off.append(
                abs(estimate_rank(D, max_rank=12, stabilise=False).rank - n_components)
            )
        assert sum(errors_off) < sum(errors_on)
        assert sum(errors_off) == 0

    def test_pure_noise_gives_a_low_rank(self) -> None:
        rng = np.random.default_rng(3)
        noise = rng.poisson(50.0, (120, 60)).astype(float)
        assert estimate_rank(noise, max_rank=10).rank <= 2

    def test_ceiling_is_reported_as_a_warning(self) -> None:
        """Ein an der Obergrenze klebender Rang ist eine Warnung, keine Antwort."""
        _, D, _ = _synthetic_window(8, noise=0.0, seed=4)
        estimate = estimate_rank(D, max_rank=2)
        assert estimate.rank == 2
        assert estimate.warning is not None
        assert "ceiling" in estimate.warning

    def test_disagreement_is_surfaced_not_hidden(self) -> None:
        """Ein Fenster, bei dem die Schätzer streiten, darf nicht still gemittelt werden."""
        _, D, _ = _synthetic_window(5, noise=3.0, separation_s=2.0, seed=7)
        estimate = estimate_rank(D, max_rank=12, spread_tolerance=0)
        if estimate.spread > 0:
            assert not estimate.agreement
            assert estimate.warning is not None

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_rank": 0, "min_rank": 1}, "is below min_rank"),
            ({"min_rank": 0}, "min_rank must be >= 1"),
        ],
    )
    def test_invalid_bounds_are_rejected(self, kwargs: dict, message: str) -> None:  # noqa: ANN001
        _, D, _ = _synthetic_window(2, seed=0)
        with pytest.raises(ValueError, match=message):
            estimate_rank(D, **kwargs)

    def test_non_matrix_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected a 2-D matrix"):
            estimate_rank(np.ones(10))


class TestRankOnTheUnsubtractedMatrix:
    def test_rank_estimation_is_unreliable_before_matrix_subtraction(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """Hält den Ist-Zustand fest, gegen den MS2.2 antritt.

        Auf der rohen Polyolefin-Matrix enthalten die Fenster median 5 und bis zu
        12 Komponenten, von denen viele Homologe mit Spektren-Cosinus > 0,98 sind.
        Die Rangschätzung trifft die wahre Komponentenzahl hier systematisch nicht
        — und kann es nicht, weil der effektive chemische Rang unter der wahren
        Zahl liegt. Genau deshalb steht die Matrix-Subtraktion vor der
        Kurvenauflösung. Der Test schreibt fest, dass dieser Zustand bekannt ist,
        statt ihn zu übersehen.
        """
        processed = preprocess(
            mixed_sample.cube, PreprocessingConfig(asls_iterations=6)
        )
        windows = detect_windows(processed)
        crowded = [
            window
            for window in windows
            if sum(
                1
                for component in mixed_sample.truth.components
                if window.start_s <= component.retention_time_s <= window.end_s
            )
            >= 6
        ]
        assert crowded, "der Benchmark muss überfüllte Fenster enthalten"

        within_one = 0
        for window in crowded:
            n_true = sum(
                1
                for component in mixed_sample.truth.components
                if window.start_s <= component.retention_time_s <= window.end_s
            )
            estimate = estimate_rank(processed.intensities[window.slice()], max_rank=14)
            if abs(estimate.rank - n_true) <= 1:
                within_one += 1

        assert within_one < len(crowded), (
            "wenn die Rangschätzung hier bereits zuverlässig wäre, wäre die "
            "Begründung für die Matrix-Subtraktion hinfällig"
        )
