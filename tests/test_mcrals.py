"""Tests für MCR-ALS und die vollständige Deconvolution-Kette.

Die Tests trennen zwei Fragen, die gern vermischt werden:

* **Rechnet der Löser richtig?** Auf konstruierten Fenstern mit bekannter Antwort
  muss die Zerlegung die Komponenten zurückgeben — dort ist das Ergebnis eindeutig
  genug, um es zu fordern.
* **Nützt er auf echten Pyrogrammen?** Dort ist die Zerlegung mehrdeutig, und die
  ehrliche Messgröße ist der Vergleich gegen die triviale Vergleichsbasis und
  gegen die Marker, auf denen ein Rezyklat-Report tatsächlich beruht.

Ein Test hält ausdrücklich eine **Grenze** fest: Naphthalin liegt bei 0,25 %
Signalanteil seines Fensters und wird auch mit großzügigem Rang nicht sauber
zurückgewonnen. Das ist kein Fehler, den man wegkonfiguriert, sondern die
Nachweisgrenze des Verfahrens, und sie gehört dokumentiert.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrecycle_analytics.core.peakshapes import gaussian
from pyrecycle_analytics.deconvolution import (
    DeconvolutionConfig,
    McrAlsOptions,
    ResolutionError,
    enforce_unimodality,
    mcr_als,
    resolve_pyrogram,
    simplisma,
)
from pyrecycle_analytics.preprocessing import PreprocessingConfig, preprocess
from pyrecycle_analytics.validation import (
    cosine_similarity,
    naive_peak_integration,
    score_markers,
    score_resolution,
)
from tests.synthetic_data import SyntheticPyrogram

BENCHMARK_MARKERS = (
    "styrene",
    "epsilon-caprolactam",
    "benzoic acid",
    "bis(2-ethylhexyl) phthalate (DEHP)",
    "2,4-diphenyl-1-butene",
    "divinyl terephthalate",
)


def _known_window(
    n_components: int = 3,
    *,
    separation_s: float = 6.0,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ein Fenster mit bekannter, eindeutiger Zerlegung."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 120.0, 150)
    mz = np.arange(40.0, 100.0)
    centers = 40.0 + separation_s * np.arange(n_components)
    C = np.column_stack([gaussian(t, center, 3.0, 1e5) for center in centers])
    S = np.zeros((n_components, mz.size))
    for row in range(n_components):
        channels = rng.choice(mz.size, size=5, replace=False)
        S[row, channels] = rng.uniform(0.4, 1.0, size=5)
        S[row] /= S[row].sum()
    D = C @ S
    if noise > 0.0:
        D = np.clip(
            D + rng.normal(0.0, noise * np.sqrt(np.clip(D, 1.0, None)), D.shape), 0.0, None
        )
    return t, mz, D, S


def _best_similarity(true_spectrum: np.ndarray, spectra: np.ndarray) -> float:
    return max(cosine_similarity(true_spectrum, row) for row in spectra)


class TestSimplisma:
    def test_selects_as_many_spectra_as_requested(self) -> None:
        _, _, D, _ = _known_window(3)
        assert simplisma(D, 3).shape == (3, D.shape[1])

    def test_spectra_are_normalised(self) -> None:
        _, _, D, _ = _known_window(3)
        spectra = simplisma(D, 3)
        occupied = spectra.sum(axis=1) > 0
        assert np.allclose(spectra[occupied].sum(axis=1), 1.0)

    def test_is_deterministic(self) -> None:
        """Ein zufälliger Start würde die Ergebnisse unreproduzierbar machen."""
        _, _, D, _ = _known_window(4, seed=1)
        assert np.array_equal(simplisma(D, 4), simplisma(D, 4))

    def test_finds_distinct_starting_points(self) -> None:
        _, _, D, _ = _known_window(3, separation_s=20.0)
        spectra = simplisma(D, 3)
        for first in range(3):
            for second in range(first + 1, 3):
                assert cosine_similarity(spectra[first], spectra[second]) < 0.99

    @pytest.mark.parametrize("n_components", [0, 10_000])
    def test_invalid_component_counts_are_rejected(self, n_components: int) -> None:
        _, _, D, _ = _known_window(2)
        with pytest.raises(ResolutionError):
            simplisma(D, n_components)


class TestUnimodality:
    def test_a_single_peak_is_left_alone(self) -> None:
        t = np.linspace(0.0, 100.0, 201)
        profile = gaussian(t, 50.0, 5.0, 1.0)
        assert np.allclose(enforce_unimodality(profile), profile, atol=1e-12)

    def test_a_second_maximum_is_suppressed(self) -> None:
        """Zwei Maxima in einem Profil heißen: zwei Verbindungen wurden vermengt."""
        t = np.linspace(0.0, 100.0, 201)
        profile = gaussian(t, 30.0, 3.0, 1.0) + gaussian(t, 70.0, 3.0, 1.0)
        corrected = enforce_unimodality(profile, tolerance=0.0)
        second_peak = corrected[t > 60.0].max()
        assert second_peak < 0.2 * corrected.max()

    def test_tolerance_absorbs_noise_on_the_flanks(self) -> None:
        t = np.linspace(0.0, 100.0, 201)
        rng = np.random.default_rng(0)
        profile = gaussian(t, 50.0, 8.0, 1.0)
        noisy = profile * (1.0 + 0.02 * rng.standard_normal(t.size))
        corrected = enforce_unimodality(np.clip(noisy, 0.0, None), tolerance=0.1)
        assert np.trapezoid(corrected, t) > 0.9 * np.trapezoid(profile, t)

    def test_short_profiles_are_returned_unchanged(self) -> None:
        profile = np.array([1.0, 2.0])
        assert np.array_equal(enforce_unimodality(profile), profile)


class TestMcrAlsOnKnownWindows:
    @pytest.mark.parametrize("n_components", [2, 3, 4])
    def test_recovers_a_clean_factorisation(self, n_components: int) -> None:
        t, mz, D, S = _known_window(n_components, separation_s=10.0, seed=n_components)
        result = mcr_als(D, n_components, retention_times=t, mz_axis=mz)

        assert result.n_components == n_components
        for row in S:
            assert _best_similarity(row, result.S) > 0.95
        assert result.lack_of_fit < 5.0

    def test_recovers_fused_peaks_that_the_tic_cannot_separate(self) -> None:
        """Der eigentliche Zweck: Trennung aus der m/z-Dimension.

        Bei 6 s Abstand und 3 s Breite zeigt der TIC einen einzigen Buckel. Nur
        weil die Spektren verschieden sind, ist die Zerlegung überhaupt möglich.
        """
        t, mz, D, S = _known_window(2, separation_s=6.0, seed=7)
        assert len(np.flatnonzero(np.diff(np.sign(np.diff(D.sum(axis=1)))) < 0)) == 1

        result = mcr_als(D, 2, retention_times=t, mz_axis=mz)
        for row in S:
            assert _best_similarity(row, result.S) > 0.9

    def test_survives_realistic_noise(self) -> None:
        t, mz, D, S = _known_window(3, separation_s=10.0, noise=1.0, seed=3)
        result = mcr_als(D, 3, retention_times=t, mz_axis=mz)
        for row in S:
            assert _best_similarity(row, result.S) > 0.85

    def test_factors_obey_the_constraints(self) -> None:
        t, mz, D, _ = _known_window(3, noise=1.0, seed=4)
        result = mcr_als(D, 3, retention_times=t, mz_axis=mz)
        assert np.all(result.C >= 0.0)
        assert np.all(result.S >= 0.0)
        assert np.allclose(result.S.sum(axis=1), 1.0)

    def test_profiles_are_unimodal_when_required(self) -> None:
        t, mz, D, _ = _known_window(3, noise=1.0, seed=5)
        result = mcr_als(
            D, 3, retention_times=t, mz_axis=mz, options=McrAlsOptions(unimodal_c=True)
        )
        for component in range(result.n_components):
            profile = result.C[:, component]
            if profile.max() <= 0:
                continue
            significant = profile > 0.05 * profile.max()
            maxima = np.flatnonzero(
                (profile[1:-1] > profile[:-2])
                & (profile[1:-1] >= profile[2:])
                & significant[1:-1]
            )
            assert maxima.size <= 1

    def test_convergence_is_reported(self) -> None:
        t, mz, D, _ = _known_window(2, separation_s=15.0)
        result = mcr_als(D, 2, retention_times=t, mz_axis=mz)
        assert result.converged
        assert result.n_iterations < 200

    def test_supplied_initialisation_is_used(self) -> None:
        t, mz, D, S = _known_window(3, separation_s=10.0, seed=9)
        result = mcr_als(D, 3, retention_times=t, mz_axis=mz, initial_spectra=S)
        assert result.diagnostics["initialisation"] == "supplied"
        assert result.lack_of_fit < 2.0

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda D: D * -1.0, "negative intensities"),
            (lambda D: D * 0.0, "empty"),
            (lambda D: D.ravel(), "expected a 2-D window"),
        ],
    )
    def test_invalid_input_is_rejected(self, mutate, message: str) -> None:  # noqa: ANN001
        _, _, D, _ = _known_window(2)
        with pytest.raises(ResolutionError, match=message):
            mcr_als(mutate(D), 2)

    def test_impossible_component_count_is_rejected(self) -> None:
        _, _, D, _ = _known_window(2)
        with pytest.raises(ResolutionError, match="outside 1.."):
            mcr_als(D, 10_000)

    def test_mismatched_initialisation_is_rejected(self) -> None:
        _, _, D, _ = _known_window(2)
        with pytest.raises(ResolutionError, match="initial_spectra has shape"):
            mcr_als(D, 2, initial_spectra=np.ones((3, D.shape[1])))


class TestPipelineAgainstTheBaseline:
    """Der Vergleich, der den MS2-Aufwand rechtfertigen muss."""

    @staticmethod
    @pytest.fixture(scope="class")
    def outcome(mixed_sample: SyntheticPyrogram):  # noqa: ANN205
        config = DeconvolutionConfig(
            preprocessing=PreprocessingConfig(asls_iterations=8)
        )
        report = resolve_pyrogram(mixed_sample.cube, config)
        resolved = score_resolution(
            mixed_sample.C,
            mixed_sample.S,
            mixed_sample.cube.retention_times,
            report.result,
            observed=mixed_sample.cube.intensities,
        )
        processed = preprocess(mixed_sample.cube, config.preprocessing)
        baseline = score_resolution(
            mixed_sample.C,
            mixed_sample.S,
            mixed_sample.cube.retention_times,
            naive_peak_integration(processed),
            observed=mixed_sample.cube.intensities,
        )
        return report, resolved, baseline

    def test_finds_substantially_more_components_than_the_baseline(
        self, outcome
    ) -> None:  # noqa: ANN001
        _, resolved, baseline = outcome
        assert resolved.recall > 1.5 * baseline.recall

    def test_area_accuracy_improves_markedly(self, outcome) -> None:  # noqa: ANN001
        """Die auffälligste Schwäche des naiven Verfahrens.

        Es integriert den ganzen TIC im Peakbereich, also auch alle koeluierenden
        Nachbarn — gemessen 90-180 % Flächenfehler. Genau das behebt die
        Kurvenauflösung.
        """
        _, resolved, baseline = outcome
        assert baseline.median_absolute_area_error > 1.0
        assert resolved.median_absolute_area_error < 0.6 * baseline.median_absolute_area_error

    def test_all_benchmark_markers_are_recovered(
        self, mixed_sample: SyntheticPyrogram, outcome
    ) -> None:  # noqa: ANN001
        """Das Kriterium, auf dem ein Rezyklat-Report beruht.

        Die triviale Vergleichsbasis findet Caprolactam und Benzoesäure gar nicht.
        """
        report, _, _ = outcome
        scores = score_markers(
            mixed_sample.C,
            mixed_sample.S,
            mixed_sample.cube.retention_times,
            report.result,
            mixed_sample.component_names,
            list(BENCHMARK_MARKERS),
        )
        missing = [score.name for score in scores if not score.found]
        assert not missing, f"nicht gefunden: {missing}"
        # Gemessene Spanne: 0,82 (Caprolactam, das sich sein Fenster mit einem
        # Restkamm teilt) bis 0,995 (PS-Dimer). Die Schranke hält den gemessenen
        # Stand fest, statt eine runde Wunschzahl zu behaupten.
        assert all(score.spectral_similarity > 0.80 for score in scores)
        assert np.median([score.spectral_similarity for score in scores]) > 0.95

    def test_baseline_misses_markers_that_the_pipeline_finds(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        processed = preprocess(
            mixed_sample.cube, PreprocessingConfig(asls_iterations=8)
        )
        scores = score_markers(
            mixed_sample.C,
            mixed_sample.S,
            mixed_sample.cube.retention_times,
            naive_peak_integration(processed),
            mixed_sample.component_names,
            list(BENCHMARK_MARKERS),
        )
        assert sum(1 for score in scores if not score.found) >= 2

    def test_matrix_subtraction_is_part_of_the_chain(self, outcome) -> None:  # noqa: ANN001
        report, _, _ = outcome
        assert report.matrix_explained_fraction > 0.2
        assert report.diagnostics["matrix_subtracted"]

    def test_report_exposes_its_own_uncertainty(self, outcome) -> None:  # noqa: ANN001
        """Fenster mit widersprüchlicher Rangschätzung müssen sichtbar sein."""
        report, _, _ = outcome
        assert isinstance(report.warnings, tuple)
        assert report.n_resolved > 0
        assert "rank_disagreements" in report.diagnostics


class TestDetectionLimit:
    def test_a_marker_below_a_percent_of_its_window_is_not_recovered(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """Hält die Nachweisgrenze fest, statt sie zu verschweigen.

        Naphthalin trägt 0,25 % zum Signal seines Fensters bei. Gemessen bleibt
        die Spektrenähnlichkeit auch bei einem Rang von 8 unter 0,5. Wer diesen
        Marker braucht, braucht eine selektivere Messung — etwa SIM auf m/z 128 —
        und nicht eine andere Einstellung der Kurvenauflösung.
        """
        report = resolve_pyrogram(
            mixed_sample.cube,
            DeconvolutionConfig(preprocessing=PreprocessingConfig(asls_iterations=8)),
        )
        scores = score_markers(
            mixed_sample.C,
            mixed_sample.S,
            mixed_sample.cube.retention_times,
            report.result,
            mixed_sample.component_names,
            ["naphthalene"],
        )
        assert not scores[0].found or scores[0].spectral_similarity < 0.9


class TestPipelineConfiguration:
    def test_chain_runs_without_matrix_subtraction(
        self, ps_sample: SyntheticPyrogram
    ) -> None:
        """Reines PS hat keine Matrix — die Kette muss trotzdem durchlaufen."""
        report = resolve_pyrogram(
            ps_sample.cube,
            DeconvolutionConfig(preprocessing=PreprocessingConfig(asls_iterations=6)),
        )
        assert report.matrix_explained_fraction == 0.0
        assert any("no polyolefin matrix" in warning for warning in report.warnings)
        assert report.result.n_components > 0

    def test_requiring_a_matrix_fails_loudly_when_there_is_none(
        self, ps_sample: SyntheticPyrogram
    ) -> None:
        from pyrecycle_analytics.matrix import MatrixSubtractionError

        with pytest.raises(MatrixSubtractionError):
            resolve_pyrogram(
                ps_sample.cube,
                DeconvolutionConfig(
                    preprocessing=PreprocessingConfig(asls_iterations=6),
                    require_matrix=True,
                ),
            )

    def test_configuration_is_frozen_and_serialisable(self) -> None:
        config = DeconvolutionConfig(rank_margin=3)
        assert DeconvolutionConfig.model_validate_json(config.model_dump_json()) == config
        with pytest.raises(ValueError, match="frozen"):
            config.rank_margin = 5  # type: ignore[misc]

    def test_rank_margin_trades_recall_against_precision(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """Dokumentiert den Kompromiss, statt eine Zahl unbegründet zu setzen."""
        base = PreprocessingConfig(asls_iterations=8)
        lean = resolve_pyrogram(
            mixed_sample.cube, DeconvolutionConfig(preprocessing=base, rank_margin=0)
        )
        generous = resolve_pyrogram(
            mixed_sample.cube, DeconvolutionConfig(preprocessing=base, rank_margin=4)
        )
        assert generous.result.n_components > lean.result.n_components
