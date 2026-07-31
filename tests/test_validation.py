"""Tests für den Bewertungs-Harness.

Der Harness ist das Messmittel für alles, was in MS2 und MS3 folgt. Wenn die
Zuordnung falsch ist, sind alle späteren Kennzahlen falsch — und zwar
unauffällig, weil sie trotzdem plausible Zahlen liefern. Entsprechend
misstrauisch sind die Tests hier: sie prüfen die Zuordnung an Fällen, bei denen
eine naive Implementierung nachweislich danebengreift.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrecycle_analytics.deconvolution.result import ResolutionError, ResolutionResult
from pyrecycle_analytics.validation import (
    cosine_similarity,
    cosine_similarity_matrix,
    explained_variance,
    lack_of_fit,
    match_components,
    naive_peak_integration,
    relative_area_error,
    score_markers,
    score_resolution,
    weighted_dot_similarity,
)
from tests.synthetic_data import SyntheticPyrogram


class TestSimilarityMetrics:
    def test_identical_vectors_have_similarity_one(self) -> None:
        vector = np.array([10.0, 5.0, 0.0, 2.0])
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_similarity_ignores_scale(self) -> None:
        """Ein Spektrum ist über sein Muster definiert, nicht über seine Höhe."""
        vector = np.array([10.0, 5.0, 1.0])
        assert cosine_similarity(vector, 37.0 * vector) == pytest.approx(1.0)

    def test_orthogonal_spectra_have_similarity_zero(self) -> None:
        assert cosine_similarity(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0])
        ) == pytest.approx(0.0)

    def test_zero_vector_gives_zero_not_nan(self) -> None:
        assert cosine_similarity(np.zeros(4), np.ones(4)) == 0.0

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity(np.ones(3), np.ones(4))

    def test_similarity_matrix_matches_pairwise_calls(self) -> None:
        rng = np.random.default_rng(0)
        rows_a = rng.random((3, 8))
        rows_b = rng.random((4, 8))
        matrix = cosine_similarity_matrix(rows_a, rows_b)
        assert matrix.shape == (3, 4)
        for i in range(3):
            for j in range(4):
                assert matrix[i, j] == pytest.approx(
                    cosine_similarity(rows_a[i], rows_b[j])
                )

    def test_mass_weighting_separates_what_plain_cosine_cannot(self) -> None:
        """Der Grund, warum es die gewichtete Variante überhaupt gibt.

        Zwei Alkane teilen sich die niedermassigen Fragmente 43/57 und
        unterscheiden sich nur im Molekülion. Für die einfache Cosinus-Metrik sind
        sie fast identisch; die Massengewichtung macht den Unterschied sichtbar.
        """
        mz = np.array([43.0, 57.0, 71.0, 198.0, 212.0])
        c14 = np.array([100.0, 92.0, 55.0, 8.0, 0.0])
        c15 = np.array([100.0, 92.0, 55.0, 0.0, 8.0])

        plain = cosine_similarity(c14, c15)
        weighted = weighted_dot_similarity(c14, c15, mz)
        assert plain > 0.99
        assert weighted < plain - 0.05

    def test_weighted_similarity_rejects_mismatched_axis(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            weighted_dot_similarity(np.ones(3), np.ones(3), np.ones(4))


class TestFitStatistics:
    def test_perfect_fit_has_zero_lack_of_fit(self) -> None:
        data = np.random.default_rng(1).random((10, 5))
        assert lack_of_fit(data, data) == pytest.approx(0.0)
        assert explained_variance(data, data) == pytest.approx(1.0)

    def test_lack_of_fit_grows_with_the_residual(self) -> None:
        data = np.ones((10, 5))
        small = lack_of_fit(data, data * 0.99)
        large = lack_of_fit(data, data * 0.8)
        assert 0.0 < small < large

    def test_zero_data_does_not_divide_by_zero(self) -> None:
        assert lack_of_fit(np.zeros((4, 4)), np.zeros((4, 4))) == 0.0

    def test_explained_variance_can_be_negative(self) -> None:
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert explained_variance(data, np.full_like(data, 100.0)) < 0.0

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            lack_of_fit(np.ones((3, 3)), np.ones((3, 4)))

    def test_relative_area_error_signs_correctly(self) -> None:
        assert relative_area_error(110.0, 100.0) == pytest.approx(0.10)
        assert relative_area_error(90.0, 100.0) == pytest.approx(-0.10)
        assert relative_area_error(0.0, 0.0) == 0.0
        assert relative_area_error(5.0, 0.0) == float("inf")


class TestComponentMatching:
    @staticmethod
    def _profile(retention_times: np.ndarray, center: float, area: float) -> np.ndarray:
        sigma = 2.0
        profile = np.exp(-0.5 * ((retention_times - center) / sigma) ** 2)
        return area * profile / np.trapezoid(profile, retention_times)

    def test_identical_factorisation_matches_perfectly(self) -> None:
        t = np.linspace(0.0, 100.0, 501)
        C = np.column_stack(
            [self._profile(t, 30.0, 100.0), self._profile(t, 70.0, 50.0)]
        )
        S = np.array([[0.6, 0.4, 0.0], [0.0, 0.3, 0.7]])

        matching = match_components(C, S, C, S, t)
        assert len(matching.matches) == 2
        assert matching.recall == 1.0
        assert matching.precision == 1.0
        assert all(match.spectral_similarity > 0.999 for match in matching.matches)
        assert all(abs(match.area_relative_error) < 1e-6 for match in matching.matches)

    def test_permuted_components_are_still_matched(self) -> None:
        """MCR-ALS liefert beliebige Reihenfolge — genau dafür existiert das Modul."""
        t = np.linspace(0.0, 100.0, 501)
        C = np.column_stack(
            [self._profile(t, 30.0, 100.0), self._profile(t, 70.0, 50.0)]
        )
        S = np.array([[0.6, 0.4, 0.0], [0.0, 0.3, 0.7]])

        matching = match_components(C, S, C[:, ::-1], S[::-1, :], t)
        assert len(matching.matches) == 2
        pairs = {(m.true_index, m.recovered_index) for m in matching.matches}
        assert pairs == {(0, 1), (1, 0)}

    def test_retention_disambiguates_near_identical_spectra(self) -> None:
        """Der Fall, an dem eine rein spektrale Zuordnung scheitert.

        Zwei Homologe mit Spektren-Cosinus > 0,99 lassen sich nur über ihre
        Retentionszeit auseinanderhalten. Ohne den Retentionsterm wäre die
        Zuordnung ein Münzwurf — und die Flächen würden systematisch vertauscht.
        """
        t = np.linspace(0.0, 200.0, 1001)
        C = np.column_stack(
            [self._profile(t, 80.0, 100.0), self._profile(t, 95.0, 20.0)]
        )
        S = np.array([[0.50, 0.49, 0.01], [0.49, 0.50, 0.01]])
        assert cosine_similarity(S[0], S[1]) > 0.99

        matching = match_components(C, S, C, S, t, retention_tolerance_s=4.0)
        assert {(m.true_index, m.recovered_index) for m in matching.matches} == {
            (0, 0),
            (1, 1),
        }
        assert all(abs(m.area_relative_error) < 1e-6 for m in matching.matches)

    def test_assignment_is_global_not_greedy(self) -> None:
        """Ein gieriges Verfahren verbaut sich hier die zweite Zuordnung.

        Die erste wahre Komponente passt spektral etwas besser zur zweiten
        gefundenen. Greedy nimmt sie und lässt für die zweite wahre Komponente nur
        noch die schlechte Alternative — die ungarische Methode optimiert die
        Summe und findet die richtige Paarung.
        """
        t = np.linspace(0.0, 200.0, 1001)
        C_true = np.column_stack(
            [self._profile(t, 60.0, 100.0), self._profile(t, 120.0, 100.0)]
        )
        C_found = np.column_stack(
            [self._profile(t, 120.2, 100.0), self._profile(t, 60.2, 100.0)]
        )
        S_true = np.array([[0.7, 0.3, 0.0], [0.0, 0.4, 0.6]])
        S_found = np.array([[0.0, 0.4, 0.6], [0.7, 0.3, 0.0]])

        matching = match_components(C_true, S_true, C_found, S_found, t)
        assert {(m.true_index, m.recovered_index) for m in matching.matches} == {
            (0, 1),
            (1, 0),
        }

    def test_missing_component_is_counted_as_a_miss(self) -> None:
        t = np.linspace(0.0, 100.0, 501)
        C = np.column_stack(
            [self._profile(t, 30.0, 100.0), self._profile(t, 70.0, 50.0)]
        )
        S = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        matching = match_components(C, S, C[:, :1], S[:1, :], t)
        assert len(matching.matches) == 1
        assert matching.missed_true == (1,)
        assert matching.recall == pytest.approx(0.5)
        assert matching.precision == pytest.approx(1.0)

    def test_invented_component_is_counted_as_spurious(self) -> None:
        t = np.linspace(0.0, 100.0, 501)
        C_true = self._profile(t, 30.0, 100.0)[:, None]
        S_true = np.array([[1.0, 0.0, 0.0]])
        C_found = np.column_stack([C_true[:, 0], self._profile(t, 80.0, 10.0)])
        S_found = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        matching = match_components(C_true, S_true, C_found, S_found, t)
        assert matching.spurious_recovered == (1,)
        assert matching.precision == pytest.approx(0.5)

    def test_low_similarity_pair_is_rejected(self) -> None:
        t = np.linspace(0.0, 100.0, 501)
        C = self._profile(t, 50.0, 100.0)[:, None]
        S_true = np.array([[1.0, 0.0, 0.0]])
        S_wrong = np.array([[0.0, 1.0, 0.0]])

        matching = match_components(
            C, S_true, C, S_wrong, t, min_spectral_similarity=0.5
        )
        assert not matching.matches
        assert matching.missed_true == (0,)
        assert matching.spurious_recovered == (0,)

    def test_hard_retention_cut_rejects_a_distant_pair(self) -> None:
        t = np.linspace(0.0, 400.0, 2001)
        C_true = self._profile(t, 100.0, 100.0)[:, None]
        C_found = self._profile(t, 300.0, 100.0)[:, None]
        S = np.array([[1.0, 0.0]])

        matching = match_components(
            C_true, S, C_found, S, t, max_retention_error_s=20.0
        )
        assert not matching.matches

    def test_empty_input_is_handled(self) -> None:
        t = np.linspace(0.0, 10.0, 11)
        matching = match_components(
            np.zeros((11, 0)), np.zeros((0, 3)), np.zeros((11, 0)), np.zeros((0, 3)), t
        )
        assert matching.matches == ()
        assert matching.recall == 0.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"retention_tolerance_s": 0.0}, "retention_tolerance_s must be > 0"),
        ],
    )
    def test_invalid_parameters_are_rejected(self, kwargs: dict, message: str) -> None:  # noqa: ANN001
        t = np.linspace(0.0, 10.0, 11)
        C = np.ones((11, 1))
        S = np.ones((1, 2))
        with pytest.raises(ValueError, match=message):
            match_components(C, S, C, S, t, **kwargs)

    def test_inconsistent_shapes_are_rejected(self) -> None:
        t = np.linspace(0.0, 10.0, 11)
        with pytest.raises(ValueError, match="components but S has"):
            match_components(
                np.ones((11, 2)), np.ones((1, 3)), np.ones((11, 1)), np.ones((1, 3)), t
            )


class TestResolutionResult:
    def test_areas_and_apexes_are_derived_correctly(self) -> None:
        t = np.linspace(0.0, 100.0, 1001)
        profile = np.exp(-0.5 * ((t - 40.0) / 2.0) ** 2)
        profile = 500.0 * profile / np.trapezoid(profile, t)
        result = ResolutionResult(
            retention_times=t, mz_axis=np.array([50.0, 51.0]),
            C=profile[:, None], S=np.array([[0.5, 0.5]]),
        )
        assert result.areas[0] == pytest.approx(500.0, rel=1e-6)
        assert result.apex_times[0] == pytest.approx(40.0, abs=0.2)
        assert result.n_components == 1

    def test_sorting_orders_components_by_apex(self) -> None:
        t = np.linspace(0.0, 100.0, 501)
        late = np.exp(-0.5 * ((t - 80.0) / 2.0) ** 2)
        early = np.exp(-0.5 * ((t - 20.0) / 2.0) ** 2)
        result = ResolutionResult(
            retention_times=t, mz_axis=np.array([50.0]),
            C=np.column_stack([late, early]), S=np.array([[1.0], [1.0]]),
        )
        assert result.sorted_by_retention().apex_times[0] < 50.0

    def test_negligible_components_can_be_dropped(self) -> None:
        t = np.linspace(0.0, 100.0, 501)
        big = np.exp(-0.5 * ((t - 50.0) / 2.0) ** 2)
        tiny = 1e-8 * np.exp(-0.5 * ((t - 20.0) / 2.0) ** 2)
        result = ResolutionResult(
            retention_times=t, mz_axis=np.array([50.0]),
            C=np.column_stack([big, tiny]), S=np.array([[1.0], [1.0]]),
        )
        assert result.drop_negligible().n_components == 1

    def test_dropping_never_returns_an_empty_result(self) -> None:
        t = np.linspace(0.0, 10.0, 11)
        result = ResolutionResult(
            retention_times=t, mz_axis=np.array([50.0]),
            C=np.zeros((11, 2)), S=np.ones((2, 1)),
        )
        assert result.drop_negligible().n_components >= 1

    @pytest.mark.parametrize(
        ("C_shape", "S_shape", "message"),
        [
            ((5, 2), (2, 3), "C has 5 rows"),
            ((11, 2), (2, 9), "S has 9 columns"),
            ((11, 3), (2, 3), "C has 3 components"),
        ],
    )
    def test_inconsistent_factors_are_rejected(
        self, C_shape: tuple[int, int], S_shape: tuple[int, int], message: str
    ) -> None:
        with pytest.raises(ResolutionError, match=message):
            ResolutionResult(
                retention_times=np.linspace(0.0, 10.0, 11),
                mz_axis=np.array([50.0, 51.0, 52.0]),
                C=np.ones(C_shape),
                S=np.ones(S_shape),
            )


class TestScoring:
    def test_perfect_resolution_scores_perfectly(
        self, noiseless_sample: SyntheticPyrogram
    ) -> None:
        result = ResolutionResult(
            retention_times=noiseless_sample.cube.retention_times,
            mz_axis=noiseless_sample.cube.mz_axis,
            C=noiseless_sample.C,
            S=noiseless_sample.S,
            method="oracle",
        )
        score = score_resolution(
            noiseless_sample.C,
            noiseless_sample.S,
            noiseless_sample.cube.retention_times,
            result,
        )
        assert score.recall == pytest.approx(1.0)
        assert score.precision == pytest.approx(1.0)
        assert score.lack_of_fit == pytest.approx(0.0, abs=1e-6)
        assert score.explained_variance == pytest.approx(1.0, abs=1e-9)
        assert score.median_absolute_area_error < 1e-6

    def test_summary_is_json_friendly(self, noiseless_sample: SyntheticPyrogram) -> None:
        import json

        result = ResolutionResult(
            retention_times=noiseless_sample.cube.retention_times,
            mz_axis=noiseless_sample.cube.mz_axis,
            C=noiseless_sample.C,
            S=noiseless_sample.S,
        )
        score = score_resolution(
            noiseless_sample.C, noiseless_sample.S,
            noiseless_sample.cube.retention_times, result,
        )
        json.dumps(score.summary())

    def test_marker_scoring_reports_named_compounds(
        self, noiseless_sample: SyntheticPyrogram
    ) -> None:
        result = ResolutionResult(
            retention_times=noiseless_sample.cube.retention_times,
            mz_axis=noiseless_sample.cube.mz_axis,
            C=noiseless_sample.C,
            S=noiseless_sample.S,
        )
        markers = score_markers(
            noiseless_sample.C,
            noiseless_sample.S,
            noiseless_sample.cube.retention_times,
            result,
            noiseless_sample.component_names,
            ["styrene", "epsilon-caprolactam"],
        )
        assert [m.name for m in markers] == ["styrene", "epsilon-caprolactam"]
        assert all(m.found for m in markers)
        assert all(m.spectral_similarity > 0.999 for m in markers)

    def test_unknown_marker_is_reported_clearly(
        self, noiseless_sample: SyntheticPyrogram
    ) -> None:
        result = ResolutionResult(
            retention_times=noiseless_sample.cube.retention_times,
            mz_axis=noiseless_sample.cube.mz_axis,
            C=noiseless_sample.C,
            S=noiseless_sample.S,
        )
        with pytest.raises(KeyError, match="not among the ground-truth components"):
            score_markers(
                noiseless_sample.C, noiseless_sample.S,
                noiseless_sample.cube.retention_times, result,
                noiseless_sample.component_names, ["polytetrafluoroethylene"],
            )


class TestNaiveBaseline:
    def test_runs_and_returns_a_consistent_factorisation(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        result = naive_peak_integration(hdpe_sample.cube)
        assert result.n_components > 5
        assert np.allclose(result.S.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(result.C >= 0.0)
        assert np.all(result.S >= 0.0)

    def test_resolves_well_separated_peaks(self) -> None:
        """Auf getrennten Peaks ist das naive Verfahren völlig ausreichend.

        Der Test hält fest, dass die Vergleichsbasis kein Strohmann ist: sie
        funktioniert dort, wo Chromatographie funktioniert.
        """
        from data_schemas.acquisition import SampleMetadata
        from data_schemas.enums import SourceFormat
        from data_schemas.pyrogram import PyrogramMetadata
        from pyrecycle_analytics.core.binning import axis_to_spec
        from pyrecycle_analytics.core.datacube import PyrogramDataCube
        from pyrecycle_analytics.core.peakshapes import gaussian

        t = np.linspace(0.0, 600.0, 3001)
        mz = np.arange(50.0, 60.0)
        C = np.column_stack(
            [gaussian(t, 150.0, 3.0, 1e6), gaussian(t, 400.0, 3.0, 5e5)]
        )
        S = np.zeros((2, mz.size))
        S[0, 0] = S[0, 3] = 0.5
        S[1, 6] = S[1, 9] = 0.5
        D = C @ S

        cube = PyrogramDataCube(
            retention_times=t, mz_axis=mz, intensities=D,
            metadata=PyrogramMetadata(
                sample=SampleMetadata(sample_id="separated"),
                source_format=SourceFormat.SYNTHETIC,
                n_scans=t.size, mz_axis=axis_to_spec(mz),
                rt_start_s=float(t[0]), rt_end_s=float(t[-1]),
            ),
        )
        result = naive_peak_integration(cube)
        score = score_resolution(C, S, t, result, observed=D)
        assert score.recall == pytest.approx(1.0)
        assert score.mean_spectral_similarity > 0.99

    def test_fails_on_fused_clusters_which_is_the_point(
        self, coelution_sample: SyntheticPyrogram
    ) -> None:
        """Die Vergleichsbasis muss dort scheitern, wo MCR-ALS gebraucht wird.

        Ein TIC-Peak über einem verschmolzenen Cluster liefert ein Mischspektrum
        aus allen Beteiligten. Findet das naive Verfahren hier schon alles, wäre
        der ganze MS2-Aufwand nicht zu rechtfertigen — der Test hält den
        Ausgangszustand fest, gegen den MCR-ALS antreten muss.
        """
        result = naive_peak_integration(coelution_sample.cube)
        score = score_resolution(
            coelution_sample.C,
            coelution_sample.S,
            coelution_sample.cube.retention_times,
            result,
            observed=coelution_sample.cube.intensities,
        )
        assert score.n_recovered < score.n_true, (
            "das naive Verfahren darf die verschmolzenen Komponenten nicht auflösen"
        )
        assert score.recall < 0.5

    def test_empty_chromatogram_does_not_crash(self, tiny_cube) -> None:  # noqa: ANN001
        empty = tiny_cube.with_intensities(np.zeros(tiny_cube.shape))
        result = naive_peak_integration(empty)
        assert result.n_components == 1
        assert result.diagnostics["n_peaks"] == 0

    def test_invalid_prominence_is_rejected(self, tiny_cube) -> None:  # noqa: ANN001
        with pytest.raises(ValueError, match="min_prominence_fraction must be > 0"):
            naive_peak_integration(tiny_cube, min_prominence_fraction=0.0)
