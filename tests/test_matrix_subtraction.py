"""Tests für die Polyolefin-Matrix-Subtraktion.

Der schärfste Test ist nicht, wie viel Matrix entfernt wird, sondern wie viel
Analyt überlebt. Ein Abzug, der den Kamm sauber wegnimmt und dabei den
Spurenmarker mitnimmt, ist wertlos — und genau das passiert, wenn die
Kontaminationsprüfung zu grob ist.

Zur Referenzgröße: gemessen wird gegen den **wahren Markerbeitrag** aus dem Ground
Truth, nicht gegen das Signal vor der Subtraktion. Letzteres enthält an der
Markerposition auch Matrixanteile — m/z 113 etwa ist neben dem Caprolactam-Ion
auch das Alkylfragment C8H17+ —, deren Entfernung korrekt ist und nicht als
Verlust zählen darf.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrecycle_analytics.deconvolution.windows import detect_windows
from pyrecycle_analytics.matrix import (
    MATRIX_INDICATOR_IONS,
    MatrixSubtractionError,
    build_matrix_model,
    detect_homologue_comb,
    subtract_polymer_matrix,
)
from pyrecycle_analytics.preprocessing import PreprocessingConfig, preprocess
from tests.synthetic_data import SyntheticPyrogram

# Marker, die den Abzug überstehen müssen: Fremdpolymere und Additive.
CRITICAL_MARKERS = (
    "styrene",
    "epsilon-caprolactam",
    "benzoic acid",
    "bis(2-ethylhexyl) phthalate (DEHP)",
    "divinyl terephthalate",
    "naphthalene",
    "2,4-diphenyl-1-butene",
    "erucamide",
    "toluene",
)


def _processed(sample: SyntheticPyrogram):  # noqa: ANN202
    return preprocess(sample.cube, PreprocessingConfig(asls_iterations=8))


def _true_marker_contribution(
    sample: SyntheticPyrogram, name: str, window_scans: int = 8
) -> tuple[float, slice, float]:
    """Wahrer Beitrag eines Markers auf seinem Quantifier-Ion, mit Fenster."""
    component = sample.truth.component_by_name(name)
    index = sample.index_of(name)
    quantifier = float(component.quantifier_mz)
    channel = sample.cube.mz_index(quantifier)
    apex = sample.cube.nearest_scan(component.retention_time_s)
    window = slice(max(apex - window_scans, 0), apex + window_scans + 1)
    contribution = float((sample.C[window, index] * sample.S[index, channel]).sum())
    return contribution, window, quantifier


class TestCombDetection:
    def test_finds_a_regular_series_in_a_polyolefin_sample(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        detection = detect_homologue_comb(_processed(hdpe_sample))
        assert detection.n_clusters >= 8
        assert 30.0 < detection.spacing_s < 60.0

    def test_cluster_spacing_matches_the_generator_retention_model(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Der erkannte Abstand muss dem echten Kettenlängenabstand entsprechen."""
        detection = detect_homologue_comb(_processed(hdpe_sample))
        alkanes = sorted(
            (
                component
                for component in hdpe_sample.truth.components
                if component.name.startswith("n-alkane")
            ),
            key=lambda component: component.retention_time_s,
        )
        true_spacing = float(
            np.median(np.diff([component.retention_time_s for component in alkanes]))
        )
        assert detection.spacing_s == pytest.approx(true_spacing, rel=0.15)

    def test_boundaries_partition_the_detected_range(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        detection = detect_homologue_comb(_processed(hdpe_sample))
        assert detection.boundaries.size == detection.n_clusters + 1
        assert np.all(np.diff(detection.boundaries) > 0)

    def test_a_sample_without_a_comb_is_rejected(
        self, ps_sample: SyntheticPyrogram
    ) -> None:
        """Reines PS hat keine Homologenreihe — der Abzug muss sich weigern.

        Stillschweigend irgendetwas als Kamm zu modellieren wäre der gefährlichere
        Ausgang: es würde Styrol-Oligomere als Matrix abziehen. Das Kriterium ist
        die Regelmäßigkeit: gemessen liegt die Abstandsstreuung eines echten Kamms
        bei 5 %, die der Peakfolge eines PS-Pyrogramms bei 15-24 %.
        """
        with pytest.raises(MatrixSubtractionError, match="not a homologous series"):
            detect_homologue_comb(_processed(ps_sample))

    def test_unknown_matrix_type_is_rejected(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        with pytest.raises(MatrixSubtractionError, match="unknown matrix_type"):
            detect_homologue_comb(_processed(hdpe_sample), matrix_type="PTFE_Backbone")

    def test_indicator_ions_outside_the_range_are_reported(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        narrow = hdpe_sample.cube.window(mz_range=(200.0, 300.0))
        with pytest.raises(MatrixSubtractionError, match="none of the indicator ions"):
            detect_homologue_comb(narrow)

    def test_every_matrix_type_is_usable(self, hdpe_sample: SyntheticPyrogram) -> None:
        processed = _processed(hdpe_sample)
        for matrix_type in MATRIX_INDICATOR_IONS:
            detection = detect_homologue_comb(processed, matrix_type=matrix_type)
            assert detection.n_clusters >= 5


class TestMarkerPreservation:
    """Das entscheidende Kriterium: der Abzug darf keinen Analyten zerstören."""

    @pytest.mark.parametrize("marker", CRITICAL_MARKERS)
    def test_marker_survives_the_subtraction(
        self, mixed_sample: SyntheticPyrogram, marker: str
    ) -> None:
        processed = _processed(mixed_sample)
        result = subtract_polymer_matrix(processed)

        contribution, window, quantifier = _true_marker_contribution(
            mixed_sample, marker
        )
        remaining = float(result.residual.eic(quantifier)[window].sum())
        assert remaining >= 0.9 * contribution, (
            f"{marker}: nur {remaining / contribution:.0%} des wahren Beitrags "
            "übrig — der Abzug frisst den Analyten"
        )

    def test_trace_pet_markers_survive_in_a_polyolefin_matrix(self) -> None:
        """Der Zielfall: 0,5 % PET, das nach Response-Korrektur bei 0,2 % liegt."""
        from tests.synthetic_data import RECIPES, SyntheticPyrogramGenerator

        sample = SyntheticPyrogramGenerator(
            seed=20240517, rt_end_s=1800.0, scan_rate_hz=2.0,
            mz_low=29, mz_high=320, carbon_range=(6, 20),
        ).generate(RECIPES["trace_pet_in_polyolefin"])
        result = subtract_polymer_matrix(_processed(sample))

        for marker in ("benzoic acid", "divinyl terephthalate"):
            contribution, window, quantifier = _true_marker_contribution(sample, marker)
            remaining = float(result.residual.eic(quantifier)[window].sum())
            assert remaining >= 0.9 * contribution

    def test_without_repair_the_subtraction_destroys_coeluting_markers(
        self, coelution_sample: SyntheticPyrogram
    ) -> None:
        """Zeigt, dass die Kontaminationsreparatur die Ursache und nicht Kosmetik ist.

        Ohne sie enthält das gemessene Clusterspektrum den koeluierenden Analyten,
        und der Abzug hebt ihn fast exakt auf.
        """
        processed = _processed(coelution_sample)
        repaired = subtract_polymer_matrix(processed, repair_contaminated=True)
        naive = subtract_polymer_matrix(processed, repair_contaminated=False)

        contribution, window, quantifier = _true_marker_contribution(
            coelution_sample, "epsilon-caprolactam"
        )
        with_repair = float(repaired.residual.eic(quantifier)[window].sum())
        without_repair = float(naive.residual.eic(quantifier)[window].sum())

        assert with_repair > without_repair
        assert without_repair < 0.8 * contribution
        assert with_repair >= 0.9 * contribution


class TestSubtractionEffect:
    def test_a_substantial_share_of_the_signal_is_attributed_to_the_matrix(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        result = subtract_polymer_matrix(_processed(mixed_sample))
        assert 0.3 < result.explained_fraction < 0.95

    def test_window_complexity_drops(self, mixed_sample: SyntheticPyrogram) -> None:
        """Der eigentliche Zweck: MCR-ALS ein lösbares Problem geben."""
        processed = _processed(mixed_sample)
        result = subtract_polymer_matrix(processed)

        def components_per_window(cube) -> list[int]:  # noqa: ANN001
            return [
                sum(
                    1
                    for component in mixed_sample.truth.components
                    if window.start_s <= component.retention_time_s <= window.end_s
                )
                for window in detect_windows(cube)
            ]

        before = components_per_window(processed)
        after = components_per_window(result.residual)
        assert np.median(after) <= np.median(before)
        assert max(after) < max(before)

    def test_residual_keeps_the_foreign_polymer_signal(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """Die Fremdpolymer-Kanäle dürfen im Residuum nicht ausgedünnt sein."""
        processed = _processed(mixed_sample)
        result = subtract_polymer_matrix(processed)
        for quantifier in (104.0, 113.0, 105.0, 149.0):
            before = float(processed.eic(quantifier).sum())
            after = float(result.residual.eic(quantifier).sum())
            assert after > 0.05 * before

    def test_matrix_and_residual_are_both_non_negative(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        result = subtract_polymer_matrix(_processed(mixed_sample))
        assert np.all(result.residual.intensities >= 0.0)
        assert np.all(result.matrix.intensities >= 0.0)

    def test_subtraction_is_recorded_in_the_audit_trail(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        result = subtract_polymer_matrix(_processed(mixed_sample))
        names = [step.name for step in result.residual.metadata.preprocessing]
        assert any(name.startswith("matrix_subtraction:") for name in names)

    def test_diagnostics_are_populated(self, mixed_sample: SyntheticPyrogram) -> None:
        result = subtract_polymer_matrix(_processed(mixed_sample))
        assert result.diagnostics["n_clusters"] >= 5
        assert result.diagnostics["cluster_spacing_s"] > 0.0
        assert "residual_fraction" in result.diagnostics


class TestMatrixModel:
    def test_amplitudes_are_non_negative(self, hdpe_sample: SyntheticPyrogram) -> None:
        """NNLS: eine negative Kammmenge wäre physikalisch unmöglich."""
        processed = _processed(hdpe_sample)
        model = build_matrix_model(processed, detect_homologue_comb(processed))
        assert np.all(model.amplitudes >= 0.0)

    def test_spectra_are_normalised_and_non_negative(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        processed = _processed(hdpe_sample)
        model = build_matrix_model(processed, detect_homologue_comb(processed))
        assert np.all(model.spectra >= 0.0)
        occupied = model.spectra.sum(axis=1) > 0
        assert np.allclose(model.spectra[occupied].sum(axis=1), 1.0)

    def test_profiles_have_unit_area(self, hdpe_sample: SyntheticPyrogram) -> None:
        processed = _processed(hdpe_sample)
        model = build_matrix_model(processed, detect_homologue_comb(processed))
        areas = np.trapezoid(model.profiles, processed.retention_times, axis=0)
        # Die Toleranz ist kein Nachgeben: das Profil wird über sein eigenes
        # Segment auf Fläche 1 normiert, hier aber über die volle Zeitachse
        # integriert, was an den Segmenträndern halbe Trapeze gegen Null addiert.
        assert np.allclose(areas[areas > 0], 1.0, rtol=1e-3)

    def test_amplitudes_follow_the_chain_length_distribution(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Die gefitteten Mengen sind bereits das Rohmaterial für MS3.

        Die Kettenlängenverteilung eines Polyolefins hat ein Maximum in der Mitte
        und fällt zu beiden Seiten ab — der Verlauf, aus dem später die
        Kettenlängen-Verschiebung des Degradations-Index berechnet wird.
        """
        processed = _processed(hdpe_sample)
        model = build_matrix_model(processed, detect_homologue_comb(processed))
        amplitudes = model.amplitudes
        peak_position = int(np.argmax(amplitudes))
        assert 0 < peak_position < amplitudes.size - 1

    def test_reconstruction_matches_the_subtracted_matrix(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        processed = _processed(hdpe_sample)
        result = subtract_polymer_matrix(processed)
        assert np.allclose(
            result.matrix.intensities,
            np.clip(result.model.reconstruct(), 0.0, None),
        )
