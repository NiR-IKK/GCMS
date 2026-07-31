"""Tests für den DegradationEngine.

Die Kennzahlen müssen zwei Dinge leisten: mit dem echten Alterungsgrad **monoton**
steigen (sonst sind sie kein Maß), und LDPE von HDPE trennen (sonst ist der
Verzweigungsindex keiner). Beides wird gegen den bekannten Alterungsgrad des
Generators geprüft.

Ein Test hält eine Anwendungsgrenze fest: der Säureanteil ist auf kaum oxidiertem
Material untergrundlimitiert und läuft dort gegen 1/3, unabhängig von der Chemie.
Die Kennzahl meldet das selbst, statt eine bedeutungslose Zahl auszugeben.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from data_schemas.enums import PolymerClass
from pyrecycle_analytics.degradation import (
    HYDROCARBON_IONS,
    OXIDATION_IONS,
    DegradationIndices,
    chain_length_statistics,
    compute_degradation_indices,
)
from pyrecycle_analytics.matrix import subtract_polymer_matrix
from pyrecycle_analytics.preprocessing import PreprocessingConfig, preprocess
from tests.synthetic_data import (
    RECIPES,
    PolymerFraction,
    PyrogramRecipe,
    SyntheticPyrogram,
    SyntheticPyrogramGenerator,
)

AGEING_LEVELS = (0.0, 0.25, 0.5, 0.75, 0.95)


@pytest.fixture(scope="module")
def ageing_generator() -> SyntheticPyrogramGenerator:
    return SyntheticPyrogramGenerator(
        seed=20240517, rt_end_s=1800.0, scan_rate_hz=2.0,
        mz_low=29, mz_high=320, carbon_range=(6, 20),
    )


@pytest.fixture(scope="module")
def ageing_series(ageing_generator: SyntheticPyrogramGenerator) -> list[DegradationIndices]:
    """Indices of the same polymer across a range of ageing levels."""
    config = PreprocessingConfig(asls_iterations=8)
    series: list[DegradationIndices] = []
    for level in AGEING_LEVELS:
        recipe = PyrogramRecipe(
            name=f"aged_{level}",
            fractions=(PolymerFraction(PolymerClass.PE_HD, 1.0, level),),
        )
        processed = preprocess(ageing_generator.generate(recipe).cube, config)
        series.append(compute_degradation_indices(processed))
    return series


class TestMonotonicity:
    def test_carbonyl_index_tracks_the_true_ageing_level(
        self, ageing_series: list[DegradationIndices]
    ) -> None:
        """Die Hauptkennzahl muss streng mit dem echten Alterungsgrad steigen."""
        values = [indices.carbonyl_index for indices in ageing_series]
        assert spearmanr(AGEING_LEVELS, values).statistic == pytest.approx(1.0)
        assert values[-1] > 3.0 * values[0]

    def test_alkene_ratio_rises_with_chain_scission(
        self, ageing_series: list[DegradationIndices]
    ) -> None:
        values = [indices.alkene_to_alkane for indices in ageing_series]
        assert spearmanr(AGEING_LEVELS, values).statistic > 0.9

    def test_branching_index_rises_with_ageing(
        self, ageing_series: list[DegradationIndices]
    ) -> None:
        """Sekundärer Radikalangriff erzeugt neue Verzweigungspunkte."""
        values = [indices.branching_index for indices in ageing_series]
        assert spearmanr(AGEING_LEVELS, values).statistic > 0.9

    def test_indices_are_bounded_and_finite(
        self, ageing_series: list[DegradationIndices]
    ) -> None:
        for indices in ageing_series:
            assert 0.0 <= indices.acid_share <= 1.0
            assert indices.carbonyl_index >= 0.0
            assert np.isfinite(indices.alkene_to_alkane)


class TestGradeDiscrimination:
    def test_branching_index_separates_ldpe_from_hdpe(
        self, ageing_generator: SyntheticPyrogramGenerator
    ) -> None:
        """Beide Sorten geben denselben Alkankamm; nur die Verzweigung trennt sie."""
        config = PreprocessingConfig(asls_iterations=8)
        hdpe = compute_degradation_indices(
            preprocess(ageing_generator.generate(RECIPES["virgin_hdpe"]).cube, config)
        )
        ldpe = compute_degradation_indices(
            preprocess(ageing_generator.generate(RECIPES["virgin_ldpe"]).cube, config)
        )
        assert ldpe.branching_index > 1.2 * hdpe.branching_index

    def test_polypropylene_is_far_more_branched_than_any_polyethylene(
        self, ageing_generator: SyntheticPyrogramGenerator
    ) -> None:
        config = PreprocessingConfig(asls_iterations=8)
        polypropylene = compute_degradation_indices(
            preprocess(ageing_generator.generate(RECIPES["virgin_pp"]).cube, config)
        )
        hdpe = compute_degradation_indices(
            preprocess(ageing_generator.generate(RECIPES["virgin_hdpe"]).cube, config)
        )
        assert polypropylene.branching_index > 3.0 * hdpe.branching_index


class TestChainLength:
    def test_mean_chain_length_falls_with_ageing(
        self, ageing_generator: SyntheticPyrogramGenerator
    ) -> None:
        """Wiederholte Verarbeitung verkürzt die Ketten."""
        config = PreprocessingConfig(asls_iterations=8)
        means: list[float] = []
        for level in (0.0, 0.5, 0.95):
            recipe = PyrogramRecipe(
                name=f"chain_{level}",
                fractions=(PolymerFraction(PolymerClass.PE_HD, 1.0, level),),
            )
            processed = preprocess(ageing_generator.generate(recipe).cube, config)
            model = subtract_polymer_matrix(processed).model
            indices = compute_degradation_indices(processed, matrix_model=model)
            assert indices.mean_carbon_number is not None
            means.append(indices.mean_carbon_number)
        assert means == sorted(means, reverse=True)

    def test_chain_statistics_come_from_the_matrix_model(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Die in MS2.2 gefitteten Amplituden *sind* die Kettenlängenverteilung."""
        processed = preprocess(
            hdpe_sample.cube, PreprocessingConfig(asls_iterations=8)
        )
        model = subtract_polymer_matrix(processed).model
        mean, spread = chain_length_statistics(model)
        assert 0.0 < mean < model.detection.n_clusters
        assert spread > 0.0

    def test_absent_without_a_model(self, hdpe_sample: SyntheticPyrogram) -> None:
        processed = preprocess(
            hdpe_sample.cube, PreprocessingConfig(asls_iterations=8)
        )
        indices = compute_degradation_indices(processed)
        assert indices.mean_carbon_number is None
        assert indices.chain_length_spread is None


class TestApplicabilityLimits:
    def test_acid_share_reports_its_own_background_limit(
        self, ageing_generator: SyntheticPyrogramGenerator
    ) -> None:
        """Auf kaum oxidiertem Material ist der Säureanteil bedeutungslos.

        Alle drei Carbonyl-Kanäle lesen dort überwiegend Untergrund, und der
        Anteil läuft gegen 1/3 unabhängig von der Chemie — gemessen exakt 0,333
        auf jungfräulichem PE. Die Kennzahl sagt das selbst.
        """
        recipe = PyrogramRecipe(
            name="virgin_for_acid",
            fractions=(PolymerFraction(PolymerClass.PE_HD, 1.0, 0.0),),
        )
        indices = compute_degradation_indices(
            preprocess(
                ageing_generator.generate(recipe).cube, PreprocessingConfig(asls_iterations=8)
            )
        )
        assert any("not informative" in note for note in indices.notes)

    def test_every_result_carries_the_reference_caveat(
        self, ageing_series: list[DegradationIndices]
    ) -> None:
        """Absolutwerte sind ohne Virgin-Referenz nicht interpretierbar."""
        for indices in ageing_series:
            assert any("virgin reference" in note for note in indices.notes)

    def test_empty_pyrogram_is_rejected(self, hdpe_sample: SyntheticPyrogram) -> None:
        empty = hdpe_sample.cube.with_intensities(np.zeros(hdpe_sample.cube.shape))
        with pytest.raises(ValueError, match="empty pyrogram"):
            compute_degradation_indices(empty)


class TestRelativeReporting:
    def test_ratios_against_a_virgin_reference(
        self, ageing_series: list[DegradationIndices]
    ) -> None:
        virgin, aged = ageing_series[0], ageing_series[-1]
        ratios = aged.relative_to(virgin)
        assert ratios["carbonyl_index"] > 3.0
        assert ratios["alkene_to_alkane"] > 1.0

    def test_comparing_incompatible_bases_is_refused(
        self, ageing_series: list[DegradationIndices]
    ) -> None:
        from dataclasses import replace

        other = replace(ageing_series[0], basis="resolved-components")
        with pytest.raises(ValueError, match="different bases"):
            ageing_series[-1].relative_to(other)

    def test_zero_reference_gives_infinity_not_a_crash(self) -> None:
        zero = DegradationIndices(0.0, 0.0, 0.0, 0.0, None, None, "ion-signal")
        nonzero = DegradationIndices(0.1, 0.2, 0.3, 0.4, None, None, "ion-signal")
        assert nonzero.relative_to(zero)["carbonyl_index"] == float("inf")
        assert zero.relative_to(zero)["carbonyl_index"] == 1.0

    def test_summary_is_json_serialisable(
        self, ageing_series: list[DegradationIndices]
    ) -> None:
        import json

        json.dumps(ageing_series[0].summary())


class TestDiagnosticIons:
    def test_acid_channel_excludes_the_column_bleed_ion(self) -> None:
        """m/z 73 ist zugleich das Siloxan-Bluten-Ion und darf nicht hinein.

        Das klassische Säurepaar ist 60 und 73. Auf einer Polysiloxan-Säule ist 73
        aber das dominierende Bluten-Ion; es mitzunehmen ließ den gemessenen
        Säureanteil mit der Alterung *fallen*, wo der wahre steigt.
        """
        assert 73.0 not in OXIDATION_IONS["acid"]
        assert OXIDATION_IONS["acid"] == (60.0,)

    def test_oxidation_and_hydrocarbon_channels_do_not_overlap(self) -> None:
        """Zähler und Nenner eines Verhältnisses dürfen kein Ion teilen."""
        oxidation = {ion for ions in OXIDATION_IONS.values() for ion in ions}
        hydrocarbon = {ion for ions in HYDROCARBON_IONS.values() for ion in ions}
        assert not (oxidation & hydrocarbon)
