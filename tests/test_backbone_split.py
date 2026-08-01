"""Tests für die Aufteilung des Polyolefin-Kamms zwischen PE und PP.

Der Kamm ist die dominante Fraktion eines Polyolefin-Rezyklats und wird vor der
Kurvenauflösung abgezogen. Ohne Aufteilung meldet der Pass ihn als *ein* Polymer,
und ein PE/PP-Blend liest sich als eine mittlere Sorte — gemessen PP bei 5,8 %
gegen 31 % Wahrheit.

Geprüft werden drei Dinge, und das dritte ist das wichtigste:

1. Reine Proben werden ihrem Material zugeordnet, Mischungen im richtigen
   Verhältnis.
2. Die Aufteilung verweigert sich, wenn die Referenzen nicht passen — lieber
   keine Zahl als eine erfundene.
3. Schlecht trennbare Referenzen werden als solche ausgewiesen und nicht als
   zwei getrennte Prozentzahlen dargestellt.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_schemas.enums import PolymerClass
from pyrecycle_analytics.matrix.backbone import (
    COLLINEARITY_LIMIT,
    BackboneEndmembers,
    BackboneSplitError,
    comb_spectrum,
    split_backbone,
)
from pyrecycle_analytics.matrix.polyolefin import (
    build_matrix_model,
    detect_homologue_comb,
)
from pyrecycle_analytics.preprocessing import PreprocessingConfig, preprocess
from tests.synthetic_data import RECIPES, SyntheticPyrogram, SyntheticPyrogramGenerator

CONFIG = PreprocessingConfig(asls_iterations=8)


def _model(sample: SyntheticPyrogram):  # noqa: ANN202
    cube = preprocess(sample.cube, CONFIG)
    return build_matrix_model(cube, detect_homologue_comb(cube)), cube


def _true_polyolefin_split(sample: SyntheticPyrogram) -> float:
    """Wahrer PP-Anteil am Polyolefin, flächenbezogen."""
    areas = sample.truth.area_by_polymer()
    polyethylene = sum(v for k, v in areas.items() if "PE" in str(k))
    polypropylene = sum(v for k, v in areas.items() if str(k) == "PP")
    total = polyethylene + polypropylene
    return polypropylene / total if total > 0.0 else 0.0


@pytest.fixture(scope="module")
def wide_generator() -> SyntheticPyrogramGenerator:
    """Ein Lauf, der den Kamm vollständig erfasst.

    Nicht die reduzierte Testmethode aus ``conftest``: die bricht bei C20 ab, und
    die Genauigkeit der Aufteilung hängt gemessen davon ab, wie viel des Kamms im
    Messfenster liegt (siehe :class:`TestAcquisitionWindow`). Die Zusicherungen
    hier gelten für ein Ofenprogramm, das die Reihe ausfahren lässt — was die
    Methode ohnehin verlangt.
    """
    return SyntheticPyrogramGenerator(seed=20240517)


@pytest.fixture(scope="module")
def references(wide_generator: SyntheticPyrogramGenerator) -> dict:  # noqa: ANN201
    return {
        PolymerClass.PE_HD: preprocess(
            wide_generator.generate(RECIPES["virgin_hdpe"]).cube, CONFIG
        ),
        PolymerClass.PE_LD: preprocess(
            wide_generator.generate(RECIPES["virgin_ldpe"]).cube, CONFIG
        ),
        PolymerClass.PP: preprocess(
            wide_generator.generate(RECIPES["virgin_pp"]).cube, CONFIG
        ),
    }


@pytest.fixture(scope="module")
def endmembers(references: dict) -> BackboneEndmembers:
    return BackboneEndmembers.from_references(references)


@pytest.fixture(scope="module")
def wide_blend(wide_generator: SyntheticPyrogramGenerator) -> SyntheticPyrogram:
    return wide_generator.generate(RECIPES["pcr_mixed_polyolefin"])


class TestCombSpectrum:
    def test_spectrum_is_normalised(self, hdpe_sample: SyntheticPyrogram) -> None:
        model, _ = _model(hdpe_sample)
        assert comb_spectrum(model).sum() == pytest.approx(1.0)

    def test_amplitude_weighting_is_not_a_plain_mean(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Die Cluster unterscheiden sich um Größenordnungen in der Größe.

        Ein ungewichtetes Mittel ließe den kleinsten, verrauschtesten Cluster so
        stark zählen wie das Maximum der Verteilung.
        """
        model, _ = _model(hdpe_sample)
        weighted = comb_spectrum(model)
        plain = model.spectra.mean(axis=0)
        plain = plain / plain.sum()
        assert not np.allclose(weighted, plain, atol=1e-4)

    def test_an_empty_model_is_rejected(self, hdpe_sample: SyntheticPyrogram) -> None:
        model, _ = _model(hdpe_sample)
        model.amplitudes = np.zeros_like(model.amplitudes)
        with pytest.raises(BackboneSplitError, match="no signal"):
            comb_spectrum(model)


class TestEndmembers:
    def test_references_become_normalised_spectra(
        self, endmembers: BackboneEndmembers
    ) -> None:
        assert endmembers.n_materials == 3
        assert endmembers.spectra.shape[1] == endmembers.mz_axis.size
        assert np.allclose(endmembers.spectra.sum(axis=1), 1.0)

    def test_polyethylene_grades_are_flagged_as_inseparable(
        self, endmembers: BackboneEndmembers
    ) -> None:
        """Der Befund, der die Berichtsform bestimmt.

        HDPE und LDPE unterscheiden sich spektral um einen Kosinus von rund
        0,999. Diese Richtung trägt praktisch keine unabhängige Information, und
        zwei Prozentzahlen daraus wären eine Scheingenauigkeit.
        """
        pairs = endmembers.collinear_pairs()
        assert len(pairs) == 1
        first, second, cosine = pairs[0]
        assert {first, second} == {PolymerClass.PE_HD, PolymerClass.PE_LD}
        assert cosine > COLLINEARITY_LIMIT

    def test_polypropylene_is_separable_from_both_grades(
        self, endmembers: BackboneEndmembers
    ) -> None:
        """Die Gegenprobe: PP ist es nicht, sonst wäre die Aufteilung sinnlos."""
        collinear = {
            frozenset((first, second)) for first, second, _ in endmembers.collinear_pairs()
        }
        assert frozenset((PolymerClass.PE_HD, PolymerClass.PP)) not in collinear
        assert frozenset((PolymerClass.PE_LD, PolymerClass.PP)) not in collinear

    def test_a_single_reference_is_rejected(self, references: dict) -> None:
        with pytest.raises(BackboneSplitError, match="at least two"):
            BackboneEndmembers.from_references(
                {PolymerClass.PP: references[PolymerClass.PP]}
            )

    def test_references_on_different_axes_are_rejected(
        self, references: dict, generator: SyntheticPyrogramGenerator
    ) -> None:
        narrow = preprocess(
            generator.generate(RECIPES["virgin_pp"]).cube, CONFIG
        ).window(mz_range=(40.0, 200.0))
        with pytest.raises(BackboneSplitError, match="different m/z axis"):
            BackboneEndmembers.from_references(
                {
                    PolymerClass.PE_LD: references[PolymerClass.PE_LD],
                    PolymerClass.PP: narrow,
                }
            )

    def test_a_sample_without_a_comb_cannot_be_a_reference(
        self, references: dict, wide_generator: SyntheticPyrogramGenerator
    ) -> None:
        polystyrene = preprocess(
            wide_generator.generate(RECIPES["virgin_ps"]).cube, CONFIG
        )
        with pytest.raises(BackboneSplitError, match="no polyolefin comb"):
            BackboneEndmembers.from_references(
                {
                    PolymerClass.PE_LD: references[PolymerClass.PE_LD],
                    PolymerClass.PS: polystyrene,
                }
            )


class TestSplitAccuracy:
    """Die Zahlen, wegen derer das Verfahren gebaut wurde."""

    @pytest.mark.parametrize(
        ("recipe", "expected"),
        [("virgin_hdpe", 0.0), ("virgin_ldpe", 0.0), ("virgin_pp", 1.0)],
    )
    def test_pure_samples_are_assigned_to_their_material(
        self,
        wide_generator: SyntheticPyrogramGenerator,
        endmembers: BackboneEndmembers,
        recipe: str,
        expected: float,
    ) -> None:
        sample = wide_generator.generate(RECIPES[recipe])
        model, cube = _model(sample)
        split = split_backbone(model, endmembers, cube.mz_axis)
        assert split.share_of(PolymerClass.PP) == pytest.approx(expected, abs=0.02)

    def test_a_blend_is_split_in_the_right_ratio(
        self, wide_blend: SyntheticPyrogram, endmembers: BackboneEndmembers
    ) -> None:
        """Gemessen 36,1 % gegen 33,9 % Wahrheit.

        Das ist die Zahl, um derentwillen das Verfahren gebaut wurde: der
        Verzweigungsindex allein setzte PP hier auf 11 % des Polyolefins.
        """
        model, cube = _model(wide_blend)
        split = split_backbone(model, endmembers, cube.mz_axis)
        assert split.share_of(PolymerClass.PP) == pytest.approx(
            _true_polyolefin_split(wide_blend), abs=0.04
        )

    def test_a_minor_polypropylene_fraction_is_found(
        self, wide_generator: SyntheticPyrogramGenerator, endmembers: BackboneEndmembers
    ) -> None:
        """PP bei knapp 5 % neben 95 % PE — der Fall, den der Index verfehlte."""
        sample = wide_generator.generate(RECIPES["trace_pet_in_polyolefin"])
        model, cube = _model(sample)
        split = split_backbone(model, endmembers, cube.mz_axis)
        assert split.share_of(PolymerClass.PP) == pytest.approx(
            _true_polyolefin_split(sample), abs=0.03
        )

    def test_shares_are_a_distribution(
        self, wide_blend: SyntheticPyrogram, endmembers: BackboneEndmembers
    ) -> None:
        model, cube = _model(wide_blend)
        split = split_backbone(model, endmembers, cube.mz_axis)
        assert sum(split.shares.values()) == pytest.approx(1.0)
        assert all(share >= 0.0 for share in split.shares.values())
        assert split.dominant is PolymerClass.PE_LD

    def test_the_unexplained_share_is_reported(
        self, wide_blend: SyntheticPyrogram, endmembers: BackboneEndmembers
    ) -> None:
        """Ohne diese Zahl wäre nicht erkennbar, ob die Referenzen überhaupt passen."""
        model, cube = _model(wide_blend)
        split = split_backbone(model, endmembers, cube.mz_axis)
        assert 0.0 <= split.residual_fraction < 0.35
        assert any("unexplained" in note for note in split.notes)


class TestAcquisitionWindow:
    """Ein abgeschnittenes Messfenster verzerrt die Aufteilung Richtung PP.

    Physikalisch erwartbar: der PE-Kamm reicht bis zu hohen Kettenlängen, das
    PP-Pyrolysat sammelt sich bei niedrigen (Trimer C9, Tetramer C12). Ein
    Ofenprogramm, das bei C20 endet, schneidet anteilig mehr PE als PP weg.

    Der Test hält das fest, weil es eine Anforderung an die Methode ist und nicht
    an den Code: die Reihe muss ausgefahren werden, sonst ist das PE/PP-Verhältnis
    systematisch zu PP verschoben.
    """

    def test_a_truncated_run_overstates_polypropylene(
        self, endmembers: BackboneEndmembers, wide_blend: SyntheticPyrogram
    ) -> None:
        truncated_generator = SyntheticPyrogramGenerator(
            seed=20240517,
            rt_start_s=60.0,
            rt_end_s=1800.0,
            scan_rate_hz=2.0,
            mz_low=29,
            mz_high=320,
            carbon_range=(6, 20),
        )
        truncated = truncated_generator.generate(RECIPES["pcr_mixed_polyolefin"])

        wide_model, wide_cube = _model(wide_blend)
        cut_model, cut_cube = _model(truncated)
        wide_pp = split_backbone(
            wide_model, endmembers, wide_cube.mz_axis
        ).share_of(PolymerClass.PP)
        cut_pp = split_backbone(
            cut_model, endmembers, cut_cube.mz_axis
        ).share_of(PolymerClass.PP)

        truth = _true_polyolefin_split(wide_blend)
        assert abs(wide_pp - truth) < 0.04, "der volle Lauf muss der genaue sein"
        assert cut_pp > wide_pp + 0.03, (
            "wenn der Abbruch bei C20 die Aufteilung nicht mehr verzerrt, ist die "
            "Warnung in der Dokumentation zu streichen"
        )


class TestSplitRefusesRatherThanGuesses:
    def test_references_that_do_not_describe_the_comb_are_refused(
        self, mixed_sample: SyntheticPyrogram, endmembers: BackboneEndmembers
    ) -> None:
        """Eine Referenz aus einer anderen Methode darf keine Zahl liefern.

        Simuliert durch eine sehr enge Toleranz: das Verfahren muss sich
        verweigern, statt die beste schlechte Anpassung als Ergebnis auszugeben.
        """
        model, cube = _model(mixed_sample)
        with pytest.raises(BackboneSplitError, match="explain only"):
            split_backbone(
                model, endmembers, cube.mz_axis, max_residual_fraction=0.001
            )

    def test_disjoint_mass_ranges_are_refused(
        self, mixed_sample: SyntheticPyrogram, endmembers: BackboneEndmembers
    ) -> None:
        model, cube = _model(mixed_sample)
        shifted = BackboneEndmembers(
            spectra=endmembers.spectra,
            polymers=endmembers.polymers,
            mz_axis=endmembers.mz_axis + 1000.0,
        )
        with pytest.raises(BackboneSplitError, match="no m/z channel"):
            split_backbone(model, shifted, cube.mz_axis)

    def test_a_narrower_sample_axis_still_works(
        self, mixed_sample: SyntheticPyrogram, endmembers: BackboneEndmembers
    ) -> None:
        """Teilüberlappung ist zulässig, solange genug Kanäle gemeinsam sind."""
        cube = preprocess(mixed_sample.cube, CONFIG).window(mz_range=(40.0, 200.0))
        model = build_matrix_model(cube, detect_homologue_comb(cube))
        split = split_backbone(model, endmembers, cube.mz_axis)
        assert sum(split.shares.values()) == pytest.approx(1.0)
