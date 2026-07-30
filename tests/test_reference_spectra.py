"""Tests for the EI reference spectra used by the simulator.

The synthetic pyrograms are only a valid benchmark if the spectra behind them are
chemically sensible. In particular the tests below pin the *discriminating*
features that the marker logic of later milestones will rely on: alkanes peak on
the alkyl series, alkenes on the alkenyl series, branched alkenes on the even-mass
ions, and oxidation products on their McLafferty ions.
"""

from __future__ import annotations

import pytest

from data_schemas.enums import MarkerRole, PolymerClass
from tests.reference_spectra import (
    REFERENCE_COMPOUNDS,
    REFERENCE_METHOD_WINDOW_S,
    SILOXANE_BLEED_SPECTRUM,
    alkadiene_spectrum,
    alkane_spectrum,
    alkanoic_acid_spectrum,
    alkanone_spectrum,
    alkene_spectrum,
    homologue_molecular_weight,
    homologue_spectrum,
    isoalkene_spectrum,
)

HOMOLOGUE_KINDS = ("alkane", "alkene", "diene", "isoalkene", "alkanone", "alkanal", "acid")


class TestHomologueSpectraWellFormed:
    @pytest.mark.parametrize("kind", HOMOLOGUE_KINDS)
    @pytest.mark.parametrize("carbon_number", [8, 12, 18, 26])
    def test_base_peak_is_normalised_to_one_hundred(self, kind: str, carbon_number: int) -> None:
        spectrum = homologue_spectrum(kind, carbon_number)
        assert max(spectrum.values()) == pytest.approx(100.0)

    @pytest.mark.parametrize("kind", HOMOLOGUE_KINDS)
    @pytest.mark.parametrize("carbon_number", [6, 10, 16, 24, 32])
    def test_no_fragment_heavier_than_the_molecular_ion(
        self, kind: str, carbon_number: int
    ) -> None:
        """A fragment cannot outweigh the molecule it came from.

        This is what makes short homologues distinguishable from long ones at all:
        the series is truncated at M, so C8 and C28 differ in their high-mass end
        even though their low-mass fragments are nearly identical.
        """
        if kind in {"diene"} and carbon_number < 4:
            pytest.skip("chain too short for this series")
        molecular_weight = homologue_molecular_weight(kind, carbon_number)
        spectrum = homologue_spectrum(kind, carbon_number)
        assert max(spectrum) <= molecular_weight

    @pytest.mark.parametrize("kind", HOMOLOGUE_KINDS)
    def test_all_intensities_are_positive(self, kind: str) -> None:
        spectrum = homologue_spectrum(kind, 14)
        assert all(intensity > 0.0 for intensity in spectrum.values())

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown homologue kind"):
            homologue_spectrum("polyester", 10)

    @pytest.mark.parametrize(
        ("builder", "minimum"),
        [
            (alkane_spectrum, 2),
            (alkene_spectrum, 3),
            (alkadiene_spectrum, 4),
            (isoalkene_spectrum, 6),
            (alkanone_spectrum, 4),
        ],
    )
    def test_chains_that_are_too_short_are_rejected(self, builder, minimum: int) -> None:  # noqa: ANN001
        with pytest.raises(ValueError):
            builder(minimum - 1)


class TestHomologueSpectraAreChemicallyDiscriminating:
    @pytest.mark.parametrize("carbon_number", [10, 16, 22, 30])
    def test_alkane_base_peak_is_an_alkyl_ion(self, carbon_number: int) -> None:
        """n-Alkanes give m/z 43 or 57 as base peak — the C3H7+/C4H9+ ions."""
        spectrum = alkane_spectrum(carbon_number)
        base_peak = max(spectrum, key=lambda mz: spectrum[mz])
        assert base_peak in {43, 57}

    @pytest.mark.parametrize("carbon_number", [10, 16, 22, 30])
    def test_alkene_base_peak_is_an_alkenyl_ion(self, carbon_number: int) -> None:
        """1-Alkenes give m/z 41 or 55 — allylic cleavage, not alkyl cleavage."""
        spectrum = alkene_spectrum(carbon_number)
        base_peak = max(spectrum, key=lambda mz: spectrum[mz])
        assert base_peak in {41, 55}

    @pytest.mark.parametrize("carbon_number", [12, 18, 24])
    def test_alkene_and_alkane_are_separable_by_the_55_over_57_ratio(
        self, carbon_number: int
    ) -> None:
        """The ratio that the degradation index is built on must actually differ.

        A 1-alkene is alkenyl-dominated and an n-alkane alkyl-dominated, so
        ``I(55)/I(57)`` must be well above 1 for the alkene and well below 1 for the
        alkane. Without that contrast, no alkene/alkane ratio could be measured.
        """
        alkene = alkene_spectrum(carbon_number)
        alkane = alkane_spectrum(carbon_number)
        alkene_ratio = alkene[55] / alkene[57]
        alkane_ratio = alkane[55] / alkane[57]
        assert alkene_ratio > 2.0
        assert alkane_ratio < 0.6
        assert alkene_ratio > 4.0 * alkane_ratio

    @pytest.mark.parametrize("carbon_number", [9, 12, 15, 18])
    def test_isoalkene_favours_even_mass_ions_over_the_linear_alkene(
        self, carbon_number: int
    ) -> None:
        """Branching diverts allylic cleavage to the even-mass ions (56, 70, 84).

        This is the mechanism behind the branching index that separates PP and LDPE
        from HDPE, so the simulated iso-alkene has to show it.
        """
        branched = isoalkene_spectrum(carbon_number)
        linear = alkene_spectrum(carbon_number)
        branched_even = sum(branched.get(mz, 0.0) for mz in (56, 70, 84))
        linear_even = sum(linear.get(mz, 0.0) for mz in (56, 70, 84))
        assert branched_even > linear_even

    def test_diene_carries_its_diagnostic_odd_series(self) -> None:
        spectrum = alkadiene_spectrum(14)
        assert spectrum.get(67, 0.0) > 0.0
        assert spectrum.get(81, 0.0) > 0.0

    def test_ketone_is_dominated_by_the_mclafferty_ion(self) -> None:
        spectrum = alkanone_spectrum(12)
        assert max(spectrum, key=lambda mz: spectrum[mz]) == 58
        assert spectrum[43] > 50.0

    def test_acid_is_dominated_by_its_sixty_seventy_three_pair(self) -> None:
        spectrum = alkanoic_acid_spectrum(14)
        assert max(spectrum, key=lambda mz: spectrum[mz]) == 60
        assert spectrum[73] > 50.0

    @pytest.mark.parametrize("carbon_number", [10, 14, 20])
    def test_oxidation_products_are_selective_against_hydrocarbons(
        self, carbon_number: int
    ) -> None:
        """m/z 44/58/60 must be essentially absent from the hydrocarbon comb.

        Trace carbonyl detection on a huge polyolefin background only works if these
        channels are quiet in the matrix. If the simulated alkanes put signal there,
        the degradation index would be testable but meaningless.
        """
        diagnostic = (44, 58, 60, 73)
        for hydrocarbon in (
            alkane_spectrum(carbon_number),
            alkene_spectrum(carbon_number),
            alkadiene_spectrum(carbon_number),
        ):
            assert all(hydrocarbon.get(mz, 0.0) == 0.0 for mz in diagnostic)

    @pytest.mark.parametrize("carbon_number", [12, 20])
    def test_adjacent_homologues_are_nearly_collinear(self, carbon_number: int) -> None:
        """Neighbouring chain lengths must be hard to tell apart.

        This is the central numerical difficulty of resolving a polyolefin matrix.
        The test asserts the difficulty is present: the cosine similarity between
        C(n) and C(n+1) alkanes has to be very high, otherwise the benchmark is
        easier than reality and MCR-ALS would pass it for the wrong reason.
        """
        first = alkane_spectrum(carbon_number)
        second = alkane_spectrum(carbon_number + 1)
        shared = set(first) | set(second)
        dot = sum(first.get(mz, 0.0) * second.get(mz, 0.0) for mz in shared)
        norm_first = sum(value * value for value in first.values()) ** 0.5
        norm_second = sum(value * value for value in second.values()) ** 0.5
        cosine = dot / (norm_first * norm_second)
        assert cosine > 0.98


class TestReferenceCompoundRegistry:
    def test_registry_is_not_empty_and_keys_match_names(self) -> None:
        assert len(REFERENCE_COMPOUNDS) > 20
        for key, compound in REFERENCE_COMPOUNDS.items():
            assert key == compound.name

    @pytest.mark.parametrize("name", sorted(REFERENCE_COMPOUNDS))
    def test_spectrum_is_well_formed(self, name: str) -> None:
        compound = REFERENCE_COMPOUNDS[name]
        assert compound.spectrum, f"{name} has an empty spectrum"
        assert max(compound.spectrum.values()) == pytest.approx(100.0), (
            f"{name} is not base-peak normalised"
        )
        assert all(intensity > 0.0 for intensity in compound.spectrum.values())
        assert max(compound.spectrum) <= compound.molecular_weight + 2, (
            f"{name} has a fragment above its molecular ion"
        )

    @pytest.mark.parametrize("name", sorted(REFERENCE_COMPOUNDS))
    def test_quantifier_ion_exists_in_the_spectrum(self, name: str) -> None:
        """A quantifier ion that is not in the spectrum would integrate noise."""
        compound = REFERENCE_COMPOUNDS[name]
        assert compound.quantifier_mz in compound.spectrum

    @pytest.mark.parametrize("name", sorted(REFERENCE_COMPOUNDS))
    def test_retention_time_lies_inside_the_reference_method(self, name: str) -> None:
        start, end = REFERENCE_METHOD_WINDOW_S
        compound = REFERENCE_COMPOUNDS[name]
        assert start < compound.retention_time_s < end

    @pytest.mark.parametrize("name", sorted(REFERENCE_COMPOUNDS))
    def test_peak_shape_scales_are_physical(self, name: str) -> None:
        compound = REFERENCE_COMPOUNDS[name]
        assert compound.peak_width_scale >= 1.0
        assert compound.tailing_scale >= 1.0

    def test_polar_compounds_tail_more_than_apolar_ones(self) -> None:
        """Acids and lactams tail badly on a 5 %-phenyl phase; aromatics do not."""
        polar = ("octadecanoic acid (stearic acid)", "epsilon-caprolactam", "benzoic acid")
        apolar = ("benzene", "toluene", "naphthalene", "styrene")
        worst_apolar = max(REFERENCE_COMPOUNDS[name].tailing_scale for name in apolar)
        best_polar = min(REFERENCE_COMPOUNDS[name].tailing_scale for name in polar)
        assert best_polar > worst_apolar

    def test_styrene_triad_is_present_with_increasing_retention(self) -> None:
        """Monomer, dimer and trimer must elute in oligomer order."""
        monomer = REFERENCE_COMPOUNDS["styrene"]
        dimer = REFERENCE_COMPOUNDS["2,4-diphenyl-1-butene"]
        trimer = REFERENCE_COMPOUNDS["2,4,6-triphenyl-1-hexene"]
        assert monomer.role is MarkerRole.MONOMER
        assert dimer.role is MarkerRole.DIMER
        assert trimer.role is MarkerRole.TRIMER
        assert monomer.retention_time_s < dimer.retention_time_s < trimer.retention_time_s

    def test_every_polymer_of_interest_has_at_least_one_marker(self) -> None:
        covered = {compound.polymer_class for compound in REFERENCE_COMPOUNDS.values()}
        required = {
            PolymerClass.PS,
            PolymerClass.PP,
            PolymerClass.PET,
            PolymerClass.PA6,
            PolymerClass.PVC,
            PolymerClass.PC,
            PolymerClass.PMMA,
            PolymerClass.NON_POLYMERIC,
        }
        assert required <= covered

    def test_additive_roles_are_tagged_as_additives(self) -> None:
        additives = [
            compound
            for compound in REFERENCE_COMPOUNDS.values()
            if compound.polymer_class is PolymerClass.NON_POLYMERIC
        ]
        assert additives
        assert all(compound.role.is_additive for compound in additives)

    def test_registry_is_read_only(self) -> None:
        """The table is shared process-wide; a test must not be able to corrupt it."""
        with pytest.raises(TypeError):
            REFERENCE_COMPOUNDS["benzene"] = REFERENCE_COMPOUNDS["toluene"]  # type: ignore[index]


class TestColumnBleed:
    def test_bleed_sits_on_the_siloxane_ions(self) -> None:
        assert set(SILOXANE_BLEED_SPECTRUM) == {73, 147, 207, 221, 281, 355}

    def test_bleed_is_base_peak_normalised(self) -> None:
        assert max(SILOXANE_BLEED_SPECTRUM.values()) == pytest.approx(100.0)

    def test_bleed_interferes_with_the_phthalate_and_pet_quantifier(self) -> None:
        """m/z 147 bleed sits next to the m/z 149 phthalate/PET quantifier.

        That adjacency is the reason baseline correction has to be per-channel; the
        test documents that the simulated interference is really there.
        """
        assert 147 in SILOXANE_BLEED_SPECTRUM
        assert REFERENCE_COMPOUNDS["divinyl terephthalate"].quantifier_mz == 149
        assert REFERENCE_COMPOUNDS["bis(2-ethylhexyl) phthalate (DEHP)"].quantifier_mz == 149
