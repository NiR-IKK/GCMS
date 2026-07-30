"""Tests for the synthetic pyrogram generator.

The generator is the measuring instrument of this project: every later claim about
deconvolution quality, marker recovery or degradation indexing is a comparison
against its ground truth. So it needs testing at two levels.

**Mathematical contract** — the properties the chemometric engines will rely on:
the clean signal is exactly ``C @ S``, each column of ``C`` integrates to the
declared area, spectra sum to one over the acquired window, and the same seed
reproduces the same cube in a different process.

**Chemical realism** — the properties that make the benchmark meaningful rather
than merely well-defined: PE really does produce fused alkane/alkene/diene
triplets, LDPE really is more branched than HDPE, ageing really does raise the
carbonyl content, response factors really do decouple mass fraction from signal
fraction, and the intended co-elutions really are unresolved.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_schemas.enums import MarkerRole, PolymerClass, RecyclateStream, SourceFormat
from data_schemas.truth import DriftTruth
from tests.synthetic_data import (
    POLYMER_RESPONSE_FACTORS,
    RECIPES,
    AdditiveSpike,
    AlkaneRetentionModel,
    BaselineModel,
    NoiseModel,
    PolymerFraction,
    PyrogramRecipe,
    SyntheticPyrogram,
    SyntheticPyrogramGenerator,
    apply_retention_drift,
)

# Any peak pair below this resolution is fused and needs curve resolution.
COELUTION_RESOLUTION = 1.0


def _integrate(profile: np.ndarray, retention_times: np.ndarray) -> float:
    return float(np.trapezoid(profile, retention_times))


class TestBilinearContract:
    """The model MCR-ALS assumes must hold exactly in the noiseless case."""

    def test_noiseless_cube_equals_c_times_s_exactly(
        self, noiseless_sample: SyntheticPyrogram
    ) -> None:
        """``D == C @ S`` to machine precision.

        This is what makes the ground truth a *factorisation* rather than an
        approximation: a curve-resolution result can be compared to C and S
        directly, with no modelling error in between.
        """
        assert np.allclose(
            noiseless_sample.cube.intensities, noiseless_sample.C @ noiseless_sample.S, atol=1e-9
        )

    def test_noiseless_cube_has_no_baseline(
        self, noiseless_sample: SyntheticPyrogram
    ) -> None:
        assert not noiseless_sample.baseline.any()
        assert not noiseless_sample.truth.baseline_included

    def test_spectra_rows_sum_to_one(self, mixed_sample: SyntheticPyrogram) -> None:
        """L1 normalisation is what makes a column of ``C`` the component's TIC."""
        assert np.allclose(mixed_sample.S.sum(axis=1), 1.0)

    def test_spectra_are_non_negative(self, mixed_sample: SyntheticPyrogram) -> None:
        assert np.all(mixed_sample.S >= 0.0)

    def test_profiles_are_non_negative(self, mixed_sample: SyntheticPyrogram) -> None:
        assert np.all(mixed_sample.C >= 0.0)

    def test_factor_shapes_are_consistent(self, mixed_sample: SyntheticPyrogram) -> None:
        n_scans, n_mz = mixed_sample.cube.shape
        n_components = mixed_sample.truth.n_components
        assert mixed_sample.C.shape == (n_scans, n_components)
        assert mixed_sample.S.shape == (n_components, n_mz)
        assert len(mixed_sample.component_names) == n_components

    def test_profile_column_integrates_to_the_declared_area(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """The truth's ``area`` must be the profile's actual integral.

        Every quantification test compares a measured area to this number, so a
        systematic offset here would be invisible and would corrupt all of them.
        """
        retention_times = mixed_sample.cube.retention_times
        for component in mixed_sample.truth.components:
            profile = mixed_sample.C[:, component.index]
            measured = _integrate(profile, retention_times)
            # Peaks near the run edges are genuinely truncated by the acquisition.
            margin = 6.0 * component.peak_width_s + 4.0 * component.tailing_s
            if (
                component.retention_time_s - margin < retention_times[0]
                or component.retention_time_s + margin > retention_times[-1]
            ):
                continue
            assert measured == pytest.approx(component.area, rel=0.02), (
                f"{component.name}: integral {measured:.4g} vs declared {component.area:.4g}"
            )

    def test_component_tic_reconstructs_the_clean_tic(
        self, noiseless_sample: SyntheticPyrogram
    ) -> None:
        """Summing the per-component TICs must give the observed TIC."""
        assert np.allclose(
            noiseless_sample.component_tic().sum(axis=1),
            noiseless_sample.cube.tic,
            rtol=1e-9,
            atol=1e-6,
        )

    def test_clean_cube_matches_the_stored_clean_matrix(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        assert np.array_equal(mixed_sample.clean_cube.intensities, mixed_sample.clean)

    def test_observed_cube_is_clean_plus_baseline_plus_noise(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """Noise must be a small perturbation, not a dominant term."""
        residual = mixed_sample.cube.intensities - (mixed_sample.clean + mixed_sample.baseline)
        assert np.abs(residual).mean() < 5.0 * mixed_sample.truth.noise_sigma + 1.0


class TestReproducibility:
    def test_same_seed_gives_an_identical_cube(self) -> None:
        first = SyntheticPyrogramGenerator(seed=5).generate(RECIPES["virgin_pp"])
        second = SyntheticPyrogramGenerator(seed=5).generate(RECIPES["virgin_pp"])
        assert np.array_equal(first.cube.intensities, second.cube.intensities)

    def test_different_seed_changes_the_noise_but_not_the_chemistry(self) -> None:
        """Composition is deterministic; only the noise realisation is random."""
        first = SyntheticPyrogramGenerator(seed=5).generate(RECIPES["virgin_pp"])
        second = SyntheticPyrogramGenerator(seed=6).generate(RECIPES["virgin_pp"])
        assert not np.array_equal(first.cube.intensities, second.cube.intensities)
        assert first.component_names == second.component_names
        assert np.allclose(first.C, second.C)
        assert np.allclose(first.S, second.S)

    def test_replicates_differ_in_noise_only(self) -> None:
        generator = SyntheticPyrogramGenerator(seed=11)
        first = generator.generate(RECIPES["virgin_ps"], replicate=1)
        second = generator.generate(RECIPES["virgin_ps"], replicate=2)
        assert not np.array_equal(first.cube.intensities, second.cube.intensities)
        assert np.allclose(first.C, second.C)
        assert first.cube.metadata.sample.replicate == 1
        assert second.cube.metadata.sample.replicate == 2

    def test_seeding_does_not_depend_on_python_string_hashing(self) -> None:
        """Recipe names are folded in with crc32, not ``hash()``.

        ``hash()`` of a string is salted per process, so using it would make
        "same seed, same recipe" reproducible only inside a single interpreter — and
        a benchmark dataset that cannot be regenerated is not a benchmark.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            from tests.synthetic_data import SyntheticPyrogramGenerator, RECIPES
            sample = SyntheticPyrogramGenerator(seed=5).generate(RECIPES["virgin_pp"])
            print(repr(float(sample.cube.intensities.sum())))
            """
        )
        outputs = set()
        for hash_seed in ("0", "1", "12345"):
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": hash_seed, "PYTHONPATH": "src:.", "PATH": "/usr/bin:/bin"},
            )
            outputs.add(completed.stdout.strip())
        assert len(outputs) == 1, f"cube changed with PYTHONHASHSEED: {outputs}"


class TestComponentIdentity:
    def test_component_names_are_unique(self, mixed_sample: SyntheticPyrogram) -> None:
        """Duplicate names would make the ground truth ambiguous.

        Two entries with the same compound would also share retention time and
        spectrum, so no algorithm could separate them and every recovery metric
        would be scoring against an unidentifiable target.
        """
        names = mixed_sample.component_names
        assert len(set(names)) == len(names)

    def test_shared_compounds_are_merged_with_their_split_preserved(self) -> None:
        """Benzene from PS and from PVC is one peak, but two attributions.

        The instrument sees a single benzene peak; the truth records how much came
        from each polymer, which is what a composition estimate has to reproduce.
        """
        recipe = PyrogramRecipe(
            name="ps_pvc_blend",
            fractions=(
                PolymerFraction(PolymerClass.PS, 0.5),
                PolymerFraction(PolymerClass.PVC, 0.5),
            ),
        )
        sample = SyntheticPyrogramGenerator(seed=3).generate(recipe)
        benzene = sample.truth.component_by_name("benzene")
        assert set(benzene.source_areas) == {"PS", "PVC"}
        assert sum(benzene.source_areas.values()) == pytest.approx(benzene.area)

    def test_merged_homologues_carry_contributions_from_both_polyolefins(self) -> None:
        """n-Nonane from PE and from PP is the same molecule."""
        recipe = PyrogramRecipe(
            name="pe_pp_blend",
            fractions=(
                PolymerFraction(PolymerClass.PE_HD, 0.5),
                PolymerFraction(PolymerClass.PP, 0.5),
            ),
        )
        sample = SyntheticPyrogramGenerator(seed=3).generate(recipe)
        nonane = sample.truth.component_by_name("n-alkane C9")
        assert set(nonane.source_areas) == {"PE-HD", "PP"}
        assert sum(nonane.source_areas.values()) == pytest.approx(nonane.area)

    def test_truth_indices_match_the_factor_columns(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        for component in mixed_sample.truth.components:
            assert mixed_sample.index_of(component.name) == component.index

    def test_components_are_ordered_by_retention_time(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """A banded ``C`` is what the windowed deconvolution expects."""
        retention_times = [
            component.retention_time_s for component in mixed_sample.truth.components
        ]
        assert retention_times == sorted(retention_times)

    def test_area_by_polymer_sums_to_the_total(self, mixed_sample: SyntheticPyrogram) -> None:
        totals = mixed_sample.truth.area_by_polymer()
        assert sum(totals.values()) == pytest.approx(mixed_sample.truth.total_area, rel=1e-9)

    def test_lookup_of_an_unknown_component_raises(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        with pytest.raises(KeyError, match="no synthetic component named"):
            mixed_sample.index_of("polytetrafluoroethylene")
        with pytest.raises(KeyError, match="no synthetic component named"):
            mixed_sample.truth.component_by_name("polytetrafluoroethylene")


class TestPolyolefinChemistry:
    def test_pe_produces_fused_alkane_alkene_diene_triplets(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """The PE triplet must be present *and* unresolved.

        A well-separated triplet would be trivially integrable and would not test
        curve resolution at all, so the resolution bound is part of the contract.
        """
        for carbon in (10, 14, 18):
            alkane = hdpe_sample.truth.component_by_name(f"n-alkane C{carbon}")
            alkene = hdpe_sample.truth.component_by_name(f"1-alkene C{carbon}")
            diene = hdpe_sample.truth.component_by_name(f"alkadiene C{carbon}")

            # Elution order on a non-polar phase: diene, then alkene, then alkane.
            assert diene.retention_time_s < alkene.retention_time_s < alkane.retention_time_s

            resolution = hdpe_sample.resolution_of(f"1-alkene C{carbon}", f"n-alkane C{carbon}")
            assert resolution < COELUTION_RESOLUTION, (
                f"C{carbon} alkene/alkane pair is resolved (R={resolution:.2f}); "
                "the benchmark would be easier than reality"
            )

    def test_alkene_to_alkane_ratio_matches_the_configured_value(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """The ratio the degradation index reads must be what the recipe declared."""
        ratios = []
        for carbon in range(9, 19):
            alkane = hdpe_sample.truth.component_by_name(f"n-alkane C{carbon}")
            alkene = hdpe_sample.truth.component_by_name(f"1-alkene C{carbon}")
            ratios.append(alkene.area / alkane.area)
        assert np.mean(ratios) == pytest.approx(0.62, rel=0.02)
        # Constant across the comb: the ratio is a property of the polymer, not of
        # chain length, so a single measurement anywhere in the comb is meaningful.
        assert np.std(ratios) < 1e-6

    def test_chain_length_distribution_peaks_in_the_expected_range(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Flash pyrolysis at 600 C peaks around C11-C16, not at the chain ends."""
        alkanes = hdpe_sample.truth.select(role=MarkerRole.HOMOLOGUE_ALKANE)
        heaviest = max(alkanes, key=lambda component: component.area)
        assert heaviest.carbon_number is not None
        assert 10 <= heaviest.carbon_number <= 17

    def test_distribution_is_skewed_towards_long_chains(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        alkanes = {
            component.carbon_number: component.area
            for component in hdpe_sample.truth.select(role=MarkerRole.HOMOLOGUE_ALKANE)
        }
        peak_carbon = max(alkanes, key=lambda carbon: alkanes[carbon])
        below = alkanes.get(peak_carbon - 4)
        above = alkanes.get(peak_carbon + 4)
        assert below is not None and above is not None
        assert above > below, "the long-chain tail must be heavier than the short-chain side"

    def test_ldpe_is_more_branched_than_hdpe(
        self, ldpe_sample: SyntheticPyrogram, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """The branching contrast the recyclate passport reports has to exist.

        Both grades give the same alkane comb, so grade discrimination rests
        entirely on the iso-alkene fraction.
        """

        def branching_index(sample: SyntheticPyrogram) -> float:
            iso = sum(
                component.area
                for component in sample.truth.select(role=MarkerRole.HOMOLOGUE_ISOALKENE)
            )
            linear = sum(
                component.area
                for component in sample.truth.select(role=MarkerRole.HOMOLOGUE_ALKANE)
            )
            return iso / linear

        assert branching_index(ldpe_sample) > 3.0 * branching_index(hdpe_sample)

    def test_pp_is_dominated_by_branched_alkenes(self) -> None:
        """PP pyrolysis gives iso-alkenes, unlike PE which gives n-alkanes."""
        sample = SyntheticPyrogramGenerator(seed=2).generate(RECIPES["virgin_pp"])
        iso = sum(
            component.area
            for component in sample.truth.select(role=MarkerRole.HOMOLOGUE_ISOALKENE)
        )
        alkane = sum(
            component.area
            for component in sample.truth.select(role=MarkerRole.HOMOLOGUE_ALKANE)
        )
        assert iso > 2.0 * alkane

    def test_pp_concentrates_signal_on_propene_oligomers(self) -> None:
        """C12/C15 stand out because PP depolymerises into propene multiples.

        C9 is excluded here because it is emitted as the named trimer marker rather
        than a generic branched alkene; it is covered by the test below.
        """
        sample = SyntheticPyrogramGenerator(seed=2).generate(RECIPES["virgin_pp"])
        iso_by_carbon = {
            component.carbon_number: component.area
            for component in sample.truth.select(role=MarkerRole.HOMOLOGUE_ISOALKENE)
        }
        for multiple in (12, 15):
            neighbours = [
                iso_by_carbon[carbon]
                for carbon in (multiple - 1, multiple + 1)
                if carbon in iso_by_carbon
            ]
            assert neighbours
            assert iso_by_carbon[multiple] > 2.0 * max(neighbours)

    def test_pp_trimer_marker_is_the_dominant_single_peak(self) -> None:
        """2,4-Dimethyl-1-heptene is *the* PP identifier and must be present.

        It is emitted with its tabulated spectrum and the trimer role rather than as
        a generic C9 branched alkene, because that is what a marker library will
        look for. It must also be substituted for the generic homologue, not added
        alongside it, or the same molecule would be counted twice.
        """
        sample = SyntheticPyrogramGenerator(seed=2).generate(RECIPES["virgin_pp"])
        marker = sample.truth.component_by_name("2,4-dimethyl-1-heptene")
        assert marker.polymer_class is PolymerClass.PP
        assert marker.role is MarkerRole.TRIMER
        assert marker.carbon_number == 9
        assert marker.quantifier_mz == 126

        assert not sample.truth.select(
            role=MarkerRole.HOMOLOGUE_ISOALKENE, carbon_number=9
        ), "the generic C9 iso-alkene must be replaced by the named marker, not duplicated"

        largest = max(sample.truth.components, key=lambda component: component.area)
        assert largest.name == "2,4-dimethyl-1-heptene"

    def test_rejects_a_non_polyolefin_for_the_polyolefin_builder(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        with pytest.raises(ValueError, match="is not a polyolefin"):
            generator.polyolefin_series(PolymerClass.PET, 1e6)


class TestDegradationChemistry:
    def test_ageing_adds_oxidation_products(
        self, hdpe_sample: SyntheticPyrogram, aged_hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Carbonyl content is the primary ageing signal."""

        def oxidised_fraction(sample: SyntheticPyrogram) -> float:
            oxidised = sum(
                component.area
                for component in sample.truth.components
                if component.role.is_oxidation_product
            )
            return oxidised / sample.truth.total_area

        virgin = oxidised_fraction(hdpe_sample)
        aged = oxidised_fraction(aged_hdpe_sample)
        assert virgin < 0.005
        assert aged > 0.03
        assert aged > 10.0 * virgin

    def test_ageing_raises_the_branching_and_alkene_ratios(
        self, hdpe_sample: SyntheticPyrogram, aged_hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Chain scission creates branch points and terminal double bonds."""

        def ratio(sample: SyntheticPyrogram, role: MarkerRole) -> float:
            numerator = sum(
                component.area for component in sample.truth.select(role=role)
            )
            denominator = sum(
                component.area
                for component in sample.truth.select(role=MarkerRole.HOMOLOGUE_ALKANE)
            )
            return numerator / denominator

        assert ratio(aged_hdpe_sample, MarkerRole.HOMOLOGUE_ISOALKENE) > ratio(
            hdpe_sample, MarkerRole.HOMOLOGUE_ISOALKENE
        )
        assert ratio(aged_hdpe_sample, MarkerRole.HOMOLOGUE_ALKENE) > ratio(
            hdpe_sample, MarkerRole.HOMOLOGUE_ALKENE
        )

    def test_ageing_shortens_the_chain_length_distribution(
        self, hdpe_sample: SyntheticPyrogram, aged_hdpe_sample: SyntheticPyrogram
    ) -> None:
        def weighted_mean_carbon(sample: SyntheticPyrogram) -> float:
            alkanes = sample.truth.select(role=MarkerRole.HOMOLOGUE_ALKANE)
            weights = np.array([component.area for component in alkanes])
            carbons = np.array([component.carbon_number for component in alkanes], dtype=float)
            return float(np.average(carbons, weights=weights))

        assert weighted_mean_carbon(aged_hdpe_sample) < weighted_mean_carbon(hdpe_sample)

    def test_acid_share_of_oxidation_products_grows_with_ageing(self) -> None:
        """Acids are terminal oxidation products, so they dominate late in ageing.

        This ordering is what lets a degradation index distinguish mild from severe
        ageing rather than merely detecting that oxidation happened.
        """
        generator = SyntheticPyrogramGenerator(seed=8)

        def acid_share(level: float) -> float:
            recipe = PyrogramRecipe(
                name=f"aged_{level}",
                fractions=(PolymerFraction(PolymerClass.PE_HD, 1.0, level),),
            )
            sample = generator.generate(recipe)
            acids = sum(
                component.area
                for component in sample.truth.select(role=MarkerRole.OXIDATION_ACID)
            )
            oxidised = sum(
                component.area
                for component in sample.truth.components
                if component.role.is_oxidation_product
            )
            return acids / oxidised

        assert acid_share(0.1) < acid_share(0.5) < acid_share(0.95)

    def test_styrenic_ageing_skews_the_marker_triad(self, ps_sample: SyntheticPyrogram) -> None:
        """Chain scission favours the monomer over the trimer.

        Identification therefore has to use the triad *ratio*, since the styrene
        peak alone changes with ageing at constant PS content.
        """
        aged = SyntheticPyrogramGenerator(seed=4).generate(
            PyrogramRecipe(
                name="aged_ps", fractions=(PolymerFraction(PolymerClass.PS, 1.0, 0.9),)
            )
        )

        def monomer_to_trimer(sample: SyntheticPyrogram) -> float:
            monomer = sample.truth.component_by_name("styrene").area
            trimer = sample.truth.component_by_name("2,4,6-triphenyl-1-hexene").area
            return monomer / trimer

        assert monomer_to_trimer(aged) > 2.0 * monomer_to_trimer(ps_sample)

    def test_styrenic_ageing_adds_aromatic_carbonyl_markers(
        self, ps_sample: SyntheticPyrogram
    ) -> None:
        aged = SyntheticPyrogramGenerator(seed=4).generate(
            PyrogramRecipe(
                name="aged_ps2", fractions=(PolymerFraction(PolymerClass.PS, 1.0, 0.9),)
            )
        )
        for marker in ("benzaldehyde", "acetophenone"):
            virgin_area = ps_sample.truth.component_by_name(marker).area
            aged_area = aged.truth.component_by_name(marker).area
            assert aged_area > 5.0 * virgin_area


class TestStyrenicMarkerTriad:
    def test_triad_is_present_in_oligomer_order(self, ps_sample: SyntheticPyrogram) -> None:
        monomer = ps_sample.truth.component_by_name("styrene")
        dimer = ps_sample.truth.component_by_name("2,4-diphenyl-1-butene")
        trimer = ps_sample.truth.component_by_name("2,4,6-triphenyl-1-hexene")

        assert monomer.role is MarkerRole.MONOMER
        assert dimer.role is MarkerRole.DIMER
        assert trimer.role is MarkerRole.TRIMER
        assert monomer.retention_time_s < dimer.retention_time_s < trimer.retention_time_s
        assert monomer.area > dimer.area > trimer.area

    def test_triad_ratio_is_the_configured_100_12_6(
        self, ps_sample: SyntheticPyrogram
    ) -> None:
        monomer = ps_sample.truth.component_by_name("styrene").area
        dimer = ps_sample.truth.component_by_name("2,4-diphenyl-1-butene").area
        trimer = ps_sample.truth.component_by_name("2,4,6-triphenyl-1-hexene").area
        assert 100.0 * dimer / monomer == pytest.approx(12.0, rel=0.01)
        assert 100.0 * trimer / monomer == pytest.approx(6.0, rel=0.01)

    def test_triad_survives_dilution_into_a_polyolefin_matrix(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """At 4.5 % PS the ratio must be unchanged — only the absolute area shrinks."""
        monomer = mixed_sample.truth.component_by_name("styrene").area
        dimer = mixed_sample.truth.component_by_name("2,4-diphenyl-1-butene").area
        assert 0.05 < dimer / monomer < 0.15

    def test_rejects_a_non_styrenic_polymer(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        with pytest.raises(ValueError, match="is not a styrenic polymer"):
            generator.styrenic_series(PolymerClass.PE_HD, 1e6)


class TestTracePolymersAndResponseFactors:
    def test_response_factors_decouple_mass_from_signal(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """1 % PVC by mass must not look like 1 % of the chromatogram.

        PVC loses most of its mass as HCl, PET and PA6 form partly non-eluting polar
        fragments, and PS over-responds. Any quantification that skips this
        correction is wrong by a factor of two or more, so the benchmark builds the
        distortion in.
        """
        areas = mixed_sample.truth.area_by_polymer()
        total = mixed_sample.truth.total_area
        mass = mixed_sample.truth.blend_fractions

        pvc_signal_share = areas["PVC"] / total
        assert pvc_signal_share < 0.5 * mass["PVC"]

        ps_signal_share = areas["PS"] / total
        assert ps_signal_share > 1.2 * mass["PS"]

    def test_response_factor_table_covers_every_simulated_polymer(self) -> None:
        for recipe in RECIPES.values():
            for fraction in recipe.fractions:
                assert fraction.polymer in POLYMER_RESPONSE_FACTORS, (
                    f"{fraction.polymer} has no response factor but appears in "
                    f"recipe {recipe.name!r}"
                )

    def test_trace_pet_markers_are_present_at_trace_level(self) -> None:
        """0.5 % PET must land near 0.2 % of the signal — genuinely trace analysis."""
        sample = SyntheticPyrogramGenerator(seed=13).generate(
            RECIPES["trace_pet_in_polyolefin"]
        )
        pet_share = sample.truth.area_by_polymer()["PET"] / sample.truth.total_area
        assert 5e-4 < pet_share < 5e-3

        benzoic_acid = sample.truth.component_by_name("benzoic acid")
        assert benzoic_acid.polymer_class is PolymerClass.PET
        assert benzoic_acid.quantifier_mz == 105

    def test_trace_marker_is_buried_in_the_tic_but_visible_on_its_quantifier_ion(
        self,
    ) -> None:
        """The reason extracted-ion work and matrix subtraction exist.

        On the total ion current the PET marker is invisible; on m/z 105 it is a
        clear peak. The test asserts both halves, because a benchmark where the
        trace marker is already visible in the TIC would not exercise the problem.
        """
        sample = SyntheticPyrogramGenerator(seed=13).generate(
            RECIPES["trace_pet_in_polyolefin"]
        )
        marker = sample.truth.component_by_name("benzoic acid")
        cube = sample.cube
        scan = cube.nearest_scan(marker.retention_time_s)

        tic_local = cube.tic[scan]
        tic_neighbourhood = np.median(
            cube.tic[max(scan - 60, 0) : min(scan + 60, cube.n_scans)]
        )
        assert tic_local < 3.0 * tic_neighbourhood, "marker should not stand out in the TIC"

        quantifier = cube.eic(105.0)
        window = quantifier[max(scan - 8, 0) : scan + 9]
        baseline_level = np.median(quantifier)
        assert window.max() > 5.0 * max(baseline_level, 1.0)

    def test_pvc_produces_its_aromatic_cascade(self, mixed_sample: SyntheticPyrogram) -> None:
        for marker in ("benzene", "toluene", "naphthalene"):
            component = mixed_sample.truth.component_by_name(marker)
            assert component.area > 0.0

    def test_pa6_marker_is_caprolactam(self, mixed_sample: SyntheticPyrogram) -> None:
        caprolactam = mixed_sample.truth.component_by_name("epsilon-caprolactam")
        assert caprolactam.polymer_class is PolymerClass.PA6
        assert caprolactam.role is MarkerRole.MONOMER
        # A lactam tails badly on a non-polar phase, unlike a hydrocarbon eluting at
        # a comparable retention time.
        neighbour = mixed_sample.truth.component_by_name("n-alkane C14")
        assert caprolactam.tailing_s > 2.0 * neighbour.tailing_s

    def test_unsupported_polymer_is_reported_clearly(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        with pytest.raises(ValueError, match="no marker pattern defined"):
            generator.condensation_polymer_series(PolymerClass.POM, 1e6)


class TestCoelution:
    def test_intended_coelutions_are_actually_unresolved(
        self, coelution_sample: SyntheticPyrogram
    ) -> None:
        """The specific overlaps the recipe promises must really be fused.

        These are the pairs that make trace foreign-polymer detection hard in a real
        PCR sample, and they are placed deliberately rather than by accident.
        """
        pairs = [
            ("styrene", "n-alkane C8"),
            ("epsilon-caprolactam", "n-alkane C14"),
            ("2,4-dimethyl-1-heptene", "1-alkene C9"),
        ]
        for first, second in pairs:
            resolution = coelution_sample.resolution_of(first, second)
            assert resolution < 1.5, (
                f"{first} / {second} resolution is {resolution:.2f}; expected an "
                "unresolved or barely resolved pair"
            )

    def test_the_cube_contains_large_fused_clusters(
        self, coelution_sample: SyntheticPyrogram
    ) -> None:
        groups = coelution_sample.truth.coeluting_groups(resolution_threshold=1.0)
        assert max(len(group) for group in groups) >= 4

    def test_coeluting_groups_partition_every_component(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        groups = mixed_sample.truth.coeluting_groups()
        flattened = [index for group in groups for index in group]
        assert sorted(flattened) == list(range(mixed_sample.truth.n_components))

    def test_coeluting_groups_are_ordered_by_retention_time(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        groups = mixed_sample.truth.coeluting_groups()
        apexes = [
            mixed_sample.truth.components[group[0]].retention_time_s for group in groups
        ]
        assert apexes == sorted(apexes)

    def test_a_stricter_threshold_produces_fewer_larger_clusters(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        loose = mixed_sample.truth.coeluting_groups(resolution_threshold=0.4)
        strict = mixed_sample.truth.coeluting_groups(resolution_threshold=2.0)
        assert len(strict) <= len(loose)


class TestRetentionDrift:
    def test_no_drift_leaves_retention_times_untouched(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        assert hdpe_sample.truth.drift.is_identity
        for component in hdpe_sample.truth.components:
            assert component.retention_time_s == pytest.approx(
                component.nominal_retention_time_s
            )
            assert component.drift_s == pytest.approx(0.0)

    def test_offset_drift_shifts_every_peak_equally(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        drifted = generator.generate(
            RECIPES["virgin_hdpe"].with_drift(DriftTruth(offset_s=4.0))
        )
        shifts = [component.drift_s for component in drifted.truth.components]
        assert np.allclose(shifts, 4.0)

    def test_quadratic_drift_moves_late_peaks_more_than_early_ones(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        """Column ageing is not a constant offset — that is why alignment is hard."""
        drifted = generator.generate(
            RECIPES["virgin_hdpe"].with_drift(DriftTruth(quadratic_factor=40.0))
        )
        components = sorted(
            drifted.truth.components, key=lambda component: component.nominal_retention_time_s
        )
        assert components[-1].drift_s > components[0].drift_s + 5.0
        # And the warp is monotone in retention time, not merely larger at the end.
        drifts = [component.drift_s for component in components]
        assert drifts == sorted(drifts)

    def test_drift_preserves_elution_order(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        """A warp that reordered peaks would be physically impossible."""
        drift = DriftTruth(
            offset_s=2.0,
            linear_factor=2e-3,
            quadratic_factor=10.0,
            oscillation_amplitude_s=1.0,
            oscillation_period_s=600.0,
        )
        drifted = generator.generate(RECIPES["virgin_hdpe"].with_drift(drift))
        nominal = np.array(
            [component.nominal_retention_time_s for component in drifted.truth.components]
        )
        observed = np.array(
            [component.retention_time_s for component in drifted.truth.components]
        )
        assert np.array_equal(np.argsort(nominal), np.argsort(observed))

    def test_drift_changes_the_cube_not_only_the_truth(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        reference = generator.generate(RECIPES["virgin_ps"])
        drifted = generator.generate(
            RECIPES["virgin_ps"].with_drift(DriftTruth(offset_s=10.0))
        )
        styrene_mz = 104.0
        reference_apex = reference.cube.retention_times[
            int(np.argmax(reference.cube.eic(styrene_mz)))
        ]
        drifted_apex = drifted.cube.retention_times[
            int(np.argmax(drifted.cube.eic(styrene_mz)))
        ]
        assert drifted_apex - reference_apex == pytest.approx(10.0, abs=1.0)

    def test_drift_helper_rejects_an_empty_window(self) -> None:
        with pytest.raises(ValueError, match="empty run window"):
            apply_retention_drift(np.array([100.0]), DriftTruth(), 500.0, 500.0)

    def test_peaks_pushed_outside_the_window_are_dropped_from_the_truth(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        """Unobservable peaks must not be scored against.

        A component whose apex leaves the acquisition window was never measured, so
        keeping it in the ground truth would penalise every recovery metric for
        something no algorithm could find.
        """
        reference = generator.generate(RECIPES["virgin_hdpe"])
        shifted = generator.generate(
            RECIPES["virgin_hdpe"].with_drift(DriftTruth(offset_s=-300.0))
        )
        assert shifted.truth.n_components < reference.truth.n_components
        for component in shifted.truth.components:
            assert (
                generator.rt_start_s
                <= component.retention_time_s
                <= generator.rt_end_s
            )

    def test_impossible_drift_is_reported_rather_than_producing_an_empty_cube(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        with pytest.raises(ValueError, match="no component elutes inside"):
            generator.generate(
                RECIPES["virgin_hdpe"].with_drift(DriftTruth(offset_s=1e5))
            )


class TestRunSeries:
    def test_series_produces_the_requested_number_of_runs(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        runs = generator.generate_series(RECIPES["virgin_ps"], 4)
        assert len(runs) == 4
        assert [run.cube.metadata.sample.replicate for run in runs] == [1, 2, 3, 4]

    def test_series_runs_have_different_retention_axes_but_equal_composition(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        """Exactly the input PARAFAC2 exists for: same chemistry, shifted axes."""
        runs = generator.generate_series(RECIPES["virgin_ps"], 3)
        styrene_apexes = [
            run.truth.component_by_name("styrene").retention_time_s for run in runs
        ]
        assert len(set(styrene_apexes)) == 3
        assert max(styrene_apexes) - min(styrene_apexes) > 0.5

        triad_ratios = [
            run.truth.component_by_name("2,4-diphenyl-1-butene").area
            / run.truth.component_by_name("styrene").area
            for run in runs
        ]
        assert np.allclose(triad_ratios, triad_ratios[0])

    def test_zero_drift_scale_gives_perfectly_aligned_runs(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        runs = generator.generate_series(RECIPES["virgin_ps"], 3, drift_scale=0.0)
        apexes = {run.truth.component_by_name("styrene").retention_time_s for run in runs}
        assert len(apexes) == 1

    def test_series_is_reproducible(self, generator: SyntheticPyrogramGenerator) -> None:
        first = generator.generate_series(RECIPES["virgin_ps"], 2)
        second = generator.generate_series(RECIPES["virgin_ps"], 2)
        for left, right in zip(first, second, strict=True):
            assert np.array_equal(left.cube.intensities, right.cube.intensities)

    def test_rejects_a_non_positive_run_count(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        with pytest.raises(ValueError, match="n_runs must be >= 1"):
            generator.generate_series(RECIPES["virgin_ps"], 0)


class TestNoiseAndBaseline:
    def test_default_run_carries_noise(self, hdpe_sample: SyntheticPyrogram) -> None:
        residual = hdpe_sample.cube.intensities - (hdpe_sample.clean + hdpe_sample.baseline)
        assert np.abs(residual).max() > 0.0

    def test_intensities_are_never_negative(self, hdpe_sample: SyntheticPyrogram) -> None:
        assert np.all(hdpe_sample.cube.intensities >= 0.0)

    def test_intensities_are_quantised_like_an_adc(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        assert np.array_equal(
            hdpe_sample.cube.intensities, np.round(hdpe_sample.cube.intensities)
        )

    def test_shot_noise_scales_with_signal(self) -> None:
        """Noise on a peak must exceed noise on the baseline — Poisson statistics.

        Constant-variance noise would make weak-signal regions look artificially
        hard and strong-signal regions artificially easy.
        """
        generator = SyntheticPyrogramGenerator(
            seed=21,
            rt_end_s=900.0,
            scan_rate_hz=2.0,
            carbon_range=(6, 14),
            mz_high=200,
            noise=NoiseModel(detector_sigma=0.0, shot_factor=1.0, quantise=False),
            baseline=BaselineModel.none(),
        )
        sample = generator.generate(RECIPES["virgin_hdpe"])
        residual = np.abs(sample.cube.intensities - sample.clean)
        strong = sample.clean > np.percentile(sample.clean[sample.clean > 0], 99)
        weak = sample.clean == 0.0
        assert residual[strong].mean() > 10.0 * max(residual[weak].mean(), 1e-9)

    def test_baseline_rises_towards_the_end_of_the_run(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Column bleed grows as the oven ramps; that is the pedestal to remove.

        The growth is asserted on a bleed channel rather than on the summed
        baseline, because the flat electronic offset spread over ~300 channels
        dominates the sum and would mask the ramp.
        """
        bleed_channel = hdpe_sample.baseline[:, hdpe_sample.cube.mz_index(207.0)]
        first_tenth = bleed_channel[: bleed_channel.size // 10].mean()
        last_tenth = bleed_channel[-bleed_channel.size // 10 :].mean()
        assert last_tenth > 50.0 * first_tenth

        # The total background still rises, just less dramatically.
        total = hdpe_sample.baseline.sum(axis=1)
        assert total[-total.size // 10 :].mean() > 1.5 * total[: total.size // 10].mean()

    def test_baseline_ramp_is_monotonic(self, hdpe_sample: SyntheticPyrogram) -> None:
        """A non-monotonic bleed would be indistinguishable from a broad peak."""
        bleed_channel = hdpe_sample.baseline[:, hdpe_sample.cube.mz_index(207.0)]
        assert np.all(np.diff(bleed_channel) >= -1e-9)

    def test_baseline_sits_on_the_siloxane_channels(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Bleed is channel-selective, which is why correction must be per-channel."""
        cube = hdpe_sample.cube
        late = cube.n_scans - 1
        siloxane = hdpe_sample.baseline[late, cube.mz_index(207.0)]
        neighbour = hdpe_sample.baseline[late, cube.mz_index(206.0)]
        assert siloxane > 20.0 * neighbour

    def test_flat_offset_is_present_on_every_channel(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        assert hdpe_sample.baseline.min() > 0.0

    def test_silent_noise_and_no_baseline_models_add_nothing(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        recipe = PyrogramRecipe(
            name="silent",
            fractions=(PolymerFraction(PolymerClass.PS, 1.0),),
            noise=NoiseModel.silent(),
            baseline=BaselineModel.none(),
        )
        sample = generator.generate(recipe)
        assert np.allclose(sample.cube.intensities, sample.C @ sample.S)

    def test_spikes_are_added_when_requested(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        recipe = PyrogramRecipe(
            name="spiky",
            fractions=(PolymerFraction(PolymerClass.PS, 1.0),),
            noise=NoiseModel(
                detector_sigma=10.0, shot_factor=0.0, spike_probability=1e-3, quantise=False
            ),
            baseline=BaselineModel.none(),
        )
        sample = generator.generate(recipe)
        residual = sample.cube.intensities - sample.clean
        assert residual.max() > 500.0


class TestMetadata:
    def test_metadata_describes_a_synthetic_acquisition(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        metadata = mixed_sample.cube.metadata
        assert metadata.source_format is SourceFormat.SYNTHETIC
        assert metadata.source_path is None
        assert metadata.sample.stream is RecyclateStream.PCR_MIXED_POLYOLEFIN
        assert metadata.n_scans == mixed_sample.cube.n_scans
        assert metadata.extra["recipe"] == "pcr_mixed_polyolefin"

    def test_nominal_composition_matches_the_recipe(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        declared = mixed_sample.cube.metadata.sample.nominal_composition
        assert declared == mixed_sample.truth.blend_fractions

    def test_axes_match_the_metadata_spec(self, mixed_sample: SyntheticPyrogram) -> None:
        cube = mixed_sample.cube
        assert cube.metadata.mz_axis.n_bins == cube.n_mz
        assert cube.metadata.mz_axis.is_nominal
        assert cube.metadata.rt_start_s == pytest.approx(cube.retention_times[0])
        assert cube.metadata.rt_end_s == pytest.approx(cube.retention_times[-1])

    def test_scan_rate_is_reflected_in_the_axis(
        self, generator: SyntheticPyrogramGenerator, mixed_sample: SyntheticPyrogram
    ) -> None:
        assert mixed_sample.cube.mean_scan_period_s == pytest.approx(
            1.0 / generator.scan_rate_hz
        )

    def test_truth_records_the_seed_and_recipe(self, mixed_sample: SyntheticPyrogram) -> None:
        assert mixed_sample.truth.recipe_name == "pcr_mixed_polyolefin"
        assert mixed_sample.truth.seed == 20240517


class TestRecipeValidation:
    def test_mass_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            PyrogramRecipe(
                name="bad",
                fractions=(
                    PolymerFraction(PolymerClass.PE_HD, 0.5),
                    PolymerFraction(PolymerClass.PP, 0.2),
                ),
            )

    def test_a_polymer_may_not_appear_twice(self) -> None:
        with pytest.raises(ValueError, match="appears twice"):
            PyrogramRecipe(
                name="bad",
                fractions=(
                    PolymerFraction(PolymerClass.PE_HD, 0.5),
                    PolymerFraction(PolymerClass.PE_HD, 0.5),
                ),
            )

    def test_an_empty_recipe_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one polymer fraction"):
            PyrogramRecipe(name="empty", fractions=())

    def test_non_positive_total_area_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_area must be > 0"):
            PyrogramRecipe(
                name="bad",
                fractions=(PolymerFraction(PolymerClass.PS, 1.0),),
                total_area=0.0,
            )

    @pytest.mark.parametrize("mass_fraction", [-0.1, 1.5])
    def test_out_of_range_mass_fraction_is_rejected(self, mass_fraction: float) -> None:
        with pytest.raises(ValueError, match="mass_fraction"):
            PolymerFraction(PolymerClass.PS, mass_fraction)

    @pytest.mark.parametrize("level", [-0.01, 1.01])
    def test_out_of_range_degradation_level_is_rejected(self, level: float) -> None:
        with pytest.raises(ValueError, match="degradation_level"):
            PolymerFraction(PolymerClass.PS, 1.0, level)

    def test_unknown_additive_is_rejected_at_construction(self) -> None:
        with pytest.raises(KeyError, match="unknown reference compound"):
            AdditiveSpike("unobtainium stearate", 1e-3)

    def test_non_positive_additive_amount_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="relative_amount must be > 0"):
            AdditiveSpike("benzene", 0.0)

    def test_every_preset_recipe_generates(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        """No preset may be broken; they are the published benchmark set."""
        for name, recipe in RECIPES.items():
            sample = generator.generate(recipe)
            assert sample.truth.n_components > 0, f"recipe {name} produced no components"
            assert sample.cube.total_signal > 0.0
            assert sample.truth.recipe_name == name


class TestGeneratorConfiguration:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"rt_start_s": 500.0, "rt_end_s": 100.0}, "empty run window"),
            ({"scan_rate_hz": 0.0}, "scan_rate_hz must be > 0"),
            ({"mz_low": 400, "mz_high": 100}, "empty m/z window"),
            ({"carbon_range": (20, 6)}, "invalid carbon_range"),
            ({"carbon_range": (2, 20)}, "invalid carbon_range"),
        ],
    )
    def test_invalid_configuration_is_rejected(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SyntheticPyrogramGenerator(**kwargs)  # type: ignore[arg-type]

    def test_scan_rate_controls_the_number_of_scans(self) -> None:
        slow = SyntheticPyrogramGenerator(rt_start_s=0.0, rt_end_s=100.0, scan_rate_hz=1.0)
        fast = SyntheticPyrogramGenerator(rt_start_s=0.0, rt_end_s=100.0, scan_rate_hz=10.0)
        assert slow.retention_times.size == 101
        assert fast.retention_times.size == 1001

    def test_points_across_a_peak_stay_sufficient_for_curve_resolution(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        """At least ~8 points across the narrowest peak, or resolution is hopeless."""
        narrowest = generator.base_sigma_s
        points_across_fwhm = 2.355 * narrowest * generator.scan_rate_hz
        assert points_across_fwhm >= 8.0

    def test_shorter_run_rescales_tabulated_retention_times(self) -> None:
        """A different method must preserve relative marker positions.

        Otherwise the intended co-elutions would drift apart when the run length is
        changed and the benchmark's difficulty would silently depend on the fixture.
        """
        short = SyntheticPyrogramGenerator(
            seed=1, rt_start_s=60.0, rt_end_s=900.0, scan_rate_hz=4.0, carbon_range=(6, 16)
        )
        sample = short.generate(RECIPES["virgin_ps"])
        styrene = sample.truth.component_by_name("styrene")
        assert 60.0 < styrene.nominal_retention_time_s < 900.0
        # Styrene sat at ~20 % into the reference run; it must stay at ~20 %.
        assert (styrene.nominal_retention_time_s - 60.0) / 840.0 == pytest.approx(
            (414.0 - 60.0) / 1740.0, rel=0.05
        )

    def test_narrow_mz_window_rejects_a_marker_it_cannot_see(self) -> None:
        """Silently dropping every ion of a component would corrupt the truth."""
        generator = SyntheticPyrogramGenerator(seed=1, mz_low=29, mz_high=40)
        with pytest.raises(ValueError, match="no ion inside the acquired window"):
            generator.generate(RECIPES["virgin_ps"])

    def test_peaks_broaden_across_the_run(
        self, generator: SyntheticPyrogramGenerator
    ) -> None:
        early = generator._width_at(generator.rt_start_s)
        late = generator._width_at(generator.rt_end_s)
        assert late[0] > early[0]
        assert late[1] > early[1]


class TestRetentionModel:
    def test_anchor_carbon_sits_at_the_anchor_time(self) -> None:
        model = AlkaneRetentionModel(reference_carbon=8, reference_rt_s=420.0)
        assert model.retention_time_s(8) == pytest.approx(420.0)

    def test_retention_increases_with_carbon_number(self) -> None:
        model = AlkaneRetentionModel()
        times = [model.retention_time_s(carbon) for carbon in range(6, 35)]
        assert times == sorted(times)

    def test_comb_spacing_compresses_at_high_carbon_numbers(self) -> None:
        """Late homologues crowd together as the oven approaches its final hold.

        That crowding is what makes the high-carbon end of a PE pyrogram harder to
        resolve than the low-carbon end.
        """
        model = AlkaneRetentionModel()
        assert model.local_spacing_s(30) < 0.8 * model.local_spacing_s(9)

    def test_spacing_is_positive_across_the_simulated_range(self) -> None:
        model = AlkaneRetentionModel()
        for carbon in range(6, 34):
            assert model.local_spacing_s(carbon) > 0.0


@pytest.mark.slow
class TestFullResolution:
    def test_production_dimensions_are_realistic(
        self, full_resolution_sample: SyntheticPyrogram
    ) -> None:
        cube = full_resolution_sample.cube
        assert cube.n_scans > 8000
        assert cube.n_mz > 350
        assert full_resolution_sample.truth.n_components > 150

    def test_full_resolution_cube_is_internally_consistent(
        self, full_resolution_sample: SyntheticPyrogram
    ) -> None:
        residual = full_resolution_sample.cube.intensities - (
            full_resolution_sample.clean + full_resolution_sample.baseline
        )
        assert np.abs(residual).mean() < 200.0
