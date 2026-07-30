"""Tests for the Pydantic data model.

The schemas are the platform's contract with the outside world: the API accepts
them, the database stores them, and every derived number is interpreted relative
to them. Validation therefore has to reject physically impossible input rather
than passing it through to a chemometric engine that will produce a confident
wrong answer.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from data_schemas import (
    AcquisitionConditions,
    ComponentTruth,
    DriftTruth,
    GcConditions,
    MarkerRole,
    MsConditions,
    MzAxisSpec,
    OvenRamp,
    PolymerClass,
    PreprocessingStep,
    PyrogramTruth,
    PyrolysisConditions,
    PyrolysisMode,
    RecyclateStream,
    SampleMetadata,
)
from data_schemas.pyrogram import round_trip_json


class TestEnumSemantics:
    def test_polyolefin_classification(self) -> None:
        assert PolymerClass.PE_LD.is_polyolefin
        assert PolymerClass.PP.is_polyolefin
        assert not PolymerClass.PET.is_polyolefin
        assert not PolymerClass.PS.is_polyolefin

    def test_styrenic_classification(self) -> None:
        assert PolymerClass.PS.is_styrenic
        assert PolymerClass.ABS.is_styrenic
        assert PolymerClass.SAN.is_styrenic
        assert not PolymerClass.PE_HD.is_styrenic

    def test_condensation_polymer_classification(self) -> None:
        """These are the trace fractions matrix subtraction is meant to expose."""
        assert PolymerClass.PET.is_polar_condensation
        assert PolymerClass.PA6.is_polar_condensation
        assert PolymerClass.PC.is_polar_condensation
        assert not PolymerClass.PP.is_polar_condensation

    def test_marker_role_groupings(self) -> None:
        assert MarkerRole.HOMOLOGUE_ALKANE.is_homologue
        assert MarkerRole.HOMOLOGUE_ISOALKENE.is_homologue
        assert not MarkerRole.MONOMER.is_homologue

        assert MarkerRole.OXIDATION_KETONE.is_oxidation_product
        assert MarkerRole.OXIDATION_ACID.is_oxidation_product
        assert not MarkerRole.HOMOLOGUE_ALKENE.is_oxidation_product

        assert MarkerRole.ADDITIVE_PLASTICISER.is_additive
        assert not MarkerRole.MONOMER.is_additive

        assert MarkerRole.MONOMER.is_oligomer
        assert MarkerRole.TRIMER.is_oligomer
        assert not MarkerRole.CONTAMINANT.is_oligomer

    def test_enums_serialise_as_their_iso_style_strings(self) -> None:
        """Values must be report-ready, since they land in the recyclate passport."""
        assert str(PolymerClass.PE_LD) == "PE-LD"
        assert str(RecyclateStream.PCR_MIXED_POLYOLEFIN) == "PCR-mixed-polyolefin"


class TestPyrolysisConditions:
    def test_defaults_describe_the_standard_analytical_method(self) -> None:
        conditions = PyrolysisConditions()
        assert conditions.temperature_c == 600.0
        assert conditions.mode is PyrolysisMode.SINGLE_SHOT

    @pytest.mark.parametrize("temperature", [50.0, 1500.0])
    def test_impossible_temperatures_are_rejected(self, temperature: float) -> None:
        with pytest.raises(ValidationError):
            PyrolysisConditions(temperature_c=temperature)

    def test_reactive_pyrolysis_requires_a_reagent(self) -> None:
        """THM without naming the reagent is not a reproducible method."""
        with pytest.raises(ValidationError, match="requires 'derivatisation_reagent'"):
            PyrolysisConditions(mode=PyrolysisMode.REACTIVE_THM)

        conditions = PyrolysisConditions(
            mode=PyrolysisMode.REACTIVE_THM, derivatisation_reagent="TMAH 25% in methanol"
        )
        assert conditions.derivatisation_reagent is not None

    def test_record_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            PyrolysisConditions().temperature_c = 700.0  # type: ignore[misc]

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PyrolysisConditions(temperture_c=600.0)  # type: ignore[call-arg]


class TestGcConditions:
    def test_program_duration_accounts_for_ramps_and_holds(self) -> None:
        conditions = GcConditions(
            initial_temp_c=40.0,
            initial_hold_min=2.0,
            oven_program=(
                OvenRamp(rate_c_per_min=10.0, target_c=140.0, hold_min=0.0),
                OvenRamp(rate_c_per_min=20.0, target_c=320.0, hold_min=5.0),
            ),
        )
        # 2 hold + 100/10 ramp + 180/20 ramp + 5 hold
        assert conditions.program_duration_min == pytest.approx(2.0 + 10.0 + 9.0 + 5.0)

    def test_isothermal_segment_contributes_only_its_hold(self) -> None:
        conditions = GcConditions(
            initial_temp_c=40.0,
            initial_hold_min=1.0,
            oven_program=(OvenRamp(rate_c_per_min=0.0, target_c=40.0, hold_min=8.0),),
        )
        assert conditions.program_duration_min == pytest.approx(9.0)

    def test_final_temperature_comes_from_the_last_segment(self) -> None:
        assert GcConditions().final_temp_c == 320.0

    def test_a_cooling_ramp_is_rejected(self) -> None:
        """GC ovens ramp upwards; a decreasing target is a data-entry error."""
        with pytest.raises(ValidationError, match="non-decreasing"):
            GcConditions(
                oven_program=(
                    OvenRamp(rate_c_per_min=10.0, target_c=300.0),
                    OvenRamp(rate_c_per_min=10.0, target_c=200.0),
                )
            )

    def test_an_empty_program_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one segment"):
            GcConditions(oven_program=())


class TestMsConditions:
    def test_reversed_mz_window_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must exceed"):
            MsConditions(mz_low=400.0, mz_high=50.0)

    def test_scan_period_is_the_inverse_of_the_rate(self) -> None:
        assert MsConditions(scan_rate_hz=4.0).scan_period_s == pytest.approx(0.25)

    def test_defaults_are_library_comparable(self) -> None:
        """70 eV is required for fragment ratios to match reference libraries."""
        assert MsConditions().electron_energy_ev == 70.0


class TestSampleMetadata:
    def test_nominal_composition_must_not_exceed_unity(self) -> None:
        with pytest.raises(ValidationError, match="must not exceed 1.0"):
            SampleMetadata(
                sample_id="S1",
                nominal_composition={"PE-LD": 0.7, "PP": 0.5},
            )

    def test_a_partial_composition_is_allowed(self) -> None:
        """Declared content is often incomplete; the rest is simply unknown."""
        sample = SampleMetadata(sample_id="S1", nominal_composition={"PE-LD": 0.8})
        assert sample.nominal_composition == {"PE-LD": 0.8}

    @pytest.mark.parametrize("fraction", [-0.1, 1.4])
    def test_out_of_range_fraction_is_rejected(self, fraction: float) -> None:
        with pytest.raises(ValidationError, match="outside"):
            SampleMetadata(sample_id="S1", nominal_composition={"PP": fraction})

    def test_empty_sample_id_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SampleMetadata(sample_id="")


class TestMzAxisSpec:
    def test_reversed_window_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must exceed"):
            MzAxisSpec(mz_low=400.0, mz_high=50.0, n_bins=100)

    def test_zero_bins_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MzAxisSpec(mz_low=29.0, mz_high=400.0, n_bins=0)


class TestPyrogramMetadata:
    def test_derived_quantities_are_computed(self, hdpe_sample) -> None:  # noqa: ANN001
        metadata = hdpe_sample.cube.metadata
        assert metadata.rt_span_s == pytest.approx(
            metadata.rt_end_s - metadata.rt_start_s
        )
        assert metadata.mean_scan_period_s == pytest.approx(
            metadata.rt_span_s / (metadata.n_scans - 1)
        )

    def test_reversed_retention_window_is_rejected(self, hdpe_sample) -> None:  # noqa: ANN001
        with pytest.raises(ValidationError, match="must not precede"):
            hdpe_sample.cube.metadata.model_copy(
                update={"rt_start_s": 900.0}
            ).model_validate(
                hdpe_sample.cube.metadata.model_dump()
                | {"rt_start_s": 900.0, "rt_end_s": 100.0}
            )

    def test_with_step_appends_without_mutating(self, hdpe_sample) -> None:  # noqa: ANN001
        metadata = hdpe_sample.cube.metadata
        before = len(metadata.preprocessing)
        extended = metadata.with_step(PreprocessingStep(name="unit"))
        assert len(extended.preprocessing) == before + 1
        assert len(metadata.preprocessing) == before

    def test_with_warnings_appends_without_mutating(self, hdpe_sample) -> None:  # noqa: ANN001
        metadata = hdpe_sample.cube.metadata
        extended = metadata.with_warnings("a", "b")
        assert extended.reader_warnings[-2:] == ("a", "b")
        assert metadata.reader_warnings != extended.reader_warnings

    def test_round_trip_helper_drops_computed_fields(self, hdpe_sample) -> None:  # noqa: ANN001
        """Computed fields are output-only and must not be fed back in.

        ``model_dump_json`` includes them, and the model forbids extra keys, so a
        naive save/load cycle would fail validation. The helper is what the ``.npz``
        archive and any on-disk benchmark rely on.
        """
        from data_schemas.pyrogram import PyrogramMetadata

        text = round_trip_json(hdpe_sample.cube.metadata)
        assert "rt_span_s" not in text
        restored = PyrogramMetadata.model_validate_json(text)
        assert restored.n_scans == hdpe_sample.cube.metadata.n_scans


class TestAcquisitionConditions:
    def test_defaults_compose_a_usable_method(self) -> None:
        conditions = AcquisitionConditions()
        assert conditions.pyrolysis.temperature_c == 600.0
        assert conditions.gc.final_temp_c == 320.0
        assert conditions.ms.scan_rate_hz == 5.0

    def test_is_json_serialisable(self) -> None:
        conditions = AcquisitionConditions()
        restored = AcquisitionConditions.model_validate_json(conditions.model_dump_json())
        assert restored == conditions


class TestTruthSchema:
    @staticmethod
    def _component(index: int, name: str, retention_time_s: float, **overrides) -> ComponentTruth:  # noqa: ANN003
        defaults: dict = {
            "index": index,
            "name": name,
            "polymer_class": PolymerClass.PE_HD,
            "role": MarkerRole.HOMOLOGUE_ALKANE,
            "retention_time_s": retention_time_s,
            "nominal_retention_time_s": retention_time_s,
            "peak_width_s": 2.0,
            "area": 1000.0,
            "base_peak_mz": 57,
            "quantifier_mz": 57,
        }
        return ComponentTruth(**(defaults | overrides))

    def test_blend_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            PyrogramTruth(
                recipe_name="r",
                seed=0,
                blend_fractions={"PE-HD": 0.5, "PP": 0.2},
                components=(self._component(0, "a", 100.0),),
            )

    def test_empty_blend_fractions_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            PyrogramTruth(
                recipe_name="r",
                seed=0,
                blend_fractions={},
                components=(self._component(0, "a", 100.0),),
            )

    def test_drift_is_recorded_on_the_component(self) -> None:
        component = self._component(
            0, "a", retention_time_s=104.0, nominal_retention_time_s=100.0
        )
        assert component.drift_s == pytest.approx(4.0)

    def test_identity_drift_is_recognised(self) -> None:
        assert DriftTruth().is_identity
        assert not DriftTruth(offset_s=1.0).is_identity
        assert not DriftTruth(oscillation_amplitude_s=0.5).is_identity

    def test_select_filters_on_every_axis(self) -> None:
        truth = PyrogramTruth(
            recipe_name="r",
            seed=0,
            blend_fractions={"PE-HD": 1.0},
            components=(
                self._component(0, "n-alkane C8", 100.0, carbon_number=8),
                self._component(
                    1, "1-alkene C8", 98.0, carbon_number=8, role=MarkerRole.HOMOLOGUE_ALKENE
                ),
                self._component(
                    2,
                    "styrene",
                    150.0,
                    polymer_class=PolymerClass.PS,
                    role=MarkerRole.MONOMER,
                    area=5.0,
                ),
            ),
        )
        assert len(truth.select(polymer_class=PolymerClass.PE_HD)) == 2
        assert len(truth.select(role=MarkerRole.HOMOLOGUE_ALKENE)) == 1
        assert len(truth.select(carbon_number=8)) == 2
        assert len(truth.select(min_area=100.0)) == 2
        assert truth.select(polymer_class=PolymerClass.PS)[0].name == "styrene"

    def test_area_by_polymer_falls_back_to_sole_attribution(self) -> None:
        """Components without an explicit split are credited to their own class."""
        truth = PyrogramTruth(
            recipe_name="r",
            seed=0,
            blend_fractions={"PE-HD": 1.0},
            components=(self._component(0, "n-alkane C8", 100.0, area=250.0),),
        )
        assert truth.area_by_polymer() == {"PE-HD": 250.0}

    def test_area_by_polymer_uses_the_recorded_split(self) -> None:
        truth = PyrogramTruth(
            recipe_name="r",
            seed=0,
            blend_fractions={"PE-HD": 0.5, "PP": 0.5},
            components=(
                self._component(
                    0,
                    "n-alkane C9",
                    100.0,
                    area=300.0,
                    source_areas={"PE-HD": 200.0, "PP": 100.0},
                ),
            ),
        )
        assert truth.area_by_polymer() == {"PE-HD": 200.0, "PP": 100.0}

    def test_coeluting_groups_fuse_close_peaks_and_split_distant_ones(self) -> None:
        truth = PyrogramTruth(
            recipe_name="r",
            seed=0,
            blend_fractions={"PE-HD": 1.0},
            components=(
                self._component(0, "a", 100.0),
                self._component(1, "b", 102.0),  # 2 s apart, sigma 2 -> R = 0.25
                self._component(2, "c", 300.0),  # far away
            ),
        )
        groups = truth.coeluting_groups(resolution_threshold=1.0)
        assert groups == ((0, 1), (2,))

    def test_coeluting_groups_of_a_single_component(self) -> None:
        truth = PyrogramTruth(
            recipe_name="r",
            seed=0,
            blend_fractions={"PE-HD": 1.0},
            components=(self._component(0, "a", 100.0),),
        )
        assert truth.coeluting_groups() == ((0,),)

    def test_component_lookup_by_name(self) -> None:
        truth = PyrogramTruth(
            recipe_name="r",
            seed=0,
            blend_fractions={"PE-HD": 1.0},
            components=(self._component(0, "n-alkane C8", 100.0),),
        )
        assert truth.component_by_name("n-alkane C8").index == 0
        with pytest.raises(KeyError, match="no synthetic component named"):
            truth.component_by_name("missing")

    def test_truth_is_json_round_trippable(self, mixed_sample) -> None:  # noqa: ANN001
        """A benchmark dataset has to survive being written to disk."""
        text = round_trip_json(mixed_sample.truth)
        restored = PyrogramTruth.model_validate_json(text)
        assert restored.n_components == mixed_sample.truth.n_components
        assert restored.components[0].name == mixed_sample.truth.components[0].name
        assert restored.total_area == pytest.approx(mixed_sample.truth.total_area)
