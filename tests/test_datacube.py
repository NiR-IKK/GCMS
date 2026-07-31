"""Tests for :class:`PyrogramDataCube`.

The cube is the shared contract between ingestion, preprocessing and every
chemometric engine. Its validation has to be strict, because a silently
transposed or misaligned matrix would produce plausible-looking nonsense rather
than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_schemas.acquisition import SampleMetadata
from data_schemas.enums import SourceFormat
from data_schemas.pyrogram import PreprocessingStep, PyrogramMetadata
from pyrecycle_analytics.core.binning import axis_to_spec
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import DataCubeError
from tests.synthetic_data import SyntheticPyrogram


def _metadata(n_scans: int, mz_axis: np.ndarray, rt: np.ndarray) -> PyrogramMetadata:
    return PyrogramMetadata(
        sample=SampleMetadata(sample_id="unit"),
        source_format=SourceFormat.SYNTHETIC,
        n_scans=n_scans,
        mz_axis=axis_to_spec(mz_axis),
        rt_start_s=float(rt[0]),
        rt_end_s=float(rt[-1]),
    )


def _cube(rt: np.ndarray, mz: np.ndarray, d: np.ndarray) -> PyrogramDataCube:
    return PyrogramDataCube(
        retention_times=rt, mz_axis=mz, intensities=d, metadata=_metadata(rt.size, mz, rt)
    )


class TestValidation:
    def test_shape_mismatch_is_rejected(self) -> None:
        rt = np.arange(5.0)
        mz = np.arange(40.0, 44.0)
        with pytest.raises(DataCubeError, match="does not match axes"):
            _cube(rt, mz, np.zeros((5, 7)))

    def test_transposed_matrix_is_rejected(self) -> None:
        """The classic mistake this validation exists to catch."""
        rt = np.arange(6.0)
        mz = np.arange(40.0, 44.0)
        with pytest.raises(DataCubeError, match="does not match axes"):
            _cube(rt, mz, np.zeros((4, 6)))

    def test_non_monotonic_retention_axis_is_rejected(self) -> None:
        rt = np.array([0.0, 2.0, 1.0, 3.0])
        mz = np.arange(40.0, 42.0)
        with pytest.raises(DataCubeError, match="retention_times must be strictly increasing"):
            _cube(rt, mz, np.zeros((4, 2)))

    def test_duplicate_retention_times_are_rejected(self) -> None:
        rt = np.array([0.0, 1.0, 1.0, 2.0])
        mz = np.arange(40.0, 42.0)
        with pytest.raises(DataCubeError, match="strictly increasing"):
            _cube(rt, mz, np.zeros((4, 2)))

    def test_non_monotonic_mz_axis_is_rejected(self) -> None:
        rt = np.arange(3.0)
        # Metadata is built from a valid axis so that the cube's own validation,
        # not the metadata spec's, is what rejects the scrambled axis.
        metadata = _metadata(3, np.array([41.0, 42.0, 44.0]), rt)
        with pytest.raises(DataCubeError, match="mz_axis must be strictly increasing"):
            PyrogramDataCube(
                retention_times=rt,
                mz_axis=np.array([44.0, 41.0, 42.0]),
                intensities=np.zeros((3, 3)),
                metadata=metadata,
            )

    def test_nan_intensities_are_rejected(self) -> None:
        rt = np.arange(3.0)
        mz = np.arange(40.0, 42.0)
        d = np.zeros((3, 2))
        d[1, 1] = np.nan
        with pytest.raises(DataCubeError, match="1 non-finite value"):
            _cube(rt, mz, d)

    def test_one_dimensional_intensities_are_rejected(self) -> None:
        rt = np.arange(3.0)
        mz = np.arange(40.0, 41.0)
        with pytest.raises(DataCubeError, match="must be 2-D"):
            _cube(rt, mz, np.zeros(3))

    def test_empty_axes_are_rejected(self) -> None:
        with pytest.raises(DataCubeError, match="retention_times must not be empty"):
            PyrogramDataCube(
                retention_times=np.array([]),
                mz_axis=np.array([40.0]),
                intensities=np.zeros((0, 1)),
                metadata=_metadata(1, np.array([40.0]), np.array([1.0, 2.0])),
            )

    def test_integer_input_is_coerced_to_float64(self) -> None:
        cube = _cube(np.arange(3.0), np.arange(40.0, 42.0), np.ones((3, 2), dtype=np.int32))
        assert cube.intensities.dtype == np.float64


class TestDerivedSignals:
    def test_tic_is_the_row_sum(self, tiny_cube: PyrogramDataCube) -> None:
        assert np.array_equal(tiny_cube.tic, np.array([3.0, 12.0, 26.0, 14.0, 5.0, 1.0]))

    def test_base_peak_chromatogram_is_the_row_maximum(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        assert np.array_equal(
            tiny_cube.base_peak_chromatogram, np.array([2.0, 6.0, 12.0, 7.0, 3.0, 1.0])
        )

    def test_base_peak_mz_identifies_the_dominant_channel(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        assert np.all(tiny_cube.base_peak_mz == 43.0)

    def test_total_signal_is_the_matrix_sum(self, tiny_cube: PyrogramDataCube) -> None:
        assert tiny_cube.total_signal == pytest.approx(61.0)

    def test_mean_spectrum_is_the_column_average(self, tiny_cube: PyrogramDataCube) -> None:
        assert np.allclose(
            tiny_cube.mean_spectrum, tiny_cube.intensities.sum(axis=0) / tiny_cube.n_scans
        )

    def test_tic_is_cached_but_consistent(self, tiny_cube: PyrogramDataCube) -> None:
        assert tiny_cube.tic is tiny_cube.tic

    def test_scan_period_reflects_the_axis(self, tiny_cube: PyrogramDataCube) -> None:
        assert tiny_cube.mean_scan_period_s == pytest.approx(1.0)


class TestExtraction:
    def test_eic_selects_a_single_nominal_channel(self, tiny_cube: PyrogramDataCube) -> None:
        assert np.array_equal(tiny_cube.eic(43.0), np.array([2.0, 6.0, 12.0, 7.0, 3.0, 1.0]))

    def test_eic_of_an_absent_mass_is_all_zero(self, tiny_cube: PyrogramDataCube) -> None:
        chromatogram = tiny_cube.eic(500.0)
        assert chromatogram.shape == (6,)
        assert not chromatogram.any()

    def test_eic_sum_adds_several_diagnostic_ions(self, tiny_cube: PyrogramDataCube) -> None:
        summed = tiny_cube.eic_sum([41.0, 43.0])
        assert np.array_equal(summed, tiny_cube.eic(41.0) + tiny_cube.eic(43.0))

    def test_eic_sum_counts_a_repeated_mass_only_once(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        """Overlapping extraction windows must not double-count a channel."""
        assert np.array_equal(tiny_cube.eic_sum([43.0, 43.0]), tiny_cube.eic(43.0))

    def test_wide_tolerance_gathers_neighbouring_channels(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        assert np.array_equal(
            tiny_cube.eic(43.0, tolerance=1.0),
            tiny_cube.eic(42.0) + tiny_cube.eic(43.0) + tiny_cube.eic(44.0),
        )

    def test_mass_spectrum_returns_an_independent_copy(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        spectrum = tiny_cube.mass_spectrum(2)
        spectrum[0] = 999.0
        assert tiny_cube.intensities[2, 0] != 999.0

    def test_mass_spectrum_at_snaps_to_the_nearest_scan(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        assert np.array_equal(tiny_cube.mass_spectrum_at(12.4), tiny_cube.mass_spectrum(2))

    def test_averaged_spectrum_subtracts_a_background_window(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        peak_only = tiny_cube.averaged_spectrum(12.0, 12.0, background_window_s=(15.0, 15.0))
        expected = np.clip(
            tiny_cube.mass_spectrum(2) - tiny_cube.mass_spectrum(5), 0.0, None
        )
        assert np.allclose(peak_only, expected)

    def test_averaged_spectrum_never_returns_negative_intensities(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        result = tiny_cube.averaged_spectrum(15.0, 15.0, background_window_s=(12.0, 12.0))
        assert np.all(result >= 0.0)

    def test_mz_index_finds_the_closest_channel(self, tiny_cube: PyrogramDataCube) -> None:
        assert tiny_cube.mz_index(42.4) == 2
        assert tiny_cube.mz_index(1000.0) == tiny_cube.n_mz - 1

    def test_scan_range_is_half_open_and_covers_the_window(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        start, stop = tiny_cube.scan_range(11.0, 13.0)
        assert (start, stop) == (1, 4)

    def test_scan_range_never_returns_an_empty_slice(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        start, stop = tiny_cube.scan_range(11.4, 11.6)
        assert stop > start

    def test_scan_range_rejects_a_reversed_window(self, tiny_cube: PyrogramDataCube) -> None:
        with pytest.raises(ValueError, match="precedes"):
            tiny_cube.scan_range(14.0, 11.0)


class TestWindowing:
    def test_window_cuts_both_axes_and_records_the_step(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        window = tiny_cube.window(11.0, 13.0, mz_range=(41.0, 43.0))
        assert window.shape == (3, 3)
        assert window.retention_times[0] == 11.0
        assert window.mz_axis[0] == 41.0
        assert window.metadata.preprocessing[-1].name == "window"
        assert window.metadata.n_scans == 3
        assert window.metadata.mz_axis.n_bins == 3

    def test_window_data_matches_the_parent(self, tiny_cube: PyrogramDataCube) -> None:
        window = tiny_cube.window(11.0, 13.0, mz_range=(41.0, 43.0))
        assert np.array_equal(window.intensities, tiny_cube.intensities[1:4, 1:4])

    def test_window_result_is_independent_of_the_parent(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        window = tiny_cube.window(11.0, 13.0)
        window.intensities[0, 0] = 12345.0
        assert tiny_cube.intensities[1, 0] != 12345.0

    def test_window_accepts_the_rt_range_tuple_form(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        """``rt_range_s=(a, b)`` mirrors ``mz_range`` and must match the positional form."""
        positional = tiny_cube.window(11.0, 13.0, mz_range=(41.0, 43.0))
        tuple_form = tiny_cube.window(rt_range_s=(11.0, 13.0), mz_range=(41.0, 43.0))
        assert np.array_equal(tuple_form.intensities, positional.intensities)
        assert np.array_equal(tuple_form.retention_times, positional.retention_times)

    def test_window_without_a_retention_range_keeps_every_scan(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        """The convenient form when only the m/z axis is being restricted."""
        result = tiny_cube.window(mz_range=(41.0, 43.0))
        assert result.n_scans == tiny_cube.n_scans
        assert result.n_mz == 3

    def test_window_rejects_both_retention_forms_at_once(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        with pytest.raises(ValueError, match="not both"):
            tiny_cube.window(11.0, 13.0, rt_range_s=(11.0, 13.0))

    def test_window_rejects_a_half_specified_retention_range(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        with pytest.raises(ValueError, match="must be given together"):
            tiny_cube.window(11.0)

    def test_window_rejects_an_mz_range_with_no_channels(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        with pytest.raises(ValueError, match="selects no channel"):
            tiny_cube.window(10.0, 15.0, mz_range=(500.0, 600.0))

    def test_window_on_a_real_pyrogram_isolates_a_cluster(
        self, coelution_sample: SyntheticPyrogram
    ) -> None:
        """The operation curve resolution uses on every peak cluster."""
        styrene = coelution_sample.truth.component_by_name("styrene")
        window = coelution_sample.cube.window(
            styrene.retention_time_s - 15.0, styrene.retention_time_s + 15.0
        )
        assert 10 < window.n_scans < coelution_sample.cube.n_scans
        assert window.n_mz == coelution_sample.cube.n_mz
        assert window.metadata.rt_start_s >= styrene.retention_time_s - 16.0


class TestTransformations:
    def test_with_intensities_appends_to_the_audit_trail(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        step = PreprocessingStep(name="unit-test", parameters={"factor": 2.0})
        doubled = tiny_cube.with_intensities(tiny_cube.intensities * 2.0, step)
        assert [entry.name for entry in doubled.metadata.preprocessing] == ["unit-test"]
        assert doubled.total_signal == pytest.approx(2.0 * tiny_cube.total_signal)

    def test_with_intensities_leaves_the_original_untouched(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        original = tiny_cube.total_signal
        tiny_cube.with_intensities(tiny_cube.intensities * 5.0)
        assert tiny_cube.total_signal == pytest.approx(original)

    def test_with_intensities_rejects_a_wrong_shape(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        with pytest.raises(DataCubeError, match="!= cube shape"):
            tiny_cube.with_intensities(np.zeros((2, 2)))

    def test_copy_is_deep(self, tiny_cube: PyrogramDataCube) -> None:
        clone = tiny_cube.copy()
        clone.intensities[0, 0] = 777.0
        clone.retention_times[0] = -5.0
        assert tiny_cube.intensities[0, 0] != 777.0
        assert tiny_cube.retention_times[0] == 10.0

    def test_audit_trail_accumulates_across_operations(
        self, tiny_cube: PyrogramDataCube
    ) -> None:
        """Provenance has to survive chaining, or a result is not reproducible."""
        stepped = tiny_cube.with_intensities(
            tiny_cube.intensities, PreprocessingStep(name="first")
        )
        windowed = stepped.window(11.0, 14.0)
        final = windowed.with_intensities(
            windowed.intensities, PreprocessingStep(name="third")
        )
        assert [entry.name for entry in final.metadata.preprocessing] == [
            "first",
            "window",
            "third",
        ]


class TestPersistence:
    def test_npz_round_trip_preserves_data_and_metadata(self, tmp_path, tiny_cube) -> None:  # noqa: ANN001
        path = tiny_cube.save_npz(tmp_path / "cube")
        assert path.suffix == ".npz"
        restored = PyrogramDataCube.load_npz(path)

        assert np.array_equal(restored.intensities, tiny_cube.intensities)
        assert np.array_equal(restored.retention_times, tiny_cube.retention_times)
        assert np.array_equal(restored.mz_axis, tiny_cube.mz_axis)
        assert restored.metadata.sample.sample_id == tiny_cube.metadata.sample.sample_id
        assert restored.metadata.source_format is tiny_cube.metadata.source_format

    def test_npz_round_trip_preserves_the_audit_trail(self, tmp_path, tiny_cube) -> None:  # noqa: ANN001
        stepped = tiny_cube.with_intensities(
            tiny_cube.intensities,
            PreprocessingStep(name="baseline:asls", parameters={"lam": 1e6}),
        )
        restored = PyrogramDataCube.load_npz(stepped.save_npz(tmp_path / "with-steps"))
        assert [entry.name for entry in restored.metadata.preprocessing] == ["baseline:asls"]
        assert restored.metadata.preprocessing[0].parameters["lam"] == 1e6

    def test_extension_is_appended_when_missing(self, tmp_path, tiny_cube) -> None:  # noqa: ANN001
        assert tiny_cube.save_npz(tmp_path / "no-suffix").name == "no-suffix.npz"

    def test_round_trip_of_a_real_pyrogram(self, tmp_path, hdpe_sample) -> None:  # noqa: ANN001
        restored = PyrogramDataCube.load_npz(hdpe_sample.cube.save_npz(tmp_path / "hdpe"))
        assert restored.shape == hdpe_sample.cube.shape
        assert restored.total_signal == pytest.approx(hdpe_sample.cube.total_signal)


class TestSummary:
    def test_summary_is_json_serialisable_and_complete(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        import json

        summary = hdpe_sample.cube.summary()
        assert set(summary) >= {
            "sample_id",
            "source_format",
            "n_scans",
            "n_mz",
            "rt_start_s",
            "rt_end_s",
            "total_signal",
            "preprocessing",
            "warnings",
        }
        json.loads(hdpe_sample.cube.to_json_summary())

    def test_repr_mentions_sample_and_dimensions(self, tiny_cube: PyrogramDataCube) -> None:
        text = repr(tiny_cube)
        assert "tiny" in text
        assert "scans=6" in text
        assert "mz_channels=5" in text
