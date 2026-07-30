"""Tests for the ragged-scan to rectangular-matrix conversion.

Binning happens once, at ingestion, and every later result inherits it. The
property that matters most is **conservation**: total ion current must survive the
conversion, because all quantification is relative to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrecycle_analytics.core.binning import (
    axis_to_spec,
    bin_ragged_scans,
    bin_scan,
    make_nominal_mz_axis,
    make_uniform_mz_axis,
)
from pyrecycle_analytics.exceptions import CorruptRawDataError


class TestNominalAxis:
    def test_covers_the_requested_window_with_integer_centres(self) -> None:
        axis = make_nominal_mz_axis(29.0, 400.0)
        assert axis[0] == 29.0
        assert axis[-1] == 400.0
        assert axis.size == 372
        assert np.array_equal(axis, np.round(axis))

    def test_expands_fractional_bounds_outwards(self) -> None:
        """A window of 28.6-400.2 must include both 28 and 401, not clip them off."""
        axis = make_nominal_mz_axis(28.6, 400.2)
        assert axis[0] == 28.0
        assert axis[-1] == 401.0

    def test_single_mass_window_is_allowed(self) -> None:
        assert np.array_equal(make_nominal_mz_axis(57.0, 57.0), np.array([57.0]))

    @pytest.mark.parametrize(
        ("low", "high", "message"),
        [(0.0, 100.0, "mz_low must be > 0"), (200.0, 100.0, "empty m/z window")],
    )
    def test_rejects_invalid_windows(self, low: float, high: float, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            make_nominal_mz_axis(low, high)


class TestUniformAxis:
    def test_bins_are_evenly_spaced_at_the_requested_width(self) -> None:
        axis = make_uniform_mz_axis(100.0, 110.0, 0.25)
        assert axis.size == 40
        assert np.allclose(np.diff(axis), 0.25)
        assert axis[0] == pytest.approx(100.125)

    @pytest.mark.parametrize(
        ("low", "high", "width", "message"),
        [
            (100.0, 110.0, 0.0, "bin_width must be > 0"),
            (110.0, 100.0, 1.0, "empty m/z window"),
        ],
    )
    def test_rejects_invalid_parameters(
        self, low: float, high: float, width: float, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            make_uniform_mz_axis(low, high, width)


class TestAxisToSpec:
    def test_recognises_a_nominal_grid(self) -> None:
        spec = axis_to_spec(make_nominal_mz_axis(35.0, 300.0))
        assert spec.is_nominal
        assert spec.bin_width == pytest.approx(1.0)
        assert spec.n_bins == 266
        # Edges sit half a bin outside the outermost centres.
        assert spec.mz_low == pytest.approx(34.5)
        assert spec.mz_high == pytest.approx(300.5)

    def test_recognises_a_high_resolution_grid(self) -> None:
        spec = axis_to_spec(make_uniform_mz_axis(100.0, 120.0, 0.02))
        assert not spec.is_nominal
        assert spec.bin_width == pytest.approx(0.02)

    def test_rejects_an_unsorted_axis(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            axis_to_spec(np.array([50.0, 49.0, 51.0]))

    def test_rejects_an_empty_axis(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            axis_to_spec(np.array([]))


class TestBinScan:
    def test_places_ions_in_their_nearest_integer_channel(self) -> None:
        axis = make_nominal_mz_axis(40.0, 45.0)
        spectrum = bin_scan(np.array([41.0, 43.2, 44.9]), np.array([10.0, 20.0, 30.0]), axis)
        assert np.array_equal(spectrum, np.array([0.0, 10.0, 0.0, 20.0, 0.0, 30.0]))

    def test_sums_rather_than_overwrites_within_a_bin(self) -> None:
        """Two ions in one channel must add: TIC conservation depends on it."""
        axis = make_nominal_mz_axis(40.0, 42.0)
        spectrum = bin_scan(np.array([41.1, 41.3, 41.4]), np.array([1.0, 2.0, 3.0]), axis)
        assert spectrum[1] == pytest.approx(6.0)

    def test_conserves_total_intensity_for_in_range_ions(self) -> None:
        rng = np.random.default_rng(0)
        mz_values = rng.uniform(50.0, 300.0, 500)
        intensities = rng.uniform(1.0, 1e5, 500)
        axis = make_nominal_mz_axis(40.0, 320.0)
        assert bin_scan(mz_values, intensities, axis).sum() == pytest.approx(intensities.sum())

    def test_drops_ions_outside_the_grid(self) -> None:
        axis = make_nominal_mz_axis(50.0, 60.0)
        spectrum = bin_scan(
            np.array([30.0, 55.0, 900.0]), np.array([100.0, 7.0, 100.0]), axis
        )
        assert spectrum.sum() == pytest.approx(7.0)

    def test_empty_scan_gives_an_all_zero_spectrum(self) -> None:
        axis = make_nominal_mz_axis(50.0, 60.0)
        spectrum = bin_scan(np.array([]), np.array([]), axis)
        assert spectrum.shape == (11,)
        assert not spectrum.any()

    def test_value_exactly_on_a_bin_edge_goes_to_the_upper_bin(self) -> None:
        """Half-open ``[edge, next_edge)`` convention, asserted explicitly."""
        axis = make_nominal_mz_axis(40.0, 42.0)
        spectrum = bin_scan(np.array([40.5]), np.array([9.0]), axis)
        assert spectrum[1] == pytest.approx(9.0)

    def test_mismatched_array_lengths_are_reported_as_corrupt_data(self) -> None:
        axis = make_nominal_mz_axis(40.0, 50.0)
        with pytest.raises(CorruptRawDataError, match="3 m/z values but 2 intensities"):
            bin_scan(np.array([41.0, 42.0, 43.0]), np.array([1.0, 2.0]), axis)


class TestBinRaggedScans:
    def test_builds_a_rectangular_matrix_from_variable_length_scans(self) -> None:
        axis = make_nominal_mz_axis(40.0, 44.0)
        scans = [
            (np.array([41.0]), np.array([5.0])),
            (np.array([41.0, 43.0, 44.0]), np.array([1.0, 2.0, 3.0])),
            (np.array([]), np.array([])),
        ]
        matrix = bin_ragged_scans(scans, axis)
        assert matrix.shape == (3, 5)
        assert matrix[0].sum() == pytest.approx(5.0)
        assert matrix[1].sum() == pytest.approx(6.0)
        assert matrix[2].sum() == pytest.approx(0.0)

    def test_conserves_total_signal_across_all_scans(self) -> None:
        rng = np.random.default_rng(3)
        axis = make_nominal_mz_axis(30.0, 200.0)
        scans = []
        expected = 0.0
        for _ in range(20):
            count = int(rng.integers(1, 40))
            mz_values = rng.uniform(35.0, 195.0, count)
            intensities = rng.uniform(1.0, 1e4, count)
            expected += intensities.sum()
            scans.append((mz_values, intensities))
        assert bin_ragged_scans(scans, axis).sum() == pytest.approx(expected)

    def test_result_is_float64_regardless_of_input_dtype(self) -> None:
        """Vendor files store float32; accumulating in float32 loses TIC precision."""
        axis = make_nominal_mz_axis(40.0, 44.0)
        scans = [(np.array([41.0], dtype=np.float32), np.array([1.0], dtype=np.float32))]
        assert bin_ragged_scans(scans, axis).dtype == np.float64
