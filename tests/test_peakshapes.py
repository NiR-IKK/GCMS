"""Tests for the chromatographic peak-shape models.

These are the functions everything else is built on: if the EMG is not
area-normalised, every quantitative claim downstream inherits the error, and if it
loses precision for tailing peaks, the polar markers (acids, caprolactam) are
simulated and fitted wrongly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyrecycle_analytics.core.peakshapes import (
    chromatographic_resolution,
    emg,
    emg_apex_time,
    fwhm_gaussian,
    gaussian,
    profile_matrix,
)


def _grid(center: float, sigma: float, tau: float = 0.0, span: float = 40.0) -> np.ndarray:
    """Dense grid wide enough that numeric integration captures the whole peak."""
    half_width = span * sigma + 12.0 * tau
    return np.linspace(center - half_width, center + half_width, 400001)


class TestGaussian:
    def test_integrates_to_requested_area(self) -> None:
        t = _grid(100.0, 2.0)
        profile = gaussian(t, center=100.0, sigma=2.0, area=1234.5)
        assert np.trapezoid(profile, t) == pytest.approx(1234.5, rel=1e-9)

    def test_apex_is_at_the_center(self) -> None:
        t = _grid(100.0, 2.0)
        profile = gaussian(t, center=100.0, sigma=2.0)
        assert t[int(np.argmax(profile))] == pytest.approx(100.0, abs=1e-3)

    def test_fwhm_matches_analytic_value(self) -> None:
        sigma = 3.0
        t = _grid(0.0, sigma)
        profile = gaussian(t, center=0.0, sigma=sigma)
        above_half = t[profile >= profile.max() / 2.0]
        measured = above_half[-1] - above_half[0]
        assert measured == pytest.approx(fwhm_gaussian(sigma), rel=1e-3)

    def test_rejects_non_positive_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma must be > 0"):
            gaussian(np.linspace(0, 1, 10), center=0.5, sigma=0.0)


class TestEmg:
    @pytest.mark.parametrize("tau", [0.05, 0.5, 2.0, 6.0, 20.0])
    def test_integrates_to_requested_area_across_tailing_range(self, tau: float) -> None:
        """Area normalisation must hold from nearly symmetric to grossly tailing.

        Peak *area* is the quantitative output of the whole platform, so an EMG
        whose integral drifts with tau would silently bias every marker ratio.
        """
        sigma = 2.0
        t = _grid(200.0, sigma, tau)
        profile = emg(t, center=200.0, sigma=sigma, tau=tau, area=1000.0)
        assert np.trapezoid(profile, t) == pytest.approx(1000.0, rel=1e-6)

    def test_reduces_to_gaussian_as_tau_vanishes(self) -> None:
        t = np.linspace(80.0, 120.0, 2001)
        assert np.allclose(
            emg(t, center=100.0, sigma=2.0, tau=0.0, area=50.0),
            gaussian(t, center=100.0, sigma=2.0, area=50.0),
        )

    def test_is_everywhere_finite_and_non_negative_for_strong_tailing(self) -> None:
        """The stable erfcx form must not produce inf/nan or negative lobes.

        The naive ``exp(...) * erfc(...)`` formulation overflows here; this is the
        regression guard for that.
        """
        t = np.linspace(0.0, 600.0, 60001)
        profile = emg(t, center=100.0, sigma=1.0, tau=40.0, area=1.0)
        assert np.all(np.isfinite(profile))
        assert np.all(profile >= 0.0)
        assert profile.max() > 0.0

    def test_tailing_shifts_apex_later_and_adds_skew(self) -> None:
        t = np.linspace(150.0, 400.0, 250001)
        symmetric = emg(t, center=200.0, sigma=3.0, tau=0.0)
        tailing = emg(t, center=200.0, sigma=3.0, tau=6.0)

        symmetric_apex = t[int(np.argmax(symmetric))]
        tailing_apex = t[int(np.argmax(tailing))]
        assert tailing_apex > symmetric_apex

        # More mass after the apex than before it — the definition of tailing.
        after = np.trapezoid(tailing[t >= tailing_apex], t[t >= tailing_apex])
        before = np.trapezoid(tailing[t <= tailing_apex], t[t <= tailing_apex])
        assert after > before

    def test_apex_estimate_tracks_the_numeric_maximum(self) -> None:
        sigma, tau = 2.0, 3.0
        t = np.linspace(150.0, 300.0, 150001)
        numeric_apex = t[int(np.argmax(emg(t, center=200.0, sigma=sigma, tau=tau)))]
        assert emg_apex_time(200.0, sigma, tau) == pytest.approx(numeric_apex, abs=1.5)

    def test_apex_estimate_is_the_center_without_tailing(self) -> None:
        assert emg_apex_time(123.0, 2.0, 0.0) == pytest.approx(123.0)

    @pytest.mark.parametrize(
        ("sigma", "tau", "message"),
        [(0.0, 1.0, "sigma must be > 0"), (1.0, -1.0, "tau must be >= 0")],
    )
    def test_rejects_invalid_shape_parameters(
        self, sigma: float, tau: float, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            emg(np.linspace(0, 10, 11), center=5.0, sigma=sigma, tau=tau)


class TestProfileMatrix:
    def test_shape_and_column_areas(self) -> None:
        t = np.linspace(0.0, 300.0, 3001)
        centers = np.array([50.0, 120.0, 200.0])
        areas = np.array([10.0, 20.0, 30.0])
        profiles = profile_matrix(
            t, centers, sigmas=np.full(3, 2.0), taus=np.full(3, 0.5), areas=areas
        )
        assert profiles.shape == (t.size, 3)
        for column, expected in enumerate(areas):
            assert np.trapezoid(profiles[:, column], t) == pytest.approx(expected, rel=1e-4)

    def test_defaults_to_unit_area_and_no_tailing(self) -> None:
        t = np.linspace(0.0, 100.0, 20001)
        profiles = profile_matrix(t, np.array([50.0]), np.array([2.0]))
        assert np.trapezoid(profiles[:, 0], t) == pytest.approx(1.0, rel=1e-6)
        assert np.allclose(profiles[:, 0], gaussian(t, 50.0, 2.0))

    @pytest.mark.parametrize("bad_field", ["sigmas", "taus", "areas"])
    def test_rejects_mismatched_parameter_lengths(self, bad_field: str) -> None:
        t = np.linspace(0.0, 100.0, 101)
        kwargs: dict[str, np.ndarray] = {
            "sigmas": np.full(2, 2.0),
            "taus": np.zeros(2),
            "areas": np.ones(2),
        }
        kwargs[bad_field] = np.ones(3)
        with pytest.raises(ValueError, match="expected 2"):
            profile_matrix(t, np.array([10.0, 20.0]), **kwargs)  # type: ignore[arg-type]


class TestResolution:
    def test_baseline_resolved_pair_exceeds_one_point_five(self) -> None:
        assert chromatographic_resolution(100.0, 2.0, 115.0, 2.0) > 1.5

    def test_fused_pair_falls_below_one(self) -> None:
        assert chromatographic_resolution(100.0, 2.0, 103.0, 2.0) < 1.0

    def test_is_symmetric_in_its_arguments(self) -> None:
        forward = chromatographic_resolution(100.0, 2.0, 110.0, 3.0)
        reverse = chromatographic_resolution(110.0, 3.0, 100.0, 2.0)
        assert forward == pytest.approx(reverse)

    def test_zero_width_peaks_are_infinitely_resolved(self) -> None:
        assert chromatographic_resolution(100.0, 0.0, 101.0, 0.0) == float("inf")
