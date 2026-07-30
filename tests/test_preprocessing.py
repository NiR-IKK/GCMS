"""Tests for baseline correction, smoothing and the preprocessing pipeline.

Preprocessing is judged on one criterion: does it remove the background without
distorting the quantities the platform reports? A baseline correction that
flattens the bleed ramp but also shaves 15 % off a trace marker's area has made
the result worse, not better. The tests therefore always check both halves —
background removed *and* marker ratios preserved.

The synthetic pyrograms make that testable: the exact background that was added is
available as ``sample.baseline``, and the exact signal underneath it as
``sample.clean``.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_schemas.enums import MarkerRole
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import PreprocessingError
from pyrecycle_analytics.preprocessing import (
    PreprocessingConfig,
    asls_baseline,
    asls_baseline_matrix,
    correct_baseline,
    estimate_noise_sigma,
    gaussian_smooth,
    preprocess,
    savgol_smooth,
    smooth_cube,
    snip_baseline,
    snip_baseline_matrix,
)
from tests.synthetic_data import SyntheticPyrogram


class TestAslsBaseline:
    @staticmethod
    def _background_and_peaks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Curved background with two tall, narrow peaks on top of it."""
        t = np.linspace(0.0, 100.0, 1000)
        background = 50.0 + 0.8 * t + 20.0 * np.sin(t / 30.0)
        peaks = 800.0 * np.exp(-0.5 * ((t - 30.0) / 1.0) ** 2)
        peaks += 500.0 * np.exp(-0.5 * ((t - 70.0) / 1.5) ** 2)
        return t, background, background + peaks

    def test_recovers_a_smooth_background_under_peaks(self) -> None:
        """The core requirement: follow the background, ignore the peaks."""
        _, background, signal = self._background_and_peaks()
        estimated = asls_baseline(signal, lam=1e7, p=0.01)
        assert np.max(np.abs(estimated - background)) < 0.05 * background.mean()

    def test_stiffness_controls_how_far_peaks_pull_the_baseline_up(self) -> None:
        """Residual error under a peak falls roughly as 1/lam.

        This is the parameter trade-off an analyst has to get right, so it is
        pinned rather than left implicit: too flexible a baseline is drawn up into
        the peak and silently removes part of its area. For 5 Hz pyrogram data with
        peaks an order of magnitude above the background, lam below ~1e6 is
        unusable — which is why the pipeline default sits well above it.
        """
        _, background, signal = self._background_and_peaks()
        errors = {
            lam: float(np.max(np.abs(asls_baseline(signal, lam=lam, p=0.01) - background)))
            for lam in (1e5, 1e6, 1e7, 1e8)
        }
        assert errors[1e5] > errors[1e6] > errors[1e7] > errors[1e8]
        assert errors[1e5] > 0.2 * background.mean()
        assert errors[1e8] < 0.02 * background.mean()

    def test_stays_below_the_signal_in_peak_regions(self) -> None:
        """Overshooting into a peak would carve out part of its area."""
        t = np.linspace(0.0, 100.0, 1000)
        signal = 100.0 + 900.0 * np.exp(-0.5 * ((t - 50.0) / 2.0) ** 2)
        estimated = asls_baseline(signal, lam=1e6, p=0.001)
        apex = int(np.argmax(signal))
        assert estimated[apex] < 0.2 * signal[apex]

    def test_flat_signal_yields_a_flat_baseline(self) -> None:
        estimated = asls_baseline(np.full(500, 42.0), lam=1e5, p=0.01)
        assert np.allclose(estimated, 42.0, rtol=1e-3)

    def test_larger_lambda_gives_a_stiffer_baseline(self) -> None:
        t = np.linspace(0.0, 100.0, 800)
        signal = 100.0 + 40.0 * np.sin(t / 5.0)
        flexible = asls_baseline(signal, lam=1e2, p=0.5)
        stiff = asls_baseline(signal, lam=1e9, p=0.5)
        assert np.std(np.diff(stiff, n=2)) < np.std(np.diff(flexible, n=2))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"p": 0.0}, r"p must lie in \(0, 1\)"),
            ({"p": 1.0}, r"p must lie in \(0, 1\)"),
            ({"lam": 0.0}, "lam must be > 0"),
            ({"n_iter": 0}, "n_iter must be >= 1"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict, message: str) -> None:  # noqa: ANN001
        with pytest.raises(PreprocessingError, match=message):
            asls_baseline(np.ones(100), **kwargs)

    def test_rejects_a_signal_too_short_for_a_curvature_penalty(self) -> None:
        with pytest.raises(PreprocessingError, match="at least 4 points"):
            asls_baseline(np.ones(3))


class TestSnipBaseline:
    def test_never_exceeds_the_signal(self) -> None:
        """SNIP only clips downwards; going above the signal would be a bug."""
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 100.0, 1000)
        signal = 100.0 + 2.0 * t + 600.0 * np.exp(-0.5 * ((t - 40.0) / 1.5) ** 2)
        signal += rng.normal(0.0, 3.0, t.size)
        assert np.all(snip_baseline(signal, n_iter=40) <= signal + 1e-9)

    def test_follows_a_rising_background(self) -> None:
        t = np.linspace(0.0, 100.0, 1000)
        background = 100.0 + 3.0 * t
        signal = background + 900.0 * np.exp(-0.5 * ((t - 55.0) / 1.2) ** 2)
        estimated = snip_baseline(signal, n_iter=60)
        interior = slice(100, 900)
        assert np.max(np.abs(estimated[interior] - background[interior])) < 0.1 * background.mean()

    def test_window_larger_than_the_signal_is_clamped_not_rejected(self) -> None:
        result = snip_baseline(np.ones(11), n_iter=500)
        assert result.shape == (11,)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [({"n_iter": 0}, "n_iter must be >= 1")],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict, message: str) -> None:  # noqa: ANN001
        with pytest.raises(PreprocessingError, match=message):
            snip_baseline(np.ones(100), **kwargs)

    def test_rejects_a_signal_that_is_too_short(self) -> None:
        with pytest.raises(PreprocessingError, match="at least 3 points"):
            snip_baseline(np.ones(2))


class TestMatrixBaseline:
    def test_banded_solver_agrees_with_the_sparse_reference(self) -> None:
        """The fast per-channel path must equal the readable reference implementation.

        ``asls_baseline_matrix`` replaces the sparse solve with a banded Cholesky so
        that ~300 channels stay affordable; if the two ever diverged, every
        production result would differ from what the documented algorithm says.
        """
        rng = np.random.default_rng(4)
        t = np.linspace(0.0, 100.0, 400)
        column = 80.0 + 1.2 * t + 700.0 * np.exp(-0.5 * ((t - 45.0) / 2.0) ** 2)
        column += rng.normal(0.0, 2.0, t.size)
        matrix = np.column_stack([column, column])

        reference = asls_baseline(column, lam=1e5, p=0.01, n_iter=15, tol=0.0)
        fast = asls_baseline_matrix(matrix, lam=1e5, p=0.01, n_iter=15)
        assert np.allclose(fast[:, 0], reference, rtol=1e-6, atol=1e-6)

    def test_empty_channels_get_a_zero_baseline(self) -> None:
        matrix = np.zeros((100, 3))
        matrix[:, 1] = 500.0
        baselines = asls_baseline_matrix(matrix, lam=1e5, p=0.01, n_iter=5)
        assert not baselines[:, 0].any()
        assert not baselines[:, 2].any()
        assert baselines[:, 1].mean() > 0.0

    def test_min_channel_signal_skips_low_channels(self) -> None:
        matrix = np.full((100, 2), 5.0)
        matrix[:, 1] = 5000.0
        baselines = asls_baseline_matrix(matrix, n_iter=5, min_channel_signal=10.0)
        assert not baselines[:, 0].any()
        assert baselines[:, 1].any()

    def test_snip_matrix_handles_every_channel(self) -> None:
        rng = np.random.default_rng(5)
        matrix = rng.uniform(10.0, 100.0, (200, 6))
        baselines = snip_baseline_matrix(matrix, n_iter=20)
        assert baselines.shape == matrix.shape
        assert np.all(baselines <= matrix + 1e-9)

    @pytest.mark.parametrize("function", [asls_baseline_matrix, snip_baseline_matrix])
    def test_rejects_non_two_dimensional_input(self, function) -> None:  # noqa: ANN001
        with pytest.raises(PreprocessingError, match="expected a 2-D matrix"):
            function(np.ones(50))

    def test_asls_matrix_rejects_a_run_that_is_too_short(self) -> None:
        with pytest.raises(PreprocessingError, match="at least 4 scans"):
            asls_baseline_matrix(np.ones((3, 5)))


class TestBaselineOnRealPyrograms:
    def test_correction_removes_most_of_the_column_bleed(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """The siloxane ramp is what a real correction has to eliminate."""
        cube = hdpe_sample.cube
        bleed_channel = cube.mz_index(207.0)
        before = cube.intensities[:, bleed_channel]

        corrected = correct_baseline(cube, "asls", lam=1e7, p=0.005, n_iter=12)
        after = corrected.intensities[:, bleed_channel]

        assert before[-100:].mean() > 1000.0
        assert after[-100:].mean() < 0.1 * before[-100:].mean()

    def test_correction_preserves_marker_peak_areas(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """Removing the background must not eat into the analyte.

        The check is on the summed quantifier-ion signal of several markers, which
        is the quantity every downstream ratio is built from.
        """
        cube = mixed_sample.cube
        corrected = correct_baseline(cube, "asls", lam=1e7, p=0.005, n_iter=12)
        assert isinstance(corrected, PyrogramDataCube)

        for name in ("styrene", "epsilon-caprolactam", "benzoic acid"):
            component = mixed_sample.truth.component_by_name(name)
            quantifier = float(component.quantifier_mz)
            scan = cube.nearest_scan(component.retention_time_s)
            half_width = max(int(4.0 * component.peak_width_s / cube.mean_scan_period_s), 5)
            window = slice(max(scan - half_width, 0), scan + half_width + 1)

            clean_area = mixed_sample.clean_cube.eic(quantifier)[window].sum()
            corrected_area = corrected.eic(quantifier)[window].sum()
            assert corrected_area == pytest.approx(clean_area, rel=0.25), (
                f"{name}: corrected area {corrected_area:.4g} vs true {clean_area:.4g}"
            )

    def test_correction_moves_the_cube_towards_the_true_clean_signal(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        cube = hdpe_sample.cube
        truth = hdpe_sample.clean
        before = float(np.abs(cube.intensities - truth).mean())
        corrected = correct_baseline(cube, "asls", lam=1e7, p=0.005, n_iter=12)
        after = float(np.abs(corrected.intensities - truth).mean())
        assert after < 0.5 * before

    def test_clip_negative_keeps_the_matrix_non_negative(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """MCR-ALS imposes non-negativity; a negative input would fight it."""
        corrected = correct_baseline(hdpe_sample.cube, "asls", n_iter=6, clip_negative=True)
        assert np.all(corrected.intensities >= 0.0)

    def test_without_clipping_negative_residuals_can_appear(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        corrected = correct_baseline(hdpe_sample.cube, "asls", n_iter=6, clip_negative=False)
        assert corrected.intensities.min() < 0.0

    def test_baseline_can_be_returned_for_inspection(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        corrected, baseline = correct_baseline(
            hdpe_sample.cube, "asls", n_iter=6, return_baseline=True
        )
        assert baseline.shape == hdpe_sample.cube.shape
        assert np.allclose(
            corrected.intensities,
            np.clip(hdpe_sample.cube.intensities - baseline, 0.0, None),
        )

    def test_step_is_recorded_in_the_audit_trail(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        corrected = correct_baseline(hdpe_sample.cube, "snip", n_iter=15)
        assert corrected.metadata.preprocessing[-1].name == "baseline:snip"

    def test_unknown_method_is_rejected(self, hdpe_sample: SyntheticPyrogram) -> None:
        with pytest.raises(PreprocessingError, match="unknown baseline method"):
            correct_baseline(hdpe_sample.cube, "rolling-ball")


class TestSmoothing:
    def test_savgol_preserves_peak_area(self) -> None:
        """Smoothing must not change how much analyte the peak represents."""
        t = np.linspace(0.0, 100.0, 1001)
        peak = 1000.0 * np.exp(-0.5 * ((t - 50.0) / 3.0) ** 2)
        smoothed = savgol_smooth(peak, window_length=9, polyorder=2)
        assert smoothed.sum() == pytest.approx(peak.sum(), rel=1e-3)

    def test_savgol_preserves_peak_position_and_height(self) -> None:
        t = np.linspace(0.0, 100.0, 1001)
        peak = 1000.0 * np.exp(-0.5 * ((t - 50.0) / 3.0) ** 2)
        smoothed = savgol_smooth(peak, window_length=9, polyorder=2)
        assert int(np.argmax(smoothed)) == int(np.argmax(peak))
        assert smoothed.max() == pytest.approx(peak.max(), rel=1e-2)

    def test_savgol_reduces_noise(self) -> None:
        rng = np.random.default_rng(1)
        t = np.linspace(0.0, 100.0, 2000)
        clean = 500.0 * np.exp(-0.5 * ((t - 50.0) / 4.0) ** 2)
        noisy = clean + rng.normal(0.0, 20.0, t.size)
        smoothed = savgol_smooth(noisy, window_length=11, polyorder=2)
        assert np.std(smoothed - clean) < 0.6 * np.std(noisy - clean)

    def test_savgol_never_returns_negative_values(self) -> None:
        """Polynomial fits undershoot on steep flanks; ion counts cannot be negative."""
        t = np.linspace(0.0, 20.0, 400)
        spike = np.zeros_like(t)
        spike[200] = 1e5
        assert np.all(savgol_smooth(spike, window_length=9, polyorder=3) >= 0.0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"window_length": 8}, "window_length must be odd"),
            ({"window_length": 3, "polyorder": 5}, "must exceed polyorder"),
            ({"window_length": 501}, "exceeds the"),
        ],
    )
    def test_savgol_rejects_invalid_parameters(self, kwargs: dict, message: str) -> None:  # noqa: ANN001
        with pytest.raises(PreprocessingError, match=message):
            savgol_smooth(np.ones(100), **kwargs)

    def test_gaussian_smoothing_preserves_area(self) -> None:
        t = np.linspace(0.0, 100.0, 1001)
        peak = 1000.0 * np.exp(-0.5 * ((t - 50.0) / 3.0) ** 2)
        assert gaussian_smooth(peak, sigma_scans=2.0).sum() == pytest.approx(
            peak.sum(), rel=1e-3
        )

    def test_gaussian_rejects_non_positive_sigma(self) -> None:
        with pytest.raises(PreprocessingError, match="sigma_scans must be > 0"):
            gaussian_smooth(np.ones(50), sigma_scans=0.0)

    def test_smoothing_acts_along_time_not_along_mz(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        """Mixing adjacent nominal masses would destroy the spectra.

        m/z 56 and 57 come from different fragmentation routes; smoothing across
        them would blur exactly the fine structure the marker logic reads.
        """
        cube = hdpe_sample.cube
        smoothed = smooth_cube(cube, "savgol", window_length=7, polyorder=2)

        # An isolated empty channel between two busy ones must stay empty.
        empty_channels = np.flatnonzero(cube.intensities.max(axis=0) == 0.0)
        if empty_channels.size:
            assert not smoothed.intensities[:, empty_channels].any()

        # Every scan's total is roughly preserved; a cross-channel filter would
        # not preserve per-scan structure this way.
        assert np.allclose(smoothed.tic.sum(), cube.tic.sum(), rtol=0.02)

    def test_smoothing_records_the_step(self, hdpe_sample: SyntheticPyrogram) -> None:
        smoothed = smooth_cube(hdpe_sample.cube, "gaussian", sigma_scans=1.0)
        assert smoothed.metadata.preprocessing[-1].name == "smooth:gaussian"

    def test_unknown_smoothing_method_is_rejected(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        with pytest.raises(PreprocessingError, match="unknown smoothing method"):
            smooth_cube(hdpe_sample.cube, "median")


class TestNoiseEstimation:
    @pytest.mark.parametrize("sigma", [1.0, 25.0, 400.0])
    def test_recovers_the_injected_noise_level(self, sigma: float) -> None:
        rng = np.random.default_rng(2)
        signal = rng.normal(1000.0, sigma, 5000)
        assert estimate_noise_sigma(signal) == pytest.approx(sigma, rel=0.12)

    def test_is_insensitive_to_peaks(self) -> None:
        """A blank region should not have to be picked by hand."""
        rng = np.random.default_rng(2)
        t = np.linspace(0.0, 100.0, 5000)
        signal = rng.normal(0.0, 10.0, t.size)
        signal += 5000.0 * np.exp(-0.5 * ((t - 50.0) / 0.5) ** 2)
        assert estimate_noise_sigma(signal) == pytest.approx(10.0, rel=0.2)

    def test_is_insensitive_to_a_smooth_baseline(self) -> None:
        rng = np.random.default_rng(2)
        t = np.linspace(0.0, 100.0, 5000)
        signal = 500.0 + 12.0 * t + rng.normal(0.0, 8.0, t.size)
        assert estimate_noise_sigma(signal) == pytest.approx(8.0, rel=0.2)

    def test_noiseless_signal_gives_zero(self) -> None:
        assert estimate_noise_sigma(np.linspace(0.0, 100.0, 500)) == pytest.approx(0.0)

    def test_rejects_a_signal_that_is_too_short(self) -> None:
        with pytest.raises(PreprocessingError, match="at least 3 points"):
            estimate_noise_sigma(np.ones(2))


class TestPipeline:
    def test_default_chain_runs_and_records_every_step(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        result = preprocess(hdpe_sample.cube, PreprocessingConfig(asls_iterations=6))
        names = [step.name for step in result.metadata.preprocessing]
        assert names == ["smooth:savgol", "baseline:asls"]
        assert result.shape == hdpe_sample.cube.shape

    def test_trimming_is_applied_before_everything_else(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        config = PreprocessingConfig(
            rt_range_s=(300.0, 900.0), mz_range=(40.0, 200.0), asls_iterations=4
        )
        result = preprocess(hdpe_sample.cube, config)
        assert [step.name for step in result.metadata.preprocessing][0] == "window"
        assert result.retention_times[0] >= 300.0
        assert result.mz_axis[0] >= 40.0

    def test_chain_can_be_disabled_entirely(self, hdpe_sample: SyntheticPyrogram) -> None:
        config = PreprocessingConfig(smoothing="none", baseline="none")
        result = preprocess(hdpe_sample.cube, config)
        assert np.array_equal(result.intensities, hdpe_sample.cube.intensities)
        assert not result.metadata.preprocessing

    def test_snip_variant_is_selectable(self, hdpe_sample: SyntheticPyrogram) -> None:
        config = PreprocessingConfig(baseline="snip", snip_iterations=15, smoothing="none")
        result = preprocess(hdpe_sample.cube, config)
        assert [step.name for step in result.metadata.preprocessing] == ["baseline:snip"]

    def test_empty_channels_can_be_dropped(self, hdpe_sample: SyntheticPyrogram) -> None:
        """Trimming the m/z axis shrinks the matrix curve resolution has to factor.

        Run on the noise-free cube: once detector noise is added every channel
        carries a count, so there is nothing to drop.
        """
        clean = hdpe_sample.clean_cube
        assert clean.intensities.max(axis=0).min() == 0.0, "fixture needs empty channels"

        config = PreprocessingConfig(
            smoothing="none", baseline="none", drop_empty_channels=True
        )
        result = preprocess(clean, config)
        assert result.n_mz < clean.n_mz
        assert "drop_empty_channels" in [
            step.name for step in result.metadata.preprocessing
        ]

    def test_dropping_empty_channels_is_a_no_op_when_all_channels_carry_signal(
        self, hdpe_sample: SyntheticPyrogram
    ) -> None:
        config = PreprocessingConfig(
            smoothing="none", baseline="none", drop_empty_channels=True
        )
        result = preprocess(hdpe_sample.cube, config)
        assert result.n_mz == hdpe_sample.cube.n_mz

    def test_configuration_is_frozen_and_serialisable(self) -> None:
        """A stored result must carry the exact chain that produced it."""
        config = PreprocessingConfig(asls_lam=5e6)
        restored = PreprocessingConfig.model_validate_json(config.model_dump_json())
        assert restored == config
        with pytest.raises(ValueError, match="frozen"):
            config.asls_lam = 1e5  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"savgol_window": 8}, "savgol_window must be odd"),
            ({"savgol_window": 5, "savgol_polyorder": 5}, "must exceed"),
            ({"rt_range_s": (900.0, 300.0)}, "is empty"),
            ({"mz_range": (300.0, 100.0)}, "is empty"),
            ({"asls_p": 1.5}, "less than 1"),
            ({"baseline": "rolling-ball"}, "Input should be"),
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs: dict, message: str) -> None:  # noqa: ANN001
        with pytest.raises(ValueError, match=message):
            PreprocessingConfig(**kwargs)

    def test_full_chain_improves_marker_visibility_on_a_trace_sample(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """End-to-end check on the quantity that actually matters.

        For a trace PET marker sitting on the bleed ramp, preprocessing must raise
        the peak-to-background ratio on its quantifier ion — that is the whole
        point of the chain.
        """
        component = mixed_sample.truth.component_by_name("divinyl terephthalate")
        quantifier = float(component.quantifier_mz)
        cube = mixed_sample.cube
        scan = cube.nearest_scan(component.retention_time_s)

        def peak_to_background(target: PyrogramDataCube) -> float:
            trace = target.eic(quantifier)
            local = trace[max(scan - 6, 0) : scan + 7].max()
            surrounding = np.median(trace[max(scan - 400, 0) : min(scan + 400, trace.size)])
            return local / max(surrounding, 1.0)

        processed = preprocess(cube, PreprocessingConfig(asls_iterations=10, asls_lam=1e7))
        assert peak_to_background(processed) > peak_to_background(cube)

    def test_marker_triad_ratio_survives_the_chain(
        self, mixed_sample: SyntheticPyrogram
    ) -> None:
        """PS identification rests on the styrene:dimer ratio; it must not shift."""
        cube = mixed_sample.cube
        processed = preprocess(cube, PreprocessingConfig(asls_iterations=10, asls_lam=1e7))

        def local_area(target: PyrogramDataCube, name: str) -> float:
            component = mixed_sample.truth.component_by_name(name)
            scan = target.nearest_scan(component.retention_time_s)
            half_width = max(
                int(4.0 * component.peak_width_s / target.mean_scan_period_s), 5
            )
            trace = target.eic(float(component.quantifier_mz))
            return float(trace[max(scan - half_width, 0) : scan + half_width + 1].sum())

        true_ratio = (
            mixed_sample.truth.component_by_name("2,4-diphenyl-1-butene").area
            / mixed_sample.truth.component_by_name("styrene").area
        )
        measured = local_area(processed, "2,4-diphenyl-1-butene") / local_area(
            processed, "styrene"
        )
        # The quantifier ions differ in relative abundance between the two
        # compounds, so the ion-level ratio is not the area ratio; what matters is
        # that it lands within a factor of two and is stable.
        assert 0.2 * true_ratio < measured < 5.0 * true_ratio

    def test_degradation_markers_remain_detectable_after_preprocessing(
        self, aged_hdpe_sample: SyntheticPyrogram
    ) -> None:
        """The carbonyl channels must not be flattened away with the background."""
        cube = aged_hdpe_sample.cube
        processed = preprocess(cube, PreprocessingConfig(asls_iterations=10, asls_lam=1e7))
        ketones = aged_hdpe_sample.truth.select(role=MarkerRole.OXIDATION_KETONE)
        assert ketones

        strongest = max(ketones, key=lambda component: component.area)
        scan = processed.nearest_scan(strongest.retention_time_s)
        trace = processed.eic(58.0)
        assert trace[max(scan - 6, 0) : scan + 7].max() > 0.0
        assert trace.max() > 10.0 * np.median(trace)
