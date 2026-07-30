"""Baseline correction, smoothing and the reproducible preprocessing chain."""

from __future__ import annotations

from pyrecycle_analytics.preprocessing.baseline import (
    asls_baseline,
    asls_baseline_matrix,
    correct_baseline,
    snip_baseline,
    snip_baseline_matrix,
)
from pyrecycle_analytics.preprocessing.pipeline import PreprocessingConfig, preprocess
from pyrecycle_analytics.preprocessing.smoothing import (
    estimate_noise_sigma,
    gaussian_smooth,
    savgol_smooth,
    smooth_cube,
)

__all__ = [
    "PreprocessingConfig",
    "asls_baseline",
    "asls_baseline_matrix",
    "correct_baseline",
    "estimate_noise_sigma",
    "gaussian_smooth",
    "preprocess",
    "savgol_smooth",
    "smooth_cube",
    "snip_baseline",
    "snip_baseline_matrix",
]
