"""Curve resolution: pulling co-eluting components apart."""

from __future__ import annotations

from pyrecycle_analytics.deconvolution.mcrals import (
    McrAlsOptions,
    enforce_unimodality,
    mcr_als,
    simplisma,
)
from pyrecycle_analytics.deconvolution.pipeline import (
    DeconvolutionConfig,
    DeconvolutionReport,
    resolve_pyrogram,
)
from pyrecycle_analytics.deconvolution.rank import RankEstimate, estimate_rank
from pyrecycle_analytics.deconvolution.result import ResolutionError, ResolutionResult
from pyrecycle_analytics.deconvolution.windows import RetentionWindow, detect_windows

__all__ = [
    "DeconvolutionConfig",
    "DeconvolutionReport",
    "McrAlsOptions",
    "RankEstimate",
    "ResolutionError",
    "ResolutionResult",
    "RetentionWindow",
    "detect_windows",
    "enforce_unimodality",
    "estimate_rank",
    "mcr_als",
    "resolve_pyrogram",
    "simplisma",
]
