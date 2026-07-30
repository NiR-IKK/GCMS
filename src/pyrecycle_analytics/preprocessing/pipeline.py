"""Declarative preprocessing pipeline.

The chain applied before deconvolution has to be identical for every sample in a
study, otherwise marker ratios are not comparable between runs. Encoding it as a
validated Pydantic configuration means the exact chain can be stored next to the
result, shipped in an API request, and replayed months later.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_schemas.pyrogram import PreprocessingStep
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.preprocessing.baseline import correct_baseline
from pyrecycle_analytics.preprocessing.smoothing import smooth_cube

__all__ = ["PreprocessingConfig", "preprocess"]


class PreprocessingConfig(BaseModel):
    """Reproducible description of the preprocessing chain.

    The default values are the ones validated against the synthetic benchmark for
    a 5 Hz, 30 m non-polar column method: they remove column bleed and shot noise
    without measurably distorting marker-triad ratios.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rt_range_s: tuple[float, float] | None = Field(
        None,
        description="Trim the run to this window before anything else. Used to cut the "
        "solvent/permanent-gas front and the terminal bleed ramp.",
    )
    mz_range: tuple[float, float] | None = Field(
        None, description="Restrict the m/z grid; e.g. drop m/z < 35 to remove air peaks."
    )

    smoothing: Literal["savgol", "gaussian", "none"] = "savgol"
    savgol_window: int = Field(7, ge=3, le=101, description="Must be odd.")
    savgol_polyorder: int = Field(2, ge=1, le=6)
    gaussian_sigma_scans: float = Field(1.0, gt=0.0, le=20.0)

    baseline: Literal["asls", "snip", "none"] = "asls"
    asls_lam: float = Field(
        1e7,
        gt=0.0,
        description="AsLS smoothness penalty. The residual a peak leaves in the "
        "baseline estimate falls roughly as 1/lam, and pyrogram peaks stand an order "
        "of magnitude above the background, so values below ~1e6 measurably shave "
        "area off real peaks. Raise further for very dense homologous-series regions.",
    )
    asls_p: float = Field(0.01, gt=0.0, lt=1.0)
    asls_iterations: int = Field(15, ge=1, le=200)
    snip_iterations: int = Field(40, ge=1, le=500)
    clip_negative: bool = True

    min_channel_signal: float = Field(
        0.0,
        ge=0.0,
        description="Channels whose maximum is at or below this value are treated as "
        "empty during baseline estimation.",
    )
    drop_empty_channels: bool = Field(
        False,
        description="Remove m/z channels that are all zero after correction. Shrinks the "
        "matrix for curve resolution but changes the m/z axis, so it is off by default.",
    )

    @model_validator(mode="after")
    def _check_window(self) -> PreprocessingConfig:
        if self.savgol_window % 2 == 0:
            raise ValueError(f"savgol_window must be odd, got {self.savgol_window}")
        if self.savgol_window <= self.savgol_polyorder:
            raise ValueError(
                f"savgol_window ({self.savgol_window}) must exceed "
                f"savgol_polyorder ({self.savgol_polyorder})"
            )
        for name, window in (("rt_range_s", self.rt_range_s), ("mz_range", self.mz_range)):
            if window is not None and window[1] <= window[0]:
                raise ValueError(f"{name}={window} is empty")
        return self


def preprocess(
    cube: PyrogramDataCube, config: PreprocessingConfig | None = None
) -> PyrogramDataCube:
    """Apply a preprocessing chain to a pyrogram.

    Order is deliberate: trim, then smooth, then baseline-correct. Smoothing first
    lets the baseline estimator see a less noisy signal, which stops AsLS from
    tracking noise spikes into the background; correcting first would instead let
    the smoother spread residual baseline steps into neighbouring scans.

    Args:
        cube: Raw ingested pyrogram.
        config: Chain description; defaults are used when omitted.

    Returns:
        A new cube whose ``metadata.preprocessing`` records every applied step.
    """
    config = config or PreprocessingConfig()
    result = cube

    if config.rt_range_s is not None or config.mz_range is not None:
        rt_start, rt_end = config.rt_range_s or result.rt_range_s
        result = result.window(rt_start, rt_end, mz_range=config.mz_range)

    if config.smoothing == "savgol":
        result = smooth_cube(
            result,
            "savgol",
            window_length=config.savgol_window,
            polyorder=config.savgol_polyorder,
        )
    elif config.smoothing == "gaussian":
        result = smooth_cube(result, "gaussian", sigma_scans=config.gaussian_sigma_scans)

    if config.baseline == "asls":
        result = correct_baseline(
            result,
            "asls",
            clip_negative=config.clip_negative,
            lam=config.asls_lam,
            p=config.asls_p,
            n_iter=config.asls_iterations,
            min_channel_signal=config.min_channel_signal,
        )
    elif config.baseline == "snip":
        result = correct_baseline(
            result,
            "snip",
            clip_negative=config.clip_negative,
            n_iter=config.snip_iterations,
            min_channel_signal=config.min_channel_signal,
        )

    if config.drop_empty_channels:
        occupied = np.flatnonzero(result.intensities.max(axis=0) > 0.0)
        if occupied.size and occupied.size < result.n_mz:
            kept_mz = result.mz_axis[occupied]
            result = result.window(
                *result.rt_range_s, mz_range=(float(kept_mz[0]), float(kept_mz[-1]))
            )
            step = PreprocessingStep(
                name="drop_empty_channels",
                parameters={"n_kept": int(occupied.size)},
            )
            result = result.with_intensities(result.intensities, step)

    return result
