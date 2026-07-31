"""Kováts retention indices from the sample's own alkane comb.

Retention time is not transferable. It moves with column length, flow, oven
program, and simply with the column ageing between Monday and Friday. The Kováts
retention index is, which is why a marker library is built on indices.

Computing an index normally requires injecting an n-alkane standard. In a
polyolefin-rich recyclate that injection is unnecessary: **the sample carries its
own ladder.** The homologous alkane series that the matrix subtraction in
Milestone 2 already located and modelled is exactly the series the Kováts scale is
defined on. The comb that makes everything else harder makes this one thing free.

The definition, for a temperature-programmed run:

    ``RI = 100 · (n + (t − tₙ) / (tₙ₊₁ − tₙ))``

for a compound eluting between the n-alkanes with ``n`` and ``n+1`` carbons.

One caveat is unavoidable and is surfaced rather than hidden: the comb detection
returns the series' *spacing* and positions but not the absolute carbon number of
its first member. That anchor has to come from outside — an operator's knowledge
of the method, a co-injected standard, or the extrapolation in
:func:`estimate_first_carbon_number`, which is an estimate and is labelled as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrecycle_analytics.exceptions import PyRecycleError

__all__ = [
    "RetentionIndexCalibration",
    "RetentionIndexError",
    "estimate_first_carbon_number",
    "calibrate_from_comb",
]


class RetentionIndexError(PyRecycleError):
    """The retention-index ladder could not be established."""


@dataclass(frozen=True, slots=True)
class RetentionIndexCalibration:
    """A Kováts ladder built from the n-alkane comb of one run.

    Attributes:
        anchor_times_s: Retention time of each ladder rung, increasing.
        carbon_numbers: Carbon number of each rung, same length.
        anchor_confidence: How firmly the absolute carbon numbering is known.
            ``"anchored"`` when supplied externally, ``"estimated"`` when
            extrapolated from the dead time. An estimated ladder gives correct
            *relative* indices but may be offset by a constant 100·k.
        note: Human-readable explanation of the anchoring.
    """

    anchor_times_s: np.ndarray
    carbon_numbers: np.ndarray
    anchor_confidence: str = "estimated"
    note: str = ""

    @property
    def n_anchors(self) -> int:
        return int(self.anchor_times_s.size)

    @property
    def covered_range_s(self) -> tuple[float, float]:
        return (float(self.anchor_times_s[0]), float(self.anchor_times_s[-1]))

    def index_of(self, retention_time_s: float) -> float:
        """Kováts index of a compound at a given retention time.

        Args:
            retention_time_s: Apex retention time in seconds.

        Returns:
            The retention index. Outside the ladder's range the outermost spacing
            is extrapolated linearly, which degrades gracefully rather than
            refusing an answer near the run's edges.
        """
        times = self.anchor_times_s
        carbons = self.carbon_numbers

        if retention_time_s <= times[0]:
            if times.size < 2:
                return float(100.0 * carbons[0])
            slope = (carbons[1] - carbons[0]) / (times[1] - times[0])
            return float(100.0 * (carbons[0] + slope * (retention_time_s - times[0])))

        if retention_time_s >= times[-1]:
            if times.size < 2:
                return float(100.0 * carbons[-1])
            slope = (carbons[-1] - carbons[-2]) / (times[-1] - times[-2])
            return float(100.0 * (carbons[-1] + slope * (retention_time_s - times[-1])))

        position = int(np.searchsorted(times, retention_time_s) - 1)
        position = int(np.clip(position, 0, times.size - 2))
        lower_time, upper_time = times[position], times[position + 1]
        lower_carbon = carbons[position]
        span = upper_time - lower_time
        fraction = (retention_time_s - lower_time) / span if span > 0 else 0.0
        step = carbons[position + 1] - carbons[position]
        return float(100.0 * (lower_carbon + step * fraction))

    def indices_of(self, retention_times_s: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`index_of`."""
        return np.array(
            [self.index_of(float(value)) for value in np.atleast_1d(retention_times_s)]
        )


def estimate_first_carbon_number(
    anchor_times_s: np.ndarray,
    *,
    dead_time_s: float,
    min_carbon: int = 5,
    max_carbon: int = 40,
) -> int:
    """Estimate the carbon number of the first comb member.

    Under a linear temperature program n-alkane retention is close to linear in
    carbon number, so fitting a line through the observed rungs and extrapolating
    back to the dead time — where an unretained compound elutes, nominally carbon
    number zero — recovers the offset.

    This is an *estimate*, and the calibration it produces is labelled as one. It
    is sensitive to the dead time, and the relationship is only approximately
    linear at the extremes. Relative indices are unaffected by an error here;
    absolute ones shift by 100 per carbon. Where absolute indices matter, supply
    the anchor explicitly.

    Args:
        anchor_times_s: Retention times of the comb members, increasing.
        dead_time_s: Hold-up time of the column, in seconds.
        min_carbon: Lower clamp on the result.
        max_carbon: Upper clamp on the result.

    Returns:
        Estimated carbon number of the first rung.

    Raises:
        RetentionIndexError: If fewer than two anchors are supplied or the dead
            time is not before the first anchor.
    """
    anchor_times_s = np.asarray(anchor_times_s, dtype=np.float64).ravel()
    if anchor_times_s.size < 2:
        raise RetentionIndexError("need at least two comb members to extrapolate")
    if dead_time_s <= 0.0 or dead_time_s >= anchor_times_s[0]:
        raise RetentionIndexError(
            f"dead_time_s ({dead_time_s}) must be positive and precede the first "
            f"comb member at {anchor_times_s[0]:.1f} s"
        )

    positions = np.arange(anchor_times_s.size, dtype=np.float64)
    slope, intercept = np.polyfit(positions, anchor_times_s, 1)
    if slope <= 0.0:  # pragma: no cover - defensive
        raise RetentionIndexError("comb retention times are not increasing")

    # Position at which the fitted line reaches the dead time; the first rung's
    # carbon number is how far it sits above that point.
    position_at_dead_time = (dead_time_s - intercept) / slope
    estimated = int(round(-position_at_dead_time))
    return int(np.clip(estimated, min_carbon, max_carbon))


def calibrate_from_comb(
    anchor_times_s: np.ndarray,
    *,
    first_carbon_number: int | None = None,
    dead_time_s: float | None = None,
) -> RetentionIndexCalibration:
    """Build a Kováts ladder from a detected alkane comb.

    Args:
        anchor_times_s: Retention times of the comb members, increasing. In
            practice ``CombDetection.apex_times_s`` from the matrix subtraction.
        first_carbon_number: Carbon number of the first member, when known. This
            is the anchored, trustworthy path.
        dead_time_s: Column hold-up time, used to *estimate* the anchor when
            ``first_carbon_number`` is not given.

    Returns:
        The calibration, labelled ``"anchored"`` or ``"estimated"``.

    Raises:
        RetentionIndexError: If fewer than two anchors are given, they are not
            increasing, or neither anchoring route is available.
    """
    anchor_times_s = np.asarray(anchor_times_s, dtype=np.float64).ravel()
    if anchor_times_s.size < 2:
        raise RetentionIndexError(
            f"a retention-index ladder needs at least two rungs, got {anchor_times_s.size}"
        )
    if np.any(np.diff(anchor_times_s) <= 0.0):
        raise RetentionIndexError("comb retention times must be strictly increasing")

    if first_carbon_number is not None:
        confidence = "anchored"
        note = f"first comb member declared as C{first_carbon_number}"
        start = int(first_carbon_number)
    elif dead_time_s is not None:
        start = estimate_first_carbon_number(anchor_times_s, dead_time_s=dead_time_s)
        confidence = "estimated"
        note = (
            f"first comb member estimated as C{start} by extrapolation to a dead "
            f"time of {dead_time_s:.1f} s; relative indices are unaffected by an "
            "error here, absolute ones shift by 100 per carbon"
        )
    else:
        raise RetentionIndexError(
            "supply either first_carbon_number (preferred) or dead_time_s to anchor "
            "the ladder; the comb gives spacing but not absolute carbon numbers"
        )

    carbons = start + np.arange(anchor_times_s.size, dtype=np.float64)
    return RetentionIndexCalibration(
        anchor_times_s=anchor_times_s,
        carbon_numbers=carbons,
        anchor_confidence=confidence,
        note=note,
    )
