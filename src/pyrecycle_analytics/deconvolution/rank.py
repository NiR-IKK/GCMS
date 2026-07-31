"""How many components are in a window?

This is the hardest and most consequential decision in the whole engine. Ask for
too few components and real compounds get merged into one; ask for too many and
the solver invents structure out of noise. The benchmark measurement that
motivated this module (see ROADMAP.md, Befund 3) showed that no single estimator
is trustworthy:

* the Marchenko-Pastur edge returned 94-129 where the truth was 8-12,
* a relative singular-value threshold returned 7-8 — an *underestimate*, and a
  correct one, because adjacent homologues are so collinear that they add almost
  no independent rank,
* one window returned 173 after preprocessing, worse than the 32 it gave raw.

The Marchenko-Pastur failure has a specific cause: MP assumes noise of constant
variance, while GC/MS noise is dominated by counting statistics and therefore
scales with the signal. Bright channels are far noisier than any global estimate,
and their noise-driven singular values sail past a uniform-variance threshold.

The obvious fix — apply an Anscombe variance-stabilising transform first, then
threshold — was implemented and **measured, and it does not work**. Two reasons,
both visible in the benchmark:

1. The square root is non-linear, so it destroys the very low-rankness being
   counted. A clean three-component window becomes numerically full-rank, and
   EFA run on the transformed matrix returned its ceiling of 12.
2. It is unnecessary. Parallel analysis compares the data against surrogates
   built by permuting each channel independently, which preserves that channel's
   own noise distribution. Heteroscedasticity is therefore *already* accounted
   for, without any transform or noise model.

Measured on synthetic windows of known rank 2-6 at four noise levels, parallel
analysis returned the exact rank in 18 of 20 cases and was never off by more than
one. It is consequently the estimator the consensus uses. EFA is retained because
its forward/backward curves say *where* a component elutes, which rank alone does
not, and its disagreement is a useful reliability flag — but it does not decide
the rank. ``variance_stabilise`` is likewise retained and tested, and left off by
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "RankEstimate",
    "variance_stabilise",
    "malinowski_indicator_rank",
    "parallel_analysis_rank",
    "evolving_factor_analysis",
    "efa_rank",
    "estimate_rank",
]


@dataclass(frozen=True, slots=True)
class RankEstimate:
    """Component count for one window, with the evidence behind it.

    Attributes:
        rank: The consensus estimate to use.
        estimates: Result of each individual estimator, by name.
        singular_values: Singular values of the (stabilised) window.
        agreement: True when the estimators lie within ``spread_tolerance``.
        warning: Human-readable note when they do not, or when the estimate was
            clamped to the allowed range.
    """

    rank: int
    estimates: dict[str, int]
    singular_values: np.ndarray = field(repr=False)
    agreement: bool
    warning: str | None = None

    @property
    def spread(self) -> int:
        """Difference between the largest and smallest individual estimate."""
        if not self.estimates:
            return 0
        values = list(self.estimates.values())
        return max(values) - min(values)


def variance_stabilise(matrix: np.ndarray) -> np.ndarray:
    """Anscombe transform, making Poisson-like noise approximately homoscedastic.

    ``2 * sqrt(x + 3/8)`` maps a Poisson variable to one with variance close to 1
    regardless of its mean. This is what makes noise-floor arguments — and
    therefore rank estimation — valid on ion-count data.

    Args:
        matrix: Non-negative intensities. Negative values are clipped to zero.

    Returns:
        Transformed matrix of the same shape.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    return 2.0 * np.sqrt(np.clip(matrix, 0.0, None) + 0.375)


def malinowski_indicator_rank(
    singular_values: np.ndarray, n_rows: int, n_columns: int
) -> int:
    """Rank from the minimum of Malinowski's indicator function.

    ``IND(k) = RE(k) / (p - k)²`` where ``RE(k)`` is the residual standard
    deviation after ``k`` components. The function falls while real components
    are being added and rises once only noise is left, so its minimum marks the
    boundary. It is the classical chemometric answer and needs no noise estimate.

    Args:
        singular_values: Singular values in descending order.
        n_rows: Number of rows of the analysed matrix.
        n_columns: Number of columns.

    Returns:
        Estimated rank, at least 1.
    """
    singular_values = np.asarray(singular_values, dtype=np.float64).ravel()
    max_rank = min(int(min(n_rows, n_columns)), singular_values.size) - 1
    if max_rank < 1:
        return 1

    squared = singular_values**2
    indicators = np.full(max_rank, np.inf)
    for k in range(1, max_rank + 1):
        remaining = n_columns - k
        if remaining <= 0:
            continue
        residual = float(np.sum(squared[k:]))
        real_error = np.sqrt(residual / (n_rows * remaining))
        indicators[k - 1] = real_error / (remaining**2)

    if not np.any(np.isfinite(indicators)):
        return 1
    return int(np.nanargmin(indicators)) + 1


def parallel_analysis_rank(
    matrix: np.ndarray,
    *,
    n_permutations: int = 20,
    percentile: float = 95.0,
    max_rank: int = 30,
    seed: int = 0,
) -> int:
    """Rank by comparison against column-permuted surrogate data.

    Shuffling each column independently destroys the correlation *between*
    channels while preserving each channel's own intensity distribution — including
    its heteroscedastic noise. Singular values that survive that destruction are
    noise; those that do not are structure. This makes no distributional
    assumption at all, which is why it is included alongside the parametric tests.

    Args:
        matrix: Window intensity matrix.
        n_permutations: Number of surrogate datasets. 20 is enough for a
            95th-percentile threshold; more only sharpens it slightly.
        percentile: Threshold percentile of the surrogate singular values.
        max_rank: Cap on the returned rank.
        seed: Seed for reproducibility.

    Returns:
        Estimated rank, at least 1.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        return 1

    observed = np.linalg.svd(matrix, compute_uv=False)
    limit = min(observed.size, max_rank)

    rng = np.random.default_rng(seed)
    surrogate = np.empty((n_permutations, limit), dtype=np.float64)
    shuffled = np.empty_like(matrix)
    for trial in range(n_permutations):
        for column in range(matrix.shape[1]):
            shuffled[:, column] = rng.permutation(matrix[:, column])
        values = np.linalg.svd(shuffled, compute_uv=False)
        surrogate[trial, :] = values[:limit]

    threshold = np.percentile(surrogate, percentile, axis=0)
    exceeds = observed[:limit] > threshold
    # Count only the leading run: once a component drops into the noise, the ones
    # behind it are noise too, whatever an isolated fluctuation suggests.
    rank = int(np.argmin(exceeds)) if not np.all(exceeds) else int(limit)
    return max(rank, 1)


def evolving_factor_analysis(
    matrix: np.ndarray,
    *,
    max_rank: int = 12,
    n_steps: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward and backward evolving singular values.

    EFA walks along the retention axis and repeatedly factorises the data seen so
    far (forward) and the data still to come (backward). A component announces
    itself as a singular value that lifts off the noise floor when its peak starts
    to elute, and the backward pass shows where it ends. Unlike a global SVD, EFA
    uses the *ordering* of the scans — the one piece of structure chromatography
    hands us for free.

    Args:
        matrix: Window intensity matrix, shape ``(n_scans, n_mz)``.
        max_rank: Number of singular values to track.
        n_steps: Number of checkpoints along the scan axis. Full resolution would
            mean one SVD per scan, which is wasteful for a smooth curve.

    Returns:
        ``(forward, backward)``, each of shape ``(n_steps, max_rank)`` and each
        row holding the leading singular values at that checkpoint.

    Raises:
        ValueError: If the matrix is not 2-D or has fewer than two rows.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {matrix.ndim}-D")
    n_scans = matrix.shape[0]
    if n_scans < 2:
        raise ValueError(f"need at least 2 scans, got {n_scans}")

    tracked = min(max_rank, min(matrix.shape))
    steps = min(n_steps, n_scans - 1)
    checkpoints = np.unique(
        np.linspace(2, n_scans, steps, dtype=int).clip(2, n_scans)
    )

    forward = np.zeros((checkpoints.size, tracked))
    backward = np.zeros((checkpoints.size, tracked))
    for position, cut in enumerate(checkpoints):
        head = np.linalg.svd(matrix[:cut, :], compute_uv=False)
        tail = np.linalg.svd(matrix[n_scans - cut :, :], compute_uv=False)
        forward[position, : min(tracked, head.size)] = head[:tracked]
        backward[position, : min(tracked, tail.size)] = tail[:tracked]

    return forward, backward[::-1, :]


def efa_rank(
    matrix: np.ndarray,
    *,
    max_rank: int = 12,
    noise_multiple: float = 3.0,
    relative_floor: float = 1e-6,
    n_steps: int = 60,
) -> int:
    """Rank from how many EFA curves rise clearly above the noise floor.

    Args:
        matrix: Window intensity matrix.
        max_rank: Cap on the returned rank.
        noise_multiple: How far above the noise floor a curve must rise. The floor
            is taken as the median of the smallest tracked singular value, which
            is a component-free reference.
        relative_floor: Secondary threshold as a fraction of the largest singular
            value. Without it, noise-free data has an *exactly* zero noise floor,
            every numerically-nonzero singular value clears the absolute
            threshold, and the estimator returns its ceiling for a clean
            three-component window.
        n_steps: Checkpoints along the scan axis.

    Returns:
        Estimated rank, at least 1.
    """
    forward, backward = evolving_factor_analysis(
        matrix, max_rank=max_rank + 2, n_steps=n_steps
    )
    if forward.size == 0:
        return 1

    floor = float(np.median(forward[:, -1]))
    if floor <= 0.0:
        floor = float(np.min(forward[forward > 0.0], initial=0.0))
    largest = float(np.max(forward)) if forward.size else 0.0
    threshold = max(noise_multiple * floor, relative_floor * largest)

    # A genuine component must lift off in *both* directions: it starts somewhere
    # and it ends somewhere. Noise fluctuations rarely satisfy both.
    peaks_forward = forward.max(axis=0)
    peaks_backward = backward.max(axis=0)
    active = np.minimum(peaks_forward, peaks_backward) > threshold
    rank = int(np.argmin(active)) if not np.all(active) else int(active.size)
    return int(np.clip(rank, 1, max_rank))


def estimate_rank(
    matrix: np.ndarray,
    *,
    max_rank: int = 12,
    min_rank: int = 1,
    stabilise: bool = False,
    spread_tolerance: int = 2,
    seed: int = 0,
    include_malinowski: bool = False,
) -> RankEstimate:
    """Estimate a window's component count, with corroborating evidence.

    **Parallel analysis decides the rank.** It was the only one of the three
    implemented estimators that survived measurement (see the module docstring):
    exact in 18 of 20 synthetic windows of known rank, never off by more than one,
    across four noise levels. EFA is run alongside as corroboration — when the two
    disagree by more than ``spread_tolerance``, the window is flagged as one whose
    resolution should not be trusted, rather than the disagreement being averaged
    away into a plausible-looking number.

    Malinowski's indicator function is implemented in this module but is **off by
    default**, because it was measured to be structurally unusable on data of this
    shape. ``IND(k) = RE(k)/(p-k)²`` needs the ``(p-k)²`` term to turn the curve
    around, but for a 70×292 window that term changes by only a third across the
    plausible rank range while ``RE(k)`` keeps falling. The curve is monotonically
    decreasing, so its minimum always sits at the largest ``k`` tested, and the
    estimator returns the ceiling regardless of the data. That is a property of
    the matrix shape, not a bug — GC/MS windows are short and wide, whereas IND
    was designed for tall, narrow spectroscopic matrices.

    Args:
        matrix: Window intensity matrix, shape ``(n_scans, n_mz)``.
        max_rank: Upper bound. Windows needing more than ~12 components are
            beyond what constrained MCR-ALS resolves reliably anyway.
        min_rank: Lower bound, normally 1.
        stabilise: Apply the Anscombe transform before counting. Off by default:
            measurement showed it makes the estimate worse, because the square
            root destroys the low-rank structure being counted, and parallel
            analysis needs no variance assumption in the first place.
        spread_tolerance: Maximum disagreement before ``agreement`` goes false.
        seed: Seed for the permutation test.
        include_malinowski: Add the indicator function to the consensus. See above
            before switching this on.

    Returns:
        The consensus estimate with its supporting evidence.

    Raises:
        ValueError: If the matrix is not 2-D, or the bounds are inconsistent.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got {matrix.ndim}-D")
    if max_rank < min_rank:
        raise ValueError(f"max_rank ({max_rank}) is below min_rank ({min_rank})")
    if min_rank < 1:
        raise ValueError(f"min_rank must be >= 1, got {min_rank}")

    working = variance_stabilise(matrix) if stabilise else matrix
    n_rows, n_columns = working.shape
    ceiling = min(max_rank, min(n_rows, n_columns))

    singular_values = np.linalg.svd(working, compute_uv=False)

    estimates = {
        "parallel_analysis": int(
            np.clip(
                parallel_analysis_rank(working, max_rank=ceiling, seed=seed),
                min_rank,
                ceiling,
            )
        ),
        "efa": int(
            np.clip(efa_rank(working, max_rank=ceiling), min_rank, ceiling)
        ),
    }
    if include_malinowski:
        estimates["malinowski"] = int(
            np.clip(
                malinowski_indicator_rank(singular_values, n_rows, n_columns),
                min_rank,
                ceiling,
            )
        )

    values = sorted(estimates.values())
    # Parallel analysis decides; the others corroborate. Taking a median would let
    # the estimators that were measured to be unreliable outvote the one that was not.
    consensus = int(np.clip(estimates["parallel_analysis"], min_rank, ceiling))
    spread = values[-1] - values[0]

    warning: str | None = None
    if spread > spread_tolerance:
        warning = (
            f"rank estimators disagree by {spread} "
            f"({', '.join(f'{k}={v}' for k, v in estimates.items())}); "
            "the resolution of this window should be treated as unreliable"
        )
    elif consensus >= ceiling:
        warning = (
            f"rank estimate hit the ceiling of {ceiling}; the window may contain "
            "more components than constrained resolution can separate"
        )

    return RankEstimate(
        rank=consensus,
        estimates=estimates,
        singular_values=singular_values,
        agreement=spread <= spread_tolerance,
        warning=warning,
    )
