"""Similarity and goodness-of-fit measures for resolved data.

These are the numbers every claim in Milestones 2 and 3 is argued with, so they
are defined once, here, rather than reimplemented per call site. All of them are
scale-invariant where that is chemically appropriate: a mass spectrum's identity
lives in its *pattern*, not its absolute height, and a curve-resolution result is
only ever determined up to a per-component scale factor.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cosine_similarity",
    "cosine_similarity_matrix",
    "weighted_dot_similarity",
    "lack_of_fit",
    "explained_variance",
    "relative_area_error",
]

_TINY = np.finfo(np.float64).tiny


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Cosine of the angle between two spectra or profiles.

    The standard measure of spectral agreement: 1.0 means identical pattern,
    independent of intensity scale. Both vectors are expected to be non-negative,
    so the result lies in ``[0, 1]``.

    Args:
        first: Vector, shape ``(n,)``.
        second: Vector of the same length.

    Returns:
        Similarity in ``[0, 1]``; 0.0 if either vector is all-zero.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    if first.size != second.size:
        raise ValueError(f"length mismatch: {first.size} vs {second.size}")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= _TINY:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def cosine_similarity_matrix(rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between two sets of row vectors.

    Args:
        rows_a: Shape ``(n_a, n_features)``.
        rows_b: Shape ``(n_b, n_features)``.

    Returns:
        Matrix of shape ``(n_a, n_b)``.

    Raises:
        ValueError: If the feature dimensions differ.
    """
    rows_a = np.atleast_2d(np.asarray(rows_a, dtype=np.float64))
    rows_b = np.atleast_2d(np.asarray(rows_b, dtype=np.float64))
    if rows_a.shape[1] != rows_b.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: {rows_a.shape[1]} vs {rows_b.shape[1]}"
        )
    norms_a = np.linalg.norm(rows_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(rows_b, axis=1, keepdims=True)
    normalised_a = rows_a / np.maximum(norms_a, _TINY)
    normalised_b = rows_b / np.maximum(norms_b, _TINY)
    return np.clip(normalised_a @ normalised_b.T, -1.0, 1.0)


def weighted_dot_similarity(
    first: np.ndarray,
    second: np.ndarray,
    mz_axis: np.ndarray,
    *,
    mass_exponent: float = 2.0,
    intensity_exponent: float = 0.5,
) -> float:
    """Mass-weighted spectral similarity, in the style of the NIST match factor.

    Plain cosine similarity is dominated by the low-mass fragments that every
    hydrocarbon shares — m/z 43 and 57 are large in almost everything in a
    polyolefin pyrogram. Weighting each channel by ``mz**mass_exponent`` shifts
    the decision onto the high-mass, structurally informative ions, which is what
    makes a match factor able to tell C14 from C15 at all.

    Args:
        first: Spectrum, shape ``(n_mz,)``.
        second: Spectrum of the same length.
        mz_axis: m/z value of each channel, same length.
        mass_exponent: Exponent on m/z. NIST uses 2.
        intensity_exponent: Exponent on intensity. NIST uses 0.5, which
            compresses the dynamic range so a single huge peak cannot decide the
            match alone.

    Returns:
        Similarity in ``[0, 1]``.

    Raises:
        ValueError: If the three arrays do not have the same length.
    """
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    mz_axis = np.asarray(mz_axis, dtype=np.float64).ravel()
    if not (first.size == second.size == mz_axis.size):
        raise ValueError(
            f"length mismatch: spectra {first.size}/{second.size}, axis {mz_axis.size}"
        )
    weights = np.power(np.maximum(mz_axis, 0.0), mass_exponent)
    weighted_first = weights * np.power(np.maximum(first, 0.0), intensity_exponent)
    weighted_second = weights * np.power(np.maximum(second, 0.0), intensity_exponent)
    return cosine_similarity(weighted_first, weighted_second)


def lack_of_fit(observed: np.ndarray, reconstructed: np.ndarray) -> float:
    """Lack of fit in percent — the standard MCR convergence and quality measure.

    ``LOF = 100 * sqrt(sum((D - D_hat)^2) / sum(D^2))``.

    A word of caution that the tests act on: a low LOF proves the model *fits*,
    not that it is *correct*. Rotational ambiguity means a wrong factorisation can
    fit perfectly. LOF is therefore reported alongside, never instead of,
    comparison against ground truth.

    Args:
        observed: Data matrix ``D``.
        reconstructed: Model ``C @ S`` of the same shape.

    Returns:
        Lack of fit in percent; 0.0 for a perfect fit.

    Raises:
        ValueError: If the shapes differ.
    """
    observed = np.asarray(observed, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if observed.shape != reconstructed.shape:
        raise ValueError(f"shape mismatch: {observed.shape} vs {reconstructed.shape}")
    total = float(np.sum(observed**2))
    if total <= _TINY:
        return 0.0
    residual = float(np.sum((observed - reconstructed) ** 2))
    return float(100.0 * np.sqrt(residual / total))


def explained_variance(observed: np.ndarray, reconstructed: np.ndarray) -> float:
    """Fraction of the data's variance captured by the model (``R²``).

    Args:
        observed: Data matrix ``D``.
        reconstructed: Model of the same shape.

    Returns:
        ``R²``, which can be negative for a model worse than the mean.

    Raises:
        ValueError: If the shapes differ.
    """
    observed = np.asarray(observed, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if observed.shape != reconstructed.shape:
        raise ValueError(f"shape mismatch: {observed.shape} vs {reconstructed.shape}")
    centred = observed - observed.mean()
    total = float(np.sum(centred**2))
    if total <= _TINY:
        return 1.0
    residual = float(np.sum((observed - reconstructed) ** 2))
    return float(1.0 - residual / total)


def relative_area_error(recovered_area: float, true_area: float) -> float:
    """Signed relative error of a recovered peak area.

    Args:
        recovered_area: Area the pipeline reported.
        true_area: Area from the ground truth.

    Returns:
        ``(recovered - true) / true``; ``inf`` when the true area is zero and the
        recovered one is not.
    """
    if abs(true_area) <= _TINY:
        return 0.0 if abs(recovered_area) <= _TINY else float("inf")
    return float((recovered_area - true_area) / true_area)
