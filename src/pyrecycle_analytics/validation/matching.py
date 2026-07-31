"""Assigning resolved components to ground-truth components.

Curve resolution returns its components in arbitrary order, so *nothing* about a
result can be scored before the recovered components have been paired with the
true ones. Getting this pairing wrong makes every downstream metric meaningless,
and it is not a trivial problem here: adjacent members of a homologous series
have spectra with cosine similarity above 0.98, so spectral agreement alone
cannot tell C14 from C15.

The assignment therefore scores a pair on spectral agreement **and** retention
proximity, and picks the globally optimal pairing with the Hungarian algorithm
rather than greedily. A greedy pass would happily consume the C15 spectrum for
the C14 slot and then mislabel everything after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from pyrecycle_analytics.validation.metrics import (
    cosine_similarity_matrix,
    relative_area_error,
)

__all__ = ["ComponentMatch", "ComponentMatching", "match_components"]


@dataclass(frozen=True, slots=True)
class ComponentMatch:
    """One recovered component paired with the true component it represents.

    Attributes:
        true_index: Column index in the ground-truth factorisation.
        recovered_index: Column index in the result being scored.
        spectral_similarity: Cosine similarity of the two mass spectra.
        retention_error_s: Recovered apex minus true apex, in seconds.
        profile_similarity: Cosine similarity of the two elution profiles.
        area_relative_error: Signed relative error of the recovered area.
    """

    true_index: int
    recovered_index: int
    spectral_similarity: float
    retention_error_s: float
    profile_similarity: float
    area_relative_error: float


@dataclass(frozen=True, slots=True)
class ComponentMatching:
    """Result of pairing a resolution against ground truth.

    Attributes:
        matches: Accepted pairs, ordered by true index.
        missed_true: True components no recovered component was assigned to.
        spurious_recovered: Recovered components that match nothing — either
            genuine artefacts or an over-estimated rank.
    """

    matches: tuple[ComponentMatch, ...]
    missed_true: tuple[int, ...]
    spurious_recovered: tuple[int, ...]

    @property
    def n_true(self) -> int:
        return len(self.matches) + len(self.missed_true)

    @property
    def n_recovered(self) -> int:
        return len(self.matches) + len(self.spurious_recovered)

    @property
    def recall(self) -> float:
        """Fraction of true components that were found."""
        return len(self.matches) / self.n_true if self.n_true else 0.0

    @property
    def precision(self) -> float:
        """Fraction of recovered components that correspond to something real."""
        return len(self.matches) / self.n_recovered if self.n_recovered else 0.0

    @property
    def mean_spectral_similarity(self) -> float:
        """Average spectral agreement over the accepted pairs."""
        if not self.matches:
            return 0.0
        return float(np.mean([match.spectral_similarity for match in self.matches]))

    @property
    def median_absolute_area_error(self) -> float:
        """Median ``|relative area error|`` over the accepted pairs."""
        if not self.matches:
            return float("inf")
        errors = [abs(match.area_relative_error) for match in self.matches]
        return float(np.median(errors))

    def match_for_true(self, true_index: int) -> ComponentMatch | None:
        """The pair covering a given true component, or ``None`` if it was missed."""
        for match in self.matches:
            if match.true_index == true_index:
                return match
        return None


def _apex_positions(profiles: np.ndarray, retention_times: np.ndarray) -> np.ndarray:
    """Retention time of each profile's maximum, shape ``(n_components,)``."""
    return retention_times[np.argmax(profiles, axis=0)]


def _areas(profiles: np.ndarray, retention_times: np.ndarray) -> np.ndarray:
    """Integrated area of each profile, shape ``(n_components,)``."""
    return np.trapezoid(profiles, retention_times, axis=0)


def match_components(
    true_profiles: np.ndarray,
    true_spectra: np.ndarray,
    recovered_profiles: np.ndarray,
    recovered_spectra: np.ndarray,
    retention_times: np.ndarray,
    *,
    retention_tolerance_s: float = 6.0,
    min_spectral_similarity: float = 0.80,
    max_retention_error_s: float | None = None,
) -> ComponentMatching:
    """Pair recovered components with true ones, optimally and globally.

    The pairing score is ``cosine(spectra) * exp(-0.5 * (Δt / tolerance)²)``. The
    retention term is what makes members of a homologous series distinguishable:
    their spectra are nearly identical, but their apexes are not.

    Args:
        true_profiles: Ground-truth ``C``, shape ``(n_scans, n_true)``.
        true_spectra: Ground-truth ``S``, shape ``(n_true, n_mz)``.
        recovered_profiles: Resolved ``C``, shape ``(n_scans, n_recovered)``.
        recovered_spectra: Resolved ``S``, shape ``(n_recovered, n_mz)``.
        retention_times: Shared time axis, shape ``(n_scans,)``.
        retention_tolerance_s: Width of the retention agreement kernel. Should be
            of the order of a peak width, not of the whole run.
        min_spectral_similarity: Pairs below this cosine are rejected and counted
            as a miss plus a spurious component.
        max_retention_error_s: Optional hard cut on apex disagreement. ``None``
            relies on the kernel alone.

    Returns:
        The accepted pairs plus the unmatched components on both sides.

    Raises:
        ValueError: If the array shapes are mutually inconsistent.
    """
    true_profiles = np.atleast_2d(np.asarray(true_profiles, dtype=np.float64))
    recovered_profiles = np.atleast_2d(np.asarray(recovered_profiles, dtype=np.float64))
    true_spectra = np.atleast_2d(np.asarray(true_spectra, dtype=np.float64))
    recovered_spectra = np.atleast_2d(np.asarray(recovered_spectra, dtype=np.float64))
    retention_times = np.asarray(retention_times, dtype=np.float64).ravel()

    n_scans = retention_times.size
    if true_profiles.shape[0] != n_scans or recovered_profiles.shape[0] != n_scans:
        raise ValueError(
            f"profiles must have {n_scans} rows, got {true_profiles.shape[0]} (true) "
            f"and {recovered_profiles.shape[0]} (recovered)"
        )
    if true_profiles.shape[1] != true_spectra.shape[0]:
        raise ValueError(
            f"true C has {true_profiles.shape[1]} components but S has "
            f"{true_spectra.shape[0]}"
        )
    if recovered_profiles.shape[1] != recovered_spectra.shape[0]:
        raise ValueError(
            f"recovered C has {recovered_profiles.shape[1]} components but S has "
            f"{recovered_spectra.shape[0]}"
        )
    if retention_tolerance_s <= 0.0:
        raise ValueError(f"retention_tolerance_s must be > 0, got {retention_tolerance_s}")

    n_true = true_profiles.shape[1]
    n_recovered = recovered_profiles.shape[1]
    if n_true == 0 or n_recovered == 0:
        return ComponentMatching(
            matches=(),
            missed_true=tuple(range(n_true)),
            spurious_recovered=tuple(range(n_recovered)),
        )

    spectral = cosine_similarity_matrix(true_spectra, recovered_spectra)
    true_apex = _apex_positions(true_profiles, retention_times)
    recovered_apex = _apex_positions(recovered_profiles, retention_times)
    delta = true_apex[:, None] - recovered_apex[None, :]
    retention_kernel = np.exp(-0.5 * (delta / retention_tolerance_s) ** 2)

    score = spectral * retention_kernel
    true_rows, recovered_columns = linear_sum_assignment(-score)

    profile_similarity = cosine_similarity_matrix(true_profiles.T, recovered_profiles.T)
    true_areas = _areas(true_profiles, retention_times)
    recovered_areas = _areas(recovered_profiles, retention_times)

    matches: list[ComponentMatch] = []
    matched_true: set[int] = set()
    matched_recovered: set[int] = set()

    for true_index, recovered_index in zip(true_rows, recovered_columns, strict=True):
        similarity = float(spectral[true_index, recovered_index])
        retention_error = float(
            recovered_apex[recovered_index] - true_apex[true_index]
        )
        if similarity < min_spectral_similarity:
            continue
        if (
            max_retention_error_s is not None
            and abs(retention_error) > max_retention_error_s
        ):
            continue
        matches.append(
            ComponentMatch(
                true_index=int(true_index),
                recovered_index=int(recovered_index),
                spectral_similarity=similarity,
                retention_error_s=retention_error,
                profile_similarity=float(profile_similarity[true_index, recovered_index]),
                area_relative_error=relative_area_error(
                    float(recovered_areas[recovered_index]), float(true_areas[true_index])
                ),
            )
        )
        matched_true.add(int(true_index))
        matched_recovered.add(int(recovered_index))

    return ComponentMatching(
        matches=tuple(sorted(matches, key=lambda match: match.true_index)),
        missed_true=tuple(index for index in range(n_true) if index not in matched_true),
        spurious_recovered=tuple(
            index for index in range(n_recovered) if index not in matched_recovered
        ),
    )
