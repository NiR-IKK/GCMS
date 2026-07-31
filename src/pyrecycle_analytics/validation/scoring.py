"""Scoring a resolution against ground truth.

This is the module that decides whether a change to the deconvolution engine was
an improvement. It deliberately reports two different things side by side:

* **Global scores** over all components — recall, precision, mean spectral
  agreement. Useful, but dominated by the ~120 homologous-series members, which
  are near-collinear and largely unresolvable in principle.
* **Marker scores** over a named list of compounds — the styrene triad,
  caprolactam, benzoic acid, the phthalates. These are what a recyclate report
  actually rests on, and they are the ones the acceptance criteria are written
  against.

Reporting only the global number would hide a pipeline that resolves the boring
comb well and loses every trace marker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrecycle_analytics.deconvolution.result import ResolutionResult
from pyrecycle_analytics.validation.matching import ComponentMatching, match_components
from pyrecycle_analytics.validation.metrics import explained_variance, lack_of_fit

__all__ = ["ResolutionScore", "MarkerScore", "score_resolution", "score_markers"]


@dataclass(frozen=True, slots=True)
class ResolutionScore:
    """Aggregate quality of a resolution against ground truth.

    Attributes:
        matching: The underlying component assignment.
        lack_of_fit: Residual of the reconstruction, in percent.
        explained_variance: ``R²`` of the reconstruction.
        n_true: Number of ground-truth components in scope.
        n_recovered: Number of components the method returned.
        n_matched: Number of accepted pairs.
        mean_spectral_similarity: Average cosine over accepted pairs.
        median_spectral_similarity: Median cosine over accepted pairs.
        median_absolute_area_error: Median ``|relative area error|``.
        median_absolute_retention_error_s: Median ``|apex error|`` in seconds.
    """

    matching: ComponentMatching
    lack_of_fit: float
    explained_variance: float
    n_true: int
    n_recovered: int
    n_matched: int
    mean_spectral_similarity: float
    median_spectral_similarity: float
    median_absolute_area_error: float
    median_absolute_retention_error_s: float

    @property
    def recall(self) -> float:
        return self.matching.recall

    @property
    def precision(self) -> float:
        return self.matching.precision

    def summary(self) -> dict[str, float | int]:
        """Flat, JSON-serialisable form for the benchmark report."""
        return {
            "n_true": self.n_true,
            "n_recovered": self.n_recovered,
            "n_matched": self.n_matched,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "mean_spectral_similarity": round(self.mean_spectral_similarity, 4),
            "median_spectral_similarity": round(self.median_spectral_similarity, 4),
            "median_abs_area_error": round(self.median_absolute_area_error, 4),
            "median_abs_retention_error_s": round(
                self.median_absolute_retention_error_s, 4
            ),
            "lack_of_fit_percent": round(self.lack_of_fit, 4),
            "explained_variance": round(self.explained_variance, 6),
        }


@dataclass(frozen=True, slots=True)
class MarkerScore:
    """How one named marker compound fared.

    Attributes:
        name: Compound name.
        found: Whether a recovered component was assigned to it.
        spectral_similarity: Cosine agreement, 0.0 if not found.
        area_relative_error: Signed relative area error, ``nan`` if not found.
        retention_error_s: Apex error in seconds, ``nan`` if not found.
        true_area: Ground-truth area, for context on how small the marker is.
    """

    name: str
    found: bool
    spectral_similarity: float
    area_relative_error: float
    retention_error_s: float
    true_area: float


def score_resolution(
    true_profiles: np.ndarray,
    true_spectra: np.ndarray,
    retention_times: np.ndarray,
    result: ResolutionResult,
    *,
    observed: np.ndarray | None = None,
    retention_tolerance_s: float = 6.0,
    min_spectral_similarity: float = 0.80,
) -> ResolutionScore:
    """Score a resolution against the true factorisation.

    Args:
        true_profiles: Ground-truth ``C``, shape ``(n_scans, n_true)``.
        true_spectra: Ground-truth ``S``, shape ``(n_true, n_mz)``.
        retention_times: Shared time axis.
        result: The factorisation being scored.
        observed: Data matrix for the fit statistics. When ``None``, the true
            factorisation's own reconstruction is used as the reference, which
            measures the model against noise-free data.
        retention_tolerance_s: Retention agreement width for the pairing.
        min_spectral_similarity: Cosine below which a pair is rejected.

    Returns:
        The aggregate score.
    """
    matching = match_components(
        true_profiles,
        true_spectra,
        result.C,
        result.S,
        retention_times,
        retention_tolerance_s=retention_tolerance_s,
        min_spectral_similarity=min_spectral_similarity,
    )

    reference = true_profiles @ true_spectra if observed is None else observed
    reconstruction = result.reconstruction

    similarities = [match.spectral_similarity for match in matching.matches]
    retention_errors = [abs(match.retention_error_s) for match in matching.matches]

    return ResolutionScore(
        matching=matching,
        lack_of_fit=lack_of_fit(reference, reconstruction),
        explained_variance=explained_variance(reference, reconstruction),
        n_true=matching.n_true,
        n_recovered=matching.n_recovered,
        n_matched=len(matching.matches),
        mean_spectral_similarity=float(np.mean(similarities)) if similarities else 0.0,
        median_spectral_similarity=(
            float(np.median(similarities)) if similarities else 0.0
        ),
        median_absolute_area_error=matching.median_absolute_area_error,
        median_absolute_retention_error_s=(
            float(np.median(retention_errors)) if retention_errors else float("inf")
        ),
    )


def score_markers(
    true_profiles: np.ndarray,
    true_spectra: np.ndarray,
    retention_times: np.ndarray,
    result: ResolutionResult,
    component_names: list[str] | tuple[str, ...],
    marker_names: list[str] | tuple[str, ...],
    *,
    retention_tolerance_s: float = 6.0,
    min_spectral_similarity: float = 0.80,
) -> tuple[MarkerScore, ...]:
    """Score a resolution on specific, named compounds.

    Args:
        true_profiles: Ground-truth ``C``.
        true_spectra: Ground-truth ``S``.
        retention_times: Shared time axis.
        result: The factorisation being scored.
        component_names: Name of each ground-truth component, in column order.
        marker_names: The compounds to report on.
        retention_tolerance_s: Retention agreement width for the pairing.
        min_spectral_similarity: Cosine below which a pair is rejected.

    Returns:
        One :class:`MarkerScore` per requested marker, in the requested order.

    Raises:
        KeyError: If a requested marker is not among the ground-truth components.
    """
    matching = match_components(
        true_profiles,
        true_spectra,
        result.C,
        result.S,
        retention_times,
        retention_tolerance_s=retention_tolerance_s,
        min_spectral_similarity=min_spectral_similarity,
    )
    name_to_index = {name: index for index, name in enumerate(component_names)}
    true_areas = np.trapezoid(true_profiles, retention_times, axis=0)

    scores: list[MarkerScore] = []
    for marker in marker_names:
        if marker not in name_to_index:
            raise KeyError(f"{marker!r} is not among the ground-truth components")
        index = name_to_index[marker]
        match = matching.match_for_true(index)
        scores.append(
            MarkerScore(
                name=marker,
                found=match is not None,
                spectral_similarity=match.spectral_similarity if match else 0.0,
                area_relative_error=(
                    match.area_relative_error if match else float("nan")
                ),
                retention_error_s=match.retention_error_s if match else float("nan"),
                true_area=float(true_areas[index]),
            )
        )
    return tuple(scores)
