"""Measuring how good a resolution is.

Nothing in Milestones 2 and 3 can be claimed without this package: curve
resolution returns components in arbitrary order, so a result must first be
paired with the ground truth before any metric means anything.
"""

from __future__ import annotations

from pyrecycle_analytics.validation.baseline_method import naive_peak_integration
from pyrecycle_analytics.validation.matching import (
    ComponentMatch,
    ComponentMatching,
    match_components,
)
from pyrecycle_analytics.validation.metrics import (
    cosine_similarity,
    cosine_similarity_matrix,
    explained_variance,
    lack_of_fit,
    relative_area_error,
    weighted_dot_similarity,
)
from pyrecycle_analytics.validation.scoring import (
    MarkerScore,
    ResolutionScore,
    score_markers,
    score_resolution,
)

__all__ = [
    "ComponentMatch",
    "ComponentMatching",
    "MarkerScore",
    "ResolutionScore",
    "cosine_similarity",
    "cosine_similarity_matrix",
    "explained_variance",
    "lack_of_fit",
    "match_components",
    "naive_peak_integration",
    "relative_area_error",
    "score_markers",
    "score_resolution",
    "weighted_dot_similarity",
]
