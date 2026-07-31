"""The output type every resolution method produces.

Whether a factorisation came from MCR-ALS, from the naive comparison baseline, or
later from PARAFAC2, it is returned in this shape so that scoring, reporting and
the marker library never have to care which method produced it.

The convention matches the ground truth from Milestone 1: rows of ``S`` sum to
one over the acquired m/z window, so each column of ``C`` *is* that component's
contribution to the total ion current, and its integral is the component's area.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyrecycle_analytics.exceptions import PyRecycleError

__all__ = ["ResolutionResult", "ResolutionError"]


class ResolutionError(PyRecycleError):
    """Curve resolution could not produce a usable factorisation."""


@dataclass(slots=True)
class ResolutionResult:
    """A bilinear factorisation ``D ≈ C @ S`` of one pyrogram or window.

    Attributes:
        retention_times: Time axis of ``C``, shape ``(n_scans,)``.
        mz_axis: m/z axis of ``S``, shape ``(n_mz,)``.
        C: Elution profiles, shape ``(n_scans, n_components)``, area-scaled.
        S: Mass spectra, shape ``(n_components, n_mz)``, rows summing to one.
        lack_of_fit: Residual as a percentage of the data norm.
        explained_variance: ``R²`` of the reconstruction.
        n_iterations: Iterations the solver used.
        converged: Whether the convergence criterion was met.
        method: Name of the producing algorithm, for provenance.
        diagnostics: Free-form solver detail (rank estimates, warnings, timings).
    """

    retention_times: np.ndarray
    mz_axis: np.ndarray
    C: np.ndarray
    S: np.ndarray
    lack_of_fit: float = 0.0
    explained_variance: float = 0.0
    n_iterations: int = 0
    converged: bool = True
    method: str = "unknown"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.retention_times = np.asarray(self.retention_times, dtype=np.float64).ravel()
        self.mz_axis = np.asarray(self.mz_axis, dtype=np.float64).ravel()
        self.C = np.atleast_2d(np.asarray(self.C, dtype=np.float64))
        self.S = np.atleast_2d(np.asarray(self.S, dtype=np.float64))

        if self.C.shape[0] != self.retention_times.size:
            raise ResolutionError(
                f"C has {self.C.shape[0]} rows but the time axis has "
                f"{self.retention_times.size} points"
            )
        if self.S.shape[1] != self.mz_axis.size:
            raise ResolutionError(
                f"S has {self.S.shape[1]} columns but the m/z axis has "
                f"{self.mz_axis.size} channels"
            )
        if self.C.shape[1] != self.S.shape[0]:
            raise ResolutionError(
                f"C has {self.C.shape[1]} components but S has {self.S.shape[0]}"
            )

    @property
    def n_components(self) -> int:
        return int(self.C.shape[1])

    @property
    def n_scans(self) -> int:
        return int(self.C.shape[0])

    @property
    def reconstruction(self) -> np.ndarray:
        """The modelled data matrix ``C @ S``."""
        return self.C @ self.S

    @property
    def areas(self) -> np.ndarray:
        """Integrated area of each component, shape ``(n_components,)``."""
        if self.n_scans < 2:
            return np.zeros(self.n_components)
        return np.trapezoid(self.C, self.retention_times, axis=0)

    @property
    def apex_times(self) -> np.ndarray:
        """Retention time of each component's maximum, shape ``(n_components,)``."""
        return self.retention_times[np.argmax(self.C, axis=0)]

    def sorted_by_retention(self) -> ResolutionResult:
        """A copy with components ordered by apex, the natural chromatographic order."""
        order = np.argsort(self.apex_times, kind="stable")
        return ResolutionResult(
            retention_times=self.retention_times.copy(),
            mz_axis=self.mz_axis.copy(),
            C=self.C[:, order].copy(),
            S=self.S[order, :].copy(),
            lack_of_fit=self.lack_of_fit,
            explained_variance=self.explained_variance,
            n_iterations=self.n_iterations,
            converged=self.converged,
            method=self.method,
            diagnostics=dict(self.diagnostics),
        )

    def drop_negligible(self, min_area_fraction: float = 1e-4) -> ResolutionResult:
        """Remove components carrying a negligible share of the total area.

        A solver asked for more components than the data contains will return
        near-empty ones; keeping them inflates the false-positive count without
        adding information.

        Args:
            min_area_fraction: Threshold relative to the largest component's area.

        Returns:
            A copy without the negligible components; never empty — the largest
            component is always retained.
        """
        areas = self.areas
        if areas.size == 0:
            return self
        largest = float(np.max(areas))
        if largest <= 0.0:
            return self
        keep = np.flatnonzero(areas >= min_area_fraction * largest)
        if keep.size == 0:
            keep = np.array([int(np.argmax(areas))])
        return ResolutionResult(
            retention_times=self.retention_times.copy(),
            mz_axis=self.mz_axis.copy(),
            C=self.C[:, keep].copy(),
            S=self.S[keep, :].copy(),
            lack_of_fit=self.lack_of_fit,
            explained_variance=self.explained_variance,
            n_iterations=self.n_iterations,
            converged=self.converged,
            method=self.method,
            diagnostics=dict(self.diagnostics),
        )

    def __repr__(self) -> str:
        return (
            f"ResolutionResult(method={self.method!r}, components={self.n_components}, "
            f"lof={self.lack_of_fit:.2f}%, converged={self.converged})"
        )
