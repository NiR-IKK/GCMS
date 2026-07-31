"""Multivariate Curve Resolution by Alternating Least Squares.

MCR-ALS factorises a window of co-eluting peaks into pure elution profiles and
pure mass spectra: ``D ≈ C @ S``, minimising the residual while both factors are
held to constraints that encode what chromatography and mass spectrometry allow.

The honest caveat, stated up front because it governs how results must be judged:
**the factorisation is not unique.** For any invertible ``T``, ``(CT)(T⁻¹S)``
fits exactly as well as ``(C)(S)``. Constraints shrink the set of admissible
``T`` but rarely to a single point. A low lack-of-fit therefore proves the model
*fits*, not that it is *right* — which is why every result in this project is
scored against ground truth rather than against its own residual.

The constraints applied here are the ones with a physical justification:

* **Non-negativity** of both factors. A concentration below zero is meaningless,
  and so is a negative ion count.
* **Unimodality** of each elution profile. A compound eluting from a
  chromatographic column produces exactly one peak; a resolved profile with two
  maxima is a sign that two compounds have been merged into one component.
  This is the constraint that does most of the work against rotational
  ambiguity, and it is available precisely because the data is chromatographic.
* **Spectral normalisation**, which fixes the scale ambiguity between ``C`` and
  ``S`` and keeps the convention used everywhere else in this project: rows of
  ``S`` sum to one, so a column of ``C`` is that component's contribution to the
  total ion current.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from pyrecycle_analytics.deconvolution.result import ResolutionError, ResolutionResult
from pyrecycle_analytics.validation.metrics import explained_variance, lack_of_fit

__all__ = ["McrAlsOptions", "simplisma", "enforce_unimodality", "mcr_als"]


@dataclass(frozen=True, slots=True)
class McrAlsOptions:
    """Solver settings for :func:`mcr_als`.

    Attributes:
        max_iterations: Upper bound on ALS sweeps.
        tolerance: Relative change in lack of fit below which the solver stops.
        absolute_tolerance: Lack of fit (in percent) below which the fit counts as
            exact. Without it, noise-free data never converges: a clean
            two-component window settles at a lack of fit near 1e-8 %, where the
            *relative* change between successive sweeps stays large forever and
            the solver runs to its iteration limit on a perfect answer.
        non_negative_c: Constrain concentrations to be non-negative.
        non_negative_s: Constrain spectra to be non-negative.
        unimodal_c: Force each elution profile to have a single maximum.
        unimodality_tolerance: Fractional rise allowed against the trend before a
            profile counts as bimodal. A little slack absorbs noise on the
            flanks without permitting a genuine second peak.
        normalise_spectra: Rescale so each spectrum sums to one after every sweep.
        min_area_fraction: Components below this share of the largest component's
            area are dropped from the final result.
    """

    max_iterations: int = 200
    tolerance: float = 1e-6
    absolute_tolerance: float = 1e-6
    non_negative_c: bool = True
    non_negative_s: bool = True
    unimodal_c: bool = True
    unimodality_tolerance: float = 0.05
    normalise_spectra: bool = True
    min_area_fraction: float = 1e-3


def simplisma(
    data: np.ndarray,
    n_components: int,
    *,
    noise_fraction: float = 0.05,
) -> np.ndarray:
    """Select the purest variables as an initial spectral estimate.

    SIMPLISMA looks for m/z channels dominated by a single component — a channel
    whose relative standard deviation across the window is high compared with its
    mean is one where something appears and disappears rather than being present
    throughout. Those channels are chosen one at a time, each subsequent choice
    being penalised for correlation with the ones already taken, so the result
    spans the window's variation instead of picking the same peak repeatedly.

    A deterministic, data-driven start matters here: ALS descends to a local
    optimum, so the starting point partly decides the answer, and a random start
    would make results irreproducible.

    Args:
        data: Window matrix, shape ``(n_scans, n_mz)``.
        n_components: Number of pure variables to select.
        noise_fraction: Offset added to the mean when computing purity, as a
            fraction of the largest mean. It stops near-empty channels, whose
            relative standard deviation is enormous and meaningless, from being
            selected.

    Returns:
        Initial spectra, shape ``(n_components, n_mz)``, taken as the data rows
        at the selected purest positions.

    Raises:
        ResolutionError: If more components are requested than the data can
            support.
    """
    data = np.asarray(data, dtype=np.float64)
    n_scans, n_mz = data.shape
    if n_components < 1:
        raise ResolutionError(f"n_components must be >= 1, got {n_components}")
    if n_components > min(n_scans, n_mz):
        raise ResolutionError(
            f"cannot extract {n_components} components from a {n_scans}x{n_mz} window"
        )

    means = data.mean(axis=0)
    deviations = data.std(axis=0)
    offset = noise_fraction * float(means.max()) if means.max() > 0 else 1.0
    purity = deviations / (means + offset)

    # Length-normalised columns; correlation between them drives the penalty.
    norms = np.sqrt((data**2).sum(axis=0)) + offset
    normalised = data / norms

    selected: list[int] = []
    weights = np.ones(n_mz)
    for _ in range(n_components):
        weighted_purity = purity * weights
        if selected:
            weighted_purity[selected] = -np.inf
        candidate = int(np.argmax(weighted_purity))
        if not np.isfinite(weighted_purity[candidate]):
            break
        selected.append(candidate)

        # Determinant-based independence weighting: a channel highly correlated
        # with those already chosen contributes little new information.
        chosen = normalised[:, selected]
        gram = chosen.T @ chosen
        try:
            base = float(np.linalg.det(gram))
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            base = 0.0
        if abs(base) < 1e-30:
            weights = np.ones(n_mz)
            continue
        for channel in range(n_mz):
            extended = np.column_stack([chosen, normalised[:, channel]])
            extended_gram = extended.T @ extended
            weights[channel] = abs(float(np.linalg.det(extended_gram))) / abs(base)
        maximum = float(weights.max())
        if maximum > 0:
            weights /= maximum

    if not selected:  # pragma: no cover - defensive
        raise ResolutionError("SIMPLISMA could not select any pure variable")

    # Use the scan where each pure variable peaks as that component's spectrum.
    spectra = np.zeros((len(selected), n_mz))
    for row, channel in enumerate(selected):
        spectra[row, :] = data[int(np.argmax(data[:, channel])), :]

    totals = spectra.sum(axis=1, keepdims=True)
    return np.divide(spectra, totals, out=np.zeros_like(spectra), where=totals > 0)


def enforce_unimodality(profile: np.ndarray, tolerance: float = 0.05) -> np.ndarray:
    """Force a single maximum on an elution profile.

    Walking outward from the apex in both directions, each point is capped at its
    inward neighbour scaled by ``1 + tolerance``. A profile that tries to rise
    again — the signature of two compounds merged into one component — is pushed
    back down, while ordinary noise on the flanks is left alone.

    Args:
        profile: One elution profile, shape ``(n_scans,)``.
        tolerance: Fractional rise permitted against the trend.

    Returns:
        The corrected profile.
    """
    profile = np.asarray(profile, dtype=np.float64).copy()
    if profile.size < 3:
        return profile

    apex = int(np.argmax(profile))
    scale = 1.0 + max(tolerance, 0.0)

    for index in range(apex - 1, -1, -1):
        ceiling = profile[index + 1] * scale
        if profile[index] > ceiling:
            profile[index] = ceiling
    for index in range(apex + 1, profile.size):
        ceiling = profile[index - 1] * scale
        if profile[index] > ceiling:
            profile[index] = ceiling
    return profile


def _solve_non_negative(design: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Column-wise non-negative least squares, ``min ||design @ x - targets||``."""
    n_unknowns = design.shape[1]
    solution = np.zeros((n_unknowns, targets.shape[1]))
    for column in range(targets.shape[1]):
        solution[:, column], _ = nnls(design, targets[:, column])
    return solution


def mcr_als(
    data: np.ndarray,
    n_components: int,
    *,
    retention_times: np.ndarray | None = None,
    mz_axis: np.ndarray | None = None,
    initial_spectra: np.ndarray | None = None,
    options: McrAlsOptions | None = None,
) -> ResolutionResult:
    """Resolve a window into pure profiles and spectra.

    Args:
        data: Window matrix ``D``, shape ``(n_scans, n_mz)``, non-negative.
        n_components: Number of components to resolve, normally from
            :func:`~pyrecycle_analytics.deconvolution.rank.estimate_rank`.
        retention_times: Time axis for the result. Defaults to scan indices.
        mz_axis: m/z axis for the result. Defaults to channel indices.
        initial_spectra: Starting estimate, shape ``(n_components, n_mz)``.
            Defaults to a SIMPLISMA selection.
        options: Solver settings.

    Returns:
        The factorisation, with lack of fit, iteration count and convergence flag.

    Raises:
        ResolutionError: If the data is not a valid non-negative 2-D matrix, or
            the requested component count exceeds what the window can support.
    """
    options = options or McrAlsOptions()
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ResolutionError(f"expected a 2-D window, got {data.ndim}-D")
    n_scans, n_mz = data.shape
    if n_components < 1 or n_components > min(n_scans, n_mz):
        raise ResolutionError(
            f"n_components={n_components} is outside 1..{min(n_scans, n_mz)} "
            f"for a {n_scans}x{n_mz} window"
        )
    if np.any(data < 0.0):
        raise ResolutionError(
            "the window contains negative intensities; baseline-correct with "
            "clipping before resolving"
        )
    if not np.any(data):
        raise ResolutionError("the window is empty")

    spectra = (
        simplisma(data, n_components)
        if initial_spectra is None
        else np.atleast_2d(np.asarray(initial_spectra, dtype=np.float64)).copy()
    )
    if spectra.shape != (n_components, n_mz):
        raise ResolutionError(
            f"initial_spectra has shape {spectra.shape}, expected {(n_components, n_mz)}"
        )

    started = time.perf_counter()
    previous_lof = np.inf
    converged = False
    iteration = 0
    profiles = np.zeros((n_scans, n_components))

    for sweep in range(1, options.max_iterations + 1):
        iteration = sweep
        # --- concentrations given spectra -------------------------------------
        if options.non_negative_c:
            profiles = _solve_non_negative(spectra.T, data.T).T
        else:
            profiles = np.linalg.lstsq(spectra.T, data.T, rcond=None)[0].T

        if options.unimodal_c:
            for component in range(n_components):
                profiles[:, component] = enforce_unimodality(
                    profiles[:, component], options.unimodality_tolerance
                )

        # A component that has collapsed to zero cannot be recovered by further
        # sweeps; restarting it from the current residual gives it a way back.
        for component in range(n_components):
            if profiles[:, component].max() <= 0.0:
                residual = data - profiles @ spectra
                profiles[:, component] = np.clip(residual.sum(axis=1), 0.0, None)

        # --- spectra given concentrations -------------------------------------
        if options.non_negative_s:
            spectra = _solve_non_negative(profiles, data)
        else:
            spectra = np.linalg.lstsq(profiles, data, rcond=None)[0]

        if options.normalise_spectra:
            totals = spectra.sum(axis=1, keepdims=True)
            scale = np.where(totals > 0, totals, 1.0)
            spectra = spectra / scale
            profiles = profiles * scale.T

        current_lof = lack_of_fit(data, profiles @ spectra)
        # An essentially perfect fit has to count as converged. Judging only the
        # *relative* change would keep iterating forever on noise-free data: once
        # the lack of fit is at 1e-15, a step to 1e-16 is a 90 % relative change
        # and meaningless.
        if current_lof <= options.absolute_tolerance:
            converged = True
            previous_lof = current_lof
            break
        if previous_lof < np.inf:
            relative_change = abs(previous_lof - current_lof) / max(previous_lof, 1e-12)
            if relative_change < options.tolerance:
                previous_lof = current_lof
                converged = True
                break
        previous_lof = current_lof

    reconstruction = profiles @ spectra
    result = ResolutionResult(
        retention_times=(
            np.arange(n_scans, dtype=np.float64)
            if retention_times is None
            else retention_times
        ),
        mz_axis=np.arange(n_mz, dtype=np.float64) if mz_axis is None else mz_axis,
        C=profiles,
        S=spectra,
        lack_of_fit=lack_of_fit(data, reconstruction),
        explained_variance=explained_variance(data, reconstruction),
        n_iterations=iteration,
        converged=converged,
        method="mcr-als",
        diagnostics={
            "requested_components": n_components,
            "seconds": round(time.perf_counter() - started, 4),
            "initialisation": "simplisma" if initial_spectra is None else "supplied",
        },
    )
    return result.drop_negligible(options.min_area_fraction).sorted_by_retention()
