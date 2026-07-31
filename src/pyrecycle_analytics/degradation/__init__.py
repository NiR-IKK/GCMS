"""Thermo-oxidative ageing indices."""

from __future__ import annotations

from pyrecycle_analytics.degradation.engine import (
    HYDROCARBON_IONS,
    OXIDATION_IONS,
    DegradationIndices,
    chain_length_statistics,
    compute_degradation_indices,
)

__all__ = [
    "HYDROCARBON_IONS",
    "OXIDATION_IONS",
    "DegradationIndices",
    "chain_length_statistics",
    "compute_degradation_indices",
]
