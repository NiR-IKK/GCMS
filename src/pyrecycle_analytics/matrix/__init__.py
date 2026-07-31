"""Modelling and removing the dominant polymer matrix."""

from __future__ import annotations

from pyrecycle_analytics.matrix.polyolefin import (
    MATRIX_INDICATOR_IONS,
    CombDetection,
    MatrixSubtractionError,
    MatrixSubtractionResult,
    PolyolefinMatrixModel,
    build_matrix_model,
    detect_homologue_comb,
    subtract_polymer_matrix,
)

__all__ = [
    "MATRIX_INDICATOR_IONS",
    "CombDetection",
    "MatrixSubtractionError",
    "MatrixSubtractionResult",
    "PolyolefinMatrixModel",
    "build_matrix_model",
    "detect_homologue_comb",
    "subtract_polymer_matrix",
]
