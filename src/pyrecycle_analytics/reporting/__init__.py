"""The digital recyclate passport and the analysis that produces it."""

from __future__ import annotations

from pyrecycle_analytics.reporting.analysis import AnalysisResult, analyse_pyrogram
from pyrecycle_analytics.reporting.passport import (
    MatrixPolyolefinEvidence,
    PassportInputs,
    build_passport,
)
from pyrecycle_analytics.reporting.reach import (
    NON_GC_AMENABLE_NOTE,
    REACH_WATCHLIST,
    RegulatedSubstance,
)
from pyrecycle_analytics.reporting.render import render_html, render_pdf

__all__ = [
    "NON_GC_AMENABLE_NOTE",
    "REACH_WATCHLIST",
    "AnalysisResult",
    "MatrixPolyolefinEvidence",
    "PassportInputs",
    "RegulatedSubstance",
    "analyse_pyrogram",
    "build_passport",
    "render_html",
    "render_pdf",
]
