"""FastAPI service for the recyclate passport.

Analysis is a slow, CPU-bound job: a full-resolution pyrogram takes seconds to
minutes to resolve. So upload and analysis are separate calls, and analysis runs
as a background task with a status endpoint, rather than the client holding a
connection open and timing out on the interesting samples.

The web layer is deliberately thin. Everything it does is available from
:func:`~pyrecycle_analytics.reporting.analyse_pyrogram` without a server, which is
how the chemistry stays testable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from data_schemas.passport import RecyclatePassport
from pyrecycle_analytics import __version__
from pyrecycle_analytics.core.datacube import PyrogramDataCube
from pyrecycle_analytics.exceptions import PyRecycleError
from pyrecycle_analytics.ingestion.registry import read_pyrogram_bytes
from pyrecycle_analytics.library.repository import MarkerLibrary
from pyrecycle_analytics.reporting.analysis import analyse_pyrogram
from pyrecycle_analytics.reporting.reach import REACH_WATCHLIST
from pyrecycle_analytics.reporting.render import render_html, render_pdf

__all__ = ["create_app", "SampleStatus"]


class SampleStatus(StrEnum):
    """Lifecycle of an uploaded sample."""

    UPLOADED = "uploaded"
    ANALYSING = "analysing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(slots=True)
class _SampleRecord:
    """Server-side state of one sample.

    In-memory by design for this milestone. Persisting analyses belongs in the
    Milestone 3 database and is a deliberate follow-up, not an oversight: a
    restart currently loses results.
    """

    sample_id: str
    filename: str
    cube: PyrogramDataCube
    status: SampleStatus = SampleStatus.UPLOADED
    passport: RecyclatePassport | None = None
    error: str | None = None
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    warnings: tuple[str, ...] = ()


class AnalysisRequest(BaseModel):
    """Options for an analysis run."""

    first_carbon_number: int | None = Field(
        None,
        ge=4,
        le=40,
        description="Carbon number of the first n-alkane cluster, if known. "
        "Anchors the retention-index ladder; without it the anchor is estimated "
        "and the passport says so.",
    )
    calibrated: bool = Field(
        False,
        description="Set only when gravimetric reference blends were measured "
        "under the same method. Leaving it false labels the percentages "
        "semi-quantitative, which is almost always the truthful answer.",
    )


class SampleSummary(BaseModel):
    """Status of one sample."""

    sample_id: str
    filename: str
    status: SampleStatus
    uploaded_at: datetime
    n_scans: int
    n_mz: int
    error: str | None = None
    warnings: tuple[str, ...] = ()


def create_app(library: MarkerLibrary | None = None) -> FastAPI:
    """Build the API.

    Args:
        library: Marker library to serve. A fresh in-memory one is seeded when
            omitted.

    Returns:
        The configured application.
    """
    marker_library = library or MarkerLibrary.in_memory()
    samples: dict[str, _SampleRecord] = {}

    app = FastAPI(
        title="PyRecycle-Analytics",
        version=__version__,
        summary="Py-GC/MS deconvolution and recyclate passports for post-consumer plastics",
    )

    def _require(sample_id: str) -> _SampleRecord:
        record = samples.get(sample_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown sample {sample_id!r}")
        return record

    def _require_passport(sample_id: str) -> RecyclatePassport:
        record = _require(sample_id)
        if record.status is SampleStatus.FAILED:
            raise HTTPException(status_code=409, detail=record.error or "analysis failed")
        if record.passport is None:
            raise HTTPException(
                status_code=409,
                detail=f"sample {sample_id!r} is {record.status}; no passport yet",
            )
        return record.passport

    def _run_analysis(sample_id: str, request: AnalysisRequest) -> None:
        record = samples[sample_id]
        record.status = SampleStatus.ANALYSING
        try:
            result = analyse_pyrogram(
                record.cube,
                marker_library,
                first_carbon_number=request.first_carbon_number,
                calibrated=request.calibrated,
            )
        except PyRecycleError as error:
            record.status = SampleStatus.FAILED
            record.error = str(error)
            return
        record.passport = result.passport
        record.warnings = result.warnings
        record.status = SampleStatus.COMPLETE

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness check with library size, so a bad deployment is visible."""
        return {
            "status": "ok",
            "version": __version__,
            "library_compounds": len(marker_library.compounds()),
            "library_patterns": len(marker_library.patterns()),
        }

    @app.post("/samples", response_model=SampleSummary, status_code=201)
    async def upload_sample(file: UploadFile) -> SampleSummary:
        """Upload a raw pyrogram (ANDI-MS ``.CDF``, mzML or mzXML)."""
        payload = await file.read()
        try:
            cube = read_pyrogram_bytes(payload, file.filename or "upload.cdf")
        except (PyRecycleError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        sample_id = uuid.uuid4().hex[:12]
        samples[sample_id] = _SampleRecord(
            sample_id=sample_id,
            filename=file.filename or "upload.cdf",
            cube=cube,
            warnings=cube.metadata.reader_warnings,
        )
        return _summary(samples[sample_id])

    @app.get("/samples", response_model=list[SampleSummary])
    def list_samples() -> list[SampleSummary]:
        """Every sample known to this instance."""
        return [_summary(record) for record in samples.values()]

    @app.get("/samples/{sample_id}", response_model=SampleSummary)
    def get_sample(sample_id: str) -> SampleSummary:
        """Status and metadata of one sample."""
        return _summary(_require(sample_id))

    @app.post("/samples/{sample_id}/analyse", response_model=SampleSummary, status_code=202)
    def analyse(
        sample_id: str, request: AnalysisRequest, background: BackgroundTasks
    ) -> SampleSummary:
        """Start an analysis.

        Returns immediately with 202; poll the sample endpoint for completion.
        Resolving a full-resolution pyrogram takes far longer than a request
        should be held open for.
        """
        record = _require(sample_id)
        if record.status is SampleStatus.ANALYSING:
            raise HTTPException(status_code=409, detail="analysis already running")
        background.add_task(_run_analysis, sample_id, request)
        record.status = SampleStatus.ANALYSING
        return _summary(record)

    @app.get("/samples/{sample_id}/passport", response_model=RecyclatePassport)
    def get_passport(sample_id: str) -> RecyclatePassport:
        """The recyclate passport as JSON."""
        return _require_passport(sample_id)

    @app.get("/samples/{sample_id}/passport.html", response_class=HTMLResponse)
    def get_passport_html(sample_id: str) -> HTMLResponse:
        """The passport as a printable HTML document."""
        return HTMLResponse(render_html(_require_passport(sample_id)))

    @app.get("/samples/{sample_id}/passport.pdf")
    def get_passport_pdf(sample_id: str) -> Response:
        """The passport as PDF."""
        import tempfile
        from pathlib import Path

        passport = _require_passport(sample_id)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = render_pdf(passport, Path(directory) / f"{sample_id}.pdf")
                payload = path.read_bytes()
        except PyRecycleError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        return Response(
            content=payload,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="passport-{sample_id}.pdf"'
                )
            },
        )

    @app.get("/library/polymers")
    def list_polymers() -> dict[str, Any]:
        """Polymers and marker patterns the library can report on."""
        return {
            "polymers": marker_library.polymer_codes(),
            "patterns": [
                {
                    "name": pattern.name,
                    "polymer": pattern.polymer.code,
                    "description": pattern.description,
                    "members": [
                        {
                            "compound": member.compound.name,
                            "role": member.role,
                            "relative_abundance": member.relative_abundance,
                        }
                        for member in pattern.members
                    ],
                }
                for pattern in marker_library.patterns()
            ],
        }

    @app.get("/library/regulated-substances")
    def list_regulated_substances() -> dict[str, Any]:
        """The regulatory watchlist, and what is outside the method's reach."""
        from pyrecycle_analytics.reporting.reach import NON_GC_AMENABLE_NOTE

        return {
            "substances": [
                {
                    "name": substance.name,
                    "cas_number": substance.cas_number,
                    "regulation": substance.regulation,
                    "limit_percent": substance.limit_percent,
                    "marker_compound": substance.marker_compound,
                    "note": substance.note,
                }
                for substance in REACH_WATCHLIST
            ],
            "out_of_scope": NON_GC_AMENABLE_NOTE,
        }

    def _summary(record: _SampleRecord) -> SampleSummary:
        return SampleSummary(
            sample_id=record.sample_id,
            filename=record.filename,
            status=record.status,
            uploaded_at=record.uploaded_at,
            n_scans=record.cube.n_scans,
            n_mz=record.cube.n_mz,
            error=record.error,
            warnings=record.warnings,
        )

    return app
