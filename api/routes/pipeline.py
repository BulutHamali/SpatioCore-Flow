"""
api/routes/pipeline.py — Pipeline submission and status endpoints.

Routes
------
POST   /pipeline/run                   Upload .h5ad → start background job
GET    /pipeline/{run_id}/status       Poll job lifecycle stage
GET    /pipeline/{run_id}/result       Full summary once COMPLETED
GET    /pipeline/{run_id}/gate-history Per-attempt Confidence Gate reports
GET    /pipeline/{run_id}/auditor-report  Full AuditorReport
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile, status

from api.models import GateSummary, ResultResponse, RunResponse, StatusResponse
from api.state import JobStatus

router = APIRouter(prefix="/pipeline")

# ── Accepted file extension ───────────────────────────────────────────────────
_ACCEPTED_EXT = ".h5ad"


# ── Background task ───────────────────────────────────────────────────────────

def _execute_pipeline(run_id: UUID, h5ad_path: str, store) -> None:
    """
    Load AnnData, build the gate + orchestrator, run the full pipeline, and
    write the result (or error) back into the job store.

    Executed in a FastAPI BackgroundTask after the HTTP response is sent.
    """
    from datetime import datetime, timezone

    store.update(run_id, status=JobStatus.RUNNING, current_stage="curator")
    try:
        # ── Load AnnData ──────────────────────────────────────────────────────
        try:
            import scanpy as sc
            adata = sc.read_h5ad(h5ad_path)
        except ImportError:
            import anndata
            adata = anndata.read_h5ad(h5ad_path)

        # ── Build pipeline components ─────────────────────────────────────────
        from logic.gates import ValidationGate
        from logic.orchestrator import SpatioFlowOrchestrator
        from api.runners import make_default_runners

        gate = ValidationGate(adata=adata)
        orchestrator = SpatioFlowOrchestrator(gate=gate)
        curator_runner, analyst_runner, auditor_runner = make_default_runners(adata)

        # ── Execute ───────────────────────────────────────────────────────────
        store.update(run_id, current_stage="analyst_attempt_1")
        result = orchestrator.run(
            adata=adata,
            curator_runner=curator_runner,
            analyst_runner=analyst_runner,
            auditor_runner=auditor_runner,
            run_id=run_id,
        )

        store.update(
            run_id,
            status=JobStatus.COMPLETED,
            current_stage=None,
            completed_at=datetime.now(timezone.utc),
            result=result,
        )

    except Exception as exc:  # noqa: BLE001
        store.update(
            run_id,
            status=JobStatus.FAILED,
            current_stage=None,
            completed_at=datetime.now(timezone.utc),
            error=str(exc),
        )
    finally:
        Path(h5ad_path).unlink(missing_ok=True)


# ── Route helpers ─────────────────────────────────────────────────────────────

def _get_store(request: Request):
    return request.app.state.job_store


def _require_job(run_id: UUID, store):
    record = store.get(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run ID {run_id} not found.",
        )
    return record


def _require_completed(record):
    if record.status == JobStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {record.error}",
        )
    if record.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Pipeline is {record.status.value}. Poll /status and retry when completed.",
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/run",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an AnnData file and start the analysis pipeline",
)
async def run_pipeline(
    request: Request,
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> RunResponse:
    """
    Upload a ``.h5ad`` file to start a new pipeline run.

    The run executes asynchronously. Poll ``GET /pipeline/{run_id}/status``
    to track progress, then retrieve results via the ``/result`` endpoint.
    """
    if not (file.filename or "").endswith(_ACCEPTED_EXT):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Only {_ACCEPTED_EXT} files are accepted. Got: {file.filename!r}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    # Persist to a temp file so the background task can read it after the
    # request handler returns (UploadFile is closed at that point).
    tmp = tempfile.NamedTemporaryFile(suffix=_ACCEPTED_EXT, delete=False)
    tmp.write(content)
    tmp.close()

    run_id = uuid4()
    store = _get_store(request)
    store.create(run_id)

    background_tasks.add_task(_execute_pipeline, run_id, tmp.name, store)

    return RunResponse(run_id=run_id, status=JobStatus.PENDING)


@router.get(
    "/{run_id}/status",
    response_model=StatusResponse,
    summary="Poll pipeline run status",
)
def get_status(run_id: UUID, request: Request) -> StatusResponse:
    store = _get_store(request)
    record = _require_job(run_id, store)
    return StatusResponse(
        run_id=record.run_id,
        status=record.status,
        current_stage=record.current_stage,
        created_at=record.created_at,
        completed_at=record.completed_at,
        error=record.error,
    )


@router.get(
    "/{run_id}/result",
    response_model=ResultResponse,
    summary="Retrieve pipeline result summary (COMPLETED runs only)",
)
def get_result(run_id: UUID, request: Request) -> ResultResponse:
    store = _get_store(request)
    record = _require_job(run_id, store)
    _require_completed(record)

    result = record.result  # OrchestratorResult
    gate_summaries = [
        GateSummary(
            attempt=vr.analyst_attempt,
            gate_decision=vr.gate_decision.value,
            computed_bcs=vr.computed_bcs,
            verified_genes=len(vr.verified_marker_genes),
            rejected_genes=len(vr.rejected_marker_genes),
            failure_reasons=vr.failure_reasons,
        )
        for vr in result.gate_history
    ]
    return ResultResponse(
        run_id=result.run_id,
        status=JobStatus.COMPLETED,
        total_analyst_attempts=result.total_analyst_attempts,
        final_bcs=result.final_validator_report.computed_bcs,
        traceability_coverage=result.auditor_report.traceability_coverage,
        total_claims=result.auditor_report.total_claims,
        pipeline_duration_ms=result.pipeline_duration_ms,
        gate_history=gate_summaries,
    )


@router.get(
    "/{run_id}/gate-history",
    response_model=list[GateSummary],
    summary="Per-attempt Confidence Gate reports",
)
def get_gate_history(run_id: UUID, request: Request) -> list[GateSummary]:
    store = _get_store(request)
    record = _require_job(run_id, store)
    _require_completed(record)

    return [
        GateSummary(
            attempt=vr.analyst_attempt,
            gate_decision=vr.gate_decision.value,
            computed_bcs=vr.computed_bcs,
            verified_genes=len(vr.verified_marker_genes),
            rejected_genes=len(vr.rejected_marker_genes),
            failure_reasons=vr.failure_reasons,
        )
        for vr in record.result.gate_history
    ]


@router.get(
    "/{run_id}/auditor-report",
    summary="Full AuditorReport (source-to-bit traceability)",
)
def get_auditor_report(run_id: UUID, request: Request) -> dict:
    store = _get_store(request)
    record = _require_job(run_id, store)
    _require_completed(record)
    return record.result.auditor_report.model_dump(mode="json")
