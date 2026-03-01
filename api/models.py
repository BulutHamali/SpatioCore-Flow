"""
api/models.py — FastAPI request / response schemas for the pipeline API.

These are *API-layer* Pydantic models — deliberately thin wrappers around the
richer internal schemas in logic/schemas.py.  Keeping them separate means the
HTTP contract can evolve independently from the internal data models.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from api.state import JobStatus


class RunResponse(BaseModel):
    """Returned immediately after POST /pipeline/run."""

    run_id: UUID
    status: JobStatus = JobStatus.PENDING


class StatusResponse(BaseModel):
    """Returned by GET /pipeline/{run_id}/status."""

    run_id: UUID
    status: JobStatus
    current_stage: str | None = Field(
        None,
        description="Active pipeline stage: 'curator', 'analyst_attempt_N', 'auditor', or null.",
    )
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = Field(None, description="Error message if status is 'failed'.")


class GateSummary(BaseModel):
    """One row in the gate history — summary of a single ValidatorReport."""

    attempt: int
    gate_decision: str
    computed_bcs: float
    verified_genes: int
    rejected_genes: int
    failure_reasons: list[str] = Field(default_factory=list)


class ResultResponse(BaseModel):
    """
    Returned by GET /pipeline/{run_id}/result once the job is COMPLETED.

    Contains the high-level summary; use the dedicated endpoints for the
    full AuditorReport or gate history.
    """

    run_id: UUID
    status: JobStatus
    total_analyst_attempts: int
    final_bcs: float = Field(..., ge=0.0, le=1.0)
    traceability_coverage: float = Field(..., ge=0.0, le=1.0)
    total_claims: int = Field(..., ge=0)
    pipeline_duration_ms: float = Field(..., ge=0.0)
    gate_history: list[GateSummary] = Field(default_factory=list)
