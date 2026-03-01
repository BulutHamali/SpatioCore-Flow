"""
api/state.py — In-process job store for pipeline run tracking.

Tracks the lifecycle of every pipeline run submitted to the API:
  PENDING → RUNNING → COMPLETED | FAILED

In a production deployment this store would be backed by PostgreSQL / Redis.
For the current scope it is an in-process dict, which is sufficient for a
single-worker server and keeps the dependency footprint minimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobRecord:
    run_id: UUID
    status: JobStatus = JobStatus.PENDING
    current_stage: str | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime | None = None
    result: Any | None = None   # OrchestratorResult once COMPLETED
    error: str | None = None    # traceback string if FAILED


class JobStore:
    """
    Thread-safe in-process job store.

    All mutations go through ``create`` / ``update`` so the store can be
    swapped for a database-backed implementation without changing the routes.
    """

    def __init__(self) -> None:
        self._jobs: dict[UUID, JobRecord] = {}

    # ── Mutations ─────────────────────────────────────────────────────────────

    def create(self, run_id: UUID) -> JobRecord:
        record = JobRecord(run_id=run_id)
        self._jobs[run_id] = record
        return record

    def update(self, run_id: UUID, **kwargs: Any) -> None:
        record = self._jobs.get(run_id)
        if record is None:
            return
        for key, value in kwargs.items():
            setattr(record, key, value)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get(self, run_id: UUID) -> JobRecord | None:
        return self._jobs.get(run_id)

    def __len__(self) -> int:
        return len(self._jobs)
