"""api/routes/health.py — Liveness probe."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="Liveness probe")
def health() -> dict:
    return {
        "status": "ok",
        "version": "2.0.0",
        "service": "SpatioCore Flow",
    }
