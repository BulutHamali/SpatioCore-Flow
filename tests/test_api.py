"""
tests/test_api.py — FastAPI route tests for SpatioCore Flow v2.0.

All tests use FastAPI's TestClient (synchronous) against a fresh app instance.
No LLM, no network — only the in-process stub runners and a synthetic AnnData.

The pipeline background task runs synchronously inside TestClient, so by the
time we poll /status we see COMPLETED (no explicit sleep required).

Synthetic dataset: 5 genes × 60 cells, normal expression.
"""

from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path

import pytest

anndata = pytest.importorskip("anndata")
numpy  = pytest.importorskip("numpy")

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.app import create_app
from api.state import JobStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

GENES   = ["CD8A", "CD4", "FOXP3", "MKI67", "EPCAM"]
N_CELLS = 60


def _make_h5ad_bytes() -> bytes:
    """Build a minimal AnnData and serialise it to an in-memory .h5ad buffer."""
    rng = np.random.default_rng(seed=99)
    X   = np.clip(rng.exponential(2.5, (N_CELLS, len(GENES))), 0.1, None).astype(np.float32)
    var = pd.DataFrame(index=pd.Index(GENES))
    obs = pd.DataFrame(index=pd.Index([f"cell_{i}" for i in range(N_CELLS)]))
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
        adata.write_h5ad(f.name)
        return Path(f.name).read_bytes()


H5AD_BYTES = _make_h5ad_bytes()   # computed once per test session


@pytest.fixture()
def client():
    """Fresh app + TestClient for each test — guarantees an empty JobStore."""
    app = create_app()
    with TestClient(app) as c:
        yield c


def _upload(client: TestClient, content: bytes = H5AD_BYTES, filename: str = "data.h5ad"):
    """Helper: POST /pipeline/run and return the JSON response."""
    return client.post(
        "/pipeline/run",
        files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
    )


def _wait_for_completion(client: TestClient, run_id: str, max_polls: int = 30) -> dict:
    """Poll /status until COMPLETED or FAILED (or timeout)."""
    for _ in range(max_polls):
        r = client.get(f"/pipeline/{run_id}/status")
        body = r.json()
        if body["status"] in (JobStatus.COMPLETED, JobStatus.FAILED):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"Pipeline {run_id} did not complete after {max_polls} polls")


# ─────────────────────────────────────────────────────────────────────────────
# 1 — Health check
# ─────────────────────────────────────────────────────────────────────────────

def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "2.0.0"
    assert body["service"] == "SpatioCore Flow"


# ─────────────────────────────────────────────────────────────────────────────
# 2 — Submit a valid .h5ad → 202 + run_id
# ─────────────────────────────────────────────────────────────────────────────

def test_submit_valid_h5ad_returns_run_id(client: TestClient) -> None:
    r = _upload(client)
    assert r.status_code == 202
    body = r.json()
    assert "run_id" in body
    assert body["status"] == JobStatus.PENDING


# ─────────────────────────────────────────────────────────────────────────────
# 3 — Status transitions to COMPLETED
# ─────────────────────────────────────────────────────────────────────────────

def test_status_reaches_completed(client: TestClient) -> None:
    run_id = _upload(client).json()["run_id"]
    status_body = _wait_for_completion(client, run_id)
    assert status_body["status"] == JobStatus.COMPLETED
    assert status_body["completed_at"] is not None
    assert status_body["error"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Result endpoint returns expected fields
# ─────────────────────────────────────────────────────────────────────────────

def test_result_contains_bcs_and_traceability(client: TestClient) -> None:
    run_id = _upload(client).json()["run_id"]
    _wait_for_completion(client, run_id)

    r = client.get(f"/pipeline/{run_id}/result")
    assert r.status_code == 200
    body = r.json()

    assert body["status"] == JobStatus.COMPLETED
    assert 0.0 <= body["final_bcs"] <= 1.0
    assert 0.0 <= body["traceability_coverage"] <= 1.0
    assert body["total_analyst_attempts"] >= 1
    assert body["pipeline_duration_ms"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5 — Gate history is present in result
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_history_in_result(client: TestClient) -> None:
    run_id = _upload(client).json()["run_id"]
    _wait_for_completion(client, run_id)

    r = client.get(f"/pipeline/{run_id}/result")
    gate_history = r.json()["gate_history"]

    assert isinstance(gate_history, list)
    assert len(gate_history) >= 1
    first = gate_history[0]
    assert first["attempt"] == 1
    assert first["gate_decision"] == "pass"
    assert 0.0 <= first["computed_bcs"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6 — Dedicated gate-history endpoint
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_history_endpoint(client: TestClient) -> None:
    run_id = _upload(client).json()["run_id"]
    _wait_for_completion(client, run_id)

    r = client.get(f"/pipeline/{run_id}/gate-history")
    assert r.status_code == 200
    history = r.json()
    assert isinstance(history, list) and len(history) >= 1
    assert all("gate_decision" in entry for entry in history)


# ─────────────────────────────────────────────────────────────────────────────
# 7 — Auditor report endpoint
# ─────────────────────────────────────────────────────────────────────────────

def test_auditor_report_endpoint(client: TestClient) -> None:
    run_id = _upload(client).json()["run_id"]
    _wait_for_completion(client, run_id)

    r = client.get(f"/pipeline/{run_id}/auditor-report")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == run_id
    assert 0.0 <= body["traceability_coverage"] <= 1.0
    assert body["total_claims"] >= 1
    assert len(body["verified_claims"]) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 8 — Unknown run_id → 404
# ─────────────────────────────────────────────────────────────────────────────

def test_unknown_run_id_returns_404(client: TestClient) -> None:
    fake_id = "00000000-0000-0000-0000-000000000000"
    for endpoint in ("status", "result", "gate-history", "auditor-report"):
        r = client.get(f"/pipeline/{fake_id}/{endpoint}")
        assert r.status_code == 404, f"Expected 404 for /{endpoint}, got {r.status_code}"


# ─────────────────────────────────────────────────────────────────────────────
# 9 — Wrong file extension → 422
# ─────────────────────────────────────────────────────────────────────────────

def test_non_h5ad_upload_returns_422(client: TestClient) -> None:
    r = _upload(client, content=b"not a real file", filename="data.csv")
    assert r.status_code == 422
    assert "h5ad" in r.json()["detail"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 10 — Empty file → 422
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_h5ad_upload_returns_422(client: TestClient) -> None:
    r = _upload(client, content=b"", filename="empty.h5ad")
    assert r.status_code == 422
    assert "empty" in r.json()["detail"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 11 — /result on still-running job → 202
# ─────────────────────────────────────────────────────────────────────────────

def test_result_on_pending_job_returns_202(client: TestClient) -> None:
    """
    Manually insert a PENDING record (no background task) and verify that
    calling /result returns 202 with a descriptive message.
    """
    from uuid import uuid4
    run_id = uuid4()
    client.app.state.job_store.create(run_id)   # PENDING, no task started

    r = client.get(f"/pipeline/{run_id}/result")
    assert r.status_code == 202
    assert "pending" in r.json()["detail"].lower()
