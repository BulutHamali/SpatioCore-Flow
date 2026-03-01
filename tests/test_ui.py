"""
tests/test_ui.py — UI layer tests for SpatioCore Flow v2.0.

Tests two distinct layers:
  1. APIClient — mocked httpx transport (no real server required)
  2. ui.components helpers — pure functions, no Streamlit runtime

Streamlit rendering (render_* functions) is not exercised here because
those functions require a live Streamlit session context.  The integration
between components and the Streamlit runtime is covered by manual smoke-
testing or end-to-end tests outside this suite.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import httpx

from ui.api_client import APIClient, APIError
from ui.components import bcs_colour, decision_badge, stage_index


# ─────────────────────────────────────────────────────────────────────────────
# APIClient — mocked transport helpers
# ─────────────────────────────────────────────────────────────────────────────

class _MockTransport(httpx.BaseTransport):
    """
    Synchronous mock transport that returns a pre-configured response.
    Registered on the httpx.Client at construction time.
    """

    def __init__(self, status_code: int, body: dict | list) -> None:
        self._status = status_code
        self._body   = body

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=self._status,
            headers={"content-type": "application/json"},
            content=json.dumps(self._body).encode(),
            request=request,
        )


def _client_with(status: int, body: dict | list) -> APIClient:
    """Return an APIClient whose underlying httpx.Client uses the mock transport."""
    c = APIClient.__new__(APIClient)
    c.base_url = "http://testserver"
    c._client  = httpx.Client(
        base_url="http://testserver",
        transport=_MockTransport(status, body),
    )
    return c


# ─────────────────────────────────────────────────────────────────────────────
# APIClient.health
# ─────────────────────────────────────────────────────────────────────────────

def test_health_ok() -> None:
    client = _client_with(200, {"status": "ok", "version": "2.0.0", "service": "SpatioCore Flow"})
    result = client.health()
    assert result["status"] == "ok"
    assert result["version"] == "2.0.0"


def test_health_error_raises_api_error() -> None:
    client = _client_with(503, {"detail": "Service unavailable"})
    with pytest.raises(APIError) as exc_info:
        client.health()
    assert exc_info.value.status_code == 503
    assert "Service unavailable" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# APIClient.submit
# ─────────────────────────────────────────────────────────────────────────────

def test_submit_returns_run_id() -> None:
    run_id = "abc-123-def"
    client = _client_with(202, {"run_id": run_id, "status": "pending"})
    result = client.submit("data.h5ad", b"fake_content")
    assert result == run_id


def test_submit_422_raises_api_error() -> None:
    client = _client_with(422, {"detail": "Only .h5ad files are accepted."})
    with pytest.raises(APIError) as exc_info:
        client.submit("data.csv", b"content")
    assert exc_info.value.status_code == 422
    assert "h5ad" in exc_info.value.detail.lower()


# ─────────────────────────────────────────────────────────────────────────────
# APIClient.get_status
# ─────────────────────────────────────────────────────────────────────────────

def test_get_status_running() -> None:
    payload = {
        "run_id":        "abc",
        "status":        "running",
        "current_stage": "analyst_attempt_1",
        "created_at":    "2025-01-01T00:00:00Z",
        "completed_at":  None,
        "error":         None,
    }
    client = _client_with(200, payload)
    result = client.get_status("abc")
    assert result["status"] == "running"
    assert result["current_stage"] == "analyst_attempt_1"


def test_get_status_404_raises() -> None:
    client = _client_with(404, {"detail": "Run ID not found."})
    with pytest.raises(APIError) as exc_info:
        client.get_status("nonexistent")
    assert exc_info.value.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# APIClient.get_result
# ─────────────────────────────────────────────────────────────────────────────

def test_get_result_completed() -> None:
    payload = {
        "run_id":                "abc",
        "status":                "completed",
        "total_analyst_attempts": 1,
        "final_bcs":             0.92,
        "traceability_coverage": 1.0,
        "total_claims":          3,
        "pipeline_duration_ms":  1200.5,
        "gate_history": [
            {
                "attempt":         1,
                "gate_decision":   "pass",
                "computed_bcs":    0.92,
                "verified_genes":  2,
                "rejected_genes":  0,
                "failure_reasons": [],
            }
        ],
    }
    client = _client_with(200, payload)
    result = client.get_result("abc")
    assert result["final_bcs"] == pytest.approx(0.92)
    assert len(result["gate_history"]) == 1


def test_get_result_still_running_raises_202() -> None:
    client = _client_with(202, {"detail": "Pipeline is running."})
    with pytest.raises(APIError) as exc_info:
        client.get_result("abc")
    assert exc_info.value.status_code == 202


# ─────────────────────────────────────────────────────────────────────────────
# APIClient.get_gate_history
# ─────────────────────────────────────────────────────────────────────────────

def test_get_gate_history_returns_list() -> None:
    payload = [
        {"attempt": 1, "gate_decision": "pass", "computed_bcs": 0.95,
         "verified_genes": 3, "rejected_genes": 0, "failure_reasons": []},
    ]
    client = _client_with(200, payload)
    history = client.get_gate_history("abc")
    assert isinstance(history, list)
    assert history[0]["gate_decision"] == "pass"


# ─────────────────────────────────────────────────────────────────────────────
# APIClient.get_auditor_report
# ─────────────────────────────────────────────────────────────────────────────

def test_get_auditor_report_fields() -> None:
    payload = {
        "audit_id":              "audit-1",
        "run_id":                "abc",
        "final_bcs":             0.92,
        "analyst_attempts_used": 1,
        "traceability_coverage": 1.0,
        "total_claims":          2,
        "verified_claims":       [],
        "unverified_claims":     [],
    }
    client = _client_with(200, payload)
    report = client.get_auditor_report("abc")
    assert report["traceability_coverage"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# components.bcs_colour
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bcs, expected_substring", [
    (0.95, "#28a745"),   # green — above threshold
    (0.70, "#ffc107"),   # yellow — below threshold but >= 85% of it
    (0.50, "#dc3545"),   # red — well below threshold
    (0.80, "#28a745"),   # exactly at threshold → green
])
def test_bcs_colour_thresholds(bcs: float, expected_substring: str) -> None:
    assert bcs_colour(bcs) == expected_substring


# ─────────────────────────────────────────────────────────────────────────────
# components.decision_badge
# ─────────────────────────────────────────────────────────────────────────────

def test_decision_badge_pass() -> None:
    label, colour = decision_badge("pass")
    assert "PASS" in label
    assert colour == "#28a745"


def test_decision_badge_loopback() -> None:
    label, colour = decision_badge("fail_loopback")
    assert "LOOPBACK" in label
    assert colour == "#ffc107"


def test_decision_badge_rejected() -> None:
    label, colour = decision_badge("fail_rejected")
    assert "REJECTED" in label
    assert colour == "#dc3545"


def test_decision_badge_unknown_falls_back() -> None:
    label, colour = decision_badge("mystery_decision")
    assert label == "MYSTERY_DECISION"
    assert colour == "#6c757d"


# ─────────────────────────────────────────────────────────────────────────────
# components.stage_index
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key, expected_idx", [
    ("curator",           0),
    ("analyst_attempt_1", 1),
    ("analyst_attempt_3", 1),
    ("auditor",           3),
    ("guardrail",         4),
    (None,               -1),
    ("unknown_stage",    -1),
])
def test_stage_index_mapping(key: str | None, expected_idx: int) -> None:
    assert stage_index(key) == expected_idx


# ─────────────────────────────────────────────────────────────────────────────
# APIClient context manager
# ─────────────────────────────────────────────────────────────────────────────

def test_api_client_context_manager_closes() -> None:
    client = _client_with(200, {"status": "ok"})
    closed_calls = []
    original_close = client._client.close
    client._client.close = lambda: closed_calls.append(1) or original_close()

    with client:
        pass

    assert len(closed_calls) == 1
