"""
ui/api_client.py — Thin synchronous httpx wrapper for the SpatioCore Flow API.

All methods raise ``APIError`` on non-2xx responses so callers can display
a single, consistent error banner rather than catching httpx internals.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

# Configurable via environment; default assumes local dev server
_DEFAULT_BASE = "http://localhost:8080"


class APIError(Exception):
    """Raised when the SpatioCore Flow API returns a non-2xx status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class APIClient:
    """
    Synchronous client for the SpatioCore Flow REST API.

    Parameters
    ----------
    base_url :
        API base URL.  Reads ``API_BASE_URL`` env var if not supplied.
    timeout :
        Per-request timeout in seconds (default 30 s).
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("API_BASE_URL", _DEFAULT_BASE)).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> APIClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _raise_for_status(self, r: httpx.Response) -> None:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise APIError(r.status_code, str(detail))

    # ── Endpoints ─────────────────────────────────────────────────────────────

    def health(self) -> dict:
        r = self._client.get("/health")
        self._raise_for_status(r)
        return r.json()

    def submit(self, filename: str, content: bytes) -> str:
        """Upload a .h5ad file and return the new run_id (str)."""
        r = self._client.post(
            "/pipeline/run",
            files={"file": (filename, content, "application/octet-stream")},
        )
        self._raise_for_status(r)
        return str(r.json()["run_id"])

    def get_status(self, run_id: str) -> dict:
        r = self._client.get(f"/pipeline/{run_id}/status")
        self._raise_for_status(r)
        return r.json()

    def get_result(self, run_id: str) -> dict:
        """Returns the result dict or raises APIError(202) if still running."""
        r = self._client.get(f"/pipeline/{run_id}/result")
        if r.status_code == 202:
            try:
                detail = r.json().get("detail", "Pipeline is running.")
            except Exception:
                detail = "Pipeline is running."
            raise APIError(202, str(detail))
        self._raise_for_status(r)
        return r.json()

    def get_gate_history(self, run_id: str) -> list[dict]:
        r = self._client.get(f"/pipeline/{run_id}/gate-history")
        self._raise_for_status(r)
        return r.json()

    def get_auditor_report(self, run_id: str) -> dict:
        r = self._client.get(f"/pipeline/{run_id}/auditor-report")
        self._raise_for_status(r)
        return r.json()
