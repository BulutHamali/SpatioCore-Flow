"""
ui/app.py — SpatioCore Flow v2.0 Streamlit Application
=======================================================
Launch with:
    streamlit run ui/app.py

The app manages a simple session-state FSM:

    IDLE  ──(upload + submit)──>  RUNNING
    RUNNING  ──(poll complete)──>  COMPLETED
    RUNNING  ──(poll failed)───>  ERROR
    COMPLETED | ERROR  ──(reset)──>  IDLE

All API communication goes through ui.api_client.APIClient.
No raw httpx or requests calls inside this file.
"""

from __future__ import annotations

import time

import streamlit as st

from ui.api_client import APIClient, APIError
from ui.components import (
    render_audit_trail,
    render_cell_types,
    render_gate_history,
    render_header,
    render_overview,
    render_stage_tracker,
)

# ── App-wide configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpatioCore Flow",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session-state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict = {
    "page":          "idle",    # idle | running | completed | error
    "run_id":        None,
    "status":        None,
    "current_stage": None,
    "result":        None,
    "auditor":       None,
    "error_msg":     None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── API client (one per session) ──────────────────────────────────────────────
@st.cache_resource
def _get_client() -> APIClient:
    return APIClient()


client = _get_client()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://img.shields.io/badge/SpatioCore_Flow-v2.0-58a6ff?style=for-the-badge",
            use_column_width=True,
        )
        st.divider()

        # ── API connectivity ──────────────────────────────────────────────────
        st.subheader("🌐 API Status")
        try:
            health = client.health()
            st.success(f"Connected — {health.get('service', 'SpatioCore Flow')}")
        except Exception:
            st.error(
                "API unreachable. Start the backend with:\n\n"
                "```\nuvicorn api.app:app --port 8080\n```"
            )

        st.divider()

        # ── Current run info ──────────────────────────────────────────────────
        if st.session_state.run_id:
            st.subheader("📋 Current Run")
            st.code(str(st.session_state.run_id), language=None)
            st.caption(f"Status: **{st.session_state.status or '—'}**")
            if st.session_state.page in ("completed", "error"):
                if st.button("🔄 Start New Run", use_container_width=True):
                    for k, v in _DEFAULTS.items():
                        st.session_state[k] = v
                    st.rerun()

        st.divider()
        st.caption(
            "SpatioCore Flow is a **research prototype**.\n\n"
            "All outputs must be verified by a qualified investigator.\n\n"
            "Not a certified medical device."
        )


# ── Pages ─────────────────────────────────────────────────────────────────────

def _page_idle() -> None:
    """Upload page — shown when no pipeline run is active."""
    render_header()

    st.markdown("### Upload AnnData Dataset")
    uploaded = st.file_uploader(
        "Select a single-cell or spatial transcriptomics file (.h5ad)",
        type=["h5ad"],
        help="AnnData HDF5 format produced by Scanpy, Squidpy, or Cell Ranger.",
    )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("ℹ️  **Accepted modalities**")
        st.markdown(
            "- Single-cell RNA-seq (10x Chromium, Smart-seq2)\n"
            "- Spatial Visium / MERFISH / Slide-seq\n"
            "- Multiome (RNA + ATAC)"
        )
    with col_b:
        st.caption("ℹ️  **Pipeline stages**")
        st.markdown(
            "1. Curator — modality detection & QC\n"
            "2. Analyst — cell-type annotation (scGPT / Geneformer)\n"
            "3. Confidence Gate — BCS validation (≥ 0.80 required)\n"
            "4. Auditor — source-to-bit traceability report"
        )

    st.divider()
    if uploaded is not None:
        st.success(f"File ready: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")
        if st.button("▶ Run Analysis", type="primary", use_container_width=True):
            _submit_pipeline(uploaded)


def _submit_pipeline(uploaded) -> None:
    """Submit the uploaded file and transition to the RUNNING page."""
    with st.spinner("Submitting pipeline…"):
        try:
            run_id = client.submit(uploaded.name, uploaded.getvalue())
            st.session_state.run_id = run_id
            st.session_state.page   = "running"
            st.session_state.status = "pending"
            st.rerun()
        except APIError as e:
            st.error(f"Submission failed: {e.detail}")


def _page_running() -> None:
    """Progress page — polls /status until COMPLETED or FAILED."""
    render_header()
    run_id = st.session_state.run_id

    st.markdown(f"### Pipeline Running  `{run_id}`")
    stage_placeholder  = st.empty()
    status_placeholder = st.empty()

    # ── Poll until terminal state ─────────────────────────────────────────────
    while True:
        try:
            status_data = client.get_status(run_id)
        except APIError as e:
            st.session_state.page      = "error"
            st.session_state.error_msg = str(e)
            st.rerun()
            return

        job_status    = status_data["status"]
        current_stage = status_data.get("current_stage")

        with stage_placeholder.container():
            render_stage_tracker(current_stage, completed=job_status == "completed")
        with status_placeholder.container():
            st.info(f"Status: **{job_status}** · Stage: `{current_stage or '—'}`")

        if job_status == "completed":
            # Fetch result + auditor report once
            try:
                result  = client.get_result(run_id)
                auditor = client.get_auditor_report(run_id)
            except APIError as e:
                st.session_state.page      = "error"
                st.session_state.error_msg = str(e)
                st.rerun()
                return

            st.session_state.page    = "completed"
            st.session_state.status  = "completed"
            st.session_state.result  = result
            st.session_state.auditor = auditor
            st.rerun()
            return

        if job_status == "failed":
            st.session_state.page      = "error"
            st.session_state.error_msg = status_data.get("error", "Unknown error")
            st.rerun()
            return

        time.sleep(1.0)


def _page_completed() -> None:
    """Results dashboard — shown once the pipeline completes successfully."""
    render_header()
    result  = st.session_state.result
    auditor = st.session_state.auditor

    st.markdown(f"### Analysis Complete  `{st.session_state.run_id}`")
    render_stage_tracker(current_stage=None, completed=True)
    st.divider()

    # ── Tabbed results layout ─────────────────────────────────────────────────
    tab_overview, tab_gate, tab_cells, tab_audit = st.tabs(
        ["📊 Overview", "🚦 Gate History", "🔬 Cell Types", "📋 Audit Trail"]
    )

    with tab_overview:
        render_overview(result, auditor)

    with tab_gate:
        render_gate_history(result.get("gate_history", []))

    with tab_cells:
        render_cell_types(auditor)

    with tab_audit:
        render_audit_trail(auditor)


def _page_error() -> None:
    """Error page — shown when the pipeline job failed."""
    render_header()
    st.error("### Pipeline Failed")
    st.markdown(f"```\n{st.session_state.error_msg}\n```")
    st.caption(f"Run ID: `{st.session_state.run_id}`")
    if st.button("🔄 Start New Run", type="primary"):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────

_PAGES = {
    "idle":      _page_idle,
    "running":   _page_running,
    "completed": _page_completed,
    "error":     _page_error,
}


def main() -> None:
    _render_sidebar()
    page_fn = _PAGES.get(st.session_state.page, _page_idle)
    page_fn()


if __name__ == "__main__":
    main()

# Streamlit discovers the app by executing the module directly —
# call main() at module level so `streamlit run ui/app.py` works.
main()
