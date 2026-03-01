"""
ui/components.py — SpatioCore Flow Streamlit render functions.

Each function is a pure rendering unit that accepts plain Python data
(dicts / lists) — no Streamlit session_state or API calls inside.
This keeps them independently testable.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# ── Colour palette (matches CLI header ANSI theme) ────────────────────────────
_GREEN  = "#28a745"
_BLUE   = "#007bff"
_CYAN   = "#17a2b8"
_YELLOW = "#ffc107"
_RED    = "#dc3545"
_GREY   = "#6c757d"

# ── Gate decision → display mapping ──────────────────────────────────────────
_DECISION_BADGE: dict[str, tuple[str, str]] = {
    "pass":           ("✅ PASS",         _GREEN),
    "fail_loopback":  ("🔄 LOOPBACK",     _YELLOW),
    "fail_rejected":  ("❌ REJECTED",     _RED),
}

# ── Stage → canonical index ───────────────────────────────────────────────────
_STAGE_ORDER = ["curator", "analyst", "gate", "auditor", "guardrail"]
_STAGE_LABEL = {
    "curator":          "Curator",
    "analyst":          "Analyst",
    "analyst_attempt_1": "Analyst",
    "analyst_attempt_2": "Analyst",
    "analyst_attempt_3": "Analyst",
    "gate":             "Gate",
    "auditor":          "Auditor",
    "guardrail":        "Guardrail",
}


# ── Public helpers (testable without Streamlit) ───────────────────────────────

def bcs_colour(bcs: float, threshold: float = 0.80) -> str:
    """Return a hex colour string for a BCS value relative to its threshold."""
    if bcs >= threshold:
        return _GREEN
    if bcs >= threshold * 0.85:
        return _YELLOW
    return _RED


def decision_badge(decision: str) -> tuple[str, str]:
    """Return (label, hex_colour) for a gate_decision string."""
    return _DECISION_BADGE.get(decision.lower(), (decision.upper(), _GREY))


def stage_index(stage_key: str | None) -> int:
    """Map a raw stage key to its position in the pipeline (0-based)."""
    if stage_key is None:
        return -1
    normalised = _STAGE_LABEL.get(stage_key.lower(), stage_key).lower()
    try:
        return _STAGE_ORDER.index(normalised)
    except ValueError:
        return -1


# ── Render functions ──────────────────────────────────────────────────────────

def render_header() -> None:
    """Top-of-page branding banner."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            border-radius: 12px;
            padding: 24px 32px 20px;
            margin-bottom: 24px;
            border: 1px solid #30363d;
        ">
            <h1 style="color:#58a6ff; font-family:monospace; margin:0; letter-spacing:2px;">
                SpatioCore Flow <span style="font-size:0.55em; color:#8b949e;">v2.0</span>
            </h1>
            <p style="color:#8b949e; margin:6px 0 0; font-size:0.9em;">
                Gated Multi-Agent Bioinformatics Orchestrator
                &nbsp;·&nbsp; scRNA-seq &amp; Spatial Transcriptomics
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stage_tracker(current_stage: str | None, completed: bool) -> None:
    """Horizontal pipeline stage indicator with colour-coded status."""
    stages = ["Curator", "Analyst", "Gate", "Auditor", "Guardrail"]
    active_idx = stage_index(current_stage)
    final_idx  = len(stages) if completed else active_idx

    cols = st.columns(len(stages))
    for i, (col, label) in enumerate(zip(cols, stages)):
        with col:
            if i < final_idx:
                st.success(f"✓ {label}")
            elif i == active_idx and not completed:
                st.info(f"⟳ {label}")
            else:
                st.markdown(
                    f"<div style='text-align:center;color:{_GREY};padding:8px 0'>· {label}</div>",
                    unsafe_allow_html=True,
                )


def render_overview(result: dict, auditor: dict) -> None:
    """
    Four headline metrics + a BCS progress bar.

    Parameters
    ----------
    result  : dict matching ResultResponse JSON
    auditor : dict matching AuditorReport JSON
    """
    bcs          = result["final_bcs"]
    coverage     = result["traceability_coverage"]
    attempts     = result["total_analyst_attempts"]
    duration_s   = result["pipeline_duration_ms"] / 1000
    total_claims = auditor.get("total_claims", 0)
    colour       = bcs_colour(bcs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧬 Final BCS",          f"{bcs:.3f}",    delta=f"{bcs - 0.80:+.3f} vs threshold")
    c2.metric("🔍 Traceability",        f"{coverage:.0%}")
    c3.metric("🔁 Analyst Attempts",    str(attempts))
    c4.metric("⏱ Duration",            f"{duration_s:.1f} s")

    st.markdown(f"**Biological Consistency Score** — `{bcs:.3f}`")
    st.markdown(
        f'<div style="background:{_GREY};border-radius:4px;height:12px;margin-bottom:16px;">'
        f'<div style="background:{colour};border-radius:4px;height:12px;width:{min(bcs, 1.0)*100:.1f}%"></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Threshold reference line label
    st.caption(f"🎯 Threshold: 0.800  ·  Verified claims: {total_claims}")


def render_gate_history(gate_history: list[dict]) -> None:
    """
    Gate attempt timeline: colour-coded table + BCS bar chart.

    Deep link: failure reasons expand per row.
    """
    if not gate_history:
        st.info("No gate history available.")
        return

    st.subheader("Confidence Gate History")

    # Summary table
    rows = []
    for g in gate_history:
        label, _ = decision_badge(g["gate_decision"])
        rows.append({
            "Attempt":          g["attempt"],
            "Decision":         label,
            "BCS":              round(g["computed_bcs"], 4),
            "Verified genes":   g["verified_genes"],
            "Rejected genes":   g["rejected_genes"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # BCS trend bar chart
    chart_data = pd.DataFrame(
        {"BCS": [g["computed_bcs"] for g in gate_history]},
        index=[f"Attempt {g['attempt']}" for g in gate_history],
    )
    st.bar_chart(chart_data, color=_CYAN)

    # Failure reasons (deep-link expanders)
    for g in gate_history:
        reasons = g.get("failure_reasons", [])
        if reasons:
            label, colour = decision_badge(g["gate_decision"])
            with st.expander(f"Attempt {g['attempt']} — {label} · details"):
                for r in reasons:
                    st.markdown(f"- {r}")


def render_cell_types(auditor: dict) -> None:
    """
    Cell-type annotation table derived from the AuditorReport verified_claims.
    Each claim expands to show raw AnnData index deep links.
    """
    claims = auditor.get("verified_claims", [])
    if not claims:
        st.info("No verified cell-type claims in this run.")
        return

    st.subheader("Cell-Type Annotations")

    summary_rows = [
        {
            "Cell type / Claim":  c["claim_text"],
            "Confidence":         round(c["confidence"], 3),
            "Source agent":       c.get("source_agent", "—"),
            "Verified":           "✓" if c.get("is_verified") else "✗",
        }
        for c in claims
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Deep-link expanders — click to see raw AnnData evidence
    st.caption("Click a claim to view supporting AnnData indices:")
    for c in claims:
        with st.expander(f"📍 {c['claim_text']}", expanded=False):
            refs = c.get("data_index_refs", [])
            spatial = c.get("spatial_refs", [])
            genes  = c.get("supporting_genes", [])

            if genes:
                st.markdown("**Supporting genes:**")
                st.code("  ".join(genes))

            if refs:
                st.markdown("**AnnData index references:**")
                for ref in refs:
                    parts = []
                    if ref.get("adata_obs_index"):
                        parts.append(f"obs['{ref['adata_obs_index']}']")
                    if ref.get("adata_var_index"):
                        parts.append(f"var['{ref['adata_var_index']}']")
                    if ref.get("layer_key"):
                        parts.append(f"layers['{ref['layer_key']}']")
                    if parts:
                        st.code(" · ".join(parts))

            if spatial:
                st.markdown("**Spatial coordinates:**")
                for s in spatial:
                    st.code(
                        f"barcode={s.get('spot_barcode','?')}  "
                        f"row={s.get('array_row','?')}  "
                        f"col={s.get('array_col','?')}"
                    )

            if not refs and not spatial:
                st.caption("No raw-data evidence links recorded.")


def render_audit_trail(auditor: dict) -> None:
    """
    Full AuditorReport summary: traceability coverage gauge + verified /
    unverified claim breakdown.
    """
    st.subheader("Audit Trail")

    coverage      = auditor.get("traceability_coverage", 0.0)
    total         = auditor.get("total_claims", 0)
    n_verified    = len(auditor.get("verified_claims", []))
    n_unverified  = len(auditor.get("unverified_claims", []))
    final_bcs     = auditor.get("final_bcs", 0.0)
    attempts_used = auditor.get("analyst_attempts_used", 1)
    colour        = bcs_colour(coverage, threshold=1.0) if coverage == 1.0 else (
        _GREEN if coverage >= 0.9 else (_YELLOW if coverage >= 0.7 else _RED)
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Traceability Coverage", f"{coverage:.0%}")
    col_b.metric("Verified Claims",       str(n_verified))
    col_c.metric("Unverified Claims",     str(n_unverified),
                 delta=f"-{n_unverified}" if n_unverified else None,
                 delta_color="inverse")

    # Provenance chain
    provenance = auditor.get("pipeline_provenance", {})
    if provenance:
        with st.expander("🔗 Pipeline Provenance"):
            for k, v in provenance.items():
                st.markdown(f"- **{k}**: `{v}`")

    # Unverified claims require investigator review
    if n_unverified:
        st.warning(
            f"⚠️  {n_unverified} unverified claim(s) require investigator review "
            "before this report can be used in a clinical context.",
            icon="⚠️",
        )

    st.caption(
        f"audit_id: `{auditor.get('audit_id', '—')}`  ·  "
        f"run_id: `{auditor.get('run_id', '—')}`  ·  "
        f"final BCS: `{final_bcs:.4f}`  ·  "
        f"analyst attempts: `{attempts_used}`"
    )
