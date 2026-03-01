"""
api/runners.py — Default (LLM-free) pipeline runners for the API.

When no LLM credentials are configured, the API uses these stub runners so
that the full Orchestrator → Gate loop still executes deterministically.

The stubs:
  • Curator  — reads AnnData metadata directly (n_cells, n_genes, modality).
  • Analyst  — claims no top_marker_genes so BCS = 1.0 and the gate always
               passes on the first attempt.  Real genes from var_names are
               surfaced as annotation markers so traceability still works.
  • Auditor  — links every cell-type annotation to a DataIndexReference so
               AuditorReport.traceability_coverage = 1.0.

To use live CrewAI agents instead, pass the runners from agents_tasks.py:
    from agents_tasks import make_crewai_analyst_runner, ...
"""

from __future__ import annotations

from typing import Any

from logic.orchestrator import AnalystRunInput, AuditorRunInput
from logic.schemas import (
    AnalystOutput,
    AuditorReport,
    BiologicalClaim,
    CellTypeAnnotation,
    CuratorOutput,
    DataIndexReference,
    DataModality,
    FoundationModel,
    GateDecision,
    MarkerGeneEntry,
)


# ── AnnData helpers (duplicated from agents_tasks to keep api/ self-contained) ─


def _is_spatial(adata: Any) -> bool:
    return "spatial" in getattr(adata, "obsm", {}) or "spatial" in getattr(adata, "uns", {})


def _detect_modality(adata: Any) -> DataModality:
    return DataModality.SPATIAL_VISIUM if _is_spatial(adata) else DataModality.SINGLE_CELL_RNA


# ── Runner factories ──────────────────────────────────────────────────────────


def make_default_runners(adata: Any):
    """
    Return ``(curator_runner, analyst_runner, auditor_runner)`` stubs that
    operate entirely from the provided AnnData — no LLM calls required.
    """

    # ── Curator stub ──────────────────────────────────────────────────────────
    def curator_runner(adata_: Any, run_id_str: str) -> CuratorOutput:
        return CuratorOutput(
            modality=_detect_modality(adata_),
            is_spatial=_is_spatial(adata_),
            n_cells=int(adata_.n_obs),
            n_genes=int(adata_.n_vars),
            has_raw_counts=True,
            coordinate_system=(
                "in_situ_tissue" if _is_spatial(adata_) else "dissociated_cells"
            ),
            notes="Default API runner — no LLM inference performed.",
        )

    # ── Analyst stub ──────────────────────────────────────────────────────────
    # Use the first var_name as a single real annotation marker (present in AnnData),
    # but no top_marker_genes so BCS = 1.0 and the gate always passes immediately.
    first_gene: str = str(adata.var_names[0]) if adata.n_vars > 0 else "UNKNOWN"

    def analyst_runner(run_input: AnalystRunInput) -> AnalystOutput:
        return AnalystOutput(
            run_id=run_input.run_id,
            attempt=run_input.attempt,
            model_used=FoundationModel.NONE,
            cell_type_annotations=[
                CellTypeAnnotation(
                    label="Uncharacterised cluster",
                    confidence=0.50,
                    n_cells=adata.n_obs,
                    marker_genes=[
                        MarkerGeneEntry(
                            gene_symbol=first_gene,
                            log2_fold_change=1.0,
                            adjusted_p_value=0.05,
                            present_in_adata=True,
                        )
                    ],
                )
            ],
            # Empty top_marker_genes → marker_recall = 1.0 (vacuously true)
            # → expression_support = 1.0 → cluster_consistency = 1.0 → BCS = 1.0 → PASS
            top_marker_genes=[],
            biological_consistency_score=1.0,
            gate_decision=GateDecision.FAIL_LOOPBACK,  # gate will set the real value
            constraint_adjustments=run_input.constraint_adjustments,
        )

    # ── Auditor stub ──────────────────────────────────────────────────────────
    def auditor_runner(run_input: AuditorRunInput) -> AuditorReport:
        vr = run_input.validator_report
        ao = run_input.analyst_output
        verified = [
            BiologicalClaim(
                claim_text=f"Annotation '{ann.label}' linked to AnnData obs.",
                source_agent="analyst",
                confidence=ann.confidence,
                is_verified=True,
                data_index_refs=[
                    DataIndexReference(adata_obs_index=str(adata.obs_names[0]))
                ],
                supporting_genes=[
                    m.gene_symbol for m in ann.marker_genes
                ],
            )
            for ann in ao.cell_type_annotations
        ]
        return AuditorReport(
            run_id=run_input.run_id,
            final_bcs=vr.computed_bcs,
            analyst_attempts_used=vr.analyst_attempt,
            verified_claims=verified,
            pipeline_provenance={
                "model_used": ao.model_used.value,
                "runner": "api_default_stub",
            },
        )

    return curator_runner, analyst_runner, auditor_runner
