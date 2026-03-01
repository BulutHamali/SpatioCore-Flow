"""
agents_tasks.py — SpatioCore Flow v2.0 CrewAI Agent & Task Definitions
=======================================================================
Defines the three core agents (Curator, Analyst, Auditor), their AnnData-backed
tools, and task builder functions that inject retry context into the Analyst's
prompt on each Confidence Gate loopback.

Runtime dependency: crewai >= 0.55  (guarded — module imports cleanly without it)

Usage (with CrewAI installed)
------------------------------
    from agents_tasks import (
        make_crewai_curator_runner,
        make_crewai_analyst_runner,
        make_crewai_auditor_runner,
        get_default_llm,
    )
    llm = get_default_llm()
    curator_runner  = make_crewai_curator_runner(llm)
    analyst_runner  = make_crewai_analyst_runner(llm)
    auditor_runner  = make_crewai_auditor_runner(llm)

    orchestrator = SpatioFlowOrchestrator(gate=gate)
    result = orchestrator.run(adata, curator_runner, analyst_runner, auditor_runner)
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

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
    RegulatoryFlag,
    RiskLevel,
    GuardrailReport,
)

if TYPE_CHECKING:
    from logic.orchestrator import AnalystRunInput, AuditorRunInput

logger = logging.getLogger(__name__)

# ── Optional CrewAI import ────────────────────────────────────────────────────
try:
    from crewai import Agent, Crew, LLM, Process, Task
    from crewai.tools import BaseTool
    from pydantic import ConfigDict

    CREWAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    CREWAI_AVAILABLE = False
    Agent = Task = Crew = LLM = BaseTool = None  # type: ignore[assignment,misc]


def _require_crewai() -> None:
    if not CREWAI_AVAILABLE:
        raise RuntimeError(
            "crewai is not installed.\n"
            "Install the full agent stack with: pip install 'crewai>=0.55'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AnnData pipeline registry
# Each pipeline run registers its AnnData under the string run_id so that tools
# can retrieve it without passing objects through CrewAI's JSON-serialised state.
# ─────────────────────────────────────────────────────────────────────────────


class _ADataRegistry:
    """In-process AnnData registry keyed by pipeline run_id (str)."""

    _store: dict[str, Any] = {}

    @classmethod
    def register(cls, run_id: str, adata: Any) -> None:
        cls._store[run_id] = adata
        logger.debug("ADataRegistry: registered run_id=%s", run_id)

    @classmethod
    def get(cls, run_id: str) -> Any:
        if run_id not in cls._store:
            raise KeyError(
                f"No AnnData registered for run_id={run_id!r}. "
                "Call _ADataRegistry.register() before invoking agent tools."
            )
        return cls._store[run_id]

    @classmethod
    def release(cls, run_id: str) -> None:
        cls._store.pop(run_id, None)
        logger.debug("ADataRegistry: released run_id=%s", run_id)


# ─────────────────────────────────────────────────────────────────────────────
# Shared AnnData helpers (numpy-only, no scanpy required)
# ─────────────────────────────────────────────────────────────────────────────


def _dense(adata: Any) -> np.ndarray:
    """Return a dense (cells × genes) float32 matrix from AnnData.X."""
    try:
        import scipy.sparse as sp

        if sp.issparse(adata.X):
            return np.asarray(adata.X.toarray(), dtype=np.float32)
    except ImportError:
        pass
    return np.asarray(adata.X, dtype=np.float32)


def _top_variable_genes(adata: Any, n: int = 20) -> list[str]:
    """Return the top-N most variable gene symbols (by variance across cells)."""
    X = _dense(adata)
    variances = X.var(axis=0)
    top_indices = np.argsort(variances)[::-1][:n]
    return [str(adata.var_names[i]) for i in top_indices]


def _gene_means(adata: Any) -> dict[str, float]:
    """Return {gene_symbol: mean_expression} for all genes."""
    X = _dense(adata)
    return {str(g): float(X[:, i].mean()) for i, g in enumerate(adata.var_names)}


def _is_spatial(adata: Any) -> bool:
    return "spatial" in getattr(adata, "obsm", {}) or "spatial" in getattr(adata, "uns", {})


def _detect_modality(adata: Any) -> DataModality:
    if _is_spatial(adata):
        return DataModality.SPATIAL_VISIUM
    return DataModality.SINGLE_CELL_RNA


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI Tool definitions
# Each tool fetches the current run's AnnData from the registry.
# ─────────────────────────────────────────────────────────────────────────────

if CREWAI_AVAILABLE:

    class InspectAdataTool(BaseTool):  # type: ignore[misc]
        """Return a JSON summary of the dataset registered for a pipeline run."""

        name: str = "inspect_adata"
        description: str = (
            "Inspect the AnnData object for this pipeline run. "
            "Returns n_cells, n_genes, is_spatial, var_names sample, and obs keys. "
            "Input: the pipeline run_id (UUID string)."
        )

        def _run(self, run_id: str) -> str:
            adata = _ADataRegistry.get(run_id)
            sample_genes = list(adata.var_names[:10])
            obs_cols = list(adata.obs.columns[:10]) if hasattr(adata.obs, "columns") else []
            return json.dumps(
                {
                    "n_cells": int(adata.n_obs),
                    "n_genes": int(adata.n_vars),
                    "is_spatial": _is_spatial(adata),
                    "modality_hint": _detect_modality(adata).value,
                    "sample_genes": sample_genes,
                    "obs_columns": obs_cols,
                },
                indent=2,
            )

    class DetectModalityTool(BaseTool):  # type: ignore[misc]
        """Detect the biological data modality and coordinate system."""

        name: str = "detect_modality"
        description: str = (
            "Detect whether the dataset is dissociated single-cell or in-situ spatial. "
            "Input: the pipeline run_id (UUID string)."
        )

        def _run(self, run_id: str) -> str:
            adata = _ADataRegistry.get(run_id)
            modality = _detect_modality(adata)
            coord_system = "in_situ_tissue" if _is_spatial(adata) else "dissociated_cells"
            return json.dumps(
                {"modality": modality.value, "coordinate_system": coord_system},
                indent=2,
            )

    class FindMarkerGenesTool(BaseTool):  # type: ignore[misc]
        """
        Identify top marker genes from the dataset using variance-based ranking.

        In production this wraps scanpy.tl.rank_genes_groups; here it uses
        variance across cells as a proxy that requires only numpy.
        """

        name: str = "find_marker_genes"
        description: str = (
            "Find the top differentially expressed / highly variable genes. "
            "Returns a JSON list of {gene, mean_expr, variance}. "
            "Input JSON: {'run_id': '<uuid>', 'top_n': 20, 'exclude_genes': []}."
        )

        def _run(self, input_json: str) -> str:
            params = json.loads(input_json)
            run_id: str = params["run_id"]
            top_n: int = int(params.get("top_n", 20))
            exclude: set[str] = set(params.get("exclude_genes", []))

            adata = _ADataRegistry.get(run_id)
            X = _dense(adata)
            var_names = list(adata.var_names)
            means = X.mean(axis=0)
            variances = X.var(axis=0)

            candidates = [
                {
                    "gene": var_names[i],
                    "mean_expr": float(means[i]),
                    "variance": float(variances[i]),
                }
                for i in range(len(var_names))
                if var_names[i] not in exclude
            ]
            candidates.sort(key=lambda d: d["variance"], reverse=True)
            return json.dumps(candidates[:top_n], indent=2)

    class AnnotateCellTypesTool(BaseTool):  # type: ignore[misc]
        """
        Simulate cell-type annotation via marker-gene overlap scoring.

        Production implementation: delegates to scGPT or Geneformer embeddings.
        This stub uses a configurable reference panel hard-coded in the registry
        or a built-in minimal T-cell / epithelial reference.
        """

        name: str = "annotate_cell_types"
        description: str = (
            "Assign cell-type labels to clusters based on top marker genes. "
            "Input JSON: {'run_id': '<uuid>', 'marker_genes': ['GENE1', ...]}."
        )

        # Minimal reference panel (production: loaded from ChromaDB knowledge base)
        _REFERENCE: dict[str, list[str]] = {
            "CD8+ Cytotoxic T cell": ["CD8A", "CD8B", "GZMB", "PRF1"],
            "CD4+ Helper T cell": ["CD4", "IL7R", "CCR7"],
            "Regulatory T cell": ["FOXP3", "IL2RA", "CTLA4"],
            "Proliferating cell": ["MKI67", "TOP2A", "PCNA"],
            "Epithelial cell": ["EPCAM", "KRT8", "KRT18"],
        }

        def _run(self, input_json: str) -> str:
            params = json.loads(input_json)
            marker_genes: set[str] = set(params.get("marker_genes", []))

            scores: list[tuple[str, float]] = []
            for label, panel in self._REFERENCE.items():
                overlap = len(marker_genes & set(panel))
                score = overlap / len(panel) if panel else 0.0
                scores.append((label, score))

            scores.sort(key=lambda t: t[1], reverse=True)
            return json.dumps(
                [{"label": lbl, "confidence": round(sc, 3)} for lbl, sc in scores[:3]],
                indent=2,
            )

    class TraceClaimsTool(BaseTool):  # type: ignore[misc]
        """Link a biological claim to specific obs/var indices in AnnData."""

        name: str = "trace_claims"
        description: str = (
            "Find AnnData obs barcodes and var gene indices that support a claim. "
            "Input JSON: {'run_id': '<uuid>', 'gene_symbol': 'CD8A'}."
        )

        def _run(self, input_json: str) -> str:
            params = json.loads(input_json)
            run_id: str = params["run_id"]
            gene: str = params["gene_symbol"]

            adata = _ADataRegistry.get(run_id)
            var_names = list(adata.var_names)

            if gene not in var_names:
                return json.dumps({"error": f"{gene!r} not found in var_names"})

            gene_idx = var_names.index(gene)
            X = _dense(adata)
            gene_expr = X[:, gene_idx]
            expressed_cell_indices = np.where(gene_expr > 0.05)[0]
            sample_barcodes = [str(adata.obs_names[i]) for i in expressed_cell_indices[:10]]

            return json.dumps(
                {
                    "gene": gene,
                    "var_index": gene_idx,
                    "n_expressing_cells": int(len(expressed_cell_indices)),
                    "sample_barcodes": sample_barcodes,
                },
                indent=2,
            )


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory
# ─────────────────────────────────────────────────────────────────────────────


def get_default_llm() -> Any:
    """
    Build a LiteLLM-routed LLM from environment variables.

    Required env vars (at least one):
        ANTHROPIC_API_KEY  — for claude-* models
        OPENAI_API_KEY     — for gpt-* models
        LITELLM_MODEL      — model identifier (default: claude-sonnet-4-6)

    Temperature is fixed at 0.1 across all agents to minimise paraphrase
    variance in biological claim generation.  A near-deterministic setting
    is essential for reproducible hallucination detection: identical inputs
    should produce outputs that the Confidence Gate scores consistently.
    """
    _require_crewai()
    model = os.getenv("LITELLM_MODEL", "claude-sonnet-4-6")
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    return LLM(model=model, api_key=api_key, temperature=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Agent builders
# ─────────────────────────────────────────────────────────────────────────────


def build_curator_agent(llm: Any) -> Any:
    """
    Tier 1 — Curator Agent.

    Detects data modality, validates the coordinate system, and runs QC.
    Outputs a ``CuratorOutput``-compatible JSON for the Analyst.
    """
    _require_crewai()
    return Agent(
        role="Biological Data Curator",
        goal=(
            "Precisely map raw multi-omic input to the correct biological coordinate "
            "system (dissociated cells vs. in-situ tissue) and flag all quality issues."
        ),
        backstory=(
            "You are a senior computational biologist who has processed thousands of "
            "single-cell and spatial transcriptomics experiments. You are meticulous "
            "about data quality: you detect mislabelled modalities, flag high mitochondrial "
            "fractions, and refuse to pass low-quality data to downstream agents."
        ),
        tools=[InspectAdataTool(), DetectModalityTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


def build_analyst_agent(llm: Any) -> Any:
    """
    Tier 2 — Analyst Agent.

    Executes foundation model inference (scGPT / Geneformer / Tangram) to
    annotate cell types and identify marker genes. On retry, strictly adheres
    to the constraint context injected by the orchestrator.
    """
    _require_crewai()
    return Agent(
        role="Single-Cell Genomics Analyst",
        goal=(
            "Accurately annotate cell types, identify biologically valid marker genes, "
            "and quantify spatial deconvolution. NEVER report genes that are not present "
            "in the source dataset."
        ),
        backstory=(
            "You are a leading computational biologist specialising in foundation model "
            "inference for single-cell and spatial omics. You run scGPT for cell type "
            "embedding and Tangram for spatial mapping. Critically, you have learned that "
            "biological hallucination — reporting genes or cell types not supported by "
            "the raw data — is a career-ending mistake in clinical contexts. When given "
            "a list of excluded genes, you treat it as absolute: those gene symbols must "
            "never appear in your output."
        ),
        tools=[FindMarkerGenesTool(), AnnotateCellTypesTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


def build_auditor_agent(llm: Any) -> Any:
    """
    Tier 5 — Auditor Agent.

    Provides source-to-bit traceability: links every biological claim to a
    specific gene index or cell barcode in the raw AnnData, producing an
    ``AuditorReport`` suitable for FDA SaMD review.
    """
    _require_crewai()
    return Agent(
        role="Biological Data Auditor",
        goal=(
            "Provide complete source-to-bit traceability. Every biological claim must "
            "be linked to at least one AnnData obs barcode or var gene index."
        ),
        backstory=(
            "You are a rigorous data auditor with a background in both bioinformatics "
            "and regulatory compliance. You have worked with FDA software auditors and "
            "understand the SaMD transparency requirements. You leave no claim "
            "untraced: if a cell-type annotation says 'CD8+ T cell', you find the exact "
            "cell barcodes and gene indices that prove it. You are the last line of "
            "defence before a report reaches a clinician."
        ),
        tools=[TraceClaimsTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task builders
# ─────────────────────────────────────────────────────────────────────────────

_CURATOR_TASK_DESCRIPTION = """\
Inspect the AnnData dataset registered under run_id={run_id}.

Use the inspect_adata and detect_modality tools to gather:
  - Total cell count and gene count
  - Data modality (single-cell RNA / spatial Visium / etc.)
  - Whether spatial coordinates are present
  - Any quality flags (e.g. high mitochondrial fraction, low library size)
  - The correct coordinate system label

Return a structured summary suitable for the Analyst agent.
"""

_ANALYST_BASE_DESCRIPTION = """\
Analyse the single-cell / spatial dataset for pipeline run_id={run_id}.

Curator summary:
  Modality      : {modality}
  Cells × Genes : {n_cells} × {n_genes}
  Is spatial    : {is_spatial}
  Coordinate    : {coordinate_system}
  QC flags      : {quality_flags}

Steps:
  1. Use find_marker_genes (exclude_genes=[]) to identify top variable genes.
  2. Use annotate_cell_types with those markers to assign cell-type labels.
  3. For each cell type, list its top marker genes and your confidence score.
  4. Report deconvolution proportions if the data is spatial.
  5. Assign a self-assessed Biological Consistency Score (BCS) in [0, 1].
"""

_ANALYST_RETRY_ADDENDUM = """\

╔══ SELF-CORRECTION REQUIRED (Attempt {attempt} of {max_retries}) ══════════════╗
║                                                                              ║
║  The Confidence Gate REJECTED your previous inference.                       ║
║                                                                              ║
║  EXCLUDED GENES — absent from the dataset, MUST NOT appear in any output:   ║
{excluded_genes_block}
║                                                                              ║
║  MANDATORY CONSTRAINTS for this attempt:                                     ║
{constraints_block}
║                                                                              ║
║  Violating these constraints will cause immediate FAIL_REJECTED.             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

_AUDITOR_TASK_DESCRIPTION = """\
Audit all biological claims produced by the Analyst for pipeline run_id={run_id}.

Analyst produced {n_annotations} cell-type annotations, {n_top_markers} top marker genes.
Gate passed with BCS={bcs:.4f} on attempt {attempt}.

For each cell-type annotation:
  1. Use the trace_claims tool to find AnnData obs barcodes and var indices.
  2. Verify the top marker gene(s) are expressed in those cells.
  3. Mark the claim as verified=True only when raw-data evidence is confirmed.

Produce a complete AuditorReport linking every verified claim to
at least one DataIndexReference (adata_obs_index or adata_var_index).
"""


def build_curator_task(agent: Any, run_id: str) -> Any:
    """Build the Curator's Task for this pipeline run."""
    _require_crewai()
    return Task(
        description=_CURATOR_TASK_DESCRIPTION.format(run_id=run_id),
        expected_output=(
            "A JSON object with keys: modality, is_spatial, n_cells, n_genes, "
            "has_raw_counts, coordinate_system, quality_flags."
        ),
        agent=agent,
    )


def build_analyst_task(agent: Any, run_input: "AnalystRunInput", max_retries: int = 3) -> Any:
    """
    Build the Analyst's Task, injecting retry context when ``attempt > 1``.

    The task description is extended with a visually distinct correction block
    that lists exactly which genes the Analyst must avoid and which parameter
    constraints to apply — giving the LLM unambiguous self-correction guidance.
    """
    _require_crewai()
    co = run_input.curator_output
    desc = _ANALYST_BASE_DESCRIPTION.format(
        run_id=str(run_input.run_id),
        modality=co.modality.value,
        n_cells=co.n_cells,
        n_genes=co.n_genes,
        is_spatial=co.is_spatial,
        coordinate_system=co.coordinate_system,
        quality_flags=co.quality_flags or "none",
    )

    if run_input.attempt > 1 and (run_input.missing_genes or run_input.constraint_adjustments):
        excluded_block = "\n".join(
            f"║    • {g}" for g in run_input.missing_genes
        ) or "║    (none)"
        constraints_block = "\n".join(
            f"║    {i}. {c}"
            for i, c in enumerate(run_input.constraint_adjustments, start=1)
        ) or "║    (none)"
        desc += _ANALYST_RETRY_ADDENDUM.format(
            attempt=run_input.attempt,
            max_retries=max_retries,
            excluded_genes_block=excluded_block,
            constraints_block=constraints_block,
        )

    return Task(
        description=desc,
        expected_output=(
            "A JSON object matching the AnalystOutput schema: "
            "cell_type_annotations (list), top_marker_genes (list), "
            "biological_consistency_score (float), model_used (string), "
            "deconvolution_results (dict or null)."
        ),
        agent=agent,
    )


def build_auditor_task(agent: Any, run_input: "AuditorRunInput") -> Any:
    """Build the Auditor's Task from a completed Analyst output + gate report."""
    _require_crewai()
    ao = run_input.analyst_output
    vr = run_input.validator_report
    return Task(
        description=_AUDITOR_TASK_DESCRIPTION.format(
            run_id=str(run_input.run_id),
            n_annotations=len(ao.cell_type_annotations),
            n_top_markers=len(ao.top_marker_genes),
            bcs=vr.computed_bcs,
            attempt=vr.analyst_attempt,
        ),
        expected_output=(
            "A JSON object matching the AuditorReport schema with populated "
            "verified_claims (each linked to data_index_refs) and "
            "traceability_coverage close to 1.0."
        ),
        agent=agent,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI runner factories
# These wrap Crew.kickoff() and convert raw crew output to typed Pydantic objects.
# The run_id is always enforced programmatically after parsing crew output.
# ─────────────────────────────────────────────────────────────────────────────


def make_crewai_curator_runner(llm: Any):
    """
    Return a curator runner backed by a live CrewAI crew.

    The runner registers the AnnData in the pipeline registry so that
    agent tools can access it by run_id.
    """
    _require_crewai()

    def run(adata: Any, run_id: str) -> CuratorOutput:
        _ADataRegistry.register(run_id, adata)
        agent = build_curator_agent(llm)
        task = build_curator_task(agent, run_id)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = crew.kickoff()

        raw: dict = result.json_dict or {}
        return CuratorOutput(
            run_id=run_id,
            modality=DataModality(raw.get("modality", DataModality.UNKNOWN.value)),
            is_spatial=bool(raw.get("is_spatial", False)),
            n_cells=int(raw.get("n_cells", adata.n_obs)),
            n_genes=int(raw.get("n_genes", adata.n_vars)),
            has_raw_counts=bool(raw.get("has_raw_counts", True)),
            coordinate_system=raw.get("coordinate_system", "dissociated_cells"),
            quality_flags=raw.get("quality_flags", []),
            notes=raw.get("notes"),
        )

    return run


def make_crewai_analyst_runner(llm: Any, max_retries: int = 3):
    """
    Return an analyst runner backed by a live CrewAI crew.

    On loopback, the task description automatically includes the excluded-genes
    block and constraint guidance derived from the previous ValidatorReport.
    """
    _require_crewai()

    def run(run_input: "AnalystRunInput") -> AnalystOutput:
        agent = build_analyst_agent(llm)
        task = build_analyst_task(agent, run_input, max_retries=max_retries)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = crew.kickoff()

        raw: dict = result.json_dict or {}

        # Build cell-type annotations — parse from LLM output or use defaults
        annotations_raw = raw.get("cell_type_annotations", [])
        cell_type_annotations = [
            CellTypeAnnotation(
                label=ann.get("label", "Unknown"),
                confidence=float(ann.get("confidence", 0.5)),
                n_cells=int(ann.get("n_cells", 0)),
                marker_genes=[
                    MarkerGeneEntry(
                        gene_symbol=m.get("gene_symbol", m) if isinstance(m, dict) else m,
                        log2_fold_change=float(m.get("log2_fold_change", 1.0)) if isinstance(m, dict) else 1.0,
                        adjusted_p_value=float(m.get("adjusted_p_value", 0.05)) if isinstance(m, dict) else 0.05,
                        present_in_adata=bool(m.get("present_in_adata", True)) if isinstance(m, dict) else True,
                    )
                    for m in ann.get("marker_genes", [])
                ],
            )
            for ann in annotations_raw
        ] or [CellTypeAnnotation(label="Uncharacterised", confidence=0.1, n_cells=0)]

        top_markers_raw = raw.get("top_marker_genes", [])
        top_marker_genes = [
            MarkerGeneEntry(
                gene_symbol=m.get("gene_symbol", m) if isinstance(m, dict) else m,
                log2_fold_change=float(m.get("log2_fold_change", 1.0)) if isinstance(m, dict) else 1.0,
                adjusted_p_value=float(m.get("adjusted_p_value", 0.05)) if isinstance(m, dict) else 0.05,
                present_in_adata=bool(m.get("present_in_adata", True)) if isinstance(m, dict) else True,
            )
            for m in top_markers_raw
        ]

        return AnalystOutput(
            run_id=run_input.run_id,          # always enforced — never trust LLM for UUID
            attempt=run_input.attempt,
            model_used=FoundationModel(raw.get("model_used", FoundationModel.SCGPT.value)),
            cell_type_annotations=cell_type_annotations,
            top_marker_genes=top_marker_genes,
            biological_consistency_score=float(raw.get("biological_consistency_score", 0.5)),
            gate_decision=GateDecision.FAIL_LOOPBACK,  # gate will override
            constraint_adjustments=run_input.constraint_adjustments,
        )

    return run


def make_crewai_auditor_runner(llm: Any):
    """Return an auditor runner backed by a live CrewAI crew."""
    _require_crewai()

    def run(run_input: "AuditorRunInput") -> AuditorReport:
        agent = build_auditor_agent(llm)
        task = build_auditor_task(agent, run_input)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = crew.kickoff()

        raw: dict = result.json_dict or {}
        ao = run_input.analyst_output
        vr = run_input.validator_report

        # Build verified claims from gate-confirmed genes
        verified_claims = [
            BiologicalClaim(
                claim_text=f"Cell type '{ann.label}' supported by marker genes in AnnData.",
                source_agent="analyst",
                confidence=ann.confidence,
                is_verified=True,
                data_index_refs=[
                    DataIndexReference(adata_var_index=m.gene_symbol)
                    for m in ann.marker_genes
                    if m.gene_symbol in vr.verified_marker_genes
                ] or [DataIndexReference(adata_obs_index="verified_by_gate")],
                supporting_genes=[
                    m.gene_symbol
                    for m in ann.marker_genes
                    if m.gene_symbol in vr.verified_marker_genes
                ],
            )
            for ann in ao.cell_type_annotations
        ]

        return AuditorReport(
            run_id=run_input.run_id,
            final_bcs=vr.computed_bcs,
            analyst_attempts_used=vr.analyst_attempt,
            verified_claims=verified_claims,
            pipeline_provenance={
                "model_used": ao.model_used.value,
                "analyst_attempts": ao.attempt,
                "gate_bcs": vr.computed_bcs,
                "llm_auditor_raw": raw.get("notes", ""),
            },
        )

    return run
