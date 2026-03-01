"""
logic/schemas.py — SpatioCore Flow v2.0 Pydantic Schemas
=========================================================
Single source of truth for all structured inter-agent communication.

Every agent in the pipeline MUST emit one of the validated models defined here.
No raw dicts are passed between nodes — only schema-validated objects.

Pipeline order (Tier → Schema):
  1. Curator      → CuratorOutput
  2. Analyst      → AnalystOutput          ← PRIMARY FOCUS
  3. Validator    → ValidatorReport
  4. Synthesizer  → SynthesizerOutput
  5. Auditor      → AuditorReport          ← PRIMARY FOCUS
  6. Guardrail    → GuardrailReport
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Shared enumerations
# ─────────────────────────────────────────────────────────────────────────────


class DataModality(str, Enum):
    """Biological data modality detected by the Curator."""

    SINGLE_CELL_RNA = "single_cell_rna"
    SINGLE_CELL_ATAC = "single_cell_atac"
    SPATIAL_VISIUM = "spatial_visium"
    SPATIAL_MERFISH = "spatial_merfish"
    SPATIAL_SLIDESEQ = "spatial_slideseq"
    MULTIOME = "multiome"
    UNKNOWN = "unknown"


class FoundationModel(str, Enum):
    """Foundation models available to the Analyst."""

    SCGPT = "scgpt"
    GENEFORMER = "geneformer"
    TANGRAM = "tangram"
    CELL2LOCATION = "cell2location"
    STARDIST = "stardist"
    VIT_IMAGING = "vit_imaging"
    NONE = "none"  # deterministic-only run


class GateDecision(str, Enum):
    """Outcome of the Confidence Gate after BCS evaluation."""

    PASS = "pass"
    FAIL_LOOPBACK = "fail_loopback"
    FAIL_REJECTED = "fail_rejected"  # exceeded max retries


class RiskLevel(str, Enum):
    """Clinical / SaMD risk classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Shared sub-models (used across multiple agent outputs)
# ─────────────────────────────────────────────────────────────────────────────


class MarkerGeneEntry(BaseModel):
    """A single validated marker gene with supporting evidence."""

    gene_symbol: str = Field(..., description="HGNC-approved gene symbol, e.g. 'CD8A'.")
    ensemble_id: str | None = Field(None, description="Ensembl gene ID, e.g. 'ENSG00000153563'.")
    log2_fold_change: float = Field(..., description="Log2 fold-change relative to background.")
    adjusted_p_value: float = Field(..., ge=0.0, le=1.0, description="Benjamini-Hochberg adjusted p-value.")
    present_in_adata: bool = Field(
        ...,
        description="Whether the gene was confirmed to exist in the source AnnData var index.",
    )

    @field_validator("gene_symbol")
    @classmethod
    def symbol_must_be_uppercase(cls, v: str) -> str:
        return v.upper().strip()


class CellTypeAnnotation(BaseModel):
    """Predicted cell-type label with confidence."""

    label: str = Field(..., description="Cell-type label, e.g. 'CD8+ Cytotoxic T cell'.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in [0, 1].")
    marker_genes: list[MarkerGeneEntry] = Field(
        default_factory=list,
        description="Top marker genes supporting this annotation.",
    )
    n_cells: int = Field(..., ge=0, description="Number of cells assigned to this label.")


class SpatialSpotReference(BaseModel):
    """Links a biological claim to a specific spatial location."""

    spot_barcode: str = Field(..., description="Visium/slide barcode or MERFISH cell ID.")
    array_row: int = Field(..., ge=0, description="Array row coordinate.")
    array_col: int = Field(..., ge=0, description="Array column coordinate.")
    pixel_x: float | None = Field(None, description="Pixel x-coordinate in full-res image.")
    pixel_y: float | None = Field(None, description="Pixel y-coordinate in full-res image.")
    tissue_section: str | None = Field(None, description="Tissue section / slide identifier.")


class DataIndexReference(BaseModel):
    """Traceability link from an agent claim to the raw AnnData index."""

    adata_obs_index: str | None = Field(None, description="AnnData .obs row label (cell barcode).")
    adata_var_index: str | None = Field(None, description="AnnData .var row label (gene name).")
    layer_key: str | None = Field(None, description="AnnData .layers key used for computation.")
    obsm_key: str | None = Field(None, description="AnnData .obsm key (e.g. 'X_umap').")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Curator Output
# ─────────────────────────────────────────────────────────────────────────────


class CuratorOutput(BaseModel):
    """
    Output of the Curator agent (Tier 1).

    Detects the data modality and maps raw input to the correct biological
    coordinate system before any downstream analysis.
    """

    run_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this pipeline run.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of Curator completion.",
    )
    modality: DataModality = Field(..., description="Detected biological data modality.")
    is_spatial: bool = Field(..., description="True when spatial coordinate metadata is present.")
    n_cells: int = Field(..., ge=0, description="Total cells / spots detected.")
    n_genes: int = Field(..., ge=0, description="Total features (genes) in the dataset.")
    has_raw_counts: bool = Field(
        ..., description="True when integer raw count matrix is available in AnnData."
    )
    quality_flags: list[str] = Field(
        default_factory=list,
        description="QC warnings, e.g. 'high_mito_fraction', 'low_library_size'.",
    )
    coordinate_system: str = Field(
        ...,
        description="Coordinate system label: 'dissociated_cells' | 'in_situ_tissue'.",
    )
    notes: str | None = Field(None, description="Free-text Curator reasoning / rationale.")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Analyst Output  ★ PRIMARY
# ─────────────────────────────────────────────────────────────────────────────


class AnalystOutput(BaseModel):
    """
    Output of the Analyst agent (Tier 2).

    The Analyst executes foundation models (scGPT, Geneformer, Tangram,
    cell2location) and returns structured biological inferences.

    All numeric claims are accompanied by a Biological Consistency Score (BCS)
    which the downstream Confidence Gate uses to decide pass / loopback / reject.

    Constraints
    -----------
    - ``biological_consistency_score`` must be in [0, 1].
    - ``gate_decision`` is set to ``FAIL_LOOPBACK`` automatically when BCS < threshold.
    - ``marker_genes`` entries with ``present_in_adata=False`` are hard evidence of
      hallucination and will force a ``FAIL_REJECTED`` gate decision.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    run_id: UUID = Field(..., description="Must match the CuratorOutput.run_id for this run.")
    attempt: int = Field(
        default=1,
        ge=1,
        description="Analyst attempt counter (incremented on each Confidence Gate loopback).",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of this Analyst attempt.",
    )

    # ── Model provenance ──────────────────────────────────────────────────────
    model_used: FoundationModel = Field(
        ..., description="Primary foundation model invoked for this inference."
    )
    model_version: str | None = Field(None, description="Model checkpoint or version tag.")
    secondary_models: list[FoundationModel] = Field(
        default_factory=list,
        description="Any additional models used (e.g. Tangram for deconvolution).",
    )

    # ── Biological inferences ─────────────────────────────────────────────────
    cell_type_annotations: list[CellTypeAnnotation] = Field(
        ...,
        min_length=1,
        description="Predicted cell-type labels with per-type confidence and marker genes.",
    )
    top_marker_genes: list[MarkerGeneEntry] = Field(
        default_factory=list,
        description="Global top differentially expressed / marker genes across all clusters.",
    )
    deconvolution_results: dict[str, float] | None = Field(
        None,
        description=(
            "Spot-level cell-type proportion estimates from Tangram/cell2location. "
            "Keys are cell-type labels; values are proportions summing to ~1.0."
        ),
    )
    pathway_enrichment: list[str] | None = Field(
        None,
        description="Top enriched biological pathways (gene ontology or KEGG terms).",
    )

    # ── Confidence Gate inputs ────────────────────────────────────────────────
    biological_consistency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Biological Consistency Score (BCS) in [0, 1]. "
            "Computed by cross-referencing model outputs against raw AnnData. "
            "BCS < 0.80 triggers a Confidence Gate FAIL_LOOPBACK."
        ),
    )
    bcs_components: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Decomposed BCS sub-scores, e.g. "
            "{'marker_gene_recall': 0.92, 'cluster_stability': 0.85, 'deconv_residual': 0.78}."
        ),
    )
    gate_decision: GateDecision = Field(
        default=GateDecision.FAIL_LOOPBACK,
        description="Confidence Gate verdict — set by the Validator, not the Analyst itself.",
    )
    constraint_adjustments: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable list of constraints applied during this attempt, "
            "e.g. 'increased min_cluster_size', 'restricted to top-2000 HVGs'."
        ),
    )

    # ── Raw data traceability ─────────────────────────────────────────────────
    data_references: list[DataIndexReference] = Field(
        default_factory=list,
        description="Links from inferences back to specific AnnData indices.",
    )

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("deconvolution_results")
    @classmethod
    def deconvolution_proportions_sum_to_one(
        cls, v: dict[str, float] | None
    ) -> dict[str, float] | None:
        if v is None:
            return v
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Deconvolution proportions must sum to ~1.0, got {total:.4f}. "
                "Normalise before emitting AnalystOutput."
            )
        return v

    @model_validator(mode="after")
    def hallucinated_markers_force_reject(self) -> AnalystOutput:
        """Any marker gene absent from AnnData is definitive hallucination evidence."""
        hallucinated = [
            m.gene_symbol
            for m in self.top_marker_genes
            if not m.present_in_adata
        ]
        if hallucinated:
            self.gate_decision = GateDecision.FAIL_REJECTED
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Validator Report
# ─────────────────────────────────────────────────────────────────────────────


class ValidatorReport(BaseModel):
    """
    Output of the Validator agent (Tier 3).

    The Validator runs Code-in-the-Loop (CitL) checks — programmatic verification
    of every Analyst claim against the raw AnnData/Squidpy objects.
    """

    run_id: UUID = Field(..., description="Must match the run_id for this pipeline execution.")
    analyst_attempt: int = Field(..., ge=1, description="Which Analyst attempt this validates.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── BCS computation ───────────────────────────────────────────────────────
    computed_bcs: float = Field(
        ..., ge=0.0, le=1.0, description="BCS value computed by this Validator run."
    )
    bcs_threshold: float = Field(
        default=0.80, ge=0.0, le=1.0, description="Minimum BCS required to pass the gate."
    )

    # ── Claim-level verdicts ──────────────────────────────────────────────────
    verified_marker_genes: list[str] = Field(
        default_factory=list,
        description="Marker gene symbols confirmed present and significant in AnnData.",
    )
    rejected_marker_genes: list[str] = Field(
        default_factory=list,
        description="Marker gene symbols NOT found in AnnData — evidence of hallucination.",
    )
    cluster_label_consistency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of predicted cluster labels that are internally consistent.",
    )

    # ── Gate decision ─────────────────────────────────────────────────────────
    gate_decision: GateDecision = Field(
        ..., description="Final gate outcome after BCS and claim verification."
    )
    failure_reasons: list[str] = Field(
        default_factory=list,
        description="Detailed reasons for FAIL_LOOPBACK or FAIL_REJECTED decisions.",
    )
    suggested_constraints: list[str] = Field(
        default_factory=list,
        description="Recommended Analyst parameter adjustments for the next attempt.",
    )

    # ── Sandbox execution metadata ────────────────────────────────────────────
    sandbox_exec_time_ms: float | None = Field(
        None, ge=0.0, description="Wall-clock time for CitL sandbox execution in milliseconds."
    )
    sandbox_stdout: str | None = Field(
        None, description="Captured stdout from the CitL sandbox (truncated to 4096 chars)."
    )

    @model_validator(mode="after")
    def gate_consistent_with_bcs(self) -> ValidatorReport:
        if self.computed_bcs >= self.bcs_threshold and self.gate_decision == GateDecision.FAIL_LOOPBACK:
            raise ValueError(
                f"BCS {self.computed_bcs} >= threshold {self.bcs_threshold} "
                "but gate_decision is FAIL_LOOPBACK — inconsistent."
            )
        return self

    # ── Orchestrator convenience ──────────────────────────────────────────────

    @property
    def is_passed(self) -> bool:
        """True when the Confidence Gate approved this Analyst attempt."""
        return self.gate_decision == GateDecision.PASS

    @property
    def missing_genes(self) -> list[str]:
        """
        Alias for ``rejected_marker_genes``.

        Surfaced as ``missing_genes`` so the orchestrator retry loop can use
        a domain-language name when building the Analyst self-correction prompt.
        """
        return self.rejected_marker_genes


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4 — Synthesizer Output
# ─────────────────────────────────────────────────────────────────────────────


class SpatialGraphNode(BaseModel):
    """A node in the Tumor Microenvironment (TME) spatial graph."""

    node_id: str = Field(..., description="Unique node identifier (e.g. spot barcode).")
    cell_type: str = Field(..., description="Dominant cell-type label at this node.")
    proportion: float = Field(..., ge=0.0, le=1.0, description="Dominant cell-type proportion.")
    spatial_ref: SpatialSpotReference


class SpatialGraphEdge(BaseModel):
    """Directed edge encoding a spatial or ligand-receptor relationship."""

    source_node_id: str
    target_node_id: str
    edge_type: str = Field(..., description="E.g. 'proximity', 'ligand_receptor', 'co_expression'.")
    weight: float = Field(..., ge=0.0, description="Edge weight / interaction strength.")


class SynthesizerOutput(BaseModel):
    """
    Output of the Synthesizer agent (Tier 4).

    Merges single-cell 'What' (RNA identity) with spatial 'Where' data to
    reconstruct the Tumor Microenvironment and predict spatial drug efficacy.
    """

    run_id: UUID = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Spatial graph ─────────────────────────────────────────────────────────
    tme_graph_nodes: list[SpatialGraphNode] = Field(
        default_factory=list, description="Nodes of the reconstructed TME spatial graph."
    )
    tme_graph_edges: list[SpatialGraphEdge] = Field(
        default_factory=list, description="Edges encoding spatial / signalling relationships."
    )

    # ── Drug penetration prediction ───────────────────────────────────────────
    drug_penetration_map: dict[str, float] | None = Field(
        None,
        description=(
            "Predicted drug penetration score per spot/node. "
            "Keys are spot barcodes; values are penetration scores in [0, 1]."
        ),
    )
    spatial_efficacy_score: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Aggregate predicted therapeutic spatial efficacy.",
    )

    # ── Integration summary ───────────────────────────────────────────────────
    integration_method: str = Field(
        ..., description="Integration method used, e.g. 'tangram' | 'cell2location'."
    )
    integration_quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Quality metric for the SC→spatial mapping."
    )
    narrative_summary: str | None = Field(
        None, description="LLM-generated synthesis narrative for the report."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 5 — Auditor Report  ★ PRIMARY
# ─────────────────────────────────────────────────────────────────────────────


class BiologicalClaim(BaseModel):
    """
    A single verifiable biological claim emitted by any upstream agent.

    Every claim MUST be traceable to at least one raw data index or
    spatial coordinate — otherwise it is flagged as unverified.
    """

    claim_id: UUID = Field(default_factory=uuid4, description="Unique claim identifier.")
    claim_text: str = Field(
        ...,
        description="Human-readable assertion, e.g. 'Spot X is dominated by CD8+ T cells'.",
    )
    source_agent: str = Field(
        ...,
        description="Agent tier that produced this claim, e.g. 'analyst' | 'synthesizer'.",
    )

    # ── Evidence links ─────────────────────────────────────────────────────────
    data_index_refs: list[DataIndexReference] = Field(
        default_factory=list,
        description="AnnData indices (obs/var) that directly support this claim.",
    )
    spatial_refs: list[SpatialSpotReference] = Field(
        default_factory=list,
        description="Spatial coordinates / barcodes that support this claim.",
    )
    supporting_genes: list[str] = Field(
        default_factory=list,
        description="Gene symbols that are direct evidence for this claim.",
    )

    # ── Verification status ───────────────────────────────────────────────────
    is_verified: bool = Field(
        default=False,
        description="True when the Auditor has confirmed evidence in raw data.",
    )
    verification_method: str | None = Field(
        None,
        description="Method used for verification, e.g. 'gene_index_lookup', 'bcs_check'.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Auditor confidence in this claim's validity."
    )

    @model_validator(mode="after")
    def verified_claims_need_evidence(self) -> BiologicalClaim:
        """Verified claims must have at least one evidence link."""
        if self.is_verified and not self.data_index_refs and not self.spatial_refs:
            raise ValueError(
                f"Claim '{self.claim_id}' marked is_verified=True but has no data_index_refs "
                "or spatial_refs. Provide at least one traceability link."
            )
        return self


class AuditorReport(BaseModel):
    """
    Output of the Auditor agent (Tier 5).

    Provides **source-to-bit traceability** — every biological claim produced
    by the pipeline is linked to a specific gene index, cell barcode, or
    pixel coordinate in the raw AnnData / spatial image.

    This report is the primary artifact ingested by:
      - PostgreSQL audit trail (durable storage)
      - Report generator (PDF / HTML)
      - The Guardrail agent for SaMD compliance review

    Traceability contract
    ---------------------
    - ``total_claims`` == len(verified_claims) + len(unverified_claims)
    - ``traceability_coverage`` == len(verified_claims) / total_claims
    - All ``verified_claims`` must have at least one DataIndexReference or
      SpatialSpotReference (enforced by BiologicalClaim validators).
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    audit_id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for this audit report."
    )
    run_id: UUID = Field(..., description="Pipeline run this audit covers.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the Auditor completed its review.",
    )
    auditor_version: str = Field(
        default="2.0.0", description="SpatioCore Flow Auditor version string."
    )

    # ── Claim registry ────────────────────────────────────────────────────────
    verified_claims: list[BiologicalClaim] = Field(
        default_factory=list,
        description="Claims confirmed with direct raw-data evidence.",
    )
    unverified_claims: list[BiologicalClaim] = Field(
        default_factory=list,
        description="Claims that could NOT be linked to raw data — require investigator review.",
    )

    # ── Traceability metrics ──────────────────────────────────────────────────
    total_claims: int = Field(
        default=0, ge=0, description="Total claims reviewed (verified + unverified)."
    )
    traceability_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of claims with confirmed raw-data evidence.",
    )

    # ── BCS audit ────────────────────────────────────────────────────────────
    final_bcs: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Biological Consistency Score from the final successful Analyst attempt.",
    )
    analyst_attempts_used: int = Field(
        ..., ge=1, description="Number of Analyst attempts before Confidence Gate passed."
    )

    # ── Spatial deep links ────────────────────────────────────────────────────
    spatial_evidence_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Maps each claim_id (str UUID) to a list of spot barcodes that generated it. "
            "Enables 'click-to-highlight' deep linking in the Streamlit UI."
        ),
    )

    # ── Provenance chain ──────────────────────────────────────────────────────
    pipeline_provenance: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Full provenance chain: input file hash, all agent run_ids, model versions, "
            "and environment snapshot (package versions, timestamp). "
            "Keys: 'input_sha256', 'curator_run_id', 'analyst_run_id', 'models_used', etc."
        ),
    )

    # ── Computed fields (auto-populated) ─────────────────────────────────────
    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def compute_traceability_metrics(self) -> AuditorReport:
        """Auto-compute total_claims and traceability_coverage from claim lists."""
        n_verified = len(self.verified_claims)
        n_unverified = len(self.unverified_claims)
        self.total_claims = n_verified + n_unverified
        self.traceability_coverage = (
            n_verified / self.total_claims if self.total_claims > 0 else 0.0
        )
        return self

    @model_validator(mode="after")
    def verified_claims_must_be_flagged(self) -> AuditorReport:
        """All claims in verified_claims must have is_verified=True."""
        bad = [str(c.claim_id) for c in self.verified_claims if not c.is_verified]
        if bad:
            raise ValueError(
                f"Claims in verified_claims must have is_verified=True. "
                f"Offending claim_ids: {bad}"
            )
        return self

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def has_unverified_claims(self) -> bool:
        return len(self.unverified_claims) > 0

    @property
    def is_fully_traced(self) -> bool:
        """True only when every single claim is backed by raw data."""
        return self.traceability_coverage == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tier 6 — Guardrail Report
# ─────────────────────────────────────────────────────────────────────────────


class RegulatoryFlag(BaseModel):
    """A single regulatory / clinical safety concern raised by the Guardrail."""

    flag_id: UUID = Field(default_factory=uuid4)
    flag_code: str = Field(
        ..., description="Short machine-readable code, e.g. 'FDA_SAMD_HIGH_RISK'."
    )
    description: str = Field(..., description="Human-readable description of the concern.")
    severity: RiskLevel
    claim_ref: UUID | None = Field(
        None,
        description="Optional link to the BiologicalClaim that triggered this flag.",
    )
    recommended_action: str | None = Field(
        None,
        description="Suggested remediation, e.g. 'Exclude from clinical report pending validation'.",
    )


class GuardrailReport(BaseModel):
    """
    Output of the Guardrail agent (Tier 6).

    Evaluates the AuditorReport against FDA SaMD risk frameworks
    and clinical literature before final report generation.
    """

    run_id: UUID = Field(...)
    audit_ref: UUID = Field(..., description="AuditorReport.audit_id this guardrail reviewed.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Risk assessment ───────────────────────────────────────────────────────
    overall_risk: RiskLevel = Field(
        ..., description="Aggregate risk level for the full pipeline output."
    )
    regulatory_flags: list[RegulatoryFlag] = Field(
        default_factory=list,
        description="All regulatory / safety flags raised during review.",
    )

    # ── Compliance status ─────────────────────────────────────────────────────
    samd_compliant: bool = Field(
        ...,
        description=(
            "True if the output meets SpatioCore Flow's internal FDA SaMD transparency "
            "criteria (full traceability + BCS >= threshold + no CRITICAL flags)."
        ),
    )
    compliance_notes: list[str] = Field(
        default_factory=list,
        description="Narrative compliance assessment items.",
    )

    # ── Output disposition ────────────────────────────────────────────────────
    approved_for_report: bool = Field(
        ...,
        description="Final gate: True allows report generation; False blocks output.",
    )
    blocking_reasons: list[str] = Field(
        default_factory=list,
        description="Reasons the output was blocked (populated when approved_for_report=False).",
    )

    @model_validator(mode="after")
    def blocked_reports_need_reasons(self) -> GuardrailReport:
        if not self.approved_for_report and not self.blocking_reasons:
            raise ValueError(
                "approved_for_report=False requires at least one entry in blocking_reasons."
            )
        return self
