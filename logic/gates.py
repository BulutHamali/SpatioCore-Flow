"""
logic/gates.py — SpatioCore Flow v2.0 Deterministic Sandbox
============================================================
The ValidationGate is the authoritative, code-driven verifier that sits
between every Analyst inference and downstream propagation.

Key design principles
---------------------
1. **Independent authority** — The gate NEVER trusts self-reported
   ``MarkerGeneEntry.present_in_adata`` flags. It re-verifies every gene
   symbol against the raw ``AnnData.var_names``.

2. **Three-component BCS** — The Biological Consistency Score is a
   weighted composite of:
     - marker_gene_recall     (weight 0.40) — fraction of claimed genes found
     - expression_support     (weight 0.35) — fraction of found genes above
                                              a mean-expression floor
     - cluster_consistency    (weight 0.25) — per-annotation marker coverage

3. **Two-tier rejection** —
     - HARD reject  : any ``top_marker_genes`` gene absent from var_names
                      (unambiguous hallucination, no retry allowed)
     - SOFT reject  : BCS < threshold after max_retries exhausted
     - LOOPBACK     : BCS < threshold but retries remain

4. **No AnnData import at module load** — scanpy / anndata are imported
   lazily so the module can be imported without the full omics stack.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from logic.schemas import AnalystOutput, GateDecision, ValidatorReport

if TYPE_CHECKING:
    pass  # AnnData type kept as Any to avoid hard runtime dependency

logger = logging.getLogger(__name__)


class ValidationGate:
    """
    Deterministic validation sandbox for Analyst outputs.

    Parameters
    ----------
    adata :
        A pre-loaded AnnData object (takes priority over ``adata_path``).
    adata_path :
        Path to an ``.h5ad`` file; loaded via ``scanpy.read_h5ad``.
        Mutually exclusive with ``adata``.
    bcs_threshold :
        Minimum BCS required to pass the gate (default 0.80).
    max_retries :
        Maximum Analyst attempts before soft-failing becomes FAIL_REJECTED.
    expression_floor :
        Genes whose mean expression is ≤ this value are treated as
        "unexpressed" for the ``expression_support`` BCS component.
        Assumes log1p-normalised counts (typical scanpy pipeline output).

    Examples
    --------
    >>> gate = ValidationGate(adata=my_adata)
    >>> report = gate.verify_biomarkers(analyst_output)
    >>> print(report.gate_decision, report.computed_bcs)
    """

    # ── Default hyper-parameters ──────────────────────────────────────────────
    BCS_PASS_THRESHOLD: float = 0.80
    MAX_RETRIES: int = 3
    EXPRESSION_FLOOR: float = 0.05  # log1p-normalised units

    # BCS component weights — must sum to 1.0
    W_MARKER_RECALL: float = 0.40
    W_EXPRESSION_SUPPORT: float = 0.35
    W_CLUSTER_CONSISTENCY: float = 0.25

    def __init__(
        self,
        adata: Any | None = None,
        adata_path: str | Path | None = None,
        bcs_threshold: float = BCS_PASS_THRESHOLD,
        max_retries: int = MAX_RETRIES,
        expression_floor: float = EXPRESSION_FLOOR,
    ) -> None:
        if adata is None and adata_path is None:
            raise ValueError("Provide either 'adata' or 'adata_path'.")
        if adata is not None and adata_path is not None:
            raise ValueError("Provide only one of 'adata' or 'adata_path', not both.")

        self.bcs_threshold = bcs_threshold
        self.max_retries = max_retries
        self.expression_floor = expression_floor

        self._adata: Any = self._load_adata(adata_path) if adata_path is not None else adata

        # Pre-built set for O(1) gene lookups — rebuilt only on explicit reload
        self._var_set: set[str] = set(self._adata.var_names)
        # Column-index map for fast expression retrieval
        self._var_index_map: dict[str, int] = {
            gene: idx for idx, gene in enumerate(self._adata.var_names)
        }

    # ── Data loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_adata(path: str | Path) -> Any:
        """Load AnnData from an .h5ad file via Scanpy (lazy import)."""
        try:
            import scanpy as sc
        except ImportError as exc:
            raise ImportError(
                "scanpy is required to load .h5ad files. "
                "Install it with: pip install scanpy"
            ) from exc
        adata = sc.read_h5ad(path)
        logger.info("Loaded AnnData from %s: %d cells × %d genes", path, adata.n_obs, adata.n_vars)
        return adata

    # ── Expression matrix helpers ─────────────────────────────────────────────

    def _dense_X(self) -> np.ndarray:
        """Return a dense (cells × genes) float32 expression matrix."""
        try:
            import scipy.sparse as sp
            if sp.issparse(self._adata.X):
                return np.asarray(self._adata.X.toarray(), dtype=np.float32)
        except ImportError:
            pass
        return np.asarray(self._adata.X, dtype=np.float32)

    def _gene_mean_expression(self, genes: list[str]) -> dict[str, float]:
        """
        Compute per-gene mean expression for genes present in var_names.

        Genes absent from var_names are silently omitted from the result.
        """
        if not genes:
            return {}

        X = self._dense_X()
        result: dict[str, float] = {}
        for gene in genes:
            col = self._var_index_map.get(gene)
            if col is not None:
                result[gene] = float(X[:, col].mean())
        return result

    # ── BCS component calculators ─────────────────────────────────────────────

    def _collect_all_marker_genes(self, analyst_output: AnalystOutput) -> list[str]:
        """
        Deduplicated union of:
          - ``top_marker_genes`` (global DE genes)
          - per-annotation ``marker_genes`` from all ``cell_type_annotations``
        """
        genes: set[str] = set()
        for m in analyst_output.top_marker_genes:
            genes.add(m.gene_symbol)
        for ann in analyst_output.cell_type_annotations:
            for m in ann.marker_genes:
                genes.add(m.gene_symbol)
        return list(genes)

    def _compute_marker_recall(
        self, all_genes: list[str]
    ) -> tuple[float, list[str], list[str]]:
        """
        Check each gene symbol against ``adata.var_names``.

        Returns
        -------
        recall :
            |found| / |total|. 1.0 when no genes are claimed (vacuously true).
        verified :
            Gene symbols confirmed in var_names.
        rejected :
            Gene symbols absent from var_names — evidence of hallucination.
        """
        if not all_genes:
            return 1.0, [], []

        verified = [g for g in all_genes if g in self._var_set]
        rejected = [g for g in all_genes if g not in self._var_set]
        recall = len(verified) / len(all_genes)
        return recall, verified, rejected

    def _compute_expression_support(self, verified_genes: list[str]) -> float:
        """
        Fraction of verified genes with mean expression > ``expression_floor``.

        Returns 1.0 when no verified genes exist (no penalty for empty slate).
        """
        if not verified_genes:
            return 1.0

        mean_expr = self._gene_mean_expression(verified_genes)
        expressed_count = sum(
            1 for g in verified_genes if mean_expr.get(g, 0.0) > self.expression_floor
        )
        return expressed_count / len(verified_genes)

    def _compute_cluster_consistency(self, analyst_output: AnalystOutput) -> float:
        """
        Per cell-type annotation: fraction of its marker genes found in var_names.

        Returns the mean across all annotations that declare ≥1 marker gene.
        Returns 1.0 when no annotation has marker genes.
        """
        per_annotation_scores: list[float] = []
        for ann in analyst_output.cell_type_annotations:
            if not ann.marker_genes:
                continue
            found = sum(1 for m in ann.marker_genes if m.gene_symbol in self._var_set)
            per_annotation_scores.append(found / len(ann.marker_genes))

        if not per_annotation_scores:
            return 1.0
        return float(np.mean(per_annotation_scores))

    # ── BCS aggregation ───────────────────────────────────────────────────────

    def compute_bcs(
        self, analyst_output: AnalystOutput
    ) -> tuple[float, dict[str, float], list[str], list[str]]:
        """
        Compute the full Biological Consistency Score for an AnalystOutput.

        Returns
        -------
        bcs : float
            Overall BCS in [0, 1], clamped after weighted summation.
        components : dict[str, float]
            Sub-scores keyed by component name (values in [0, 1]).
        verified_genes : list[str]
            All marker genes confirmed in AnnData var_names.
        rejected_genes : list[str]
            All marker genes NOT found — includes both top and annotation markers.
        """
        all_genes = self._collect_all_marker_genes(analyst_output)

        marker_recall, verified_genes, rejected_genes = self._compute_marker_recall(all_genes)
        expression_support = self._compute_expression_support(verified_genes)
        cluster_consistency = self._compute_cluster_consistency(analyst_output)

        bcs = (
            self.W_MARKER_RECALL * marker_recall
            + self.W_EXPRESSION_SUPPORT * expression_support
            + self.W_CLUSTER_CONSISTENCY * cluster_consistency
        )
        bcs = float(np.clip(bcs, 0.0, 1.0))

        components: dict[str, float] = {
            "marker_gene_recall": round(marker_recall, 6),
            "expression_support": round(expression_support, 6),
            "cluster_consistency": round(cluster_consistency, 6),
        }

        return bcs, components, verified_genes, rejected_genes

    # ── Gate decision logic ───────────────────────────────────────────────────

    def _find_hallucinated_top_markers(self, analyst_output: AnalystOutput) -> list[str]:
        """
        Return ``top_marker_genes`` symbols absent from var_names.

        These are the hard-evidence hallucinations that trigger immediate
        FAIL_REJECTED regardless of BCS value.
        """
        return [
            m.gene_symbol
            for m in analyst_output.top_marker_genes
            if m.gene_symbol not in self._var_set
        ]

    def _decide_gate(
        self,
        bcs: float,
        hallucinated_top_markers: list[str],
        analyst_attempt: int,
    ) -> tuple[GateDecision, list[str], list[str]]:
        """
        Apply the two-tier gate logic and return a verdict with supporting text.

        Returns
        -------
        decision : GateDecision
        failure_reasons : list[str]
        suggested_constraints : list[str]
        """
        failure_reasons: list[str] = []
        suggestions: list[str] = []

        # ── Tier 1: hard rejection for hallucinated top markers ───────────────
        if hallucinated_top_markers:
            failure_reasons.append(
                f"HALLUCINATION DETECTED — top_marker_genes not present in "
                f"AnnData.var_names: {hallucinated_top_markers}. "
                "No loopback allowed; inference is definitively invalid."
            )
            suggestions.append(
                "Restrict model output to gene symbols present in the "
                "source dataset's var_names index before emitting AnalystOutput."
            )
            return GateDecision.FAIL_REJECTED, failure_reasons, suggestions

        # ── Tier 2: BCS threshold gate ────────────────────────────────────────
        if bcs >= self.bcs_threshold:
            return GateDecision.PASS, [], []

        # BCS below threshold — accumulate failure context
        failure_reasons.append(
            f"BCS {bcs:.4f} is below the required threshold of {self.bcs_threshold:.4f}."
        )

        if analyst_attempt >= self.max_retries:
            failure_reasons.append(
                f"Maximum retries ({self.max_retries}) exhausted. "
                "Inference permanently rejected."
            )
            return GateDecision.FAIL_REJECTED, failure_reasons, suggestions

        # Suggest parameter adjustments for the next Analyst attempt
        suggestions.extend(
            [
                "Increase min_cluster_size to improve cluster marker stability.",
                "Restrict embedding space to top-2000 highly variable genes (HVGs).",
                "Increase n_neighbors for more robust KNN graph construction.",
                "Re-run with a higher min_dist UMAP parameter for cleaner separation.",
            ]
        )
        return GateDecision.FAIL_LOOPBACK, failure_reasons, suggestions

    # ── Public API ────────────────────────────────────────────────────────────

    def verify_biomarkers(
        self,
        analyst_output: AnalystOutput,
        analyst_attempt: int = 1,
        sandbox_exec_time_ms: float | None = None,
    ) -> ValidatorReport:
        """
        Verify all marker gene claims in *analyst_output* against the AnnData.

        This is the primary entry point for the Confidence Gate. Call it once
        per Analyst attempt and use the returned :class:`ValidatorReport` to
        decide whether to propagate or loop back.

        Parameters
        ----------
        analyst_output :
            Structured output from the Analyst agent (Tier 2).
        analyst_attempt :
            1-based attempt counter — incremented by the orchestrator on each
            loopback. Once ``analyst_attempt >= max_retries``, a failing BCS
            becomes FAIL_REJECTED instead of FAIL_LOOPBACK.
        sandbox_exec_time_ms :
            If the caller wraps this in an outer CitL sandbox and wants to
            record that timing, pass it here. Otherwise the gate's own
            wall-clock time is used.

        Returns
        -------
        ValidatorReport
            Fully structured gate verdict ready for the orchestrator.
        """
        t0 = time.perf_counter()

        # Step 1 — compute BCS and gene verification
        bcs, components, verified_genes, rejected_genes = self.compute_bcs(analyst_output)

        # Step 2 — identify hard-reject candidates from top_marker_genes only
        hallucinated_top_markers = self._find_hallucinated_top_markers(analyst_output)

        # Step 3 — apply gate logic
        decision, failure_reasons, suggestions = self._decide_gate(
            bcs=bcs,
            hallucinated_top_markers=hallucinated_top_markers,
            analyst_attempt=analyst_attempt,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1_000

        logger.info(
            "ValidationGate | attempt=%d bcs=%.4f decision=%s "
            "verified=%d rejected=%d elapsed=%.1fms",
            analyst_attempt,
            bcs,
            decision.value,
            len(verified_genes),
            len(rejected_genes),
            elapsed_ms,
        )

        return ValidatorReport(
            run_id=analyst_output.run_id,
            analyst_attempt=analyst_attempt,
            computed_bcs=round(bcs, 6),
            bcs_threshold=self.bcs_threshold,
            verified_marker_genes=verified_genes,
            rejected_marker_genes=rejected_genes,
            cluster_label_consistency=round(components["cluster_consistency"], 6),
            gate_decision=decision,
            failure_reasons=failure_reasons,
            suggested_constraints=suggestions,
            sandbox_exec_time_ms=round(
                sandbox_exec_time_ms if sandbox_exec_time_ms is not None else elapsed_ms,
                3,
            ),
        )

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n_cells(self) -> int:
        return int(self._adata.n_obs)

    @property
    def n_genes(self) -> int:
        return int(self._adata.n_vars)

    def __repr__(self) -> str:
        return (
            f"ValidationGate("
            f"n_cells={self.n_cells}, "
            f"n_genes={self.n_genes}, "
            f"bcs_threshold={self.bcs_threshold}, "
            f"max_retries={self.max_retries})"
        )
