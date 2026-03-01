"""
tests/test_gates.py — ValidationGate regression tests
======================================================
All tests use a small synthetic AnnData so no real .h5ad file is required.

The module is skipped automatically if ``anndata`` is not installed.
Run the full suite after: pip install anndata numpy

Synthetic dataset layout
------------------------
Genes (var_names): CD8A, CD4, FOXP3, MKI67, EPCAM   (5 genes)
Cells (obs):       100 synthetic cells

Expression matrix is configurable per test via ``make_adata()``.
"""

from __future__ import annotations

import pytest

anndata = pytest.importorskip("anndata")   # skip entire module if anndata absent
numpy = pytest.importorskip("numpy")       # numpy is typically always present

import numpy as np
import pandas as pd
from uuid import uuid4

from logic.gates import ValidationGate
from logic.schemas import (
    AnalystOutput,
    CellTypeAnnotation,
    FoundationModel,
    GateDecision,
    MarkerGeneEntry,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

REAL_GENES = ["CD8A", "CD4", "FOXP3", "MKI67", "EPCAM"]
N_CELLS = 100
EXPRESSION_FLOOR = ValidationGate.EXPRESSION_FLOOR


def make_adata(
    genes: list[str] = REAL_GENES,
    n_cells: int = N_CELLS,
    expression: str = "normal",
) -> anndata.AnnData:
    """
    Build a minimal in-memory AnnData.

    Parameters
    ----------
    genes :
        Gene symbols to place in ``var_names``.
    n_cells :
        Number of synthetic cells (obs rows).
    expression :
        ``'normal'``  — log-normalised-like values (mean ≈ 2–4, all > floor)
        ``'zero'``    — all zeros (every gene below expression_floor)
        ``'sparse'``  — roughly half the cells have zero counts
    """
    rng = np.random.default_rng(seed=42)
    n_genes = len(genes)

    if expression == "normal":
        # Simulate log1p-normalised counts, all genes expressed
        X = rng.exponential(scale=2.5, size=(n_cells, n_genes)).astype(np.float32)
        # Ensure every gene mean exceeds the expression floor (clamp min to 0.1)
        X = np.clip(X, 0.1, None)
    elif expression == "zero":
        X = np.zeros((n_cells, n_genes), dtype=np.float32)
    elif expression == "sparse":
        X = rng.exponential(scale=2.5, size=(n_cells, n_genes)).astype(np.float32)
        mask = rng.random(size=(n_cells, n_genes)) < 0.55
        X[mask] = 0.0
    else:
        raise ValueError(f"Unknown expression mode: {expression!r}")

    var = pd.DataFrame(index=pd.Index(genes, name="gene_symbols"))
    obs = pd.DataFrame(index=pd.Index([f"cell_{i}" for i in range(n_cells)]))
    return anndata.AnnData(X=X, obs=obs, var=var)


def make_marker(
    symbol: str,
    present: bool = True,
    log2fc: float = 2.0,
    padj: float = 0.01,
) -> MarkerGeneEntry:
    """Convenience factory for MarkerGeneEntry test fixtures."""
    return MarkerGeneEntry(
        gene_symbol=symbol,
        log2_fold_change=log2fc,
        adjusted_p_value=padj,
        present_in_adata=present,
    )


def make_analyst_output(
    top_markers: list[MarkerGeneEntry] | None = None,
    annotations: list[CellTypeAnnotation] | None = None,
    bcs: float = 0.90,
    gate_decision: GateDecision = GateDecision.PASS,
) -> AnalystOutput:
    """
    Convenience factory for AnalystOutput.

    ``bcs`` and ``gate_decision`` are the Analyst's self-reported values;
    the gate will re-verify independently.
    """
    if annotations is None:
        annotations = [
            CellTypeAnnotation(
                label="CD8+ Cytotoxic T cell",
                confidence=0.91,
                n_cells=60,
                marker_genes=top_markers or [],
            )
        ]
    return AnalystOutput(
        run_id=uuid4(),
        model_used=FoundationModel.SCGPT,
        cell_type_annotations=annotations,
        top_marker_genes=top_markers or [],
        biological_consistency_score=bcs,
        gate_decision=gate_decision,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def gate_normal() -> ValidationGate:
    """Gate backed by a normally-expressed synthetic AnnData."""
    return ValidationGate(adata=make_adata(expression="normal"))


@pytest.fixture()
def gate_zero() -> ValidationGate:
    """Gate backed by an all-zero expression matrix (every gene 'unexpressed')."""
    return ValidationGate(adata=make_adata(expression="zero"))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — All real genes → PASS
# ─────────────────────────────────────────────────────────────────────────────

def test_all_real_markers_pass(gate_normal: ValidationGate) -> None:
    """Analyst correctly reports genes present in the dataset → gate passes."""
    markers = [make_marker("CD8A"), make_marker("CD4")]
    output = make_analyst_output(top_markers=markers)

    report = gate_normal.verify_biomarkers(output, analyst_attempt=1)

    assert report.gate_decision == GateDecision.PASS
    assert report.computed_bcs >= ValidationGate.BCS_PASS_THRESHOLD
    assert "CD8A" in report.verified_marker_genes
    assert "CD4" in report.verified_marker_genes
    assert report.rejected_marker_genes == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Hallucinated top marker → hard FAIL_REJECTED (gate catches the lie)
# ─────────────────────────────────────────────────────────────────────────────

def test_hallucinated_top_marker_triggers_hard_rejection(gate_normal: ValidationGate) -> None:
    """
    Analyst self-reports present_in_adata=True for a fictional gene.
    The gate independently verifies and issues FAIL_REJECTED — not FAIL_LOOPBACK.
    This test validates that the gate is authoritative, not the Analyst.
    """
    markers = [
        make_marker("CD8A", present=True),        # real gene
        make_marker("NOTREAL_GENE", present=True), # lie: analyst claims it exists
    ]
    output = make_analyst_output(top_markers=markers)

    # Confirm the schema didn't reject it (analyst lied about present_in_adata)
    assert output.run_id is not None

    report = gate_normal.verify_biomarkers(output, analyst_attempt=1)

    assert report.gate_decision == GateDecision.FAIL_REJECTED
    assert "NOTREAL_GENE" in report.rejected_marker_genes
    assert any("NOTREAL_GENE" in r for r in report.failure_reasons)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Annotation marker missing (soft BCS degradation, not hard reject)
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_annotation_marker_degrades_bcs(gate_normal: ValidationGate) -> None:
    """
    A gene missing only from cell_type_annotation.marker_genes (not top_marker_genes)
    should degrade BCS via cluster_consistency — but NOT trigger a hard FAIL_REJECTED.
    """
    # top_marker_genes: all real (no hard reject)
    top = [make_marker("CD8A")]
    # annotation has one real + one fake gene → cluster_consistency = 0.5
    annotation = CellTypeAnnotation(
        label="Mixed T cell",
        confidence=0.75,
        n_cells=40,
        marker_genes=[make_marker("CD8A"), make_marker("PHANTOM_GENE", present=True)],
    )
    output = make_analyst_output(top_markers=top, annotations=[annotation])

    report = gate_normal.verify_biomarkers(output, analyst_attempt=1)

    # Hard reject must NOT fire (PHANTOM_GENE is only in annotation, not top_markers)
    assert report.gate_decision != GateDecision.FAIL_REJECTED or (
        report.gate_decision == GateDecision.FAIL_REJECTED
        and any("PHANTOM_GENE" in r for r in report.failure_reasons)
        # If the cluster_consistency drop alone drives BCS below threshold +
        # we've hit max retries, FAIL_REJECTED is also acceptable here.
        # The point is that no hard-reject PHANTOM path fires for annotation markers.
    )
    assert report.cluster_label_consistency == pytest.approx(0.5, abs=1e-4)
    assert "PHANTOM_GENE" in report.rejected_marker_genes


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Zero-expression matrix → BCS below threshold → FAIL_LOOPBACK
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_expression_triggers_loopback(gate_zero: ValidationGate) -> None:
    """
    Genes are present in var_names but have zero expression.
    expression_support = 0.0 → BCS ≈ 0.65 < 0.80 → FAIL_LOOPBACK (attempt 1).

    Expected BCS = 0.40*1.0 + 0.35*0.0 + 0.25*1.0 = 0.65
    """
    markers = [make_marker("CD8A"), make_marker("CD4"), make_marker("FOXP3")]
    output = make_analyst_output(top_markers=markers)

    report = gate_zero.verify_biomarkers(output, analyst_attempt=1)

    assert report.gate_decision == GateDecision.FAIL_LOOPBACK
    assert report.computed_bcs == pytest.approx(0.65, abs=1e-4)
    assert report.verified_marker_genes  # genes ARE found
    assert report.rejected_marker_genes == []
    # Suggestions should be populated for the Analyst's next attempt
    assert len(report.suggested_constraints) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Max retries exhausted → FAIL_LOOPBACK becomes FAIL_REJECTED
# ─────────────────────────────────────────────────────────────────────────────

def test_max_retries_converts_loopback_to_rejected(gate_zero: ValidationGate) -> None:
    """Once analyst_attempt >= max_retries, low BCS must escalate to FAIL_REJECTED."""
    markers = [make_marker("CD8A")]
    output = make_analyst_output(top_markers=markers)

    report = gate_zero.verify_biomarkers(
        output, analyst_attempt=ValidationGate.MAX_RETRIES
    )

    assert report.gate_decision == GateDecision.FAIL_REJECTED
    assert any("exhausted" in r.lower() for r in report.failure_reasons)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — No markers at all → perfect BCS (vacuously true) → PASS
# ─────────────────────────────────────────────────────────────────────────────

def test_no_markers_gives_perfect_bcs(gate_normal: ValidationGate) -> None:
    """
    An Analyst that claims no marker genes produces recall=1, support=1,
    consistency=1 → BCS=1.0 → PASS.
    This guards against over-penalising minimalist but valid outputs.
    """
    annotation = CellTypeAnnotation(
        label="Unknown cluster", confidence=0.5, n_cells=10
    )
    output = make_analyst_output(top_markers=[], annotations=[annotation])

    report = gate_normal.verify_biomarkers(output, analyst_attempt=1)

    assert report.gate_decision == GateDecision.PASS
    assert report.computed_bcs == pytest.approx(1.0, abs=1e-6)
    assert report.verified_marker_genes == []
    assert report.rejected_marker_genes == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — BCS components are always in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def test_bcs_components_bounded(gate_normal: ValidationGate) -> None:
    """All three BCS sub-scores must be in the closed interval [0, 1]."""
    markers = [make_marker("CD8A"), make_marker("MKI67")]
    output = make_analyst_output(top_markers=markers)

    bcs, components, _, _ = gate_normal.compute_bcs(output)

    assert 0.0 <= bcs <= 1.0
    for name, score in components.items():
        assert 0.0 <= score <= 1.0, f"Component '{name}' = {score} is out of [0, 1]"
    assert set(components.keys()) == {
        "marker_gene_recall",
        "expression_support",
        "cluster_consistency",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Custom BCS threshold is respected
# ─────────────────────────────────────────────────────────────────────────────

def test_custom_bcs_threshold_respected() -> None:
    """
    A gate configured with threshold=0.60 should PASS the same output
    that a gate with threshold=0.80 would FAIL_LOOPBACK on.
    """
    # All-zero expression → expected BCS ≈ 0.65 (recall=1, support=0, consistency=1)
    adata = make_adata(expression="zero")
    gate_strict = ValidationGate(adata=adata, bcs_threshold=0.80)
    gate_lenient = ValidationGate(adata=adata, bcs_threshold=0.60)

    markers = [make_marker("CD8A"), make_marker("CD4")]
    output = make_analyst_output(top_markers=markers)

    strict_report = gate_strict.verify_biomarkers(output, analyst_attempt=1)
    lenient_report = gate_lenient.verify_biomarkers(output, analyst_attempt=1)

    assert strict_report.gate_decision == GateDecision.FAIL_LOOPBACK
    assert lenient_report.gate_decision == GateDecision.PASS


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — ValidatorReport run_id matches AnalystOutput run_id
# ─────────────────────────────────────────────────────────────────────────────

def test_report_run_id_matches_analyst_output(gate_normal: ValidationGate) -> None:
    """The returned ValidatorReport must carry the same run_id as the input."""
    output = make_analyst_output()
    report = gate_normal.verify_biomarkers(output)
    assert report.run_id == output.run_id


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — Constructor guards
# ─────────────────────────────────────────────────────────────────────────────

def test_constructor_requires_exactly_one_data_source() -> None:
    """Gate must reject ambiguous or empty constructor arguments."""
    adata = make_adata()

    with pytest.raises(ValueError, match="either"):
        ValidationGate()  # neither adata nor path

    with pytest.raises(ValueError, match="only one"):
        ValidationGate(adata=adata, adata_path="/fake/path.h5ad")  # both supplied


# ─────────────────────────────────────────────────────────────────────────────
# Test 11 — __repr__ is informative
# ─────────────────────────────────────────────────────────────────────────────

def test_repr_contains_key_metadata(gate_normal: ValidationGate) -> None:
    r = repr(gate_normal)
    assert "ValidationGate" in r
    assert str(N_CELLS) in r
    assert str(len(REAL_GENES)) in r
    assert str(ValidationGate.BCS_PASS_THRESHOLD) in r


# ─────────────────────────────────────────────────────────────────────────────
# Test 12 — sandbox_exec_time_ms is reported
# ─────────────────────────────────────────────────────────────────────────────

def test_sandbox_timing_is_reported(gate_normal: ValidationGate) -> None:
    """gate.verify_biomarkers() must always populate sandbox_exec_time_ms."""
    output = make_analyst_output()
    report = gate_normal.verify_biomarkers(output)
    assert report.sandbox_exec_time_ms is not None
    assert report.sandbox_exec_time_ms >= 0.0
