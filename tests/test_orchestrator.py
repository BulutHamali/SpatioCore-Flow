"""
tests/test_orchestrator.py — Orchestrator retry-loop tests
===========================================================
All tests use stub runners (plain Python callables) and a real
ValidationGate backed by an in-memory synthetic AnnData.

No CrewAI, no LLM calls, no network — pure orchestration logic.

Synthetic dataset
-----------------
Genes in AnnData: CD8A, CD4, FOXP3, MKI67, EPCAM  (5 genes)
Expression: normally distributed (all genes above expression floor)

Retry scenarios exercised
--------------------------
A. First-attempt PASS         — gate clears immediately
B. One loopback → then PASS   — FAIL_LOOPBACK on attempt 1, PASS on attempt 2
C. missing_genes forwarded    — rejected gene propagates to next AnalystRunInput
D. constraint_adjustments fwd — suggested constraints propagate correctly
E. Hard reject (FAIL_REJECTED)— hallucinated top marker → immediate exception
F. Retry budget exhausted     — all attempts FAIL_LOOPBACK → exception
"""

from __future__ import annotations

import pytest

anndata = pytest.importorskip("anndata")
numpy = pytest.importorskip("numpy")

import numpy as np
import pandas as pd
from uuid import uuid4, UUID

from logic.gates import ValidationGate
from logic.orchestrator import (
    AnalystRunInput,
    AuditorRunInput,
    OrchestratorResult,
    PipelineRejectionError,
    SpatioFlowOrchestrator,
)
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
    ValidatorReport,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

REAL_GENES = ["CD8A", "CD4", "FOXP3", "MKI67", "EPCAM"]
N_CELLS = 80


def make_adata(genes: list[str] = REAL_GENES, expression: str = "normal") -> anndata.AnnData:
    rng = np.random.default_rng(seed=7)
    n = len(genes)
    if expression == "normal":
        X = np.clip(rng.exponential(2.5, (N_CELLS, n)), 0.1, None).astype(np.float32)
    elif expression == "zero":
        X = np.zeros((N_CELLS, n), dtype=np.float32)
    else:
        raise ValueError(expression)
    var = pd.DataFrame(index=pd.Index(genes))
    obs = pd.DataFrame(index=pd.Index([f"cell_{i}" for i in range(N_CELLS)]))
    return anndata.AnnData(X=X, obs=obs, var=var)


def make_marker(symbol: str, present: bool = True) -> MarkerGeneEntry:
    return MarkerGeneEntry(
        gene_symbol=symbol,
        log2_fold_change=2.0,
        adjusted_p_value=0.01,
        present_in_adata=present,
    )


def make_analyst_output(
    run_id: UUID,
    top_markers: list[MarkerGeneEntry],
    attempt: int = 1,
    annotation_markers: list[MarkerGeneEntry] | None = None,
) -> AnalystOutput:
    """Build an AnalystOutput with the given top_markers."""
    return AnalystOutput(
        run_id=run_id,
        attempt=attempt,
        model_used=FoundationModel.SCGPT,
        cell_type_annotations=[
            CellTypeAnnotation(
                label="CD8+ T cell",
                confidence=0.90,
                n_cells=40,
                marker_genes=annotation_markers or top_markers,
            )
        ],
        top_marker_genes=top_markers,
        biological_consistency_score=0.88,
        gate_decision=GateDecision.FAIL_LOOPBACK,  # gate will override
    )


def stub_curator_runner(adata, run_id_str: str) -> CuratorOutput:
    return CuratorOutput(
        modality=DataModality.SINGLE_CELL_RNA,
        is_spatial=False,
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
        has_raw_counts=True,
        coordinate_system="dissociated_cells",
    )


def stub_auditor_runner(run_input: AuditorRunInput) -> AuditorReport:
    ao = run_input.analyst_output
    vr = run_input.validator_report
    verified = [
        BiologicalClaim(
            claim_text=f"Annotation '{ann.label}' traced to AnnData.",
            source_agent="analyst",
            confidence=ann.confidence,
            is_verified=True,
            data_index_refs=[DataIndexReference(adata_obs_index="cell_0")],
        )
        for ann in ao.cell_type_annotations
    ]
    return AuditorReport(
        run_id=run_input.run_id,
        final_bcs=vr.computed_bcs,
        analyst_attempts_used=vr.analyst_attempt,
        verified_claims=verified,
    )


@pytest.fixture()
def gate_normal() -> ValidationGate:
    return ValidationGate(adata=make_adata(expression="normal"))


@pytest.fixture()
def gate_zero() -> ValidationGate:
    """Gate where all genes are present but unexpressed — BCS ≈ 0.65 → loopback."""
    return ValidationGate(adata=make_adata(expression="zero"))


@pytest.fixture()
def orchestrator_normal(gate_normal: ValidationGate) -> SpatioFlowOrchestrator:
    return SpatioFlowOrchestrator(gate=gate_normal)


@pytest.fixture()
def orchestrator_zero(gate_zero: ValidationGate) -> SpatioFlowOrchestrator:
    return SpatioFlowOrchestrator(gate=gate_zero)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario A — First-attempt PASS
# ─────────────────────────────────────────────────────────────────────────────

def test_passes_on_first_attempt(
    orchestrator_normal: SpatioFlowOrchestrator,
) -> None:
    """
    Analyst returns real genes with normal expression → gate passes on attempt 1.
    OrchestratorResult must show total_analyst_attempts == 1.
    """
    adata = make_adata()
    fixed_run_id = uuid4()

    def analyst_runner(run_input: AnalystRunInput) -> AnalystOutput:
        return make_analyst_output(
            run_id=run_input.run_id,
            top_markers=[make_marker("CD8A"), make_marker("CD4")],
            attempt=run_input.attempt,
        )

    result = orchestrator_normal.run(
        adata=adata,
        curator_runner=stub_curator_runner,
        analyst_runner=analyst_runner,
        auditor_runner=stub_auditor_runner,
        run_id=fixed_run_id,
    )

    assert isinstance(result, OrchestratorResult)
    assert result.success is True
    assert result.total_analyst_attempts == 1
    assert result.final_validator_report.is_passed is True
    assert result.run_id == fixed_run_id
    assert result.auditor_report.run_id == fixed_run_id


# ─────────────────────────────────────────────────────────────────────────────
# Scenario B — One loopback, then PASS
# ─────────────────────────────────────────────────────────────────────────────

def test_one_loopback_then_pass(
    orchestrator_zero: SpatioFlowOrchestrator,
) -> None:
    """
    Attempt 1: genes present but expression = 0 → BCS ≈ 0.65 → FAIL_LOOPBACK.
    Attempt 2: analyst returns no top_marker_genes → BCS = 1.0 → PASS.
    """
    adata = make_adata(expression="zero")
    call_count = {"n": 0}

    def analyst_runner(run_input: AnalystRunInput) -> AnalystOutput:
        call_count["n"] += 1
        if run_input.attempt == 1:
            # Returns genes that exist but are unexpressed → loopback
            return make_analyst_output(
                run_id=run_input.run_id,
                top_markers=[make_marker("CD8A")],
                attempt=run_input.attempt,
            )
        # Attempt 2: no top markers → BCS = 1.0 → PASS
        return AnalystOutput(
            run_id=run_input.run_id,
            attempt=run_input.attempt,
            model_used=FoundationModel.SCGPT,
            cell_type_annotations=[
                CellTypeAnnotation(label="Sparse cluster", confidence=0.6, n_cells=10)
            ],
            top_marker_genes=[],
            biological_consistency_score=1.0,
            gate_decision=GateDecision.FAIL_LOOPBACK,
        )

    result = orchestrator_zero.run(
        adata=adata,
        curator_runner=stub_curator_runner,
        analyst_runner=analyst_runner,
        auditor_runner=stub_auditor_runner,
    )

    assert result.total_analyst_attempts == 2
    assert call_count["n"] == 2
    assert result.final_validator_report.is_passed is True
    assert len(result.gate_history) == 2
    assert result.gate_history[0].gate_decision == GateDecision.FAIL_LOOPBACK
    assert result.gate_history[1].gate_decision == GateDecision.PASS


# ─────────────────────────────────────────────────────────────────────────────
# Scenario C — missing_genes forwarded to the next AnalystRunInput
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_genes_forwarded_to_analyst_on_retry(
    orchestrator_normal: SpatioFlowOrchestrator,
) -> None:
    """
    Attempt 1 includes a phantom annotation marker (PHANTOM1).
    The gate rejects it via BCS degradation (not hard-reject — it's in
    annotation only, not in top_marker_genes).
    Attempt 2 must receive missing_genes=['PHANTOM1'].
    """
    adata = make_adata()
    received_inputs: list[AnalystRunInput] = []

    def analyst_runner(run_input: AnalystRunInput) -> AnalystOutput:
        received_inputs.append(run_input)
        if run_input.attempt == 1:
            # top_markers: real gene (no hard reject)
            # annotation_markers: real + phantom (cluster_consistency = 0.5)
            return make_analyst_output(
                run_id=run_input.run_id,
                top_markers=[make_marker("CD8A")],
                annotation_markers=[make_marker("CD8A"), make_marker("PHANTOM1", present=True)],
                attempt=1,
            )
        # Attempt 2: clean output
        return make_analyst_output(
            run_id=run_input.run_id,
            top_markers=[make_marker("CD4")],
            attempt=run_input.attempt,
        )

    orchestrator_normal.run(
        adata=adata,
        curator_runner=stub_curator_runner,
        analyst_runner=analyst_runner,
        auditor_runner=stub_auditor_runner,
    )

    assert len(received_inputs) == 2
    # Attempt 1 should have no prior rejection context
    assert received_inputs[0].missing_genes == []

    # Attempt 2 MUST receive the phantom gene as missing
    assert "PHANTOM1" in received_inputs[1].missing_genes


# ─────────────────────────────────────────────────────────────────────────────
# Scenario D — constraint_adjustments forwarded
# ─────────────────────────────────────────────────────────────────────────────

def test_constraint_adjustments_forwarded_to_analyst(
    orchestrator_zero: SpatioFlowOrchestrator,
) -> None:
    """
    Gate's suggested_constraints from attempt 1 must appear in attempt 2's
    AnalystRunInput.constraint_adjustments.
    """
    adata = make_adata(expression="zero")
    received_inputs: list[AnalystRunInput] = []

    def analyst_runner(run_input: AnalystRunInput) -> AnalystOutput:
        received_inputs.append(run_input)
        if run_input.attempt == 1:
            return make_analyst_output(
                run_id=run_input.run_id,
                top_markers=[make_marker("CD8A")],
                attempt=1,
            )
        return AnalystOutput(
            run_id=run_input.run_id,
            attempt=run_input.attempt,
            model_used=FoundationModel.SCGPT,
            cell_type_annotations=[
                CellTypeAnnotation(label="Adjusted cluster", confidence=0.7, n_cells=5)
            ],
            top_marker_genes=[],
            biological_consistency_score=1.0,
            gate_decision=GateDecision.FAIL_LOOPBACK,
        )

    orchestrator_zero.run(
        adata=adata,
        curator_runner=stub_curator_runner,
        analyst_runner=analyst_runner,
        auditor_runner=stub_auditor_runner,
    )

    assert len(received_inputs) == 2
    # The constraints from the gate (loopback suggestions) must be forwarded
    assert len(received_inputs[1].constraint_adjustments) > 0
    # Spot-check one known suggestion from the gate
    all_constraints = " ".join(received_inputs[1].constraint_adjustments).lower()
    assert "cluster" in all_constraints or "hvg" in all_constraints or "variable" in all_constraints


# ─────────────────────────────────────────────────────────────────────────────
# Scenario E — Hard FAIL_REJECTED raises PipelineRejectionError immediately
# ─────────────────────────────────────────────────────────────────────────────

def test_hard_reject_raises_immediately(
    orchestrator_normal: SpatioFlowOrchestrator,
) -> None:
    """
    Analyst includes 'GHOST_GENE' in top_marker_genes (not in AnnData).
    Gate issues FAIL_REJECTED on attempt 1 — no loopback, no second call.
    """
    adata = make_adata()
    call_count = {"n": 0}

    def analyst_runner(run_input: AnalystRunInput) -> AnalystOutput:
        call_count["n"] += 1
        return make_analyst_output(
            run_id=run_input.run_id,
            top_markers=[
                make_marker("CD8A"),                   # real
                make_marker("GHOST_GENE", present=True),  # hallucinated (analyst lies)
            ],
            attempt=run_input.attempt,
        )

    with pytest.raises(PipelineRejectionError) as exc_info:
        orchestrator_normal.run(
            adata=adata,
            curator_runner=stub_curator_runner,
            analyst_runner=analyst_runner,
            auditor_runner=stub_auditor_runner,
        )

    # Must raise on the very first attempt — no retries
    assert call_count["n"] == 1
    err = exc_info.value
    assert err.attempt == 1
    assert err.validator_report.gate_decision == GateDecision.FAIL_REJECTED
    assert "GHOST_GENE" in str(err)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario F — Retry budget exhausted → PipelineRejectionError
# ─────────────────────────────────────────────────────────────────────────────

def test_retry_budget_exhausted_raises(
    orchestrator_zero: SpatioFlowOrchestrator,
) -> None:
    """
    Every Analyst attempt triggers FAIL_LOOPBACK (zero expression, BCS ≈ 0.65).
    After max_retries attempts the orchestrator raises PipelineRejectionError.
    """
    adata = make_adata(expression="zero")
    call_count = {"n": 0}

    def always_failing_analyst(run_input: AnalystRunInput) -> AnalystOutput:
        call_count["n"] += 1
        return make_analyst_output(
            run_id=run_input.run_id,
            top_markers=[make_marker("FOXP3")],  # real gene, but expression = 0
            attempt=run_input.attempt,
        )

    with pytest.raises(PipelineRejectionError) as exc_info:
        orchestrator_zero.run(
            adata=adata,
            curator_runner=stub_curator_runner,
            analyst_runner=always_failing_analyst,
            auditor_runner=stub_auditor_runner,
        )

    max_r = orchestrator_zero.gate.max_retries
    assert call_count["n"] == max_r, (
        f"Expected {max_r} analyst calls, got {call_count['n']}"
    )
    err = exc_info.value
    assert err.attempt == max_r


# ─────────────────────────────────────────────────────────────────────────────
# Gate history is always complete
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_history_matches_attempt_count(
    orchestrator_zero: SpatioFlowOrchestrator,
) -> None:
    """gate_history must contain one ValidatorReport per analyst attempt."""
    adata = make_adata(expression="zero")

    def analyst_runner(run_input: AnalystRunInput) -> AnalystOutput:
        if run_input.attempt < orchestrator_zero.gate.max_retries:
            return make_analyst_output(
                run_id=run_input.run_id,
                top_markers=[make_marker("MKI67")],
                attempt=run_input.attempt,
            )
        # Final attempt: no markers → PASS
        return AnalystOutput(
            run_id=run_input.run_id,
            attempt=run_input.attempt,
            model_used=FoundationModel.SCGPT,
            cell_type_annotations=[
                CellTypeAnnotation(label="Clean", confidence=0.8, n_cells=1)
            ],
            top_marker_genes=[],
            biological_consistency_score=1.0,
            gate_decision=GateDecision.FAIL_LOOPBACK,
        )

    result = orchestrator_zero.run(
        adata=adata,
        curator_runner=stub_curator_runner,
        analyst_runner=analyst_runner,
        auditor_runner=stub_auditor_runner,
    )

    max_r = orchestrator_zero.gate.max_retries
    assert len(result.gate_history) == max_r
    assert result.total_analyst_attempts == max_r
    # All earlier reports are loopbacks, last is PASS
    for report in result.gate_history[:-1]:
        assert report.gate_decision == GateDecision.FAIL_LOOPBACK
    assert result.gate_history[-1].is_passed


# ─────────────────────────────────────────────────────────────────────────────
# run_id propagates end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_run_id_propagates_through_all_outputs(
    orchestrator_normal: SpatioFlowOrchestrator,
) -> None:
    """The same UUID must appear in CuratorOutput, AnalystOutput, ValidatorReport, AuditorReport."""
    adata = make_adata()
    fixed_id = uuid4()

    def analyst_runner(run_input: AnalystRunInput) -> AnalystOutput:
        return make_analyst_output(
            run_id=run_input.run_id,
            top_markers=[make_marker("EPCAM")],
            attempt=run_input.attempt,
        )

    result = orchestrator_normal.run(
        adata=adata,
        curator_runner=stub_curator_runner,
        analyst_runner=analyst_runner,
        auditor_runner=stub_auditor_runner,
        run_id=fixed_id,
    )

    assert result.run_id == fixed_id
    assert result.analyst_output.run_id == fixed_id
    assert result.final_validator_report.run_id == fixed_id
    assert result.auditor_report.run_id == fixed_id
