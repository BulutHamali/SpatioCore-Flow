"""
tests/test_schemas.py — Schema validation tests for SpatioCore Flow v2.0.

These tests serve as the regression baseline for the Pydantic contract layer.
All schema invariants documented in logic/schemas.py are exercised here.
"""

from __future__ import annotations

import pytest
from uuid import uuid4

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
    GuardrailReport,
    MarkerGeneEntry,
    RegulatoryFlag,
    RiskLevel,
    SpatialSpotReference,
    ValidatorReport,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID = uuid4()


def _valid_marker(present: bool = True) -> MarkerGeneEntry:
    return MarkerGeneEntry(
        gene_symbol="CD8A",
        ensemble_id="ENSG00000153563",
        log2_fold_change=2.5,
        adjusted_p_value=0.001,
        present_in_adata=present,
    )


def _valid_cell_type() -> CellTypeAnnotation:
    return CellTypeAnnotation(
        label="CD8+ Cytotoxic T cell",
        confidence=0.92,
        marker_genes=[_valid_marker()],
        n_cells=450,
    )


def _valid_analyst_output(**overrides) -> dict:
    base = dict(
        run_id=RUN_ID,
        model_used=FoundationModel.SCGPT,
        cell_type_annotations=[_valid_cell_type()],
        biological_consistency_score=0.88,
        gate_decision=GateDecision.PASS,
    )
    base.update(overrides)
    return base


def _valid_claim(verified: bool = True) -> BiologicalClaim:
    kwargs = dict(
        claim_text="Spot X is dominated by CD8+ T cells",
        source_agent="analyst",
        confidence=0.91,
        is_verified=verified,
    )
    if verified:
        kwargs["data_index_refs"] = [
            DataIndexReference(adata_obs_index="ACGTACGT-1", adata_var_index="CD8A")
        ]
    return BiologicalClaim(**kwargs)


def _valid_auditor_report(**overrides) -> dict:
    base = dict(
        run_id=RUN_ID,
        final_bcs=0.88,
        analyst_attempts_used=1,
        verified_claims=[_valid_claim(verified=True)],
    )
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# CuratorOutput
# ─────────────────────────────────────────────────────────────────────────────


def test_curator_output_basic():
    curator = CuratorOutput(
        modality=DataModality.SPATIAL_VISIUM,
        is_spatial=True,
        n_cells=3000,
        n_genes=33000,
        has_raw_counts=True,
        coordinate_system="in_situ_tissue",
    )
    assert curator.is_spatial is True
    assert curator.modality == DataModality.SPATIAL_VISIUM


# ─────────────────────────────────────────────────────────────────────────────
# MarkerGeneEntry
# ─────────────────────────────────────────────────────────────────────────────


def test_marker_gene_symbol_uppercased():
    m = MarkerGeneEntry(
        gene_symbol="cd8a",
        log2_fold_change=1.5,
        adjusted_p_value=0.05,
        present_in_adata=True,
    )
    assert m.gene_symbol == "CD8A"


def test_marker_gene_p_value_bounds():
    with pytest.raises(Exception):
        MarkerGeneEntry(
            gene_symbol="TP53",
            log2_fold_change=1.0,
            adjusted_p_value=1.5,   # invalid > 1
            present_in_adata=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# AnalystOutput  ★
# ─────────────────────────────────────────────────────────────────────────────


def test_analyst_output_valid():
    out = AnalystOutput(**_valid_analyst_output())
    assert out.biological_consistency_score == 0.88
    assert out.gate_decision == GateDecision.PASS


def test_analyst_output_hallucinated_marker_forces_reject():
    """Any top_marker_genes entry with present_in_adata=False → FAIL_REJECTED."""
    data = _valid_analyst_output(
        top_marker_genes=[_valid_marker(present=False)],
        gate_decision=GateDecision.PASS,  # would be overridden by validator
    )
    out = AnalystOutput(**data)
    assert out.gate_decision == GateDecision.FAIL_REJECTED


def test_analyst_output_deconv_proportions_must_sum_to_one():
    data = _valid_analyst_output(
        deconvolution_results={"T cell": 0.5, "B cell": 0.3}  # sums to 0.8
    )
    with pytest.raises(Exception, match="sum to ~1.0"):
        AnalystOutput(**data)


def test_analyst_output_deconv_valid():
    data = _valid_analyst_output(
        deconvolution_results={"T cell": 0.6, "B cell": 0.4}
    )
    out = AnalystOutput(**data)
    assert abs(sum(out.deconvolution_results.values()) - 1.0) < 0.01


def test_analyst_output_attempt_increments():
    out1 = AnalystOutput(**_valid_analyst_output(attempt=1))
    out2 = AnalystOutput(**_valid_analyst_output(attempt=2))
    assert out2.attempt == out1.attempt + 1


def test_analyst_output_requires_at_least_one_cell_type():
    data = _valid_analyst_output(cell_type_annotations=[])
    with pytest.raises(Exception):
        AnalystOutput(**data)


def test_analyst_output_bcs_bounds():
    with pytest.raises(Exception):
        AnalystOutput(**_valid_analyst_output(biological_consistency_score=1.5))
    with pytest.raises(Exception):
        AnalystOutput(**_valid_analyst_output(biological_consistency_score=-0.1))


# ─────────────────────────────────────────────────────────────────────────────
# ValidatorReport
# ─────────────────────────────────────────────────────────────────────────────


def test_validator_report_gate_consistent_with_bcs():
    """BCS above threshold + FAIL_LOOPBACK is an invalid combination."""
    with pytest.raises(Exception, match="inconsistent"):
        ValidatorReport(
            run_id=RUN_ID,
            analyst_attempt=1,
            computed_bcs=0.90,       # above threshold
            bcs_threshold=0.80,
            cluster_label_consistency=0.95,
            gate_decision=GateDecision.FAIL_LOOPBACK,  # contradiction
        )


def test_validator_report_pass_on_high_bcs():
    report = ValidatorReport(
        run_id=RUN_ID,
        analyst_attempt=1,
        computed_bcs=0.90,
        bcs_threshold=0.80,
        cluster_label_consistency=0.95,
        gate_decision=GateDecision.PASS,
    )
    assert report.gate_decision == GateDecision.PASS


# ─────────────────────────────────────────────────────────────────────────────
# BiologicalClaim
# ─────────────────────────────────────────────────────────────────────────────


def test_verified_claim_needs_evidence():
    with pytest.raises(Exception, match="traceability link"):
        BiologicalClaim(
            claim_text="Some claim",
            source_agent="analyst",
            confidence=0.9,
            is_verified=True,
            data_index_refs=[],   # empty — should raise
            spatial_refs=[],
        )


def test_unverified_claim_no_evidence_allowed():
    claim = BiologicalClaim(
        claim_text="Unverified claim",
        source_agent="synthesizer",
        confidence=0.4,
        is_verified=False,
    )
    assert not claim.is_verified


# ─────────────────────────────────────────────────────────────────────────────
# AuditorReport  ★
# ─────────────────────────────────────────────────────────────────────────────


def test_auditor_report_computes_traceability():
    verified = _valid_claim(verified=True)
    unverified = _valid_claim(verified=False)
    report = AuditorReport(
        run_id=RUN_ID,
        final_bcs=0.88,
        analyst_attempts_used=1,
        verified_claims=[verified],
        unverified_claims=[unverified],
    )
    assert report.total_claims == 2
    assert report.traceability_coverage == pytest.approx(0.5)


def test_auditor_report_fully_traced_property():
    report = AuditorReport(**_valid_auditor_report())
    assert report.is_fully_traced is True


def test_auditor_report_has_unverified_property():
    unverified = _valid_claim(verified=False)
    report = AuditorReport(
        **_valid_auditor_report(unverified_claims=[unverified])
    )
    assert report.has_unverified_claims is True


def test_auditor_report_verified_claims_must_be_flagged():
    """verified_claims list entries must have is_verified=True."""
    bad_claim = _valid_claim(verified=False)  # is_verified=False
    with pytest.raises(Exception, match="is_verified=True"):
        AuditorReport(
            run_id=RUN_ID,
            final_bcs=0.88,
            analyst_attempts_used=1,
            verified_claims=[bad_claim],  # should raise
        )


def test_auditor_report_zero_claims():
    report = AuditorReport(
        run_id=RUN_ID,
        final_bcs=0.85,
        analyst_attempts_used=2,
    )
    assert report.total_claims == 0
    assert report.traceability_coverage == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GuardrailReport
# ─────────────────────────────────────────────────────────────────────────────


def test_guardrail_blocked_needs_reasons():
    audit_id = uuid4()
    with pytest.raises(Exception, match="blocking_reasons"):
        GuardrailReport(
            run_id=RUN_ID,
            audit_ref=audit_id,
            overall_risk=RiskLevel.HIGH,
            samd_compliant=False,
            approved_for_report=False,
            blocking_reasons=[],   # must not be empty when blocked
        )


def test_guardrail_approved_report():
    audit_id = uuid4()
    report = GuardrailReport(
        run_id=RUN_ID,
        audit_ref=audit_id,
        overall_risk=RiskLevel.LOW,
        samd_compliant=True,
        approved_for_report=True,
    )
    assert report.approved_for_report is True
