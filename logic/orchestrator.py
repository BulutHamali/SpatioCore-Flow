"""
logic/orchestrator.py — SpatioCore Flow v2.0 Pipeline Orchestrator
===================================================================
The SpatioFlowOrchestrator manages the full agent pipeline and implements
the Confidence Gate retry loop defined in the README architecture.

Pipeline execution order
------------------------
  1. Curator runner  → CuratorOutput
  2. Analyst runner  → AnalystOutput   ─┐
     ValidationGate  → ValidatorReport  │  repeated up to max_retries
       • PASS        → proceed          │
       • FAIL_LOOPBACK → re-prompt      │  (missing_genes + constraints injected)
       • FAIL_REJECTED → raise          ─┘
  3. Auditor runner  → AuditorReport
  → OrchestratorResult

Dependency injection
--------------------
The orchestrator is decoupled from CrewAI: it accepts plain Python callables
for each agent step.  ``agents_tasks.py`` provides the live CrewAI-backed
runners; tests supply lightweight stubs — no LLM calls required.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from logic.gates import ValidationGate
from logic.schemas import (
    AnalystOutput,
    AuditorReport,
    CuratorOutput,
    GateDecision,
    ValidatorReport,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data transfer objects for runner callables
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AnalystRunInput:
    """
    All context the Analyst runner needs for one attempt.

    ``missing_genes`` and ``constraint_adjustments`` are empty on attempt 1
    and populated from the previous ``ValidatorReport`` on each loopback.
    """

    run_id: UUID
    curator_output: CuratorOutput
    attempt: int
    missing_genes: list[str] = field(default_factory=list)
    constraint_adjustments: list[str] = field(default_factory=list)


@dataclass
class AuditorRunInput:
    """Context passed to the Auditor runner after the gate has passed."""

    run_id: UUID
    analyst_output: AnalystOutput
    validator_report: ValidatorReport
    curator_output: CuratorOutput


# ─────────────────────────────────────────────────────────────────────────────
# Runner callable type aliases (documentation only — Python uses duck-typing)
# ─────────────────────────────────────────────────────────────────────────────

CuratorRunner = Callable[[Any, str], CuratorOutput]
AnalystRunner = Callable[[AnalystRunInput], AnalystOutput]
AuditorRunner = Callable[[AuditorRunInput], AuditorReport]


# ─────────────────────────────────────────────────────────────────────────────
# Result & error types
# ─────────────────────────────────────────────────────────────────────────────


class OrchestratorResult(BaseModel):
    """
    Immutable record of a completed pipeline run.

    Contains the full provenance chain:
    Curator → Analyst (all attempts) → Gate history → Auditor.
    """

    run_id: UUID = Field(..., description="Pipeline run identifier (propagated to all agents).")
    curator_output: CuratorOutput
    analyst_output: AnalystOutput = Field(
        ..., description="The final, gate-approved Analyst output."
    )
    final_validator_report: ValidatorReport = Field(
        ..., description="The ValidatorReport that returned GateDecision.PASS."
    )
    auditor_report: AuditorReport
    total_analyst_attempts: int = Field(
        ..., ge=1, description="Number of Analyst attempts before gate passed."
    )
    gate_history: list[ValidatorReport] = Field(
        default_factory=list,
        description="All ValidatorReports in chronological order (useful for debugging).",
    )
    pipeline_duration_ms: float = Field(..., ge=0.0)
    success: bool = True

    model_config = {"arbitrary_types_allowed": True}


class PipelineRejectionError(Exception):
    """
    Raised when the Confidence Gate permanently rejects an inference.

    Triggered by:
      - ``GateDecision.FAIL_REJECTED`` at any attempt (hallucinated genes)
      - ``GateDecision.FAIL_LOOPBACK`` after all retries are exhausted
        (this should not happen in practice — the gate converts the final
        loopback to FAIL_REJECTED internally, but the orchestrator guards
        defensively).
    """

    def __init__(
        self,
        run_id: UUID,
        attempt: int,
        validator_report: ValidatorReport,
    ) -> None:
        self.run_id = run_id
        self.attempt = attempt
        self.validator_report = validator_report
        reasons = "; ".join(validator_report.failure_reasons) or "unknown"
        super().__init__(
            f"[run={run_id}] Pipeline permanently rejected at attempt {attempt}. "
            f"Gate decision: {validator_report.gate_decision.value}. "
            f"Reasons: {reasons}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


class SpatioFlowOrchestrator:
    """
    Coordinates the full SpatioCore Flow pipeline with a Confidence Gate
    retry loop between the Analyst (Tier 2) and Validator (Tier 3).

    Parameters
    ----------
    gate :
        A configured ``ValidationGate`` instance with the AnnData pre-loaded.
        The gate's ``max_retries`` controls how many Analyst attempts are allowed.

    Examples
    --------
    With live CrewAI agents::

        gate = ValidationGate(adata=adata)
        orchestrator = SpatioFlowOrchestrator(gate=gate)
        result = orchestrator.run(
            adata=adata,
            curator_runner=make_crewai_curator_runner(llm),
            analyst_runner=make_crewai_analyst_runner(llm),
            auditor_runner=make_crewai_auditor_runner(llm),
        )

    With test stubs (no LLM required)::

        result = orchestrator.run(
            adata=adata,
            curator_runner=my_stub_curator,
            analyst_runner=my_stub_analyst,
            auditor_runner=my_stub_auditor,
        )
    """

    def __init__(self, gate: ValidationGate) -> None:
        self.gate = gate

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        adata: Any,
        curator_runner: CuratorRunner,
        analyst_runner: AnalystRunner,
        auditor_runner: AuditorRunner,
        run_id: UUID | None = None,
    ) -> OrchestratorResult:
        """
        Execute the full pipeline for one AnnData input.

        Parameters
        ----------
        adata :
            The AnnData object (or path) for this run. Passed verbatim to the
            curator runner; the gate was already initialised with it.
        curator_runner :
            ``(adata, run_id_str) -> CuratorOutput``
        analyst_runner :
            ``(AnalystRunInput) -> AnalystOutput``
        auditor_runner :
            ``(AuditorRunInput) -> AuditorReport``
        run_id :
            Optional pre-assigned UUID (useful for deterministic tests).
            A fresh UUID4 is generated when not supplied.

        Returns
        -------
        OrchestratorResult

        Raises
        ------
        PipelineRejectionError
            When the gate issues FAIL_REJECTED (hallucinated genes or retry
            budget exhausted).
        """
        t0 = time.perf_counter()
        run_id = run_id or uuid4()
        gate_history: list[ValidatorReport] = []

        logger.info("=" * 60)
        logger.info("SpatioFlow run=%s STARTING", run_id)

        # ── Step 1: Curator ───────────────────────────────────────────────────
        curator_output = self._run_curator(curator_runner, adata, run_id)

        # ── Step 2: Analyst + Confidence Gate retry loop ──────────────────────
        analyst_output, final_validator_report = self._analyst_gate_loop(
            analyst_runner=analyst_runner,
            curator_output=curator_output,
            run_id=run_id,
            gate_history=gate_history,
        )

        # ── Step 3: Auditor ───────────────────────────────────────────────────
        auditor_report = self._run_auditor(
            auditor_runner=auditor_runner,
            run_id=run_id,
            analyst_output=analyst_output,
            validator_report=final_validator_report,
            curator_output=curator_output,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        logger.info("SpatioFlow run=%s COMPLETE in %.0fms", run_id, elapsed_ms)

        return OrchestratorResult(
            run_id=run_id,
            curator_output=curator_output,
            analyst_output=analyst_output,
            final_validator_report=final_validator_report,
            auditor_report=auditor_report,
            total_analyst_attempts=len(gate_history),
            gate_history=gate_history,
            pipeline_duration_ms=round(elapsed_ms, 2),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_curator(
        self,
        curator_runner: CuratorRunner,
        adata: Any,
        run_id: UUID,
    ) -> CuratorOutput:
        logger.info("[run=%s] ── Curator starting", run_id)
        curator_output = curator_runner(adata, str(run_id))
        logger.info(
            "[run=%s] Curator done: modality=%s cells=%d genes=%d",
            run_id,
            curator_output.modality.value,
            curator_output.n_cells,
            curator_output.n_genes,
        )
        return curator_output

    def _analyst_gate_loop(
        self,
        analyst_runner: AnalystRunner,
        curator_output: CuratorOutput,
        run_id: UUID,
        gate_history: list[ValidatorReport],
    ) -> tuple[AnalystOutput, ValidatorReport]:
        """
        Run the Analyst → Gate loop, forwarding rejection context on each loopback.

        Returns the gate-approved (AnalystOutput, ValidatorReport) pair.
        Raises PipelineRejectionError on permanent rejection.
        """
        missing_genes: list[str] = []
        constraint_adjustments: list[str] = []

        for attempt in range(1, self.gate.max_retries + 1):
            logger.info(
                "[run=%s] ── Analyst attempt %d/%d | missing_genes=%s",
                run_id,
                attempt,
                self.gate.max_retries,
                missing_genes or "[]",
            )

            run_input = AnalystRunInput(
                run_id=run_id,
                curator_output=curator_output,
                attempt=attempt,
                missing_genes=missing_genes,
                constraint_adjustments=constraint_adjustments,
            )
            analyst_output = analyst_runner(run_input)

            validator_report = self.gate.verify_biomarkers(analyst_output, attempt)
            gate_history.append(validator_report)

            logger.info(
                "[run=%s] Gate decision: %s (BCS=%.4f, verified=%d, rejected=%d)",
                run_id,
                validator_report.gate_decision.value,
                validator_report.computed_bcs,
                len(validator_report.verified_marker_genes),
                len(validator_report.rejected_marker_genes),
            )

            if validator_report.is_passed:
                return analyst_output, validator_report

            if validator_report.gate_decision == GateDecision.FAIL_REJECTED:
                raise PipelineRejectionError(
                    run_id=run_id,
                    attempt=attempt,
                    validator_report=validator_report,
                )

            # FAIL_LOOPBACK — harvest rejection context and loop
            missing_genes = validator_report.missing_genes
            constraint_adjustments = validator_report.suggested_constraints
            logger.info(
                "[run=%s] Looping back: forwarding %d missing genes, %d constraints",
                run_id,
                len(missing_genes),
                len(constraint_adjustments),
            )

        # All attempts exhausted — the final gate report should already be
        # FAIL_REJECTED (the gate converts the last loopback). Raise regardless.
        raise PipelineRejectionError(
            run_id=run_id,
            attempt=self.gate.max_retries,
            validator_report=gate_history[-1],
        )

    def _run_auditor(
        self,
        auditor_runner: AuditorRunner,
        run_id: UUID,
        analyst_output: AnalystOutput,
        validator_report: ValidatorReport,
        curator_output: CuratorOutput,
    ) -> AuditorReport:
        logger.info("[run=%s] ── Auditor starting", run_id)
        run_input = AuditorRunInput(
            run_id=run_id,
            analyst_output=analyst_output,
            validator_report=validator_report,
            curator_output=curator_output,
        )
        auditor_report = auditor_runner(run_input)
        logger.info(
            "[run=%s] Auditor done: traceability=%.0f%% claims=%d",
            run_id,
            auditor_report.traceability_coverage * 100,
            auditor_report.total_claims,
        )
        return auditor_report
