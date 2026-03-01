"""
SpatioCore Flow v2.0 — Gated Multi-Agent Biological Orchestrator
Entry point for CLI and Streamlit UI modes.
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spatiocoreflow",
        description="SpatioCore Flow: Gated multi-agent single-cell & spatial transcriptomics pipeline.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Streamlit UI alongside the pipeline.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to an AnnData (.h5ad) input file.",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the splash screen (useful for CI or scripted runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Splash screen ─────────────────────────────────────────────────────────
    # Suppressed when --no-banner is set or stdout is not a TTY (piped output).
    show_banner = not args.no_banner and sys.stdout.isatty()
    if show_banner:
        from cli_header import print_header
        print_header(animate=True)

    # ── Mode dispatch ─────────────────────────────────────────────────────────
    if args.ui:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "ui/app.py"],
            check=True,
        )
    elif args.input:
        _run_pipeline(args.input)
    else:
        print(
            "SpatioCore Flow — CLI mode.\n"
            "  --input <file.h5ad>   run the analysis pipeline\n"
            "  --ui                  launch the Streamlit UI\n"
            "  --no-banner           suppress the splash screen\n"
        )


def _run_pipeline(input_path: str) -> None:
    """Bootstrap and execute the full agent pipeline for a single .h5ad file."""
    import os
    from pathlib import Path

    path = Path(input_path)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {path}")

    try:
        import scanpy as sc
        adata = sc.read_h5ad(path)
    except ImportError:
        # Fallback: anndata direct (scanpy not installed)
        import anndata
        adata = anndata.read_h5ad(path)

    from logic.gates import ValidationGate
    from logic.orchestrator import SpatioFlowOrchestrator
    from agents_tasks import (
        get_default_llm,
        make_crewai_curator_runner,
        make_crewai_analyst_runner,
        make_crewai_auditor_runner,
    )

    llm  = get_default_llm()
    gate = ValidationGate(adata=adata)
    orchestrator = SpatioFlowOrchestrator(gate=gate)

    result = orchestrator.run(
        adata=adata,
        curator_runner=make_crewai_curator_runner(llm),
        analyst_runner=make_crewai_analyst_runner(llm),
        auditor_runner=make_crewai_auditor_runner(llm),
    )

    print(
        f"\n[DONE] Pipeline complete.\n"
        f"  Analyst attempts : {result.total_analyst_attempts}\n"
        f"  Final BCS        : {result.final_validator_report.computed_bcs:.4f}\n"
        f"  Traceability     : {result.auditor_report.traceability_coverage:.0%}\n"
        f"  Claims verified  : {result.auditor_report.total_claims}\n"
    )


if __name__ == "__main__":
    main()
