"""
cli_header.py — SpatioCore Flow v2.0 Splash Screen
===================================================
Terminal branding for the CLI entry point.

Call ``print_header()`` once at startup.
Use ``--no-banner`` in main.py to suppress for non-interactive / CI runs.
"""

from __future__ import annotations

import os
import sys
import time

# ── ANSI colour codes ─────────────────────────────────────────────────────────
GREEN     = "\033[92m"
BLUE      = "\033[94m"
CYAN      = "\033[96m"
BOLD      = "\033[1m"
UNDERLINE = "\033[4m"
RESET     = "\033[0m"

# ── ASCII art banner ──────────────────────────────────────────────────────────
HEADER_ART = rf"""
{CYAN}
   ____              _   _        ____                ______ _
  / ___| _ __   __ _| |_(_) ___  / ___|___  _ __ ___  |  ____| |
  \___ \| '_ \ / _` | __| |/ _ \| |   / _ \| '__/ _ \ | |__  | |
   ___) | |_) | (_| | |_| | (_) | |__| (_) | | |  __/ |  __| | |___
  |____/| .__/ \__,_|\__|_|\___/ \____\___/|_|  \___| |_|    |_____|
        |_|

{RESET}"""

# ── Agent flow sub-header ─────────────────────────────────────────────────────
SUB_HEADER = f"""
{BOLD}{UNDERLINE}Multimodal Multi-Agent Bioinformatics Orchestrator{RESET}

{BOLD}[AGENT FLOW]{RESET}
1. Curator    ({BLUE}Modality Map{RESET})  ────>
2. Analyst    ({BLUE}Model Execute{RESET}) ────>
3. Synthesizer({BLUE}Spatial Fusion{RESET})────>
4. Auditor    ({GREEN}Verify{RESET})       ────>
5. Guardrail  ({GREEN}Clinical Check{RESET})────>
{BOLD}Result{RESET}
"""

# ── System info box ───────────────────────────────────────────────────────────
SYSTEM_INFO = f"""
{BLUE}[SYSTEM INFO]{RESET}
Orchestration: CrewAI (Sequential)
Core Models:   scGPT, Geneformer, scATAC-seq, Image-ViT
Target:        Spatial Biology & Single-Cell Genomics
"""


# ── Rendering helpers ─────────────────────────────────────────────────────────

def slow_print(text: str, delay: float = 0.005) -> None:
    """Stream text character-by-character for dramatic effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)


def loading_animation(duration: float = 2.0) -> None:
    """Braille spinner that resolves to an 'Online' status message."""
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    end_time = time.time() + duration
    idx = 0
    print(f"\n{GREEN}Status:{RESET}", end=" ")
    while time.time() < end_time:
        sys.stdout.write(
            f"\r{GREEN}Status:{RESET} {frames[idx]} Initializing SpatioCore Nexus..."
        )
        sys.stdout.flush()
        time.sleep(0.08)
        idx = (idx + 1) % len(frames)
    sys.stdout.write(
        f"\r{GREEN}Status:{RESET}  {BOLD}SpatioCore Nexus is Online.{RESET}   \n\n"
    )
    sys.stdout.flush()


# ── Public entry point ────────────────────────────────────────────────────────

def print_header(animate: bool = True) -> None:
    """
    Clear the terminal and render the full SpatioCore Flow splash screen.

    Parameters
    ----------
    animate :
        When ``False`` the spinner is skipped (useful for ``--no-banner``
        fast-path or non-TTY environments).
    """
    os.system("cls" if os.name == "nt" else "clear")
    print(HEADER_ART)
    slow_print(SUB_HEADER, delay=0.001)
    print(SYSTEM_INFO)
    if animate:
        loading_animation()


if __name__ == "__main__":
    print_header()
