#!/usr/bin/env python3
"""Run one memory measurement for each paper signature benchmark cell."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from memory_benchmark import run_signature_memory_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the paper signature memory benchmark",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an existing signature_memory run directory",
    )
    args = parser.parse_args()
    run_signature_memory_benchmark(
        REPO_ROOT / "config" / "paper_signatures_sweep.yaml",
        resume_dir=args.resume,
    )
