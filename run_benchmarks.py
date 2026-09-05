#!/usr/bin/env python3
"""Run all sweeps required for the paper heatmaps."""

import argparse
from pathlib import Path

from paper_runner import PAPER_SWEEPS, run_paper_sweeps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="Existing benchmark directory")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failures while preserving successful results")
    parser.add_argument("--plots-only", action="store_true", help="Regenerate all plots without running benchmarks")
    parser.add_argument("--retry-exclude-library", action="append", default=[], help="Keep failures for this library without retrying (repeatable)")
    args = parser.parse_args()
    run_paper_sweeps(
        PAPER_SWEEPS, "combined_benchmark", resume_dir=args.resume,
        retry_failed=args.retry_failed, plots_only=args.plots_only,
        retry_exclude_libraries=args.retry_exclude_library,
    )
