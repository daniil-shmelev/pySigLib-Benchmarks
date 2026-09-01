#!/usr/bin/env python3
"""Run all sweeps required for the paper heatmaps."""

from paper_runner import PAPER_SWEEPS, run_paper_sweeps


if __name__ == "__main__":
    run_paper_sweeps(PAPER_SWEEPS, "combined_benchmark")
