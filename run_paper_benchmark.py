#!/usr/bin/env python3
"""Run every problem-specific paper benchmark."""

from paper_runner import PAPER_SWEEPS, run_paper_sweeps


if __name__ == "__main__":
    run_paper_sweeps(PAPER_SWEEPS, "paper_benchmark")
