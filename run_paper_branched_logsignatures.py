#!/usr/bin/env python3
"""Run forward and backward branched-log-signature benchmarks for the paper."""

from paper_runner import run_paper_sweeps


if __name__ == "__main__":
    run_paper_sweeps(
        ("paper_branched_logsignatures_sweep.yaml",),
        "paper_branched_logsignatures",
    )
