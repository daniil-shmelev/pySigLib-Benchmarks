#!/usr/bin/env python3
"""Run signature-to-log and direct-BCH benchmarks for the paper."""

from paper_runner import run_paper_sweeps


if __name__ == "__main__":
    run_paper_sweeps(
        (
            "paper_logsignatures_sweep.yaml",
            "paper_bch_logsignatures_sweep.yaml",
        ),
        "paper_logsignatures",
    )
