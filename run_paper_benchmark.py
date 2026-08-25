#!/usr/bin/env python3
"""Run every problem-specific paper benchmark."""

from paper_runner import run_paper_sweeps


if __name__ == "__main__":
    run_paper_sweeps(
        (
            "paper_signatures_sweep.yaml",
            "paper_logsignatures_sweep.yaml",
            "paper_branched_signatures_sweep.yaml",
            "paper_branched_logsignatures_sweep.yaml",
            "paper_signature_kernel_sweep.yaml",
        ),
        "paper_benchmark",
    )
