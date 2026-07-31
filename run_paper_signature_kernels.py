#!/usr/bin/env python3
"""Run forward and backward signature-kernel benchmarks for the paper."""

from paper_runner import run_paper_sweeps


if __name__ == "__main__":
    run_paper_sweeps(
        ("paper_signature_kernel_sweep.yaml",),
        "paper_signature_kernels",
    )
