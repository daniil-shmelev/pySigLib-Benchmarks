#!/usr/bin/env python3
"""Shared runner for paper benchmark groups."""

import json
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from orchestrator import run_orchestrator


PAPER_SWEEPS = (
    "paper_signatures_sweep.yaml",
    "paper_logsignatures_sweep.yaml",
    "paper_branched_signatures_sweep.yaml",
    "paper_branched_logsignatures_sweep.yaml",
    "paper_signature_kernel_sweep.yaml",
    "paper_polynomial_signature_kernel_sweep.yaml",
)


def run_paper_sweeps(sweep_names: Sequence[str], summary_prefix: str) -> Path:
    started_at = datetime.now().astimezone()
    summary_path = REPO_ROOT / "runs" / (
        f"{summary_prefix}_{started_at.strftime('%Y%m%d_%H%M%S')}.json"
    )
    benchmark_started = time.monotonic()
    result_paths = [
        run_orchestrator(REPO_ROOT / "config" / sweep_name)
        for sweep_name in sweep_names
    ]
    benchmark_wall_time = time.monotonic() - benchmark_started

    summary = {
        "status": "plotting",
        "started_at": started_at.isoformat(),
        "benchmark_completed_at": datetime.now().astimezone().isoformat(),
        "benchmark_wall_time_seconds": benchmark_wall_time,
        "sweep_configs": list(sweep_names),
        "run_directories": [str(path.parent) for path in result_paths],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plot_started = time.monotonic()
    for result_path in result_paths:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "src" / "plotting.py"),
                str(result_path.parent),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    plot_wall_time = time.monotonic() - plot_started

    summary.update({
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "plot_wall_time_seconds": plot_wall_time,
        "total_wall_time_seconds": time.monotonic() - benchmark_started,
    })
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Paper benchmark summary written to: {summary_path}")
    return summary_path
