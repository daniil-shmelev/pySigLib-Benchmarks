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
    "paper_bch_logsignatures_sweep.yaml",
    "paper_branched_signatures_sweep.yaml",
    "paper_branched_logsignatures_sweep.yaml",
    "paper_signature_kernel_sweep.yaml",
    "paper_polynomial_signature_kernel_sweep.yaml",
)


def run_paper_sweeps(
    sweep_names: Sequence[str],
    summary_prefix: str,
    *,
    resume_dir: Path | None = None,
    retry_failed: bool = False,
    plots_only: bool = False,
    retry_exclude_libraries: Sequence[str] = (),
) -> Path:
    """Keep one benchmark's plots and named sweep data in a single directory."""
    if (retry_failed or plots_only) and resume_dir is None:
        raise ValueError("Retrying or plotting an existing benchmark requires --resume")
    if retry_failed and plots_only:
        raise ValueError("--retry-failed and --plots-only cannot be combined")
    if retry_exclude_libraries and not retry_failed:
        raise ValueError("--retry-exclude-library requires --retry-failed")
    if resume_dir is None:
        started_at = datetime.now().astimezone()
        group_dir = REPO_ROOT / "runs" / f"{summary_prefix}_{started_at:%Y%m%d_%H%M%S}"
        group_dir.mkdir(parents=True, exist_ok=False)
        summary = {
            "started_at": started_at.isoformat(),
            "sweep_configs": list(sweep_names),
            "run_directories": [
                f"data/{Path(name).stem.removeprefix('paper_').removesuffix('_sweep')}"
                for name in sweep_names
            ],
        }
    else:
        group_dir = resume_dir.resolve()
        summary = json.loads((group_dir / "summary.json").read_text(encoding="utf-8"))
    summary_path = group_dir / "summary.json"

    def save_summary():
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    if not plots_only:
        summary["status"] = "benchmarking"
        save_summary()
        benchmark_started = time.monotonic()
        for sweep_name, relative_dir in zip(summary["sweep_configs"], summary["run_directories"], strict=True):
            run_dir = group_dir / relative_dir
            if run_dir.exists():
                run_orchestrator(
                    resume_dir=run_dir, retry_failed=retry_failed,
                    retry_exclude_libraries=retry_exclude_libraries,
                )
            else:
                run_orchestrator(REPO_ROOT / "config" / sweep_name, output_dir=run_dir)
        summary["benchmark_wall_time_seconds"] = summary.get("benchmark_wall_time_seconds", 0) + time.monotonic() - benchmark_started
        summary["benchmark_completed_at"] = datetime.now().astimezone().isoformat()

    summary["status"] = "plotting"
    save_summary()

    plot_started = time.monotonic()
    for relative_dir in summary["run_directories"]:
        run_dir = group_dir / relative_dir
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "src" / "plotting.py"),
                str(run_dir),
                "--plot-dir", str(group_dir / "plots"),
                "--filename-prefix", run_dir.name + "_",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    plot_wall_time = time.monotonic() - plot_started

    summary.update({
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "plot_wall_time_seconds": summary.get("plot_wall_time_seconds", 0) + plot_wall_time,
    })
    summary["total_wall_time_seconds"] = summary.get("benchmark_wall_time_seconds", 0) + summary["plot_wall_time_seconds"]
    save_summary()
    print(f"Benchmark directory: {group_dir}")
    print(f"All plots: {group_dir / 'plots'}")
    return summary_path
