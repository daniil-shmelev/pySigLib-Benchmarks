"""One user-facing directory for an entire paper benchmark."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paper_runner


def test_grouped_run_resume_and_plots_only(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_runner, "REPO_ROOT", tmp_path)
    calls = []
    plots = []

    def fake_orchestrator(config_path=None, *, output_dir=None, resume_dir=None, retry_failed=False, retry_exclude_libraries=()):
        calls.append((config_path, output_dir, resume_dir, retry_failed, retry_exclude_libraries))
        run_dir = output_dir or resume_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        result = run_dir / "results.csv"
        result.touch()
        return result

    monkeypatch.setattr(paper_runner, "run_orchestrator", fake_orchestrator)
    monkeypatch.setattr(paper_runner.subprocess, "run", lambda command, **kwargs: plots.append(command))
    summary_path = paper_runner.run_paper_sweeps(paper_runner.PAPER_SWEEPS, "combined_benchmark")
    group = summary_path.parent
    summary = json.loads(summary_path.read_text())
    assert list((tmp_path / "runs").iterdir()) == [group]
    assert summary["status"] == "complete"
    assert summary["run_directories"][:3] == ["data/signatures", "data/logsignatures", "data/bch_logsignatures"]
    assert len(calls) == len(plots) == 7
    assert all(call[1].parent == group / "data" for call in calls)
    assert all(command[command.index("--plot-dir") + 1] == str(group / "plots") for command in plots)
    prefixes = [command[command.index("--filename-prefix") + 1] for command in plots]
    assert len(set(prefixes)) == 7

    calls.clear()
    plots.clear()
    # Resume the saved group, not the current default list of sweeps.
    paper_runner.run_paper_sweeps([], "unused", resume_dir=group, retry_failed=True, retry_exclude_libraries=["log-signatures-pytorch"])
    assert len(calls) == len(plots) == 7
    assert all(call[0] is None and call[1] is None and call[2].parent == group / "data" and call[3] for call in calls)
    assert all(call[4] == ["log-signatures-pytorch"] for call in calls)

    calls.clear()
    plots.clear()
    paper_runner.run_paper_sweeps([], "unused", resume_dir=group, plots_only=True)
    assert calls == []
    assert len(plots) == 7


def test_group_is_recorded_before_first_benchmark_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_runner, "REPO_ROOT", tmp_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(paper_runner, "run_orchestrator", interrupt)
    with pytest.raises(KeyboardInterrupt):
        paper_runner.run_paper_sweeps(paper_runner.PAPER_SWEEPS, "combined_benchmark")
    summary_path, = (tmp_path / "runs").glob("*/summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "benchmarking"
    assert len(summary["run_directories"]) == 7


@pytest.mark.parametrize("kwargs", [{"retry_failed": True}, {"plots_only": True}])
def test_resume_options_require_existing_group(kwargs):
    with pytest.raises(ValueError, match="requires --resume"):
        paper_runner.run_paper_sweeps([], "unused", **kwargs)
