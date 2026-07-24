"""Tests for benchmark orchestration and resume semantics."""

import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import orchestrator


def test_python_adapter_expands_repo_root_in_environment(tmp_path, monkeypatch):
    script = tmp_path / "adapter.py"
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "ensure_compatibility_wheel", lambda *args: None)
    captured = {}

    def fake_run(command, *, env=None):
        captured.update(env)
        return '{"library": "example"}\n'

    monkeypatch.setattr(orchestrator, "run_subprocess_capture", fake_run)
    result = orchestrator.run_python_adapter(
        "example",
        {"script": "adapter.py"},
        {},
        env_overrides={"EXAMPLE_PATH": "{repo_root}/dependency"},
    )

    assert result == {"library": "example"}
    assert captured["EXAMPLE_PATH"] == str(tmp_path / "dependency")


def test_compatibility_wheel_is_installed_once(tmp_path, monkeypatch):
    packages_dir = tmp_path / "adapter-packages"
    monkeypatch.setattr(
        orchestrator,
        "COMPATIBILITY_PACKAGES_DIR",
        packages_dir,
    )
    calls = []

    def fake_run(command, *, env=None):
        calls.append(command)
        target = Path(command[command.index("--target") + 1])
        (target / "log_signatures_pytorch").mkdir()
        return ""

    monkeypatch.setattr(orchestrator, "run_subprocess_capture", fake_run)
    url = f"https://example.test/package.whl#sha256={'a' * 64}"
    library_config = {"compatibility_wheel": {"url": url}}

    first = orchestrator.ensure_compatibility_wheel(
        "example",
        library_config,
        ["example-extra"],
        {},
    )
    second = orchestrator.ensure_compatibility_wheel(
        "example",
        library_config,
        ["example-extra"],
        {},
    )

    assert first == second
    assert len(calls) == 1
    assert "--ignore-requires-python" in calls[0]
    assert "--no-deps" in calls[0]
    assert (first / ".complete").read_text(encoding="utf-8") == url


def _result(task_config):
    return {
        **task_config,
        "language": "python",
        "library": "test-library",
        "method": "signature",
        "path_type": "ndarray",
        "t_ms": 1.0,
        "t_ms_mean": 1.0,
        "t_ms_std": 0.0,
        "samples_ms": [1.0],
        "alloc_bytes": 0,
    }


def test_failed_run_appends_results_and_resumes_missing_tasks(tmp_path, monkeypatch):
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        "\n".join([
            "path_kind: sin",
            "seed: 1",
            "Ns: [4, 8]",
            "Ds: [2]",
            "Ms: [2]",
            "operations: [signature]",
            "batch_size: 2",
            "repeats: 2",
            "runs_dir: runs",
        ]),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "\n".join([
            "libraries:",
            "  broken:",
            "    type: python",
            "    script: broken.py",
            "    operations: [signature]",
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "REGISTRY_CONFIG", registry_path)

    first_calls = []

    def fail_second_task(library_name, library_config, task_config, **kwargs):
        first_calls.append(task_config["N"])
        if task_config["N"] == 8:
            raise RuntimeError("adapter failed")
        return _result(task_config)

    monkeypatch.setattr(orchestrator, "run_python_adapter", fail_second_task)

    with pytest.raises(RuntimeError, match="adapter failed"):
        orchestrator.run_orchestrator(sweep_path)

    run_dir = next((tmp_path / "runs").iterdir())
    csv_path = run_dir / "results.csv"
    with csv_path.open(newline="", encoding="utf-8") as results_file:
        first_rows = list(csv.DictReader(results_file))
    assert first_calls == [4, 8]
    assert [int(row["N"]) for row in first_rows] == [4]

    resumed_calls = []

    def finish_run(library_name, library_config, task_config, **kwargs):
        resumed_calls.append(task_config["N"])
        return _result(task_config)

    monkeypatch.setattr(orchestrator, "run_python_adapter", finish_run)
    orchestrator.run_orchestrator(resume_dir=run_dir)

    with csv_path.open(newline="", encoding="utf-8") as results_file:
        final_rows = list(csv.DictReader(results_file))
    assert resumed_calls == [8]
    assert [int(row["N"]) for row in final_rows] == [4, 8]
    assert len((run_dir / orchestrator.COMPLETED_TASKS_FILE).read_text().splitlines()) == 2
