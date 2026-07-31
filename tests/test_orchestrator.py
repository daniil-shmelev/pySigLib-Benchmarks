"""Tests for benchmark orchestration and resume semantics."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import orchestrator


def test_run_metadata_captures_hardware_and_dirty_patch(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text("Ns: [4]\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")

    responses = {
        ("git", "status", "--porcelain"): " M benchmark.py",
        ("git", "diff", "--binary", "HEAD"): "diff --git a/x b/x",
        ("git", "rev-parse", "HEAD"): "abc123",
        ("lscpu",): "Model name: Test CPU",
        ("free", "-b"): "Mem: 1000",
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
            "--format=csv,noheader",
        ): "Test GPU, GPU-123, 1.0, 1000 MiB, 100 W",
        ("nvidia-smi", "-q"): "GPU full details",
    }
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        orchestrator,
        "_command_output",
        lambda command: responses.get(tuple(command)),
    )

    orchestrator.write_run_metadata(run_dir, sweep_path)

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["git_commit"] == "abc123"
    assert metadata["git_dirty"] is True
    assert metadata["cpu"] == "Model name: Test CPU"
    assert metadata["memory"] == "Mem: 1000"
    assert metadata["gpu"].startswith("Test GPU")
    assert metadata["nvidia_smi"] == "GPU full details"
    assert (run_dir / "git_status.txt").read_text(encoding="utf-8").endswith("\n")
    assert (run_dir / "git_diff.patch").read_text(encoding="utf-8").endswith("\n")
    assert (run_dir / "uv.lock").read_text(encoding="utf-8") == "lock"


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


def test_unsupported_task_is_recorded_and_not_retried(tmp_path, monkeypatch):
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
            "  limited:",
            "    type: python",
            "    script: limited.py",
            "    operations: [signature]",
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "REGISTRY_CONFIG", registry_path)
    calls = []

    def skip_first_task(library_name, library_config, task_config, **kwargs):
        calls.append(task_config["N"])
        if task_config["N"] == 4:
            raise subprocess.CalledProcessError(
                1,
                ["limited-adapter"],
                stderr=orchestrator.UNSUPPORTED_ERROR_MARKERS[0],
            )
        return _result(task_config)

    monkeypatch.setattr(orchestrator, "run_python_adapter", skip_first_task)
    orchestrator.run_orchestrator(sweep_path)

    run_dir = next((tmp_path / "runs").iterdir())
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as results_file:
        result_rows = list(csv.DictReader(results_file))
    with (
        run_dir / orchestrator.SKIPPED_TASKS_FILE
    ).open(newline="", encoding="utf-8") as skipped_file:
        skipped_rows = list(csv.DictReader(skipped_file))

    assert calls == [4, 8]
    assert [int(row["N"]) for row in result_rows] == [8]
    assert [int(row["N"]) for row in skipped_rows] == [4]
    assert skipped_rows[0]["reason"] == orchestrator.UNSUPPORTED_ERROR_MARKERS[0]
    assert len((run_dir / orchestrator.COMPLETED_TASKS_FILE).read_text().splitlines()) == 2

    resumed_calls = []
    monkeypatch.setattr(
        orchestrator,
        "run_python_adapter",
        lambda *args, **kwargs: resumed_calls.append(args),
    )
    orchestrator.run_orchestrator(resume_dir=run_dir)
    assert resumed_calls == []
