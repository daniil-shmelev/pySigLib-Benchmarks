"""Tests for dedicated signature memory measurements."""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import memory_benchmark
import orchestrator
from common import BenchmarkAdapter


def _adapter_config(**overrides):
    config = {
        "N": 16,
        "d": 2,
        "m": 2,
        "path_kind": "brownian",
        "operation": "signature",
        "repeats": 3,
        "warmup_iterations": 1,
        "batch_size": 1,
        "backend": "cpu",
        "memory_benchmark": True,
    }
    config.update(overrides)
    return config


def test_memory_mode_runs_one_warmup_and_one_measured_call():
    events = []
    calls = []
    adapter = BenchmarkAdapter(
        _adapter_config(_call_event_callback=events.append)
    )
    _, allocated, samples = adapter.manual_timing_loop(
        lambda: calls.append(True)
    )

    assert calls == [True, True]
    assert allocated == 0
    assert len(samples) == 1
    assert events == [
        {"status": "call_start", "phase": "warmup", "iteration": 0},
        {"status": "call_end", "phase": "warmup", "iteration": 0},
        {"status": "call_start", "phase": "measured", "iteration": 0},
        {"status": "call_end", "phase": "measured", "iteration": 0},
    ]


def test_gpu_memory_summary_records_peak_and_delta():
    metrics = BenchmarkAdapter._summarize_gpu_memory(
        {
            "source": "test_allocator",
            "current_allocated_bytes": 100,
        },
        {
            "source": "test_allocator",
            "peak_allocated_bytes": 350,
            "peak_reserved_bytes": 500,
        },
    )

    assert metrics == {
        "gpu_memory_source": "test_allocator",
        "gpu_baseline_allocated_bytes": 100,
        "gpu_peak_allocated_bytes": 350,
        "gpu_peak_allocated_delta_bytes": 250,
        "gpu_peak_reserved_bytes": 500,
    }


def test_memory_mode_keeps_gpu_metrics_when_call_fails(monkeypatch):
    adapter = BenchmarkAdapter(_adapter_config(backend="gpu"))
    snapshots = iter([
        {
            "source": "test_allocator",
            "current_allocated_bytes": 100,
            "peak_allocated_bytes": 100,
        },
        {
            "source": "test_allocator",
            "current_allocated_bytes": 100,
            "peak_allocated_bytes": 500,
            "peak_reserved_bytes": 600,
        },
    ])
    monkeypatch.setattr(
        adapter,
        "_gpu_memory_snapshot",
        lambda reset_peak=False: next(snapshots),
    )

    def raise_oom():
        raise MemoryError("test OOM")

    with pytest.raises(MemoryError, match="test OOM"):
        adapter.manual_timing_loop(raise_oom)

    assert adapter._memory_metrics["gpu_peak_allocated_bytes"] == 500
    assert adapter._memory_metrics["gpu_peak_allocated_delta_bytes"] == 400


def test_oom_classifier_rejects_unrelated_errors():
    assert memory_benchmark._is_oom(
        RuntimeError("zoom worker disconnected")
    ) is False
    assert memory_benchmark._is_oom(
        RuntimeError("CUDA error: out of memory")
    ) is True


def test_worker_adds_sampled_host_memory_to_result(monkeypatch):
    class FakeProcess:
        @staticmethod
        def poll():
            return None

    worker = orchestrator.PythonAdapterWorker("example", {})
    worker.process = FakeProcess()
    worker._measure_memory = True
    worker._output_queue = orchestrator.queue.Queue()
    for payload in (
        {"status": "call_start", "phase": "measured", "iteration": 0},
        {"status": "call_end", "phase": "measured", "iteration": 0},
        {"status": "result", "result": {"library": "example"}},
    ):
        worker._output_queue.put(
            orchestrator.PYTHON_WORKER_PROTOCOL_PREFIX + json.dumps(payload)
        )
    samples = iter([100, 350])
    monkeypatch.setattr(worker, "_host_memory_bytes", lambda: next(samples))

    response = worker._read_response()

    assert response["result"]["host_baseline_bytes"] == 100
    assert response["result"]["host_peak_bytes"] == 350
    assert response["result"]["host_peak_delta_bytes"] == 250


def test_memory_benchmark_records_oom_and_continues(tmp_path, monkeypatch):
    config_path = tmp_path / "paper_signatures_sweep.yaml"
    config_path.write_text(
        "libraries: [example]\n"
        "Ns: [16]\n"
        "Ds: [2, 3]\n"
        "Ms: [2]\n"
        "operations: [signature]\n"
        "backends: [cpu]\n"
        "path_kind: brownian\n"
        "batch_size: 1\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "libraries_registry.yaml"
    registry_path.write_text(
        "libraries:\n"
        "  example:\n"
        "    type: python\n"
        "    script: adapter.py\n"
        "    backend: cpu\n"
        "    operations: [signature]\n",
        encoding="utf-8",
    )

    class FakeWorker:
        def __init__(self, *args):
            self.last_memory_metrics = {
                "host_memory_source": "process_tree_rss",
                "host_baseline_bytes": 100,
                "host_peak_bytes": 1024,
                "host_peak_delta_bytes": 924,
            }

        def close(self):
            pass

    seen_configs = []

    def fake_run(*args, **kwargs):
        task_config = args[2]
        seen_configs.append(task_config)
        if task_config["d"] == 2:
            raise orchestrator.BenchmarkWorkerOOM("worker received SIGKILL")
        return {
            "N": task_config["N"],
            "d": task_config["d"],
            "m": task_config["m"],
            "batch_size": task_config["batch_size"],
            "seed": task_config["seed"],
            "path_kind": task_config["path_kind"],
            "operation": task_config["operation"],
            "backend": task_config["backend"],
            "language": "python",
            "library": "example",
            "method": "signature",
            "path_type": "array",
            "host_memory_source": "process_tree_rss",
            "host_baseline_bytes": 200,
            "host_peak_bytes": 400,
            "host_peak_delta_bytes": 200,
        }

    monkeypatch.setattr(memory_benchmark, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(memory_benchmark, "PythonAdapterWorker", FakeWorker)
    monkeypatch.setattr(memory_benchmark, "run_python_adapter", fake_run)
    monkeypatch.setattr(memory_benchmark, "write_run_metadata", lambda *args: None)
    monkeypatch.setattr(memory_benchmark, "record_run_completion", lambda *args: None)

    results_path = memory_benchmark.run_signature_memory_benchmark(
        config_path,
        registry_path=registry_path,
    )

    with results_path.open("r", encoding="utf-8", newline="") as output:
        results = list(csv.DictReader(output))
    with (results_path.parent / "failed_tasks.csv").open(
        "r", encoding="utf-8", newline=""
    ) as output:
        failures = list(csv.DictReader(output))

    assert len(results) == 1
    assert results[0]["memory_status"] == "ok"
    assert results[0]["host_peak_bytes"] == "400"
    assert len(failures) == 1
    assert failures[0]["memory_status"] == "oom"
    assert failures[0]["host_peak_bytes"] == "1024"
    assert len(seen_configs) == 2
    assert all(config["repeats"] == 1 for config in seen_configs)
    assert all(config["warmup_iterations"] == 1 for config in seen_configs)
