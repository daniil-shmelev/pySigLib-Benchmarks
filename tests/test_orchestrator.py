"""Tests for benchmark orchestration and resume semantics."""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import orchestrator


def test_task_timeout_does_not_change_resume_identity():
    config = {"N": 1000, "d": 16, "m": 3, "operation": "logsignature"}
    assert orchestrator.make_task_id("example", "gpu", config) == orchestrator.make_task_id(
        "example", "gpu", {**config, "task_timeout_seconds": 120},
    )


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


def test_python_adapter_expands_cpu_count_in_environment(tmp_path, monkeypatch):
    script = tmp_path / "adapter.py"
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(orchestrator, "ensure_compatibility_wheel", lambda *args: None)

    _, _, env = orchestrator.prepare_python_adapter_process(
        "example",
        {"script": "adapter.py"},
        {"OMP_NUM_THREADS": "{cpu_count}"},
    )

    assert env["OMP_NUM_THREADS"] == "12"


def test_rust_adapter_uses_cached_toolchain(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for key in ("CARGO_HOME", "RUSTUP_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    cargo_dir = tmp_path / ".cache" / "puccinialin" / "cargo"
    monkeypatch.setattr(
        orchestrator.shutil, "which",
        lambda command, *, path: str(cargo_dir / "bin" / command)
        if path == str(cargo_dir / "bin") else None,
    )

    _, _, env = orchestrator.prepare_python_adapter_process(
        "signature-rs", {"script": "adapter.py"}, None,
    )

    assert env["PATH"] == str(cargo_dir / "bin") + orchestrator.os.pathsep + "/usr/bin"
    assert env["CARGO_HOME"] == str(cargo_dir)
    assert env["RUSTUP_HOME"] == str(cargo_dir.parent / "rustup")


def test_rust_discovery_preserves_toolchain_already_on_path(monkeypatch):
    env = {"PATH": "/custom/bin", "CARGO_HOME": "/custom/cargo", "RUSTUP_HOME": "/custom/rustup"}
    original = env.copy()
    monkeypatch.setattr(orchestrator.shutil, "which", lambda *args, **kwargs: "/custom/bin/cargo")
    orchestrator.discover_rust_toolchain(env)
    assert env == original


def test_rust_discovery_without_installation_leaves_environment_unchanged(monkeypatch):
    env = {"PATH": "/usr/bin"}
    monkeypatch.setattr(orchestrator.shutil, "which", lambda *args, **kwargs: None)
    orchestrator.discover_rust_toolchain(env)
    assert env == {"PATH": "/usr/bin"}


def test_build_task_config_applies_only_matching_backend_config():
    sweep = {
        "task_timeout_seconds": 120,
        "library_configs": {
            "example": {
                "shared": "value",
                "backend_configs": {
                    "cpu": {"n_jobs": -1},
                },
            },
        },
    }
    kwargs = {
        "N": 10,
        "d": 2,
        "m": 3,
        "path_kind": "brownian",
        "operation": "signature",
        "repeats": 3,
        "batch_size": 4,
        "seed": 1,
    }

    cpu = orchestrator.build_task_config(
        sweep,
        "example",
        backend_name="cpu",
        **kwargs,
    )
    gpu = orchestrator.build_task_config(
        sweep,
        "example",
        backend_name="gpu",
        **kwargs,
    )

    assert cpu["shared"] == "value"
    assert cpu["task_timeout_seconds"] == 120
    assert cpu["n_jobs"] == -1
    assert "backend_configs" not in cpu
    assert gpu["shared"] == "value"
    assert "n_jobs" not in gpu


def test_python_worker_reuses_process_and_recovers_after_task_error(tmp_path):
    adapter_path = tmp_path / "adapter.py"
    adapter_path.write_text(
        "\n".join([
            "import os",
            "import sys",
            "common_loaded_before_adapter = 'common' in sys.modules",
            "from common import BenchmarkAdapter",
            "task_count = 0",
            "class TestAdapter(BenchmarkAdapter):",
            "    def _run_benchmark(self):",
            "        global task_count",
            "        task_count += 1",
            "        if self.N == 4:",
            "            print('native diagnostic', end='', flush=True)",
            "            raise RuntimeError('expected task failure')",
            "        return {'pid': os.getpid(), 'task_count': task_count, 'common_loaded_before_adapter': common_loaded_before_adapter}",
        ]),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(orchestrator.PYTHON_ADAPTER_WORKER), str(adapter_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )

    def response():
        while True:
            line = process.stdout.readline()
            assert line != ""
            protocol_index = line.find(orchestrator.PYTHON_WORKER_PROTOCOL_PREFIX)
            if protocol_index >= 0:
                return json.loads(
                    line[
                        protocol_index
                        + len(orchestrator.PYTHON_WORKER_PROTOCOL_PREFIX):
                    ]
                )

    assert response()["status"] == "ready"
    base_config = {
        "d": 2,
        "m": 2,
        "path_kind": "sin",
        "operation": "signature",
        "repeats": 1,
        "batch_size": 1,
    }
    process.stdin.write(json.dumps({**base_config, "N": 4}) + "\n")
    process.stdin.flush()
    assert response()["status"] == "error"

    process.stdin.write(json.dumps({**base_config, "N": 8}) + "\n")
    process.stdin.flush()
    result = response()
    assert result["status"] == "result"
    assert result["result"]["pid"] == process.pid
    assert result["result"]["task_count"] == 2
    assert result["result"]["common_loaded_before_adapter"] is False

    process.stdin.close()
    assert process.wait(timeout=5) == 0


def test_worker_parser_accepts_protocol_after_unterminated_diagnostic():
    response = {"status": "result", "result": {"value": 1}}
    stdout = io.StringIO(
        "native diagnostic"
        + orchestrator.PYTHON_WORKER_PROTOCOL_PREFIX
        + json.dumps(response)
        + "\n"
    )

    class FakeProcess:
        def __init__(self):
            self.stdout = stdout

        @staticmethod
        def poll():
            return None

    worker = orchestrator.PythonAdapterWorker("example", {})
    worker.process = FakeProcess()
    worker._output_queue = orchestrator.queue.Queue()
    worker._output_queue.put(stdout.readline())

    assert worker._read_response() == response
    assert worker.diagnostics == ["native diagnostic"]


def test_worker_times_out_an_active_kernel_call(monkeypatch):
    class FakeProcess:
        @staticmethod
        def poll():
            return None

    worker = orchestrator.PythonAdapterWorker("example", {})
    worker.process = FakeProcess()
    worker._output_queue = orchestrator.queue.Queue()
    worker._output_queue.put(
        orchestrator.PYTHON_WORKER_PROTOCOL_PREFIX
        + json.dumps({
            "status": "call_start",
            "phase": "measured",
            "iteration": 0,
            "timeout_seconds": 0.01,
        })
        + "\n"
    )
    killed = []
    monkeypatch.setattr(
        worker,
        "_kill_process_tree",
        lambda: killed.append(True),
    )

    with pytest.raises(orchestrator.BenchmarkCallTimeout):
        worker._read_response()

    assert killed == [True]


@pytest.mark.parametrize("events", [[], ["call_start", "call_end"], ["diagnostic"]])
def test_task_timeout_includes_setup_and_is_not_reset_by_events(monkeypatch, events):
    worker = orchestrator.PythonAdapterWorker("example", {})
    worker.process = object()
    worker._output_queue = orchestrator.queue.Queue()
    for status in events:
        worker._output_queue.put(
            orchestrator.PYTHON_WORKER_PROTOCOL_PREFIX + json.dumps({"status": status})
            if status != "diagnostic" else "compiling..."
        )
    killed = []
    monkeypatch.setattr(worker, "_kill_process_tree", lambda: killed.append(True))
    with pytest.raises(orchestrator.BenchmarkTaskTimeout, match="including setup and compilation"):
        worker._read_response(task_timeout_seconds=0.01)
    assert killed == [True]


def test_killing_systemd_worker_targets_service_not_only_launcher(monkeypatch):
    worker = orchestrator.PythonAdapterWorker("example", {})
    worker.process = type("Process", (), {"pid": 123, "wait": lambda *args, **kwargs: 0})()
    worker._systemd_unit = "sig-benchmark-test.service"
    commands = []
    monkeypatch.setattr(orchestrator.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(orchestrator.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(orchestrator.os, "killpg", lambda *args: None)
    worker._kill_process_tree()
    assert commands == [["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGKILL", worker._systemd_unit]]


def test_worker_memory_limit_kills_only_worker_and_reports_oom(monkeypatch):
    class FakeProcess:
        pid = 123

        @staticmethod
        def poll():
            return None

    worker = orchestrator.PythonAdapterWorker("example", {})
    worker.process = FakeProcess()
    worker._output_queue = orchestrator.queue.Queue()
    worker._memory_limit_bytes = 1024
    monkeypatch.setattr(worker, "_worker_resident_bytes", lambda: 2048)
    killed = []
    monkeypatch.setattr(
        worker,
        "_kill_process_tree",
        lambda: killed.append(True),
    )

    with pytest.raises(orchestrator.BenchmarkWorkerOOM, match="exceeding"):
        worker._read_response()

    assert killed == [True]


def test_sigkill_without_oom_evidence_is_runtime_error():
    class FakeProcess:
        @staticmethod
        def poll():
            return -orchestrator.WORKER_SIGKILL

    worker = orchestrator.PythonAdapterWorker("example", {})
    worker.process = FakeProcess()
    worker._output_queue = orchestrator.queue.Queue()
    worker._output_queue.put(None)

    with pytest.raises(RuntimeError, match="SIGKILL"):
        worker._read_response()


def test_cgroup_oom_diagnostic_is_reported_as_worker_oom():
    class FakeProcess:
        @staticmethod
        def poll():
            return 1

    worker = orchestrator.PythonAdapterWorker("example", {})
    worker.process = FakeProcess()
    worker.diagnostics = ["Finished with result: oom-kill"]
    worker._output_queue = orchestrator.queue.Queue()
    worker._output_queue.put(None)

    with pytest.raises(orchestrator.BenchmarkWorkerOOM, match="due to OOM"):
        worker._read_response()


def test_oom_classification_requires_explicit_memory_evidence():
    assert orchestrator.is_oom_failure(
        "RuntimeError",
        "zoom worker disconnected",
    ) is False
    assert orchestrator.is_oom_failure(
        "RuntimeError",
        "random kernel error",
    ) is False
    assert orchestrator.is_oom_failure(
        "OutOfMemoryError",
        "CUDA allocation failed",
    ) is True
    assert orchestrator.is_oom_failure(
        "RuntimeError",
        "CUDA error: out of memory",
    ) is True


def test_python_worker_classifies_adapter_oom_response(monkeypatch):
    worker = orchestrator.PythonAdapterWorker("example", {})
    worker.process = type(
        "FakeProcess",
        (),
        {
            "stdin": type(
                "FakeStdin",
                (),
                {"write": lambda self, value: None, "flush": lambda self: None},
            )(),
        },
    )()
    worker.scope_key = "scope"
    monkeypatch.setattr(worker, "_task_scope_key", lambda config: "scope")
    monkeypatch.setattr(worker, "_start", lambda: None)
    monkeypatch.setattr(
        worker,
        "_read_response",
        lambda: {
            "status": "error",
            "error_type": "JaxRuntimeError",
            "error": "RESOURCE_EXHAUSTED: Out of memory while allocating",
            "traceback": "",
        },
    )

    with pytest.raises(orchestrator.BenchmarkWorkerOOM, match="Out of memory"):
        worker.run({"memory_sample_interval_seconds": 0.005})


def test_python_worker_does_not_classify_random_adapter_error_as_oom(
    monkeypatch,
):
    worker = orchestrator.PythonAdapterWorker("example", {})
    worker.process = type(
        "FakeProcess",
        (),
        {
            "stdin": type(
                "FakeStdin",
                (),
                {"write": lambda self, value: None, "flush": lambda self: None},
            )(),
        },
    )()
    worker.scope_key = "scope"
    monkeypatch.setattr(worker, "_task_scope_key", lambda config: "scope")
    monkeypatch.setattr(worker, "_start", lambda: None)
    monkeypatch.setattr(
        worker,
        "_read_response",
        lambda: {
            "status": "error",
            "error_type": "RuntimeError",
            "error": "random kernel error",
            "traceback": "",
        },
    )

    with pytest.raises(RuntimeError, match="random kernel error"):
        worker.run({"memory_sample_interval_seconds": 0.005})


def test_worker_does_not_repeat_failed_startup(monkeypatch):
    worker = orchestrator.PythonAdapterWorker("example", {})
    starts = []

    def fail_start():
        starts.append(True)
        raise RuntimeError("No CMAKE_CUDA_COMPILER could be found")

    monkeypatch.setattr(worker, "_start", fail_start)
    for _ in range(2):
        with pytest.raises(orchestrator.BenchmarkWorkerStartupError, match="CMAKE_CUDA_COMPILER"):
            worker.run({})
        worker.close()
    assert len(starts) == 1


def test_cgroup_worker_forwards_changed_and_inherited_environment(monkeypatch):
    captured = {}
    script_path = Path("adapter.py")
    inherited = {
        "PATH": "/usr/local/cuda/bin:/usr/bin",
        "CUDACXX": "/usr/local/cuda/bin/nvcc",
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVCC_APPEND_FLAGS": "--std=c++20",
        "CARGO_HOME": "/home/test/.cache/puccinialin/cargo",
        "RUSTUP_HOME": "/home/test/.cache/puccinialin/rustup",
        "RUSTUP_TOOLCHAIN": "stable",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(
        "UV_PROJECT_ENVIRONMENT",
        "/home/test/benchmark-venv",
    )

    class FakeProcess:
        stdout = io.StringIO()

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(orchestrator.os, "name", "posix")
    monkeypatch.setattr(
        orchestrator,
        "prepare_python_adapter_process",
        lambda *args: (
            script_path,
            ["uv", "run"],
            {
                **inherited,
                "PYTHONPATH": "/tmp/adapter-package",
                "UV": "/opt/uv/bin/uv",
                "UV_PROJECT_ENVIRONMENT": "/home/test/benchmark-venv",
            },
        ),
    )
    monkeypatch.setattr(
        orchestrator.shutil,
        "which",
        lambda command, **kwargs: f"/usr/bin/{command}",
    )
    monkeypatch.setattr(
        orchestrator,
        "systemd_user_manager_available",
        lambda env: True,
    )
    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)

    worker = orchestrator.PythonAdapterWorker("example", {})
    worker._memory_limit_bytes = 1024
    monkeypatch.setattr(worker, "_read_response", lambda: {"status": "ready"})
    worker._start()

    assert "--setenv=PYTHONPATH=/tmp/adapter-package" in captured["command"]
    for key, value in inherited.items():
        assert f"--setenv={key}={value}" in captured["command"]
    assert (
        "--setenv=UV_PROJECT_ENVIRONMENT=/home/test/benchmark-venv"
        in captured["command"]
    )
    assert "/opt/uv/bin/uv" in captured["command"]
    assert captured["env"]["PYTHONPATH"] == "/tmp/adapter-package"


def test_worker_without_user_systemd_starts_directly(monkeypatch):
    captured = {}
    script_path = Path("adapter.py")

    class FakeProcess:
        stdout = io.StringIO()

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(orchestrator.os, "name", "posix")
    monkeypatch.setattr(
        orchestrator,
        "prepare_python_adapter_process",
        lambda *args: (
            script_path,
            ["uv", "run"],
            {"PATH": "/usr/bin", "UV": "/opt/uv/bin/uv"},
        ),
    )
    monkeypatch.setattr(
        orchestrator.shutil,
        "which",
        lambda command, **kwargs: f"/usr/bin/{command}",
    )
    monkeypatch.setattr(
        orchestrator,
        "systemd_user_manager_available",
        lambda env: False,
    )
    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)

    worker = orchestrator.PythonAdapterWorker("example", {})
    worker._memory_limit_bytes = 1024
    monkeypatch.setattr(worker, "_read_response", lambda: {"status": "ready"})
    worker._start()

    assert captured["command"][0] == "/opt/uv/bin/uv"
    assert "systemd-run" not in captured["command"]


def test_worker_memory_limit_does_not_change_task_identity():
    config = {"N": 1000, "d": 4, "m": 3, "operation": "signature"}

    without_limit = orchestrator.make_task_id("example", "cpu", config)
    with_limit = orchestrator.make_task_id(
        "example",
        "cpu",
        {**config, "worker_memory_limit_gb": 24},
    )

    assert with_limit == without_limit


def test_shape_scoped_worker_reuses_only_matching_shapes():
    worker = orchestrator.PythonAdapterWorker(
        "example",
        {"worker_scope": "shape"},
    )
    base = {
        "N": 256,
        "d": 4,
        "m": 2,
        "batch_size": 32,
        "path_kind": "fbm",
        "seed": 7,
    }

    assert worker._task_scope_key(base) == worker._task_scope_key({
        **base,
        "m": 3,
        "operation": "logsignature",
    })
    assert worker._task_scope_key(base) != worker._task_scope_key({
        **base,
        "d": 8,
    })


def test_shape_restart_preserves_worker_memory_limit(monkeypatch):
    worker = orchestrator.PythonAdapterWorker(
        "example",
        {"worker_scope": "shape"},
    )
    worker.process = object()
    worker.scope_key = ("shape", 16, 2, 1, "brownian", 0)
    observed_limits = []

    def fake_close():
        worker.process = None
        worker.scope_key = None
        worker._memory_limit_bytes = None

    class FakeStdin:
        def write(self, value):
            pass

        def flush(self):
            pass

    class FakeProcess:
        stdin = FakeStdin()

    def fake_start():
        observed_limits.append(worker._memory_limit_bytes)
        worker.process = FakeProcess()

    monkeypatch.setattr(worker, "close", fake_close)
    monkeypatch.setattr(worker, "_start", fake_start)
    monkeypatch.setattr(
        worker,
        "_read_response",
        lambda: {"status": "result", "result": {}},
    )

    worker.run({
        "N": 32,
        "d": 4,
        "m": 2,
        "batch_size": 1,
        "path_kind": "brownian",
        "seed": 0,
        "worker_memory_limit_gb": 16,
    })

    assert observed_limits == [16 * 1024 ** 3]


def test_orchestrator_passes_one_worker_to_all_backend_tasks(tmp_path, monkeypatch):
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        "\n".join([
            "path_kind: sin",
            "Ns: [4, 8]",
            "Ds: [2]",
            "Ms: [2]",
            "operations: [signature]",
            "batch_size: 1",
            "repeats: 1",
            "runs_dir: runs",
        ]),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "\n".join([
            "libraries:",
            "  example:",
            "    type: python",
            "    script: adapter.py",
            "    operations: [signature]",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    workers = []

    def fake_adapter(library_name, library_config, task_config, **kwargs):
        workers.append(kwargs["worker"])
        return _result(task_config)

    monkeypatch.setattr(orchestrator, "run_python_adapter", fake_adapter)
    output_dir = tmp_path / "group" / "data" / "signatures"
    result = orchestrator.run_orchestrator(sweep_path, registry_path=registry_path, output_dir=output_dir)

    assert len(workers) == 2
    assert workers[0] is workers[1]
    assert result == output_dir / "results.csv"
    assert result.is_file()


def test_orchestrator_uses_fixed_path_length_per_block(tmp_path, monkeypatch):
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        "\n".join([
            "path_kind: sin",
            "Ns: [4, 8]",
            "Ds: [2]",
            "Ms: [2]",
            "operations: [signature, logsignature]",
            "backends: [cpu]",
            "path_lengths:",
            "  - d: 2",
            "    m: 2",
            "    operation: signature",
            "    backends: {cpu: 32}",
            "  - d: 2",
            "    m: 2",
            "    operation: logsignature",
            "    backends: {cpu: 64}",
            "batch_size: 1",
            "repeats: 1",
            "runs_dir: runs",
        ]),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "\n".join([
            "libraries:",
            "  example:",
            "    type: python",
            "    script: adapter.py",
            "    operations: [signature, logsignature]",
            "    backends:",
            "      - name: cpu",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    calls = []

    def fake_adapter(library_name, library_config, task_config, **kwargs):
        calls.append((task_config["operation"], task_config["N"]))
        return _result(task_config)

    monkeypatch.setattr(orchestrator, "run_python_adapter", fake_adapter)
    orchestrator.run_orchestrator(sweep_path, registry_path=registry_path)

    assert calls == [("signature", 32), ("logsignature", 64)]


def test_orchestrator_runs_only_selected_libraries(tmp_path, monkeypatch):
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        "\n".join([
            "path_kind: sin",
            "libraries: [selected]",
            "Ns: [4]",
            "Ds: [2]",
            "Ms: [2]",
            "operations: [signature]",
            "batch_size: 1",
            "repeats: 1",
            "runs_dir: runs",
        ]),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "\n".join([
            "libraries:",
            "  selected:",
            "    type: python",
            "    script: selected.py",
            "    operations: [signature]",
            "  excluded:",
            "    type: python",
            "    script: excluded.py",
            "    operations: [signature]",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    calls = []

    def fake_adapter(library_name, library_config, task_config, **kwargs):
        calls.append(library_name)
        return _result(task_config)

    monkeypatch.setattr(orchestrator, "run_python_adapter", fake_adapter)
    orchestrator.run_orchestrator(sweep_path, registry_path=registry_path)

    assert calls == ["selected"]


def test_adaptive_sweep_uses_common_path_length(tmp_path, monkeypatch):
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        "\n".join([
            "path_kind: sin",
            "Ns: [4]",
            "Ds: [2]",
            "Ms: [2]",
            "operations: [signature]",
            "backends: [cpu]",
            "batch_size: 1",
            "repeats: 3",
            "warmup_iterations: 1",
            "timing_statistic: min",
            "adaptive_min_time_ms: 100",
            "adaptive_growth_factor: 2",
            "runs_dir: runs",
        ]),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "\n".join([
            "libraries:",
            "  fast:",
            "    type: python",
            "    script: fast.py",
            "    backend: cpu",
            "    operations: [signature]",
            "  slow:",
            "    type: python",
            "    script: slow.py",
            "    backend: cpu",
            "    operations: [signature]",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    calls = []

    def fake_adapter(library_name, library_config, task_config, **kwargs):
        calls.append((library_name, task_config["N"]))
        multiplier = 10 if library_name == "fast" else 20
        t_ms = task_config["N"] * multiplier
        return {
            **_result(task_config),
            "library": library_name,
            "t_ms": t_ms,
            "t_ms_mean": t_ms,
            "samples_ms": [t_ms] * 3,
        }

    monkeypatch.setattr(orchestrator, "run_python_adapter", fake_adapter)
    csv_path = orchestrator.run_orchestrator(
        sweep_path,
        registry_path=registry_path,
    )

    with csv_path.open(newline="", encoding="utf-8") as results_file:
        rows = list(csv.DictReader(results_file))
    with (
        csv_path.parent / orchestrator.ADAPTIVE_BLOCKS_FILE
    ).open(newline="", encoding="utf-8") as blocks_file:
        blocks = list(csv.DictReader(blocks_file))

    assert calls == [
        ("fast", 4),
        ("fast", 12),
        ("slow", 4),
        ("slow", 8),
        ("slow", 12),
    ]
    assert {int(row["N"]) for row in rows} == {12}
    assert min(float(row["t_ms"]) for row in rows) >= 100
    assert blocks[0]["nominal_N"] == "4"
    assert blocks[0]["final_N"] == "12"


def test_adaptive_sweep_records_oom_and_continues(tmp_path, monkeypatch):
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        "\n".join([
            "path_kind: sin",
            "Ns: [4]",
            "Ds: [2]",
            "Ms: [2]",
            "operations: [signature]",
            "backends: [cpu]",
            "batch_size: 1",
            "repeats: 3",
            "adaptive_min_time_ms: 100",
            "runs_dir: runs",
        ]),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "\n".join([
            "libraries:",
            "  healthy:",
            "    type: python",
            "    script: healthy.py",
            "    backend: cpu",
            "    operations: [signature]",
            "  oom:",
            "    type: python",
            "    script: oom.py",
            "    backend: cpu",
            "    operations: [signature]",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)

    def fake_adapter(library_name, library_config, task_config, **kwargs):
        if library_name == "oom":
            raise MemoryError("out of memory")
        t_ms = task_config["N"] * 30
        return {
            **_result(task_config),
            "library": library_name,
            "t_ms": t_ms,
            "t_ms_mean": t_ms,
            "samples_ms": [t_ms] * 3,
        }

    monkeypatch.setattr(orchestrator, "run_python_adapter", fake_adapter)
    csv_path = orchestrator.run_orchestrator(
        sweep_path,
        registry_path=registry_path,
    )

    with csv_path.open(newline="", encoding="utf-8") as results_file:
        rows = list(csv.DictReader(results_file))
    with (
        csv_path.parent / orchestrator.FAILED_TASKS_FILE
    ).open(newline="", encoding="utf-8") as failed_file:
        failures = list(csv.DictReader(failed_file))

    assert len(rows) == 1
    assert rows[0]["library"] == "healthy"
    assert failures[0]["library"] == "oom"
    assert failures[0]["error_type"] == "MemoryError"
    assert failures[0]["reason"] == "out of memory"


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


def test_failed_task_is_recorded_and_not_retried(tmp_path, monkeypatch):
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        "\n".join([
            "path_kind: sin",
            "seed: 1",
            "Ns: [4, 8, 16]",
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

    orchestrator.run_orchestrator(sweep_path)

    run_dir = next((tmp_path / "runs").iterdir())
    csv_path = run_dir / "results.csv"
    with csv_path.open(newline="", encoding="utf-8") as results_file:
        result_rows = list(csv.DictReader(results_file))
    with (
        run_dir / orchestrator.FAILED_TASKS_FILE
    ).open(newline="", encoding="utf-8") as failed_file:
        failed_rows = list(csv.DictReader(failed_file))

    assert first_calls == [4, 8, 16]
    assert [int(row["N"]) for row in result_rows] == [4, 16]
    assert [int(row["N"]) for row in failed_rows] == [8]
    assert failed_rows[0]["error_type"] == "RuntimeError"
    assert failed_rows[0]["reason"] == "adapter failed"
    assert len((run_dir / orchestrator.COMPLETED_TASKS_FILE).read_text().splitlines()) == 3

    resumed_calls = []

    def finish_run(library_name, library_config, task_config, **kwargs):
        resumed_calls.append(task_config["N"])
        return _result(task_config)

    monkeypatch.setattr(orchestrator, "run_python_adapter", finish_run)
    orchestrator.run_orchestrator(resume_dir=run_dir)

    with csv_path.open(newline="", encoding="utf-8") as results_file:
        final_rows = list(csv.DictReader(results_file))
    assert resumed_calls == []
    assert [int(row["N"]) for row in final_rows] == [4, 16]

    # Retry only recorded failures even if the saved grid has an unrun case.
    saved_sweep = run_dir / "benchmark_sweep.yaml"
    saved_sweep.write_text(saved_sweep.read_text().replace("Ns: [4, 8, 16]", "Ns: [4, 8, 16, 32]"))
    orchestrator.run_orchestrator(resume_dir=run_dir, retry_failed=True)
    with csv_path.open(newline="", encoding="utf-8") as results_file:
        final_rows = list(csv.DictReader(results_file))
    assert resumed_calls == [8]
    assert [int(row["N"]) for row in final_rows] == [4, 16, 8]
    with (run_dir / orchestrator.FAILED_TASKS_FILE).open(newline="", encoding="utf-8") as file:
        assert list(csv.DictReader(file)) == []
    backup_dir, = run_dir.glob("before-retry-*")
    with (backup_dir / orchestrator.FAILED_TASKS_FILE).open(newline="", encoding="utf-8") as file:
        assert [int(row["N"]) for row in csv.DictReader(file)] == [8]
    assert len((backup_dir / orchestrator.COMPLETED_TASKS_FILE).read_text().splitlines()) == 3
    orchestrator.run_orchestrator(resume_dir=run_dir, retry_failed=True)
    assert resumed_calls == [8]


def test_retry_exclusion_preserves_failures_and_successes(tmp_path):
    success_id, retry_id, excluded_id = (character * 64 for character in "abc")
    orchestrator.initialize_results_csv(tmp_path / "results.csv")
    with (tmp_path / "results.csv").open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=orchestrator.RESULT_FIELDS)
        writer.writerow({"task_id": success_id})
    failed_path = tmp_path / orchestrator.FAILED_TASKS_FILE
    orchestrator.initialize_failed_tasks_csv(failed_path)
    with failed_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=orchestrator.FAILED_TASK_FIELDS)
        writer.writerows([
            {"task_id": success_id, "library": "working"},
            {"task_id": retry_id, "library": "retry"},
            {"task_id": excluded_id, "library": "log-signatures-pytorch", "reason": "original failure"},
        ])
    completed_path = tmp_path / orchestrator.COMPLETED_TASKS_FILE
    completed_path.write_text("\n".join([success_id, retry_id, excluded_id]) + "\n")
    original_failures = failed_path.read_bytes()
    original_results = (tmp_path / "results.csv").read_bytes()

    assert orchestrator.reopen_failed_tasks(tmp_path, ["log-signatures-pytorch"]) == {retry_id}

    assert orchestrator.load_completed_tasks(completed_path) == {success_id, excluded_id}
    with failed_path.open(newline="") as file:
        remaining = list(csv.DictReader(file))
    assert [row["task_id"] for row in remaining] == [excluded_id]
    assert remaining[0]["reason"] == "original failure"
    assert (tmp_path / "results.csv").read_bytes() == original_results
    backup, = tmp_path.glob("before-retry-*")
    assert (backup / orchestrator.FAILED_TASKS_FILE).read_bytes() == original_failures
    orchestrator.reopen_failed_tasks(tmp_path, ["log-signatures-pytorch"])
    assert list(tmp_path.glob("before-retry-*")) == [backup]


@pytest.mark.parametrize("reason", orchestrator.UNSUPPORTED_ERROR_MARKERS)
def test_unsupported_task_is_recorded_and_not_retried(tmp_path, monkeypatch, reason):
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
                stderr=reason,
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
    assert skipped_rows[0]["reason"] == reason
    with (run_dir / orchestrator.FAILED_TASKS_FILE).open(newline="", encoding="utf-8") as file:
        assert list(csv.DictReader(file)) == []
    assert len((run_dir / orchestrator.COMPLETED_TASKS_FILE).read_text().splitlines()) == 2

    resumed_calls = []
    monkeypatch.setattr(
        orchestrator,
        "run_python_adapter",
        lambda *args, **kwargs: resumed_calls.append(args),
    )
    orchestrator.run_orchestrator(resume_dir=run_dir)
    assert resumed_calls == []
