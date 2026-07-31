"""Orchestrator for signature benchmark suite"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from tqdm import tqdm


# Get script directory (src/)
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
COMPATIBILITY_PACKAGES_DIR = REPO_ROOT / ".adapter_packages"
PYTHON_ADAPTER_WORKER = SRC_DIR / "python_adapter_worker.py"
PYTHON_WORKER_PROTOCOL_PREFIX = "__PYSIGLIB_BENCHMARK_WORKER__"

# Configuration paths
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_SWEEP_CONFIG = CONFIG_DIR / "benchmark_sweep.yaml"
REGISTRY_CONFIG = CONFIG_DIR / "libraries_registry.yaml"

RESULT_FIELDS = [
    "task_id",
    "N",
    "d",
    "m",
    "batch_size",
    "seed",
    "path_kind",
    "operation",
    "backend",
    "language",
    "library",
    "method",
    "path_type",
    "timing_statistic",
    "t_ms",
    "t_ms_mean",
    "t_ms_std",
    "samples_ms",
    "alloc_bytes",
]
COMPLETED_TASKS_FILE = "completed_tasks.txt"
SKIPPED_TASKS_FILE = "skipped_tasks.csv"
FAILED_TASKS_FILE = "failed_tasks.csv"
ADAPTIVE_BLOCKS_FILE = "adaptive_blocks.csv"
SKIPPED_TASK_FIELDS = [
    "task_id",
    "library",
    "backend",
    "operation",
    "N",
    "d",
    "m",
    "batch_size",
    "reason",
]
FAILED_TASK_FIELDS = [
    *SKIPPED_TASK_FIELDS[:-1],
    "error_type",
    "reason",
]
ADAPTIVE_BLOCK_FIELDS = [
    "block_id",
    "nominal_N",
    "final_N",
    "d",
    "m",
    "operation",
    "backend",
    "fastest_t_ms",
]
UNSUPPORTED_ERROR_MARKERS = (
    "CUDA branched sig: num_trees > 1024 not supported",
)


class BenchmarkCallTimeout(TimeoutError):
    """A benchmark kernel call exceeded its configured deadline."""


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file"""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_run_folder(runs_dir: Path) -> Path:
    """Create timestamped run folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_dir}")
    return run_dir


def make_task_id(
    library_name: str,
    backend_name: str,
    task_config: Dict[str, Any],
) -> str:
    """Return a stable ID for one adapter invocation."""
    payload = {
        "library": library_name,
        "backend": backend_name,
        "config": task_config,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_completed_tasks(path: Path) -> set[str]:
    """Load fully committed task IDs, ignoring a possibly torn final line."""
    if not path.exists():
        return set()
    return {
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if len(line) == 64
    }


def initialize_results_csv(csv_path: Path) -> None:
    """Create a durable results file before any benchmark subprocess starts."""
    with csv_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        results_file.flush()
        os.fsync(results_file.fileno())


def initialize_skipped_tasks_csv(csv_path: Path) -> None:
    """Create the durable record of unsupported benchmark tasks."""
    with csv_path.open("w", newline="", encoding="utf-8") as skipped_file:
        writer = csv.DictWriter(skipped_file, fieldnames=SKIPPED_TASK_FIELDS)
        writer.writeheader()
        skipped_file.flush()
        os.fsync(skipped_file.fileno())


def initialize_failed_tasks_csv(csv_path: Path) -> None:
    """Create the durable record of failed benchmark tasks."""
    with csv_path.open("w", newline="", encoding="utf-8") as failed_file:
        writer = csv.DictWriter(failed_file, fieldnames=FAILED_TASK_FIELDS)
        writer.writeheader()
        failed_file.flush()
        os.fsync(failed_file.fileno())


def initialize_adaptive_blocks_csv(csv_path: Path) -> None:
    """Create the durable record of completed adaptive comparison blocks."""
    with csv_path.open("w", newline="", encoding="utf-8") as blocks_file:
        writer = csv.DictWriter(blocks_file, fieldnames=ADAPTIVE_BLOCK_FIELDS)
        writer.writeheader()
        blocks_file.flush()
        os.fsync(blocks_file.fileno())


def load_adaptive_blocks(csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", newline="", encoding="utf-8") as blocks_file:
        return {
            row["block_id"]: row
            for row in csv.DictReader(blocks_file)
        }


def discard_uncommitted_rows(
    csv_path: Path,
    completed_tasks: set[str],
    fieldnames: Sequence[str] = RESULT_FIELDS,
) -> None:
    """Remove rows written by a task that crashed before its completion marker."""
    with csv_path.open("r", newline="", encoding="utf-8") as results_file:
        reader = csv.DictReader(results_file)
        if reader.fieldnames != list(fieldnames):
            raise ValueError(
                f"Cannot resume {csv_path}: results schema does not support resume"
            )
        rows = [
            row
            for row in reader
            if row.get("task_id") in completed_tasks
        ]

    temp_path = csv_path.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(
            results_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        results_file.flush()
        os.fsync(results_file.fileno())
    os.replace(temp_path, csv_path)


def append_task_results(
    writer: csv.DictWriter,
    results_file,
    completed_file,
    task_id: str,
    result: Any,
) -> int:
    """Append one complete adapter result and then commit its task ID."""
    rows = result if isinstance(result, list) else [result]
    if not rows:
        raise RuntimeError(f"Task {task_id} returned no result rows")

    for row in rows:
        row["task_id"] = task_id
        writer.writerow(row)
    results_file.flush()
    os.fsync(results_file.fileno())

    completed_file.write(task_id + "\n")
    completed_file.flush()
    os.fsync(completed_file.fileno())
    return len(rows)


def append_skipped_task(
    writer: csv.DictWriter,
    skipped_file,
    completed_file,
    task_id: str,
    library_name: str,
    backend_name: str,
    task_config: Dict[str, Any],
    reason: str,
) -> None:
    """Record an unsupported task, then commit it so resume will not retry it."""
    writer.writerow({
        "task_id": task_id,
        "library": library_name,
        "backend": backend_name,
        "operation": task_config["operation"],
        "N": task_config["N"],
        "d": task_config["d"],
        "m": task_config["m"],
        "batch_size": task_config["batch_size"],
        "reason": reason,
    })
    skipped_file.flush()
    os.fsync(skipped_file.fileno())

    completed_file.write(task_id + "\n")
    completed_file.flush()
    os.fsync(completed_file.fileno())


def append_failed_task(
    writer: csv.DictWriter,
    failed_file,
    completed_file,
    task_id: str,
    library_name: str,
    backend_name: str,
    task_config: Dict[str, Any],
    error: Exception,
) -> None:
    """Record a failed task, then commit it so resume will not retry it."""
    writer.writerow({
        "task_id": task_id,
        "library": library_name,
        "backend": backend_name,
        "operation": task_config["operation"],
        "N": task_config["N"],
        "d": task_config["d"],
        "m": task_config["m"],
        "batch_size": task_config["batch_size"],
        "error_type": type(error).__name__,
        "reason": str(error),
    })
    failed_file.flush()
    os.fsync(failed_file.fileno())

    completed_file.write(task_id + "\n")
    completed_file.flush()
    os.fsync(completed_file.fileno())


def unsupported_adapter_reason(error: Exception) -> Optional[str]:
    """Return a stable reason for a known deterministic capability failure."""
    details = [str(error)]
    if isinstance(error, subprocess.CalledProcessError):
        details.extend([error.stdout or "", error.stderr or ""])
    output = "\n".join(details)
    return next(
        (marker for marker in UNSUPPORTED_ERROR_MARKERS if marker in output),
        None,
    )


def count_result_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8") as results_file:
        return sum(1 for _ in csv.DictReader(results_file))


def _command_output(cmd: List[str]) -> Optional[str]:
    """Return short system metadata without making benchmark execution depend on it."""
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip()
    return output or None


def write_run_metadata(run_dir: Path, sweep_config: Path) -> None:
    """Save environment and hardware metadata needed to identify a paper run."""
    git_status = _command_output(["git", "status", "--porcelain"])
    git_diff = _command_output(["git", "diff", "--binary", "HEAD"])
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "git_status": git_status,
        "hostname": platform.node(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu": _command_output(["lscpu"]),
        "memory": _command_output(["free", "-b"]),
        "sweep_config": str(sweep_config),
        "uv_version": _command_output(["uv", "--version"]),
        "julia_version": _command_output(["julia", "--version"]),
        "nvcc_version": _command_output(["nvcc", "--version"]),
        "gpu": _command_output([
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
            "--format=csv,noheader",
        ]),
        "nvidia_smi": _command_output(["nvidia-smi", "-q"]),
        "environment": {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "CUDA_HOME",
                "CMAKE_ARGS",
                "JAX_ENABLE_X64",
                "JAX_PLATFORM_NAME",
                "JAX_PLATFORMS",
                "KERAS_BACKEND",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TORCH_CUDA_ARCH_LIST",
                "VECLIB_MAXIMUM_THREADS",
                "XLA_FLAGS",
            )
            if key in os.environ
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if git_status:
        (run_dir / "git_status.txt").write_text(
            git_status + "\n",
            encoding="utf-8",
        )
    if git_diff:
        (run_dir / "git_diff.patch").write_text(
            git_diff + "\n",
            encoding="utf-8",
        )

    lockfile = REPO_ROOT / "uv.lock"
    if lockfile.exists():
        shutil.copy2(lockfile, run_dir / "uv.lock")


def record_run_completion(run_dir: Path, wall_time_seconds: float) -> None:
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_at"] = datetime.now().astimezone().isoformat()
    metadata["wall_time_seconds"] = wall_time_seconds
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _progress_write(line: str, *, file=None) -> None:
    tqdm.write(line.rstrip("\n"), file=file)


def run_subprocess_capture(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """
    Run an adapter subprocess and capture its output.

    Adapter stdout is the machine-readable result channel. Capturing stdout and
    stderr keeps dependency/runtime warnings from corrupting the progress bar;
    captured output is surfaced by the caller if the subprocess fails.
    """
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    return completed.stdout


def ensure_compatibility_wheel(
    library_name: str,
    library_config: Dict[str, Any],
    extras: Sequence[str],
    env: Dict[str, str],
) -> Optional[Path]:
    """Install a pinned compatibility wheel into an ignored local cache."""
    wheel_config = library_config.get("compatibility_wheel")
    if wheel_config is None:
        return None

    url = str(wheel_config["url"])
    if not url.startswith("https://") or "#sha256=" not in url:
        raise ValueError(
            f"{library_name}: compatibility wheel must use HTTPS and include a sha256 fragment"
        )

    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    target = COMPATIBILITY_PACKAGES_DIR / f"{library_name}-{cache_key}"
    marker = target / ".complete"
    if marker.is_file() and marker.read_text(encoding="utf-8") == url:
        return target
    if target.exists():
        raise RuntimeError(
            f"{library_name}: incomplete compatibility cache at {target}; "
            "remove it and retry"
        )

    COMPATIBILITY_PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{library_name}-",
        dir=COMPATIBILITY_PACKAGES_DIR,
    ) as temporary_directory:
        temporary_target = Path(temporary_directory)
        command = ["uv", "run"]
        for extra in extras:
            command.extend(["--extra", extra])
        command.extend(
            [
                "--with",
                "pip",
                "python",
                "-m",
                "pip",
                "install",
                "--target",
                str(temporary_target),
                "--no-deps",
                "--ignore-requires-python",
                url,
            ]
        )
        run_subprocess_capture(command, env=env)
        (temporary_target / ".complete").write_text(url, encoding="utf-8")
        os.replace(temporary_target, target)

    return target


def backend_variants(library_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return backend variants for a library.

    A missing backend list keeps existing registry entries as a single default
    run. Entries may set a label and environment variables, e.g.
    {"name": "cpu", "env": {"JAX_PLATFORM_NAME": "cpu"}}.
    """
    variants = library_config.get("backends")
    if not variants:
        backend = library_config.get("backend", "")
        env = library_config.get("env", {})
        return [{"name": backend, "env": env}]

    normalized = []
    for variant in variants:
        if isinstance(variant, str):
            normalized.append({"name": variant, "env": {}})
        else:
            normalized.append({
                "name": variant.get("name", ""),
                "env": variant.get("env", {}),
            })
    return normalized


def prepare_python_adapter_process(
    library_name: str,
    library_config: Dict[str, Any],
    env_overrides: Optional[Dict[str, str]],
) -> tuple[Path, List[str], Dict[str, str]]:
    script_path = REPO_ROOT / library_config["script"]
    extras = list(library_config.get("extras", []))
    deps = library_config.get("deps", [])
    env = os.environ.copy()
    if env_overrides:
        env.update({
            key: str(value)
            .replace("{repo_root}/", str(REPO_ROOT) + os.sep)
            .replace("{repo_root}\\", str(REPO_ROOT) + os.sep)
            .replace("{repo_root}", str(REPO_ROOT))
            for key, value in env_overrides.items()
        })

    compatibility_path = ensure_compatibility_wheel(
        library_name,
        library_config,
        extras,
        env,
    )
    if compatibility_path is not None:
        pythonpath = [str(compatibility_path)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    command = ["uv", "run"]
    for extra in extras:
        command.extend(["--extra", extra])
    for dep in deps:
        command.extend(["--with", dep])
    return script_path, command, env


class PythonAdapterWorker:
    """Run every task for one Python library/backend in one process."""

    def __init__(
        self,
        library_name: str,
        library_config: Dict[str, Any],
        env_overrides: Optional[Dict[str, str]] = None,
    ):
        self.library_name = library_name
        self.library_config = library_config
        self.env_overrides = env_overrides
        self.process = None
        self.diagnostics: List[str] = []
        self.scope = library_config.get("worker_scope")
        self.scope_key = None
        self._output_queue = None
        self._reader_thread = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()

    def _read_response(self) -> Dict[str, Any]:
        if self.process is None or self._output_queue is None:
            raise RuntimeError(f"{self.library_name}: worker is not running")

        deadline = None
        active_call = None
        while True:
            timeout = None
            if deadline is not None:
                timeout = max(0.0, deadline - time.monotonic())
            try:
                line = self._output_queue.get(timeout=timeout)
            except queue.Empty:
                phase = active_call.get("phase", "call")
                iteration = active_call.get("iteration", 0) + 1
                timeout_seconds = active_call["timeout_seconds"]
                self._kill_process_tree()
                raise BenchmarkCallTimeout(
                    f"{self.library_name}: {phase} call {iteration} exceeded "
                    f"{timeout_seconds:g} seconds"
                )

            if line is None:
                return_code = self.process.poll()
                details = "\n".join(self.diagnostics[-20:])
                raise RuntimeError(
                    f"{self.library_name}: worker exited with code {return_code}"
                    + (f"\n{details}" if details else "")
                )
            line = line.rstrip("\r\n")
            protocol_index = line.find(PYTHON_WORKER_PROTOCOL_PREFIX)
            if protocol_index >= 0:
                diagnostic = line[:protocol_index]
                if diagnostic:
                    self.diagnostics.append(diagnostic)
                payload = json.loads(
                    line[protocol_index + len(PYTHON_WORKER_PROTOCOL_PREFIX):]
                )
                status = payload.get("status")
                if status == "call_start":
                    timeout_seconds = float(payload["timeout_seconds"])
                    active_call = {
                        **payload,
                        "timeout_seconds": timeout_seconds,
                    }
                    deadline = time.monotonic() + timeout_seconds
                    continue
                if status == "call_end":
                    active_call = None
                    deadline = None
                    continue
                return payload
            if line:
                self.diagnostics.append(line)

    @staticmethod
    def _read_output(stdout, output_queue: queue.Queue) -> None:
        for line in stdout:
            output_queue.put(line)
        output_queue.put(None)

    def _kill_process_tree(self) -> None:
        if self.process is None:
            return
        process = self.process
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _start(self) -> None:
        if self.process is not None:
            return
        script_path, command, env = prepare_python_adapter_process(
            self.library_name,
            self.library_config,
            self.env_overrides,
        )
        command.extend([
            "python",
            str(PYTHON_ADAPTER_WORKER),
            str(script_path),
        ])
        popen_kwargs = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_kwargs,
        )
        self._output_queue = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._read_output,
            args=(self.process.stdout, self._output_queue),
            daemon=True,
        )
        self._reader_thread.start()
        response = self._read_response()
        if response.get("status") != "ready":
            raise RuntimeError(
                f"{self.library_name}: worker startup failed: "
                f"{response.get('error', response)}\n"
                f"{response.get('traceback', '')}"
            )
        if self.scope is None:
            self.scope = response.get("worker_scope", "backend")

    def _task_scope_key(self, task_config: Dict[str, Any]) -> tuple:
        if self.scope == "backend" or self.scope is None:
            return ("backend",)
        if self.scope == "shape":
            return (
                "shape",
                task_config["N"],
                task_config["d"],
                task_config.get("batch_size", 1),
                task_config.get("path_kind", ""),
                task_config.get("seed", 0),
            )
        if self.scope == "task":
            return ("task", json.dumps(task_config, sort_keys=True))
        raise ValueError(
            f"{self.library_name}: invalid worker_scope {self.scope!r}"
        )

    def run(self, task_config: Dict[str, Any]) -> Any:
        if (
            self.process is not None
            and self.scope_key != self._task_scope_key(task_config)
        ):
            self.close()
        self._start()
        if self.scope_key is None:
            self.scope_key = self._task_scope_key(task_config)
        if self.process is None or self.process.stdin is None:
            raise RuntimeError(f"{self.library_name}: worker is not running")
        self.process.stdin.write(json.dumps(task_config) + "\n")
        self.process.stdin.flush()
        response = self._read_response()
        status = response.get("status")
        if status == "result":
            return response.get("result")
        raise RuntimeError(
            f"{self.library_name}: adapter task failed: "
            f"{response.get('error', response)}\n"
            f"{response.get('traceback', '')}"
        )

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.process = None
        self.scope_key = None
        self._output_queue = None
        self._reader_thread = None


def run_python_adapter(
    library_name: str,
    library_config: Dict[str, Any],
    task_config: Dict[str, Any],
    *,
    env_overrides: Optional[Dict[str, str]] = None,
    worker: Optional[PythonAdapterWorker] = None,
) -> Any:
    """
    Run a Python adapter using uv with project extras.

    Args:
        library_name: Name of the library
        library_config: Library configuration from registry
        task_config: Task parameters (N, d, m, etc.)

    Returns:
        Benchmark result dictionary
    """
    if worker is not None:
        return worker.run(task_config)

    try:
        script_path, cmd, env = prepare_python_adapter_process(
            library_name,
            library_config,
            env_overrides,
        )
        cmd.extend(["python", str(script_path), json.dumps(task_config)])

        stdout = run_subprocess_capture(cmd, env=env)

        # Parse JSON output from stdout
        output_lines = stdout.strip().split('\n')
        for line in output_lines:
            line = line.strip()
            if line.startswith('{'):
                return json.loads(line)

        raise RuntimeError(
            f"No JSON output from {library_name}. Captured stdout:\n{stdout}"
        )

    except subprocess.CalledProcessError as e:
        if unsupported_adapter_reason(e) is not None:
            raise
        _progress_write(f"Error running {library_name}:", file=sys.stderr)
        _progress_write(f"  command: {shlex.join(e.cmd)}", file=sys.stderr)
        if e.stdout:
            _progress_write(f"  stdout: {e.stdout}", file=sys.stderr)
        if e.stderr:
            _progress_write(f"  stderr: {e.stderr}", file=sys.stderr)
        raise


def run_julia_adapter(
    library_name: str,
    library_config: Dict[str, Any],
    task_config: Dict[str, Any],
    *,
    env_overrides: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run a Julia adapter with project environment.

    Args:
        library_name: Name of the library
        library_config: Library configuration from registry
        task_config: Task parameters (N, d, m, etc.)

    Returns:
        List of benchmark result dictionaries
    """
    julia_dir = REPO_ROOT / library_config["dir"]
    script = julia_dir / library_config["script"]

    # Build Julia command
    # JULIA_PROJECT=<dir> julia <script> '<json_config>'
    env = os.environ.copy()
    env["JULIA_PROJECT"] = str(julia_dir)
    if env_overrides:
        env.update({key: str(value) for key, value in env_overrides.items()})

    cmd = [
        "julia",
        str(script),
        json.dumps(task_config)
    ]

    try:
        stdout = run_subprocess_capture(cmd, env=env)

        # Parse JSON output from stdout (one line per benchmark result)
        outputs: List[Dict[str, Any]] = []
        output_lines = stdout.strip().split('\n')
        for line in output_lines:
            line = line.strip()
            if line.startswith('{'):
                outputs.append(json.loads(line))

        if outputs:
            return outputs

        raise RuntimeError(
            f"No JSON output from {library_name}. Captured stdout:\n{stdout}"
        )

    except subprocess.CalledProcessError as e:
        _progress_write(f"Error running {library_name}:", file=sys.stderr)
        _progress_write(f"  command: {shlex.join(e.cmd)}", file=sys.stderr)
        if e.stdout:
            _progress_write(f"  stdout: {e.stdout}", file=sys.stderr)
        if e.stderr:
            _progress_write(f"  stderr: {e.stderr}", file=sys.stderr)
        raise


def format_task_label(
    library_name: str,
    backend_name: str,
    operation: str,
    *,
    N: int,
    d: int,
    m: int,
    batch_size: int,
) -> str:
    backend_suffix = f"[{backend_name}]" if backend_name else ""
    return (
        f"{library_name}{backend_suffix}.{operation} "
        f"N={N} d={d} m={m} batch={batch_size}"
    )


def build_task_config(
    sweep: Dict[str, Any],
    library_name: str,
    *,
    N: int,
    d: int,
    m: int,
    path_kind: str,
    operation: str,
    repeats: int,
    batch_size: int,
    seed: int,
    backend_name: str,
) -> Dict[str, Any]:
    task_config = {
        "N": N,
        "d": d,
        "m": m,
        "path_kind": path_kind,
        "operation": operation,
        "repeats": repeats,
        "batch_size": batch_size,
        "seed": seed,
        "backend": backend_name,
    }
    for key in (
        "logsig_method",
        "log_sig_method",
        "num_chunks",
        "sig_kernel_order",
        "sig_kernel_solver",
        "sig_kernel_dyadic_order",
        "sig_kernel_max_batch",
        "warmup_iterations",
        "timing_statistic",
        "call_timeout_seconds",
        "clear_input_caches_after_task",
    ):
        if key in sweep:
            task_config[key] = sweep[key]
    task_config.update(
        sweep.get("library_configs", {}).get(library_name, {})
    )
    return task_config


def _adaptive_block_id(
    nominal_N: int,
    d: int,
    m: int,
    operation: str,
    backend_name: str,
) -> str:
    encoded = json.dumps(
        [nominal_N, d, m, operation, backend_name],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_sweep_cases(
    sweep: Dict[str, Any],
    Ns: Sequence[int],
    Ds: Sequence[int],
    Ms: Sequence[int],
    operations: Sequence[str],
) -> List[Dict[str, Any]]:
    fixed_lengths = sweep.get("path_lengths")
    if fixed_lengths is None:
        return [
            {
                "N": int(N),
                "d": int(d),
                "m": int(m),
                "operation": operation,
                "backend": None,
            }
            for N in Ns
            for d in Ds
            for m in Ms
            for operation in operations
        ]
    if not isinstance(fixed_lengths, list):
        raise ValueError("path_lengths must be a list")

    cases = []
    seen = set()
    for entry in fixed_lengths:
        if not isinstance(entry, dict):
            raise ValueError("each path_lengths entry must be a mapping")
        try:
            d = int(entry["d"])
            m = int(entry["m"])
            operation = str(entry["operation"])
            backends = entry["backends"]
        except KeyError as error:
            raise ValueError(
                f"path_lengths entry is missing {error.args[0]}"
            ) from error
        if not isinstance(backends, dict) or not backends:
            raise ValueError("path_lengths backends must be a non-empty mapping")
        for backend_name, value in backends.items():
            N = int(value)
            if N < 2:
                raise ValueError("fixed path lengths must be >= 2")
            key = (d, m, operation, str(backend_name))
            if key in seen:
                raise ValueError(
                    "duplicate fixed path length for "
                    f"d={d}, m={m}, operation={operation}, "
                    f"backend={backend_name}"
                )
            seen.add(key)
            cases.append({
                "N": N,
                "d": d,
                "m": m,
                "operation": operation,
                "backend": str(backend_name),
            })
    return cases


def remove_adaptive_inputs(
    run_dir: Path,
    *,
    seed: int,
    N: int,
    d: int,
    batch_size: int,
    path_count: int,
) -> None:
    inputs_dir = run_dir / "inputs"
    for logical_seed in range(seed, seed + path_count):
        cache_name = (
            f"brownian_seed{logical_seed}_N{N}_d{d}_batch{batch_size}.npy"
        )
        cache_path = inputs_dir / cache_name
        cache_path.unlink(missing_ok=True)
        cache_path.with_suffix(".npy.sha256").unlink(missing_ok=True)
        for temporary_path in inputs_dir.glob(f".{cache_name}.*.tmp"):
            temporary_path.unlink(missing_ok=True)


def run_adaptive_tasks(
    *,
    sweep: Dict[str, Any],
    libraries: Dict[str, Dict[str, Any]],
    library_backends: Dict[str, List[Dict[str, Any]]],
    Ns: Sequence[int],
    Ds: Sequence[int],
    Ms: Sequence[int],
    operations: Sequence[str],
    path_kind: str,
    repeats: int,
    batch_size: int,
    seed: int,
    run_dir: Path,
    csv_path: Path,
    completed_path: Path,
    skipped_path: Path,
    failed_path: Path,
    blocks_path: Path,
    completed_tasks: set[str],
    completed_blocks: Dict[str, Dict[str, str]],
) -> None:
    minimum_time_ms = float(sweep["adaptive_min_time_ms"])
    growth_factor = int(sweep.get("adaptive_growth_factor", 2))
    maximum_growth_factor = int(
        sweep.get("adaptive_max_growth_factor", 16)
    )
    maximum_N = int(sweep.get("adaptive_max_N", 1048576))
    maximum_input_bytes = int(
        sweep.get("adaptive_max_input_bytes", 1073741824)
    )
    if minimum_time_ms <= 0:
        raise ValueError("adaptive_min_time_ms must be > 0")
    if growth_factor <= 1:
        raise ValueError("adaptive_growth_factor must be > 1")
    if maximum_growth_factor < growth_factor:
        raise ValueError(
            "adaptive_max_growth_factor must be >= adaptive_growth_factor"
        )

    backend_names = sorted({
        backend.get("name", "")
        for backends in library_backends.values()
        for backend in backends
    })
    blocks = [
        (nominal_N, d, m, operation, backend_name)
        for nominal_N in Ns
        for d in Ds
        for m in Ms
        for operation in operations
        for backend_name in backend_names
        if any(
            operation in libraries[library_name].get("operations", [])
            and any(
                backend.get("name", "") == backend_name
                for backend in library_backends[library_name]
            )
            for library_name in libraries
        )
    ]
    previous_final_N: Dict[tuple[int, int, str, str], int] = {}

    with (
        csv_path.open("a", newline="", encoding="utf-8") as results_file,
        skipped_path.open("a", newline="", encoding="utf-8") as skipped_file,
        failed_path.open("a", newline="", encoding="utf-8") as failed_file,
        blocks_path.open("a", newline="", encoding="utf-8") as blocks_file,
        completed_path.open("a", encoding="utf-8") as completed_file,
        tqdm(total=len(blocks), unit="block", dynamic_ncols=True, leave=True) as progress,
    ):
        writer = csv.DictWriter(
            results_file,
            fieldnames=RESULT_FIELDS,
            extrasaction="ignore",
        )
        skipped_writer = csv.DictWriter(
            skipped_file,
            fieldnames=SKIPPED_TASK_FIELDS,
            extrasaction="ignore",
        )
        failed_writer = csv.DictWriter(
            failed_file,
            fieldnames=FAILED_TASK_FIELDS,
            extrasaction="ignore",
        )
        blocks_writer = csv.DictWriter(
            blocks_file,
            fieldnames=ADAPTIVE_BLOCK_FIELDS,
        )

        for nominal_N, d, m, operation, backend_name in blocks:
                block_id = _adaptive_block_id(
                    nominal_N,
                    d,
                    m,
                    operation,
                    backend_name,
                )
                series_key = (d, m, operation, backend_name)
                if block_id in completed_blocks:
                    previous_final_N[series_key] = int(
                        completed_blocks[block_id]["final_N"]
                    )
                    progress.update(1)
                    continue

                previous_N = previous_final_N.get(series_key)
                candidate_N = nominal_N
                if previous_N is not None and candidate_N <= previous_N:
                    candidate_N = previous_N * growth_factor

                participants = []
                for library_name, library_config in libraries.items():
                    if operation not in library_config.get("operations", []):
                        continue
                    for backend in library_backends[library_name]:
                        if backend.get("name", "") == backend_name:
                            participants.append(
                                (library_name, library_config, backend)
                            )

                final_results = []
                terminal_failures = []
                terminal_skips = []
                fastest_t_ms = None
                calibrated = []
                used_path_lengths = set()
                path_count = 2 if operation.startswith("signaturekernel") else 1

                def execute(participant, task_N, worker):
                    library_name, library_config, backend = participant
                    task_config = build_task_config(
                        sweep,
                        library_name,
                        N=task_N,
                        d=d,
                        m=m,
                        path_kind=path_kind,
                        operation=operation,
                        repeats=repeats,
                        batch_size=batch_size,
                        seed=seed,
                        backend_name=backend_name,
                    )
                    if path_kind.lower() == "brownian":
                        task_config["input_cache_dir"] = str(run_dir / "inputs")
                        used_path_lengths.add(task_N)
                    task_label = format_task_label(
                        library_name,
                        backend_name,
                        operation,
                        N=task_N,
                        d=d,
                        m=m,
                        batch_size=batch_size,
                    )
                    progress.set_description_str(task_label, refresh=True)
                    progress.set_postfix_str("calibrating", refresh=True)
                    if library_config["type"] == "python":
                        result = run_python_adapter(
                            library_name,
                            library_config,
                            task_config,
                            env_overrides=backend.get("env", {}),
                            worker=worker,
                        )
                    elif library_config["type"] == "julia":
                        result = run_julia_adapter(
                            library_name,
                            library_config,
                            task_config,
                            env_overrides=backend.get("env", {}),
                        )
                    else:
                        raise ValueError(
                            f"Unknown library type: {library_config['type']}"
                        )
                    if result is None:
                        raise RuntimeError(f"{task_label} returned no result")
                    rows = result if isinstance(result, list) else [result]
                    return task_config, result, rows

                for participant in participants:
                    library_name, library_config, backend = participant
                    participant_N = candidate_N
                    worker = (
                        PythonAdapterWorker(
                            library_name,
                            library_config,
                            backend.get("env", {}),
                        )
                        if library_config["type"] == "python"
                        else None
                    )
                    try:
                        while True:
                            input_bytes = (
                                path_count * batch_size * participant_N * d * 4
                            )
                            if (
                                participant_N > maximum_N
                                or input_bytes > maximum_input_bytes
                            ):
                                if participant_N > maximum_N:
                                    reason = (
                                        "adaptive path length exceeded maximum "
                                        f"N={maximum_N}"
                                    )
                                else:
                                    reason = (
                                        "adaptive input exceeded maximum size "
                                        f"{maximum_input_bytes} bytes"
                                    )
                                task_config = build_task_config(
                                    sweep,
                                    library_name,
                                    N=participant_N,
                                    d=d,
                                    m=m,
                                    path_kind=path_kind,
                                    operation=operation,
                                    repeats=repeats,
                                    batch_size=batch_size,
                                    seed=seed,
                                    backend_name=backend_name,
                                )
                                terminal_failures.append((
                                    participant,
                                    task_config,
                                    RuntimeError(reason),
                                ))
                                break
                            try:
                                task_config, result, rows = execute(
                                    participant,
                                    participant_N,
                                    worker,
                                )
                            except Exception as error:
                                task_config = build_task_config(
                                    sweep,
                                    library_name,
                                    N=participant_N,
                                    d=d,
                                    m=m,
                                    path_kind=path_kind,
                                    operation=operation,
                                    repeats=repeats,
                                    batch_size=batch_size,
                                    seed=seed,
                                    backend_name=backend_name,
                                )
                                unsupported_reason = unsupported_adapter_reason(error)
                                if unsupported_reason is None:
                                    terminal_failures.append(
                                        (participant, task_config, error)
                                    )
                                else:
                                    terminal_skips.append(
                                        (participant, task_config, unsupported_reason)
                                    )
                                break
                            participant_time = min(
                                float(row["t_ms"])
                                for row in rows
                            )
                            if participant_time >= minimum_time_ms:
                                calibrated.append((
                                    participant,
                                    participant_N,
                                    task_config,
                                    result,
                                    rows,
                                ))
                                break
                            if participant_time <= 0:
                                step_growth = maximum_growth_factor
                            else:
                                step_growth = math.ceil(
                                    minimum_time_ms / participant_time
                                )
                                step_growth = max(
                                    growth_factor,
                                    min(maximum_growth_factor, step_growth),
                                )
                            participant_N *= step_growth
                    finally:
                        if worker is not None:
                            worker.close()

                if calibrated:
                    candidate_N = max(item[1] for item in calibrated)
                    for calibrated_item in calibrated:
                        participant, required_N, task_config, result, rows = (
                            calibrated_item
                        )
                        if required_N == candidate_N:
                            final_results.append(
                                (participant, task_config, result, rows)
                            )
                            continue
                        library_name, library_config, backend = participant
                        worker = (
                            PythonAdapterWorker(
                                library_name,
                                library_config,
                                backend.get("env", {}),
                            )
                            if library_config["type"] == "python"
                            else None
                        )
                        try:
                            task_config, result, rows = execute(
                                participant,
                                candidate_N,
                                worker,
                            )
                            if min(float(row["t_ms"]) for row in rows) < minimum_time_ms:
                                raise RuntimeError(
                                    "runtime remained below adaptive minimum at "
                                    f"common N={candidate_N}"
                                )
                            final_results.append(
                                (participant, task_config, result, rows)
                            )
                        except Exception as error:
                            task_config = build_task_config(
                                sweep,
                                library_name,
                                N=candidate_N,
                                d=d,
                                m=m,
                                path_kind=path_kind,
                                operation=operation,
                                repeats=repeats,
                                batch_size=batch_size,
                                seed=seed,
                                backend_name=backend_name,
                            )
                            unsupported_reason = unsupported_adapter_reason(error)
                            if unsupported_reason is None:
                                terminal_failures.append(
                                    (participant, task_config, error)
                                )
                            else:
                                terminal_skips.append(
                                    (participant, task_config, unsupported_reason)
                                )
                        finally:
                            if worker is not None:
                                worker.close()

                    if final_results:
                        fastest_t_ms = min(
                            float(row["t_ms"])
                            for _, _, _, rows in final_results
                            for row in rows
                        )

                if path_kind.lower() == "brownian":
                    for used_N in used_path_lengths:
                        remove_adaptive_inputs(
                            run_dir,
                            seed=seed,
                            N=used_N,
                            d=d,
                            batch_size=batch_size,
                            path_count=path_count,
                        )

                for participant, task_config, result, _ in final_results:
                    library_name, _, _ = participant
                    task_id = make_task_id(
                        library_name,
                        backend_name,
                        task_config,
                    )
                    if task_id not in completed_tasks:
                        append_task_results(
                            writer,
                            results_file,
                            completed_file,
                            task_id,
                            result,
                        )
                        completed_tasks.add(task_id)

                for participant, task_config, error in terminal_failures:
                    library_name, _, _ = participant
                    task_id = make_task_id(
                        library_name,
                        backend_name,
                        task_config,
                    )
                    if task_id not in completed_tasks:
                        append_failed_task(
                            failed_writer,
                            failed_file,
                            completed_file,
                            task_id,
                            library_name,
                            backend_name,
                            task_config,
                            error,
                        )
                        completed_tasks.add(task_id)

                for participant, task_config, reason in terminal_skips:
                    library_name, _, _ = participant
                    task_id = make_task_id(
                        library_name,
                        backend_name,
                        task_config,
                    )
                    if task_id not in completed_tasks:
                        append_skipped_task(
                            skipped_writer,
                            skipped_file,
                            completed_file,
                            task_id,
                            library_name,
                            backend_name,
                            task_config,
                            reason,
                        )
                        completed_tasks.add(task_id)

                blocks_writer.writerow({
                    "block_id": block_id,
                    "nominal_N": nominal_N,
                    "final_N": candidate_N,
                    "d": d,
                    "m": m,
                    "operation": operation,
                    "backend": backend_name,
                    "fastest_t_ms": "" if fastest_t_ms is None else fastest_t_ms,
                })
                blocks_file.flush()
                os.fsync(blocks_file.fileno())
                completed_blocks[block_id] = {
                    "final_N": str(candidate_N),
                }
                previous_final_N[series_key] = candidate_N
                progress.set_postfix_str("done", refresh=True)
                progress.update(1)


def run_orchestrator(
    config_path: Optional[Path] = None,
    resume_dir: Optional[Path] = None,
    registry_path: Optional[Path] = None,
):
    """
    Main orchestrator logic

    Args:
        config_path: Optional sweep config (default: config/benchmark_sweep.yaml)
        resume_dir: Existing run directory to continue
        registry_path: Optional library registry (default: config/libraries_registry.yaml)
    """
    started_at = time.monotonic()
    if config_path is not None and resume_dir is not None:
        raise ValueError("config_path and resume_dir cannot be used together")
    if registry_path is not None and resume_dir is not None:
        raise ValueError("registry_path and resume_dir cannot be used together")

    if resume_dir is not None:
        run_dir = resume_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        sweep_config = run_dir / "benchmark_sweep.yaml"
        registry_config = run_dir / "libraries_registry.yaml"
        print(f"Resuming run folder: {run_dir}")
    else:
        sweep_config = config_path if config_path else DEFAULT_SWEEP_CONFIG
        registry_config = registry_path if registry_path else REGISTRY_CONFIG

    # Load configurations
    sweep = load_yaml(sweep_config)
    registry = load_yaml(registry_config)

    # Extract sweep parameters
    Ns = sweep.get("Ns", [200, 400, 800])
    Ds = sweep.get("Ds", [2, 5, 7])
    Ms = sweep.get("Ms", [2, 3, 4])
    path_kind = sweep.get("path_kind", "sin")
    operations = sweep.get("operations", ["signature", "logsignature"])
    registered_libraries = registry.get("libraries", {})
    selected_libraries = sweep.get("libraries")
    if selected_libraries is None:
        libraries = registered_libraries
    else:
        if not isinstance(selected_libraries, list) or not all(
            isinstance(name, str) for name in selected_libraries
        ):
            raise ValueError("libraries must be a list of registry names")
        if len(selected_libraries) != len(set(selected_libraries)):
            raise ValueError("libraries must not contain duplicates")
        unknown_libraries = [
            name for name in selected_libraries if name not in registered_libraries
        ]
        if unknown_libraries:
            raise ValueError(
                "Unknown libraries in sweep: " + ", ".join(unknown_libraries)
            )
        libraries = {
            name: registered_libraries[name]
            for name in selected_libraries
        }
    selected_backends = sweep.get("backends")
    if selected_backends is not None:
        selected_backends = {str(backend) for backend in selected_backends}
    repeats = sweep.get("repeats", 10)
    warmup_iterations = int(sweep.get("warmup_iterations", 3))
    if warmup_iterations < 0:
        raise ValueError(
            "warmup_iterations must be >= 0, "
            f"got {warmup_iterations}"
        )
    seed = int(sweep.get("seed", 0))
    batch_size = int(sweep.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    adaptive_min_time_ms = sweep.get("adaptive_min_time_ms")
    call_timeout_seconds = sweep.get("call_timeout_seconds")
    if call_timeout_seconds is not None and float(call_timeout_seconds) <= 0:
        raise ValueError("call_timeout_seconds must be > 0")
    runs_dir = REPO_ROOT / sweep.get("runs_dir", "runs")

    if resume_dir is None:
        run_dir = setup_run_folder(runs_dir)
        (run_dir / "benchmark_sweep.yaml").write_text(
            sweep_config.read_text(encoding="utf-8"),
            encoding="utf-8"
        )
        (run_dir / "libraries_registry.yaml").write_text(
            registry_config.read_text(encoding="utf-8"),
            encoding="utf-8"
        )
        write_run_metadata(run_dir, sweep_config)

    csv_path = run_dir / "results.csv"
    completed_path = run_dir / COMPLETED_TASKS_FILE
    skipped_path = run_dir / SKIPPED_TASKS_FILE
    failed_path = run_dir / FAILED_TASKS_FILE
    blocks_path = run_dir / ADAPTIVE_BLOCKS_FILE
    if resume_dir is None:
        initialize_results_csv(csv_path)
        initialize_skipped_tasks_csv(skipped_path)
        initialize_failed_tasks_csv(failed_path)
        if adaptive_min_time_ms is not None:
            initialize_adaptive_blocks_csv(blocks_path)
        completed_tasks: set[str] = set()
        completed_blocks: Dict[str, Dict[str, str]] = {}
    else:
        if not csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {csv_path}")
        completed_tasks = load_completed_tasks(completed_path)
        discard_uncommitted_rows(csv_path, completed_tasks)
        if skipped_path.exists():
            discard_uncommitted_rows(
                skipped_path,
                completed_tasks,
                SKIPPED_TASK_FIELDS,
            )
        else:
            initialize_skipped_tasks_csv(skipped_path)
        if failed_path.exists():
            discard_uncommitted_rows(
                failed_path,
                completed_tasks,
                FAILED_TASK_FIELDS,
            )
        else:
            initialize_failed_tasks_csv(failed_path)
        if adaptive_min_time_ms is not None:
            if not blocks_path.exists():
                initialize_adaptive_blocks_csv(blocks_path)
            completed_blocks = load_adaptive_blocks(blocks_path)
        else:
            completed_blocks = {}

    print("\n" + "=" * 60)
    print("Signature Benchmark Orchestrator")
    print("=" * 60)
    print(f"Path kind: {path_kind}")
    print(f"Ns: {Ns}")
    print(f"Ds: {Ds}")
    print(f"Ms: {Ms}")
    print(f"Operations: {operations}")
    if selected_backends is not None:
        print(f"Backends: {sorted(selected_backends)}")
    print(f"Batch size: {batch_size}")
    print(f"Repeats: {repeats}")
    print(f"Warmup iterations: {warmup_iterations}")
    if adaptive_min_time_ms is not None:
        print(f"Adaptive minimum time: {float(adaptive_min_time_ms):g} ms")
    if call_timeout_seconds is not None:
        print(f"Call timeout: {float(call_timeout_seconds):g} seconds")
    print(f"Seed: {seed}")
    print(f"Libraries: {list(libraries)}")
    print("=" * 60)

    # Run benchmarks
    library_backends = {
        name: [
            backend
            for backend in backend_variants(config)
            if selected_backends is None or backend.get("name", "") in selected_backends
        ]
        for name, config in libraries.items()
    }
    fixed_cases = build_sweep_cases(sweep, Ns, Ds, Ms, operations)
    fixed_N_by_block = {
        (case["d"], case["m"], case["operation"], case["backend"]): case["N"]
        for case in fixed_cases
        if case["backend"] is not None
    }
    if fixed_N_by_block:
        configured_backends = {
            backend.get("name", "")
            for backends in library_backends.values()
            for backend in backends
        }
        expected_blocks = {
            (int(d), int(m), operation, backend_name)
            for d in Ds
            for m in Ms
            for operation in operations
            for backend_name in configured_backends
        }
        missing_blocks = expected_blocks - set(fixed_N_by_block)
        extra_blocks = set(fixed_N_by_block) - expected_blocks
        if missing_blocks or extra_blocks:
            raise ValueError(
                "path_lengths must define exactly one length for each "
                "(d, m, operation, backend) block"
            )
        Ns = [0]
    backend_task_count = sum(len(backends) for backends in library_backends.values())
    total_tasks = len(Ns) * len(Ds) * len(Ms) * len(operations) * backend_task_count

    if adaptive_min_time_ms is not None:
        run_adaptive_tasks(
            sweep=sweep,
            libraries=libraries,
            library_backends=library_backends,
            Ns=Ns,
            Ds=Ds,
            Ms=Ms,
            operations=operations,
            path_kind=path_kind,
            repeats=repeats,
            batch_size=batch_size,
            seed=seed,
            run_dir=run_dir,
            csv_path=csv_path,
            completed_path=completed_path,
            skipped_path=skipped_path,
            failed_path=failed_path,
            blocks_path=blocks_path,
            completed_tasks=completed_tasks,
            completed_blocks=completed_blocks,
        )
        record_run_completion(run_dir, time.monotonic() - started_at)
        print("\n" + "=" * 60)
        print("Benchmark complete!")
        print(f"Results written to: {csv_path}")
        print(f"Total benchmarks: {count_result_rows(csv_path)}")
        print(f"Failed tasks: {count_result_rows(failed_path)}")
        print("=" * 60)
        return csv_path

    with (
        csv_path.open("a", newline="", encoding="utf-8") as results_file,
        skipped_path.open("a", newline="", encoding="utf-8") as skipped_file,
        failed_path.open("a", newline="", encoding="utf-8") as failed_file,
        completed_path.open("a", encoding="utf-8") as completed_file,
        tqdm(
            total=total_tasks,
            unit="bench",
            dynamic_ncols=True,
            leave=True,
        ) as progress,
    ):
        writer = csv.DictWriter(
            results_file,
            fieldnames=RESULT_FIELDS,
            extrasaction="ignore",
        )
        skipped_writer = csv.DictWriter(
            skipped_file,
            fieldnames=SKIPPED_TASK_FIELDS,
            extrasaction="ignore",
        )
        failed_writer = csv.DictWriter(
            failed_file,
            fieldnames=FAILED_TASK_FIELDS,
            extrasaction="ignore",
        )
        for library_name, library_config in libraries.items():
            lib_operations = set(library_config.get("operations", []))
            backends = library_backends[library_name]

            for backend in backends:
                backend_name = backend.get("name", "")
                python_worker = (
                    PythonAdapterWorker(
                        library_name,
                        library_config,
                        backend.get("env", {}),
                    )
                    if library_config["type"] == "python"
                    else None
                )

                for N in Ns:
                    for d in Ds:
                        for m in Ms:
                            for operation in operations:
                                case_N = fixed_N_by_block.get(
                                    (int(d), int(m), operation, backend_name),
                                    N,
                                )
                                task_label = format_task_label(
                                    library_name,
                                    backend_name,
                                    operation,
                                    N=case_N,
                                    d=d,
                                    m=m,
                                    batch_size=batch_size,
                                )
                                progress.set_description_str(task_label, refresh=True)

                                # Skip if library doesn't support this operation.
                                if operation not in lib_operations:
                                    progress.set_postfix_str("skipped: unsupported", refresh=True)
                                    progress.update(1)
                                    continue

                                # Prepare task configuration
                                task_config = build_task_config(
                                    sweep,
                                    library_name,
                                    N=case_N,
                                    d=d,
                                    m=m,
                                    path_kind=path_kind,
                                    operation=operation,
                                    repeats=repeats,
                                    batch_size=batch_size,
                                    seed=seed,
                                    backend_name=backend_name,
                                )
                                task_id = make_task_id(
                                    library_name,
                                    backend_name,
                                    task_config,
                                )
                                if path_kind.lower() == "brownian":
                                    task_config["input_cache_dir"] = str(
                                        run_dir / "inputs"
                                    )
                                if task_id in completed_tasks:
                                    progress.set_postfix_str("skipped: complete", refresh=True)
                                    progress.update(1)
                                    continue

                                progress.set_postfix_str("running", refresh=True)

                                # Run adapter based on type
                                try:
                                    if library_config["type"] == "python":
                                        result = run_python_adapter(
                                            library_name,
                                            library_config,
                                            task_config,
                                            env_overrides=backend.get("env", {}),
                                            worker=python_worker,
                                        )
                                    elif library_config["type"] == "julia":
                                        result = run_julia_adapter(
                                            library_name,
                                            library_config,
                                            task_config,
                                            env_overrides=backend.get("env", {}),
                                        )
                                    else:
                                        raise ValueError(
                                            "Unknown library type: "
                                            f"{library_config['type']}"
                                        )

                                    if result is None:
                                        raise RuntimeError(
                                            f"{task_label} returned no result"
                                        )

                                    append_task_results(
                                        writer,
                                        results_file,
                                        completed_file,
                                        task_id,
                                        result,
                                    )
                                    completed_tasks.add(task_id)
                                    progress.set_postfix_str("done", refresh=True)

                                except Exception as e:
                                    unsupported_reason = unsupported_adapter_reason(e)
                                    if unsupported_reason is not None:
                                        append_skipped_task(
                                            skipped_writer,
                                            skipped_file,
                                            completed_file,
                                            task_id,
                                            library_name,
                                            backend_name,
                                            task_config,
                                            unsupported_reason,
                                        )
                                        completed_tasks.add(task_id)
                                        progress.set_postfix_str(
                                            "skipped: unsupported",
                                            refresh=True,
                                        )
                                        _progress_write(
                                            f"Skipped {task_label}: "
                                            f"{unsupported_reason}",
                                            file=sys.stderr,
                                        )
                                        continue

                                    progress.set_postfix_str("failed", refresh=True)
                                    append_failed_task(
                                        failed_writer,
                                        failed_file,
                                        completed_file,
                                        task_id,
                                        library_name,
                                        backend_name,
                                        task_config,
                                        e,
                                    )
                                    completed_tasks.add(task_id)
                                    _progress_write(
                                        f"Failed {task_label}: {e}",
                                        file=sys.stderr,
                                    )
                                    if python_worker is not None:
                                        python_worker.close()
                                    continue
                                finally:
                                    progress.update(1)

                if python_worker is not None:
                    python_worker.close()

        progress.set_description_str("benchmarks complete", refresh=True)
        progress.set_postfix_str("", refresh=True)

    print("\n" + "=" * 60)
    print(f"Benchmark complete!")
    print(f"Results written to: {csv_path}")
    print(f"Total benchmarks: {count_result_rows(csv_path)}")
    skipped_count = count_result_rows(skipped_path)
    if skipped_count:
        print(f"Unsupported tasks: {skipped_count} (see {skipped_path})")
    failed_count = count_result_rows(failed_path)
    if failed_count:
        print(f"Failed tasks: {failed_count} (see {failed_path})")
    print("=" * 60)

    record_run_completion(run_dir, time.monotonic() - started_at)
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signature benchmark orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  uv run src/orchestrator.py

  # Run with custom config
  uv run src/orchestrator.py config/smoke.yaml

  # Continue an interrupted run
  uv run src/orchestrator.py --resume runs/benchmark_TIMESTAMP
        """
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=None,
        help="Path to benchmark sweep config (default: config/benchmark_sweep.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue an existing run directory using its saved configurations",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to a library registry (default: config/libraries_registry.yaml)",
    )

    args = parser.parse_args()
    run_orchestrator(
        config_path=args.config,
        resume_dir=args.resume,
        registry_path=args.registry,
    )
