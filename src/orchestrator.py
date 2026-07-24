"""Orchestrator for signature benchmark suite"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from tqdm import tqdm


# Get script directory (src/)
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
COMPATIBILITY_PACKAGES_DIR = REPO_ROOT / ".adapter_packages"

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
    "t_ms",
    "t_ms_mean",
    "t_ms_std",
    "samples_ms",
    "alloc_bytes",
]
COMPLETED_TASKS_FILE = "completed_tasks.txt"


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


def discard_uncommitted_rows(csv_path: Path, completed_tasks: set[str]) -> None:
    """Remove rows written by a task that crashed before its completion marker."""
    with csv_path.open("r", newline="", encoding="utf-8") as results_file:
        reader = csv.DictReader(results_file)
        if reader.fieldnames != RESULT_FIELDS:
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
            fieldnames=RESULT_FIELDS,
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
    """Save the minimum environment metadata needed to identify a paper run."""
    git_status = _command_output(["git", "status", "--porcelain"])
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "sweep_config": str(sweep_config),
        "uv_version": _command_output(["uv", "--version"]),
        "julia_version": _command_output(["julia", "--version"]),
        "gpu": _command_output([
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]),
        "environment": {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORM_NAME",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "XLA_FLAGS",
            )
            if key in os.environ
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lockfile = REPO_ROOT / "uv.lock"
    if lockfile.exists():
        shutil.copy2(lockfile, run_dir / "uv.lock")


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


def run_python_adapter(
    library_name: str,
    library_config: Dict[str, Any],
    task_config: Dict[str, Any],
    *,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run a Python adapter using uv with project extras.

    Args:
        library_name: Name of the library
        library_config: Library configuration from registry
        task_config: Task parameters (N, d, m, etc.)

    Returns:
        Benchmark result dictionary
    """
    script_path = REPO_ROOT / library_config["script"]
    extras = list(library_config.get("extras", []))
    deps = library_config.get("deps", [])

    try:
        env = os.environ.copy()
        if env_overrides:
            env.update({key: str(value) for key, value in env_overrides.items()})

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

        # Build uv command with project optional dependencies. Adapters add
        # src/ to sys.path themselves.
        cmd = ["uv", "run"]
        for extra in extras:
            cmd.extend(["--extra", extra])
        for dep in deps:  # Legacy support for older registry entries.
            cmd.extend(["--with", dep])
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


def run_orchestrator(
    config_path: Optional[Path] = None,
    resume_dir: Optional[Path] = None,
):
    """
    Main orchestrator logic

    Args:
        config_path: Optional sweep config (default: config/benchmark_sweep.yaml)
        resume_dir: Existing run directory to continue
    """
    if config_path is not None and resume_dir is not None:
        raise ValueError("config_path and resume_dir cannot be used together")

    if resume_dir is not None:
        run_dir = resume_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        sweep_config = run_dir / "benchmark_sweep.yaml"
        registry_config = run_dir / "libraries_registry.yaml"
        print(f"Resuming run folder: {run_dir}")
    else:
        sweep_config = config_path if config_path else DEFAULT_SWEEP_CONFIG
        registry_config = REGISTRY_CONFIG

    # Load configurations
    sweep = load_yaml(sweep_config)
    registry = load_yaml(registry_config)

    # Extract sweep parameters
    Ns = sweep.get("Ns", [200, 400, 800])
    Ds = sweep.get("Ds", [2, 5, 7])
    Ms = sweep.get("Ms", [2, 3, 4])
    path_kind = sweep.get("path_kind", "sin")
    operations = sweep.get("operations", ["signature", "logsignature"])
    selected_backends = sweep.get("backends")
    if selected_backends is not None:
        selected_backends = {str(backend) for backend in selected_backends}
    repeats = sweep.get("repeats", 10)
    seed = int(sweep.get("seed", 0))
    batch_size = int(sweep.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
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
    if resume_dir is None:
        initialize_results_csv(csv_path)
        completed_tasks: set[str] = set()
    else:
        if not csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {csv_path}")
        completed_tasks = load_completed_tasks(completed_path)
        discard_uncommitted_rows(csv_path, completed_tasks)

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
    print(f"Seed: {seed}")
    print(f"Libraries: {list(registry.get('libraries', {}).keys())}")
    print("=" * 60)

    libraries = registry.get("libraries", {})

    # Run benchmarks
    library_backends = {
        name: [
            backend
            for backend in backend_variants(config)
            if selected_backends is None or backend.get("name", "") in selected_backends
        ]
        for name, config in libraries.items()
    }
    backend_task_count = sum(len(backends) for backends in library_backends.values())
    total_tasks = len(Ns) * len(Ds) * len(Ms) * len(operations) * backend_task_count

    with (
        csv_path.open("a", newline="", encoding="utf-8") as results_file,
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
        for library_name, library_config in libraries.items():
            lib_operations = set(library_config.get("operations", []))
            backends = library_backends[library_name]

            for backend in backends:
                backend_name = backend.get("name", "")

                for N in Ns:
                    for d in Ds:
                        for m in Ms:
                            for operation in operations:
                                task_label = format_task_label(
                                    library_name,
                                    backend_name,
                                    operation,
                                    N=N,
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
                                ):
                                    if key in sweep:
                                        task_config[key] = sweep[key]
                                library_task_config = (
                                    sweep.get("library_configs", {}).get(library_name, {})
                                )
                                task_config.update(library_task_config)
                                task_id = make_task_id(
                                    library_name,
                                    backend_name,
                                    task_config,
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
                                    progress.set_postfix_str("failed", refresh=True)
                                    _progress_write(
                                        f"Failed {task_label}: {e}",
                                        file=sys.stderr,
                                    )
                                    raise
                                finally:
                                    progress.update(1)

        progress.set_description_str("benchmarks complete", refresh=True)
        progress.set_postfix_str("", refresh=True)

    print("\n" + "=" * 60)
    print(f"Benchmark complete!")
    print(f"Results written to: {csv_path}")
    print(f"Total benchmarks: {count_result_rows(csv_path)}")
    print("=" * 60)

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

    args = parser.parse_args()
    run_orchestrator(config_path=args.config, resume_dir=args.resume)
