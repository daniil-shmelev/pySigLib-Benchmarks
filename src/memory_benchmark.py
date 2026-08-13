"""Peak-memory benchmark orchestration for signature adapters."""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from orchestrator import (
    REPO_ROOT,
    BenchmarkWorkerOOM,
    PythonAdapterWorker,
    backend_variants,
    build_sweep_cases,
    build_task_config,
    discard_uncommitted_rows,
    format_task_label,
    load_completed_tasks,
    load_yaml,
    make_task_id,
    record_run_completion,
    run_python_adapter,
    unsupported_adapter_reason,
    write_run_metadata,
)

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
    "memory_status",
    "host_memory_source",
    "host_baseline_bytes",
    "host_peak_bytes",
    "host_peak_delta_bytes",
    "gpu_memory_source",
    "gpu_baseline_allocated_bytes",
    "gpu_peak_allocated_bytes",
    "gpu_peak_allocated_delta_bytes",
    "gpu_peak_reserved_bytes",
]
TASK_FIELDS = [
    "task_id",
    "library",
    "backend",
    "operation",
    "N",
    "d",
    "m",
    "batch_size",
]
FAILED_FIELDS = [
    *TASK_FIELDS,
    "memory_status",
    "host_memory_source",
    "host_baseline_bytes",
    "host_peak_bytes",
    "host_peak_delta_bytes",
    "gpu_memory_source",
    "gpu_baseline_allocated_bytes",
    "gpu_peak_allocated_bytes",
    "gpu_peak_allocated_delta_bytes",
    "gpu_peak_reserved_bytes",
    "error_type",
    "reason",
]
SKIPPED_FIELDS = [*TASK_FIELDS, "reason"]


def _write_header(path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        csv.DictWriter(output, fieldnames=fields).writeheader()


def _flush_row(
    writer: csv.DictWriter,
    output,
    completed_output,
    task_id: str,
    row: dict[str, Any],
) -> None:
    writer.writerow(row)
    output.flush()
    completed_output.write(task_id + "\n")
    completed_output.flush()


def _task_identity(
    task_id: str,
    library: str,
    backend: str,
    task_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "library": library,
        "backend": backend,
        "operation": task_config["operation"],
        "N": task_config["N"],
        "d": task_config["d"],
        "m": task_config["m"],
        "batch_size": task_config["batch_size"],
    }


def _is_oom(error: Exception) -> bool:
    error_text = f"{type(error).__name__} {error}".lower()
    return (
        isinstance(error, (BenchmarkWorkerOOM, MemoryError))
        or "out of memory" in error_text
        or "oom" in error_text
    )


def _selected_libraries(
    sweep: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    registered = registry.get("libraries", {})
    selected = sweep.get("libraries")
    if selected is None:
        return registered
    if not isinstance(selected, list) or not all(
        isinstance(name, str) for name in selected
    ):
        raise ValueError("libraries must be a list of registry names")
    unknown = [name for name in selected if name not in registered]
    if unknown:
        raise ValueError("Unknown libraries in sweep: " + ", ".join(unknown))
    return {name: registered[name] for name in selected}


def run_signature_memory_benchmark(
    config_path: Path,
    *,
    registry_path: Path | None = None,
    resume_dir: Path | None = None,
) -> Path:
    """Run one peak-memory measurement for every configured signature cell."""
    started = time.monotonic()
    registry_path = registry_path or REPO_ROOT / "config" / "libraries_registry.yaml"
    if resume_dir is None:
        sweep_path = config_path.resolve()
        registry_source = registry_path.resolve()
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "runs" / f"signature_memory_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "memory_sweep.yaml").write_text(
            sweep_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (run_dir / "libraries_registry.yaml").write_text(
            registry_source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        write_run_metadata(run_dir, sweep_path)
    else:
        run_dir = resume_dir.resolve()
        sweep_path = run_dir / "memory_sweep.yaml"
        registry_source = run_dir / "libraries_registry.yaml"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    sweep = load_yaml(sweep_path)
    registry = load_yaml(registry_source)
    libraries = _selected_libraries(sweep, registry)
    selected_backends = sweep.get("backends")
    if selected_backends is not None:
        selected_backends = {str(value) for value in selected_backends}
    library_backends = {
        name: [
            backend
            for backend in backend_variants(config)
            if (
                selected_backends is None
                or backend.get("name", "") in selected_backends
            )
        ]
        for name, config in libraries.items()
    }

    Ns = [int(value) for value in sweep.get("Ns", [10000])]
    Ds = [int(value) for value in sweep.get("Ds", [2, 4, 8, 16])]
    Ms = [int(value) for value in sweep.get("Ms", [2, 3])]
    operations = [str(value) for value in sweep.get("operations", ["signature"])]
    path_kind = str(sweep.get("path_kind", "brownian"))
    batch_size = int(sweep.get("batch_size", 1))
    seed = int(sweep.get("seed", 0))
    sample_interval = float(sweep.get("memory_sample_interval_seconds", 0.005))
    if sample_interval <= 0:
        raise ValueError("memory_sample_interval_seconds must be > 0")
    if resume_dir is None:
        (run_dir / "memory_measurement.json").write_text(
            json.dumps(
                {
                    "gpu_metrics": "framework_allocator",
                    "host_metrics": "cgroup_or_process_tree",
                    "host_sample_interval_seconds": sample_interval,
                    "measured_calls": 1,
                    "mode": "peak_memory",
                    "warmup_iterations": 1,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    cases = build_sweep_cases(sweep, Ns, Ds, Ms, operations)
    if any(case["backend"] is not None for case in cases):
        raise ValueError("fixed backend-specific path_lengths are not yet supported")
    total_tasks = len(cases) * sum(
        len(backends) for backends in library_backends.values()
    )

    results_path = run_dir / "memory_results.csv"
    failures_path = run_dir / "failed_tasks.csv"
    skipped_path = run_dir / "skipped_tasks.csv"
    completed_path = run_dir / "completed_tasks.txt"
    if resume_dir is None:
        _write_header(results_path, RESULT_FIELDS)
        _write_header(failures_path, FAILED_FIELDS)
        _write_header(skipped_path, SKIPPED_FIELDS)
        completed_tasks: set[str] = set()
    else:
        completed_tasks = load_completed_tasks(completed_path)
        discard_uncommitted_rows(results_path, completed_tasks, RESULT_FIELDS)
        discard_uncommitted_rows(failures_path, completed_tasks, FAILED_FIELDS)
        discard_uncommitted_rows(skipped_path, completed_tasks, SKIPPED_FIELDS)

    print("\n" + "=" * 60)
    print("Signature Memory Benchmark")
    print("=" * 60)
    print("Calls per cell: 1")
    print("Warmup calls: 1")
    print(f"Host sampling interval: {sample_interval * 1000:g} ms")
    print(f"Operations: {operations}")
    print(f"Backends: {sorted(selected_backends) if selected_backends else 'all'}")
    print(f"Libraries: {list(libraries)}")
    print("=" * 60)

    with (
        results_path.open("a", newline="", encoding="utf-8") as results_output,
        failures_path.open("a", newline="", encoding="utf-8") as failures_output,
        skipped_path.open("a", newline="", encoding="utf-8") as skipped_output,
        completed_path.open("a", encoding="utf-8") as completed_output,
        tqdm(total=total_tasks, unit="cell", dynamic_ncols=True) as progress,
    ):
        results_writer = csv.DictWriter(
            results_output,
            fieldnames=RESULT_FIELDS,
            extrasaction="ignore",
        )
        failures_writer = csv.DictWriter(
            failures_output,
            fieldnames=FAILED_FIELDS,
            extrasaction="ignore",
        )
        skipped_writer = csv.DictWriter(
            skipped_output,
            fieldnames=SKIPPED_FIELDS,
            extrasaction="ignore",
        )
        for library_name, library_config in libraries.items():
            if library_config.get("type") != "python":
                raise ValueError(
                    f"Memory benchmark requires a Python adapter: {library_name}"
                )
            supported_operations = set(library_config.get("operations", []))
            for backend in library_backends[library_name]:
                backend_name = backend.get("name", "")
                worker = PythonAdapterWorker(
                    library_name,
                    library_config,
                    backend.get("env", {}),
                )
                try:
                    for case in cases:
                        task_config = build_task_config(
                            sweep,
                            library_name,
                            N=case["N"],
                            d=case["d"],
                            m=case["m"],
                            path_kind=path_kind,
                            operation=case["operation"],
                            repeats=1,
                            batch_size=batch_size,
                            seed=seed,
                            backend_name=backend_name,
                        )
                        task_config.update({
                            "memory_benchmark": True,
                            "memory_framework": library_config.get(
                                "memory_framework"
                            ),
                            "memory_sample_interval_seconds": sample_interval,
                            "repeats": 1,
                            "warmup_iterations": 1,
                        })
                        if path_kind.lower() == "brownian":
                            task_config["input_cache_dir"] = str(run_dir / "inputs")
                        task_id = make_task_id(
                            library_name,
                            backend_name,
                            task_config,
                        )
                        task_label = format_task_label(
                            library_name,
                            backend_name,
                            task_config["operation"],
                            N=task_config["N"],
                            d=task_config["d"],
                            m=task_config["m"],
                            batch_size=batch_size,
                        )
                        progress.set_description_str(task_label, refresh=True)
                        if task_id in completed_tasks:
                            progress.update(1)
                            continue
                        identity = _task_identity(
                            task_id,
                            library_name,
                            backend_name,
                            task_config,
                        )
                        if task_config["operation"] not in supported_operations:
                            _flush_row(
                                skipped_writer,
                                skipped_output,
                                completed_output,
                                task_id,
                                {**identity, "reason": "unsupported operation"},
                            )
                            completed_tasks.add(task_id)
                            progress.update(1)
                            continue
                        try:
                            result = run_python_adapter(
                                library_name,
                                library_config,
                                task_config,
                                env_overrides=backend.get("env", {}),
                                worker=worker,
                            )
                            if result is None:
                                raise RuntimeError(f"{task_label} returned no result")
                            _flush_row(
                                results_writer,
                                results_output,
                                completed_output,
                                task_id,
                                {
                                    **result,
                                    "task_id": task_id,
                                    "memory_status": "ok",
                                },
                            )
                        except Exception as error:  # noqa: BLE001
                            unsupported_reason = unsupported_adapter_reason(error)
                            if unsupported_reason is not None:
                                _flush_row(
                                    skipped_writer,
                                    skipped_output,
                                    completed_output,
                                    task_id,
                                    {**identity, "reason": unsupported_reason},
                                )
                            else:
                                _flush_row(
                                    failures_writer,
                                    failures_output,
                                    completed_output,
                                    task_id,
                                    {
                                        **identity,
                                        **worker.last_memory_metrics,
                                        "memory_status": (
                                            "oom" if _is_oom(error) else "failed"
                                        ),
                                        "error_type": type(error).__name__,
                                        "reason": str(error),
                                    },
                                )
                                print(f"\nFailed {task_label}: {error}", file=sys.stderr)
                            worker.close()
                        completed_tasks.add(task_id)
                        progress.update(1)
                finally:
                    worker.close()

    record_run_completion(run_dir, time.monotonic() - started)
    print(f"Memory results written to: {results_path}")
    return results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run signature memory benchmarks")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "config" / "paper_signatures_sweep.yaml",
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run_signature_memory_benchmark(args.config, resume_dir=args.resume)
