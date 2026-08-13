"""Base benchmark adapter with manual timing loop"""

import gc
import hashlib
import json
import os
import statistics
import sys
import time
import tracemalloc
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .paths import make_path, make_path_batch


@lru_cache(maxsize=4)
def _load_stochastic_input(cache_path: str) -> np.ndarray:
    return np.load(cache_path, allow_pickle=False)


_PREPARED_INPUTS: OrderedDict[tuple[str, int], tuple[np.ndarray, Any]] = OrderedDict()
_MAX_PREPARED_INPUTS = 8


def clear_cached_inputs() -> None:
    _load_stochastic_input.cache_clear()
    _PREPARED_INPUTS.clear()
    gc.collect()


class BenchmarkAdapter:
    """
    Base class for benchmark adapters.

    Implements the manual timing loop pattern to ensure fairness:
    - Warmup phase (untimed)
    - Disable garbage collection
    - Manual loop with perf_counter
    - Average timing calculation
    - Re-enable garbage collection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adapter with configuration.

        Args:
            config: Configuration dictionary containing:
                - N: Number of path points
                - d: Dimension
                - m: Signature truncation level
                - path_kind: "linear", "sin", or "brownian"
                - operation: "signature", "logsignature", "sig_backprop",
                  "branchedsignature_nonplanar", "branchedsignature_planar",
                  or "signaturekernel"
                - repeats: Number of timing repetitions
                - warmup_iterations: Number of untimed warmup iterations
                - timing_statistic: Summary stored in t_ms ("median" or "min")
                - batch_size: Number of paths per timed kernel call
                - backend: Optional backend label, e.g. "cpu" or "gpu"
        """
        self.config = config
        self.N = config["N"]
        self.d = config["d"]
        self.m = config["m"]
        self.path_kind = config["path_kind"]
        self.operation = config["operation"]
        self.repeats = config["repeats"]
        if self.repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {self.repeats}")
        self.warmup_iterations = int(config.get("warmup_iterations", 3))
        if self.warmup_iterations < 0:
            raise ValueError(
                "warmup_iterations must be >= 0, "
                f"got {self.warmup_iterations}"
            )
        self.timing_statistic = config.get("timing_statistic", "median")
        if self.timing_statistic not in ("median", "min"):
            raise ValueError(
                "timing_statistic must be 'median' or 'min', "
                f"got {self.timing_statistic!r}"
            )
        self.call_timeout_seconds = config.get("call_timeout_seconds")
        self._call_event_callback = config.get("_call_event_callback")
        self.memory_benchmark = bool(config.get("memory_benchmark", False))
        self._memory_metrics: Dict[str, Any] = {}
        self.batch_size = int(config.get("batch_size", 1))
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        self.backend = config.get("backend", "")
        self.seed = int(config.get("seed", 0))
        self._input_index = 0

    def random_cotangent(self, shape) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal(tuple(shape)).astype(np.float32)

    def _cached_stochastic_input(self, logical_seed: int) -> Optional[np.ndarray]:
        """Load a saved stochastic input, or generate and save it atomically."""
        cache_dir = self.config.get("input_cache_dir")
        if cache_dir is None or self.path_kind.lower() != "brownian":
            return None

        cache_path = Path(cache_dir) / (
            f"brownian_seed{logical_seed}_N{self.N}_d{self.d}"
            f"_batch{self.batch_size}.npy"
        )
        checksum_path = cache_path.with_suffix(".npy.sha256")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not cache_path.exists():
            if self.batch_size == 1:
                path = make_path(self.d, self.N, self.path_kind)
            else:
                path = make_path_batch(
                    self.d,
                    self.N,
                    self.path_kind,
                    self.batch_size,
                )

            temporary_path = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.tmp"
            )
            with temporary_path.open("wb") as cache_file:
                np.save(cache_file, path, allow_pickle=False)
                cache_file.flush()
                os.fsync(cache_file.fileno())
            os.replace(temporary_path, cache_path)

        if not checksum_path.exists():
            digest_builder = hashlib.sha256()
            with cache_path.open("rb") as cache_file:
                for chunk in iter(lambda: cache_file.read(1024 * 1024), b""):
                    digest_builder.update(chunk)
            digest = digest_builder.hexdigest()
            temporary_checksum = checksum_path.with_name(
                f".{checksum_path.name}.{os.getpid()}.tmp"
            )
            temporary_checksum.write_text(
                f"{digest}  {cache_path.name}\n",
                encoding="utf-8",
            )
            os.replace(temporary_checksum, checksum_path)

        path = _load_stochastic_input(str(cache_path.resolve()))
        expected_shape = (
            (self.N, self.d)
            if self.batch_size == 1
            else (self.batch_size, self.N, self.d)
        )
        if path.shape != expected_shape or path.dtype != np.float32:
            raise ValueError(
                f"Invalid cached input {cache_path}: expected float32 shape "
                f"{expected_shape}, got {path.dtype} shape {path.shape}"
            )
        return path

    def cached_prepared_input(
        self,
        namespace: str,
        path: np.ndarray,
        prepare: Callable[[], Any],
    ) -> Any:
        """Reuse a host or device input while its source path remains cached."""
        key = (namespace, id(path))
        cached = _PREPARED_INPUTS.get(key)
        if cached is not None and cached[0] is path:
            _PREPARED_INPUTS.move_to_end(key)
            return cached[1]

        prepared = prepare()
        _PREPARED_INPUTS[key] = (path, prepared)
        _PREPARED_INPUTS.move_to_end(key)
        while len(_PREPARED_INPUTS) > _MAX_PREPARED_INPUTS:
            _PREPARED_INPUTS.popitem(last=False)
        return prepared

    def make_path_input(self):
        """Generate this benchmark's input path, batched when requested."""
        # Stochastic generators use NumPy's global RNG. Reset it for each
        # logical input so every library/backend receives the same path data.
        logical_seed = self.seed + self._input_index
        np.random.seed(logical_seed)
        self._input_index += 1

        cached_path = self._cached_stochastic_input(logical_seed)
        if cached_path is not None:
            return cached_path

        if self.batch_size == 1:
            return make_path(self.d, self.N, self.path_kind)
        return make_path_batch(self.d, self.N, self.path_kind, self.batch_size)

    def manual_timing_loop(
        self,
        func: Callable[[], Any],
        warmup_iterations: Optional[int] = None,
    ) -> Tuple[float, int, List[float]]:
        """
        Execute manual timing loop with warmup and GC disabled.

        Args:
            func: Function to time (should be a closure/lambda wrapping the kernel)
            warmup_iterations: Number of warmup runs before timing

        Returns:
            Tuple of (summary_time_ms, alloc_bytes, samples_ms):
                - summary_time_ms: Selected timing statistic in milliseconds
                - alloc_bytes: Average net traced Python heap change per iteration
                - samples_ms: Raw time for every measured iteration
        """
        if warmup_iterations is None:
            warmup_iterations = self.warmup_iterations
        if warmup_iterations < 0:
            raise ValueError(
                "warmup_iterations must be >= 0, "
                f"got {warmup_iterations}"
            )

        def invoke(phase: str, iteration: int) -> None:
            emit_call_event = (
                self._call_event_callback is not None
                and (
                    self.call_timeout_seconds is not None
                    or self.memory_benchmark
                )
            )
            if emit_call_event:
                event = {
                    "status": "call_start",
                    "phase": phase,
                    "iteration": iteration,
                }
                if self.call_timeout_seconds is not None:
                    event["timeout_seconds"] = self.call_timeout_seconds
                self._call_event_callback(event)
            try:
                func()
            finally:
                if emit_call_event:
                    self._call_event_callback({
                        "status": "call_end",
                        "phase": phase,
                        "iteration": iteration,
                    })

        if self.memory_benchmark:
            for iteration in range(warmup_iterations):
                gpu_before = self._gpu_memory_snapshot(reset_peak=True)
                try:
                    invoke("warmup", iteration)
                finally:
                    self._memory_metrics = self._summarize_gpu_memory(
                        gpu_before,
                        self._gpu_memory_snapshot(),
                    )
            gc.collect()
            gpu_before = self._gpu_memory_snapshot(reset_peak=True)
            t0 = time.perf_counter()
            try:
                invoke("measured", 0)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                gpu_after = self._gpu_memory_snapshot()
                self._memory_metrics = self._summarize_gpu_memory(
                    gpu_before,
                    gpu_after,
                )
            return elapsed_ms, 0, [elapsed_ms]

        # Warmup phase (untimed)
        for iteration in range(warmup_iterations):
            invoke("warmup", iteration)

        # Timed phase with GC disabled and allocation tracking
        gc.disable()
        tracemalloc.start()
        try:
            mem0_current, mem0_peak = tracemalloc.get_traced_memory()

            samples_ms = []
            for iteration in range(self.repeats):
                t0 = time.perf_counter()
                invoke("measured", iteration)
                samples_ms.append((time.perf_counter() - t0) * 1000.0)

            mem1_current, mem1_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
            gc.enable()

        if self.timing_statistic == "min":
            summary_time_ms = min(samples_ms)
        else:
            summary_time_ms = statistics.median(samples_ms)

        # This tracks only the net Python heap change visible to tracemalloc.
        # It excludes native allocations, framework pools, and device memory.
        total_alloc_bytes = mem1_current - mem0_current
        avg_alloc_bytes = total_alloc_bytes // self.repeats if self.repeats > 0 else 0

        return summary_time_ms, avg_alloc_bytes, samples_ms

    def _gpu_memory_snapshot(self, reset_peak: bool = False) -> Dict[str, Any]:
        """Read allocator memory statistics for the active GPU framework."""
        if self.backend != "gpu":
            return {}

        memory_framework = self.config.get("memory_framework")
        torch = sys.modules.get("torch")
        if (
            memory_framework in (None, "torch")
            and torch is not None
            and torch.cuda.is_available()
        ):
            device = torch.cuda.current_device()
            torch.cuda.synchronize(device)
            if reset_peak:
                torch.cuda.reset_peak_memory_stats(device)
            return {
                "source": "torch_cuda_allocator",
                "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }

        if memory_framework not in (None, "jax"):
            return {}
        jax = sys.modules.get("jax")
        if jax is None:
            return {}
        try:
            device = next(
                device for device in jax.devices()
                if device.platform == "gpu"
            )
            stats = device.memory_stats() or {}
        except (RuntimeError, StopIteration):
            return {}
        return {
            "source": "jax_device_stats",
            "current_allocated_bytes": stats.get("bytes_in_use"),
            "peak_allocated_bytes": stats.get("peak_bytes_in_use"),
            "peak_reserved_bytes": stats.get("peak_pool_bytes"),
        }

    @staticmethod
    def _summarize_gpu_memory(
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert framework snapshots into result columns."""
        if not before and not after:
            return {}
        baseline = before.get("current_allocated_bytes")
        peak = after.get("peak_allocated_bytes")
        delta = None
        if baseline is not None and peak is not None:
            delta = max(0, int(peak) - int(baseline))
        return {
            "gpu_memory_source": after.get("source") or before.get("source"),
            "gpu_baseline_allocated_bytes": baseline,
            "gpu_peak_allocated_bytes": peak,
            "gpu_peak_allocated_delta_bytes": delta,
            "gpu_peak_reserved_bytes": after.get("peak_reserved_bytes"),
        }

    def run_signature(self, path, d: int, m: int) -> Optional[Callable]:
        """
        Prepare and return kernel for signature computation.

        This method should be overridden by subclasses to return a callable
        that performs only the kernel computation (no setup).

        Args:
            path: The input path (format depends on library)
            d: Dimension
            m: Signature level

        Returns:
            Callable that performs the signature computation, or None if not supported
        """
        raise NotImplementedError("Subclasses must implement run_signature")

    def run_logsignature(self, path, d: int, m: int) -> Optional[Callable]:
        """
        Prepare and return kernel for logsignature computation.

        This method should be overridden by subclasses to return a callable
        that performs only the kernel computation (no setup).

        Args:
            path: The input path (format depends on library)
            d: Dimension
            m: Signature level

        Returns:
            Callable that performs the logsignature computation, or None if not supported
        """
        return None  # Default: not supported

    def run_sig_backprop(self, path, d: int, m: int) -> Optional[Callable]:
        """
        Prepare and return a signature backpropagation kernel.

        This method should be overridden by subclasses to return a callable
        that performs only the kernel computation (no setup).

        Args:
            path: The input path (format depends on library)
            d: Dimension
            m: Signature level

        Returns:
            Callable that performs signature backpropagation, or None if unsupported
        """
        return None  # Default: not supported

    def run_branchedsignature(
        self,
        path,
        d: int,
        m: int,
        *,
        planar: bool,
    ) -> Optional[Callable]:
        """
        Prepare and return kernel for branched signature computation.

        This method should be overridden by subclasses to return a callable
        that performs only the kernel computation (no setup).

        Args:
            path: The input path (format depends on library)
            d: Dimension
            m: Signature level
            planar: Whether to compute planar rather than non-planar trees

        Returns:
            Callable that performs the branched signature computation, or None
            if not supported
        """
        return None  # Default: not supported

    def run_signaturekernel(self, path1, path2, d: int, m: int) -> Optional[Callable]:
        """
        Prepare and return kernel for signature-kernel matrix computation.

        This method should be overridden by subclasses to return a callable
        that performs only the kernel computation (no setup).

        Args:
            path1: The first input path batch (format depends on library)
            path2: The second input path batch (format depends on library)
            d: Dimension
            m: Signature-kernel approximation parameter for this benchmark

        Returns:
            Callable that computes the signature-kernel matrix, or None
            if not supported
        """
        return None  # Default: not supported

    def run(self) -> None:
        """
        Execute the benchmark and output results as JSON to stdout.

        This is the main entry point called by the orchestrator.
        """
        try:
            result = self._run_benchmark()
            if result is not None:
                # Output as single-line JSON for orchestrator to parse
                print(json.dumps(result), flush=True)
        except Exception as e:
            error_result = {
                "error": str(e),
                "N": self.N,
                "d": self.d,
                "m": self.m,
                "operation": self.operation,
            }
            print(json.dumps(error_result), file=sys.stderr, flush=True)
            sys.exit(1)

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """
        Internal method to run the actual benchmark.

        Returns:
            Dictionary with benchmark results, or None if operation not supported
        """
        raise NotImplementedError("Subclasses must implement _run_benchmark")

    def output_result(
        self,
        t_ms: float,
        alloc_bytes: int,
        samples_ms: List[float],
        library: str,
        method: str,
        path_type: str = "ndarray",
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Format benchmark result for output.

        Args:
            t_ms: Time in milliseconds
            alloc_bytes: Bytes allocated
            samples_ms: Raw timing samples in milliseconds
            library: Library name
            method: Method name
            path_type: Path type descriptor
            language: Programming language

        Returns:
            Formatted result dictionary
        """
        result = {
            "N": self.N,
            "d": self.d,
            "m": self.m,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "path_kind": self.path_kind,
            "operation": self.operation,
            "backend": self.backend,
            "language": language,
            "library": library,
            "method": method,
            "path_type": path_type,
            "timing_statistic": self.timing_statistic,
            "t_ms": t_ms,
            "t_ms_mean": statistics.fmean(samples_ms),
            "t_ms_std": statistics.pstdev(samples_ms),
            "samples_ms": samples_ms,
            "alloc_bytes": alloc_bytes,
        }
        result.update(self._memory_metrics)
        return result
