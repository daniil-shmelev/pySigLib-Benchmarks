"""Base benchmark adapter with manual timing loop"""

import gc
import json
import statistics
import sys
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .paths import make_path, make_path_batch


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
                - path_kind: "linear", "sin", or "fbm"
                - operation: "signature", "logsignature", "sigdiff",
                  "branchedsignature_nonplanar", "branchedsignature_planar",
                  or "signaturekernel"
                - repeats: Number of timing repetitions
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
        self.batch_size = int(config.get("batch_size", 1))
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        self.backend = config.get("backend", "")
        self.seed = int(config.get("seed", 0))
        self._input_index = 0

    def make_path_input(self):
        """Generate this benchmark's input path, batched when requested."""
        # fbm uses NumPy's legacy global RNG internally. Reset it for each
        # logical input so every library/backend receives the same path data.
        np.random.seed(self.seed + self._input_index)
        self._input_index += 1
        if self.batch_size == 1:
            return make_path(self.d, self.N, self.path_kind)
        return make_path_batch(self.d, self.N, self.path_kind, self.batch_size)

    def manual_timing_loop(
        self,
        func: Callable[[], Any],
        warmup_iterations: int = 3
    ) -> Tuple[float, int, List[float]]:
        """
        Execute manual timing loop with warmup and GC disabled.

        Args:
            func: Function to time (should be a closure/lambda wrapping the kernel)
            warmup_iterations: Number of warmup runs before timing

        Returns:
            Tuple of (median_time_ms, alloc_bytes, samples_ms):
                - median_time_ms: Median time per iteration in milliseconds
                - alloc_bytes: Average bytes allocated per iteration
                - samples_ms: Raw time for every measured iteration
        """
        # Warmup phase (untimed)
        for _ in range(warmup_iterations):
            func()

        # Timed phase with GC disabled and allocation tracking
        gc.disable()
        tracemalloc.start()
        try:
            mem0_current, mem0_peak = tracemalloc.get_traced_memory()

            samples_ms = []
            for _ in range(self.repeats):
                t0 = time.perf_counter()
                func()
                samples_ms.append((time.perf_counter() - t0) * 1000.0)

            mem1_current, mem1_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
            gc.enable()

        median_time_ms = statistics.median(samples_ms)

        # Calculate average bytes allocated per iteration
        total_alloc_bytes = mem1_current - mem0_current
        avg_alloc_bytes = total_alloc_bytes // self.repeats if self.repeats > 0 else 0

        return median_time_ms, avg_alloc_bytes, samples_ms

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

    def run_sigdiff(self, path, d: int, m: int) -> Optional[Callable]:
        """
        Prepare and return kernel for signature differentiation.

        This method should be overridden by subclasses to return a callable
        that performs only the kernel computation (no setup).

        Args:
            path: The input path (format depends on library)
            d: Dimension
            m: Signature level

        Returns:
            Callable that performs the sigdiff computation, or None if not supported
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
        return {
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
            "t_ms": t_ms,
            "t_ms_mean": statistics.fmean(samples_ms),
            "t_ms_std": statistics.pstdev(samples_ms),
            "samples_ms": samples_ms,
            "alloc_bytes": alloc_bytes,
        }
