#!/usr/bin/env python3
"""RoughPy adapter for signature benchmarks."""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from common import BenchmarkAdapter


class RoughPyAdapter(BenchmarkAdapter):
    """CPU adapter for RoughPy's single-precision stream API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import roughpy

        self.roughpy = roughpy

    def _prepare_paths(
        self,
        path: np.ndarray,
        d: int,
        m: int,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Prepare f32 increments, indices, and the reusable algebra context."""
        paths = np.ascontiguousarray(path, dtype=np.float32)
        if paths.ndim == 2:
            paths = paths[np.newaxis, ...]

        increments = np.ascontiguousarray(np.diff(paths, axis=1), dtype=np.float32)
        indices = np.arange(paths.shape[1] - 1, dtype=np.float32)
        context = self.roughpy.get_context(d, m, self.roughpy.SPReal)
        return increments, indices, context

    def _stream_kernel(
        self,
        path: np.ndarray,
        d: int,
        m: int,
        operation: str,
    ) -> Callable:
        increments, indices, context = self._prepare_paths(path, d, m)
        roughpy = self.roughpy

        def kernel():
            results = []
            for sample in increments:
                stream = roughpy.LieIncrementStream.from_increments(
                    sample,
                    indices=indices,
                    ctx=context,
                )
                results.append(getattr(stream, operation)())
            return results

        return kernel

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        return self._stream_kernel(path, d, m, "signature")

    def run_logsignature(
        self,
        path: np.ndarray,
        d: int,
        m: int,
    ) -> Optional[Callable]:
        return self._stream_kernel(path, d, m, "log_signature")

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        path = self.make_path_input()

        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "LieIncrementStream.signature(SPReal,batch_loop)"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = "LieIncrementStream.log_signature(SPReal,batch_loop)"
        else:
            return None

        if kernel is None:
            return None

        t_ms, alloc_bytes, samples_ms = self.manual_timing_loop(kernel)
        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            samples_ms=samples_ms,
            library="RoughPy",
            method=method,
            path_type="ndarray -> LieIncrementStream",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_roughpy.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    RoughPyAdapter(json.loads(sys.argv[1])).run()
