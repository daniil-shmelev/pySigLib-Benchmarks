#!/usr/bin/env python3
"""PathSig adapter for native batched CUDA benchmarks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from common import BenchmarkAdapter


class PathSigAdapter(BenchmarkAdapter):
    """Benchmark PathSig's compiled CUDA signature modules."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import pathsig
        import torch

        if self.backend != "gpu":
            raise ValueError("PathSig is GPU-only")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.pathsig = pathsig
        self.torch = torch

    @staticmethod
    def _prepare_paths(path: np.ndarray, d: int) -> np.ndarray:
        paths = np.ascontiguousarray(path, dtype=np.float32)
        if paths.ndim == 2:
            paths = paths[np.newaxis, ...]
        if paths.ndim != 3 or paths.shape[-1] != d:
            raise ValueError(
                f"path must have shape (N, {d}) or (batch_size, N, {d}), "
                f"got {paths.shape}"
            )
        return paths

    def _path_tensor(self, path: np.ndarray, d: int):
        paths = self._prepare_paths(path, d)
        return self.torch.as_tensor(
            paths,
            dtype=self.torch.float32,
            device="cuda",
        )

    def _synchronized(self, function: Callable[[], Any]) -> Callable[[], Any]:
        def kernel():
            result = function()
            self.torch.cuda.synchronize()
            return result

        return kernel

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Callable:
        path = self._path_tensor(path, d)
        module = self.pathsig.Signature(depth=m).eval()
        return self._synchronized(lambda: module(path))

    def run_logsignature(self, path: np.ndarray, d: int, m: int) -> Callable:
        path = self._path_tensor(path, d)
        projection = self.pathsig.projections.lyndon(depth=m, path_dim=d)
        module = self.pathsig.LogSignature(
            depth=m,
            projection=projection,
        ).eval()
        return self._synchronized(lambda: module(path))

    def run_sigdiff(self, path: np.ndarray, d: int, m: int) -> Callable:
        path = self._path_tensor(path, d).requires_grad_(True)
        module = self.pathsig.Signature(depth=m).eval()

        def gradient():
            signature = module(path)
            return self.torch.autograd.grad(signature.sum(), path)

        return self._synchronized(gradient)

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        path = self.make_path_input()
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "Signature(full)"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = "LogSignature(LyndonProjection)"
        elif self.operation == "sigdiff":
            kernel = self.run_sigdiff(path, self.d, self.m)
            method = "autograd.grad(Signature)"
        else:
            return None

        t_ms, alloc_bytes, samples_ms = self.manual_timing_loop(kernel)
        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            samples_ms=samples_ms,
            library="pathsig",
            method=method,
            path_type="torch.Tensor",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_pathsig.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    PathSigAdapter(json.loads(sys.argv[1])).run()
