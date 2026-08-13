#!/usr/bin/env python3
"""Signatory adapter for native batched CPU and GPU benchmarks."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from common import BenchmarkAdapter


class SignatoryAdapter(BenchmarkAdapter):
    """Benchmark Signatory's native leading-batch PyTorch API."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        import signatory
        import torch

        self.signatory = signatory
        self.torch = torch

    def _path_tensor(self, path: np.ndarray, *, requires_grad: bool = False):
        if self.backend == "gpu":
            if not self.torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            device = "cuda"
        else:
            device = "cpu"
        namespace = f"signatory.{device}.grad={requires_grad}"
        return self.cached_prepared_input(
            namespace,
            path,
            lambda: self.torch.as_tensor(
                np.ascontiguousarray(path, dtype=np.float32),
                dtype=self.torch.float32,
                device=device,
            ).requires_grad_(requires_grad),
        )

    def _synchronized(self, function: Callable[[], Any]) -> Callable[[], Any]:
        def kernel():
            result = function()
            if self.backend == "gpu":
                self.torch.cuda.synchronize()
            return result

        return kernel

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Callable:
        path = self._path_tensor(path)
        module = self.signatory.Signature(depth=m).eval()
        return self._synchronized(lambda: module(path))

    def run_logsignature(self, path: np.ndarray, d: int, m: int) -> Callable:
        path = self._path_tensor(path)
        module = self.signatory.LogSignature(depth=m, mode="brackets").eval()
        return self._synchronized(lambda: module(path))

    def run_sig_backprop(self, path: np.ndarray, d: int, m: int) -> Callable:
        path = self._path_tensor(path, requires_grad=True)
        module = self.signatory.Signature(depth=m).eval()
        output = module(path)
        cotangent = self.torch.as_tensor(
            self.random_cotangent(tuple(output.shape)),
            dtype=output.dtype,
            device=output.device,
        )

        def gradient():
            return self.torch.autograd.grad(
                output,
                path,
                cotangent,
                retain_graph=True,
            )

        return self._synchronized(gradient)

    def run_logsignature_backprop(
        self, path: np.ndarray, d: int, m: int
    ) -> Callable:
        path = self._path_tensor(path, requires_grad=True)
        module = self.signatory.LogSignature(depth=m, mode="brackets").eval()
        output = module(path)
        cotangent = self.torch.as_tensor(
            self.random_cotangent(tuple(output.shape)),
            dtype=output.dtype,
            device=output.device,
        )

        def gradient():
            return self.torch.autograd.grad(
                output,
                path,
                cotangent,
                retain_graph=True,
            )

        return self._synchronized(gradient)

    def _run_benchmark(self) -> dict[str, Any] | None:
        path = self.make_path_input()
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "Signature"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = "LogSignature(brackets)"
        elif self.operation == "sig_backprop":
            kernel = self.run_sig_backprop(path, self.d, self.m)
            method = "Signature.backward(random_cotangent)"
        elif self.operation == "logsignature_backprop":
            kernel = self.run_logsignature_backprop(path, self.d, self.m)
            method = "LogSignature(brackets).backward(random_cotangent)"
        else:
            return None

        t_ms, alloc_bytes, samples_ms = self.manual_timing_loop(kernel)
        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            samples_ms=samples_ms,
            library="signatory",
            method=method,
            path_type="torch.Tensor",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_signatory.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    SignatoryAdapter(json.loads(sys.argv[1])).run()
