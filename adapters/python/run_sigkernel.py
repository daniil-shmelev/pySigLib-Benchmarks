#!/usr/bin/env python3
"""Sigkernel adapter for PyTorch signature-kernel benchmarks."""

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


class SigkernelAdapter(BenchmarkAdapter):
    """Benchmark Sigkernel's differentiable signature-PDE Gram API."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        import sigkernel
        import torch

        self.sigkernel = sigkernel
        self.torch = torch
        default_dtype = "float32" if self.backend == "gpu" else "float64"
        self.dtype_name = str(config.get("dtype", default_dtype))
        if self.dtype_name == "float32":
            self.numpy_dtype = np.float32
            self.torch_dtype = torch.float32
        elif self.dtype_name == "float64":
            self.numpy_dtype = np.float64
            self.torch_dtype = torch.float64
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype_name}")
        self.dyadic_order = int(config.get("sig_kernel_dyadic_order", 0))
        self.configured_max_batch = int(
            config.get("sig_kernel_max_batch", -1)
        )

    def _path_tensor(
        self,
        path: np.ndarray,
        *,
        requires_grad: bool = False,
    ):
        if self.backend == "gpu":
            if not self.torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            device = "cuda"
        else:
            device = "cpu"
        namespace = (
            f"sigkernel.{device}.{self.dtype_name}.grad={requires_grad}"
        )
        return self.cached_prepared_input(
            namespace,
            path,
            lambda: self.torch.as_tensor(
                np.ascontiguousarray(path, dtype=self.numpy_dtype),
                dtype=self.torch_dtype,
                device=device,
            ).requires_grad_(requires_grad),
        )

    def _signature_kernel(self):
        return self.sigkernel.SigKernel(
            self.sigkernel.LinearKernel(),
            self.dyadic_order,
        )

    def _max_batch(self, X, Y) -> int:
        if self.configured_max_batch > 0:
            return self.configured_max_batch
        return max(int(X.shape[0]), int(Y.shape[0]))

    def _synchronize(self, result):
        if self.backend == "gpu":
            self.torch.cuda.synchronize()
        return result

    def run_signaturekernel(
        self,
        path1: np.ndarray,
        path2: np.ndarray,
        d: int,
        m: int,
    ) -> Callable:
        X = self._path_tensor(path1)
        Y = self._path_tensor(path2)
        signature_kernel = self._signature_kernel()
        max_batch = self._max_batch(X, Y)

        def kernel():
            result = signature_kernel.compute_Gram(
                X,
                Y,
                sym=False,
                max_batch=max_batch,
            )
            return self._synchronize(result)

        return kernel

    def run_signaturekernel_backprop(
        self,
        path1: np.ndarray,
        path2: np.ndarray,
        d: int,
        m: int,
    ) -> Callable:
        X = self._path_tensor(path1, requires_grad=True)
        Y = self._path_tensor(path2)
        signature_kernel = self._signature_kernel()
        max_batch = self._max_batch(X, Y)
        output = signature_kernel.compute_Gram(
            X,
            Y,
            sym=False,
            max_batch=max_batch,
        )
        self._synchronize(output)
        cotangent = self.torch.as_tensor(
            self.random_cotangent(tuple(output.shape)),
            dtype=X.dtype,
            device=X.device,
        )

        def gradient():
            X.grad = None
            output.backward(cotangent, retain_graph=True)
            return self._synchronize(X.grad)

        return gradient

    def run_signaturekernel_fwd_bwd(self, path1, path2, d, m) -> Callable:
        X = self._path_tensor(path1, requires_grad=True)
        Y = self._path_tensor(path2)
        signature_kernel = self._signature_kernel()
        max_batch = self._max_batch(X, Y)
        cotangent = self.torch.as_tensor(
            self.random_cotangent((X.shape[0], Y.shape[0])),
            dtype=X.dtype, device=X.device,
        )

        def kernel():
            # compute_Gram also prepares derivatives: it must be timed here.
            output = signature_kernel.compute_Gram(X, Y, sym=False, max_batch=max_batch)
            gradient, = self.torch.autograd.grad(output, X, cotangent)
            return self._synchronize((output, gradient))

        return kernel

    def _run_benchmark(self) -> dict[str, Any] | None:
        if self.operation not in (
            "signaturekernel",
            "signaturekernel_backprop",
            "signaturekernel_fwd_bwd",
        ):
            return None

        path1 = self.make_path_input()
        path2 = self.make_path_input()
        if self.operation == "signaturekernel_fwd_bwd":
            kernel = self.run_signaturekernel_fwd_bwd(path1, path2, self.d, self.m)
            method_prefix = "forward+backward(SigKernel.compute_Gram)"
        elif self.operation == "signaturekernel_backprop":
            kernel = self.run_signaturekernel_backprop(
                path1,
                path2,
                self.d,
                self.m,
            )
            method_prefix = "backward(SigKernel.compute_Gram)"
        else:
            kernel = self.run_signaturekernel(
                path1,
                path2,
                self.d,
                self.m,
            )
            method_prefix = "SigKernel.compute_Gram"
        t_ms, alloc_bytes, samples_ms = self.manual_timing_loop(kernel)
        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            samples_ms=samples_ms,
            library="sigkernel",
            method=(
                f"{method_prefix}(dyadic_order={self.dyadic_order},linear,"
                f"dtype={self.dtype_name})"
            ),
            path_type="torch.Tensor",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_sigkernel.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    SigkernelAdapter(json.loads(sys.argv[1])).run()
