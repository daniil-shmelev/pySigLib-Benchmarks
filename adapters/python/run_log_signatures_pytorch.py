#!/usr/bin/env python3
"""log-signatures-pytorch adapter for batched CPU and GPU benchmarks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from common import BenchmarkAdapter


class LogSignaturesPyTorchAdapter(BenchmarkAdapter):
    """Use the library's native leading-batch PyTorch API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import torch
        from log_signatures_pytorch import log_signature, signature

        self.torch = torch
        self.signature = signature
        self.log_signature = log_signature

    def _path_tensor(self, path: np.ndarray):
        if self.backend == "gpu":
            if not self.torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            device = "cuda"
        else:
            device = "cpu"
        return self.cached_prepared_input(
            f"log-signatures-pytorch.{device}",
            path,
            lambda: self.torch.as_tensor(
                np.ascontiguousarray(path, dtype=np.float32),
                dtype=self.torch.float32,
                device=device,
            ),
        )

    def _prepare(self, function: Callable[[Any], Any], path: Any) -> Callable[[], Any]:
        if self.backend == "gpu":
            function = self.torch.compile(function, mode="reduce-overhead")

        def synchronized():
            result = function(path)
            if self.backend == "gpu":
                self.torch.cuda.synchronize()
            return result

        return synchronized

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Callable:
        path = self._path_tensor(path)
        return self._prepare(
            lambda path_arg: self.signature(path_arg, depth=m),
            path,
        )

    def run_logsignature(self, path: np.ndarray, d: int, m: int) -> Callable:
        path = self._path_tensor(path)
        return self._prepare(
            lambda path_arg: self.log_signature(
                path_arg,
                depth=m,
                method="default",
                mode="words",
            ),
            path,
        )

    def run_sigdiff(self, path: np.ndarray, d: int, m: int) -> Callable:
        path = self._path_tensor(path)

        def loss(path_arg):
            return self.signature(path_arg, depth=m).sum()

        gradient = self.torch.func.grad(loss)
        return self._prepare(gradient, path)

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        path = self.make_path_input()
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "signature"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = "log_signature(default,words)"
        elif self.operation == "sigdiff":
            kernel = self.run_sigdiff(path, self.d, self.m)
            method = "torch.func.grad(signature)"
        else:
            return None

        if self.backend == "gpu":
            method += "[torch.compile(reduce-overhead)]"
        t_ms, alloc_bytes, samples_ms = self.manual_timing_loop(kernel)
        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            samples_ms=samples_ms,
            library="log-signatures-pytorch",
            method=method,
            path_type="torch.Tensor",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: run_log_signatures_pytorch.py '<json_config>'",
            file=sys.stderr,
        )
        sys.exit(1)

    LogSignaturesPyTorchAdapter(json.loads(sys.argv[1])).run()
