#!/usr/bin/env python3
"""signature-rs Python adapter for CPU log-signature benchmarks."""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from common import BenchmarkAdapter


class SignatureRSAdapter(BenchmarkAdapter):
    """CPU adapter for the float32 signature-py binding."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import signature_py

        self.signature_py = signature_py

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

    def run_logsignature(
        self,
        path: np.ndarray,
        d: int,
        m: int,
    ) -> Callable:
        paths = self._prepare_paths(path, d)
        builder = self.signature_py.LogSignatureBuilder(
            max_degree=m,
            num_dimensions=d,
        )

        def kernel():
            return [builder.build_from_path(sample) for sample in paths]

        return kernel

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        if self.operation != "logsignature":
            return None

        kernel = self.run_logsignature(self.make_path_input(), self.d, self.m)
        t_ms, alloc_bytes, samples_ms = self.manual_timing_loop(kernel)
        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            samples_ms=samples_ms,
            library="signature-rs",
            method="LogSignatureBuilder.build_from_path(f32,batch_loop)",
            path_type="ndarray",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_signature_rs.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    SignatureRSAdapter(json.loads(sys.argv[1])).run()
