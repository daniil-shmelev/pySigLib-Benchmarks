#!/usr/bin/env python3
"""TensorDev adapter for signature benchmarks."""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from common import BenchmarkAdapter


class TensorDevAdapter(BenchmarkAdapter):
    """Adapter for TensorDev's JAX signature implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        import jax
        import jax.numpy as jnp
        import tensordev

        self.jax = jax
        self.jnp = jnp
        self.tensordev = tensordev

    def _path_array(self, path: np.ndarray):
        """Convert path data to a contiguous float32 JAX array during setup."""
        return self.cached_prepared_input(
            "tensordev.jax",
            path,
            lambda: self.jnp.asarray(
                np.ascontiguousarray(path, dtype=np.float32),
                dtype=self.jnp.float32,
            ),
        )

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """Prepare TensorDev's native batched signature computation."""
        path_jax = self._path_array(path)

        def signature_fn(path_arg):
            return self.tensordev.path_signature(path_arg, trunc=m)

        signature_fn = self.jax.jit(signature_fn)

        return lambda: self.jax.block_until_ready(signature_fn(path_jax))

    def run_logsignature(
        self, path: np.ndarray, d: int, m: int
    ) -> Optional[Callable]:
        """Prepare TensorDev's expanded word-basis tensor log-signature."""
        path_jax = self._path_array(path)

        def logsignature_fn(path_arg):
            signature = self.tensordev.path_signature(path_arg, trunc=m)
            return self.tensordev.tensor_logarithm(
                signature[1:],
                trunc=m,
                output_zero_level=False,
            )

        logsignature_fn = self.jax.jit(logsignature_fn)

        return lambda: self.jax.block_until_ready(logsignature_fn(path_jax))

    def run_sig_backprop(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """Prepare differentiation through TensorDev's path signature."""
        path_jax = self._path_array(path)

        def signature_fn(path_arg):
            return self.tensordev.path_signature(path_arg, trunc=m)[1:]

        output, pullback = self.jax.vjp(signature_fn, path_jax)
        cotangent = self.jax.tree.map(
            lambda level: self.jnp.asarray(
                self.random_cotangent(level.shape),
                dtype=level.dtype,
            ),
            output,
        )
        backprop_fn = self.jax.jit(lambda cotangent_arg: pullback(cotangent_arg)[0])
        return lambda: backprop_fn(cotangent).block_until_ready()

    def run_logsignature_backprop(
        self, path: np.ndarray, d: int, m: int
    ) -> Optional[Callable]:
        path_jax = self._path_array(path)

        def logsignature_fn(path_arg):
            signature = self.tensordev.path_signature(path_arg, trunc=m)
            return self.tensordev.tensor_logarithm(
                signature[1:],
                trunc=m,
                output_zero_level=False,
            )

        output, pullback = self.jax.vjp(logsignature_fn, path_jax)
        cotangent = self.jax.tree.map(
            lambda level: self.jnp.asarray(
                self.random_cotangent(level.shape),
                dtype=level.dtype,
            ),
            output,
        )
        backprop_fn = self.jax.jit(lambda cotangent_arg: pullback(cotangent_arg)[0])
        return lambda: backprop_fn(cotangent).block_until_ready()

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """Execute the configured benchmark."""
        path = self.make_path_input()

        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "path_signature"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = "tensor_logarithm(path_signature, word_basis)"
        elif self.operation == "sig_backprop":
            kernel = self.run_sig_backprop(path, self.d, self.m)
            method = "vjp(path_signature)"
        elif self.operation == "logsignature_backprop":
            kernel = self.run_logsignature_backprop(path, self.d, self.m)
            method = "vjp(tensor_logarithm(path_signature, word_basis))"
        else:
            return None

        if kernel is None:
            return None

        t_ms, alloc_bytes, samples_ms = self.manual_timing_loop(kernel)

        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            samples_ms=samples_ms,
            library="tensordev",
            method=method,
            path_type="jax.Array",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_tensordev.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    TensorDevAdapter(json.loads(sys.argv[1])).run()
