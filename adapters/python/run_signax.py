#!/usr/bin/env python3
"""signax adapter for signature benchmarks"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Add src directory to path for common module
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from common import BenchmarkAdapter


class SignaxAdapter(BenchmarkAdapter):
    """Adapter for signax library"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Import here to avoid import errors if not available.
        import jax
        import jax.numpy as jnp
        import signax

        self.jax = jax
        self.jnp = jnp
        self.signax = signax
        self.num_chunks = int(config.get("num_chunks", 1))

    def _path_array(self, path: np.ndarray):
        """Convert path data to a contiguous JAX array during setup."""
        return self.cached_prepared_input(
            "signax.jax",
            path,
            lambda: self.jnp.asarray(
                np.ascontiguousarray(path, dtype=np.float32),
                dtype=self.jnp.float32,
            ),
        )

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path_jax = self._path_array(path)
        num_chunks = self.num_chunks

        def signature_fn(path_arg):
            return self.signax.signature(
                path_arg,
                depth=m,
                stream=False,
                flatten=True,
                num_chunks=num_chunks,
            )

        signature_fn = self.jax.jit(signature_fn)

        return lambda: signature_fn(path_jax).block_until_ready()

    def run_logsignature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare logsignature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path_jax = self._path_array(path)
        num_chunks = self.num_chunks

        def logsignature_fn(path_arg):
            return self.signax.logsignature(
                path_arg,
                depth=m,
                stream=False,
                flatten=True,
                num_chunks=num_chunks,
            )

        logsignature_fn = self.jax.jit(logsignature_fn)

        return lambda: logsignature_fn(path_jax).block_until_ready()

    def run_sig_backprop(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature backpropagation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path_jax = self._path_array(path)
        num_chunks = self.num_chunks

        def signature_fn(path_arg):
            return self.signax.signature(
                path_arg,
                depth=m,
                stream=False,
                flatten=True,
                num_chunks=num_chunks,
            )

        output, pullback = self.jax.vjp(signature_fn, path_jax)
        cotangent = self.jnp.asarray(
            self.random_cotangent(output.shape),
            dtype=output.dtype,
        )
        backprop_fn = self.jax.jit(lambda cotangent_arg: pullback(cotangent_arg)[0])
        return lambda: backprop_fn(cotangent).block_until_ready()

    def run_logsignature_backprop(
        self, path: np.ndarray, d: int, m: int
    ) -> Optional[Callable]:
        path_jax = self._path_array(path)
        num_chunks = self.num_chunks

        def logsignature_fn(path_arg):
            return self.signax.logsignature(
                path_arg,
                depth=m,
                stream=False,
                flatten=True,
                num_chunks=num_chunks,
            )

        output, pullback = self.jax.vjp(logsignature_fn, path_jax)
        cotangent = self.jnp.asarray(
            self.random_cotangent(output.shape),
            dtype=output.dtype,
        )
        backprop_fn = self.jax.jit(lambda cotangent_arg: pullback(cotangent_arg)[0])
        return lambda: backprop_fn(cotangent).block_until_ready()

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """Execute the benchmark"""
        # Generate path input during setup. signax dispatches rank-3 paths to
        # its batched vmap path internally.
        path = self.make_path_input()

        # Select operation
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = f"signature(num_chunks={self.num_chunks})"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = f"logsignature(num_chunks={self.num_chunks})"
        elif self.operation == "sig_backprop":
            kernel = self.run_sig_backprop(path, self.d, self.m)
            method = f"vjp(signature, num_chunks={self.num_chunks})"
        elif self.operation == "logsignature_backprop":
            kernel = self.run_logsignature_backprop(path, self.d, self.m)
            method = f"vjp(logsignature, num_chunks={self.num_chunks})"
        else:
            # Operation not supported
            return None

        if kernel is None:
            return None

        # Run manual timing loop
        t_ms, alloc_bytes, samples_ms = self.manual_timing_loop(kernel)

        # Format and return result
        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            samples_ms=samples_ms,
            library="signax",
            method=method,
            path_type="jax.Array",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_signax.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    # Parse configuration from command line
    config = json.loads(sys.argv[1])

    # Create and run adapter
    adapter = SignaxAdapter(config)
    adapter.run()
