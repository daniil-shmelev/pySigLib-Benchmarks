#!/usr/bin/env python3
"""keras_sig adapter for signature benchmarks"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Add src directory to path for common module
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from common import BenchmarkAdapter


class KerasSigAdapter(BenchmarkAdapter):
    """Adapter for keras_sig library"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # keras_sig imports Keras at module import time, so select the JAX backend
        # before importing it. This keeps timing comparable with the JAX adapters.
        os.environ["KERAS_BACKEND"] = "jax"

        # Import here to avoid import errors if not available.
        import jax
        import jax.numpy as jnp
        from keras_sig import jax_gpu_signature

        self.jax = jax
        self.jnp = jnp
        self.jax_gpu_signature = jax_gpu_signature

    def _path_array(self, path: np.ndarray):
        """Convert path data to a batched contiguous JAX array during setup."""
        if path.ndim == 2:
            path = path[None, :, :]
        elif path.ndim != 3:
            raise ValueError(
                f"path must have shape (N, d) or (batch_size, N, d), got {path.shape}"
            )
        path_np = np.ascontiguousarray(path, dtype=np.float32)
        return self.jnp.asarray(path_np, dtype=self.jnp.float32)

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path_jax = self._path_array(path)
        squeeze_output = self.batch_size == 1

        def signature_fn(path_arg):
            sig = self.jax_gpu_signature(path_arg, depth=m, stream=False)
            if squeeze_output:
                return self.jnp.squeeze(sig, axis=0)
            return sig

        signature_fn = self.jax.jit(signature_fn)

        return lambda: signature_fn(path_jax).block_until_ready()

    def run_sigdiff(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature differentiation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path_jax = self._path_array(path)

        def loss_fn(path_arg):
            sig = self.jax_gpu_signature(path_arg, depth=m, stream=False)
            return self.jnp.sum(sig)

        grad_fn = self.jax.jit(self.jax.grad(loss_fn))

        return lambda: grad_fn(path_jax).block_until_ready()

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """Execute the benchmark"""
        # Generate path input during setup. keras_sig's JAX API is batch-first.
        path = self.make_path_input()

        # Select operation
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "jax_gpu_signature"
        elif self.operation == "sigdiff":
            kernel = self.run_sigdiff(path, self.d, self.m)
            method = "jax.grad(jax_gpu_signature)"
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
            library="keras_sig",
            method=method,
            path_type="jax.Array",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_keras_sig.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    # Parse configuration from command line
    config = json.loads(sys.argv[1])

    # Create and run adapter
    adapter = KerasSigAdapter(config)
    adapter.run()
