#!/usr/bin/env python3
"""polysigkernel adapter for signature-kernel benchmarks"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Add src directory to path for common module
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from common import BenchmarkAdapter


class PolySigKernelAdapter(BenchmarkAdapter):
    """Adapter for polysigkernel library"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Import here to avoid import errors if not available.
        import jax
        import jax.numpy as jnp
        from polysigkernel.sigkernel import SigKernel as SigKernel_polynomial

        self.jax = jax
        self.jnp = jnp
        self.SigKernel_polynomial = SigKernel_polynomial
        self.solver = config.get("sig_kernel_solver", "monomial_approx")

    def _path_batch(self, path: np.ndarray):
        """Convert path data to a rank-3 JAX array during setup."""
        path_np = np.ascontiguousarray(path, dtype=np.float32)
        if path_np.ndim == 2:
            path_np = path_np[None, :, :]
        return self.cached_prepared_input(
            "polysigkernel.jax",
            path,
            lambda: self.jnp.asarray(path_np, dtype=self.jnp.float32),
        )

    def run_signaturekernel(
        self,
        path1: np.ndarray,
        path2: np.ndarray,
        d: int,
        m: int,
    ) -> Optional[Callable]:
        """
        Prepare signature-kernel matrix computation.

        Returns a closure that performs only the kernel computation (no setup).
        """
        X = self._path_batch(path1)
        Y = self._path_batch(path2)
        order = int(self.config.get("sig_kernel_order", m))
        solver = self.solver
        max_batch_config = self.config.get("sig_kernel_max_batch")
        if max_batch_config is None:
            max_batch = None
        else:
            max_batch_int = int(max_batch_config)
            max_batch = None if max_batch_int <= 0 else max_batch_int

        sig_kernel = self.SigKernel_polynomial(
            order=order,
            static_kernel="linear",
            solver=solver,
            add_time=False,
        )

        return lambda: sig_kernel.kernel_matrix(X, Y, max_batch=max_batch).block_until_ready()

    def run_signaturekernel_backprop(
        self,
        path1: np.ndarray,
        path2: np.ndarray,
        d: int,
        m: int,
    ) -> Optional[Callable]:
        X = self._path_batch(path1)
        Y = self._path_batch(path2)
        order = int(self.config.get("sig_kernel_order", m))
        max_batch_config = self.config.get("sig_kernel_max_batch")
        if max_batch_config is None:
            max_batch = None
        else:
            max_batch_int = int(max_batch_config)
            max_batch = None if max_batch_int <= 0 else max_batch_int
        sig_kernel = self.SigKernel_polynomial(
            order=order,
            static_kernel="linear",
            solver=self.solver,
            add_time=False,
        )

        def signaturekernel_fn(X_arg):
            return sig_kernel.kernel_matrix(X_arg, Y, max_batch=max_batch)

        output, pullback = self.jax.vjp(signaturekernel_fn, X)
        cotangent = self.jnp.asarray(
            self.random_cotangent(output.shape),
            dtype=output.dtype,
        )
        backprop_fn = self.jax.jit(lambda cotangent_arg: pullback(cotangent_arg)[0])
        return lambda: backprop_fn(cotangent).block_until_ready()

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """Execute the benchmark"""
        if self.operation not in (
            "signaturekernel",
            "signature_kernel",
            "sigkernel",
            "signaturekernel_backprop",
        ):
            return None

        path1 = self.make_path_input()
        path2 = self.make_path_input()
        if self.operation == "signaturekernel_backprop":
            kernel = self.run_signaturekernel_backprop(
                path1,
                path2,
                self.d,
                self.m,
            )
            method_prefix = "vjp(SigKernel_polynomial)"
        else:
            kernel = self.run_signaturekernel(path1, path2, self.d, self.m)
            method_prefix = "SigKernel_polynomial"
        if kernel is None:
            return None

        t_ms, alloc_bytes, samples_ms = self.manual_timing_loop(kernel)

        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            samples_ms=samples_ms,
            library="polysigkernel",
            method=(
                method_prefix
                +
                f"(order={int(self.config.get('sig_kernel_order', self.m))}, "
                f"solver={self.solver})"
            ),
            path_type="jax.Array",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_polysigkernel.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    config = json.loads(sys.argv[1])

    adapter = PolySigKernelAdapter(config)
    adapter.run()
