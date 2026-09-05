#!/usr/bin/env python3
"""Sigkerax adapter for JAX signature-kernel benchmarks."""

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


class SigkeraxAdapter(BenchmarkAdapter):
    """Benchmark Sigkerax's batched JAX signature-PDE kernel."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        import jax
        import jax.numpy as jnp
        from sigkerax.sigkernel import SigKernel
        from sigkerax.solver import FiniteDifferenceSolver

        self.jax = jax
        self.jnp = jnp
        self.dtype_name = str(config.get("dtype", "float32"))
        if self.dtype_name == "float32":
            self.numpy_dtype = np.float32
            self.jax_dtype = jnp.float32
        elif self.dtype_name == "float64":
            jax.config.update("jax_enable_x64", True)
            self.numpy_dtype = np.float64
            self.jax_dtype = jnp.float64
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype_name}")
        self.SigKernel = SigKernel
        self.refinement_factor = float(
            config.get("sig_kernel_refinement_factor", 1)
        )
        self._install_current_jax_compatibility(FiniteDifferenceSolver)

    def _install_current_jax_compatibility(self, solver_class) -> None:
        """Adapt two obsolete JAX annotations in Sigkerax 0.2.1."""
        if getattr(solver_class, "_sig_benchmarks_compatibility", False):
            return

        update_impl = solver_class._solution_diag_update.__wrapped__
        solver_class._solution_diag_update = self.jax.jit(
            update_impl,
            static_argnums=(0,),
        )

        def solve_single_device(solver, X, Y, directions):
            return solver._solve(X, Y, directions)

        solver_class.solve = self.jax.jit(
            solve_single_device,
            static_argnums=(0,),
        )
        solver_class._sig_benchmarks_compatibility = True

    def _path_batch(self, path: np.ndarray):
        path_np = np.ascontiguousarray(path, dtype=self.numpy_dtype)
        if path_np.ndim == 2:
            path_np = path_np[None, :, :]
        return self.cached_prepared_input(
            f"sigkerax.jax.{self.dtype_name}",
            path,
            lambda: self.jnp.asarray(path_np, dtype=self.jax_dtype),
        )

    def _signature_kernel(self):
        return self.SigKernel(
            refinement_factor=self.refinement_factor,
            static_kernel_kind="linear",
            scales=self.jnp.asarray([1.0], dtype=self.jax_dtype),
            add_time=False,
        )

    def run_signaturekernel(
        self,
        path1: np.ndarray,
        path2: np.ndarray,
        d: int,
        m: int,
    ) -> Callable:
        X = self._path_batch(path1)
        Y = self._path_batch(path2)
        signature_kernel = self._signature_kernel()

        def kernel():
            result = signature_kernel.kernel_matrix(X, Y)[..., 0]
            return result.block_until_ready()

        return kernel

    def run_signaturekernel_backprop(
        self,
        path1: np.ndarray,
        path2: np.ndarray,
        d: int,
        m: int,
    ) -> Callable:
        X = self._path_batch(path1)
        Y = self._path_batch(path2)
        signature_kernel = self._signature_kernel()

        def signaturekernel_fn(X_arg):
            return signature_kernel.kernel_matrix(X_arg, Y)[..., 0]

        output, pullback = self.jax.vjp(signaturekernel_fn, X)
        cotangent = self.jnp.asarray(
            self.random_cotangent(output.shape),
            dtype=output.dtype,
        )
        backprop_fn = self.jax.jit(
            lambda cotangent_arg: pullback(cotangent_arg)[0]
        )
        return lambda: backprop_fn(cotangent).block_until_ready()

    def run_signaturekernel_fwd_bwd(self, path1, path2, d, m) -> Callable:
        X, Y = self._path_batch(path1), self._path_batch(path2)
        signature_kernel = self._signature_kernel()
        cotangent = self.jnp.asarray(
            self.random_cotangent((X.shape[0], Y.shape[0])), dtype=X.dtype,
        )

        @self.jax.jit
        def step(X_arg, Y_arg, cotangent_arg):
            output, pullback = self.jax.vjp(
                lambda x: signature_kernel.kernel_matrix(x, Y_arg)[..., 0], X_arg,
            )
            return output, pullback(cotangent_arg)[0]

        # Dynamic arguments prevent compiling a cached gradient for fixed inputs.
        return lambda: self.jax.block_until_ready(step(X, Y, cotangent))

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
            method_prefix = "forward+backward(SigKernel.kernel_matrix)"
        elif self.operation == "signaturekernel_backprop":
            kernel = self.run_signaturekernel_backprop(
                path1,
                path2,
                self.d,
                self.m,
            )
            method_prefix = "vjp(SigKernel.kernel_matrix)"
        else:
            kernel = self.run_signaturekernel(
                path1,
                path2,
                self.d,
                self.m,
            )
            method_prefix = "SigKernel.kernel_matrix"

        t_ms, alloc_bytes, samples_ms = self.manual_timing_loop(kernel)
        return self.output_result(
            t_ms=t_ms,
            alloc_bytes=alloc_bytes,
            samples_ms=samples_ms,
            library="sigkerax",
            method=(
                f"{method_prefix}(refinement_factor="
                f"{self.refinement_factor:g},linear,dtype={self.dtype_name})"
            ),
            path_type="jax.Array",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_sigkerax.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    SigkeraxAdapter(json.loads(sys.argv[1])).run()
