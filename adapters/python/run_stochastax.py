#!/usr/bin/env python3
"""stochastax adapter for signature benchmarks"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Add src directory to path for common module
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from common import BenchmarkAdapter


class StochastaxAdapter(BenchmarkAdapter):
    """Adapter for stochastax library"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Import here to avoid import errors if not available.
        import jax

        import jax.numpy as jnp
        from stochastax.control_lifts.log_signature import compute_log_signature
        from stochastax.control_lifts.path_signature import compute_path_signature
        from stochastax.control_lifts.branched_signature_ito import (
            GLHopfAlgebra,
            MKWHopfAlgebra,
            compute_nonplanar_branched_signature,
            compute_planar_branched_signature,
        )
        from stochastax.hopf_algebras.shuffle import ShuffleHopfAlgebra

        self.jax = jax
        self.jnp = jnp
        self.GLHopfAlgebra = GLHopfAlgebra
        self.MKWHopfAlgebra = MKWHopfAlgebra
        self.compute_log_signature = compute_log_signature
        self.compute_nonplanar_branched_signature = compute_nonplanar_branched_signature
        self.compute_path_signature = compute_path_signature
        self.compute_planar_branched_signature = compute_planar_branched_signature
        self.ShuffleHopfAlgebra = ShuffleHopfAlgebra
        self.log_signature_type = config.get("log_signature_type", "Lyndon words")

    def _path_array(self, path: np.ndarray):
        """Convert path data to a contiguous JAX array during setup."""
        return self.cached_prepared_input(
            "stochastax.jax",
            path,
            lambda: self.jnp.asarray(
                np.ascontiguousarray(path, dtype=np.float32),
                dtype=self.jnp.float32,
            ),
        )

    def _zero_cov_increments(self, path):
        """Create zero quadratic-variation increments for ordinary branched signatures."""
        steps = max(int(path.shape[-2]) - 1, 0)
        dim = int(path.shape[-1])
        return self.jnp.zeros((*path.shape[:-2], steps, dim, dim), dtype=path.dtype)

    def _vjp_kernel(self, function: Callable, path) -> Callable:
        output, pullback = self.jax.vjp(function, path)
        cotangent = self.jnp.asarray(
            self.random_cotangent(output.shape),
            dtype=output.dtype,
        )
        backprop_fn = self.jax.jit(lambda cotangent_arg: pullback(cotangent_arg)[0])
        return lambda: backprop_fn(cotangent).block_until_ready()

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path_jax = self._path_array(path)
        hopf = self.ShuffleHopfAlgebra.build(d, m)

        def signature_one(path_arg):
            return self.compute_path_signature(path_arg, m, hopf, mode="full").flatten()

        if path_jax.ndim == 3:
            signature_fn = self.jax.jit(self.jax.vmap(signature_one))
        else:
            signature_fn = self.jax.jit(signature_one)

        return lambda: signature_fn(path_jax).block_until_ready()

    def run_logsignature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare logsignature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path_jax = self._path_array(path)
        hopf = self.ShuffleHopfAlgebra.build(d, m)
        log_signature_type = self.log_signature_type

        def logsignature_one(path_arg):
            return self.compute_log_signature(
                path_arg,
                m,
                hopf,
                log_signature_type,
                mode="full",
            ).flatten()

        if path_jax.ndim == 3:
            logsignature_fn = self.jax.jit(self.jax.vmap(logsignature_one))
        else:
            logsignature_fn = self.jax.jit(logsignature_one)

        return lambda: logsignature_fn(path_jax).block_until_ready()

    def run_sig_backprop(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature backpropagation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path_jax = self._path_array(path)
        hopf = self.ShuffleHopfAlgebra.build(d, m)

        def signature_one(path_arg):
            sig = self.compute_path_signature(path_arg, m, hopf, mode="full").flatten()
            return sig

        if path_jax.ndim == 3:
            signature_fn = self.jax.vmap(signature_one)
        else:
            signature_fn = signature_one
        return self._vjp_kernel(signature_fn, path_jax)

    def run_logsignature_backprop(
        self, path: np.ndarray, d: int, m: int
    ) -> Optional[Callable]:
        path_jax = self._path_array(path)
        hopf = self.ShuffleHopfAlgebra.build(d, m)
        log_signature_type = self.log_signature_type

        def logsignature_one(path_arg):
            return self.compute_log_signature(
                path_arg,
                m,
                hopf,
                log_signature_type,
                mode="full",
            ).flatten()

        if path_jax.ndim == 3:
            logsignature_fn = self.jax.vmap(logsignature_one)
        else:
            logsignature_fn = logsignature_one
        return self._vjp_kernel(logsignature_fn, path_jax)

    def run_branchedsignature(
        self,
        path: np.ndarray,
        d: int,
        m: int,
        *,
        planar: bool,
    ) -> Optional[Callable]:
        """
        Prepare branched signature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path_jax = self._path_array(path)
        cov_increments = self._zero_cov_increments(path_jax)

        if planar:
            hopf = self.MKWHopfAlgebra.build(d, m)

            def branchedsignature_one(path_arg, cov_arg):
                return self.compute_planar_branched_signature(
                    path_arg,
                    m,
                    hopf,
                    "full",
                    cov_arg,
                ).flatten()

        else:
            hopf = self.GLHopfAlgebra.build(d, m)

            def branchedsignature_one(path_arg, cov_arg):
                return self.compute_nonplanar_branched_signature(
                    path_arg,
                    m,
                    hopf,
                    "full",
                    cov_arg,
                ).flatten()

        if path_jax.ndim == 3:
            branchedsignature_fn = self.jax.jit(self.jax.vmap(branchedsignature_one))
        else:
            branchedsignature_fn = self.jax.jit(branchedsignature_one)

        return lambda: branchedsignature_fn(path_jax, cov_increments).block_until_ready()

    def run_branchedsignature_backprop(
        self,
        path: np.ndarray,
        d: int,
        m: int,
        *,
        planar: bool,
    ) -> Optional[Callable]:
        path_jax = self._path_array(path)
        cov_increments = self._zero_cov_increments(path_jax)
        if planar:
            hopf = self.MKWHopfAlgebra.build(d, m)

            def branchedsignature_one(path_arg, cov_arg):
                return self.compute_planar_branched_signature(
                    path_arg,
                    m,
                    hopf,
                    "full",
                    cov_arg,
                ).flatten()
        else:
            hopf = self.GLHopfAlgebra.build(d, m)

            def branchedsignature_one(path_arg, cov_arg):
                return self.compute_nonplanar_branched_signature(
                    path_arg,
                    m,
                    hopf,
                    "full",
                    cov_arg,
                ).flatten()

        if path_jax.ndim == 3:
            vmapped = self.jax.vmap(branchedsignature_one)
            function = lambda path_arg: vmapped(path_arg, cov_increments)
        else:
            function = lambda path_arg: branchedsignature_one(
                path_arg,
                cov_increments,
            )
        return self._vjp_kernel(function, path_jax)

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """Execute the benchmark"""
        # Generate path input during setup. Stochastax single-path APIs are
        # vmapped by the adapter when batch_size > 1.
        path = self.make_path_input()

        # Select operation
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "compute_path_signature"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = f"compute_log_signature({self.log_signature_type})"
        elif self.operation == "sig_backprop":
            kernel = self.run_sig_backprop(path, self.d, self.m)
            method = "vjp(compute_path_signature)"
        elif self.operation == "logsignature_backprop":
            kernel = self.run_logsignature_backprop(path, self.d, self.m)
            method = f"vjp(compute_log_signature({self.log_signature_type}))"
        elif self.operation in ("branchedsignature", "branchedsignature_nonplanar"):
            kernel = self.run_branchedsignature(path, self.d, self.m, planar=False)
            method = "compute_nonplanar_branched_signature"
        elif self.operation == "branchedsignature_planar":
            kernel = self.run_branchedsignature(path, self.d, self.m, planar=True)
            method = "compute_planar_branched_signature"
        elif self.operation == "branchedsignature_nonplanar_backprop":
            kernel = self.run_branchedsignature_backprop(
                path,
                self.d,
                self.m,
                planar=False,
            )
            method = "vjp(compute_nonplanar_branched_signature)"
        elif self.operation == "branchedsignature_planar_backprop":
            kernel = self.run_branchedsignature_backprop(
                path,
                self.d,
                self.m,
                planar=True,
            )
            method = "vjp(compute_planar_branched_signature)"
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
            library="stochastax",
            method=method,
            path_type="jax.Array",
            language="python",
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_stochastax.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    # Parse configuration from command line
    config = json.loads(sys.argv[1])

    # Create and run adapter
    adapter = StochastaxAdapter(config)
    adapter.run()
