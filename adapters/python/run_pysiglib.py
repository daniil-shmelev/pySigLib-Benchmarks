#!/usr/bin/env python3
"""pysiglib adapter for signature benchmarks"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Add src directory to path for common module
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from common import BenchmarkAdapter


class PySigLibAdapter(BenchmarkAdapter):
    """Adapter for pysiglib library"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Import here to avoid import errors if not available.
        import jax
        import jax.numpy as jnp
        import pysiglib.jax_api as pysiglib

        self.jax = jax
        self.jnp = jnp
        self.pysiglib = pysiglib
        self.log_sig_method = int(config.get("log_sig_method", 2))

    def _path_array(self, path: np.ndarray):
        """Convert path data to a contiguous JAX array during setup."""
        return self.cached_prepared_input(
            "pysiglib.jax",
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
        # Setup phase (untimed): ensure path is contiguous and on the JAX backend
        path = self._path_array(path)

        def signature_fn(path_arg):
            return self.pysiglib.signature(path_arg, degree=m)

        signature_fn = self.jax.jit(signature_fn)

        # Return kernel closure
        return lambda: signature_fn(path).block_until_ready()

    def run_logsignature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare logsignature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Setup phase (untimed): ensure path is contiguous and prepare cached
        # log signature data when the selected method requires it.
        path = self._path_array(path)
        if self.log_sig_method in (1, 2):
            device = (
                "cuda" if self.backend == "gpu"
                else "cpu" if self.backend == "cpu"
                else "both"
            )
            self.pysiglib.prepare_log_sig(
                d,
                m,
                method=self.log_sig_method,
                device=device,
            )
        log_sig_method = self.log_sig_method
        use_scalar_term = log_sig_method in (1, 2)

        def logsignature_fn(path_arg):
            return self.pysiglib.log_sig(
                path_arg,
                m,
                method=log_sig_method,
                scalar_term=use_scalar_term,
            )

        logsignature_fn = self.jax.jit(logsignature_fn)

        # Return kernel closure
        return lambda: logsignature_fn(path).block_until_ready()

    def run_sig_backprop(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature backpropagation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Setup phase (untimed): ensure path is contiguous and on the JAX backend
        path = self._path_array(path)

        def signature_fn(path_arg):
            return self.pysiglib.signature(path_arg, degree=m)

        output, pullback = self.jax.vjp(signature_fn, path)
        cotangent = self.jnp.asarray(
            self.random_cotangent(output.shape),
            dtype=output.dtype,
        )
        backprop_fn = self.jax.jit(lambda cotangent_arg: pullback(cotangent_arg)[0])
        return lambda: backprop_fn(cotangent).block_until_ready()

    def run_logsignature_backprop(
        self, path: np.ndarray, d: int, m: int
    ) -> Optional[Callable]:
        path = self._path_array(path)
        if self.log_sig_method in (1, 2):
            device = (
                "cuda" if self.backend == "gpu"
                else "cpu" if self.backend == "cpu"
                else "both"
            )
            self.pysiglib.prepare_log_sig(
                d,
                m,
                method=self.log_sig_method,
                device=device,
            )
        log_sig_method = self.log_sig_method
        use_scalar_term = log_sig_method in (1, 2)

        def logsignature_fn(path_arg):
            return self.pysiglib.log_sig(
                path_arg,
                m,
                method=log_sig_method,
                scalar_term=use_scalar_term,
            )

        output, pullback = self.jax.vjp(logsignature_fn, path)
        cotangent = self.jnp.asarray(
            self.random_cotangent(output.shape),
            dtype=output.dtype,
        )
        backprop_fn = self.jax.jit(lambda cotangent_arg: pullback(cotangent_arg)[0])
        return lambda: backprop_fn(cotangent).block_until_ready()

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
        if (
            self.backend == "gpu"
            and self.pysiglib.branched_sig_length(
                d,
                m,
                planar=planar,
                scalar_term=False,
            ) > 1024
        ):
            raise RuntimeError(
                "CUDA branched sig: num_trees > 1024 not supported"
            )

        # Setup phase (untimed): ensure path is contiguous and prepare cached
        # tree/coproduct data for the selected planar convention.
        path = self._path_array(path)
        self.pysiglib.prepare_branched_sig(d, m, planar=planar)

        def branchedsignature_fn(path_arg):
            return self.pysiglib.branched_sig(path_arg, degree=m, planar=planar)

        branchedsignature_fn = self.jax.jit(branchedsignature_fn)

        return lambda: branchedsignature_fn(path).block_until_ready()

    def run_branchedsignature_backprop(
        self,
        path: np.ndarray,
        d: int,
        m: int,
        *,
        planar: bool,
    ) -> Optional[Callable]:
        if (
            self.backend == "gpu"
            and self.pysiglib.branched_sig_length(
                d,
                m,
                planar=planar,
                scalar_term=False,
            ) > 1024
        ):
            raise RuntimeError(
                "CUDA branched sig: num_trees > 1024 not supported"
            )
        path = self._path_array(path)
        self.pysiglib.prepare_branched_sig(d, m, planar=planar)

        def branchedsignature_fn(path_arg):
            return self.pysiglib.branched_sig(
                path_arg,
                degree=m,
                planar=planar,
            )

        output, pullback = self.jax.vjp(branchedsignature_fn, path)
        cotangent = self.jnp.asarray(
            self.random_cotangent(output.shape),
            dtype=output.dtype,
        )
        backprop_fn = self.jax.jit(lambda cotangent_arg: pullback(cotangent_arg)[0])
        return lambda: backprop_fn(cotangent).block_until_ready()

    def run_signaturekernel(
        self,
        path1: np.ndarray,
        path2: np.ndarray,
        d: int,
        m: int,
    ) -> Optional[Callable]:
        """
        Prepare signature-kernel Gram matrix computation.

        Returns a closure that performs only the kernel computation (no setup).
        """
        path1 = self._path_array(path1)
        path2 = self._path_array(path2)
        dyadic_order = int(self.config.get("sig_kernel_dyadic_order", m))
        max_batch = int(self.config.get("sig_kernel_max_batch", -1))

        def signaturekernel_fn(path1_arg, path2_arg):
            return self.pysiglib.sig_kernel_gram(
                path1_arg,
                path2_arg,
                dyadic_order=dyadic_order,
                static_kernel=None,
                time_aug=False,
                max_batch=max_batch,
            )

        signaturekernel_fn = self.jax.jit(signaturekernel_fn)

        return lambda: signaturekernel_fn(path1, path2).block_until_ready()

    def run_signaturekernel_backprop(
        self,
        path1: np.ndarray,
        path2: np.ndarray,
        d: int,
        m: int,
    ) -> Optional[Callable]:
        path1 = self._path_array(path1)
        path2 = self._path_array(path2)
        dyadic_order = int(self.config.get("sig_kernel_dyadic_order", m))
        max_batch = int(self.config.get("sig_kernel_max_batch", -1))

        def signaturekernel_fn(path1_arg):
            return self.pysiglib.sig_kernel_gram(
                path1_arg,
                path2,
                dyadic_order=dyadic_order,
                static_kernel=None,
                time_aug=False,
                max_batch=max_batch,
            )

        output, pullback = self.jax.vjp(signaturekernel_fn, path1)
        cotangent = self.jnp.asarray(
            self.random_cotangent(output.shape),
            dtype=output.dtype,
        )
        backprop_fn = self.jax.jit(lambda cotangent_arg: pullback(cotangent_arg)[0])
        return lambda: backprop_fn(cotangent).block_until_ready()

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """Execute the benchmark"""
        # Generate path input during setup. pySigLib accepts leading batch
        # dimensions, so batch_size > 1 uses native batched execution.
        path = self.make_path_input()

        # Select operation
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "signature"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = f"log_sig(method={self.log_sig_method})"
        elif self.operation == "sig_backprop":
            kernel = self.run_sig_backprop(path, self.d, self.m)
            method = "vjp(signature)"
        elif self.operation == "logsignature_backprop":
            kernel = self.run_logsignature_backprop(path, self.d, self.m)
            method = f"vjp(log_sig(method={self.log_sig_method}))"
        elif self.operation in ("branchedsignature", "branchedsignature_nonplanar"):
            kernel = self.run_branchedsignature(path, self.d, self.m, planar=False)
            method = "branched_sig(planar=False)"
        elif self.operation == "branchedsignature_planar":
            kernel = self.run_branchedsignature(path, self.d, self.m, planar=True)
            method = "branched_sig(planar=True)"
        elif self.operation == "branchedsignature_nonplanar_backprop":
            kernel = self.run_branchedsignature_backprop(
                path,
                self.d,
                self.m,
                planar=False,
            )
            method = "vjp(branched_sig(planar=False))"
        elif self.operation == "branchedsignature_planar_backprop":
            kernel = self.run_branchedsignature_backprop(
                path,
                self.d,
                self.m,
                planar=True,
            )
            method = "vjp(branched_sig(planar=True))"
        elif self.operation in ("signaturekernel", "signature_kernel", "sigkernel"):
            path2 = self.make_path_input()
            kernel = self.run_signaturekernel(path, path2, self.d, self.m)
            method = (
                "sig_kernel_gram"
                f"(dyadic_order={int(self.config.get('sig_kernel_dyadic_order', self.m))})"
            )
        elif self.operation == "signaturekernel_backprop":
            path2 = self.make_path_input()
            kernel = self.run_signaturekernel_backprop(
                path,
                path2,
                self.d,
                self.m,
            )
            method = (
                "vjp(sig_kernel_gram)"
                f"(dyadic_order={int(self.config.get('sig_kernel_dyadic_order', self.m))})"
            )
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
            library="pysiglib",
            method=method,
            path_type="jax.Array",
            language="python"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_pysiglib.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    # Parse configuration from command line
    config = json.loads(sys.argv[1])

    # Create and run adapter
    adapter = PySigLibAdapter(config)
    adapter.run()
