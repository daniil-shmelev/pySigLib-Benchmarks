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
        import pysiglib
        import torch
        from pysiglib.log_sig_backprop import _log_sig_from_path_backprop

        self.pysiglib = pysiglib
        self.torch = torch
        self.log_sig_from_path_backprop = _log_sig_from_path_backprop
        self.log_sig_method = int(config.get("log_sig_method", 2))
        self.n_jobs = int(config.get("n_jobs", 1))

    def _path_array(self, path: np.ndarray):
        """Prepare the standard API input array outside the timed region."""
        if self.backend == "gpu":
            return self.cached_prepared_input(
                "pysiglib.standard.gpu",
                path,
                lambda: self.torch.as_tensor(
                    np.ascontiguousarray(path, dtype=np.float32),
                    dtype=self.torch.float32,
                    device="cuda",
                ),
            )

        return self.cached_prepared_input(
            "pysiglib.standard.cpu",
            path,
            lambda: np.ascontiguousarray(path, dtype=np.float32),
        )

    def _cotangent(self, output):
        values = self.random_cotangent(output.shape)
        if isinstance(output, np.ndarray):
            return np.asarray(values, dtype=output.dtype)
        return self.torch.as_tensor(
            values,
            dtype=output.dtype,
            device=output.device,
        )

    def _synchronize(self, result):
        if self.backend == "gpu":
            self.torch.cuda.synchronize()
        return result

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path = self._path_array(path)

        def signature_fn():
            result = self.pysiglib.signature(
                path,
                degree=m,
                n_jobs=self.n_jobs,
            )
            return self._synchronize(result)

        return signature_fn

    def run_logsignature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare logsignature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
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

        def logsignature_fn():
            result = self.pysiglib.log_sig(
                path,
                m,
                method=log_sig_method,
                scalar_term=use_scalar_term,
                n_jobs=self.n_jobs,
            )
            return self._synchronize(result)

        return logsignature_fn

    def run_sig_backprop(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature backpropagation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        path = self._path_array(path)
        output = self.pysiglib.signature(
            path,
            degree=m,
            n_jobs=self.n_jobs,
        )
        self._synchronize(output)
        cotangent = self._cotangent(output)

        def backprop_fn():
            result = self.pysiglib.sig_backprop(
                path,
                output,
                cotangent,
                m,
                n_jobs=self.n_jobs,
            )
            return self._synchronize(result)

        return backprop_fn

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
        if log_sig_method == 3:
            output = self.pysiglib.log_sig(
                path,
                m,
                method=3,
                n_jobs=self.n_jobs,
            )
            self._synchronize(output)
            cotangent = self._cotangent(output)

            def backprop_fn():
                result = self.log_sig_from_path_backprop(
                    cotangent,
                    path,
                    m,
                    n_jobs=self.n_jobs,
                )
                return self._synchronize(result)

            return backprop_fn
        use_scalar_term = log_sig_method in (1, 2)
        signature = self.pysiglib.signature(
            path,
            degree=m,
            scalar_term=use_scalar_term,
            n_jobs=self.n_jobs,
        )
        output = self.pysiglib.sig_to_log_sig(
            signature,
            d,
            m,
            method=log_sig_method,
            n_jobs=self.n_jobs,
        )
        self._synchronize(output)
        cotangent = self._cotangent(output)

        def backprop_fn():
            sig_cotangent = self.pysiglib.sig_to_log_sig_backprop(
                signature,
                cotangent,
                d,
                m,
                method=log_sig_method,
                n_jobs=self.n_jobs,
            )
            result = self.pysiglib.sig_backprop(
                path,
                signature,
                sig_cotangent,
                m,
                n_jobs=self.n_jobs,
            )
            return self._synchronize(result)

        return backprop_fn

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

        path = self._path_array(path)
        self.pysiglib.prepare_branched_sig(d, m, planar=planar)

        def branchedsignature_fn():
            result = self.pysiglib.branched_sig(
                path,
                degree=m,
                planar=planar,
                n_jobs=self.n_jobs,
            )
            return self._synchronize(result)

        return branchedsignature_fn

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
        output = self.pysiglib.branched_sig(
            path,
            degree=m,
            planar=planar,
            n_jobs=self.n_jobs,
        )
        self._synchronize(output)
        cotangent = self._cotangent(output)

        def backprop_fn():
            result = self.pysiglib.branched_sig_backprop(
                path,
                output,
                cotangent,
                m,
                planar=planar,
                n_jobs=self.n_jobs,
            )
            return self._synchronize(result)

        return backprop_fn

    def run_branchedlogsignature(
        self,
        path: np.ndarray,
        d: int,
        m: int,
        *,
        planar: bool,
    ) -> Optional[Callable]:
        """Prepare a planar or non-planar branched log-signature kernel."""
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

        def branchedlogsignature_fn():
            result = self.pysiglib.branched_log_sig(
                path,
                degree=m,
                planar=planar,
                n_jobs=self.n_jobs,
            )
            return self._synchronize(result)

        return branchedlogsignature_fn

    def run_branchedlogsignature_backprop(
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
        branched_signature = self.pysiglib.branched_sig(
            path,
            degree=m,
            planar=planar,
            n_jobs=self.n_jobs,
        )
        output = self.pysiglib.branched_sig_to_log_sig(
            branched_signature,
            d,
            m,
            planar=planar,
            n_jobs=self.n_jobs,
        )
        self._synchronize(output)
        cotangent = self._cotangent(output)

        def backprop_fn():
            branched_signature_cotangent = (
                self.pysiglib.branched_sig_to_log_sig_backprop(
                    branched_signature,
                    cotangent,
                    d,
                    m,
                    planar=planar,
                    n_jobs=self.n_jobs,
                )
            )
            result = self.pysiglib.branched_sig_backprop(
                path,
                branched_signature,
                branched_signature_cotangent,
                m,
                planar=planar,
                n_jobs=self.n_jobs,
            )
            return self._synchronize(result)

        return backprop_fn

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

        def signaturekernel_fn():
            result = self.pysiglib.sig_kernel_gram(
                path1,
                path2,
                dyadic_order=dyadic_order,
                static_kernel=None,
                time_aug=False,
                n_jobs=self.n_jobs,
                max_batch=max_batch,
            )
            return self._synchronize(result)

        return signaturekernel_fn

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

        output = self.pysiglib.sig_kernel_gram(
            path1,
            path2,
            dyadic_order=dyadic_order,
            static_kernel=None,
            time_aug=False,
            n_jobs=self.n_jobs,
            max_batch=max_batch,
        )
        self._synchronize(output)
        cotangent = self._cotangent(output)

        def backprop_fn():
            result, _ = self.pysiglib.sig_kernel_gram_backprop(
                cotangent,
                path1,
                path2,
                dyadic_order=dyadic_order,
                static_kernel=None,
                time_aug=False,
                left_deriv=True,
                right_deriv=False,
                n_jobs=self.n_jobs,
                max_batch=max_batch,
            )
            return self._synchronize(result)

        return backprop_fn

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
            method = "sig_backprop"
        elif self.operation == "logsignature_backprop":
            kernel = self.run_logsignature_backprop(path, self.d, self.m)
            method = f"log_sig_backprop(method={self.log_sig_method})"
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
            method = "branched_sig_backprop(planar=False)"
        elif self.operation == "branchedsignature_planar_backprop":
            kernel = self.run_branchedsignature_backprop(
                path,
                self.d,
                self.m,
                planar=True,
            )
            method = "branched_sig_backprop(planar=True)"
        elif self.operation == "branchedlogsignature_nonplanar":
            kernel = self.run_branchedlogsignature(
                path,
                self.d,
                self.m,
                planar=False,
            )
            method = "branched_log_sig(planar=False)"
        elif self.operation == "branchedlogsignature_planar":
            kernel = self.run_branchedlogsignature(
                path,
                self.d,
                self.m,
                planar=True,
            )
            method = "branched_log_sig(planar=True)"
        elif self.operation == "branchedlogsignature_nonplanar_backprop":
            kernel = self.run_branchedlogsignature_backprop(
                path,
                self.d,
                self.m,
                planar=False,
            )
            method = "branched_log_sig_backprop(planar=False)"
        elif self.operation == "branchedlogsignature_planar_backprop":
            kernel = self.run_branchedlogsignature_backprop(
                path,
                self.d,
                self.m,
                planar=True,
            )
            method = "branched_log_sig_backprop(planar=True)"
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
                "sig_kernel_gram_backprop"
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
            path_type=(
                "torch.Tensor" if self.backend == "gpu" else "numpy.ndarray"
            ),
            language="python",
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
