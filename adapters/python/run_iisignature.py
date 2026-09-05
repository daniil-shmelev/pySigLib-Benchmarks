#!/usr/bin/env python3
"""iisignature adapter for signature benchmarks"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Add src directory to path for common module
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from common import BenchmarkAdapter


class IISignatureAdapter(BenchmarkAdapter):
    """Adapter for iisignature library"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Import here to avoid import errors if not available
        import iisignature
        self.iisignature = iisignature
        self.logsig_method = config.get("logsig_method", "S")

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Setup phase (untimed): ensure path is contiguous
        path = np.ascontiguousarray(path, dtype=np.float32)

        # Return kernel closure
        return lambda: self.iisignature.sig(path, m)

    def run_logsignature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare logsignature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        if d < 2:
            # iisignature sometimes doesn't like d=1 for logsig
            return None

        # Setup phase (untimed): prepare basis and ensure path is contiguous
        path = np.ascontiguousarray(path, dtype=np.float32)
        basis = self.iisignature.prepare(d, m, self.logsig_method)

        # Return kernel closure
        return lambda: self.iisignature.logsig(path, basis, self.logsig_method)

    def run_sig_backprop(
        self, path: np.ndarray, d: int, m: int
    ) -> Optional[Callable]:
        path = np.ascontiguousarray(path, dtype=np.float32)
        output = self.iisignature.sig(path, m)
        cotangent = self.random_cotangent(output.shape)
        return lambda: self.iisignature.sigbackprop(cotangent, path, m)

    def run_logsignature_backprop(
        self, path: np.ndarray, d: int, m: int
    ) -> Optional[Callable]:
        if d < 2:
            return None
        if not set(str(self.logsig_method).upper()) & {"S", "A", "X"}:
            # iisignature's direct BCH methods C and O have no backward API.
            raise NotImplementedError(
                "iisignature BCH logsignature backprop is unsupported"
            )
        path = np.ascontiguousarray(path, dtype=np.float32)
        basis = self.iisignature.prepare(d, m, self.logsig_method)
        output = self.iisignature.logsig(path, basis, self.logsig_method)
        cotangent = self.random_cotangent(output.shape)
        return lambda: self.iisignature.logsigbackprop(
            cotangent,
            path,
            basis,
            self.logsig_method,
        )

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """Execute the benchmark"""
        # Generate path input during setup. When batch_size > 1 this has shape
        # (batch_size, N, d), which iisignature handles natively.
        path = self.make_path_input()

        # Select operation
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "sig"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = f"logsig({self.logsig_method})"
        elif self.operation == "sig_backprop":
            kernel = self.run_sig_backprop(path, self.d, self.m)
            method = "sigbackprop"
        elif self.operation == "logsignature_backprop":
            kernel = self.run_logsignature_backprop(path, self.d, self.m)
            method = f"logsigbackprop({self.logsig_method})"
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
            library="iisignature",
            method=method,
            path_type="ndarray",
            language="python"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_iisignature.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    # Parse configuration from command line
    config = json.loads(sys.argv[1])

    # Add logsig_method if not present (default from benchmark_sweep.yaml)
    if "logsig_method" not in config:
        config["logsig_method"] = "S"

    # Create and run adapter
    adapter = IISignatureAdapter(config)
    adapter.run()
