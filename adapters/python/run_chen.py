#!/usr/bin/env python3
"""chen-signatures adapter for signature benchmarks"""

import json
import sys
import ctypes
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _preload_julia_libstdcxx() -> None:
    """
    Prefer Julia's bundled libstdc++ before importing juliacall-backed chen.

    Older Linux distributions can have a system libstdc++ that is too old for
    juliaup's Julia build. Loading Julia's copy first avoids a process-wide
    LD_LIBRARY_PATH requirement for this adapter.
    """
    if not sys.platform.startswith("linux"):
        return

    juliaup_dir = Path.home() / ".julia" / "juliaup"
    candidates = sorted(
        juliaup_dir.glob("julia-*/lib/julia/libstdc++.so.6"),
        reverse=True,
    )

    for candidate in candidates:
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_julia_libstdcxx()

# Add src directory to path for common module
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from common import BenchmarkAdapter


class ChenSignaturesAdapter(BenchmarkAdapter):
    """Adapter for chen-signatures library"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Import here to avoid import errors if not available
        import chen
        self.chen = chen

    def run_signature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Setup phase (untimed): ensure path is contiguous
        path = np.ascontiguousarray(path, dtype=np.float32)

        # Return kernel closure
        return lambda: self.chen.sig(path, m)

    def run_logsignature(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare logsignature computation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Check if logsig methods are available
        if not (hasattr(self.chen, "logsig") and hasattr(self.chen, "prepare_logsig")):
            return None

        # Setup phase (untimed): prepare basis and ensure path is contiguous
        path = np.ascontiguousarray(path, dtype=np.float32)
        basis = self.chen.prepare_logsig(d, m)

        # Return kernel closure
        return lambda: self.chen.logsig(path, basis)

    def run_sigdiff(self, path: np.ndarray, d: int, m: int) -> Optional[Callable]:
        """
        Prepare signature differentiation kernel.

        Returns a closure that performs only the kernel (no setup).
        """
        # Import PyTorch dependencies
        try:
            from chen.torch import sig_torch
            import torch
        except ImportError:
            return None

        # Setup phase (untimed): prepare path as numpy array
        path_np = np.ascontiguousarray(path, dtype=np.float32)

        # Return kernel closure that converts to torch, computes sig, and backprop
        def kernel():
            path_t = torch.tensor(path_np, dtype=torch.float32, requires_grad=True)
            sig = sig_torch(path_t, m)
            loss = sig.sum()
            loss.backward()

        return kernel

    def _run_benchmark(self) -> Optional[Dict[str, Any]]:
        """Execute the benchmark"""
        if self.batch_size > 1:
            # The installed Python wrapper only exposes single-path ChenSignatures.
            # Avoid benchmarking a Python loop as a stand-in for native batching.
            return None

        # Generate path
        path = self.make_path_input()

        # Select operation
        if self.operation == "signature":
            kernel = self.run_signature(path, self.d, self.m)
            method = "sig"
            path_type = "ndarray"
        elif self.operation == "logsignature":
            kernel = self.run_logsignature(path, self.d, self.m)
            method = "logsig(prepared)"
            path_type = "ndarray"
        elif self.operation == "sigdiff":
            kernel = self.run_sigdiff(path, self.d, self.m)
            method = "sigdiff"
            path_type = "torch"
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
            library="chen-signatures",
            method=method,
            path_type=path_type,
            language="python"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_chen.py '<json_config>'", file=sys.stderr)
        sys.exit(1)

    # Parse configuration from command line
    config = json.loads(sys.argv[1])

    # Create and run adapter
    adapter = ChenSignaturesAdapter(config)
    adapter.run()
