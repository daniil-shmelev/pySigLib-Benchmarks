"""Tests for signature-kernel adapter paths."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from adapters.python.run_polysigkernel import PolySigKernelAdapter
from adapters.python.run_pysiglib import PySigLibAdapter
from common.paths import make_path


def _config(operation: str) -> dict:
    return {
        "N": 4,
        "d": 2,
        "m": 2,
        "path_kind": "linear",
        "operation": operation,
        "repeats": 1,
        "batch_size": 2,
        "backend": "cpu",
    }


def _path_batch() -> np.ndarray:
    path = make_path(2, 4, "linear").astype(np.float32)
    return np.stack([path, path + 0.1], axis=0)


def test_polysigkernel_signaturekernel_matches_direct_api():
    """polysigkernel adapter calls SigKernel.kernel_matrix on path batches."""
    pytest.importorskip("jax")
    polysigkernel = pytest.importorskip("polysigkernel")

    adapter = PolySigKernelAdapter(_config("signaturekernel"))
    path1 = _path_batch()
    path2 = _path_batch() + 0.2

    kernel = adapter.run_signaturekernel(path1, path2, 2, 2)
    got = np.asarray(kernel())

    X = adapter._path_batch(path1)
    Y = adapter._path_batch(path2)
    expected = np.asarray(
        polysigkernel.SigKernel(
            order=2,
            static_kernel="linear",
            solver=adapter.solver,
            add_time=False,
        ).kernel_matrix(X, Y)
    )

    assert got.dtype == np.float32
    assert got.shape == (2, 2)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_pysiglib_signaturekernel_matches_jax_api():
    """pySigLib adapter uses the JAX signature-kernel Gram API."""
    pytest.importorskip("jax")
    pytest.importorskip("pysiglib.jax_api")

    adapter = PySigLibAdapter(_config("signaturekernel"))
    path1 = _path_batch()
    path2 = _path_batch() + 0.2

    kernel = adapter.run_signaturekernel(path1, path2, 2, 2)
    got = np.asarray(kernel())

    X = adapter._path_array(path1)
    Y = adapter._path_array(path2)
    expected = np.asarray(adapter.pysiglib.sig_kernel_gram(X, Y, dyadic_order=2))

    assert got.dtype == np.float32
    assert got.shape == (2, 2)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
