"""Tests for JAX-backed signature adapters."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from adapters.python.run_keras_sig import KerasSigAdapter
from adapters.python.run_signax import SignaxAdapter
from common.paths import make_path


def _config(operation: str) -> dict:
    return {
        "N": 4,
        "d": 2,
        "m": 2,
        "path_kind": "linear",
        "operation": operation,
        "repeats": 1,
    }


def test_signax_signature_matches_direct_api():
    """signax adapter uses the flattened top-level signature API."""
    pytest.importorskip("jax")
    signax = pytest.importorskip("signax")

    adapter = SignaxAdapter(_config("signature"))
    path = make_path(2, 4, "linear")

    kernel = adapter.run_signature(path, 2, 2)
    got = np.asarray(kernel())

    path_jax = adapter._path_array(path)
    expected = np.asarray(
        signax.signature(
            path_jax,
            depth=2,
            stream=False,
            flatten=True,
            num_chunks=adapter.num_chunks,
        )
    )

    assert got.dtype == np.float32
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_signax_logsignature_matches_direct_api():
    """signax adapter uses the flattened top-level logsignature API."""
    pytest.importorskip("jax")
    signax = pytest.importorskip("signax")

    adapter = SignaxAdapter(_config("logsignature"))
    path = make_path(2, 4, "linear")

    kernel = adapter.run_logsignature(path, 2, 2)
    got = np.asarray(kernel())

    path_jax = adapter._path_array(path)
    expected = np.asarray(
        signax.logsignature(
            path_jax,
            depth=2,
            stream=False,
            flatten=True,
            num_chunks=adapter.num_chunks,
        )
    )

    assert got.dtype == np.float32
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_keras_sig_signature_matches_direct_api():
    """keras_sig adapter uses the direct JAX GPU signature API."""
    os.environ["KERAS_BACKEND"] = "jax"
    pytest.importorskip("jax")
    pytest.importorskip("keras_sig")

    adapter = KerasSigAdapter(_config("signature"))
    path = make_path(2, 4, "linear")

    kernel = adapter.run_signature(path, 2, 2)
    got = np.asarray(kernel())

    path_jax = adapter._path_array(path)
    expected = np.asarray(
        adapter.jnp.squeeze(
            adapter.jax_gpu_signature(path_jax, depth=2, stream=False),
            axis=0,
        )
    )

    assert got.dtype == np.float32
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
