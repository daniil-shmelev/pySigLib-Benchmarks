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
from adapters.python.run_tensordev import TensorDevAdapter
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


def test_tensordev_signature_matches_direct_api():
    """TensorDev adapter uses its native batched path-signature API."""
    pytest.importorskip("jax")
    tensordev = pytest.importorskip("tensordev")

    adapter = TensorDevAdapter(_config("signature"))
    path = make_path(2, 4, "linear")

    got = adapter.run_signature(path, 2, 2)()
    expected = tensordev.path_signature(adapter._path_array(path), trunc=2)

    got_flat = np.asarray(tensordev.tensor_to_flat(got, start_at_level_one=True))
    expected_flat = np.asarray(
        tensordev.tensor_to_flat(expected, start_at_level_one=True)
    )

    assert got_flat.dtype == np.float32
    assert got_flat.shape == expected_flat.shape
    np.testing.assert_allclose(got_flat, expected_flat, rtol=1e-6, atol=1e-6)


def test_tensordev_logsignature_matches_direct_api():
    """TensorDev adapter computes its expanded tensor log-signature."""
    pytest.importorskip("jax")
    tensordev = pytest.importorskip("tensordev")

    adapter = TensorDevAdapter(_config("logsignature"))
    path = make_path(2, 4, "linear")

    got = adapter.run_logsignature(path, 2, 2)()
    signature = tensordev.path_signature(adapter._path_array(path), trunc=2)
    expected = tensordev.tensor_logarithm(
        signature[1:],
        trunc=2,
        output_zero_level=False,
    )

    assert len(got) == len(expected)
    for got_level, expected_level in zip(got, expected):
        got_level = np.asarray(got_level)
        expected_level = np.asarray(expected_level)
        assert got_level.dtype == np.float32
        assert got_level.shape == expected_level.shape
        np.testing.assert_allclose(got_level, expected_level, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("adapter_class", "package", "method_name", "operation"),
    [
        (SignaxAdapter, "signax", "run_sig_backprop", "sig_backprop"),
        (
            SignaxAdapter,
            "signax",
            "run_logsignature_backprop",
            "logsignature_backprop",
        ),
        (KerasSigAdapter, "keras_sig", "run_sig_backprop", "sig_backprop"),
        (TensorDevAdapter, "tensordev", "run_sig_backprop", "sig_backprop"),
        (
            TensorDevAdapter,
            "tensordev",
            "run_logsignature_backprop",
            "logsignature_backprop",
        ),
    ],
)
def test_jax_backprop_returns_path_gradient(
    adapter_class,
    package,
    method_name,
    operation,
):
    if package == "keras_sig":
        os.environ["KERAS_BACKEND"] = "jax"
    pytest.importorskip("jax")
    pytest.importorskip(package)
    adapter = adapter_class(_config(operation))
    path = make_path(2, 4, "linear")

    gradient = np.asarray(getattr(adapter, method_name)(path, 2, 2)())

    expected_shape = (
        tuple(adapter._path_array(path).shape)
        if package == "keras_sig"
        else path.shape
    )
    assert gradient.shape == expected_shape
    assert np.isfinite(gradient).all()
