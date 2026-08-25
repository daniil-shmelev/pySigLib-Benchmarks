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
from adapters.python.run_sigkerax import SigkeraxAdapter
from adapters.python.run_sigkernel import SigkernelAdapter
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


def test_pysiglib_signaturekernel_matches_standard_api():
    """pySigLib adapter uses the standard signature-kernel Gram API."""
    pytest.importorskip("pysiglib")

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


def test_polysigkernel_backprop_matches_direct_vjp():
    pytest.importorskip("jax")
    pytest.importorskip("polysigkernel")

    adapter = PolySigKernelAdapter(_config("signaturekernel_backprop"))
    path1 = _path_batch()
    path2 = _path_batch() + 0.2
    got = np.asarray(
        adapter.run_signaturekernel_backprop(path1, path2, 2, 2)()
    )

    X = adapter._path_batch(path1)
    Y = adapter._path_batch(path2)
    sig_kernel = adapter.SigKernel_polynomial(
        order=2,
        static_kernel="linear",
        solver=adapter.solver,
        add_time=False,
    )
    output, pullback = adapter.jax.vjp(
        lambda X_arg: sig_kernel.kernel_matrix(X_arg, Y),
        X,
    )
    cotangent = adapter.jnp.asarray(
        adapter.random_cotangent(output.shape),
        dtype=output.dtype,
    )
    expected = np.asarray(pullback(cotangent)[0])

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_pysiglib_signaturekernel_backprop_matches_standard_api():
    pytest.importorskip("pysiglib")

    adapter = PySigLibAdapter(_config("signaturekernel_backprop"))
    path1 = _path_batch()
    path2 = _path_batch() + 0.2
    got = np.asarray(
        adapter.run_signaturekernel_backprop(path1, path2, 2, 2)()
    )

    X = adapter._path_array(path1)
    Y = adapter._path_array(path2)
    output = adapter.pysiglib.sig_kernel_gram(
        X,
        Y,
        dyadic_order=2,
    )
    cotangent = np.asarray(
        adapter.random_cotangent(output.shape), dtype=output.dtype
    )
    expected, _ = adapter.pysiglib.sig_kernel_gram_backprop(
        cotangent,
        X,
        Y,
        dyadic_order=2,
        left_deriv=True,
        right_deriv=False,
    )

    np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_sigkerax_signaturekernel_matches_direct_api():
    pytest.importorskip("jax")
    pytest.importorskip("sigkerax")

    config = _config("signaturekernel")
    config["sig_kernel_refinement_factor"] = 1
    adapter = SigkeraxAdapter(config)
    path1 = _path_batch()
    path2 = _path_batch() + 0.2

    got = np.asarray(adapter.run_signaturekernel(path1, path2, 2, 2)())
    X = adapter._path_batch(path1)
    Y = adapter._path_batch(path2)
    expected = np.asarray(adapter._signature_kernel().kernel_matrix(X, Y)[..., 0])

    assert got.dtype == np.float32
    assert got.shape == (2, 2)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_sigkerax_backprop_matches_direct_vjp():
    pytest.importorskip("jax")
    pytest.importorskip("sigkerax")

    config = _config("signaturekernel_backprop")
    config["sig_kernel_refinement_factor"] = 1
    adapter = SigkeraxAdapter(config)
    path1 = _path_batch()
    path2 = _path_batch() + 0.2
    got = np.asarray(
        adapter.run_signaturekernel_backprop(path1, path2, 2, 2)()
    )

    X = adapter._path_batch(path1)
    Y = adapter._path_batch(path2)
    output, pullback = adapter.jax.vjp(
        lambda X_arg: adapter._signature_kernel().kernel_matrix(
            X_arg,
            Y,
        )[..., 0],
        X,
    )
    cotangent = adapter.jnp.asarray(
        adapter.random_cotangent(output.shape),
        dtype=output.dtype,
    )
    expected = np.asarray(pullback(cotangent)[0])

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_sigkernel_signaturekernel_matches_direct_api():
    pytest.importorskip("sigkernel")

    config = _config("signaturekernel")
    config["sig_kernel_dyadic_order"] = 0
    adapter = SigkernelAdapter(config)
    path1 = _path_batch()
    path2 = _path_batch() + 0.2

    got = np.asarray(adapter.run_signaturekernel(path1, path2, 2, 2)())
    X = adapter._path_tensor(path1)
    Y = adapter._path_tensor(path2)
    expected = np.asarray(
        adapter._signature_kernel().compute_Gram(
            X,
            Y,
            sym=False,
            max_batch=2,
        )
    )

    assert got.dtype == np.float64
    assert got.shape == (2, 2)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_sigkernel_backprop_matches_direct_autograd():
    pytest.importorskip("sigkernel")

    config = _config("signaturekernel_backprop")
    config["sig_kernel_dyadic_order"] = 0
    adapter = SigkernelAdapter(config)
    path1 = _path_batch()
    path2 = _path_batch() + 0.2
    got = np.asarray(
        adapter.run_signaturekernel_backprop(path1, path2, 2, 2)()
    )

    X = adapter._path_tensor(path1, requires_grad=True)
    Y = adapter._path_tensor(path2)
    output = adapter._signature_kernel().compute_Gram(
        X,
        Y,
        sym=False,
        max_batch=2,
    )
    cotangent = adapter.torch.as_tensor(
        adapter.random_cotangent(tuple(output.shape)),
        dtype=output.dtype,
        device=output.device,
    )
    expected = adapter.torch.autograd.grad(
        output,
        X,
        cotangent,
        retain_graph=True,
    )[0]

    np.testing.assert_allclose(
        got,
        expected.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
