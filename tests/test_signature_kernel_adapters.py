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


def test_pysiglib_polynomial_kernel_matches_standard_api():
    pytest.importorskip("pysiglib")

    config = _config("signaturekernel")
    config["sig_kernel_method"] = "polynomial"
    config["sig_kernel_order"] = 2
    adapter = PySigLibAdapter(config)
    path1 = _path_batch()
    path2 = _path_batch() + 0.2

    got = np.asarray(adapter.run_signaturekernel(path1, path2, 2, 2)())
    X = adapter._path_array(path1)
    Y = adapter._path_array(path2)
    expected = np.asarray(
        adapter.pysiglib.sig_kernel_gram(
            X,
            Y,
            method="polynomial",
            order=2,
        )
    )

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


def test_pysiglib_signaturekernel_backprop_matches_standard_api(monkeypatch):
    pytest.importorskip("pysiglib")

    adapter = PySigLibAdapter(_config("signaturekernel_backprop"))
    path1 = _path_batch()
    path2 = _path_batch() + 0.2
    forward = adapter.pysiglib.sig_kernel_gram
    backprop = adapter.pysiglib.sig_kernel_gram_backprop
    calls = {}

    def checked_forward(*args, **kwargs):
        calls["forward_return_grid"] = kwargs.get("return_grid")
        return forward(*args, **kwargs)

    def checked_backprop(*args, **kwargs):
        calls["backprop_k_grid"] = kwargs.get("k_grid")
        calls["backprop_return_grid"] = kwargs.get("return_grid")
        return backprop(*args, **kwargs)

    monkeypatch.setattr(adapter.pysiglib, "sig_kernel_gram", checked_forward)
    monkeypatch.setattr(
        adapter.pysiglib,
        "sig_kernel_gram_backprop",
        checked_backprop,
    )
    got = np.asarray(
        adapter.run_signaturekernel_backprop(path1, path2, 2, 2)()
    )

    X = adapter._path_array(path1)
    Y = adapter._path_array(path2)
    k_grid = forward(
        X,
        Y,
        dyadic_order=2,
        return_grid=True,
    )
    output = k_grid[..., -1, -1]
    cotangent = np.asarray(
        adapter.random_cotangent(output.shape), dtype=output.dtype
    )
    expected, _ = backprop(
        cotangent,
        X,
        Y,
        dyadic_order=2,
        left_deriv=True,
        right_deriv=False,
        k_grid=k_grid,
        return_grid=False,
    )

    assert calls["forward_return_grid"] is True
    assert calls["backprop_k_grid"] is not None
    assert calls["backprop_return_grid"] is False
    np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_pysiglib_polynomial_backprop_does_not_use_a_pde_grid(monkeypatch):
    pytest.importorskip("pysiglib")

    config = _config("signaturekernel_backprop")
    config["sig_kernel_method"] = "polynomial"
    config["sig_kernel_order"] = 2
    adapter = PySigLibAdapter(config)
    path1 = _path_batch()
    path2 = _path_batch() + 0.2
    backprop = adapter.pysiglib.sig_kernel_gram_backprop
    calls = {}

    def checked_backprop(*args, **kwargs):
        calls.update(kwargs)
        return backprop(*args, **kwargs)

    monkeypatch.setattr(
        adapter.pysiglib,
        "sig_kernel_gram_backprop",
        checked_backprop,
    )
    got = np.asarray(
        adapter.run_signaturekernel_backprop(path1, path2, 2, 2)()
    )

    X = adapter._path_array(path1)
    Y = adapter._path_array(path2)
    output = adapter.pysiglib.sig_kernel_gram(
        X,
        Y,
        method="polynomial",
        order=2,
    )
    cotangent = np.asarray(
        adapter.random_cotangent(output.shape),
        dtype=output.dtype,
    )
    expected, _ = backprop(
        cotangent,
        X,
        Y,
        method="polynomial",
        order=2,
        left_deriv=True,
        right_deriv=False,
        return_grid=False,
    )

    assert calls["method"] == "polynomial"
    assert calls["order"] == 2
    assert calls["k_grid"] is None
    assert calls["return_grid"] is False
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


def test_sigkernel_backprop_matches_direct_autograd(monkeypatch):
    pytest.importorskip("sigkernel")

    config = _config("signaturekernel_backprop")
    config["sig_kernel_dyadic_order"] = 0
    adapter = SigkernelAdapter(config)
    path1 = _path_batch()
    path2 = _path_batch() + 0.2
    signature_kernel = adapter._signature_kernel()
    compute_gram = signature_kernel.compute_Gram
    calls = []

    def checked_compute_gram(*args, **kwargs):
        calls.append(None)
        return compute_gram(*args, **kwargs)

    monkeypatch.setattr(adapter, "_signature_kernel", lambda: signature_kernel)
    monkeypatch.setattr(
        signature_kernel,
        "compute_Gram",
        checked_compute_gram,
    )
    gradient = adapter.run_signaturekernel_backprop(path1, path2, 2, 2)

    assert len(calls) == 1
    got = np.asarray(gradient())
    assert len(calls) == 1

    X = adapter._path_tensor(path1, requires_grad=True)
    Y = adapter._path_tensor(path2)
    output = compute_gram(
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


def test_sigkernel_backprop_benchmark_times_backward_call(monkeypatch):
    adapter = SigkernelAdapter.__new__(SigkernelAdapter)
    adapter.operation = "signaturekernel_backprop"
    adapter.d = 2
    adapter.m = 3
    adapter.dyadic_order = 0
    adapter.dtype_name = "float64"
    paths = iter([object(), object()])
    backward_kernel = object()

    monkeypatch.setattr(adapter, "make_path_input", lambda: next(paths))
    monkeypatch.setattr(
        adapter,
        "run_signaturekernel_backprop",
        lambda *args: backward_kernel,
    )
    timed_kernels = []

    def timing_result(kernel):
        timed_kernels.append(kernel)
        return 250.0, 0, [250.0, 270.0, 260.0]

    monkeypatch.setattr(adapter, "manual_timing_loop", timing_result)
    monkeypatch.setattr(adapter, "output_result", lambda **kwargs: kwargs)

    result = adapter._run_benchmark()

    assert timed_kernels == [backward_kernel]
    assert result["t_ms"] == 250.0
    assert result["samples_ms"] == [250.0, 270.0, 260.0]
    assert result["alloc_bytes"] == 0
    assert result["method"].startswith("backward(SigKernel.compute_Gram)")
