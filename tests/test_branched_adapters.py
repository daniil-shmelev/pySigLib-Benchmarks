"""Tests for branched signature adapter paths."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from adapters.python.run_pysiglib import PySigLibAdapter
from adapters.python.run_stochastax import StochastaxAdapter
from common.paths import make_path


def _config(operation: str) -> dict:
    return {
        "N": 4,
        "d": 2,
        "m": 2,
        "path_kind": "linear",
        "operation": operation,
        "repeats": 1,
        "backend": "cpu",
    }


@pytest.mark.parametrize(
    ("planar", "operation"),
    [
        (False, "branchedsignature_nonplanar"),
        (True, "branchedsignature_planar"),
    ],
)
def test_stochastax_branchedsignature_matches_direct_api(planar, operation):
    pytest.importorskip("jax")
    pytest.importorskip("stochastax")
    from stochastax.control_lifts.branched_signature_ito import (
        GLHopfAlgebra,
        MKWHopfAlgebra,
        compute_nonplanar_branched_signature,
        compute_planar_branched_signature,
    )

    adapter = StochastaxAdapter(_config(operation))
    path = make_path(2, 4, "linear")
    got = np.asarray(adapter.run_branchedsignature(path, 2, 2, planar=planar)())
    path_jax = adapter._path_array(path)
    cov_increments = adapter._zero_cov_increments(path_jax)
    if planar:
        hopf = MKWHopfAlgebra.build(2, 2)
        expected = compute_planar_branched_signature(
            path_jax,
            2,
            hopf,
            "full",
            cov_increments,
        ).flatten()
    else:
        hopf = GLHopfAlgebra.build(2, 2)
        expected = compute_nonplanar_branched_signature(
            path_jax,
            2,
            hopf,
            "full",
            cov_increments,
        ).flatten()

    np.testing.assert_allclose(got, np.asarray(expected), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("planar", [False, True])
def test_stochastax_branchedsignature_backprop_returns_path_gradient(planar):
    pytest.importorskip("jax")
    pytest.importorskip("stochastax")

    adapter = StochastaxAdapter(
        _config("branchedsignature_planar_backprop")
    )
    path = make_path(2, 4, "linear")
    gradient = np.asarray(
        adapter.run_branchedsignature_backprop(
            path,
            2,
            2,
            planar=planar,
        )()
    )

    assert gradient.shape == path.shape
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize(
    ("planar", "operation"),
    [
        (False, "branchedlogsignature_nonplanar"),
        (True, "branchedlogsignature_planar"),
    ],
)
def test_stochastax_branchedlogsignature_matches_direct_api(planar, operation):
    pytest.importorskip("jax")
    pytest.importorskip("stochastax")
    from stochastax.control_lifts.branched_signature_ito import (
        GLHopfAlgebra,
        MKWHopfAlgebra,
        compute_nonplanar_branched_signature,
        compute_planar_branched_signature,
    )

    adapter = StochastaxAdapter(_config(operation))
    path = make_path(2, 4, "linear")
    got = np.asarray(
        adapter.run_branchedlogsignature(path, 2, 2, planar=planar)()
    )
    path_jax = adapter._path_array(path)
    cov_increments = adapter._zero_cov_increments(path_jax)
    if planar:
        hopf = MKWHopfAlgebra.build(2, 2)
        signature = compute_planar_branched_signature(
            path_jax,
            2,
            hopf,
            "full",
            cov_increments,
        )
    else:
        hopf = GLHopfAlgebra.build(2, 2)
        signature = compute_nonplanar_branched_signature(
            path_jax,
            2,
            hopf,
            "full",
            cov_increments,
        )

    np.testing.assert_allclose(
        got,
        np.asarray(signature.log().flatten()),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("planar", [False, True])
def test_stochastax_branchedlogsignature_backprop_returns_path_gradient(planar):
    pytest.importorskip("jax")
    pytest.importorskip("stochastax")

    adapter = StochastaxAdapter(
        _config("branchedlogsignature_planar_backprop")
    )
    path = make_path(2, 4, "linear")
    gradient = np.asarray(
        adapter.run_branchedlogsignature_backprop(
            path,
            2,
            2,
            planar=planar,
        )()
    )

    assert gradient.shape == path.shape
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize(
    ("planar", "operation"),
    [
        (False, "branchedsignature_nonplanar"),
        (True, "branchedsignature_planar"),
    ],
)
def test_pysiglib_branchedsignature_matches_standard_api(planar, operation):
    """pySigLib adapter uses the standard branched signature API."""
    pytest.importorskip("pysiglib")

    adapter = PySigLibAdapter(_config(operation))
    path = make_path(2, 4, "linear")

    kernel = adapter.run_branchedsignature(path, 2, 2, planar=planar)
    got = np.asarray(kernel())

    path_array = adapter._path_array(path)
    adapter.pysiglib.prepare_branched_sig(2, 2, planar=planar)
    expected = np.asarray(
        adapter.pysiglib.branched_sig(path_array, degree=2, planar=planar)
    )
    expected_len = adapter.pysiglib.branched_sig_length(2, 2, planar=planar)

    assert got.dtype == np.float32
    assert got.shape == (expected_len,)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("planar", [False, True])
def test_pysiglib_branchedsignature_backprop_matches_standard_api(planar):
    pytest.importorskip("pysiglib")

    adapter = PySigLibAdapter(
        _config("branchedsignature_planar_backprop")
    )
    path = make_path(2, 4, "linear")
    got = np.asarray(
        adapter.run_branchedsignature_backprop(
            path,
            2,
            2,
            planar=planar,
        )()
    )

    path_array = adapter._path_array(path)
    adapter.pysiglib.prepare_branched_sig(2, 2, planar=planar)
    output = adapter.pysiglib.branched_sig(
        path_array,
        degree=2,
        planar=planar,
    )
    cotangent = np.asarray(
        adapter.random_cotangent(output.shape), dtype=output.dtype
    )
    expected = np.asarray(
        adapter.pysiglib.branched_sig_backprop(
            path_array,
            output,
            cotangent,
            2,
            planar=planar,
        )
    )

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
@pytest.mark.parametrize(
    ("planar", "operation"),
    [
        (False, "branchedlogsignature_nonplanar"),
        (True, "branchedlogsignature_planar"),
    ],
)
def test_pysiglib_branchedlogsignature_matches_standard_api(
    planar,
    operation,
    monkeypatch,
):
    pytest.importorskip("pysiglib")

    adapter = PySigLibAdapter(_config(operation))
    path = make_path(2, 4, "linear")
    prepare_branched_log_sig = adapter.pysiglib.prepare_branched_log_sig
    prepare_calls = []

    def checked_prepare(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return prepare_branched_log_sig(*args, **kwargs)

    monkeypatch.setattr(
        adapter.pysiglib,
        "prepare_branched_log_sig",
        checked_prepare,
    )
    got = np.asarray(
        adapter.run_branchedlogsignature(
            path,
            2,
            2,
            planar=planar,
        )()
    )

    path_array = adapter._path_array(path)
    method = adapter._branched_log_sig_method(planar)
    assert prepare_calls == [
        (
            (2, 2),
            {
                "method": method,
                "planar": planar,
                "device": "cpu",
            },
        )
    ]
    prepare_branched_log_sig(
        2,
        2,
        method=method,
        planar=planar,
        device="cpu",
    )
    expected = np.asarray(
        adapter.pysiglib.branched_log_sig(
            path_array,
            degree=2,
            method=method,
            planar=planar,
        )
    )

    assert got.dtype == np.float32
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("planar", [False, True])
def test_pysiglib_branchedlogsignature_backprop_matches_standard_api(
    planar,
    monkeypatch,
):
    pytest.importorskip("pysiglib")

    adapter = PySigLibAdapter(
        _config("branchedlogsignature_planar_backprop")
    )
    path = make_path(2, 4, "linear")
    prepare_branched_log_sig = adapter.pysiglib.prepare_branched_log_sig
    prepare_calls = []

    def checked_prepare(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return prepare_branched_log_sig(*args, **kwargs)

    monkeypatch.setattr(
        adapter.pysiglib,
        "prepare_branched_log_sig",
        checked_prepare,
    )
    got = np.asarray(
        adapter.run_branchedlogsignature_backprop(
            path,
            2,
            2,
            planar=planar,
        )()
    )

    path_array = adapter._path_array(path)
    method = adapter._branched_log_sig_method(planar)
    assert prepare_calls == [
        (
            (2, 2),
            {
                "method": method,
                "planar": planar,
                "device": "cpu",
            },
        )
    ]
    prepare_branched_log_sig(
        2,
        2,
        method=method,
        planar=planar,
        device="cpu",
    )
    branched_signature = adapter.pysiglib.branched_sig(
        path_array,
        degree=2,
        planar=planar,
    )
    output = adapter.pysiglib.branched_sig_to_log_sig(
        branched_signature,
        2,
        2,
        method=method,
        planar=planar,
    )
    cotangent = np.asarray(
        adapter.random_cotangent(output.shape),
        dtype=output.dtype,
    )
    branched_signature_cotangent = (
        adapter.pysiglib.branched_sig_to_log_sig_backprop(
            branched_signature,
            cotangent,
            2,
            2,
            method=method,
            planar=planar,
        )
    )
    expected = np.asarray(
        adapter.pysiglib.branched_sig_backprop(
            path_array,
            branched_signature,
            branched_signature_cotangent,
            2,
            planar=planar,
        )
    )

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
