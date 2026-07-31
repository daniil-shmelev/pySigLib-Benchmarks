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
    }


@pytest.mark.parametrize(
    ("planar", "operation"),
    [
        (False, "branchedsignature_nonplanar"),
        (True, "branchedsignature_planar"),
    ],
)
def test_stochastax_branchedsignature_matches_direct_api(planar, operation):
    """Stochastax adapter calls the matching planar/non-planar branched API."""
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

    kernel = adapter.run_branchedsignature(path, 2, 2, planar=planar)
    got = np.asarray(kernel())

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

    expected = np.asarray(expected)
    assert got.dtype == np.float32
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("planar", "operation"),
    [
        (False, "branchedsignature_nonplanar"),
        (True, "branchedsignature_planar"),
    ],
)
def test_pysiglib_branchedsignature_matches_jax_api(planar, operation):
    """pySigLib adapter uses the JAX branched signature API."""
    pytest.importorskip("jax")
    pytest.importorskip("pysiglib.jax_api")

    adapter = PySigLibAdapter(_config(operation))
    path = make_path(2, 4, "linear")

    kernel = adapter.run_branchedsignature(path, 2, 2, planar=planar)
    got = np.asarray(kernel())

    path_jax = adapter._path_array(path)
    adapter.pysiglib.prepare_branched_sig(2, 2, planar=planar)
    expected = np.asarray(
        adapter.pysiglib.branched_sig(path_jax, degree=2, planar=planar)
    )
    expected_len = adapter.pysiglib.branched_sig_length(2, 2, planar=planar)

    assert got.dtype == np.float32
    assert got.shape == (expected_len,)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("planar", [False, True])
def test_pysiglib_branchedsignature_backprop_matches_direct_vjp(planar):
    pytest.importorskip("jax")
    pytest.importorskip("pysiglib.jax_api")

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

    path_jax = adapter._path_array(path)
    adapter.pysiglib.prepare_branched_sig(2, 2, planar=planar)
    output, pullback = adapter.jax.vjp(
        lambda path_arg: adapter.pysiglib.branched_sig(
            path_arg,
            degree=2,
            planar=planar,
        ),
        path_jax,
    )
    cotangent = adapter.jnp.asarray(
        adapter.random_cotangent(output.shape),
        dtype=output.dtype,
    )
    expected = np.asarray(pullback(cotangent)[0])

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


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
