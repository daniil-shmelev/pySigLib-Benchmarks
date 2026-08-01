"""Tests for the pySigLib standard API adapter."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

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
        "backend": "cpu",
        "log_sig_method": 2,
    }


def test_signature_matches_standard_api():
    pytest.importorskip("pysiglib")
    adapter = PySigLibAdapter(_config("signature"))
    path = make_path(2, 4, "linear")

    got = np.asarray(adapter.run_signature(path, 2, 2)())
    path_array = adapter._path_array(path)
    expected = np.asarray(adapter.pysiglib.signature(path_array, degree=2))

    assert adapter.pysiglib.__name__ == "pysiglib"
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_signature_backprop_matches_standard_api():
    pytest.importorskip("pysiglib")
    adapter = PySigLibAdapter(_config("sig_backprop"))
    path = make_path(2, 4, "linear")

    got = np.asarray(adapter.run_sig_backprop(path, 2, 2)())
    path_array = adapter._path_array(path)
    signature = adapter.pysiglib.signature(path_array, degree=2)
    cotangent = np.asarray(
        adapter.random_cotangent(signature.shape), dtype=signature.dtype
    )
    expected = np.asarray(
        adapter.pysiglib.sig_backprop(
            path_array,
            signature,
            cotangent,
            2,
        )
    )

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_logsignature_backprop_matches_standard_api():
    pytest.importorskip("pysiglib")
    adapter = PySigLibAdapter(_config("logsignature_backprop"))
    path = make_path(2, 4, "linear")

    got = np.asarray(adapter.run_logsignature_backprop(path, 2, 2)())
    path_array = adapter._path_array(path)
    adapter.pysiglib.prepare_log_sig(2, 2, method=2, device="cpu")
    signature = adapter.pysiglib.signature(
        path_array,
        degree=2,
        scalar_term=True,
    )
    log_signature = adapter.pysiglib.sig_to_log_sig(
        signature,
        2,
        2,
        method=2,
    )
    cotangent = np.asarray(
        adapter.random_cotangent(log_signature.shape),
        dtype=log_signature.dtype,
    )
    signature_cotangent = adapter.pysiglib.sig_to_log_sig_backprop(
        signature,
        cotangent,
        2,
        2,
        method=2,
    )
    expected = np.asarray(
        adapter.pysiglib.sig_backprop(
            path_array,
            signature,
            signature_cotangent,
            2,
        )
    )

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_logsignature_method_3_backprop_matches_native_api():
    pytest.importorskip("pysiglib")
    config = _config("logsignature_backprop")
    config["log_sig_method"] = 3
    adapter = PySigLibAdapter(config)
    path = make_path(2, 4, "linear")

    got = np.asarray(adapter.run_logsignature_backprop(path, 2, 2)())
    path_array = adapter._path_array(path)
    log_signature = adapter.pysiglib.log_sig(path_array, 2, method=3)
    cotangent = np.asarray(
        adapter.random_cotangent(log_signature.shape),
        dtype=log_signature.dtype,
    )
    expected = np.asarray(
        adapter.log_sig_from_path_backprop(
            cotangent,
            path_array,
            2,
        )
    )

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
