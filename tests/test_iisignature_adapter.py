"""Tests for iisignature backward adapters."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from adapters.python.run_iisignature import IISignatureAdapter


def _config(operation: str) -> dict:
    return {
        "N": 5,
        "d": 2,
        "m": 2,
        "path_kind": "linear",
        "operation": operation,
        "repeats": 1,
        "batch_size": 2,
        "backend": "cpu",
        "seed": 17,
    }


def _path() -> np.ndarray:
    base = np.arange(10, dtype=np.float32).reshape(5, 2) / 10
    return np.stack((base, base + 0.1))


def test_sig_backprop_uses_seeded_random_cotangent():
    iisignature = pytest.importorskip("iisignature")
    adapter = IISignatureAdapter(_config("sig_backprop"))
    path = _path()
    output = iisignature.sig(path, 2)
    cotangent = adapter.random_cotangent(output.shape)

    got = adapter.run_sig_backprop(path, 2, 2)()
    expected = iisignature.sigbackprop(cotangent, path, 2)

    assert not np.all(cotangent == 1)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_logsignature_backprop_uses_seeded_random_cotangent():
    iisignature = pytest.importorskip("iisignature")
    adapter = IISignatureAdapter(_config("logsignature_backprop"))
    path = _path()
    basis = iisignature.prepare(2, 2, "S")
    output = iisignature.logsig(path, basis, "S")
    cotangent = adapter.random_cotangent(output.shape)

    got = adapter.run_logsignature_backprop(path, 2, 2)()
    expected = iisignature.logsigbackprop(cotangent, path, basis, "S")

    assert not np.all(cotangent == 1)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("method", ["C", "O"])
def test_compiled_bch_method_does_not_claim_backprop_support(monkeypatch, method):
    package = types.ModuleType("iisignature")
    package.prepare = lambda *args, **kwargs: pytest.fail(
        "BCH backprop must be rejected before preparation"
    )
    monkeypatch.setitem(sys.modules, "iisignature", package)
    config = _config("logsignature_backprop")
    config["logsig_method"] = method

    adapter = IISignatureAdapter(config)

    with pytest.raises(NotImplementedError, match="iisignature BCH logsignature backprop is unsupported"):
        adapter.run_logsignature_backprop(_path(), 2, 2)
