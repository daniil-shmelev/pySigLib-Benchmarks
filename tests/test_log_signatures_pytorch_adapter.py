"""Tests for the log-signatures-pytorch adapter."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from adapters.python.run_log_signatures_pytorch import (
    LogSignaturesPyTorchAdapter,
)


def _config(operation):
    return {
        "N": 5,
        "d": 2,
        "m": 2,
        "path_kind": "linear",
        "operation": operation,
        "repeats": 1,
        "batch_size": 3,
        "backend": "cpu",
    }


@pytest.mark.parametrize(
    "operation",
    ["signature", "logsignature", "sig_backprop", "logsignature_backprop"],
)
def test_cpu_adapter_uses_native_float32_batch(monkeypatch, operation):
    torch = pytest.importorskip("torch")
    package = types.ModuleType("log_signatures_pytorch")

    def signature(path, depth):
        return path.sum(dim=-2).repeat_interleave(depth, dim=-1)

    def log_signature(path, depth, method, mode):
        assert method == "default"
        assert mode == "words"
        return path.mean(dim=-2).repeat_interleave(depth, dim=-1)

    package.signature = signature
    package.log_signature = log_signature
    monkeypatch.setitem(sys.modules, "log_signatures_pytorch", package)

    adapter = LogSignaturesPyTorchAdapter(_config(operation))
    path = np.zeros((3, 5, 2), dtype=np.float64)
    if operation == "signature":
        result = adapter.run_signature(path, 2, 2)()
    elif operation == "logsignature":
        result = adapter.run_logsignature(path, 2, 2)()
    elif operation == "sig_backprop":
        result = adapter.run_sig_backprop(path, 2, 2)()
    else:
        result = adapter.run_logsignature_backprop(path, 2, 2)()

    assert result.dtype == torch.float32
    assert result.shape[0] == 3


@pytest.mark.parametrize(
    "operation",
    ["logsignature", "logsignature_backprop"],
)
def test_cpu_adapter_uses_configured_bch_method(monkeypatch, operation):
    torch = pytest.importorskip("torch")
    package = types.ModuleType("log_signatures_pytorch")

    def signature(path, depth):
        return path.sum(dim=-2).repeat_interleave(depth, dim=-1)

    def log_signature(path, depth, method, mode):
        assert method == "bch_sparse"
        assert mode == "hall"
        return path.mean(dim=-2).repeat_interleave(depth, dim=-1)

    package.signature = signature
    package.log_signature = log_signature
    monkeypatch.setitem(sys.modules, "log_signatures_pytorch", package)
    config = _config(operation)
    config.update({
        "log_signature_method": "bch_sparse",
        "log_signature_mode": "hall",
    })

    adapter = LogSignaturesPyTorchAdapter(config)
    path = np.zeros((3, 5, 2), dtype=np.float64)
    if operation == "logsignature":
        result = adapter.run_logsignature(path, 2, 2)()
    else:
        result = adapter.run_logsignature_backprop(path, 2, 2)()

    assert result.dtype == torch.float32
    assert result.shape[0] == 3
