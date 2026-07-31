"""Tests for the Signatory adapter."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from adapters.python.run_signatory import SignatoryAdapter


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
    package = types.ModuleType("signatory")

    class Signature(torch.nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.depth = depth

        def forward(self, path):
            return path.sum(dim=-2).repeat_interleave(self.depth, dim=-1)

    class LogSignature(torch.nn.Module):
        def __init__(self, depth, mode):
            super().__init__()
            assert mode == "brackets"
            self.depth = depth

        def forward(self, path):
            return path.mean(dim=-2).repeat_interleave(self.depth, dim=-1)

    package.Signature = Signature
    package.LogSignature = LogSignature
    monkeypatch.setitem(sys.modules, "signatory", package)

    adapter = SignatoryAdapter(_config(operation))
    path = np.zeros((3, 5, 2), dtype=np.float64)
    if operation == "signature":
        result = adapter.run_signature(path, 2, 2)()
    elif operation == "logsignature":
        result = adapter.run_logsignature(path, 2, 2)()
    elif operation == "sig_backprop":
        result = adapter.run_sig_backprop(path, 2, 2)()[0]
    else:
        result = adapter.run_logsignature_backprop(path, 2, 2)()[0]

    assert result.dtype == torch.float32
    assert result.shape[0] == 3
