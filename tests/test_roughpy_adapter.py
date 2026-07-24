"""Tests for the RoughPy signature adapter."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from adapters.python.run_roughpy import RoughPyAdapter
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


@pytest.mark.parametrize("operation", ["signature", "logsignature"])
def test_roughpy_adapter_uses_f32_and_preserves_batch(operation):
    pytest.importorskip("roughpy")

    adapter = RoughPyAdapter(_config(operation))
    path = _path_batch()
    if operation == "signature":
        kernel = adapter.run_signature(path, 2, 2)
    else:
        kernel = adapter.run_logsignature(path, 2, 2)

    results = kernel()

    assert len(results) == 2
    assert all(np.asarray(result).dtype == np.float32 for result in results)
