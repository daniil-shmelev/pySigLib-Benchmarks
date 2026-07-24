"""Tests for the signature-rs Python adapter."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from adapters.python.run_signature_rs import SignatureRSAdapter
from common.paths import make_path


def _config() -> dict:
    return {
        "N": 4,
        "d": 5,
        "m": 2,
        "path_kind": "linear",
        "operation": "logsignature",
        "repeats": 1,
        "batch_size": 2,
        "backend": "cpu",
    }


def test_signature_rs_adapter_supports_f32_multidimensional_batches():
    pytest.importorskip("signature_py")

    adapter = SignatureRSAdapter(_config())
    path = make_path(5, 4, "linear").astype(np.float64)
    batch = np.stack([path, path + 0.1], axis=0)

    prepared = adapter._prepare_paths(batch, 5)
    results = adapter.run_logsignature(batch, 5, 2)()

    assert prepared.dtype == np.float32
    assert prepared.flags.c_contiguous
    assert len(results) == 2
    assert all(len(result.coefficients) == 15 for result in results)
