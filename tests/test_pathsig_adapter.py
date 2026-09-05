"""Tests for the PathSig benchmark adapter."""

import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from adapters.python.run_pathsig import PathSigAdapter


def test_prepare_paths_adds_batch_axis_and_uses_float32():
    path = np.zeros((5, 2), dtype=np.float64)

    paths = PathSigAdapter._prepare_paths(path, d=2)

    assert paths.shape == (1, 5, 2)
    assert paths.dtype == np.float32
    assert paths.flags.c_contiguous


def test_pathsig_is_registered_as_gpu_only():
    registry = yaml.safe_load(
        (REPO_ROOT / "config" / "libraries_registry.yaml").read_text(
            encoding="utf-8"
        )
    )

    config = registry["libraries"]["pathsig"]
    assert config["backend"] == "gpu"
    assert config["env"]["NVCC_APPEND_FLAGS"] == "--std=c++20"
    assert config["operations"] == [
        "signature",
        "logsignature",
        "sig_backprop",
        "logsignature_backprop",
    ]
