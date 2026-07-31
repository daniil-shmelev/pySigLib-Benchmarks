"""Unit tests for path generation utilities"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from common.adapter import BenchmarkAdapter
from common.paths import make_path_batch, make_path_linear, make_path_sin, make_path


class TestPathGeneration:
    """Test suite for path generation functions"""

    def test_linear_path_shape(self):
        """Test that linear path has correct shape"""
        N, d = 100, 3
        path = make_path_linear(d, N)
        assert path.shape == (N, d), f"Expected shape ({N}, {d}), got {path.shape}"

    def test_linear_path_first_dimension(self):
        """Test that first dimension is linear time [0, 1]"""
        N, d = 50, 4
        path = make_path_linear(d, N)
        # First column should be linearly spaced from 0 to 1
        expected_first_col = np.linspace(0.0, 1.0, N, dtype=np.float32)
        np.testing.assert_array_equal(path[:, 0], expected_first_col)

    def test_linear_path_other_dimensions(self):
        """Test that other dimensions are 2*t"""
        N, d = 50, 4
        path = make_path_linear(d, N)
        ts = np.linspace(0.0, 1.0, N, dtype=np.float32)
        for i in range(1, d):
            np.testing.assert_array_equal(path[:, i], 2.0 * ts)

    def test_linear_path_1d(self):
        """Test linear path with d=1"""
        N, d = 30, 1
        path = make_path_linear(d, N)
        assert path.shape == (N, 1)
        expected = np.linspace(0.0, 1.0, N, dtype=np.float32)[:, None]
        np.testing.assert_array_equal(path, expected)

    def test_sin_path_shape(self):
        """Test that sinusoidal path has correct shape"""
        N, d = 100, 5
        path = make_path_sin(d, N)
        assert path.shape == (N, d), f"Expected shape ({N}, {d}), got {path.shape}"

    def test_sin_path_values(self):
        """Test that sinusoidal path has correct values"""
        N, d = 10, 2
        path = make_path_sin(d, N)
        ts = np.linspace(0.0, 1.0, N)
        omega = 2.0 * np.pi

        # First dimension: sin(2π * 1 * t)
        expected_col0 = np.sin(omega * 1 * ts)
        np.testing.assert_allclose(path[:, 0], expected_col0, rtol=1e-6, atol=1e-6)

        # Second dimension: sin(2π * 2 * t)
        expected_col1 = np.sin(omega * 2 * ts)
        np.testing.assert_allclose(path[:, 1], expected_col1, rtol=1e-6, atol=1e-6)

    def test_sin_path_bounds(self):
        """Test that sinusoidal path values are within [-1, 1]"""
        N, d = 100, 3
        path = make_path_sin(d, N)
        assert np.all(path >= -1.0) and np.all(path <= 1.0), \
            "Sinusoidal path values should be in [-1, 1]"

    def test_make_path_dispatcher_linear(self):
        """Test make_path dispatcher with 'linear' kind"""
        N, d = 20, 3
        path1 = make_path(d, N, "linear")
        path2 = make_path_linear(d, N)
        np.testing.assert_array_equal(path1, path2)

    def test_make_path_dispatcher_sin(self):
        """Test make_path dispatcher with 'sin' kind"""
        N, d = 20, 3
        path1 = make_path(d, N, "sin")
        path2 = make_path_sin(d, N)
        np.testing.assert_array_equal(path1, path2)

    def test_make_path_case_insensitive(self):
        """Test that make_path is case-insensitive"""
        N, d = 15, 2
        path_lower = make_path(d, N, "linear")
        path_upper = make_path(d, N, "LINEAR")
        path_mixed = make_path(d, N, "LiNeAr")
        np.testing.assert_array_equal(path_lower, path_upper)
        np.testing.assert_array_equal(path_lower, path_mixed)

    def test_make_path_invalid_kind(self):
        """Test that make_path raises error for invalid kind"""
        with pytest.raises(ValueError, match="Unknown path_kind"):
            make_path(2, 10, "invalid")

    def test_make_path_batch_shape(self):
        """Test that batched paths have a leading batch dimension."""
        batch_size, N, d = 4, 20, 3
        paths = make_path_batch(d, N, "linear", batch_size)
        assert paths.shape == (batch_size, N, d)

    def test_make_path_batch_repeats_deterministic_paths(self):
        """Test that deterministic path batches repeat the base path."""
        batch = make_path_batch(2, 10, "sin", 3)
        expected = make_path(2, 10, "sin")
        for path in batch:
            np.testing.assert_array_equal(path, expected)

    def test_make_path_batch_invalid_size(self):
        """Test that batch_size must be positive."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            make_path_batch(2, 10, "linear", 0)

    def test_path_dtype(self):
        """All shared inputs use the benchmark's canonical float32 precision."""
        path_linear = make_path_linear(3, 50)
        path_sin = make_path_sin(3, 50)
        assert path_linear.dtype == np.float32
        assert path_sin.dtype == np.float32

    def test_path_contiguous(self):
        """Test that paths are C-contiguous arrays"""
        path_linear = make_path_linear(3, 50)
        path_sin = make_path_sin(3, 50)
        assert path_linear.flags['C_CONTIGUOUS']
        assert path_sin.flags['C_CONTIGUOUS']

    def test_seeded_fbm_inputs_are_saved_and_reused(self, tmp_path):
        """Logical inputs are durable, reproducible, and checksummed."""
        config = {
            "N": 16,
            "d": 2,
            "m": 2,
            "path_kind": "fbm",
            "operation": "signature",
            "repeats": 2,
            "batch_size": 2,
            "seed": 1234,
            "input_cache_dir": str(tmp_path),
        }
        first = BenchmarkAdapter(config)
        second = BenchmarkAdapter(config)

        first_x = first.make_path_input()
        first_y = first.make_path_input()
        second_x = second.make_path_input()
        second_y = second.make_path_input()

        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_y, second_y)
        assert not np.array_equal(first_x, first_y)

        cache_files = sorted(tmp_path.glob("*.npy"))
        checksum_files = sorted(tmp_path.glob("*.npy.sha256"))
        assert len(cache_files) == 2
        assert len(checksum_files) == 2
        for cache_path in cache_files:
            digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            checksum = cache_path.with_suffix(".npy.sha256").read_text(
                encoding="utf-8"
            )
            assert checksum == f"{digest}  {cache_path.name}\n"

    def test_timing_loop_retains_raw_samples(self):
        adapter = BenchmarkAdapter({
            "N": 4,
            "d": 2,
            "m": 2,
            "path_kind": "linear",
            "operation": "signature",
            "repeats": 3,
        })

        median_ms, alloc_bytes, samples_ms = adapter.manual_timing_loop(lambda: None)

        assert len(samples_ms) == 3
        assert median_ms == np.median(samples_ms)
        assert alloc_bytes >= 0
