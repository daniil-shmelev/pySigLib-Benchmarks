"""Path generation utilities"""

import math
from typing import Literal

import numpy as np


def make_path_linear(d: int, N: int) -> np.ndarray:
    """
    Generate linear path: [t, 2t, 2t, ...]

    Args:
        d: Dimension of the path
        N: Number of points

    Returns:
        Array of shape (N, d)
    """
    ts = np.linspace(0.0, 1.0, N, dtype=np.float32)
    path = np.empty((N, d), dtype=np.float32)
    path[:, 0] = ts
    if d > 1:
        path[:, 1:] = 2.0 * ts[:, None]
    return path


def make_path_sin(d: int, N: int) -> np.ndarray:
    """
    Generate sinusoidal path: [sin(2π·1·t), sin(2π·2·t), ...]

    Args:
        d: Dimension of the path
        N: Number of points

    Returns:
        Array of shape (N, d)
    """
    ts = np.linspace(0.0, 1.0, N, dtype=np.float32)
    omega = np.float32(2.0 * math.pi)
    ks = np.arange(1, d + 1, dtype=np.float32)
    path = np.sin(omega * ts[:, None] * ks[None, :])
    return path


def make_path_brownian(d: int, N: int) -> np.ndarray:
    """Generate a standard Brownian motion on an N-point grid over [0, 1]."""
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    path = np.zeros((N, d), dtype=np.float32)
    if N == 1:
        return path

    increments = np.random.standard_normal((N - 1, d)).astype(np.float32)
    increments *= np.float32(math.sqrt(1.0 / (N - 1)))
    np.cumsum(increments, axis=0, dtype=np.float32, out=path[1:])
    return path


def make_path_brownian_batch(d: int, N: int, batch_size: int) -> np.ndarray:
    """Generate independent standard Brownian paths in one vectorized call."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")

    paths = np.zeros((batch_size, N, d), dtype=np.float32)
    if N == 1:
        return paths

    increments = np.random.standard_normal(
        (batch_size, N - 1, d)
    ).astype(np.float32)
    increments *= np.float32(math.sqrt(1.0 / (N - 1)))
    np.cumsum(increments, axis=1, dtype=np.float32, out=paths[:, 1:])
    return paths


def make_path(
    d: int,
    N: int,
    kind: Literal["linear", "sin", "brownian"],
) -> np.ndarray:
    """
    Generate path of specified kind.

    Args:
        d: Dimension of the path
        N: Number of points
        kind: "linear", "sin", or "brownian"

    Returns:
        Array of shape (N, d)
    """
    kind = kind.lower()

    if kind == "linear":
        return make_path_linear(d, N)
    elif kind == "sin":
        return make_path_sin(d, N)
    elif kind == "brownian":
        return make_path_brownian(d, N)
    else:
        raise ValueError(f"Unknown path_kind: {kind}")


def make_path_batch(
    d: int,
    N: int,
    kind: Literal["linear", "sin", "brownian"],
    batch_size: int,
) -> np.ndarray:
    """
    Generate a batch of paths with shape (batch_size, N, d).

    Deterministic path generators produce repeated paths. Stochastic generators
    Brownian paths are independently generated in one vectorized call.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if kind.lower() == "brownian":
        return make_path_brownian_batch(d, N, batch_size)

    return np.stack([make_path(d, N, kind) for _ in range(batch_size)], axis=0)
