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


def make_path_fbm(d: int, N: int, hurst: float = 0.33) -> np.ndarray:
    """
    Generate a fractional Brownian motion: [X_1, X_2, ...]

    Args:
        d: Dimension of the path
        N: Number of points

    Returns:
        Array of shape (N, d)
    """
    from fbm import fbm

    path = np.stack(
        [fbm(n=N - 1, hurst=hurst, length=1, method="daviesharte") for _ in range(d)],
        axis=-1,
    )
    return np.asarray(path, dtype=np.float32)


def make_path(d: int, N: int, kind: Literal["linear", "sin", "fbm"]) -> np.ndarray:
    """
    Generate path of specified kind.

    Args:
        d: Dimension of the path
        N: Number of points
        kind: "linear", "sin", or "fbm"

    Returns:
        Array of shape (N, d)
    """
    kind = kind.lower()

    if kind == "linear":
        return make_path_linear(d, N)
    elif kind == "sin":
        return make_path_sin(d, N)
    elif kind == "fbm":
        return make_path_fbm(d, N)
    else:
        raise ValueError(f"Unknown path_kind: {kind}")


def make_path_batch(
    d: int,
    N: int,
    kind: Literal["linear", "sin", "fbm"],
    batch_size: int,
) -> np.ndarray:
    """
    Generate a batch of paths with shape (batch_size, N, d).

    Deterministic path generators produce repeated paths. Stochastic generators
    such as fBM are called once per batch element.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    return np.stack([make_path(d, N, kind) for _ in range(batch_size)], axis=0)
