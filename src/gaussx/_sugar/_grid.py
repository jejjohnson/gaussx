"""Grid construction and cubic interpolation weights for SKI/KISS-GP."""

from __future__ import annotations

import jax.numpy as jnp


def create_grid(
    grid_sizes: list[int] | tuple[int, ...],
    grid_bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> list[jnp.ndarray]:
    """Create a regular grid from per-dimension sizes and bounds.

    Args:
        grid_sizes: Number of points per dimension, length ``D``.
        grid_bounds: ``(lo, hi)`` bounds per dimension, length ``D``.

    Returns:
        List of 1-D arrays, one per dimension.
    """
    return [
        jnp.linspace(lo, hi, n)
        for n, (lo, hi) in zip(grid_sizes, grid_bounds, strict=True)
    ]


def grid_data(grid: list[jnp.ndarray]) -> jnp.ndarray:
    """Expand a grid to the full Cartesian product of grid points.

    Args:
        grid: List of 1-D arrays, one per dimension.

    Returns:
        Shape ``(prod(sizes), D)`` array of all grid points.
    """
    meshes = jnp.meshgrid(*grid, indexing="ij")
    return jnp.stack([g.ravel() for g in meshes], axis=-1)


def _cubic_weights_1d(t: jnp.ndarray) -> jnp.ndarray:
    """Cubic convolution interpolation weights (Keys, 1981).

    Args:
        t: Fractional position in [0, 1], shape ``(B,)``.

    Returns:
        Weights for the 4 neighboring points, shape ``(B, 4)``.
    """
    t2 = t * t
    t3 = t2 * t
    w0 = -0.5 * t3 + t2 - 0.5 * t
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w3 = 0.5 * t3 - 0.5 * t2
    return jnp.stack([w0, w1, w2, w3], axis=-1)


def cubic_interpolation_weights(
    x_target: jnp.ndarray,
    grid: list[jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute cubic interpolation indices and weights for SKI.

    For each target point, finds the 4^D nearest grid neighbors and
    computes cubic convolution interpolation weights.

    Args:
        x_target: Target points, shape ``(B, D)``.
        grid: List of 1-D arrays, one per dimension (length ``D``).

    Returns:
        Tuple ``(indices, weights)`` both of shape ``(B, 4^D)``:
        - ``indices``: Flat indices into the grid (product of sizes).
        - ``weights``: Interpolation weights summing to ~1.
    """
    D = len(grid)
    B = x_target.shape[0]
    grid_sizes = [g.shape[0] for g in grid]

    for d, n in enumerate(grid_sizes):
        if n < 4:
            msg = (
                f"Cubic interpolation requires at least 4 grid points per "
                f"dimension, but dimension {d} has {n}."
            )
            raise ValueError(msg)

    # Compute strides for manual flat indexing
    strides = []
    s = 1
    for d in range(D - 1, -1, -1):
        strides.append(s)
        s *= grid_sizes[d]
    strides = strides[::-1]  # strides[d] = prod(sizes[d+1:])

    # Per-dimension: 4 indices and weights
    dim_indices = []  # each (B, 4)
    dim_weights = []  # each (B, 4)

    for d in range(D):
        g = grid[d]
        n = g.shape[0]
        x_d = x_target[:, d]

        # Cell spacing
        h = (g[-1] - g[0]) / (n - 1)

        # Continuous index
        cont_idx = (x_d - g[0]) / h

        # Integer index of the cell (clamp to valid range for 4 neighbors)
        cell = jnp.floor(cont_idx).astype(jnp.int32)
        # The 4 neighbors are at cell-1, cell, cell+1, cell+2
        cell = jnp.clip(cell, 1, n - 3)

        t = cont_idx - cell.astype(x_d.dtype)  # fractional part

        # 4 neighbor indices
        idx = jnp.stack([cell - 1, cell, cell + 1, cell + 2], axis=-1)  # (B, 4)
        idx = jnp.clip(idx, 0, n - 1)

        dim_indices.append(idx)
        dim_weights.append(_cubic_weights_1d(t))

    # Cross-dimensional outer product
    # Start with first dimension
    flat_indices = dim_indices[0] * strides[0]  # (B, 4)
    weights = dim_weights[0]  # (B, 4)

    for d in range(1, D):
        # Outer product: expand existing (B, K) x new (B, 4) -> (B, K*4)
        K = flat_indices.shape[1]
        # Expand flat_indices: (B, K, 1) + dim_indices[d] * stride: (B, 1, 4)
        fi_exp = flat_indices[:, :, None]  # (B, K, 1)
        di_exp = dim_indices[d][:, None, :] * strides[d]  # (B, 1, 4)
        flat_indices = (fi_exp + di_exp).reshape(B, K * 4)

        w_exp = weights[:, :, None]  # (B, K, 1)
        dw_exp = dim_weights[d][:, None, :]  # (B, 1, 4)
        weights = (w_exp * dw_exp).reshape(B, K * 4)

    return flat_indices, weights
