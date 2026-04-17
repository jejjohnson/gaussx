"""Grid construction and cubic interpolation weights for SKI/KISS-GP."""

from __future__ import annotations

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, Int


def create_grid(
    grid_sizes: list[int] | tuple[int, ...],
    grid_bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> list[Float[Array, " n"]]:
    """Create a regular grid from per-dimension sizes and bounds.

    Args:
        grid_sizes: Number of points per dimension, length D.
        grid_bounds: (lo, hi) bounds per dimension, length D.

    Returns:
        List of 1-D arrays, one per dimension.
    """
    return [
        jnp.linspace(lo, hi, n)
        for n, (lo, hi) in zip(grid_sizes, grid_bounds, strict=True)
    ]


def grid_data(grid: list[Float[Array, " n"]]) -> Float[Array, "G D"]:
    """Expand a grid to the full Cartesian product of grid points.

    Args:
        grid: List of 1-D arrays, one per dimension.

    Returns:
        Shape ``(prod(sizes), D)`` array of all grid points.
    """
    meshes = jnp.meshgrid(*grid, indexing="ij")  # D arrays of shape sizes
    stacked = jnp.stack(meshes, axis=0)  # (D, *sizes)
    return rearrange(stacked, "D ... -> (...) D")


def _cubic_weights_1d(t: Float[Array, " B"]) -> Float[Array, "B four"]:
    """Cubic convolution interpolation weights (Keys, 1981).

    For fractional position t ∈ [0, 1], the four weights are::

        w₋₁ = −½t³ + t² − ½t
        w₀  = 3/2 t³ − 5/2 t² + 1
        w₁  = −3/2 t³ + 2t² + ½t
        w₂  = ½t³ − ½t²

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
    return jnp.stack([w0, w1, w2, w3], axis=-1)  # (B, 4)


def cubic_interpolation_weights(
    x_target: Float[Array, "B D"],
    grid: list[Float[Array, " n"]],
) -> tuple[Int[Array, "B K"], Float[Array, "B K"]]:
    """Compute cubic interpolation indices and weights for SKI.

    For each target point, finds the 4ᴰ nearest grid neighbors and
    computes cubic convolution interpolation weights.

    Args:
        x_target: Target points, shape ``(B, D)``.
        grid: List of 1-D arrays, one per dimension (length D).
            Each dimension must have ≥ 4 points.

    Returns:
        Tuple ``(indices, weights)`` both of shape ``(B, 4ᴰ)``:

        - ``indices``: Flat indices into the grid (product of sizes).
        - ``weights``: Interpolation weights summing to ≈ 1.

    Raises:
        ValueError: If any grid dimension has fewer than 4 points.
    """
    D = len(grid)
    grid_sizes = [g.shape[0] for g in grid]

    for d, n in enumerate(grid_sizes):
        if n < 4:
            msg = (
                f"Cubic interpolation requires at least 4 grid points per "
                f"dimension, but dimension {d} has {n}."
            )
            raise ValueError(msg)

    # Strides for manual flat indexing: strides[d] = ∏ sizes[d+1:]
    strides = []
    s = 1
    for d in range(D - 1, -1, -1):
        strides.append(s)
        s *= grid_sizes[d]
    strides = strides[::-1]

    # Per-dimension: 4 indices and weights
    dim_indices = []  # each (B, 4)
    dim_weights = []  # each (B, 4)

    for d in range(D):
        g = grid[d]
        n = g.shape[0]
        x_d = x_target[:, d]  # (B,)

        h = (g[-1] - g[0]) / (n - 1)  # cell spacing
        cont_idx = (x_d - g[0]) / h  # continuous index, (B,)

        # Integer cell index, clamped for 4 neighbors
        cell = jnp.floor(cont_idx).astype(jnp.int32)
        cell = jnp.clip(cell, 1, n - 3)  # (B,)

        t = cont_idx - cell.astype(x_d.dtype)  # fractional part, (B,)

        # 4 neighbor indices: cell−1, cell, cell+1, cell+2
        idx = jnp.stack([cell - 1, cell, cell + 1, cell + 2], axis=-1)  # (B, 4)
        idx = jnp.clip(idx, 0, n - 1)

        dim_indices.append(idx)
        dim_weights.append(_cubic_weights_1d(t))

    # Cross-dimensional outer product of indices/weights
    flat_indices = dim_indices[0] * strides[0]  # (B, 4)
    weights = dim_weights[0]  # (B, 4)

    for d in range(1, D):
        # Outer product: (B, K) × (B, 4) → (B, K·4)
        fi_exp = flat_indices[:, :, None]  # (B, K, 1)
        di_exp = dim_indices[d][:, None, :] * strides[d]  # (B, 1, 4)
        flat_indices = rearrange(fi_exp + di_exp, "B K four -> B (K four)")

        w_exp = weights[:, :, None]  # (B, K, 1)
        dw_exp = dim_weights[d][:, None, :]  # (B, 1, 4)
        weights = rearrange(w_exp * dw_exp, "B K four -> B (K four)")

    return flat_indices, weights
