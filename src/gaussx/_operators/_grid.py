"""Regular grids and cubic interpolation weights for SKI / KISS-GP.

Inputs live in ℝ^{D_x}. The grid is the inducing set ℐ: one axis
g_d ∈ ℝ^{N_d} of evenly spaced points per input dimension d = 1, …, D_x,
with spacing h_d, and the inducing inputs X_ℐ ∈ ℝ^{N_ℐ × D_x} are their
Cartesian product, N_ℐ = ∏_d N_d, in row-major order. The interpolation
weight matrix W_interp ∈ ℝ^{N × N_ℐ} maps grid values to N target inputs
X ∈ ℝ^{N × D_x}; cubic convolution gives each row 4^{D_x} nonzero weights,
the products of four per-axis weights w_k(δ) of the fractional offset
δ ∈ [0, 1) of the target from its grid cell.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from gaussx._einx import rearrange


def create_grid(
    grid_sizes: list[int] | tuple[int, ...],
    grid_bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> list[Float[Array, " n"]]:
    """Regular grid axes g_d = linspace(lo_d, hi_d, N_d), one per dimension.

        g_d,i = lo_d + i (hi_d − lo_d) / (N_d − 1),   i = 0, …, N_d − 1

    Args:
        grid_sizes: Number of points N_d per dimension, D_x entries.
        grid_bounds: Bounds (lo_d, hi_d) per dimension, D_x entries.

    Returns:
        The D_x axes g_d, each a 1-D array of N_d points.

    Raises:
        ValueError: If ``grid_sizes`` and ``grid_bounds`` differ in length.

    Examples:
        A 5 × 3 grid on [0, 1] × [−1, 1]:

        >>> import gaussx
        >>> grid = gaussx.create_grid([5, 3], [(0.0, 1.0), (-1.0, 1.0)])
        >>> [g.shape for g in grid]
        [(5,), (3,)]
        >>> [round(float(v), 2) for v in grid[0]]
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    return [
        # g_d = linspace(lo_d, hi_d, N_d)
        jnp.linspace(lo, hi, n)
        for n, (lo, hi) in zip(grid_sizes, grid_bounds, strict=True)
    ]


def grid_data(grid: list[Float[Array, " n"]]) -> Float[Array, "G D"]:
    """Inducing inputs X_ℐ, the Cartesian product of the grid axes.

        X_ℐ = g_1 × … × g_D,   N_ℐ = ∏_d N_d rows, row-major order

    Args:
        grid: The D_x axes g_d, each a 1-D array of N_d points.

    Returns:
        X_ℐ, of shape (N_ℐ, D_x); the last axis varies fastest down the rows.

    Examples:
        A 2 × 3 grid has six points, the second axis varying fastest:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> grid = [jnp.array([0.0, 1.0]), jnp.array([0.0, 0.5, 1.0])]
        >>> X = gaussx.grid_data(grid)
        >>> X.shape
        (6, 2)
        >>> X[:3].tolist()
        [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]
    """
    # D_x coordinate arrays of the grid's shape (N_1, …, N_D)
    meshes = jnp.meshgrid(*grid, indexing="ij")
    # jnp, not einx: stacks a run-time number of arrays (a constructor)
    stacked = jnp.stack(meshes, axis=0)
    # X_ℐ, one grid point per row
    return rearrange(stacked, "D ... -> (...) D")


def _cubic_weights_1d(t: Float[Array, " B"]) -> Float[Array, "B four"]:
    """Cubic convolution interpolation weights (Keys, 1981).

        w_−1(δ) = −½δ³ + δ² − ½δ
        w_0(δ)  = 3/2 δ³ − 5/2 δ² + 1
        w_1(δ)  = −3/2 δ³ + 2δ² + ½δ
        w_2(δ)  = ½δ³ − ½δ²

    Args:
        t: Fractional offsets δ ∈ [0, 1] of the targets from their cells.

    Returns:
        Weights (w_−1, w_0, w_1, w_2) of the four neighbouring grid points,
        one row per target.

    References:
        Keys (1981), "Cubic convolution interpolation for digital image
        processing".
    """
    # δ²
    t2 = t * t
    # δ³
    t3 = t2 * t
    # w_−1 = −½δ³ + δ² − ½δ
    w0 = -0.5 * t3 + t2 - 0.5 * t
    # w_0 = 3/2 δ³ − 5/2 δ² + 1
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    # w_1 = −3/2 δ³ + 2δ² + ½δ
    w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    # w_2 = ½δ³ − ½δ²
    w3 = 0.5 * t3 - 0.5 * t2
    # jnp, not einx: the target axis is empty for N = 0 targets
    return jnp.stack([w0, w1, w2, w3], axis=-1)


def cubic_interpolation_weights(
    x_target: Float[Array, "B D"],
    grid: list[Float[Array, " n"]],
) -> tuple[Int[Array, "B K"], Float[Array, "B K"]]:
    """Sparse rows of the cubic SKI interpolation matrix W_interp.

        [W_interp]_n,j = ∏_d w_k_d(δ_n,d),   j = ∑_d (i_n,d + k_d) s_d

    Each target x_n takes the 4^{D_x} grid points around it: on axis d the
    neighbours i_n,d + k_d, k_d ∈ {−1, 0, 1, 2}, of its cell i_n,d, weighted
    by the Keys cubic weights of its fractional offset δ_n,d.

    Algorithm:
        1. Compute the row-major strides s_d = ∏_{d′>d} N_d′ of the grid.
        2. For each axis d, find the continuous index
           ξ = (x_n,d − g_d,0) / h_d, the cell i = clip(⌊ξ⌋, 1, N_d − 3)
           and the offset δ = ξ − i, then the four neighbour indices and
           their weights w_k(δ).
        3. Combine the axes by an outer product over the neighbours: flat
           indices add, weights multiply.

    Args:
        x_target: Target inputs X, one x_n ∈ ℝ^{D_x} per row.
        grid: The D_x axes g_d, each evenly spaced with N_d ≥ 4 points.

    Returns:
        Tuple ``(indices, weights)``, both with 4^{D_x} entries per target:

        - ``indices``: Flat row-major indices j into the N_ℐ grid points.
        - ``weights``: The weights [W_interp]_n,j, summing to ≈ 1 per row.

    Raises:
        ValueError: If any grid axis has fewer than 4 points.

    Examples:
        Four 1-D neighbours per target, with weights summing to one:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> grid = gaussx.create_grid([10], [(0.0, 1.0)])
        >>> x = jnp.array([[0.31], [0.75]])
        >>> indices, weights = gaussx.cubic_interpolation_weights(x, grid)
        >>> indices.shape, weights.shape
        ((2, 4), (2, 4))
        >>> bool(jnp.allclose(weights.sum(-1), 1.0, atol=1e-5))
        True

        In 2-D each target has 4² = 16 neighbours:

        >>> grid2 = gaussx.create_grid([6, 5], [(0.0, 1.0), (0.0, 1.0)])
        >>> x2 = jnp.array([[0.5, 0.5]])
        >>> gaussx.cubic_interpolation_weights(x2, grid2)[0].shape
        (1, 16)
    """
    D = len(grid)
    # N_d
    grid_sizes = [g.shape[0] for g in grid]

    for d, n in enumerate(grid_sizes):
        if n < 4:
            msg = (
                f"Cubic interpolation requires at least 4 grid points per "
                f"dimension, but dimension {d} has {n}."
            )
            raise ValueError(msg)

    # s_d = ∏_{d′>d} N_d′, the row-major strides for manual flat indexing
    strides = []
    s = 1
    for d in range(D - 1, -1, -1):
        strides.append(s)
        s *= grid_sizes[d]
    strides = strides[::-1]

    # Per axis: the four neighbour indices and their weights
    dim_indices = []
    dim_weights = []

    for d in range(D):
        g = grid[d]
        n = g.shape[0]
        # x_n,d
        x_d = x_target[:, d]

        # h_d = (g_d,N_d−1 − g_d,0) / (N_d − 1)
        h = (g[-1] - g[0]) / (n - 1)
        # ξ = (x_n,d − g_d,0) / h_d
        cont_idx = (x_d - g[0]) / h

        # i = clip(⌊ξ⌋, 1, N_d − 3), so all four neighbours exist
        cell = jnp.floor(cont_idx).astype(jnp.int32)
        cell = jnp.clip(cell, 1, n - 3)

        # δ = ξ − i
        t = cont_idx - cell.astype(x_d.dtype)

        # i + k, k ∈ {−1, 0, 1, 2}
        # jnp, not einx: the target axis is empty for N = 0 targets
        idx = jnp.stack([cell - 1, cell, cell + 1, cell + 2], axis=-1)
        idx = jnp.clip(idx, 0, n - 1)

        dim_indices.append(idx)
        # w_k(δ)
        dim_weights.append(_cubic_weights_1d(t))

    # j = (i_1 + k_1) s_1 for the first axis
    flat_indices = dim_indices[0] * strides[0]
    # ∏_d w_k_d, starting from the first axis
    weights = dim_weights[0]

    # Reached only for D_x ≥ 2, where N = 0 targets were never supported
    # (they raise an einx AxisSizeError); 1-D grids skip this loop.
    for d in range(1, D):
        # (i_d + k_d) s_d
        axis_offsets = dim_indices[d] * strides[d]
        # j + (i_d + k_d) s_d over every pair of neighbours
        flat_indices = einx.add("B K, B four -> B (K four)", flat_indices, axis_offsets)
        # w w_k_d(δ_d) over every pair of neighbours
        weights = einx.multiply("B K, B four -> B (K four)", weights, dim_weights[d])

    return flat_indices, weights
