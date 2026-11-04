"""Regular grids and cubic interpolation weights for SKI / KISS-GP.

Inputs live in ℝ^{D_x}. The grid is the inducing set ℐ: one axis
g_d ∈ ℝ^{N_d} of evenly spaced points per input dimension d = 1, …, D_x,
with spacing h_d, and the inducing inputs X_ℐ ∈ ℝ^{N_ℐ × D_x} are their
Cartesian product, N_ℐ = ∏_d N_d, in row-major order. The interpolation
weight matrix W_interp ∈ ℝ^{N × N_ℐ} maps grid values to N target inputs
X ∈ ℝ^{N × D_x}; cubic convolution gives each row 4^{D_x} nonzero weights,
the products of four per-axis weights w_k(δ) of the fractional offset
δ ∈ [0, 1) of the target from its grid cell. Multilinear interpolation gives
2^{D_x} nonzero weights on axes that may be non-uniform.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._einx import einsum, rearrange


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

    return _combine_axes(dim_indices, dim_weights, _row_major_strides(grid_sizes))


def linear_interpolation_weights(
    x_target: Float[Array, "B D"],
    grid: list[Float[Array, " n"]],
    *,
    clamp: bool = True,
) -> tuple[Int[Array, "B K"], Float[Array, "B K"]]:
    """Sparse rows of the multilinear interpolation matrix on a rectilinear grid.

        i_d = clip(searchsorted(g_d, p_d) − 1, 0, N_d − 2)
        t_d = (p_d − g_d,i_d) / (g_d,i_d+1 − g_d,i_d)
        [W]_n,j = ∏_d t_d^{c_d} (1 − t_d)^{1 − c_d},   j = ∑_d (i_d + c_d) s_d

    Each target p takes the K = 2^D corners c ∈ {0, 1}^D of its grid cell, in
    the index and weight format of `cubic_interpolation_weights`. The axes may
    be non-uniform. Weights sum to one per target, and grid nodes are
    reproduced exactly. The result is piecewise linear in p, as in the KISS-GP
    setting of Wilson & Nickisch (2015), arXiv:1503.01057.

    Algorithm:
        1. Compute the row-major strides s_d = ∏_{d′>d} N_d′ of the grid.
        2. For each axis, bracket p_d by its left node i_d and offset t_d,
           clipped to [0, 1] when ``clamp`` is set.
        3. Combine the axes over the corners: flat indices add, weights
           multiply.

    Args:
        x_target: Target points p, one per row.
        grid: The D axes g_d, each strictly increasing with N_d ≥ 2 points.
        clamp: When ``True``, a target outside the grid takes the edge value;
            when ``False``, it extrapolates linearly from the edge cell.

    Returns:
        Tuple ``(indices, weights)``, both with 2^D entries per target:

        - ``indices``: Flat row-major indices j into the ∏_d N_d grid points.
        - ``weights``: The weights [W]_n,j.

    Raises:
        ValueError: If any grid axis has fewer than 2 points.

    Examples:
        A point a quarter of the way along a non-uniform cell:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> axis = jnp.array([0.0, 1.0, 3.0])
        >>> indices, weights = gaussx.linear_interpolation_weights(
        ...     jnp.array([[1.5]]), [axis]
        ... )
        >>> indices.tolist(), [round(float(v), 4) for v in weights[0]]
        ([[1, 2]], [0.75, 0.25])

        In 2-D each target has 2² = 4 corners:

        >>> gaussx.linear_interpolation_weights(
        ...     jnp.array([[0.5, 0.5]]), [axis, axis]
        ... )[0].shape
        (1, 4)
    """
    # N_d
    grid_sizes = [axis.shape[0] for axis in grid]
    for d, n in enumerate(grid_sizes):
        if n < 2:
            msg = (
                f"Linear interpolation requires at least 2 grid points per "
                f"dimension, but dimension {d} has {n}."
            )
            raise ValueError(msg)

    dim_indices = []
    dim_weights = []
    for d, axis in enumerate(grid):
        # p_d
        points = x_target[:, d]
        # i_d, the left node of the cell around p_d
        left = jnp.clip(jnp.searchsorted(axis, points) - 1, 0, grid_sizes[d] - 2)
        # t_d, the offset of p_d in its cell
        offset = (points - axis[left]) / (axis[left + 1] - axis[left])
        if clamp:
            offset = jnp.clip(offset, 0.0, 1.0)
        # i_d, i_d + 1, one row per target
        dim_indices.append(jnp.column_stack([left, left + 1]))
        # 1 − t_d, t_d, one row per target
        dim_weights.append(jnp.column_stack([1.0 - offset, offset]))

    return _combine_axes(dim_indices, dim_weights, _row_major_strides(grid_sizes))


def interpolation_operator(
    indices: Int[Array, "B K"],
    weights: Float[Array, "B K"],
    grid_size: int,
) -> lx.AbstractLinearOperator:
    """The interpolation matrix W as a matrix-free operator, B × G.

        (W f)_n = ∑_k w_n,k f_{j_n,k}

    Each matvec is a gather at O(B K), and ``W.T`` is the matching scatter-add
    (`jax.linear_transpose` of the gather). Takes the output of
    `linear_interpolation_weights` or `cubic_interpolation_weights`, and the
    indices may be traced. `gaussx.InterpolatedOperator` is the W K Wᵀ form.

    Args:
        indices: Flat grid indices j, one row per target.
        weights: The weights w, one row per target.
        grid_size: The number G of grid points.

    Returns:
        W, acting on the flattened grid values.

    Examples:
        Sample a field on a 3-point axis at x = 1.5, then apply the transpose:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> axis = jnp.array([0.0, 1.0, 3.0])
        >>> indices, weights = gaussx.linear_interpolation_weights(
        ...     jnp.array([[1.5]]), [axis]
        ... )
        >>> W = gaussx.interpolation_operator(indices, weights, grid_size=3)
        >>> [round(float(v), 4) for v in W.mv(jnp.array([0.0, 4.0, 8.0]))]
        [5.0]
        >>> [round(float(v), 4) for v in W.T.mv(jnp.array([1.0]))]
        [0.0, 0.75, 0.25]
    """

    def gather(field: Float[Array, " G"]) -> Float[Array, " B"]:
        # (W f)_n = ∑_k w_n,k f_{j_n,k}
        return einsum(weights, field[indices], "b k, b k -> b")

    return lx.FunctionLinearOperator(
        gather, jax.ShapeDtypeStruct((grid_size,), weights.dtype)
    )


def _row_major_strides(grid_sizes: list[int]) -> list[int]:
    """Row-major strides s_d = ∏_{d′>d} N_d′ of a grid, for flat indexing.

    Args:
        grid_sizes: The axis sizes N_d.

    Returns:
        The strides s_d, one per axis.
    """
    strides = []
    stride = 1
    for size in reversed(grid_sizes):
        strides.append(stride)
        stride *= size
    return strides[::-1]


def _combine_axes(
    dim_indices: list[Int[Array, "B P"]],
    dim_weights: list[Float[Array, "B P"]],
    strides: list[int],
) -> tuple[Int[Array, "B K"], Float[Array, "B K"]]:
    """Combine per-axis neighbours: flat indices add and weights multiply.

    Args:
        dim_indices: Per axis, the P neighbour indices of each target.
        dim_weights: Per axis, the P neighbour weights of each target.
        strides: The row-major strides s_d.

    Returns:
        Tuple ``(indices, weights)``, with P^D entries per target.
    """
    # j = (i_1 + k_1) s_1 for the first axis
    flat_indices = dim_indices[0] * strides[0]
    # ∏_d w_k_d, starting from the first axis
    weights = dim_weights[0]
    # Reached only for D_x ≥ 2, where N = 0 targets were never supported
    # (they raise an einx AxisSizeError); 1-D grids skip this loop.
    for axis_indices, axis_weights, stride in zip(
        dim_indices[1:], dim_weights[1:], strides[1:], strict=True
    ):
        # (i_d + k_d) s_d
        axis_offsets = axis_indices * stride
        # j + (i_d + k_d) s_d over every pair of neighbours
        flat_indices = einx.add("B K, B P -> B (K P)", flat_indices, axis_offsets)
        # w w_k_d(δ_d) over every pair of neighbours
        weights = einx.multiply("B K, B P -> B (K P)", weights, axis_weights)
    return flat_indices, weights
