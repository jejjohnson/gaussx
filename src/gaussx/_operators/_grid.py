"""Regular grids and the interpolation operator for SKI / KISS-GP.

Inputs live in ℝ^{D_x}. The grid is the inducing set ℐ: one axis
g_d ∈ ℝ^{N_d} of evenly spaced points per input dimension d = 1, …, D_x,
with spacing h_d, and the inducing inputs X_ℐ ∈ ℝ^{N_ℐ × D_x} are their
Cartesian product, N_ℐ = ∏_d N_d, in row-major order. The interpolation
weight matrix W_interp ∈ ℝ^{N × N_ℐ} maps grid values to N target inputs
X ∈ ℝ^{N × D_x}, one sparse row of ``(indices, weights)`` per target.

The stencils themselves come from functax (gh-813), whose ``Plan.to_gaussx()``
returns ``(indices, weights)`` in the layout `InterpolatedOperator` and
`interpolation_operator` take.
"""

from __future__ import annotations

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


def interpolation_operator(
    indices: Int[Array, "B K"],
    weights: Float[Array, "B K"],
    grid_size: int,
) -> lx.AbstractLinearOperator:
    """The interpolation matrix W as a matrix-free operator, B × G.

        (W f)_n = ∑_k w_n,k f_{j_n,k}

    Each matvec is a gather at O(B K), and ``W.T`` is the matching scatter-add
    (`jax.linear_transpose` of the gather). Takes any ``(indices, weights)``
    stencil in that row layout, such as functax's ``Plan.to_gaussx()``, and the
    indices may be traced. `gaussx.InterpolatedOperator` is the W K Wᵀ form.

    Args:
        indices: Flat grid indices j, one row per target.
        weights: The weights w, one row per target.
        grid_size: The number G of grid points.

    Returns:
        W, acting on the flattened grid values.

    Examples:
        Sample a field on the axis (0, 1, 3) at x = 1.5, whose linear stencil is
        0.75 on node 1 and 0.25 on node 2, then apply the transpose:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> indices, weights = jnp.array([[1, 2]]), jnp.array([[0.75, 0.25]])
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
