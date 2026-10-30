"""Tridiagonal solves A x = b, single and batched.

A is an n × n tridiagonal matrix given by its three bands: the sub-diagonal
a, the main diagonal d and the super-diagonal c, so A_ii = d_i,
A_i+1,i = a_i and A_i,i+1 = c_i.
"""

from __future__ import annotations

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float


def tridiagonal_solve(
    lower: Float[Array, " n_minus_1"],
    diag: Float[Array, " n"],
    upper: Float[Array, " n_minus_1"],
    rhs: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve a tridiagonal system A x = b from its three bands.

        x = A⁻¹ b,   A = tridiag(a, d, c)

    A thin wrapper over `lineax.TridiagonalLinearOperator` and the
    `lineax.Tridiagonal` solver, which calls
    ``jax.lax.linalg.tridiagonal_solve`` (LAPACK or cuSPARSE).

    Args:
        lower: Sub-diagonal a of A, length n − 1.
        diag: Main diagonal d of A, length n.
        upper: Super-diagonal c of A, length n − 1.
        rhs: Right-hand side b, length n.

    Returns:
        The solution x = A⁻¹ b.

    Examples:
        A diagonally dominant 3 × 3 system:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> lower = jnp.array([1.0, 1.0])
        >>> diag = jnp.array([4.0, 4.0, 4.0])
        >>> upper = jnp.array([1.0, 1.0])
        >>> rhs = jnp.array([1.0, 2.0, 3.0])
        >>> x = gaussx.tridiagonal_solve(lower, diag, upper, rhs)
        >>> dense = jnp.diag(diag) + jnp.diag(lower, -1) + jnp.diag(upper, 1)
        >>> bool(jnp.allclose(dense @ x, rhs, atol=1e-5))
        True
    """
    # A = tridiag(a, d, c)
    operator = lx.TridiagonalLinearOperator(diag, lower, upper)
    # x = A⁻¹ b
    solution = lx.linear_solve(operator, rhs, solver=lx.Tridiagonal())
    return solution.value


def tridiagonal_solve_batched(
    lower: Float[Array, "*batch n_minus_1"],
    diag: Float[Array, "*batch n"],
    upper: Float[Array, "*batch n_minus_1"],
    rhs: Float[Array, "*batch n"],
) -> Float[Array, "*batch n"]:
    """Solve independent tridiagonal systems over leading batch axes.

        x = A⁻¹ b,   A = tridiag(a, d, c),   for each batch element

    Maps `tridiagonal_solve` over every leading axis, one ``vmap`` per axis.

    Args:
        lower: Sub-diagonals a, length n − 1 along the last axis.
        diag: Main diagonals d, length n along the last axis.
        upper: Super-diagonals c, length n − 1 along the last axis.
        rhs: Right-hand sides b, length n along the last axis.

    Returns:
        The solutions x, with the batch axes of ``diag``.

    Examples:
        Two systems at once:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> lower = jnp.ones((2, 2))
        >>> diag = jnp.full((2, 3), 4.0)
        >>> upper = jnp.ones((2, 2))
        >>> rhs = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        >>> x = gaussx.tridiagonal_solve_batched(lower, diag, upper, rhs)
        >>> x.shape
        (2, 3)
        >>> single = gaussx.tridiagonal_solve(lower[1], diag[1], upper[1], rhs[1])
        >>> bool(jnp.allclose(x[1], single))
        True
    """
    num_batch_axes = diag.ndim - 1
    batched_solve = tridiagonal_solve
    for _ in range(num_batch_axes):
        batched_solve = eqx.filter_vmap(batched_solve)
    return batched_solve(lower, diag, upper, rhs)
