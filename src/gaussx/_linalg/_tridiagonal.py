"""Tridiagonal (Thomas-algorithm) solves."""

from __future__ import annotations

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float


def solve_tridiagonal(
    lower: Float[Array, " n_minus_1"],
    diag: Float[Array, " n"],
    upper: Float[Array, " n_minus_1"],
    rhs: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve a tridiagonal system ``A x = d``.

    Thin wrapper over `lineax.TridiagonalLinearOperator`, which delegates
    to ``jax.lax.linalg.tridiagonal_solve`` (LAPACK / cuSPARSE).

    Args:
        lower: Sub-diagonal of ``A``, length ``n - 1``.
        diag: Main diagonal of ``A``, length ``n``.
        upper: Super-diagonal of ``A``, length ``n - 1``.
        rhs: Right-hand side, length ``n``.

    Returns:
        The solution ``x``, length ``n``.
    """
    operator = lx.TridiagonalLinearOperator(diag, lower, upper)
    return lx.linear_solve(operator, rhs, solver=lx.Tridiagonal()).value


def solve_tridiagonal_batched(
    lower: Float[Array, "*batch n_minus_1"],
    diag: Float[Array, "*batch n"],
    upper: Float[Array, "*batch n_minus_1"],
    rhs: Float[Array, "*batch n"],
) -> Float[Array, "*batch n"]:
    """Solve independent tridiagonal systems over leading batch dimensions.

    Applies `solve_tridiagonal` vmapped over all leading dimensions.

    Args:
        lower: Sub-diagonals, shape ``(*batch, n - 1)``.
        diag: Main diagonals, shape ``(*batch, n)``.
        upper: Super-diagonals, shape ``(*batch, n - 1)``.
        rhs: Right-hand sides, shape ``(*batch, n)``.

    Returns:
        Solutions, shape ``(*batch, n)``.
    """
    n_batch = diag.ndim - 1
    fn = solve_tridiagonal
    for _ in range(n_batch):
        fn = eqx.filter_vmap(fn)
    return fn(lower, diag, upper, rhs)
