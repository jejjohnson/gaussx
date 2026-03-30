"""Structured solve: A x = b with dispatch on operator type."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._low_rank_update import LowRankUpdate


def solve(
    operator: lx.AbstractLinearOperator,
    vector: jnp.ndarray,
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> jnp.ndarray:
    """Solve ``A x = b`` with structural dispatch.

    Args:
        operator: The linear operator A.
        vector: The right-hand side b.
        solver: Optional lineax solver override for the fallback path.

    Returns:
        The solution x.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _solve_diagonal(operator, vector)
    if isinstance(operator, BlockDiag):
        return _solve_block_diag(operator, vector)
    if isinstance(operator, Kronecker):
        return _solve_kronecker(operator, vector)
    if isinstance(operator, LowRankUpdate):
        return _solve_low_rank(operator, vector)
    return _solve_fallback(operator, vector, solver)


def _solve_diagonal(
    operator: lx.DiagonalLinearOperator, vector: jnp.ndarray
) -> jnp.ndarray:
    diag = lx.diagonal(operator)
    return vector / diag


def _solve_block_diag(operator: BlockDiag, vector: jnp.ndarray) -> jnp.ndarray:
    results = []
    offset = 0
    for op in operator.operators:
        size = op.in_size()
        block = jax.lax.dynamic_slice(vector, (offset,), (size,))
        results.append(solve(op, block))
        offset += size
    return jnp.concatenate(results)


def _solve_kronecker(operator: Kronecker, vector: jnp.ndarray) -> jnp.ndarray:
    """Solve (A1 kron A2 kron ... kron Ak) x = b.

    Uses the same reshape trick as Kronecker.mv but with
    per-factor solves: solve(A_i^T, ...) instead of mat @ ....
    """
    from einops import rearrange

    x = vector
    for i in range(len(operator.operators) - 1, -1, -1):
        op = operator.operators[i]
        n_in = op.in_size()
        x = rearrange(x, "(r c) -> r c", c=n_in)
        # In the mv path we do: x = x @ mat.T
        # For solve, we replace mat.T with inv(mat).T = inv(mat^T)
        # So: x = x @ inv(mat).T, i.e. solve mat^T @ y.T = x.T
        mat = op.as_matrix()
        x = jnp.linalg.solve(mat, x.T).T
        x = rearrange(x, "r c -> (c r)")
    return x


def _solve_low_rank(operator: LowRankUpdate, vector: jnp.ndarray) -> jnp.ndarray:
    """Woodbury identity: (L + U D V^T)^{-1} b.

    (L + U D V^T)^{-1} = L^{-1} - L^{-1} U C^{-1} V^T L^{-1}
    where C = D^{-1} + V^T L^{-1} U  (k x k capacitance matrix).
    """
    U, d, V = operator.U, operator.d, operator.V

    # Step 1: L^{-1} b
    Linv_b = solve(operator.base, vector)

    # Step 2: L^{-1} U  (n x k)
    Linv_U = jnp.stack(
        [solve(operator.base, U[:, j]) for j in range(U.shape[1])], axis=1
    )

    # Step 3: Capacitance matrix C = D^{-1} + V^T L^{-1} U  (k x k)
    C = jnp.diag(1.0 / d) + V.T @ Linv_U

    # Step 4: C^{-1} V^T L^{-1} b  (k,)
    Cinv_VtLinvb = jnp.linalg.solve(C, V.T @ Linv_b)

    # Step 5: L^{-1} U C^{-1} V^T L^{-1} b  (n,)
    correction = Linv_U @ Cinv_VtLinvb

    return Linv_b - correction


def _solve_fallback(
    operator: lx.AbstractLinearOperator,
    vector: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None,
) -> jnp.ndarray:
    if solver is None:
        solver = lx.AutoLinearSolver(well_posed=True)
    return lx.linear_solve(operator, vector, solver).value
