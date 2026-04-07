"""Structured solve: A x = b with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import (
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
)
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._svd_low_rank_update import SVDLowRankUpdate


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
        return _solve_block_diag(operator, vector, solver)
    if isinstance(operator, Kronecker):
        return _solve_kronecker(operator, vector, solver)
    if isinstance(operator, SVDLowRankUpdate):
        return _solve_svd_low_rank(operator, vector, solver)
    if isinstance(operator, LowRankUpdate):
        return _solve_low_rank(operator, vector, solver)
    if isinstance(operator, KroneckerSum):
        return _solve_kronecker_sum(operator, vector)
    if isinstance(operator, BlockTriDiag):
        return _solve_block_tridiag(operator, vector)
    if isinstance(operator, LowerBlockTriDiag):
        return _solve_lower_block_tridiag(operator, vector)
    if isinstance(operator, UpperBlockTriDiag):
        return _solve_upper_block_tridiag(operator, vector)
    return _solve_fallback(operator, vector, solver)


def _solve_diagonal(
    operator: lx.DiagonalLinearOperator, vector: jnp.ndarray
) -> jnp.ndarray:
    diag = lx.diagonal(operator)
    return vector / diag


def _solve_block_diag(
    operator: BlockDiag,
    vector: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None,
) -> jnp.ndarray:
    results = []
    offset = 0
    for op in operator.operators:
        size = op.in_size()
        block = jax.lax.dynamic_slice(vector, (offset,), (size,))
        results.append(solve(op, block, solver=solver))
        offset += size
    return jnp.concatenate(results)


def _solve_kronecker(
    operator: Kronecker,
    vector: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None,
) -> jnp.ndarray:
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
        solve_factor = ft.partial(solve, op, solver=solver)
        x = jax.vmap(solve_factor)(x)
        x = rearrange(x, "r c -> (c r)")
    return x


def _solve_low_rank(
    operator: LowRankUpdate,
    vector: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None,
) -> jnp.ndarray:
    """Woodbury identity: (L + U D V^T)^{-1} b.

    (L + U D V^T)^{-1} = L^{-1} - L^{-1} U C^{-1} V^T L^{-1}
    where C = D^{-1} + V^T L^{-1} U  (k x k capacitance matrix).
    """
    U, d, V = operator.U, operator.d, operator.V

    # Step 1: L^{-1} b
    Linv_b = solve(operator.base, vector, solver=solver)

    # Step 2: L^{-1} U  (n x k)
    Linv_U = jnp.stack(
        [solve(operator.base, U[:, j], solver=solver) for j in range(U.shape[1])],
        axis=1,
    )

    # Step 3: Capacitance matrix C = D^{-1} + V^T L^{-1} U  (k x k)
    C = jnp.diag(1.0 / d) + V.T @ Linv_U

    # Step 4: C^{-1} V^T L^{-1} b  (k,)
    Cinv_VtLinvb = jnp.linalg.solve(C, V.T @ Linv_b)

    # Step 5: L^{-1} U C^{-1} V^T L^{-1} b  (n,)
    correction = Linv_U @ Cinv_VtLinvb

    return Linv_b - correction


def _solve_svd_low_rank(
    operator: SVDLowRankUpdate,
    vector: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None,
) -> jnp.ndarray:
    """Woodbury identity for SVDLowRankUpdate: (L + U S V^T)^{-1} b.

    Same as _solve_low_rank but uses S (singular values) instead of d.
    """
    U, S, V = operator.U, operator.S, operator.V

    # Step 1: L^{-1} b
    Linv_b = solve(operator.base, vector, solver=solver)

    # Step 2: L^{-1} U  (n x k)
    Linv_U = jnp.stack(
        [solve(operator.base, U[:, j], solver=solver) for j in range(U.shape[1])],
        axis=1,
    )

    # Step 3: Capacitance matrix C = S^{-1} + V^T L^{-1} U  (k x k)
    C = jnp.diag(1.0 / S) + V.T @ Linv_U

    # Step 4: C^{-1} V^T L^{-1} b  (k,)
    Cinv_VtLinvb = jnp.linalg.solve(C, V.T @ Linv_b)

    # Step 5: L^{-1} U C^{-1} V^T L^{-1} b  (n,)
    correction = Linv_U @ Cinv_VtLinvb

    return Linv_b - correction


def _solve_kronecker_sum(
    operator: KroneckerSum,
    vector: jnp.ndarray,
) -> jnp.ndarray:
    """Solve (A (+) B) x = b via eigendecomposition.

    (A (+) B) = (Q_A (x) Q_B) diag(lambda_A_i + lambda_B_j) (Q_A (x) Q_B)^T.
    """
    from einops import rearrange

    evals_a, Q_a = jnp.linalg.eigh(operator.A.as_matrix())
    evals_b, Q_b = jnp.linalg.eigh(operator.B.as_matrix())
    n_a, n_b = operator._n_a, operator._n_b
    # Rotate into eigenbasis: c = (Q_A^T (x) Q_B^T) b
    X = rearrange(vector, "(a b) -> b a", a=n_a, b=n_b)
    C = Q_b.T @ X @ Q_a  # (n_b, n_a)
    # Divide by eigenvalues
    eig_mat = evals_a[None, :] + evals_b[:, None]  # (n_b, n_a)
    C = C / eig_mat
    # Rotate back: x = (Q_A (x) Q_B) c
    result = Q_b @ C @ Q_a.T
    return rearrange(result, "b a -> (a b)")


def _solve_block_tridiag(
    operator: BlockTriDiag,
    vector: jnp.ndarray,
) -> jnp.ndarray:
    """Solve via block-banded Cholesky then forward/backward substitution."""
    from gaussx._primitives._cholesky import cholesky

    L = cholesky(operator)
    # Forward solve: L y = b
    y = solve(L, vector)
    # Backward solve: L^T x = y
    return solve(L.T, y)


def _solve_lower_block_tridiag(
    operator: LowerBlockTriDiag,
    vector: jnp.ndarray,
) -> jnp.ndarray:
    """Forward substitution for lower block-bidiagonal system."""
    N = operator._num_blocks
    d = operator._block_size
    from einops import rearrange

    b = rearrange(vector, "(N d) -> N d", N=N, d=d)

    def body_fn(carry, k):
        x_prev = carry
        rhs = b[k] - jax.lax.cond(
            k > 0,
            lambda: operator.sub_diagonal[k - 1] @ x_prev,
            lambda: jnp.zeros(d, dtype=b.dtype),
        )
        x_k = jax.scipy.linalg.solve_triangular(operator.diagonal[k], rhs, lower=True)
        return x_k, x_k

    _, x_all = jax.lax.scan(body_fn, jnp.zeros(d, dtype=b.dtype), jnp.arange(N))
    return rearrange(x_all, "N d -> (N d)")


def _solve_upper_block_tridiag(
    operator: UpperBlockTriDiag,
    vector: jnp.ndarray,
) -> jnp.ndarray:
    """Backward substitution for upper block-bidiagonal system."""
    N = operator._num_blocks
    d = operator._block_size
    from einops import rearrange

    b = rearrange(vector, "(N d) -> N d", N=N, d=d)

    def body_fn(carry, k):
        x_next = carry
        idx = N - 1 - k
        rhs = b[idx] - jax.lax.cond(
            idx < N - 1,
            lambda: operator.super_diagonal[idx] @ x_next,
            lambda: jnp.zeros(d, dtype=b.dtype),
        )
        x_k = jax.scipy.linalg.solve_triangular(
            operator.diagonal[idx], rhs, lower=False
        )
        return x_k, x_k

    _, x_rev = jax.lax.scan(body_fn, jnp.zeros(d, dtype=b.dtype), jnp.arange(N))
    return rearrange(jnp.flip(x_rev, axis=0), "N d -> (N d)")


def _solve_fallback(
    operator: lx.AbstractLinearOperator,
    vector: jnp.ndarray,
    solver: lx.AbstractLinearSolver | None,
) -> jnp.ndarray:
    if solver is None:
        solver = lx.AutoLinearSolver(well_posed=True)
    return lx.linear_solve(operator, vector, solver).value
