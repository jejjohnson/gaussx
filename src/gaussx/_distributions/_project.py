"""Projection: K_XZ @ K_ZZ^{-1} via Cholesky solve."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx


def project(
    K_XZ: jnp.ndarray,
    L_Z: lx.AbstractLinearOperator,
) -> jnp.ndarray:
    """Compute A_X = K_XZ @ K_ZZ^{-1} via Cholesky solve.

    Solves ``L_Z @ L_Z^T @ A_X^T = K_XZ^T`` using forward/backward
    substitution.  Used in sparse variational GPs to project test
    points onto the inducing space.

    Args:
        K_XZ: Cross-covariance matrix, shape ``(B, M)``.
        L_Z: Lower-triangular Cholesky factor of K_ZZ, shape ``(M, M)``.

    Returns:
        Projection matrix A_X, shape ``(B, M)``.
    """
    # Solve L_Z @ Y = K_XZ^T, then L_Z^T @ A_X^T = Y
    # Equivalently, solve (L_Z @ L_Z^T) @ A_X^T = K_XZ^T per column
    solver = lx.Triangular()

    def _solve_col(kxz_row):
        # Solve L_Z y = kxz_row
        y = lx.linear_solve(L_Z, kxz_row, solver).value
        # Solve L_Z^T a = y
        return lx.linear_solve(L_Z.T, y, solver).value

    return jax.vmap(_solve_col)(K_XZ)
