"""Projection: K_XZ @ K_ZZ^{-1} via Cholesky solve."""

from __future__ import annotations

import jax
import lineax as lx
from jaxtyping import Array, Float


def project(
    K_XZ: Float[Array, "B M"],
    L_Z: lx.AbstractLinearOperator,
) -> Float[Array, "B M"]:
    """Compute A_X = K_XZ @ K_ZZ^{-1} via Cholesky solve.

    Solves ``L_Z @ L_Z^T @ A_X^T = K_XZ^T`` using forward/backward
    substitution.  Used in sparse variational GPs to project test
    points onto the inducing space.

    Args:
        K_XZ: Cross-covariance matrix, shape ``(B, M)``.
        L_Z: Lower-triangular Cholesky factor of K_ZZ, shape ``(M, M)``.
            It need not carry ``lineax.lower_triangular_tag``: the tag is
            added when missing (the factor is lower-triangular by
            precondition), and ``L_Z.T`` is then upper-triangular.

    Returns:
        Projection matrix A_X, shape ``(B, M)``.

    Raises:
        ValueError: If ``L_Z`` is not ``(M, M)`` with ``M = K_XZ.shape[1]``.
    """
    M = K_XZ.shape[-1]
    if L_Z.in_size() != M or L_Z.out_size() != M:
        raise ValueError(
            f"L_Z must be ({M}, {M}) to match K_XZ's {M} columns, got "
            f"({L_Z.out_size()}, {L_Z.in_size()})."
        )
    if not lx.is_lower_triangular(L_Z):
        # lx.Triangular checks tags, so an untagged factor (e.g. built from
        # jnp.linalg.cholesky) would be rejected.
        L_Z = lx.TaggedLinearOperator(L_Z, lx.lower_triangular_tag)
    # Solve L_Z @ Y = K_XZ^T, then L_Z^T @ A_X^T = Y
    # Equivalently, solve (L_Z @ L_Z^T) @ A_X^T = K_XZ^T per column
    solver = lx.Triangular()

    def _solve_col(kxz_row):
        # Solve L_Z y = kxz_row
        y = lx.linear_solve(L_Z, kxz_row, solver).value
        # Solve L_Z^T a = y
        return lx.linear_solve(L_Z.T, y, solver).value

    return jax.vmap(_solve_col)(K_XZ)
