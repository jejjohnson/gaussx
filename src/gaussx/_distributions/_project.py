"""Inducing projection A_X = K_Xℐ K_ℐℐ⁻¹ through the Cholesky factor of K_ℐℐ.

K_Xℐ is the kernel matrix between N inputs X and the N_ℐ inducing inputs
X_ℐ, K_ℐℐ = L_ℐℐ L_ℐℐᵀ the inducing kernel matrix with lower Cholesky
factor L_ℐℐ, and k_ℐn the kernel column of input n, row n of K_Xℐ.
"""

from __future__ import annotations

import jax
import lineax as lx
from jaxtyping import Array, Float


def project(
    K_XZ: Float[Array, "B M"],
    L_Z: lx.AbstractLinearOperator,
) -> Float[Array, "B M"]:
    """Projection A_X = K_Xℐ K_ℐℐ⁻¹ onto the inducing inputs via Cholesky solves.

        A_X = K_Xℐ K_ℐℐ⁻¹,   K_ℐℐ = L_ℐℐ L_ℐℐᵀ

    Used in sparse variational GPs to project test inputs onto the
    inducing space. Row n of A_X is (K_ℐℐ⁻¹ k_ℐn)ᵀ.

    Algorithm:
        1. Tag L_ℐℐ lower-triangular if it is not already.
        2. For each input n, solve L_ℐℐ v_n = k_ℐn by forward substitution.
        3. Solve L_ℐℐᵀ a_n = v_n by back substitution; a_n is row n of A_X.

    Args:
        K_XZ: Kernel matrix K_Xℐ between the inputs and the inducing inputs.
        L_Z: Lower Cholesky factor L_ℐℐ of K_ℐℐ. It need not carry
            ``lineax.lower_triangular_tag``: the tag is added when missing
            (the factor is lower-triangular by precondition), and
            ``L_Z.T`` is then upper-triangular.

    Returns:
        The projection A_X, one row per input.

    Raises:
        ValueError: If ``L_Z`` is not N_ℐ × N_ℐ with N_ℐ the number of
            columns of ``K_XZ``.

    Examples:
        A diagonal inducing kernel matrix K_ℐℐ = 4 I:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> factor = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(2), lx.lower_triangular_tag
        ... )
        >>> cross = jnp.array([[2.0, 0.0], [0.0, 4.0]])
        >>> projection = gaussx.project(cross, factor)
        >>> bool(jnp.allclose(projection, jnp.array([[0.5, 0.0], [0.0, 1.0]])))
        True

        An untagged factor from ``jnp.linalg.cholesky`` matches a dense solve:

        >>> inducing = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        >>> untagged = lx.MatrixLinearOperator(jnp.linalg.cholesky(inducing))
        >>> cross = jnp.array([[1.0, 0.2], [0.3, 0.7], [0.0, 1.0]])
        >>> projection = gaussx.project(cross, untagged)
        >>> projection.shape
        (3, 2)
        >>> expected = jnp.linalg.solve(inducing, cross.T).T
        >>> bool(jnp.allclose(projection, expected, atol=1e-5))
        True

        A factor of the wrong size is rejected:

        >>> wide = jnp.ones((2, 3))
        >>> gaussx.project(wide, factor)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: L_Z must be (3, 3) to match K_XZ's 3 columns, got (2, 2).
    """
    inducing_size = K_XZ.shape[-1]
    if L_Z.in_size() != inducing_size or L_Z.out_size() != inducing_size:
        raise ValueError(
            f"L_Z must be ({inducing_size}, {inducing_size}) to match K_XZ's "
            f"{inducing_size} columns, got ({L_Z.out_size()}, {L_Z.in_size()})."
        )
    if not lx.is_lower_triangular(L_Z):
        # lx.Triangular checks tags, so an untagged factor (e.g. built from
        # jnp.linalg.cholesky) would be rejected.
        L_Z = lx.TaggedLinearOperator(L_Z, lx.lower_triangular_tag)
    solver = lx.Triangular()

    def _solve_row(kernel_column):
        # v_n = L_ℐℐ⁻¹ k_ℐn
        forward = lx.linear_solve(L_Z, kernel_column, solver).value
        # a_n = L_ℐℐ⁻ᵀ v_n = K_ℐℐ⁻¹ k_ℐn
        return lx.linear_solve(L_Z.T, forward, solver).value

    # A_X, one row a_nᵀ per input
    return jax.vmap(_solve_row)(K_XZ)
