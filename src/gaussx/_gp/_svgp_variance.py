"""SVGP variance adjustment Q = K_ℐℐ⁻¹ Σ_ℐ K_ℐℐ⁻¹ − K_ℐℐ⁻¹.

K_ℐℐ is the N_ℐ × N_ℐ kernel matrix of the inducing inputs X_ℐ and Σ_ℐ the
covariance of the variational distribution q(f_ℐ) = 𝒩(μ_ℐ, Σ_ℐ) over the
inducing values f_ℐ.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._primitives._inv import inv


def svgp_variance_adjustment(
    K_zz_op: lx.AbstractLinearOperator,
    S_u: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """SVGP variance adjustment operator Q = K_ℐℐ⁻¹ Σ_ℐ K_ℐℐ⁻¹ − K_ℐℐ⁻¹.

        Q = (K_ℐℐ⁻¹ Σ_ℐ − I) K_ℐℐ⁻¹

    Q carries every sparse-GP predictive variance: for a test input x_*
    with inducing column k_ℐ* = k(X_ℐ, x_*),

        σ²_* = k(x_*, x_*) − k_ℐ*ᵀ K_ℐℐ⁻¹ k_ℐ* + k_ℐ*ᵀ K_ℐℐ⁻¹ Σ_ℐ K_ℐℐ⁻¹ k_ℐ*
             = k(x_*, x_*) + k_ℐ*ᵀ Q k_ℐ*

    The result is a linear operator, but it is built from dense N_ℐ × N_ℐ
    intermediates.

    Algorithm:
        1. Take the lazy inverse K_ℐℐ⁻¹ with `gaussx.inv`.
        2. Densify K_ℐℐ⁻¹ Σ_ℐ column by column.
        3. Subtract the identity, then densify (K_ℐℐ⁻¹ Σ_ℐ − I) K_ℐℐ⁻¹.

    Args:
        K_zz_op: Inducing kernel matrix K_ℐℐ, as an operator.
        S_u: Variational covariance Σ_ℐ of q(f_ℐ), as an operator.

    Returns:
        The adjustment Q, as a dense `lineax.MatrixLinearOperator`.

    Examples:
        Q vanishes when q(f_ℐ) equals the prior, Σ_ℐ = K_ℐℐ:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> K_zz = lx.MatrixLinearOperator(
        ...     jnp.array([[2.0, 0.5], [0.5, 1.0]]), lx.positive_semidefinite_tag
        ... )
        >>> Q = gaussx.svgp_variance_adjustment(K_zz, K_zz)
        >>> Q.as_matrix().shape
        (2, 2)
        >>> bool(jnp.allclose(Q.as_matrix(), 0.0, atol=1e-5))
        True

        It matches the dense formula:

        >>> S_u = lx.MatrixLinearOperator(
        ...     0.5 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> K_inv = jnp.linalg.inv(K_zz.as_matrix())
        >>> expected = K_inv @ (0.5 * jnp.eye(2)) @ K_inv - K_inv
        >>> Q = gaussx.svgp_variance_adjustment(K_zz, S_u)
        >>> bool(jnp.allclose(Q.as_matrix(), expected, atol=1e-5))
        True
    """
    # K_ℐℐ⁻¹, lazy
    inducing_inverse = inv(K_zz_op)
    # K_ℐℐ⁻¹ Σ_ℐ
    inverse_covariance_product = _compose_dense(inducing_inverse, S_u)
    inverse_covariance_op = lx.MatrixLinearOperator(inverse_covariance_product)
    inducing_size = K_zz_op.out_structure().shape[0]
    # −I
    negative_identity = lx.DiagonalLinearOperator(
        -jnp.ones(inducing_size, dtype=inverse_covariance_product.dtype)
    )
    # K_ℐℐ⁻¹ Σ_ℐ − I
    shifted = lx.AddLinearOperator(inverse_covariance_op, negative_identity)
    # Q = (K_ℐℐ⁻¹ Σ_ℐ − I) K_ℐℐ⁻¹
    adjustment = _compose_dense(shifted, inducing_inverse)
    return lx.MatrixLinearOperator(adjustment)


def _compose_dense(
    left: lx.AbstractLinearOperator,
    right: lx.AbstractLinearOperator,
) -> jax.Array:
    """Dense product A B of two operators, one column of B at a time.

        A B = [A b_1, …, A b_n]

    Args:
        left: Left operator A.
        right: Right operator B, materialised.

    Returns:
        The dense product A B.
    """
    right_matrix = right.as_matrix()
    # A b_j for every column b_j of B
    return jax.vmap(lambda col: left.mv(col), in_axes=1, out_axes=1)(right_matrix)
