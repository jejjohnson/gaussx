"""SVGP variance adjustment operator."""

from __future__ import annotations

import jax
import lineax as lx

from gaussx._primitives._inv import inv


def svgp_variance_adjustment(
    K_zz_op: lx.AbstractLinearOperator,
    S_u: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    r"""Compute the SVGP variance adjustment operator.

    Builds the operator ``Q = K_{zz}^{-1} S_u K_{zz}^{-1} - K_{zz}^{-1}``
    which appears in every sparse GP predictive variance computation::

        Var[f_*] = k_{**} - k_{*z} (K_{zz}^{-1} - Q) k_{z*}

    The returned value is exposed as a linear operator, but the current
    implementation materializes dense ``(M, M)`` intermediates while building
    it.

    Args:
        K_zz_op: Inducing-point covariance operator, shape ``(M, M)``.
        S_u: Variational covariance operator, shape ``(M, M)``.

    Returns:
        Operator ``Q`` of shape ``(M, M)`` such that
        ``Q @ v = K_{zz}^{-1} S_u K_{zz}^{-1} v - K_{zz}^{-1} v``.
    """
    K_inv = inv(K_zz_op)
    K_inv_S = lx.MatrixLinearOperator(
        _compose_dense(K_inv, S_u),
    )
    # Q = K_inv @ S_u @ K_inv - K_inv
    # Build as (K_inv @ S_u - I) @ K_inv
    import jax.numpy as jnp

    M = K_zz_op.out_structure().shape[0]
    K_inv_S_minus_I = lx.AddLinearOperator(
        K_inv_S,
        lx.DiagonalLinearOperator(-jnp.ones(M)),
    )
    Q_dense = _compose_dense(K_inv_S_minus_I, K_inv)
    return lx.MatrixLinearOperator(Q_dense)


def _compose_dense(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
) -> jax.Array:
    """Compose two operators by materializing: (A @ B).as_matrix()."""
    import jax

    B_mat = B.as_matrix()
    return jax.vmap(lambda col: A.mv(col), in_axes=1, out_axes=1)(B_mat)
