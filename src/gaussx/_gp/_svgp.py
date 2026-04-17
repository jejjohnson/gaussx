"""Whitened SVGP forward pass sugar."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from einops import reduce
from jaxtyping import Array, Float

from gaussx._primitives._cholesky import cholesky


def whitened_svgp_predict(
    K_zz_op: lx.AbstractLinearOperator,
    K_xz: Float[Array, "N M"],
    u_mean: Float[Array, " M"],
    u_chol: Float[Array, "M M"],
    K_xx_diag: Float[Array, " N"],
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Whitened SVGP prediction: mean and variance at test points.

    Computes the predictive mean and variance for a sparse variational
    GP using the whitened parameterization::

        L_{zz} = cholesky(K_{zz})
        A = L_{zz}^{-1} K_{zx}           (triangular solve)
        f_{loc} = A^T u_{mean}
        Q_{xx} = sum(A^2, axis=0)         (prior variance reduction)
        W = u_{chol}^T A
        S_{contrib} = sum(W^2, axis=0)    (posterior variance contribution)
        f_{var} = K_{xx,diag} - Q_{xx} + S_{contrib}

    Args:
        K_zz_op: Inducing-point covariance operator, shape ``(M, M)``.
        K_xz: Cross-covariance matrix, shape ``(N, M)``.
        u_mean: Whitened variational mean, shape ``(M,)``.
        u_chol: Whitened variational Cholesky factor, shape ``(M, M)``.
            Lower-triangular matrix such that the variational covariance
            in whitened space is ``u_chol @ u_chol^T``.
        K_xx_diag: Prior diagonal variances at test points, shape ``(N,)``.

    Returns:
        Tuple ``(f_loc, f_var)`` — predictive mean shape ``(N,)`` and
        predictive variance shape ``(N,)``.
    """
    L_zz = cholesky(K_zz_op)

    # A = L_zz^{-1} K_xz^T  -> shape (M, N)
    # Solve L_zz @ A_col = K_xz^T_col for each column of K_xzᵀ
    from gaussx._linalg._linalg import solve_columns

    K_zx = K_xz.T  # (M, N)
    A = solve_columns(L_zz, K_zx)

    # Predictive mean: f_loc = A^T @ u_mean = K_xz @ L_zz^{-T} @ u_mean
    f_loc = A.T @ u_mean

    # Prior variance reduction: Q_xx = Σₘ Aₘₙ²
    Q_xx = reduce(A**2, "M N -> N", "sum")

    # Posterior variance contribution: W = u_cholᵀ @ A, S = Σₖ Wₖₙ²
    W = u_chol.T @ A
    S_contrib = reduce(W**2, "K N -> N", "sum")

    # Predictive variance
    f_var = jnp.clip(K_xx_diag - Q_xx + S_contrib, 0.0)

    return f_loc, f_var
