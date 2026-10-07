"""Collapsed ELBO (Titsias variational bound) for sparse GP regression."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jax import lax
from jaxtyping import Array, Float

from gaussx._deprecation import renamed_kwargs, warn_deprecated
from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._einx import einsum, rearrange
from gaussx._linalg._linalg import solve_columns
from gaussx._primitives._cholesky import cholesky
from gaussx._strategies._base import AbstractSolverStrategy


@renamed_kwargs(K_diag="K_xx_diag")
def collapsed_elbo(
    y: Float[Array, " N"],
    K_xx_diag: Float[Array, " N"],
    K_xz: Float[Array, "N M"],
    K_zz: Float[Array, "M M"] | lx.AbstractLinearOperator,
    noise_var: float,
    *,
    jitter: float = 1e-6,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    """Collapsed ELBO (Titsias bound) for sparse GP regression.

    Computes the variational lower bound on the log marginal likelihood
    using the matrix determinant lemma for O(NM² + M³) cost:

        ELBO = log 𝒩(y | 0, Q_ff + σ²I) − ½σ⁻² tr(K_ff − Q_ff)

    where Q_ff = K_xz K_zz⁻¹ K_xzᵀ is the Nyström approximation. With
    ``L L^T = K_zz`` and ``V = L⁻¹ K_xzᵀ`` (structured triangular solves),
    ``B = I + σ⁻² V Vᵀ`` is the only dense ``(M, M)`` factorisation.

    Args:
        y: Observations, shape ``(N,)``.
        K_xx_diag: Diagonal of the full kernel matrix K_ff, shape ``(N,)``.
            (Formerly ``K_diag``; the old keyword is deprecated.)
        K_xz: Cross-covariance between data and inducing points,
            shape ``(N, M)``.
        K_zz: Inducing point kernel matrix, ``(M, M)`` array or PSD
            operator. A `Kronecker` / `BlockDiag` operator is factorised
            per factor.
        noise_var: Observation noise variance σ² (scalar).
        jitter: Diagonal jitter added to ``K_zz`` before the Cholesky
            factorisation, for a dense array, `lineax.MatrixLinearOperator`
            or `lineax.DiagonalLinearOperator`. It is **not** added to other
            structured operators (it would destroy their structure):
            include any jitter in the operator itself, e.g. per factor.
        solver: Deprecated and ignored (the Cholesky factorisations take no
            solver); passing one warns. Removed in the next minor release.

    Returns:
        Scalar ELBO value.
    """
    if solver is not None:
        warn_deprecated(
            "collapsed_elbo(solver=...) is ignored and deprecated; it will be "
            "removed in the next minor release."
        )
    N = y.shape[0]
    K_zz_op = _jittered(K_zz, jitter)
    M = K_zz_op.in_size()

    # L_zz L_zzᵀ = K_zz (+ jitter · I);  V = L_zz⁻¹ K_xzᵀ  (M, N)
    L_zz = cholesky(K_zz_op)
    V = solve_columns(L_zz, rearrange(K_xz, "n m -> m n"))

    # B = I_M + σ⁻² V Vᵀ
    B = jnp.eye(M, dtype=V.dtype) + (1.0 / noise_var) * einsum(
        V, V, "i n, j n -> i j"
    )  # (M, M)
    L_B = cholesky(  # (M, M)
        lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag)
    ).as_matrix()

    # log|Q_ff + σ²I| = N log σ² + log|B|
    from gaussx._primitives._logdet import cholesky_logdet

    log_det = N * jnp.log(noise_var) + cholesky_logdet(L_B)

    # Quadratic form via Woodbury:
    # yᵀ (Q_ff + σ²I)⁻¹ y = σ⁻²(‖y‖² − σ⁻² ‖L_B⁻¹ V y‖²)
    Vy = V @ y  # (M,)
    LBinv_Vy = lax.linalg.triangular_solve(
        L_B,
        Vy,
        left_side=True,
        lower=True,
    )  # (M,)
    quad = (1.0 / noise_var) * (
        jnp.sum(y**2) - (1.0 / noise_var) * jnp.sum(LBinv_Vy**2)
    )

    # Trace penalty: −½σ⁻² (tr(K_ff) − tr(Q_ff))
    # where tr(Q_ff) = ‖V‖²_F
    trace_penalty = -0.5 / noise_var * (jnp.sum(K_xx_diag) - jnp.sum(V**2))

    return -0.5 * (log_det + quad + N * _LOG_2PI) + trace_penalty


def _jittered(
    K_zz: Float[Array, "M M"] | lx.AbstractLinearOperator, jitter: float
) -> lx.AbstractLinearOperator:
    """``K_zz + jitter I`` for dense / diagonal inputs; others unchanged."""
    if isinstance(K_zz, lx.DiagonalLinearOperator):
        return lx.DiagonalLinearOperator(lx.diagonal(K_zz) + jitter)
    if isinstance(K_zz, lx.MatrixLinearOperator):
        K_zz = K_zz.as_matrix()
    if isinstance(K_zz, lx.AbstractLinearOperator):
        return K_zz
    M = K_zz.shape[0]
    return lx.MatrixLinearOperator(
        K_zz + jitter * jnp.eye(M, dtype=K_zz.dtype), lx.positive_semidefinite_tag
    )
