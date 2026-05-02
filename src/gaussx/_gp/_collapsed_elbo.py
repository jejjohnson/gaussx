"""Collapsed ELBO (Titsias variational bound) for sparse GP regression."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jax import lax
from jaxtyping import Array, Float

from gaussx._primitives._cholesky import cholesky
from gaussx._strategies._base import AbstractSolverStrategy


def collapsed_elbo(
    y: Float[Array, " N"],
    K_diag: Float[Array, " N"],
    K_xz: Float[Array, "N M"],
    K_zz: Float[Array, "M M"],
    noise_var: float,
    *,
    jitter: float = 1e-6,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    """Collapsed ELBO (Titsias bound) for sparse GP regression.

    Computes the variational lower bound on the log marginal likelihood
    using the matrix determinant lemma for O(NM² + M³) cost::

        ELBO = log 𝒩(y | 0, Q_ff + σ²I) − ½σ⁻² tr(K_ff − Q_ff)

    where Q_ff = K_xz K_zz⁻¹ K_xzᵀ is the Nyström approximation.

    Args:
        y: Observations, shape ``(N,)``.
        K_diag: Diagonal of full kernel matrix K_ff, shape ``(N,)``.
        K_xz: Cross-covariance between data and inducing points,
            shape ``(N, M)``.
        K_zz: Inducing point kernel matrix, shape ``(M, M)``.
        noise_var: Observation noise variance σ² (scalar).
        jitter: Diagonal jitter for numerical stability in Cholesky
            decomposition of K_zz.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch. This parameter
            is accepted for API consistency but is not currently used by the
            Cholesky decompositions in this function.

    Returns:
        Scalar ELBO value.
    """
    del solver  # cholesky does not accept a solver; parameter reserved for future use
    N = y.shape[0]
    M = K_zz.shape[0]
    log_2pi = jnp.log(2.0 * jnp.pi)

    # L_zz L_zzᵀ = K_zz + jitter · I
    K_zz_jitter = K_zz + jitter * jnp.eye(M)
    L_zz = cholesky(  # (M, M)
        lx.MatrixLinearOperator(K_zz_jitter, lx.positive_semidefinite_tag)
    ).as_matrix()

    # V = L_zz⁻¹ K_xzᵀ
    V = lax.linalg.triangular_solve(
        L_zz,
        K_xz.T,
        left_side=True,
        lower=True,
    )  # (M, N)

    # B = I_M + σ⁻² V Vᵀ
    B = jnp.eye(M) + (1.0 / noise_var) * (V @ V.T)  # (M, M)
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
    trace_penalty = -0.5 / noise_var * (jnp.sum(K_diag) - jnp.sum(V**2))

    return -0.5 * (log_det + quad + N * log_2pi) + trace_penalty
