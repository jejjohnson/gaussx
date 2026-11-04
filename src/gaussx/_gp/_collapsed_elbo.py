"""Collapsed ELBO (Titsias variational bound) for sparse GP regression.

X stacks the N training inputs and X_ℐ the N_ℐ inducing inputs of the
inducing set ℐ, so K_XX is the N × N prior kernel matrix, K_Xℐ the N × N_ℐ
cross-covariance (K_ℐX = K_Xℐᵀ), K_ℐℐ = L_ℐℐ L_ℐℐᵀ the inducing kernel
matrix and Q_XX = K_Xℐ K_ℐℐ⁻¹ K_ℐX the Nyström matrix; σ_ε² is the
observation noise variance.
"""

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
from gaussx._primitives._logdet import cholesky_logdet
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

        ELBO = log 𝒩(y; 0, Q_XX + σ_ε² I) − tr(K_XX − Q_XX) / (2σ_ε²)

    The optimal q(f_ℐ) is integrated out analytically. With
    V_X = L_ℐℐ⁻¹ K_ℐX, so that Q_XX = V_Xᵀ V_X, the matrix determinant lemma
    and Woodbury's identity reduce everything to the N_ℐ × N_ℐ matrix
    B = I + σ_ε⁻² V_X V_Xᵀ = L_B L_Bᵀ, at O(N N_ℐ² + N_ℐ³) cost:

        log det(Q_XX + σ_ε² I) = N log σ_ε² + log det B
        yᵀ (Q_XX + σ_ε² I)⁻¹ y = σ_ε⁻² (‖y‖² − σ_ε⁻² ‖L_B⁻¹ V_X y‖²)
        tr(Q_XX) = ‖V_X‖_F²

    K_ℐℐ is factorised with `gaussx.cholesky`, so its triangular solves are
    structured; B is the only dense factorisation.

    Algorithm:
        1. Add the jitter to K_ℐℐ (dense and diagonal inputs only), factorise
           K_ℐℐ = L_ℐℐ L_ℐℐᵀ and whiten V_X = L_ℐℐ⁻¹ K_ℐX.
        2. Form B = I + σ_ε⁻² V_X V_Xᵀ and factorise B = L_B L_Bᵀ.
        3. Take log det(Q_XX + σ_ε² I) from σ_ε² and L_B.
        4. Take the quadratic form from one triangular solve with L_B.
        5. Subtract the trace penalty tr(K_XX − Q_XX) / (2σ_ε²).

    Args:
        y: Observations y.
        K_xx_diag: Prior variances diag(K_XX) at the training inputs.
            (Formerly ``K_diag``; the old keyword is deprecated.)
        K_xz: Cross-covariance K_Xℐ between the training inputs and the
            inducing inputs.
        K_zz: Inducing kernel matrix K_ℐℐ, as a dense array or a PSD
            operator. A `Kronecker` / `BlockDiag` operator is factorised
            per factor.
        noise_var: Observation noise variance σ_ε².
        jitter: Diagonal jitter added to ``K_zz`` before the Cholesky
            factorisation, for a dense array, `lineax.MatrixLinearOperator`
            or `lineax.DiagonalLinearOperator`. It is **not** added to other
            structured operators (it would destroy their structure):
            include any jitter in the operator itself, e.g. per factor.
        solver: Deprecated and ignored (the Cholesky factorisations take no
            solver); passing one warns. Removed in gaussx 0.7.0.

    Returns:
        The scalar ELBO.

    Examples:
        With the inducing inputs at the training inputs, Q_XX = K_XX, the
        trace penalty vanishes and the bound is the exact log marginal
        likelihood:

        >>> import jax.numpy as jnp
        >>> import jax.scipy.stats as jss
        >>> import gaussx
        >>> K = jnp.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.5], [0.2, 0.5, 1.0]])
        >>> y = jnp.array([0.3, -0.2, 0.5])
        >>> elbo = gaussx.collapsed_elbo(y, jnp.diag(K), K, K, 0.1, jitter=0.0)
        >>> exact = jss.multivariate_normal.logpdf(
        ...     y, jnp.zeros(3), K + 0.1 * jnp.eye(3)
        ... )
        >>> bool(jnp.allclose(elbo, exact, atol=1e-4))
        True

        Fewer inducing inputs give a lower bound:

        >>> sparse = gaussx.collapsed_elbo(y, jnp.diag(K), K[:, :1], K[:1, :1], 0.1)
        >>> bool(sparse < elbo)
        True

    See Also:
        `variational_elbo_gaussian`: The uncollapsed (SVGP; Hensman et al.,
        2013) bound, which keeps q(f_ℐ) explicit.

    References:
        Titsias, M. (2009). Variational learning of inducing variables in
        sparse Gaussian processes. *AISTATS*.
    """
    if solver is not None:
        warn_deprecated(
            "collapsed_elbo(solver=...) is ignored and deprecated; it will be "
            "removed in gaussx 0.7.0."
        )
    num_samples = y.shape[0]
    # K_ℐℐ + jitter I, where that keeps the structure
    inducing_operator = _jittered(K_zz, jitter)
    inducing_size = inducing_operator.in_size()

    # --- whiten ---
    # K_ℐℐ = L_ℐℐ L_ℐℐᵀ, structured where possible
    inducing_factor = cholesky(inducing_operator)
    # K_ℐX = K_Xℐᵀ
    cross_covariance_t = rearrange(K_xz, "sample inducing -> inducing sample")
    # V_X = L_ℐℐ⁻¹ K_ℐX
    whitened_cross = solve_columns(inducing_factor, cross_covariance_t)

    # --- B = I + σ_ε⁻² V_X V_Xᵀ ---
    identity = jnp.eye(inducing_size, dtype=whitened_cross.dtype)
    # σ_ε⁻²
    noise_precision = 1.0 / noise_var
    # V_X V_Xᵀ
    whitened_gram = einsum(
        whitened_cross, whitened_cross, "row sample, col sample -> row col"
    )
    # σ_ε⁻² V_X V_Xᵀ
    scaled_gram = noise_precision * whitened_gram
    # B = I + σ_ε⁻² V_X V_Xᵀ
    b_matrix = identity + scaled_gram
    b_operator = lx.MatrixLinearOperator(b_matrix, lx.positive_semidefinite_tag)
    # B = L_B L_Bᵀ
    b_factor = cholesky(b_operator).as_matrix()

    # --- log-determinant ---
    # log σ_ε²
    log_noise_variance = jnp.log(noise_var)
    # N log σ_ε²
    noise_logdet = num_samples * log_noise_variance
    # log det B
    b_logdet = cholesky_logdet(b_factor)
    # log det(Q_XX + σ_ε² I) = N log σ_ε² + log det B
    logdet = noise_logdet + b_logdet

    # --- quadratic form, by Woodbury ---
    # V_X y
    whitened_obs = whitened_cross @ y
    # L_B⁻¹ V_X y
    b_whitened_obs = lax.linalg.triangular_solve(
        b_factor,
        whitened_obs,
        left_side=True,
        lower=True,
    )
    # σ_ε⁻², recomputed rather than reused so the traced graph is unchanged
    outer_precision = 1.0 / noise_var
    # y²
    obs_squared = y**2
    # ‖y‖²
    obs_norm = jnp.sum(obs_squared)
    # σ_ε⁻², as above
    inner_precision = 1.0 / noise_var
    # (L_B⁻¹ V_X y)²
    b_whitened_obs_squared = b_whitened_obs**2
    # ‖L_B⁻¹ V_X y‖²
    b_whitened_norm = jnp.sum(b_whitened_obs_squared)
    # σ_ε⁻² ‖L_B⁻¹ V_X y‖²
    explained_norm = inner_precision * b_whitened_norm
    # ‖y‖² − σ_ε⁻² ‖L_B⁻¹ V_X y‖²
    woodbury_norm = obs_norm - explained_norm
    # yᵀ (Q_XX + σ_ε² I)⁻¹ y = σ_ε⁻² (‖y‖² − σ_ε⁻² ‖L_B⁻¹ V_X y‖²)
    mahalanobis = outer_precision * woodbury_norm

    # --- trace penalty ---
    # −1 / (2σ_ε²)
    negative_half_precision = -0.5 / noise_var
    # tr(K_XX)
    prior_trace = jnp.sum(K_xx_diag)
    # V_X²
    whitened_cross_squared = whitened_cross**2
    # tr(Q_XX) = ‖V_X‖_F²
    nystrom_trace = jnp.sum(whitened_cross_squared)
    # tr(K_XX − Q_XX)
    trace_gap = prior_trace - nystrom_trace
    # −tr(K_XX − Q_XX) / (2σ_ε²)
    trace_penalty = negative_half_precision * trace_gap

    # log det(Q_XX + σ_ε² I) + yᵀ (Q_XX + σ_ε² I)⁻¹ y + N log 2π
    gaussian_terms = logdet + mahalanobis + num_samples * _LOG_2PI
    # log 𝒩(y; 0, Q_XX + σ_ε² I)
    log_marginal = -0.5 * gaussian_terms
    # ELBO = log 𝒩(y; 0, Q_XX + σ_ε² I) − tr(K_XX − Q_XX) / (2σ_ε²)
    return log_marginal + trace_penalty


def _jittered(
    inducing_kernel: Float[Array, "M M"] | lx.AbstractLinearOperator,
    jitter: float,
) -> lx.AbstractLinearOperator:
    """K_ℐℐ + jitter I for dense and diagonal inputs; other operators unchanged.

    Args:
        inducing_kernel: Inducing kernel matrix K_ℐℐ, as an array or an
            operator.
        jitter: Diagonal jitter added to dense and diagonal K_ℐℐ.

    Returns:
        A PSD operator: K_ℐℐ + jitter I for a dense array,
        `lineax.MatrixLinearOperator` or `lineax.DiagonalLinearOperator`,
        otherwise ``inducing_kernel`` itself.
    """
    if isinstance(inducing_kernel, lx.DiagonalLinearOperator):
        # diag(K_ℐℐ) + jitter
        jittered_diagonal = lx.diagonal(inducing_kernel) + jitter
        return lx.DiagonalLinearOperator(jittered_diagonal)
    if isinstance(inducing_kernel, lx.MatrixLinearOperator):
        inducing_kernel = inducing_kernel.as_matrix()
    if isinstance(inducing_kernel, lx.AbstractLinearOperator):
        return inducing_kernel
    inducing_size = inducing_kernel.shape[0]
    identity = jnp.eye(inducing_size, dtype=inducing_kernel.dtype)
    # jitter I
    jitter_matrix = jitter * identity
    # K_ℐℐ + jitter I
    jittered_kernel = inducing_kernel + jitter_matrix
    return lx.MatrixLinearOperator(jittered_kernel, lx.positive_semidefinite_tag)
