"""KL divergence of a Cholesky-parameterised q(f_ℐ) from the GP prior p(f_ℐ).

Each of the R outputs, indexed by r, has the variational distribution
q(f_ℐ,r) = 𝒩(μ_ℐ,r, L_ℐ,r L_ℐ,rᵀ) over the N_ℐ inducing values, and the
prior is p(f_ℐ) = 𝒩(0, K_ℐℐ) with K_ℐℐ = L_ℐℐ L_ℐℐᵀ, or 𝒩(0, I) for the
whitened values z_ℐ = L_ℐℐ⁻¹ f_ℐ.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import rearrange, reduce
from gaussx._gp._base_conditional import _as_psd_operator
from gaussx._linalg._linalg import solve_columns
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._logdet import cholesky_logdet
from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractSolverStrategy


def gauss_kl(
    q_mu: Float[Array, "M R"],
    q_sqrt: Float[Array, "R M M"] | Float[Array, "M R"],
    K: Float[Array, "M M"] | lx.AbstractLinearOperator | None = None,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, ""]:
    """KL(first ‖ second): KL[q(f_ℐ) ‖ p(f_ℐ)] with p(f_ℐ) = 𝒩(0, K_ℐℐ).

        KL[q ‖ p] = ½ ∑_r (tr(K_ℐℐ⁻¹ L_ℐ,r L_ℐ,rᵀ) + μ_ℐ,rᵀ K_ℐℐ⁻¹ μ_ℐ,r
                           − N_ℐ + log det K_ℐℐ − log det(L_ℐ,r L_ℐ,rᵀ))

    with q(f_ℐ,r) = 𝒩(μ_ℐ,r, L_ℐ,r L_ℐ,rᵀ) for each output r. Without
    ``K`` the prior is the whitened 𝒩(0, I), and every K_ℐℐ above is I.

    This is the Cholesky-parameterised variant of `gaussian_kl` for GP and
    SVGP models: it never builds a covariance, and takes the roots L_ℐ,r
    either full or diagonal.

    - **Full** ``q_sqrt``: one lower-triangular root L_ℐ,r per output,
      stacked over outputs first.
    - **Diagonal** ``q_sqrt``: the diagonals of L_ℐ,r, the standard
      deviations of q(f_ℐ,r), one column per output.

    For lineax-operator covariances, use `gaussian_kl` instead.

    Algorithm:
        1. Factorise K_ℐℐ = L_ℐℐ L_ℐℐᵀ with `gaussx.cholesky` (structured
           where possible), unless the prior is whitened.
        2. Diagonal roots: take the quadratic term ‖L_ℐℐ⁻¹ μ_ℐ,r‖², the trace
           ∑_i [L_ℐ,r]_ii² [K_ℐℐ⁻¹]_ii from diag(K_ℐℐ⁻¹), and the
           log-determinants from the factors, summed over outputs.
        3. Full roots: per output, take the quadratic term ‖L_ℐℐ⁻¹ μ_ℐ,r‖²,
           the trace ‖L_ℐℐ⁻¹ L_ℐ,r‖_F² and the log-determinants, then sum
           over outputs.

    Args:
        q_mu: Variational means μ_ℐ,r, one column per output.
        q_sqrt: Variational roots L_ℐ,r, full (stacked over outputs first)
            or diagonal (one column per output).
        K: Prior covariance K_ℐℐ, as a dense array or a PSD operator (a
            `Kronecker` / `BlockDiag` operator is factorised per factor).
            ``None`` uses the whitened prior 𝒩(0, I).
        solver: Deprecated and ignored (the Cholesky factorisation takes
            no solver); passing one warns. Removed in gaussx
            0.7.0.

    Returns:
        The scalar KL[q(f_ℐ) ‖ p(f_ℐ)], summed over the R outputs.

    Examples:
        q equal to the whitened prior has zero divergence:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> q_mu = jnp.zeros((3, 2))
        >>> q_sqrt = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
        >>> kl = gaussx.gauss_kl(q_mu, q_sqrt)
        >>> kl.shape, bool(jnp.allclose(kl, 0.0))
        ((), True)

        Diagonal roots agree with the matching full roots:

        >>> K = jnp.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 1.5]])
        >>> q_mu = jnp.array([[0.5, -1.0], [0.0, 0.3], [1.0, 0.2]])
        >>> q_diag = jnp.array([[0.5, 0.8], [1.0, 0.4], [0.3, 0.6]])
        >>> q_full = jnp.stack([jnp.diag(q_diag[:, 0]), jnp.diag(q_diag[:, 1])])
        >>> diagonal = gaussx.gauss_kl(q_mu, q_diag, K)
        >>> full = gaussx.gauss_kl(q_mu, q_full, K)
        >>> bool(jnp.allclose(diagonal, full, atol=1e-5))
        True

        One output matches `gaussx.gaussian_kl` on dense covariances:

        >>> import lineax as lx
        >>> q_cov = lx.MatrixLinearOperator(
        ...     jnp.diag(q_diag[:, 0] ** 2), lx.positive_semidefinite_tag
        ... )
        >>> p_cov = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        >>> reference = gaussx.gaussian_kl(q_mu[:, 0], q_cov, jnp.zeros(3), p_cov)
        >>> single = gaussx.gauss_kl(q_mu[:, :1], q_diag[:, :1], K)
        >>> bool(jnp.allclose(single, reference, atol=1e-5))
        True

    See Also:
        `gaussian_kl`: KL between two multivariate normals with lineax
        covariance operators.
    """
    if solver is not None:
        warn_deprecated(
            "gauss_kl(solver=...) is ignored and deprecated; it will be "
            "removed in gaussx 0.7.0."
        )
    inducing_size = q_mu.shape[0]
    num_outputs = q_mu.shape[1]

    is_diagonal = q_sqrt.ndim == 2

    # K_ℐℐ = L_ℐℐ L_ℐℐᵀ, structured where gaussx.cholesky preserves it
    prior_factor = cholesky(_as_psd_operator(K)) if K is not None else None

    if is_diagonal:
        # [L_ℐ,r]_ii², the variances of q(f_ℐ,r)
        q_variance = q_sqrt**2

        if prior_factor is not None:
            # L_ℐℐ⁻¹ μ_ℐ,r
            whitened_mean = solve_columns(prior_factor, q_mu)
            # (L_ℐℐ⁻¹ μ_ℐ,r)²
            whitened_mean_squared = whitened_mean**2
            # ∑_r μ_ℐ,rᵀ K_ℐℐ⁻¹ μ_ℐ,r = ∑_r ‖L_ℐℐ⁻¹ μ_ℐ,r‖²
            mahalanobis = jnp.sum(whitened_mean_squared)

            # [K_ℐℐ⁻¹]_jj = ‖L_ℐℐ⁻¹ e_j‖², from one batched triangular solve
            # against I, with no inverse formed
            identity = jnp.eye(inducing_size, dtype=q_mu.dtype)
            # L_ℐℐ⁻¹
            inverse_factor = solve_columns(prior_factor, identity)
            # [L_ℐℐ⁻¹]²
            inverse_factor_squared = inverse_factor**2
            # [K_ℐℐ⁻¹]_jj = ∑_i [L_ℐℐ⁻¹]_ij²
            inverse_diagonal = reduce(inverse_factor_squared, "row col -> col", "sum")
            # [L_ℐ,r]_ii² [K_ℐℐ⁻¹]_ii
            weighted_variance = einx.multiply(
                "inducing output, inducing -> inducing output",
                q_variance,
                inverse_diagonal,
            )
            # ∑_r tr(K_ℐℐ⁻¹ L_ℐ,r L_ℐ,rᵀ)
            trace_term = jnp.sum(weighted_variance)

            # log det K_ℐℐ
            prior_logdet = _factor_logdet(prior_factor)
            # log [L_ℐ,r]_ii²
            log_q_variance = jnp.log(q_variance)
            # ∑_r log det(L_ℐ,r L_ℐ,rᵀ)
            q_logdet = jnp.sum(log_q_variance)
            # R log det K_ℐℐ
            total_prior_logdet = num_outputs * prior_logdet
            # ∑_r (log det K_ℐℐ − log det(L_ℐ,r L_ℐ,rᵀ))
            logdet_difference = total_prior_logdet - q_logdet
        else:
            # whitened prior, K_ℐℐ = I
            # μ_ℐ,r²
            q_mu_squared = q_mu**2
            # ∑_r ‖μ_ℐ,r‖²
            mahalanobis = jnp.sum(q_mu_squared)
            # ∑_r tr(L_ℐ,r L_ℐ,rᵀ)
            trace_term = jnp.sum(q_variance)
            # log [L_ℐ,r]_ii²
            log_q_variance = jnp.log(q_variance)
            # ∑_r log det(L_ℐ,r L_ℐ,rᵀ)
            q_logdet = jnp.sum(log_q_variance)
            # −∑_r log det(L_ℐ,r L_ℐ,rᵀ)
            logdet_difference = -q_logdet

        # ∑_r (log det K_ℐℐ − log det(L_ℐ,r L_ℐ,rᵀ)) − R N_ℐ
        kl_terms = logdet_difference - inducing_size * num_outputs
        # … + ∑_r tr(K_ℐℐ⁻¹ L_ℐ,r L_ℐ,rᵀ)
        kl_terms = kl_terms + trace_term
        # … + ∑_r μ_ℐ,rᵀ K_ℐℐ⁻¹ μ_ℐ,r
        kl_terms = kl_terms + mahalanobis
        # KL[q ‖ p] = ½ (…)
        return 0.5 * kl_terms

    # Full roots, stacked over outputs. The prior log-determinant is the
    # same for every output, so it is taken once, outside the vmap.
    # log det K_ℐℐ
    prior_logdet = _factor_logdet(prior_factor) if prior_factor is not None else 0.0

    def _kl_one_output(
        q_mu_r: Float[Array, " M"], q_sqrt_r: Float[Array, "M M"]
    ) -> Float[Array, ""]:
        # log det(L_ℐ,r L_ℐ,rᵀ)
        q_logdet = cholesky_logdet(q_sqrt_r)

        if prior_factor is not None:
            # L_ℐℐ⁻¹ μ_ℐ,r
            whitened_mean = solve(prior_factor, q_mu_r)
            # (L_ℐℐ⁻¹ μ_ℐ,r)²
            whitened_mean_squared = whitened_mean**2
            # μ_ℐ,rᵀ K_ℐℐ⁻¹ μ_ℐ,r = ‖L_ℐℐ⁻¹ μ_ℐ,r‖²
            mahalanobis = jnp.sum(whitened_mean_squared)
            # L_ℐℐ⁻¹ L_ℐ,r
            whitened_root = solve_columns(prior_factor, q_sqrt_r)
            # (L_ℐℐ⁻¹ L_ℐ,r)²
            whitened_root_squared = whitened_root**2
            # tr(K_ℐℐ⁻¹ L_ℐ,r L_ℐ,rᵀ) = ‖L_ℐℐ⁻¹ L_ℐ,r‖_F²
            trace_term = jnp.sum(whitened_root_squared)
            # log det K_ℐℐ − log det(L_ℐ,r L_ℐ,rᵀ)
            logdet_difference = prior_logdet - q_logdet
        else:
            # whitened prior, K_ℐℐ = I
            # μ_ℐ,r²
            q_mu_squared = q_mu_r**2
            # ‖μ_ℐ,r‖²
            mahalanobis = jnp.sum(q_mu_squared)
            # L_ℐ,r²
            q_sqrt_squared = q_sqrt_r**2
            # tr(L_ℐ,r L_ℐ,rᵀ) = ‖L_ℐ,r‖_F²
            trace_term = jnp.sum(q_sqrt_squared)
            # −log det(L_ℐ,r L_ℐ,rᵀ)
            logdet_difference = -q_logdet

        # log det K_ℐℐ − log det(L_ℐ,r L_ℐ,rᵀ) − N_ℐ
        kl_terms = logdet_difference - inducing_size
        # … + tr(K_ℐℐ⁻¹ L_ℐ,r L_ℐ,rᵀ)
        kl_terms = kl_terms + trace_term
        # … + μ_ℐ,rᵀ K_ℐℐ⁻¹ μ_ℐ,r
        kl_terms = kl_terms + mahalanobis
        # KL[q(f_ℐ,r) ‖ p(f_ℐ)] = ½ (…)
        return 0.5 * kl_terms

    # μ_ℐ,r stacked over outputs first, to match L_ℐ,r
    q_mu_by_output = rearrange(q_mu, "inducing output -> output inducing")
    # KL[q(f_ℐ,r) ‖ p(f_ℐ)], one per output
    kl_per_output = jax.vmap(_kl_one_output)(q_mu_by_output, q_sqrt)
    # KL[q ‖ p] = ∑_r KL[q(f_ℐ,r) ‖ p(f_ℐ)]
    return jnp.sum(kl_per_output)


def _factor_logdet(factor: lx.AbstractLinearOperator) -> Float[Array, ""]:
    """Log-determinant from a Cholesky factor, structured or dense.

        log det(L Lᵀ) = 2 ∑_i log [L]_ii

    Args:
        factor: Lower Cholesky factor L, as an operator.

    Returns:
        The scalar log det(L Lᵀ).
    """
    # diag(L)
    factor_diagonal = diag(factor)
    # log [L]_ii
    log_diagonal = jnp.log(factor_diagonal)
    # ∑_i log [L]_ii
    half_logdet = jnp.sum(log_diagonal)
    # log det(L Lᵀ) = 2 ∑_i log [L]_ii
    return 2.0 * half_logdet
