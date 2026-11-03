"""Closed-form KL divergence between two Gaussians with operator covariances.

p = 𝒩(μ_p, Σ_p) and q = 𝒩(μ_q, Σ_q) share the event size D_x; L_p and L_q
are the lower Cholesky factors, L_p L_pᵀ = Σ_p and L_q L_qᵀ = Σ_q.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._linalg import trace_product
from gaussx._primitives._inv import inv
from gaussx._primitives._logdet import _cholesky_logdet, _dense_psd_matrix, logdet
from gaussx._primitives._solve import solve


def gaussian_kl(
    p_loc: Float[Array, " N"],
    p_cov: lx.AbstractLinearOperator,
    q_loc: Float[Array, " N"],
    q_cov: lx.AbstractLinearOperator,
) -> Float[Array, ""]:
    """KL(first ‖ second): KL[p ‖ q] between two multivariate normals.

        KL[p ‖ q] = ½ (tr(Σ_q⁻¹ Σ_p) + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) − D_x
                       + log det Σ_q − log det Σ_p)

    with p = 𝒩(μ_p, Σ_p) and q = 𝒩(μ_q, Σ_q). This is the canonical KL for
    lineax-operator covariances; the specialised variants compute the same
    quantity in other parameterisations:

    - `kl_standard_normal`: the case q = 𝒩(0, I), with no solve.
    - `gauss_kl`: the Cholesky-parameterised form for GP and SVGP models,
      with multi-output and diagonal ``q_sqrt``.
    - `kl_divergence`: the Bregman-divergence form on natural parameters
      of the exponential family.

    Algorithm:
        1. When both covariances are dense and PSD, factor Σ_p = L_p L_pᵀ
           and Σ_q = L_q L_qᵀ; then tr(Σ_q⁻¹ Σ_p) = ‖L_q⁻¹ L_p‖_F², the
           quadratic term is ‖L_q⁻¹ (μ_q − μ_p)‖² and both log-determinants
           come from the factors.
        2. Otherwise take tr(Σ_q⁻¹ Σ_p) through the lazy inverse of Σ_q, and
           the solve and log-determinants through structural dispatch.
        3. Combine the terms.

    Args:
        p_loc: Mean μ_p of the first Gaussian p.
        p_cov: Covariance Σ_p of p.
        q_loc: Mean μ_q of the second Gaussian q.
        q_cov: Covariance Σ_q of q.

    Returns:
        The scalar KL divergence KL[p ‖ q].

    Examples:
        A shifted, wider second Gaussian:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> p_cov = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
        >>> q_cov = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(2), lx.positive_semidefinite_tag
        ... )
        >>> q_loc = jnp.array([1.0, 0.0])
        >>> kl = gaussx.gaussian_kl(jnp.zeros(2), p_cov, q_loc, q_cov)
        >>> round(float(kl), 4)
        0.4431

        The divergence of a Gaussian from itself is zero:

        >>> self_kl = gaussx.gaussian_kl(q_loc, q_cov, q_loc, q_cov)
        >>> bool(jnp.allclose(self_kl, 0.0, atol=1e-6))
        True

        Diagonal operators take the structured path and agree:

        >>> p_diagonal = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> q_diagonal = lx.DiagonalLinearOperator(2.0 * jnp.ones(2))
        >>> structured = gaussx.gaussian_kl(
        ...     jnp.zeros(2), p_diagonal, q_loc, q_diagonal
        ... )
        >>> bool(jnp.allclose(structured, kl))
        True
    """
    event_size = p_loc.shape[-1]
    # μ_q − μ_p
    mean_difference = q_loc - p_loc

    p_matrix, q_matrix = _dense_psd_matrix(p_cov), _dense_psd_matrix(q_cov)
    if p_matrix is not None and q_matrix is not None:
        # Two Choleskys for the whole KL (gh-329).
        # L_p L_pᵀ = Σ_p
        p_factor = jnp.linalg.cholesky(p_matrix)
        # L_q L_qᵀ = Σ_q
        q_factor = jnp.linalg.cholesky(q_matrix)
        # L_q⁻¹ L_p
        whitened_factor = jax.scipy.linalg.solve_triangular(
            q_factor, p_factor, lower=True
        )
        # L_q⁻¹ (μ_q − μ_p)
        whitened_difference = jax.scipy.linalg.solve_triangular(
            q_factor, mean_difference, lower=True
        )
        # log det Σ_q − log det Σ_p
        logdet_difference = _cholesky_logdet(q_factor) - _cholesky_logdet(p_factor)
        # tr(Σ_q⁻¹ Σ_p) = ‖L_q⁻¹ L_p‖_F², a full reduction with no axis
        trace_term = jnp.sum(whitened_factor**2)
        # (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) = ‖L_q⁻¹ (μ_q − μ_p)‖²
        mahalanobis = whitened_difference @ whitened_difference
        # KL[p ‖ q] = ½ (tr + quadratic − D_x + log det Σ_q − log det Σ_p)
        return 0.5 * (trace_term + mahalanobis - event_size + logdet_difference)

    # Σ_q⁻¹, lazy
    q_precision = inv(q_cov)
    # tr(Σ_q⁻¹ Σ_p)
    trace_term = trace_product(q_precision, p_cov)

    # Σ_q⁻¹ (μ_q − μ_p)
    solved_difference = solve(q_cov, mean_difference)
    # (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p), a full reduction with no axis
    mahalanobis = jnp.sum(mean_difference * solved_difference)

    # log det Σ_q
    q_logdet = logdet(q_cov)
    # log det Σ_p
    p_logdet = logdet(p_cov)

    # KL[p ‖ q] = ½ (tr + quadratic − D_x + log det Σ_q − log det Σ_p)
    return 0.5 * (trace_term + mahalanobis - event_size + q_logdet - p_logdet)
