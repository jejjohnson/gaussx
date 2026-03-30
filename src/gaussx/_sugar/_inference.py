"""Inference sugar: expected log-lik, trace correction, cavity, Newton update."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._primitives._inv import inv
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve
from gaussx._primitives._trace import trace
from gaussx._sugar._gaussian import _LOG_2PI


def log_marginal_likelihood(
    loc: jnp.ndarray,
    cov_operator: lx.AbstractLinearOperator,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """GP log marginal likelihood.

    Computes::

        log p(y) = -0.5 * (y-mu)^T K^{-1} (y-mu) - 0.5 * log|K| - N/2 * log(2pi)

    Equivalent to ``gaussian_log_prob`` but named for GP convention.

    Args:
        loc: Prior mean, shape ``(N,)``.
        cov_operator: Covariance operator K, shape ``(N, N)``.
        y: Observations, shape ``(N,)``.

    Returns:
        Scalar log marginal likelihood.
    """
    N = y.shape[-1]
    residual = y - loc
    alpha = solve(cov_operator, residual)
    quad = residual @ alpha
    ld = logdet(cov_operator)
    return -0.5 * (quad + ld + N * _LOG_2PI)


def gaussian_expected_log_lik(
    y: jnp.ndarray,
    q_mu: jnp.ndarray,
    q_cov: lx.AbstractLinearOperator,
    noise: lx.AbstractLinearOperator,
) -> jnp.ndarray:
    r"""Expected log-likelihood ``E_q[log N(y | f, R)]``.

    Computes::

        E_q[log N(y|f,R)] = log N(y | q_mu, R) - 0.5 * tr(R^{-1} q_cov)

    Core to variational inference ELBO computation.

    Args:
        y: Observations, shape ``(N,)``.
        q_mu: Variational mean, shape ``(N,)``.
        q_cov: Variational covariance operator, shape ``(N, N)``.
        noise: Noise covariance operator R, shape ``(N, N)``.

    Returns:
        Scalar expected log-likelihood.
    """
    N = y.shape[-1]
    residual = y - q_mu
    alpha = solve(noise, residual)
    quad = residual @ alpha
    ld = logdet(noise)

    # Trace correction: tr(R^{-1} q_cov)
    R_inv = inv(noise)
    from gaussx._sugar._linalg import trace_product

    tr_term = trace_product(R_inv, q_cov)

    return -0.5 * (N * _LOG_2PI + ld + quad + tr_term)


def trace_correction(
    K_xx: lx.AbstractLinearOperator,
    K_xz: jnp.ndarray,
    K_zz: lx.AbstractLinearOperator,
) -> jnp.ndarray:
    """Trace term in Titsias collapsed ELBO.

    Computes::

        tr(K_xx) - tr(K_xz^T K_zz^{-1} K_xz)

    This is the "trace correction" that penalizes the Nystrom
    approximation error.

    Args:
        K_xx: Full covariance, shape ``(N, N)``.
        K_xz: Cross-covariance, shape ``(N, M)``.
        K_zz: Inducing covariance, shape ``(M, M)``.

    Returns:
        Scalar trace correction.
    """
    tr_full = trace(K_xx)

    # tr(K_xz^T K_zz^{-1} K_xz) = sum_ij W_ij * K_xz_ij
    # where W = K_zz^{-1} K_xz^T reshaped, but easier:
    # tr(A^T B) = sum(A * B), so tr(K_xz^T W) where W_col = K_zz^{-1} K_xz_col
    W = jax.vmap(lambda row: solve(K_zz, row))(K_xz)  # (N, M)
    tr_approx = jnp.sum(K_xz * W)

    return tr_full - tr_approx


def cavity_distribution(
    post_mean: jnp.ndarray,
    post_cov: lx.AbstractLinearOperator,
    site_nat1: jnp.ndarray,
    site_nat2: lx.AbstractLinearOperator,
    power: float = 1.0,
) -> tuple[jnp.ndarray, lx.AbstractLinearOperator]:
    """Compute EP cavity distribution by removing a site.

    Computes::

        cav_prec = post_prec - power * site_nat2
        cav_cov  = inv(cav_prec)
        cav_mean = cav_cov @ (post_prec @ post_mean - power * site_nat1)

    Args:
        post_mean: Posterior mean, shape ``(N,)``.
        post_cov: Posterior covariance operator.
        site_nat1: Site natural parameter (precision-weighted mean).
        site_nat2: Site natural parameter (precision).
        power: Power EP fraction (default 1.0 for standard EP).

    Returns:
        Tuple ``(cav_mean, cav_cov)``.
    """
    post_prec = inv(post_cov)
    cav_prec_mat = post_prec.as_matrix() - power * site_nat2.as_matrix()
    cav_prec = lx.MatrixLinearOperator(cav_prec_mat)
    cav_cov = inv(cav_prec)

    eta1_cav = post_prec.mv(post_mean) - power * site_nat1
    cav_mean = cav_cov.mv(eta1_cav)

    return cav_mean, cav_cov


def newton_update(
    mean: jnp.ndarray,
    jacobian: jnp.ndarray,
    hessian: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert a Newton step to natural pseudo-likelihood parameters.

    Computes::

        nat1 = jacobian - hessian @ mean
        nat2 = -hessian

    Used in Laplace/Newton-based approximate inference to convert
    function-space derivatives into site natural parameters.

    Args:
        mean: Current mean, shape ``(N,)`` or ``(D,)``.
        jacobian: First derivative of log-likelihood, shape ``(N,)``.
        hessian: Second derivative (negative definite), shape ``(N, N)``.

    Returns:
        Tuple ``(nat1, nat2)`` — site natural parameters.
    """
    nat1 = jacobian - hessian @ mean
    nat2 = -hessian
    return nat1, nat2


def process_noise_covariance(
    A: jnp.ndarray,
    Pinf: jnp.ndarray,
) -> jnp.ndarray:
    """Compute process noise from stationary covariance.

    Computes::

        Q = Pinf - A @ Pinf @ A^T

    For a discrete-time state-space model with stationary covariance
    ``Pinf`` and transition matrix ``A``.

    Args:
        A: State transition matrix, shape ``(N, N)``.
        Pinf: Stationary covariance, shape ``(N, N)``.

    Returns:
        Process noise covariance Q, shape ``(N, N)``.
    """
    return Pinf - A @ Pinf @ A.T
