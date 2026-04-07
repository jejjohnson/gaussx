"""Gaussian distribution in exponential family form."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from einops import einsum

from gaussx._primitives._inv import inv
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve


class GaussianExpFam(eqx.Module):
    r"""Gaussian in natural (exponential family) parameters.

    .. math::

        q(x \mid \eta) = h(x) \exp(\eta^T T(x) - A(\eta))

    where:

    - Natural parameters: ``eta1 = Lambda @ mu``, ``eta2 = -0.5 * Lambda``
    - Sufficient statistics: ``T(x) = [x, x x^T]``
    - Log-partition: ``A(eta) = -0.25 * eta1^T eta2^{-1} eta1 - 0.5 * log|-2 eta2|``
    - Base measure: ``h(x) = (2 pi)^{-N/2}``

    Attributes:
        eta1: Natural location parameter, shape ``(N,)``.
        eta2: Natural precision-like operator, shape ``(N, N)``.
            Represents ``-0.5 * Lambda`` where Lambda is the precision.
    """

    eta1: jnp.ndarray
    eta2: lx.AbstractLinearOperator

    @staticmethod
    def from_mean_cov(
        mu: jnp.ndarray,
        Sigma: lx.AbstractLinearOperator,
    ) -> GaussianExpFam:
        """Construct from mean and covariance.

        Args:
            mu: Mean vector, shape ``(N,)``.
            Sigma: Covariance operator, shape ``(N, N)``.

        Returns:
            A ``GaussianExpFam`` instance.
        """
        eta1 = solve(Sigma, mu)
        eta2 = -0.5 * inv(Sigma)
        return GaussianExpFam(eta1=eta1, eta2=eta2)

    @staticmethod
    def from_mean_prec(
        mu: jnp.ndarray,
        Lambda: lx.AbstractLinearOperator,
    ) -> GaussianExpFam:
        """Construct from mean and precision.

        Args:
            mu: Mean vector, shape ``(N,)``.
            Lambda: Precision operator, shape ``(N, N)``.

        Returns:
            A ``GaussianExpFam`` instance.
        """
        eta1 = Lambda.mv(mu)
        eta2 = -0.5 * Lambda
        return GaussianExpFam(eta1=eta1, eta2=eta2)


def to_expectation(
    expfam: GaussianExpFam,
) -> tuple[jnp.ndarray, lx.AbstractLinearOperator]:
    """Convert natural to expectation parameters.

    Args:
        expfam: Gaussian in natural form.

    Returns:
        Tuple ``(mu, Sigma)`` — mean vector and covariance operator.
    """
    neg2_eta2 = -2.0 * expfam.eta2
    mu = solve(neg2_eta2, expfam.eta1)
    Sigma = inv(neg2_eta2)
    return mu, Sigma


def to_natural(
    mu: jnp.ndarray,
    Sigma: lx.AbstractLinearOperator,
) -> tuple[jnp.ndarray, lx.AbstractLinearOperator]:
    """Convert expectation to natural parameters.

    Args:
        mu: Mean vector, shape ``(N,)``.
        Sigma: Covariance operator, shape ``(N, N)``.

    Returns:
        Tuple ``(eta1, eta2)`` — natural parameters.
    """
    eta1 = solve(Sigma, mu)
    eta2 = -0.5 * inv(Sigma)
    return eta1, eta2


def log_partition(expfam: GaussianExpFam) -> jnp.ndarray:
    r"""Log-partition function ``A(eta)``.

    .. math::

        A(\eta) = -\frac{1}{4} \eta_1^T \eta_2^{-1} \eta_1
                  - \frac{1}{2} \log|-2\eta_2|

    Args:
        expfam: Gaussian in natural form.

    Returns:
        Scalar log-partition value.
    """
    neg2_eta2 = -2.0 * expfam.eta2
    N = neg2_eta2.in_size()

    # -0.25 * eta1^T @ eta2^{-1} @ eta1
    # eta2^{-1} = (-0.5 Lambda)^{-1} = -2 Sigma
    # So -0.25 * eta1^T @ (-2 Sigma) @ eta1 = 0.5 * eta1^T Sigma eta1
    eta2_inv_eta1 = solve(expfam.eta2, expfam.eta1)
    quad = -0.25 * (expfam.eta1 @ eta2_inv_eta1)

    # -0.5 * log|-2 eta2| = -0.5 * logdet(Lambda)
    ld = -0.5 * logdet(neg2_eta2)

    # Add base measure contribution: N/2 * log(2pi)
    log_2pi = jnp.log(2.0 * jnp.pi)
    return quad + ld + 0.5 * N * log_2pi


def fisher_info(
    expfam: GaussianExpFam,
) -> lx.AbstractLinearOperator:
    r"""Fisher information matrix ``F(eta) = nabla^2 A(eta)``.

    For a Gaussian, the Fisher information in terms of the
    covariance is ``Sigma^{-1}`` (the precision matrix).

    Args:
        expfam: Gaussian in natural form.

    Returns:
        Precision operator (the Fisher information matrix).
    """
    # Lambda = -2 * eta2
    return -2.0 * expfam.eta2


def sufficient_stats(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute sufficient statistics ``T(x) = [x, x x^T]``.

    Args:
        x: Data vector, shape ``(N,)`` or batch ``(B, N)``.

    Returns:
        Tuple ``(x, outer_product)`` where outer_product has
        shape ``(N, N)`` or ``(B, N, N)``.
    """
    if x.ndim == 1:
        return x, jnp.outer(x, x)
    # Batched: (B, N) -> (B, N, N)
    return x, einsum(x, x, "b i, b j -> b i j")


def kl_divergence(
    q: GaussianExpFam,
    p: GaussianExpFam,
) -> jnp.ndarray:
    """KL divergence ``KL(q || p)`` via Bregman divergence.

    .. math::

        KL(q || p) = A(eta_p) - A(eta_q) - (eta_p - eta_q)^T nabla A(eta_q)

    Args:
        q: First Gaussian (the "true" distribution).
        p: Second Gaussian (the "approximate" distribution).

    Returns:
        Scalar KL divergence.
    """
    A_p = log_partition(p)
    A_q = log_partition(q)

    # grad A(eta_q) w.r.t eta1 = mu_q, w.r.t eta2 = mu_q mu_q^T + Sigma_q
    # The linear term: (eta_p - eta_q)^T grad A(eta_q)
    # For eta1 part: (eta1_p - eta1_q)^T mu_q
    mu_q, Sigma_q = to_expectation(q)

    delta_eta1 = p.eta1 - q.eta1
    linear_eta1 = delta_eta1 @ mu_q

    # For eta2 part: tr((eta2_p - eta2_q)^T (mu_q mu_q^T + Sigma_q))
    # = tr((eta2_p - eta2_q) @ (mu mu^T + Sigma))
    delta_eta2_mat = p.eta2.as_matrix() - q.eta2.as_matrix()
    moment2 = jnp.outer(mu_q, mu_q) + Sigma_q.as_matrix()
    linear_eta2 = jnp.sum(delta_eta2_mat * moment2)

    return A_p - A_q - linear_eta1 - linear_eta2
