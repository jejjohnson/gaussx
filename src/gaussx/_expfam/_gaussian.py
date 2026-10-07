"""Gaussian distribution in exponential family form."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._einx import einsum
from gaussx._expfam._natural import mean_cov_to_natural, natural_to_mean_cov
from gaussx._linalg._linalg import trace_product
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve


class GaussianExpFam(eqx.Module):
    r"""Gaussian in natural (exponential family) parameters.

    $$
    q(x \mid \eta) = h(x) \exp(\eta^T T(x) - A(\eta))
    $$

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

    eta1: Float[Array, " N"]
    eta2: lx.AbstractLinearOperator

    @staticmethod
    def from_mean_cov(
        mu: Float[Array, " N"],
        Sigma: lx.AbstractLinearOperator,
    ) -> GaussianExpFam:
        """Construct from mean and covariance.

        Args:
            mu: Mean vector, shape ``(N,)``.
            Sigma: Covariance operator, shape ``(N, N)``.

        Returns:
            A ``GaussianExpFam`` instance.
        """
        eta1, eta2 = mean_cov_to_natural(mu, Sigma)
        return GaussianExpFam(eta1=eta1, eta2=eta2)

    @staticmethod
    def from_mean_prec(
        mu: Float[Array, " N"],
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


def to_mean_cov(
    expfam: GaussianExpFam,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator]:
    """Convert natural parameters to the mean and covariance.

    The inverse of `GaussianExpFam.from_mean_cov`. These are **not** the
    expectation parameters ``(mu, mu mu^T + Sigma)``: for those, use
    `gaussx.natural_to_expectation`.

    Args:
        expfam: Gaussian in natural form.

    Returns:
        Tuple ``(mu, Sigma)`` — mean vector and covariance operator.
    """
    return natural_to_mean_cov(expfam.eta1, expfam.eta2)


def to_expectation(
    expfam: GaussianExpFam,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator]:
    """Deprecated alias of `to_mean_cov`.

    Despite its name it returns the mean and covariance ``(mu, Sigma)``,
    not the expectation parameters ``(mu, mu mu^T + Sigma)`` (see
    `gaussx.natural_to_expectation`). It will be removed in a future
    release.

    Args:
        expfam: Gaussian in natural form.

    Returns:
        Tuple ``(mu, Sigma)`` — mean vector and covariance operator.
    """
    warnings.warn(
        "to_expectation is deprecated: it returns (mu, Sigma), not expectation "
        "parameters. Use gaussx.to_mean_cov (same result), or "
        "gaussx.natural_to_expectation for (mu, mu mu^T + Sigma).",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_mean_cov(expfam)


def to_natural(
    mu: Float[Array, " N"],
    Sigma: lx.AbstractLinearOperator,
) -> tuple[Float[Array, " N"], lx.AbstractLinearOperator]:
    """Deprecated alias of `gaussx.mean_cov_to_natural`.

    It takes the mean and covariance, not expectation parameters. It will
    be removed in a future release.

    Args:
        mu: Mean vector, shape ``(N,)``.
        Sigma: Covariance operator, shape ``(N, N)``.

    Returns:
        Tuple ``(eta1, eta2)`` — natural parameters.
    """
    warnings.warn(
        "to_natural is deprecated: it takes (mu, Sigma), not expectation "
        "parameters. Use gaussx.mean_cov_to_natural (same result).",
        DeprecationWarning,
        stacklevel=2,
    )
    return mean_cov_to_natural(mu, Sigma)


def log_partition(expfam: GaussianExpFam) -> Float[Array, ""]:
    r"""Log-partition function ``A(eta)``.

    $$
    A(\eta) = -\frac{1}{4} \eta_1^T \eta_2^{-1} \eta_1
              - \frac{1}{2} \log|-2\eta_2|
    $$

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
    return quad + ld + 0.5 * N * _LOG_2PI


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


def sufficient_stats(
    x: Float[Array, "*batch N"],
) -> tuple[Float[Array, "*batch N"], Float[Array, "*batch N N"]]:
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
) -> Float[Array, ""]:
    """KL divergence ``KL(q || p)`` via the Bregman-divergence form on
    natural parameters.

    Exponential-family expression of the KL divergence in terms of the
    log-partition ``A`` and the natural parameters of ``q`` and ``p``.
    Mathematically equivalent to
    `dist_kl_divergence`.

    The current implementation evaluates the Bregman form by routing
    through `to_mean_cov` for the natural-gradient term
    ``(eta_p - eta_q)^T nabla A(eta_q)``. The second-moment contraction
    splits into a quadratic form (operator matvecs) plus
    `gaussx.trace_product`, so structured ``eta2`` / ``Sigma_q``
    operators are never materialized. The benefit relative to
    `dist_kl_divergence` is keeping the gradient flowing in
    natural-parameter space (suitable inside a natural-gradient loop).

    $$
    KL(q || p) = A(eta_p) - A(eta_q) - (eta_p - eta_q)^T nabla A(eta_q)
    $$

    Args:
        q: First Gaussian (the "true" distribution).
        p: Second Gaussian (the "approximate" distribution).

    Returns:
        Scalar KL divergence.

    See Also:
        `dist_kl_divergence`: General KL
        in mean/covariance form with lineax operators.
    """
    A_p = log_partition(p)
    A_q = log_partition(q)

    # grad A(eta_q) is the expectation parameters: mu_q w.r.t. eta1 and
    # mu_q mu_q^T + Sigma_q w.r.t. eta2. Build them from (mu_q, Sigma_q).
    # The linear term: (eta_p - eta_q)^T grad A(eta_q)
    # For eta1 part: (eta1_p - eta1_q)^T mu_q
    mu_q, Sigma_q = to_mean_cov(q)

    delta_eta1 = p.eta1 - q.eta1
    linear_eta1 = delta_eta1 @ mu_q

    # For eta2 part: tr((eta2_p - eta2_q) @ (mu mu^T + Sigma))
    # = mu^T (eta2_p - eta2_q) mu + tr(eta2_p Sigma) - tr(eta2_q Sigma).
    # Quadratic form via matvecs + structured trace_product — no
    # materialization of eta2 or Sigma_q.
    quad = mu_q @ (p.eta2.mv(mu_q) - q.eta2.mv(mu_q))
    linear_eta2 = quad + trace_product(p.eta2, Sigma_q) - trace_product(q.eta2, Sigma_q)

    return A_p - A_q - linear_eta1 - linear_eta2
