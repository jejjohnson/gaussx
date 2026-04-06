"""Likelihood objects for analytical dispatch in expected log-likelihood."""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._sugar._gaussian import _LOG_2PI


class AbstractLikelihood(eqx.Module):
    """Base class for likelihood functions with optional analytical ELL.

    Subclasses that support closed-form expected log-likelihood under
    a Gaussian variational distribution should override
    ``analytical_expected_log_likelihood``.
    """

    @abc.abstractmethod
    def log_prob(
        self,
        f: Float[Array, " N"],
    ) -> Float[Array, ""]:
        """Evaluate ``log p(y | f)`` for fixed observations.

        Args:
            f: Latent function values, shape ``(N,)``.

        Returns:
            Scalar log-likelihood.
        """
        ...

    def has_analytical_ell(self) -> bool:
        """Whether this likelihood supports closed-form ELL."""
        return False

    def analytical_expected_log_likelihood(
        self,
        q_mu: Float[Array, " N"],
        q_cov: lx.AbstractLinearOperator,
    ) -> Float[Array, ""]:
        """Closed-form ``E_q[log p(y | f)]`` where ``q = N(q_mu, q_cov)``.

        Args:
            q_mu: Variational mean, shape ``(N,)``.
            q_cov: Variational covariance operator, shape ``(N, N)``.

        Returns:
            Scalar expected log-likelihood.

        Raises:
            NotImplementedError: If no analytical form exists.
        """
        msg = f"{type(self).__name__} has no analytical ELL"
        raise NotImplementedError(msg)


class GaussianLikelihood(AbstractLikelihood):
    r"""Gaussian likelihood ``log N(y | f, noise_var * I)``.

    Supports closed-form expected log-likelihood::

        E_q[log N(y | f, \sigma^2 I)]
            = log N(y | q_\mu, \sigma^2 I)
              - 0.5 / \sigma^2 \cdot tr(q_{cov})

    Args:
        y: Observed targets, shape ``(N,)``.
        noise_var: Observation noise variance (scalar).
    """

    y: Float[Array, " N"]
    noise_var: float

    def log_prob(
        self,
        f: Float[Array, " N"],
    ) -> Float[Array, ""]:
        """Evaluate ``log N(y | f, noise_var * I)``."""
        N = self.y.shape[-1]
        residual = self.y - f
        return -0.5 * (
            N * _LOG_2PI
            + N * jnp.log(self.noise_var)
            + jnp.sum(residual**2) / self.noise_var
        )

    def has_analytical_ell(self) -> bool:
        """Gaussian likelihood has closed-form ELL."""
        return True

    def analytical_expected_log_likelihood(
        self,
        q_mu: Float[Array, " N"],
        q_cov: lx.AbstractLinearOperator,
    ) -> Float[Array, ""]:
        r"""Closed-form ``E_q[log N(y | f, \sigma^2 I)]``.

        Uses::

            E_q[log N(y|f,R)] = log N(y | q_mu, R) - 0.5 tr(R^{-1} q_cov)

        where ``R = noise_var * I``.
        """
        from gaussx._primitives._trace import trace

        N = self.y.shape[-1]
        residual = self.y - q_mu
        quad = jnp.sum(residual**2) / self.noise_var
        log_noise = N * jnp.log(self.noise_var)
        tr_term = trace(q_cov) / self.noise_var
        return -0.5 * (N * _LOG_2PI + log_noise + quad + tr_term)
