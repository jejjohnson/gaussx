"""Non-Gaussian likelihood functions for variational inference."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from gaussx._einx import rearrange
from gaussx._quadrature._likelihood import AbstractLikelihood


class BernoulliLikelihood(AbstractLikelihood):
    r"""Bernoulli likelihood with logit link.

    Args:
        y: Binary observations, shape ``(N,)``.
    """

    y: Float[Array, " N"]

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate Bernoulli log-likelihood with logit link."""
        return jnp.sum(
            self.y * jax.nn.log_sigmoid(f) + (1.0 - self.y) * jax.nn.log_sigmoid(-f)
        )


class PoissonLikelihood(AbstractLikelihood):
    r"""Poisson likelihood with log link.

    Args:
        y: Count observations, shape ``(N,)``.
    """

    y: Float[Array, " N"]

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate Poisson log-likelihood with log link."""
        return jnp.sum(self.y * f - jnp.exp(f) - jax.scipy.special.gammaln(self.y + 1))


class StudentTLikelihood(AbstractLikelihood):
    r"""Student-t likelihood for robust regression.

    Args:
        y: Observations, shape ``(N,)``.
        df: Degrees of freedom (> 0).
        scale: Scale parameter (> 0).
    """

    y: Float[Array, " N"]
    df: float
    scale: float

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate Student-t log-likelihood."""
        df = self.df
        scale = self.scale
        residual = self.y - f
        half_df = 0.5 * df
        half_dfp1 = 0.5 * (df + 1.0)

        log_norm = (
            jax.scipy.special.gammaln(half_dfp1)
            - jax.scipy.special.gammaln(half_df)
            - 0.5 * jnp.log(df * jnp.pi * scale**2)
        )
        log_kernel = -half_dfp1 * jnp.log1p(residual**2 / (df * scale**2))
        return jnp.sum(log_norm + log_kernel)


class SoftmaxLikelihood(AbstractLikelihood):
    r"""Softmax (categorical) likelihood for multi-class classification.

    Args:
        y: Integer class labels, shape ``(N,)``.
        num_classes: Number of classes C.
    """

    y: Int[Array, " N"]
    num_classes: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True, default=1)

    def __init__(self, y: Int[Array, " N"], num_classes: int):
        self.y = y
        self.num_classes = num_classes
        self.latent_dim = num_classes

    def log_prob(self, f: Float[Array, " NC"]) -> Float[Array, ""]:
        """Evaluate softmax log-likelihood."""
        f_2d = rearrange(f, "(N C) -> N C", C=self.num_classes)
        log_probs = jax.nn.log_softmax(f_2d, axis=-1)
        return jnp.sum(log_probs[jnp.arange(self.y.shape[0]), self.y])


class HeteroscedasticGaussianLikelihood(AbstractLikelihood):
    r"""Heteroscedastic Gaussian likelihood with input-dependent noise.

    Args:
        y: Observations, shape ``(N,)``.
    """

    y: Float[Array, " N"]
    latent_dim: int = eqx.field(static=True, default=2)

    def log_prob(self, f: Float[Array, " 2N"]) -> Float[Array, ""]:
        """Evaluate heteroscedastic Gaussian log-likelihood."""
        N = self.y.shape[0]
        f_mean = f[:N]
        f_noise = f[N:]
        noise_std = jax.nn.softplus(f_noise)
        noise_var = noise_std**2

        log_2pi = jnp.log(2.0 * jnp.pi)
        residual = self.y - f_mean
        return jnp.sum(-0.5 * (log_2pi + jnp.log(noise_var) + residual**2 / noise_var))
