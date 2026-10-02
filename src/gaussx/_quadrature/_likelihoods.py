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

    Attributes:
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

    Attributes:
        y: Count observations, shape ``(N,)``.
    """

    y: Float[Array, " N"]

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate Poisson log-likelihood with log link."""
        return jnp.sum(self.y * f - jnp.exp(f) - jax.scipy.special.gammaln(self.y + 1))


class BinomialLikelihood(AbstractLikelihood):
    r"""Binomial likelihood with logit link.

    $y_i \sim \operatorname{Bin}(n_i, \sigma(f_i))$, so

    $$
    \log p(y\mid f) = \sum_i \log\binom{n_i}{y_i} + y_i\log\sigma(f_i)
        + (n_i - y_i)\log\sigma(-f_i),
    $$

    with site gradient $y_i - n_i\sigma(f_i)$ and Hessian
    $-n_i\sigma(f_i)\sigma(-f_i)$ (log-concave). ``n_trials = 1`` is
    `BernoulliLikelihood`.

    Attributes:
        y: Success counts, shape ``(N,)``.
        n_trials: Numbers of trials $n_i$, a scalar or shape ``(N,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        import gaussx as gx

        lik = gx.BinomialLikelihood(jnp.array([3.0, 0.0, 5.0]), n_trials=5.0)
        lp = lik.log_prob(jnp.zeros(3))
        grad, hess = lik.site_derivatives(jnp.zeros(3))  # [0.5, -2.5, 2.5], -1.25
        ```
    """

    y: Float[Array, " N"]
    n_trials: Float[Array, " N"] | float

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate the binomial log-likelihood with logit link."""
        y, n = self.y, self.n_trials
        gammaln = jax.scipy.special.gammaln
        log_binom = gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0)
        return jnp.sum(
            log_binom + y * jax.nn.log_sigmoid(f) + (n - y) * jax.nn.log_sigmoid(-f)
        )

    def site_derivatives(
        self, f: Float[Array, " N"]
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Closed-form ``(y − nσ(f), −nσ(f)σ(−f))``."""
        p = jax.nn.sigmoid(f)
        return self.y - self.n_trials * p, -self.n_trials * p * jax.nn.sigmoid(-f)


class NegativeBinomialLikelihood(AbstractLikelihood):
    r"""Negative-binomial likelihood with log link (NB2, a gamma-Poisson mixture).

    $y_i$ has mean $\mu_i = e^{f_i}$ and variance $\mu_i + \mu_i^2/r$, with
    $r$ the ``concentration`` (R-INLA's ``size``; numpyro's
    ``NegativeBinomial2`` concentration). $r$ is the dispersion
    hyperparameter: a leaf, so it can be traced and differentiated as part
    of $\theta$. With $s_i = \sigma(f_i - \log r) = \mu_i/(r + \mu_i)$,

    $$
    \log p(y\mid f) = \sum_i \log\frac{\Gamma(y_i + r)}{\Gamma(r)\,y_i!}
        + r\log r + y_if_i - (r + y_i)\log(r + e^{f_i}),
    $$

    with site gradient $y_i - (r + y_i)s_i$ and Hessian
    $-(r + y_i)s_i(1 - s_i)$ (log-concave). $r\to\infty$ is
    `PoissonLikelihood`.

    Attributes:
        y: Count observations, shape ``(N,)``.
        concentration: $r > 0$, a scalar or shape ``(N,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        import gaussx as gx

        lik = gx.NegativeBinomialLikelihood(jnp.array([0.0, 4.0]), concentration=2.0)
        lp = lik.log_prob(jnp.log(jnp.array([1.0, 3.0])))
        grad, hess = lik.site_derivatives(jnp.zeros(2))
        ```
    """

    y: Float[Array, " N"]
    concentration: Float[Array, " N"] | float

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate the negative-binomial log-likelihood with log link."""
        y, r = self.y, self.concentration
        gammaln = jax.scipy.special.gammaln
        log_r = jnp.log(r)
        return jnp.sum(
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1.0)
            + r * log_r
            + y * f
            - (r + y) * jnp.logaddexp(log_r, f)
        )

    def site_derivatives(
        self, f: Float[Array, " N"]
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Closed-form ``(y − (r + y)s, −(r + y)s(1 − s))``, ``s = σ(f − log r)``."""
        y, r = self.y, self.concentration
        shifted = f - jnp.log(r)
        s = jax.nn.sigmoid(shifted)
        return y - (r + y) * s, -(r + y) * s * jax.nn.sigmoid(-shifted)


class StudentTLikelihood(AbstractLikelihood):
    r"""Student-t likelihood for robust regression.

    Attributes:
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

    Attributes:
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
