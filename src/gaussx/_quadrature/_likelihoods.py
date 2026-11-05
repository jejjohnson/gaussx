"""Non-Gaussian likelihood functions for variational inference."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from gaussx._einx import rearrange
from gaussx._quadrature._likelihood import AbstractLikelihood


class BernoulliLikelihood(AbstractLikelihood):
    """Bernoulli likelihood with logit link.

        log p(y ∣ f) = ∑ᵢ yᵢ log σ(fᵢ) + (1 − yᵢ) log σ(−fᵢ)

    with σ the logistic sigmoid.

    Attributes:
        y: Binary observations y.

    Examples:
        At f = 0 every site contributes log ½:

        >>> import jax.numpy as jnp
        >>> likelihood = BernoulliLikelihood(jnp.array([1.0, 0.0]))
        >>> log_prob = likelihood.log_prob(jnp.zeros(2))
        >>> bool(jnp.allclose(log_prob, 2.0 * jnp.log(0.5)))
        True
    """

    y: Float[Array, " N"]

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate the Bernoulli log-likelihood with logit link.

        Args:
            f: Latent logits f.

        Returns:
            The scalar log-likelihood.
        """
        # yᵢ log σ(fᵢ)
        success_term = self.y * jax.nn.log_sigmoid(f)
        # (1 − yᵢ) log σ(−fᵢ)
        failure_term = (1.0 - self.y) * jax.nn.log_sigmoid(-f)
        # log p(y ∣ f) = ∑ᵢ yᵢ log σ(fᵢ) + (1 − yᵢ) log σ(−fᵢ)
        return jnp.sum(success_term + failure_term)


class PoissonLikelihood(AbstractLikelihood):
    """Poisson likelihood with log link.

        log p(y ∣ f) = ∑ᵢ yᵢ fᵢ − exp(fᵢ) − log Γ(yᵢ + 1)

    Attributes:
        y: Count observations y.

    Examples:
        At f = 0 a zero count contributes −1:

        >>> import jax.numpy as jnp
        >>> likelihood = PoissonLikelihood(jnp.zeros(3))
        >>> bool(jnp.allclose(likelihood.log_prob(jnp.zeros(3)), -3.0))
        True
    """

    y: Float[Array, " N"]

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate the Poisson log-likelihood with log link.

        Args:
            f: Latent log-rates f.

        Returns:
            The scalar log-likelihood.
        """
        # yᵢ fᵢ − exp(fᵢ)
        rate_term = self.y * f - jnp.exp(f)
        # log yᵢ! = log Γ(yᵢ + 1)
        log_factorial = jax.scipy.special.gammaln(self.y + 1)
        # log p(y ∣ f) = ∑ᵢ yᵢ fᵢ − exp(fᵢ) − log Γ(yᵢ + 1)
        return jnp.sum(rate_term - log_factorial)


class BinomialLikelihood(AbstractLikelihood):
    """Binomial likelihood with logit link.

        yᵢ ~ Binomial(nᵢ, σ(fᵢ))
        log p(y ∣ f) = ∑ᵢ log C(nᵢ, yᵢ) + yᵢ log σ(fᵢ) + (nᵢ − yᵢ) log σ(−fᵢ)

    with σ the logistic sigmoid and C(nᵢ, yᵢ) the binomial coefficient. The site
    gradient is yᵢ − nᵢ σ(fᵢ) and the Hessian −nᵢ σ(fᵢ) σ(−fᵢ) (log-concave).
    ``n_trials = 1`` is `BernoulliLikelihood`.

    Attributes:
        y: Success counts y.
        n_trials: Numbers of trials nᵢ, a scalar or one per site.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx as gx
        >>> lik = gx.BinomialLikelihood(jnp.array([3.0, 0.0, 5.0]), n_trials=5.0)
        >>> lik.log_prob(jnp.zeros(3)).shape
        ()
        >>> gradient, hessian_diagonal = lik.site_derivatives(jnp.zeros(3))
        >>> bool(jnp.allclose(gradient, jnp.array([0.5, -2.5, 2.5])))
        True
        >>> bool(jnp.allclose(hessian_diagonal, -1.25))
        True
    """

    y: Float[Array, " N"]
    n_trials: Float[Array, " N"] | float

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate the binomial log-likelihood with logit link.

        Args:
            f: Latent logits f.

        Returns:
            The scalar log-likelihood.
        """
        successes, trials = self.y, self.n_trials
        gammaln = jax.scipy.special.gammaln
        # log C(nᵢ, yᵢ) = log Γ(nᵢ + 1) − log Γ(yᵢ + 1) − log Γ(nᵢ − yᵢ + 1)
        log_binomial = (
            gammaln(trials + 1.0)
            - gammaln(successes + 1.0)
            - gammaln(trials - successes + 1.0)
        )
        # log C(nᵢ, yᵢ) + yᵢ log σ(fᵢ)
        success_term = log_binomial + successes * jax.nn.log_sigmoid(f)
        # (nᵢ − yᵢ) log σ(−fᵢ)
        failure_term = (trials - successes) * jax.nn.log_sigmoid(-f)
        # log p(y ∣ f) = ∑ᵢ log C(nᵢ, yᵢ) + yᵢ log σ(fᵢ) + (nᵢ − yᵢ) log σ(−fᵢ)
        return jnp.sum(success_term + failure_term)

    def site_derivatives(
        self, f: Float[Array, " N"]
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Closed-form site derivatives of the binomial log-likelihood.

            ∂ log p / ∂fᵢ = yᵢ − nᵢ σ(fᵢ)
            ∂² log p / ∂fᵢ² = −nᵢ σ(fᵢ) σ(−fᵢ)

        Args:
            f: Latent logits f.

        Returns:
            Tuple ``(gradient, hessian_diagonal)`` holding the two site-wise
            derivatives.
        """
        # σ(fᵢ)
        success_probability = jax.nn.sigmoid(f)
        # ∂ log p / ∂fᵢ = yᵢ − nᵢ σ(fᵢ)
        gradient = self.y - self.n_trials * success_probability
        # ∂² log p / ∂fᵢ² = −nᵢ σ(fᵢ) σ(−fᵢ)
        hessian_diagonal = -self.n_trials * success_probability * jax.nn.sigmoid(-f)
        return gradient, hessian_diagonal


class NegativeBinomialLikelihood(AbstractLikelihood):
    """Negative-binomial likelihood with log link (NB2, a gamma-Poisson mixture).

        log p(y ∣ f) = ∑ᵢ log(Γ(yᵢ + r) / (Γ(r) yᵢ!)) + r log r + yᵢ fᵢ
                       − (r + yᵢ) log(r + exp(fᵢ))

    yᵢ has mean μᵢ = exp(fᵢ) and variance μᵢ + μᵢ²/r, with r the
    ``concentration`` (R-INLA's ``size``; numpyro's ``NegativeBinomial2``
    concentration). r is the dispersion hyperparameter: a leaf, so it can be
    traced and differentiated as part of θ. With sᵢ = σ(fᵢ − log r) =
    μᵢ/(r + μᵢ), the site gradient is yᵢ − (r + yᵢ) sᵢ and the Hessian
    −(r + yᵢ) sᵢ (1 − sᵢ) (log-concave). r → ∞ is `PoissonLikelihood`.

    Attributes:
        y: Count observations y.
        concentration: Concentration r > 0, a scalar or one per site.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx as gx
        >>> counts = jnp.array([0.0, 4.0])
        >>> lik = gx.NegativeBinomialLikelihood(counts, concentration=2.0)
        >>> lik.log_prob(jnp.log(jnp.array([1.0, 3.0]))).shape
        ()
        >>> gradient, hessian_diagonal = lik.site_derivatives(jnp.zeros(2))
        >>> bool(jnp.allclose(gradient, jnp.array([-2.0 / 3.0, 2.0])))
        True
        >>> bool(jnp.all(hessian_diagonal < 0.0))
        True
    """

    y: Float[Array, " N"]
    concentration: Float[Array, " N"] | float

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate the negative-binomial log-likelihood with log link.

        Args:
            f: Latent log-means f.

        Returns:
            The scalar log-likelihood.
        """
        counts, concentration = self.y, self.concentration
        gammaln = jax.scipy.special.gammaln
        # log r
        log_concentration = jnp.log(concentration)
        # log(Γ(yᵢ + r) / (Γ(r) yᵢ!)) + r log r
        log_normaliser = (
            gammaln(counts + concentration)
            - gammaln(concentration)
            - gammaln(counts + 1.0)
            + concentration * log_concentration
        )
        # log(Γ(yᵢ + r) / (Γ(r) yᵢ!)) + r log r + yᵢ fᵢ
        linear_term = log_normaliser + counts * f
        # (r + yᵢ) log(r + exp(fᵢ))
        saturation_term = (concentration + counts) * jnp.logaddexp(log_concentration, f)
        # log p(y ∣ f) = ∑ᵢ log(Γ(yᵢ + r) / (Γ(r) yᵢ!)) + r log r + yᵢ fᵢ
        #                − (r + yᵢ) log(r + exp(fᵢ))
        return jnp.sum(linear_term - saturation_term)

    def site_derivatives(
        self, f: Float[Array, " N"]
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Closed-form site derivatives of the negative-binomial log-likelihood.

            ∂ log p / ∂fᵢ = yᵢ − (r + yᵢ) sᵢ
            ∂² log p / ∂fᵢ² = −(r + yᵢ) sᵢ (1 − sᵢ),    sᵢ = σ(fᵢ − log r)

        Args:
            f: Latent log-means f.

        Returns:
            Tuple ``(gradient, hessian_diagonal)`` holding the two site-wise
            derivatives.
        """
        counts, concentration = self.y, self.concentration
        # fᵢ − log r
        shifted = f - jnp.log(concentration)
        # sᵢ = σ(fᵢ − log r)
        success_probability = jax.nn.sigmoid(shifted)
        # ∂ log p / ∂fᵢ = yᵢ − (r + yᵢ) sᵢ
        gradient = counts - (concentration + counts) * success_probability
        # ∂² log p / ∂fᵢ² = −(r + yᵢ) sᵢ (1 − sᵢ), with 1 − sᵢ = σ(log r − fᵢ)
        hessian_diagonal = (
            -(concentration + counts) * success_probability * jax.nn.sigmoid(-shifted)
        )
        return gradient, hessian_diagonal


class StudentTLikelihood(AbstractLikelihood):
    """Student-t likelihood for robust regression.

        log p(y ∣ f) = ∑ᵢ log Γ((df + 1)/2) − log Γ(df/2) − ½ log(df π scale²)
                       − ((df + 1)/2) log(1 + (yᵢ − fᵢ)² / (df scale²))

    Attributes:
        y: Observations y.
        df: Degrees of freedom (> 0).
        scale: Scale parameter (> 0).

    Examples:
        One degree of freedom is the Cauchy density, 1/π at the centre:

        >>> import jax.numpy as jnp
        >>> likelihood = StudentTLikelihood(jnp.zeros(1), df=1.0, scale=1.0)
        >>> log_prob = likelihood.log_prob(jnp.zeros(1))
        >>> bool(jnp.allclose(log_prob, -jnp.log(jnp.pi)))
        True
    """

    y: Float[Array, " N"]
    df: float
    scale: float

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        """Evaluate the Student-t log-likelihood.

        Args:
            f: Latent locations f.

        Returns:
            The scalar log-likelihood.
        """
        df = self.df
        scale = self.scale
        # rᵢ = yᵢ − fᵢ
        residual = self.y - f
        # df / 2
        half_df = 0.5 * df
        # (df + 1) / 2
        half_df_plus_one = 0.5 * (df + 1.0)

        # log Γ((df + 1)/2) − log Γ(df/2) − ½ log(df π scale²)
        log_normaliser = (
            jax.scipy.special.gammaln(half_df_plus_one)
            - jax.scipy.special.gammaln(half_df)
            - 0.5 * jnp.log(df * jnp.pi * scale**2)
        )
        # −((df + 1)/2) log(1 + rᵢ² / (df scale²))
        log_kernel = -half_df_plus_one * jnp.log1p(residual**2 / (df * scale**2))
        # log p(y ∣ f) = ∑ᵢ log normaliser + log kernelᵢ
        return jnp.sum(log_normaliser + log_kernel)


class SoftmaxLikelihood(AbstractLikelihood):
    """Softmax (categorical) likelihood for multi-class classification.

        log p(y ∣ f) = ∑ᵢ (f_i,yᵢ − log ∑_c exp(f_i,c))

    with the latents f_i,c of C classes per site, flattened site by site.

    Args:
        y: Integer class labels yᵢ.
        num_classes: Number of classes C.

    Attributes:
        latent_dim: Latents per site, equal to ``num_classes``.

    Examples:
        Zero logits make every class equally likely:

        >>> import jax.numpy as jnp
        >>> likelihood = SoftmaxLikelihood(jnp.array([0, 2]), num_classes=3)
        >>> likelihood.latent_dim
        3
        >>> log_prob = likelihood.log_prob(jnp.zeros(6))
        >>> bool(jnp.allclose(log_prob, 2.0 * jnp.log(1.0 / 3.0)))
        True
    """

    y: Int[Array, " N"]
    num_classes: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True, default=1)

    def __init__(self, y: Int[Array, " N"], num_classes: int):
        self.y = y
        self.num_classes = num_classes
        self.latent_dim = num_classes

    def log_prob(self, f: Float[Array, " NC"]) -> Float[Array, ""]:
        """Evaluate the softmax log-likelihood.

        Args:
            f: Flattened latents f_i,c, site by site.

        Returns:
            The scalar log-likelihood.
        """
        # The axis names stay: einx prints this pattern when a size mismatch raises.
        latents = rearrange(f, "(N C) -> N C", C=self.num_classes)
        # log p(c ∣ fᵢ) = f_i,c − log ∑_c′ exp(f_i,c′)
        log_probabilities = jax.nn.log_softmax(latents, axis=-1)
        site_indices = jnp.arange(self.y.shape[0])
        # log p(yᵢ ∣ fᵢ)
        site_log_probabilities = log_probabilities[site_indices, self.y]
        # log p(y ∣ f) = ∑ᵢ log p(yᵢ ∣ fᵢ)
        return jnp.sum(site_log_probabilities)


class HeteroscedasticGaussianLikelihood(AbstractLikelihood):
    """Heteroscedastic Gaussian likelihood with input-dependent noise.

        log p(y ∣ f) = ∑ᵢ −½ (log 2π + log σᵢ² + (yᵢ − f_mean,i)² / σᵢ²)
        σᵢ = softplus(f_noise,i)

    with the latents f = (f_mean, f_noise) concatenated.

    Attributes:
        y: Observations y.
        latent_dim: Latents per site, ``2``.

    Examples:
        At zero residual and unit noise each site gives −½ log 2π:

        >>> import jax.numpy as jnp
        >>> likelihood = HeteroscedasticGaussianLikelihood(jnp.zeros(1))
        >>> unit_noise_latent = jnp.log(jnp.expm1(1.0))
        >>> f = jnp.array([0.0, unit_noise_latent])
        >>> log_prob = likelihood.log_prob(f)
        >>> bool(jnp.allclose(log_prob, -0.5 * jnp.log(2.0 * jnp.pi)))
        True
    """

    y: Float[Array, " N"]
    latent_dim: int = eqx.field(static=True, default=2)

    def log_prob(self, f: Float[Array, " 2N"]) -> Float[Array, ""]:
        """Evaluate the heteroscedastic Gaussian log-likelihood.

        Args:
            f: Concatenated latents (f_mean, f_noise).

        Returns:
            The scalar log-likelihood.
        """
        obs_size = self.y.shape[0]
        latent_mean = f[:obs_size]
        latent_noise = f[obs_size:]
        # σᵢ = softplus(f_noise,i)
        noise_std = jax.nn.softplus(latent_noise)
        # σᵢ²
        noise_var = noise_std**2

        # log 2π
        log_2pi = jnp.log(2.0 * jnp.pi)
        # rᵢ = yᵢ − f_mean,i
        residual = self.y - latent_mean
        # log 2π + log σᵢ²
        normaliser = log_2pi + jnp.log(noise_var)
        # log p(yᵢ ∣ fᵢ) = −½ (log 2π + log σᵢ² + rᵢ² / σᵢ²)
        site_log_densities = -0.5 * (normaliser + residual**2 / noise_var)
        # log p(y ∣ f) = ∑ᵢ log p(yᵢ ∣ fᵢ)
        return jnp.sum(site_log_densities)
