"""Likelihood objects for analytical dispatch in the expected log-likelihood."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._distributions._gaussian import _LOG_2PI, gaussian_log_prob
from gaussx._primitives._trace import trace


class AbstractLikelihood(eqx.Module):
    """Base class for likelihood functions with optional analytical ELL.

        log p(y ∣ f),    𝔼_q[log p(y ∣ f)]

    Subclasses that support a closed-form expected log-likelihood under a
    Gaussian variational distribution should override ``has_analytical_ell``
    to return ``True`` and implement ``analytical_expected_log_likelihood``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> likelihood = gaussx.PoissonLikelihood(jnp.array([1.0, 2.0]))
        >>> isinstance(likelihood, AbstractLikelihood)
        True
        >>> likelihood.has_analytical_ell()
        False
    """

    @abc.abstractmethod
    def log_prob(
        self,
        f: Float[Array, " N"],
    ) -> Float[Array, ""]:
        """Evaluate log p(y ∣ f) for fixed observations.

        Args:
            f: Latent function values f.

        Returns:
            The scalar log-likelihood.
        """
        ...

    def site_derivatives(
        self,
        f: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Per-site gradient and Hessian diagonal of log p(y ∣ f) in f.

            log p(y ∣ f) = ∑ᵢ log p(yᵢ ∣ fᵢ)

        For a likelihood that factorises over sites the Hessian is diagonal and
        this returns (∂ log p / ∂fᵢ, ∂² log p / ∂fᵢ²), the inputs of
        `gaussx.newton_update`'s diagonal path and of `argminax.solve.laplace_mode`. The
        default differentiates ``log_prob``: one reverse pass for the gradient
        and one forward pass of it along ``1`` for the diagonal (exact only when
        the Hessian is diagonal, so not for `SoftmaxLikelihood` or
        `HeteroscedasticGaussianLikelihood`). Subclasses may override it with
        closed forms.

        Args:
            f: Latent function values f.

        Returns:
            Tuple ``(gradient, hessian_diagonal)`` holding the two site-wise
            derivatives.

        Examples:
            A unit-variance Gaussian has gradient y − f and curvature −1:

            >>> import jax.numpy as jnp
            >>> import gaussx
            >>> likelihood = gaussx.GaussianLikelihood(jnp.array([1.0, 2.0]), 1.0)
            >>> gradient, hessian_diagonal = likelihood.site_derivatives(jnp.zeros(2))
            >>> bool(jnp.allclose(gradient, jnp.array([1.0, 2.0])))
            True
            >>> bool(jnp.allclose(hessian_diagonal, -jnp.ones(2)))
            True
        """
        # ∂ log p / ∂f
        gradient_fn = jax.grad(self.log_prob)
        # The all-ones tangent reads off the Hessian diagonal
        tangent = jnp.ones_like(f)
        return jax.jvp(gradient_fn, (f,), (tangent,))

    def has_analytical_ell(self) -> bool:
        """Whether this likelihood supports a closed-form ELL.

        Returns:
            ``True`` if ``analytical_expected_log_likelihood`` is implemented.
        """
        return False

    def analytical_expected_log_likelihood(
        self,
        q_mu: Float[Array, " N"],
        q_cov: lx.AbstractLinearOperator,
    ) -> Float[Array, ""]:
        """Closed-form 𝔼_q[log p(y ∣ f)] where q(f) = 𝒩(μ_f, Σ_f).

        Args:
            q_mu: Variational mean μ_f.
            q_cov: Variational covariance Σ_f, as an operator.

        Returns:
            The scalar expected log-likelihood.

        Raises:
            NotImplementedError: If no analytical form exists.

        Examples:
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> import gaussx
            >>> likelihood = gaussx.PoissonLikelihood(jnp.ones(2))
            >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
            >>> likelihood.analytical_expected_log_likelihood(jnp.zeros(2), covariance)
            Traceback (most recent call last):
            ...
            NotImplementedError: PoissonLikelihood has no analytical ELL
        """
        msg = f"{type(self).__name__} has no analytical ELL"
        raise NotImplementedError(msg)


class GaussianLikelihood(AbstractLikelihood):
    """Gaussian likelihood with isotropic noise, with a closed-form ELL.

        log p(y ∣ f) = log 𝒩(y; f, σ_ε² I)
        𝔼_q[log p(y ∣ f)] = log 𝒩(y; μ_f, σ_ε² I) − tr(Σ_f) / (2 σ_ε²)

    Attributes:
        y: Observed targets y.
        noise_var: Observation noise variance σ_ε².

    Examples:
        The ELL is the density at the mean less the trace correction:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> likelihood = GaussianLikelihood(jnp.zeros(2), 1.0)
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> mean = jnp.zeros(2)
        >>> ell = likelihood.analytical_expected_log_likelihood(mean, covariance)
        >>> bool(jnp.allclose(ell, likelihood.log_prob(mean) - 1.0))
        True
    """

    y: Float[Array, " N"]
    noise_var: float

    def log_prob(
        self,
        f: Float[Array, " N"],
    ) -> Float[Array, ""]:
        """Evaluate log 𝒩(y; f, σ_ε² I).

        Args:
            f: Latent function values f.

        Returns:
            The scalar log-likelihood.
        """
        obs_size = self.y.shape[-1]
        # r = y − f
        residual = self.y - f
        # D_y log 2π + D_y log σ_ε²
        normaliser = obs_size * _LOG_2PI + obs_size * jnp.log(self.noise_var)
        # ‖r‖₂² / σ_ε²
        quadratic_term = jnp.sum(residual**2) / self.noise_var
        # log 𝒩(y; f, σ_ε² I) = −½ (D_y log 2π + D_y log σ_ε² + ‖r‖₂² / σ_ε²)
        return -0.5 * (normaliser + quadratic_term)

    def has_analytical_ell(self) -> bool:
        """Report the closed-form ELL of the Gaussian likelihood.

        Returns:
            Always ``True``.
        """
        return True

    def analytical_expected_log_likelihood(
        self,
        q_mu: Float[Array, " N"],
        q_cov: lx.AbstractLinearOperator,
    ) -> Float[Array, ""]:
        """Closed-form 𝔼_q[log 𝒩(y; f, σ_ε² I)] where q(f) = 𝒩(μ_f, Σ_f).

            𝔼_q[log p(y ∣ f)] = log 𝒩(y; μ_f, Σ_ε) − ½ tr(Σ_ε⁻¹ Σ_f)

        with Σ_ε = σ_ε² I. Delegates the log-density term to
        `gaussx.gaussian_log_prob` (which exploits the diagonal noise structure)
        and computes the trace correction directly via the structural
        ``trace(q_cov) / noise_var`` shortcut, so Kronecker/BlockDiag-structured
        ``q_cov`` keeps its O(n) ``prod(trace_factor)`` / per-block ``trace``
        fast paths instead of materializing through
        ``trace_product(Σ_ε⁻¹, q_cov)``.

        Args:
            q_mu: Variational mean μ_f.
            q_cov: Variational covariance Σ_f, as an operator.

        Returns:
            The scalar expected log-likelihood.
        """
        obs_size = self.y.shape[-1]
        # Σ_ε = σ_ε² I
        obs_noise_covariance = lx.DiagonalLinearOperator(
            jnp.full(obs_size, self.noise_var)
        )
        # log 𝒩(y; μ_f, Σ_ε)
        log_density = gaussian_log_prob(q_mu, obs_noise_covariance, self.y)
        # tr(Σ_ε⁻¹ Σ_f) = tr(Σ_f) / σ_ε², with ``trace`` dispatching on the
        # structure of Σ_f (Kronecker, BlockDiag, …) through gaussx primitives
        trace_term = trace(q_cov) / self.noise_var
        # 𝔼_q[log p(y ∣ f)] = log 𝒩(y; μ_f, Σ_ε) − ½ tr(Σ_ε⁻¹ Σ_f)
        return log_density - 0.5 * trace_term
