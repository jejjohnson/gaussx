"""Tests for unified expected_log_likelihood and elbo (#54, #55)."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest

from gaussx._uncertain._expectations import (
    elbo,
    expected_log_likelihood,
    log_likelihood_expectation,
)
from gaussx._uncertain._gauss_hermite import GaussHermiteIntegrator
from gaussx._uncertain._likelihood import (
    AbstractLikelihood,
    GaussianLikelihood,
)
from gaussx._uncertain._taylor import TaylorIntegrator
from gaussx._uncertain._types import GaussianState


def _make_state():
    mean = jnp.array([1.0, 2.0])
    cov = lx.MatrixLinearOperator(
        jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        lx.positive_semidefinite_tag,
    )
    return GaussianState(mean=mean, cov=cov)


class TestGaussianLikelihood:
    def test_log_prob_shape(self):
        """log_prob should return a scalar."""
        y = jnp.array([1.0, 2.0])
        lik = GaussianLikelihood(y=y, noise_var=0.1)
        f = jnp.array([1.1, 1.9])
        result = lik.log_prob(f)
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_log_prob_value(self):
        """log_prob should match manual Gaussian log-density."""
        y = jnp.array([1.0, 2.0])
        noise_var = 0.5
        lik = GaussianLikelihood(y=y, noise_var=noise_var)
        f = jnp.array([1.0, 2.0])  # exact match

        result = lik.log_prob(f)
        # log N(y|f, sigma^2 I) with y==f: -0.5 * N * log(2*pi*sigma^2)
        N = 2
        expected = -0.5 * N * jnp.log(2 * jnp.pi * noise_var)
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_has_analytical_ell(self):
        """GaussianLikelihood should support analytical ELL."""
        lik = GaussianLikelihood(y=jnp.array([1.0]), noise_var=0.1)
        assert lik.has_analytical_ell()


class TestExpectedLogLikelihood:
    def test_gaussian_analytical(self):
        """Analytical ELL should match numerical GH integration."""
        state = _make_state()
        y = jnp.array([1.5, 2.5])
        noise_var = 0.5
        lik = GaussianLikelihood(y=y, noise_var=noise_var)

        # Analytical path (no integrator needed)
        ell_analytical = expected_log_likelihood(lik, state)

        # Numerical path via GH quadrature
        integrator = GaussHermiteIntegrator(order=20)
        ell_numerical = log_likelihood_expectation(lik.log_prob, state, integrator)

        assert jnp.allclose(ell_analytical, ell_numerical, atol=1e-4)

    def test_gaussian_analytical_no_integrator(self):
        """Gaussian likelihood should work without an integrator."""
        state = _make_state()
        lik = GaussianLikelihood(y=jnp.array([1.0, 2.0]), noise_var=0.1)
        result = expected_log_likelihood(lik, state)
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_non_conjugate_requires_integrator(self):
        """Non-conjugate likelihood without integrator should raise."""
        state = _make_state()

        class BernoulliLikelihood(AbstractLikelihood):
            y: jnp.ndarray

            def log_prob(self, f):
                p = jax.nn.sigmoid(f)
                return jnp.sum(self.y * jnp.log(p) + (1 - self.y) * jnp.log(1 - p))

        lik = BernoulliLikelihood(y=jnp.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="no analytical ELL"):
            expected_log_likelihood(lik, state)

    def test_non_conjugate_with_integrator(self):
        """Non-conjugate likelihood should work with an integrator."""
        state = _make_state()

        class QuadLikelihood(AbstractLikelihood):
            def log_prob(self, f):
                return -0.5 * jnp.sum(f**2)

        lik = QuadLikelihood()
        integrator = TaylorIntegrator(order=1)
        result = expected_log_likelihood(lik, state, integrator)
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_analytical_matches_sugar(self):
        """Analytical ELL should match the existing sugar function."""
        from gaussx._sugar._inference import gaussian_expected_log_lik

        state = _make_state()
        y = jnp.array([1.5, 2.5])
        noise_var = 0.5
        lik = GaussianLikelihood(y=y, noise_var=noise_var)

        ell_unified = expected_log_likelihood(lik, state)

        noise_op = lx.MatrixLinearOperator(
            noise_var * jnp.eye(2),
            lx.positive_semidefinite_tag,
        )
        ell_sugar = gaussian_expected_log_lik(y, state.mean, state.cov, noise_op)

        assert jnp.allclose(ell_unified, ell_sugar, atol=1e-6)


class TestElbo:
    def test_gaussian_elbo(self):
        """ELBO = ELL - KL for Gaussian likelihood."""
        state = _make_state()
        y = jnp.array([1.5, 2.5])
        lik = GaussianLikelihood(y=y, noise_var=0.5)
        kl = jnp.array(0.5)

        result = elbo(lik, state, kl)
        ell = expected_log_likelihood(lik, state)
        assert jnp.allclose(result, ell - kl, atol=1e-10)

    def test_elbo_with_integrator(self):
        """ELBO should work with numerical integration."""
        state = _make_state()

        class QuadLikelihood(AbstractLikelihood):
            def log_prob(self, f):
                return -0.5 * jnp.sum(f**2)

        lik = QuadLikelihood()
        kl = jnp.array(1.0)
        integrator = GaussHermiteIntegrator(order=10)

        result = elbo(lik, state, kl, integrator)
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_elbo_less_than_ell(self):
        """ELBO should be less than ELL for positive KL."""
        state = _make_state()
        lik = GaussianLikelihood(y=jnp.array([1.0, 2.0]), noise_var=0.1)
        kl = jnp.array(2.0)

        ell = expected_log_likelihood(lik, state)
        result = elbo(lik, state, kl)
        assert result < ell
