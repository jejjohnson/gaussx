"""Tests for Gaussian expectation functions."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._quadrature._expectations import (
    cost_expectation,
    gradient_expectation,
    log_likelihood_expectation,
    mean_expectation,
)
from gaussx._quadrature._monte_carlo import MonteCarloIntegrator
from gaussx._quadrature._taylor import TaylorIntegrator
from gaussx._quadrature._types import GaussianState


def _make_state():
    mean = jnp.array([1.0, 2.0])
    cov = lx.MatrixLinearOperator(
        jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        lx.positive_semidefinite_tag,
    )
    return GaussianState(mean=mean, cov=cov)


class TestMeanExpectation:
    def test_linear_fn(self):
        """E[Ax + b] = A @ mu + b for linear functions."""
        state = _make_state()
        integrator = TaylorIntegrator(order=1)

        A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        b = jnp.array([1.0, -1.0])
        fn = lambda x: A @ x + b

        result = mean_expectation(fn, state, integrator)
        expected = A @ state.mean + b
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_output_shape(self):
        """Output shape should match function output."""
        state = _make_state()
        integrator = TaylorIntegrator(order=1)

        fn = lambda x: jnp.array([x[0], x[1], x[0] + x[1]])
        result = mean_expectation(fn, state, integrator)
        assert result.shape == (3,)


class TestGradientExpectation:
    def test_linear_fn(self):
        """E[nabla (a^T x)] = a for linear functions."""
        state = _make_state()
        integrator = TaylorIntegrator(order=1)

        a = jnp.array([3.0, -1.0])
        fn = lambda x: jnp.dot(a, x)

        result = gradient_expectation(fn, state, integrator)
        assert jnp.allclose(result, a, atol=1e-6)

    def test_quadratic_fn_mc(self):
        """E[nabla x^2] = 2 * mu for quadratic via MC."""
        state = _make_state()
        integrator = MonteCarloIntegrator(n_samples=20000, key=jax.random.key(42))

        fn = lambda x: jnp.sum(x**2)
        result = gradient_expectation(fn, state, integrator)
        expected = 2.0 * state.mean
        assert jnp.allclose(result, expected, atol=0.2)


class TestLogLikelihoodExpectation:
    def test_returns_scalar(self):
        """Should return a finite scalar."""
        state = _make_state()
        integrator = TaylorIntegrator(order=1)

        def log_lik(x):
            return -0.5 * jnp.sum(x**2)

        result = log_likelihood_expectation(log_lik, state, integrator)
        assert result.shape == ()
        assert jnp.isfinite(result)


class TestCostExpectation:
    def test_quadratic_cost(self):
        """Expected quadratic cost should be finite and positive."""
        state = _make_state()
        integrator = TaylorIntegrator(order=1)

        pred_fn = lambda x: x  # identity prediction
        cost_fn = lambda pred, target: jnp.sum((pred - target) ** 2)
        target = jnp.zeros(2)

        result = cost_expectation(pred_fn, cost_fn, state, target, integrator)
        assert result.shape == ()
        assert jnp.isfinite(result)
        assert result > 0  # state mean != target
