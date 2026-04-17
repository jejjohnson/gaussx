"""Tests for GaussHermiteIntegrator."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._quadrature._gauss_hermite import GaussHermiteIntegrator
from gaussx._quadrature._taylor import TaylorIntegrator
from gaussx._quadrature._types import GaussianState


def _make_state():
    mean = jnp.array([1.0, 2.0])
    cov = lx.MatrixLinearOperator(
        jnp.array([[1.0, 0.3], [0.3, 0.5]]),
        lx.positive_semidefinite_tag,
    )
    return GaussianState(mean=mean, cov=cov)


def _linear_fn(x):
    A = jnp.array([[2.0, 1.0], [0.0, 3.0]])
    b = jnp.array([1.0, -1.0])
    return A @ x + b


class TestGaussHermiteIntegrator:
    def test_linear_fn_exact_mean(self):
        """GH should be exact for linear functions (mean)."""
        state = _make_state()
        integrator = GaussHermiteIntegrator(order=5)
        result = integrator.integrate(_linear_fn, state)

        expected_mean = _linear_fn(state.mean)
        assert jnp.allclose(result.state.mean, expected_mean, atol=1e-6)

    def test_linear_fn_exact_cov(self):
        """GH should be exact for linear functions (covariance)."""
        state = _make_state()
        integrator = GaussHermiteIntegrator(order=5)
        result = integrator.integrate(_linear_fn, state)

        A = jnp.array([[2.0, 1.0], [0.0, 3.0]])
        Sigma = state.cov.as_matrix()
        expected_cov = A @ Sigma @ A.T
        assert jnp.allclose(result.state.cov.as_matrix(), expected_cov, atol=1e-5)

    def test_cross_covariance(self):
        """Should compute input-output cross-covariance."""
        state = _make_state()
        integrator = GaussHermiteIntegrator(order=5)
        result = integrator.integrate(_linear_fn, state)

        assert result.cross_cov is not None
        A = jnp.array([[2.0, 1.0], [0.0, 3.0]])
        Sigma = state.cov.as_matrix()
        expected_cross = Sigma @ A.T
        assert jnp.allclose(result.cross_cov, expected_cross, atol=1e-5)

    def test_polynomial_exact(self):
        """GH with sufficient order should be exact for polynomials.

        E[x^2] = mu^2 + sigma^2, so for a 1D quadratic
        with order >= 2, the mean should be exact.
        """
        mean = jnp.array([3.0])
        cov = lx.MatrixLinearOperator(
            jnp.array([[2.0]]),
            lx.positive_semidefinite_tag,
        )
        state = GaussianState(mean=mean, cov=cov)
        integrator = GaussHermiteIntegrator(order=5)

        def quadratic_fn(x):
            return jnp.array([x[0] ** 2])

        result = integrator.integrate(quadratic_fn, state)
        # E[x^2] = mu^2 + sigma^2 = 9 + 2 = 11
        assert jnp.allclose(result.state.mean[0], 11.0, atol=1e-6)

    def test_matches_unscented_on_smooth(self):
        """GH should roughly match Taylor on linear functions."""
        state = _make_state()
        gh = GaussHermiteIntegrator(order=10)
        taylor = TaylorIntegrator(order=1)

        gh_result = gh.integrate(_linear_fn, state)
        taylor_result = taylor.integrate(_linear_fn, state)

        assert jnp.allclose(gh_result.state.mean, taylor_result.state.mean, atol=1e-5)
        assert jnp.allclose(
            gh_result.state.cov.as_matrix(),
            taylor_result.state.cov.as_matrix(),
            atol=1e-4,
        )

    def test_output_shapes(self):
        """Output dimension can differ from input."""
        state = _make_state()
        integrator = GaussHermiteIntegrator(order=5)

        def expand_fn(x):
            return jnp.array([x[0], x[1], x[0] + x[1]])

        result = integrator.integrate(expand_fn, state)
        assert result.state.mean.shape == (3,)
        assert result.state.cov.as_matrix().shape == (3, 3)
        assert result.cross_cov.shape == (2, 3)

    def test_jit(self):
        """Should be JIT-compatible."""
        state = _make_state()
        integrator = GaussHermiteIntegrator(order=5)
        result1 = integrator.integrate(_linear_fn, state)
        jitted = jax.jit(integrator.integrate, static_argnums=(0,))
        result2 = jitted(_linear_fn, state)
        assert jnp.allclose(result1.state.mean, result2.state.mean, atol=1e-10)

    def test_1d_integration(self):
        """Should work for 1D inputs."""
        mean = jnp.array([0.0])
        cov = lx.MatrixLinearOperator(
            jnp.array([[1.0]]),
            lx.positive_semidefinite_tag,
        )
        state = GaussianState(mean=mean, cov=cov)
        integrator = GaussHermiteIntegrator(order=20)

        # E[x] = 0 under N(0,1)
        result = integrator.integrate(lambda x: x, state)
        assert jnp.allclose(result.state.mean[0], 0.0, atol=1e-10)
