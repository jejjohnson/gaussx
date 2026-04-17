"""Tests for integrator implementations."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._quadrature._adf import AssumedDensityFilter
from gaussx._quadrature._monte_carlo import MonteCarloIntegrator
from gaussx._quadrature._taylor import TaylorIntegrator
from gaussx._quadrature._types import GaussianState
from gaussx._quadrature._unscented import UnscentedIntegrator


def _linear_fn(x):
    """Linear function: f(x) = A @ x + b."""
    A = jnp.array([[2.0, 1.0], [0.0, 3.0]])
    b = jnp.array([1.0, -1.0])
    return A @ x + b


def _make_state():
    """Create a simple 2D Gaussian state."""
    mean = jnp.array([1.0, 2.0])
    cov = lx.MatrixLinearOperator(
        jnp.array([[1.0, 0.3], [0.3, 0.5]]),
        lx.positive_semidefinite_tag,
    )
    return GaussianState(mean=mean, cov=cov)


class TestTaylorIntegrator:
    def test_linear_fn_exact_mean(self):
        """Taylor should be exact for linear functions (mean)."""
        state = _make_state()
        integrator = TaylorIntegrator(order=1)
        result = integrator.integrate(_linear_fn, state)

        expected_mean = _linear_fn(state.mean)
        assert jnp.allclose(result.state.mean, expected_mean, atol=1e-10)

    def test_linear_fn_exact_cov(self):
        """Taylor should be exact for linear functions (covariance)."""
        state = _make_state()
        integrator = TaylorIntegrator(order=1)
        result = integrator.integrate(_linear_fn, state)

        A = jnp.array([[2.0, 1.0], [0.0, 3.0]])
        Sigma = state.cov.as_matrix()
        expected_cov = A @ Sigma @ A.T
        assert jnp.allclose(result.state.cov.as_matrix(), expected_cov, atol=1e-10)

    def test_cross_covariance(self):
        """Should compute input-output cross-covariance."""
        state = _make_state()
        integrator = TaylorIntegrator(order=1)
        result = integrator.integrate(_linear_fn, state)

        assert result.cross_cov is not None
        A = jnp.array([[2.0, 1.0], [0.0, 3.0]])
        Sigma = state.cov.as_matrix()
        expected_cross = Sigma @ A.T
        assert jnp.allclose(result.cross_cov, expected_cross, atol=1e-10)

    def test_second_order(self):
        """Second-order Taylor should handle quadratic corrections."""
        state = _make_state()
        integrator = TaylorIntegrator(order=2)

        def quadratic_fn(x):
            return jnp.array([x[0] ** 2, x[1]])

        result = integrator.integrate(quadratic_fn, state)
        # Mean should include Hessian correction for x[0]^2
        # E[x^2] = mu^2 + sigma^2, correction = 0.5 * tr(H @ Sigma)
        # H = [[2, 0], [0, 0]] for first output
        correction = 0.5 * 2.0 * state.cov.as_matrix()[0, 0]
        expected_mean_0 = state.mean[0] ** 2 + correction
        assert jnp.allclose(result.state.mean[0], expected_mean_0, atol=1e-6)

    def test_second_order_covariance(self):
        """2nd-order Taylor includes Hessian covariance correction by default."""
        state = GaussianState(
            mean=jnp.array([1.0]),
            cov=lx.MatrixLinearOperator(
                jnp.array([[0.5]]),
                lx.positive_semidefinite_tag,
            ),
        )
        integrator = TaylorIntegrator(order=2)

        def quadratic_fn(x):
            return jnp.array([x[0] ** 2])

        result = integrator.integrate(quadratic_fn, state)

        expected_cov = jnp.array([[2.5]])
        assert jnp.allclose(result.state.cov.as_matrix(), expected_cov, atol=1e-6)

    def test_second_order_mean_only_same_cov_as_first(self):
        """order=2 + correct_variance=False matches order=1 covariance."""
        state = _make_state()

        def quadratic_fn(x):
            return jnp.array([x[0] ** 2, x[1]])

        result_1st = TaylorIntegrator(order=1).integrate(quadratic_fn, state)
        result_2nd = TaylorIntegrator(order=2, correct_variance=False).integrate(
            quadratic_fn, state
        )

        assert jnp.allclose(
            result_2nd.state.cov.as_matrix(),
            result_1st.state.cov.as_matrix(),
            atol=1e-10,
        )

    def test_second_order_default_larger_than_first(self):
        """order=2 default gives larger variance than first-order."""
        state = _make_state()

        def quadratic_fn(x):
            return jnp.array([x[0] ** 2, x[1]])

        result_1st = TaylorIntegrator(order=1).integrate(quadratic_fn, state)
        result_2nd = TaylorIntegrator(order=2).integrate(quadratic_fn, state)

        # Diagonal variances should be >= first-order (correction is PSD)
        diag_1st = jnp.diag(result_1st.state.cov.as_matrix())
        diag_2nd = jnp.diag(result_2nd.state.cov.as_matrix())
        assert jnp.all(diag_2nd >= diag_1st - 1e-10)

    def test_jit(self):
        """Should be JIT-compatible."""
        state = _make_state()
        integrator = TaylorIntegrator(order=1)

        result1 = integrator.integrate(_linear_fn, state)
        jitted = jax.jit(integrator.integrate, static_argnums=(0,))
        result2 = jitted(_linear_fn, state)
        assert jnp.allclose(result1.state.mean, result2.state.mean, atol=1e-10)


class TestUnscentedIntegrator:
    def test_linear_fn_exact_mean(self):
        """Unscented should be exact for linear functions (mean)."""
        state = _make_state()
        integrator = UnscentedIntegrator(alpha=1.0, beta=0.0, kappa=0.0)
        result = integrator.integrate(_linear_fn, state)

        expected_mean = _linear_fn(state.mean)
        assert jnp.allclose(result.state.mean, expected_mean, atol=1e-6)

    def test_linear_fn_exact_cov(self):
        """Unscented should be exact for linear functions (covariance)."""
        state = _make_state()
        integrator = UnscentedIntegrator(alpha=1.0, beta=0.0, kappa=0.0)
        result = integrator.integrate(_linear_fn, state)

        A = jnp.array([[2.0, 1.0], [0.0, 3.0]])
        Sigma = state.cov.as_matrix()
        expected_cov = A @ Sigma @ A.T
        assert jnp.allclose(result.state.cov.as_matrix(), expected_cov, atol=1e-5)

    def test_cross_covariance_shape(self):
        """Cross-covariance should have correct shape."""
        state = _make_state()
        integrator = UnscentedIntegrator()
        result = integrator.integrate(_linear_fn, state)
        assert result.cross_cov.shape == (2, 2)

    def test_output_shapes(self):
        """Output dimension can differ from input."""
        state = _make_state()
        integrator = UnscentedIntegrator()

        def expand_fn(x):
            return jnp.array([x[0], x[1], x[0] + x[1]])

        result = integrator.integrate(expand_fn, state)
        assert result.state.mean.shape == (3,)
        assert result.state.cov.as_matrix().shape == (3, 3)
        assert result.cross_cov.shape == (2, 3)

    def test_jit(self):
        """Should be JIT-compatible."""
        state = _make_state()
        integrator = UnscentedIntegrator()
        result1 = integrator.integrate(_linear_fn, state)
        jitted = jax.jit(integrator.integrate, static_argnums=(0,))
        result2 = jitted(_linear_fn, state)
        assert jnp.allclose(result1.state.mean, result2.state.mean, atol=1e-10)


class TestMonteCarloIntegrator:
    def test_linear_fn_approx_mean(self):
        """MC should approximate linear function mean well."""
        state = _make_state()
        integrator = MonteCarloIntegrator(n_samples=10000, key=jax.random.key(42))
        result = integrator.integrate(_linear_fn, state)

        expected_mean = _linear_fn(state.mean)
        assert jnp.allclose(result.state.mean, expected_mean, atol=0.1)

    def test_output_shapes(self):
        """Should produce correct output shapes."""
        state = _make_state()
        integrator = MonteCarloIntegrator(n_samples=100, key=jax.random.key(0))
        result = integrator.integrate(_linear_fn, state)

        assert result.state.mean.shape == (2,)
        assert result.state.cov.as_matrix().shape == (2, 2)
        assert result.cross_cov.shape == (2, 2)

    def test_cross_covariance(self):
        """Cross-covariance should exist and be finite."""
        state = _make_state()
        integrator = MonteCarloIntegrator(n_samples=1000, key=jax.random.key(1))
        result = integrator.integrate(_linear_fn, state)
        assert result.cross_cov is not None
        assert jnp.all(jnp.isfinite(result.cross_cov))

    def test_jit(self):
        """Should be JIT-compatible via eqx.filter_jit."""
        state = _make_state()
        integrator = MonteCarloIntegrator(n_samples=100, key=jax.random.key(0))
        result1 = integrator.integrate(_linear_fn, state)

        @eqx.filter_jit
        def run(integ, s):
            return integ.integrate(_linear_fn, s)

        result2 = run(integrator, state)
        assert jnp.allclose(result1.state.mean, result2.state.mean, atol=1e-10)


class TestAssumedDensityFilter:
    def test_basic_integration(self):
        """ADF should produce valid output moments."""
        state = _make_state()
        integrator = AssumedDensityFilter(n_samples=5000, key=jax.random.key(42))
        result = integrator.integrate(_linear_fn, state)

        assert result.state.mean.shape == (2,)
        assert jnp.all(jnp.isfinite(result.state.mean))
        # Covariance should be PSD
        eigvals = jnp.linalg.eigvalsh(result.state.cov.as_matrix())
        assert jnp.all(eigvals > 0)

    def test_diagnostics(self):
        """integrate_with_diagnostics should return diagnostic dict."""
        state = _make_state()
        integrator = AssumedDensityFilter(n_samples=2000, key=jax.random.key(0))
        _result, diagnostics = integrator.integrate_with_diagnostics(_linear_fn, state)

        assert diagnostics is not None
        assert "skewness" in diagnostics
        assert "kurtosis" in diagnostics
        assert "min_eigval" in diagnostics
        assert "condition_number" in diagnostics

    def test_linear_fn_gaussian_diagnostics(self):
        """For linear functions, kurtosis should be close to 3 (Gaussian)."""
        state = _make_state()
        integrator = AssumedDensityFilter(n_samples=50000, key=jax.random.key(123))
        _, diagnostics = integrator.integrate_with_diagnostics(_linear_fn, state)
        # Gaussian kurtosis = 3
        assert jnp.allclose(diagnostics["kurtosis"], 3.0, atol=0.3)

    def test_adaptive_regularization(self):
        """Adaptive regularization should scale with output variance."""
        state = _make_state()
        adf_adaptive = AssumedDensityFilter(
            n_samples=1000,
            regularization=1e-6,
            adaptive_regularization=True,
            key=jax.random.key(0),
        )
        adf_fixed = AssumedDensityFilter(
            n_samples=1000,
            regularization=1e-6,
            adaptive_regularization=False,
            key=jax.random.key(0),
        )

        result_adaptive = adf_adaptive.integrate(_linear_fn, state)
        result_fixed = adf_fixed.integrate(_linear_fn, state)

        # Both should produce valid results
        assert jnp.all(jnp.isfinite(result_adaptive.state.mean))
        assert jnp.all(jnp.isfinite(result_fixed.state.mean))

    def test_jit(self):
        """Should be JIT-compatible via eqx.filter_jit."""
        state = _make_state()
        integrator = AssumedDensityFilter(n_samples=100, key=jax.random.key(0))
        result1 = integrator.integrate(_linear_fn, state)

        @eqx.filter_jit
        def run(integ, s):
            return integ.integrate(_linear_fn, s)

        result2 = run(integrator, state)
        assert jnp.allclose(result1.state.mean, result2.state.mean, atol=1e-10)
