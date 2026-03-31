"""Tests for uncertain GP predictions."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._uncertain._gp_predict import (
    kernel_expectations,
    uncertain_gp_predict,
    uncertain_gp_predict_mc,
)
from gaussx._uncertain._monte_carlo import MonteCarloIntegrator
from gaussx._uncertain._types import GaussianState


def _rbf_kernel(x, y, lengthscale=1.0):
    """Simple RBF kernel."""
    diff = x - y
    return jnp.exp(-0.5 * jnp.sum(diff**2) / lengthscale**2)


def _make_state_1d():
    """1D uncertain input."""
    mean = jnp.array([0.0])
    cov = lx.MatrixLinearOperator(
        jnp.array([[0.1]]),
        lx.positive_semidefinite_tag,
    )
    return GaussianState(mean=mean, cov=cov)


class TestKernelExpectations:
    def test_psi0_positive(self):
        """Psi_0 = E[k(x,x)] should be positive."""
        state = _make_state_1d()
        X_train = jnp.array([[0.0], [1.0], [2.0]])
        integrator = MonteCarloIntegrator(n_samples=2000, key=jax.random.key(0))

        Psi_0, _Psi_1, _Psi_2 = kernel_expectations(
            _rbf_kernel, state, X_train, integrator
        )
        assert Psi_0 > 0

    def test_psi1_shape(self):
        """Psi_1 should have shape (N_train,)."""
        state = _make_state_1d()
        X_train = jnp.array([[0.0], [1.0]])
        integrator = MonteCarloIntegrator(n_samples=1000, key=jax.random.key(0))

        _, Psi_1, _ = kernel_expectations(_rbf_kernel, state, X_train, integrator)
        assert Psi_1.shape == (2,)

    def test_psi2_shape_and_symmetry(self):
        """Psi_2 should be (N_train, N_train) and symmetric."""
        state = _make_state_1d()
        X_train = jnp.array([[0.0], [1.0], [2.0]])
        integrator = MonteCarloIntegrator(n_samples=2000, key=jax.random.key(0))

        _, _, Psi_2 = kernel_expectations(_rbf_kernel, state, X_train, integrator)
        assert Psi_2.shape == (3, 3)
        assert jnp.allclose(Psi_2, Psi_2.T, atol=0.1)


class TestUncertainGPPredict:
    def test_returns_scalar_moments(self):
        """Should return scalar mean and variance."""
        state = _make_state_1d()
        X_train = jnp.array([[0.0], [1.0]])
        N_train = X_train.shape[0]

        # Build kernel matrix
        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X_train))(X_train)
        K = K + 0.1 * jnp.eye(N_train)
        K_inv = lx.MatrixLinearOperator(jnp.linalg.inv(K))

        y_train = jnp.array([0.5, -0.3])
        alpha = jnp.linalg.solve(K, y_train)

        integrator = MonteCarloIntegrator(n_samples=2000, key=jax.random.key(0))

        mean, var = uncertain_gp_predict(
            _rbf_kernel, X_train, alpha, K_inv, state, integrator
        )
        assert mean.shape == ()
        assert var.shape == ()
        assert jnp.isfinite(mean)
        assert var >= 0


class TestUncertainGPPredictMC:
    def test_basic(self):
        """MC prediction should return finite moments."""
        state = _make_state_1d()

        def predict_fn(x):
            # Simple mock GP predictor
            return jnp.sum(x), jnp.array(0.1)

        mean, var = uncertain_gp_predict_mc(
            predict_fn,
            state,
            n_particles=200,
            key=jax.random.key(0),
        )
        assert mean.shape == ()
        assert var.shape == ()
        assert jnp.isfinite(mean)
        assert var >= 0

    def test_law_of_total_variance(self):
        """Variance should be >= mean of predictive variances."""
        mean_state = jnp.array([0.0])
        cov = lx.MatrixLinearOperator(
            jnp.array([[1.0]]),
            lx.positive_semidefinite_tag,
        )
        state = GaussianState(mean=mean_state, cov=cov)

        pred_var = 0.5

        def predict_fn(x):
            return jnp.sum(x), jnp.array(pred_var)

        _, var = uncertain_gp_predict_mc(
            predict_fn,
            state,
            n_particles=5000,
            key=jax.random.key(42),
        )
        # var = var(means) + mean(vars) >= mean(vars) = pred_var
        assert var >= pred_var - 0.1
