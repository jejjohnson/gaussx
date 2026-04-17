"""Tests for uncertain SVGP, VGP, and BGPLVM predictions."""

import jax
import jax.numpy as jnp
import lineax as lx

from gaussx._quadrature._gp_predict import (
    uncertain_bgplvm_predict,
    uncertain_gp_predict,
    uncertain_svgp_predict,
    uncertain_vgp_predict,
)
from gaussx._quadrature._monte_carlo import MonteCarloIntegrator
from gaussx._quadrature._types import GaussianState


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


def _make_Q_operator(K, S):
    """Build Q = K^{-1} S K^{-1} - K^{-1} densely."""
    K_inv = jnp.linalg.inv(K)
    Q = K_inv @ S @ K_inv - K_inv
    return lx.MatrixLinearOperator(Q)


class TestUncertainSVGPPredict:
    def test_returns_scalar_moments(self):
        """Should return scalar mean and variance."""
        state = _make_state_1d()
        M = 3
        Z = jnp.array([[0.0], [1.0], [2.0]])

        K_zz = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(Z))(Z)
        K_zz = K_zz + 0.01 * jnp.eye(M)
        S_u = 0.5 * jnp.eye(M)
        alpha = jnp.array([0.3, -0.1, 0.2])
        Q = _make_Q_operator(K_zz, S_u)

        integrator = MonteCarloIntegrator(n_samples=2000, key=jax.random.key(0))
        mean, var = uncertain_svgp_predict(_rbf_kernel, Z, alpha, Q, state, integrator)
        assert mean.shape == ()
        assert var.shape == ()
        assert jnp.isfinite(mean)
        assert var >= 0

    def test_zero_alpha_zero_mean(self):
        """With alpha=0, predictive mean should be zero."""
        state = _make_state_1d()
        M = 2
        Z = jnp.array([[0.0], [1.0]])

        K_zz = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(Z))(Z)
        K_zz = K_zz + 0.01 * jnp.eye(M)
        S_u = 0.5 * jnp.eye(M)
        alpha = jnp.zeros(M)
        Q = _make_Q_operator(K_zz, S_u)

        integrator = MonteCarloIntegrator(n_samples=2000, key=jax.random.key(0))
        mean, _ = uncertain_svgp_predict(_rbf_kernel, Z, alpha, Q, state, integrator)
        assert jnp.allclose(mean, 0.0, atol=0.1)

    def test_reduces_to_exact_gp_when_q_is_negative_k_inv(self):
        """SVGP matches exact uncertain GP when Z=X and Q=-K^{-1}."""
        state = _make_state_1d()
        X_train = jnp.array([[0.0], [1.0], [2.0]])

        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X_train))(X_train)
        K = K + 0.1 * jnp.eye(X_train.shape[0])
        K_inv_dense = jnp.linalg.inv(K)
        K_inv = lx.MatrixLinearOperator(K_inv_dense)

        y = jnp.array([0.5, -0.3, 0.1])
        alpha = jnp.linalg.solve(K, y)
        Q = lx.MatrixLinearOperator(-K_inv_dense)

        mean_gp, var_gp = uncertain_gp_predict(
            _rbf_kernel,
            X_train,
            alpha,
            K_inv,
            state,
            MonteCarloIntegrator(n_samples=5000, key=jax.random.key(42)),
        )
        mean_svgp, var_svgp = uncertain_svgp_predict(
            _rbf_kernel,
            X_train,
            alpha,
            Q,
            state,
            MonteCarloIntegrator(n_samples=5000, key=jax.random.key(42)),
        )

        assert jnp.allclose(mean_gp, mean_svgp, atol=0.15)
        assert jnp.allclose(var_gp, var_svgp, atol=0.15)


class TestUncertainVGPPredict:
    def test_returns_scalar_moments(self):
        """Should return scalar mean and variance."""
        state = _make_state_1d()
        N_train = 3
        X_train = jnp.array([[0.0], [1.0], [2.0]])

        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X_train))(X_train)
        K = K + 0.1 * jnp.eye(N_train)
        S = 0.5 * jnp.eye(N_train)
        m = jnp.array([0.5, -0.3, 0.1])
        alpha = jnp.linalg.solve(K, m)
        Q = _make_Q_operator(K, S)

        integrator = MonteCarloIntegrator(n_samples=2000, key=jax.random.key(0))
        mean, var = uncertain_vgp_predict(
            _rbf_kernel, X_train, alpha, Q, state, integrator
        )
        assert mean.shape == ()
        assert var.shape == ()
        assert jnp.isfinite(mean)
        assert var >= 0

    def test_reduces_to_exact_gp_when_q_is_negative_k_inv(self):
        """VGP matches exact uncertain GP when Q=-K^{-1}."""
        state = _make_state_1d()
        X_train = jnp.array([[0.0], [1.0], [2.0]])

        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X_train))(X_train)
        K = K + 0.1 * jnp.eye(X_train.shape[0])
        K_inv_dense = jnp.linalg.inv(K)
        K_inv = lx.MatrixLinearOperator(K_inv_dense)

        y = jnp.array([0.5, -0.3, 0.1])
        alpha = jnp.linalg.solve(K, y)
        Q = lx.MatrixLinearOperator(-K_inv_dense)

        mean_gp, var_gp = uncertain_gp_predict(
            _rbf_kernel,
            X_train,
            alpha,
            K_inv,
            state,
            MonteCarloIntegrator(n_samples=5000, key=jax.random.key(7)),
        )
        mean_vgp, var_vgp = uncertain_vgp_predict(
            _rbf_kernel,
            X_train,
            alpha,
            Q,
            state,
            MonteCarloIntegrator(n_samples=5000, key=jax.random.key(7)),
        )

        assert jnp.allclose(mean_gp, mean_vgp, atol=0.15)
        assert jnp.allclose(var_gp, var_vgp, atol=0.15)


class TestUncertainBGPLVMPredict:
    def test_multi_output_shapes(self):
        """Should return (D_out,) mean and variance."""
        state = _make_state_1d()
        N_train = 3
        D_out = 4
        X_train = jnp.array([[0.0], [1.0], [2.0]])

        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X_train))(X_train)
        K = K + 0.1 * jnp.eye(N_train)
        K_inv = lx.MatrixLinearOperator(jnp.linalg.inv(K))

        Y = jax.random.normal(jax.random.key(1), (N_train, D_out))
        alpha = jnp.linalg.solve(K, Y)

        integrator = MonteCarloIntegrator(n_samples=2000, key=jax.random.key(0))
        mean, var = uncertain_bgplvm_predict(
            _rbf_kernel, X_train, alpha, K_inv, state, integrator
        )
        assert mean.shape == (D_out,)
        assert var.shape == (D_out,)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(var >= 0)

    def test_zero_alpha_zero_mean(self):
        """With alpha=0, all output means should be zero."""
        state = _make_state_1d()
        N_train = 2
        D_out = 3
        X_train = jnp.array([[0.0], [1.0]])

        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X_train))(X_train)
        K = K + 0.1 * jnp.eye(N_train)
        K_inv = lx.MatrixLinearOperator(jnp.linalg.inv(K))
        alpha = jnp.zeros((N_train, D_out))

        integrator = MonteCarloIntegrator(n_samples=2000, key=jax.random.key(0))
        mean, _ = uncertain_bgplvm_predict(
            _rbf_kernel, X_train, alpha, K_inv, state, integrator
        )
        assert jnp.allclose(mean, 0.0, atol=0.1)

    def test_single_output_matches_gp(self):
        """With D_out=1, should be consistent with uncertain_gp_predict."""
        state = _make_state_1d()
        N_train = 3
        X_train = jnp.array([[0.0], [1.0], [2.0]])

        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel(x, y))(X_train))(X_train)
        K = K + 0.1 * jnp.eye(N_train)
        K_inv = lx.MatrixLinearOperator(jnp.linalg.inv(K))
        y = jnp.array([0.5, -0.3, 0.1])
        alpha_1d = jnp.linalg.solve(K, y)

        integrator = MonteCarloIntegrator(n_samples=5000, key=jax.random.key(42))

        # Exact GP prediction
        mean_gp, var_gp = uncertain_gp_predict(
            _rbf_kernel, X_train, alpha_1d, K_inv, state, integrator
        )

        # BGPLVM with D_out=1
        alpha_2d = alpha_1d[:, None]
        integrator2 = MonteCarloIntegrator(n_samples=5000, key=jax.random.key(42))
        mean_bg, var_bg = uncertain_bgplvm_predict(
            _rbf_kernel, X_train, alpha_2d, K_inv, state, integrator2
        )

        assert jnp.allclose(mean_gp, mean_bg[0], atol=0.15)
        assert jnp.allclose(var_gp, var_bg[0], atol=0.15)
