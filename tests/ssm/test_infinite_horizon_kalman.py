"""Tests for infinite-horizon Kalman filter and smoother."""

import jax
import jax.numpy as jnp

from gaussx import (
    dare,
    infinite_horizon_filter,
    infinite_horizon_smoother,
    kalman_filter,
)


class TestInfiniteHorizonFilter:
    def _stable_system(self, getkey):
        """Create a stable system for testing."""
        D, M = 3, 2
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        return A, H, Q, R

    def test_output_shapes(self, getkey):
        """Output shapes match time steps and state dimension."""
        A, H, Q, R = self._stable_system(getkey)
        D, M, T = A.shape[0], H.shape[0], 50
        obs = jax.random.normal(getkey(), (T, M))
        state = infinite_horizon_filter(A, H, Q, R, obs)
        assert state.filtered_means.shape == (T, D)
        assert state.filtered_covs.shape == (T, D, D)
        assert state.predicted_means.shape == (T, D)
        assert state.predicted_covs.shape == (T, D, D)
        assert state.log_likelihood.shape == ()

    def test_log_likelihood_finite(self, getkey):
        """Log-likelihood is finite."""
        A, H, Q, R = self._stable_system(getkey)
        T = 50
        obs = jax.random.normal(getkey(), (T, H.shape[0]))
        state = infinite_horizon_filter(A, H, Q, R, obs)
        assert jnp.isfinite(state.log_likelihood)

    def test_matches_standard_kf_at_convergence(self, getkey):
        """Filtered means converge to standard KF for long sequences."""
        A, H, Q, R = self._stable_system(getkey)
        D, M, T = A.shape[0], H.shape[0], 500
        obs = jax.random.normal(getkey(), (T, M))

        # Standard KF
        init_mean = jnp.zeros(D)
        init_cov = jnp.eye(D)
        kf_state = kalman_filter(A, H, Q, R, obs, init_mean, init_cov)

        # Infinite-horizon KF
        ih_state = infinite_horizon_filter(A, H, Q, R, obs)

        # Compare last 100 time steps (after convergence)
        assert jnp.allclose(
            kf_state.filtered_means[-100:],
            ih_state.filtered_means[-100:],
            atol=0.1,
        )

    def test_precomputed_dare(self, getkey):
        """Accepts precomputed DARE result."""
        A, H, Q, R = self._stable_system(getkey)
        T = 50
        obs = jax.random.normal(getkey(), (T, H.shape[0]))
        dare_result = dare(A, H, Q, R)
        state = infinite_horizon_filter(
            A,
            H,
            Q,
            R,
            obs,
            dare_result=dare_result,
        )
        assert jnp.isfinite(state.log_likelihood)

    def test_jit_compatible(self, getkey):
        """Works under jax.jit."""
        A, H, Q, R = self._stable_system(getkey)
        T = 30
        obs = jax.random.normal(getkey(), (T, H.shape[0]))
        dare_result = dare(A, H, Q, R)
        state = jax.jit(
            lambda o: infinite_horizon_filter(
                A,
                H,
                Q,
                R,
                o,
                dare_result=dare_result,
            )
        )(obs)
        assert jnp.isfinite(state.log_likelihood)


class TestInfiniteHorizonSmoother:
    def test_output_shapes(self, getkey):
        """Smoothed outputs have correct shapes."""
        D, M, T = 3, 2, 50
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        obs = jax.random.normal(getkey(), (T, M))
        dare_result = dare(A, H, Q, R)
        filt = infinite_horizon_filter(
            A,
            H,
            Q,
            R,
            obs,
            dare_result=dare_result,
        )
        s_means, s_covs = infinite_horizon_smoother(filt, A, dare_result, Q)
        assert s_means.shape == (T, D)
        assert s_covs.shape == (T, D, D)

    def test_smoother_jit_compatible(self, getkey):
        """Smoother works under jax.jit."""
        D, M, T = 3, 2, 30
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        obs = jax.random.normal(getkey(), (T, M))
        dare_result = dare(A, H, Q, R)
        filt = infinite_horizon_filter(
            A,
            H,
            Q,
            R,
            obs,
            dare_result=dare_result,
        )

        @jax.jit
        def smooth(filt):
            return infinite_horizon_smoother(filt, A, dare_result, Q)

        s_means, _s_covs = smooth(filt)
        assert jnp.all(jnp.isfinite(s_means))


def test_infinite_horizon_filter_obs_noise_diagonal_operator(getkey):
    """infinite_horizon_filter with operator-typed R matches the array form."""
    import lineax as lx

    N, M, T = 3, 2, 8
    A = 0.9 * jnp.eye(N)
    H = jax.random.normal(getkey(), (M, N)) * 0.5
    Q = 0.1 * jnp.eye(N)
    R_diag = jnp.array([0.3, 0.5])
    R = jnp.diag(R_diag)
    obs = jax.random.normal(getkey(), (T, M))

    ref = infinite_horizon_filter(A, H, Q, R, obs)
    op = infinite_horizon_filter(A, H, Q, lx.DiagonalLinearOperator(R_diag), obs)
    assert jnp.allclose(ref.filtered_means, op.filtered_means, atol=1e-5)
    assert jnp.allclose(ref.log_likelihood, op.log_likelihood, atol=1e-4)


def test_infinite_horizon_filter_woodbury_innovation_matches_dense(getkey):
    """Woodbury innovation path matches dense steady-state innovations."""
    import lineax as lx

    N, M, T = 4, 32, 8
    A = 0.8 * jnp.eye(N)
    H = jax.random.normal(getkey(), (M, N)) * 0.2
    Q = 0.05 * jnp.eye(N)
    R_diag = 0.3 + 0.1 * jnp.linspace(0.0, 1.0, M)
    R = jnp.diag(R_diag)
    obs = jax.random.normal(getkey(), (T, M))

    ref = infinite_horizon_filter(A, H, Q, R, obs)
    got = infinite_horizon_filter(
        A,
        H,
        Q,
        lx.DiagonalLinearOperator(R_diag),
        obs,
        woodbury_innovation=True,
    )

    assert jnp.allclose(ref.filtered_means, got.filtered_means, atol=1e-5)
    assert jnp.allclose(ref.log_likelihood, got.log_likelihood, atol=1e-4)
