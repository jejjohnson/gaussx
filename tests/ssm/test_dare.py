"""Tests for DARE solver."""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import scipy.linalg

from gaussx import dare


class TestDARE:
    @pytest.mark.slow
    def test_converges_stable_system(self, getkey):
        """DARE converges for a stable system."""
        D, M = 3, 2
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        result = dare(A, H, Q, R)
        assert result.converged
        assert result.P_inf.shape == (D, D)
        assert result.K_inf.shape == (D, M)

    def test_satisfies_dare(self, getkey):
        """P_inf satisfies the DARE fixed-point equation."""
        D, M = 3, 2
        A = 0.8 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        result = dare(A, H, Q, R, max_iter=200)

        P = result.P_inf
        # One predict-update step from P should return P (fixed point).
        P_pred = A @ P @ A.T + Q
        S = H @ P_pred @ H.T + R
        K = jnp.linalg.solve(S, H @ P_pred).T
        P_updated = (jnp.eye(D) - K @ H) @ P_pred
        assert jnp.allclose(P, P_updated, atol=1e-6)

    def test_p_inf_symmetric(self, getkey):
        """Steady-state covariance is symmetric."""
        D, M = 4, 2
        A = 0.85 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.3
        Q = 0.2 * jnp.eye(D)
        R = jnp.eye(M)
        result = dare(A, H, Q, R)
        assert jnp.allclose(result.P_inf, result.P_inf.T, atol=1e-8)

    def test_p_inf_positive_definite(self, getkey):
        """Steady-state covariance is positive definite."""
        D, M = 3, 2
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(getkey(), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        result = dare(A, H, Q, R)
        eigvals = jnp.linalg.eigvalsh(result.P_inf)
        assert jnp.all(eigvals > 0)

    def test_jit_compatible(self):
        """Works under jax.jit."""
        D, M = 2, 1
        A = 0.9 * jnp.eye(D)
        H = jnp.ones((M, D))
        Q = 0.1 * jnp.eye(D)
        R = jnp.eye(M)
        result = jax.jit(dare)(A, H, Q, R)
        assert result.converged

    def test_p_init_is_deprecated_and_ignored(self):
        """The doubling algorithm needs no initial guess (gh-294)."""
        D, M = 3, 2
        A = 0.9 * jnp.eye(D)
        H = jax.random.normal(jax.random.key(0), (M, D)) * 0.5
        Q = 0.1 * jnp.eye(D)
        R = 0.5 * jnp.eye(M)
        with pytest.warns(DeprecationWarning, match="P_init"):
            result = dare(A, H, Q, R, P_init=jnp.eye(D))
        assert result.converged
        assert jnp.array_equal(result.P_inf, dare(A, H, Q, R).P_inf)


def test_dare_obs_noise_diagonal_operator(getkey):
    """dare with operator-typed R matches the array form."""
    import lineax as lx

    D, M = 3, 2
    A = 0.9 * jnp.eye(D)
    H = jax.random.normal(getkey(), (M, D)) * 0.5
    Q = 0.1 * jnp.eye(D)
    R_diag = jnp.array([0.3, 0.5])
    R = jnp.diag(R_diag)
    ref = dare(A, H, Q, R)
    op = dare(A, H, Q, lx.DiagonalLinearOperator(R_diag))
    assert jnp.allclose(ref.P_inf, op.P_inf, atol=1e-5)
    assert jnp.allclose(ref.K_inf, op.K_inf, atol=1e-5)


def _scipy_filtered(A, H, Q, R):
    """scipy's DARE gives the predicted P⁻; convert to the filtered P."""
    A, H, Q, R = (np.asarray(x) for x in (A, H, Q, R))
    P_pred = scipy.linalg.solve_discrete_are(A.T, H.T, Q, R)
    return P_pred - P_pred @ H.T @ np.linalg.solve(H @ P_pred @ H.T + R, H @ P_pred)


@pytest.mark.parametrize("a", [0.5, 0.9, 0.99, 0.999, 0.9999])
def test_slow_dynamics_match_scipy(a):
    # gh-294: at a = 0.999 the old 100-step fixed-point iteration stopped
    # 21% short and reported converged=False.
    A, H = jnp.array([[a]]), jnp.array([[1.0]])
    Q, R = jnp.array([[1e-4]]), jnp.array([[1.0]])
    result = dare(A, H, Q, R)
    assert result.converged
    np.testing.assert_allclose(result.P_inf, _scipy_filtered(A, H, Q, R), rtol=1e-10)


def test_random_stable_system_matches_scipy():
    k_a, k_h, k_q = jax.random.split(jax.random.key(0), 3)
    A = jax.random.normal(k_a, (3, 3))
    A = 0.95 * A / jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    H = jax.random.normal(k_h, (2, 3))
    L = jax.random.normal(k_q, (3, 3))
    Q = 0.1 * L @ L.T + 0.01 * jnp.eye(3)
    R = 0.5 * jnp.eye(2)
    result = dare(A, H, Q, R)
    assert result.converged
    np.testing.assert_allclose(result.P_inf, _scipy_filtered(A, H, Q, R), rtol=1e-9)
    P_pred = A @ result.P_inf @ A.T + Q
    K = P_pred @ H.T @ jnp.linalg.inv(H @ P_pred @ H.T + R)
    np.testing.assert_allclose(result.K_inf, K, rtol=1e-9)


def test_reports_non_convergence():
    A, H = jnp.array([[0.999]]), jnp.array([[1.0]])
    Q, R = jnp.array([[1e-4]]), jnp.array([[1.0]])
    assert not dare(A, H, Q, R, max_iter=2).converged


def _grad_test_system(theta):
    """A stable LTI system smooth in four scalar parameters (gh-97)."""
    k_a, k_h = jax.random.split(jax.random.key(1))
    A0 = jax.random.normal(k_a, (3, 3))
    A0 = A0 / jnp.max(jnp.abs(jnp.linalg.eigvals(A0)))
    H = jax.random.normal(k_h, (2, 3)).at[0, 0].add(theta[3])
    A = 0.9 * jnp.tanh(theta[0]) * A0
    Q = jnp.diag(jnp.exp(theta[1] + jnp.arange(3.0)))
    R = jnp.exp(theta[2]) * jnp.array([[1.0, 0.3], [0.3, 1.0]])
    return A, H, Q, R


_THETA = jnp.array([1.2, -1.0, -0.5, 0.3])
_WEIGHTS = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.mark.slow
def test_grad_matches_scipy_finite_differences():
    """Reverse-mode gradients through dare take the implicit path (gh-97).

    The oracle is a central difference of scipy's DARE solution, so it
    shares no code with gaussx; h = 1e-6 in float64 leaves a truncation
    plus rounding error of about 1e-9 relative.
    """

    def loss(theta):
        return jnp.sum(_WEIGHTS * dare(*_grad_test_system(theta)).P_inf)

    def loss_scipy(theta):
        return float(
            np.sum(np.asarray(_WEIGHTS) * _scipy_filtered(*_grad_test_system(theta)))
        )

    g = jax.jit(jax.grad(loss))(_THETA)
    h = 1e-6
    fd = np.array(
        [
            (loss_scipy(_THETA.at[i].add(h)) - loss_scipy(_THETA.at[i].add(-h)))
            / (2 * h)
            for i in range(_THETA.size)
        ]
    )
    np.testing.assert_allclose(g, fd, rtol=1e-6)
    # Forward mode goes through the same implicit rule.
    np.testing.assert_allclose(jax.jit(jax.jacfwd(loss))(_THETA), g, rtol=1e-10)


@pytest.mark.slow
def test_grad_of_gain_and_woodbury_path_agree():
    """K_inf is differentiable too, and the Woodbury innovation path agrees."""

    def loss(theta, woodbury):
        A, H, Q, R = _grad_test_system(theta)
        R_op = lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag)
        return jnp.sum(dare(A, H, Q, R_op, woodbury_innovation=woodbury).K_inf)

    grad = jax.jit(jax.grad(loss), static_argnums=1)
    g_dense = grad(_THETA, False)
    g_wood = grad(_THETA, True)
    assert jnp.all(jnp.isfinite(g_dense))
    np.testing.assert_allclose(g_wood, g_dense, rtol=1e-8)


@pytest.mark.slow
def test_grad_scalar_closed_form():
    """Scalar DARE: dP⁻/dq = 1 / (1 - a² r² / (P⁻ + r)²) (gh-97).

    Differentiating P⁻ = a² P⁻ r / (P⁻ + r) + q implicitly gives the
    closed form; reverse mode used to fail on the doubling while_loop.
    """
    a, r, q = 0.95, 0.5, 0.2

    def p_pred(q):
        A, H, R = jnp.array([[a]]), jnp.array([[1.0]]), jnp.array([[r]])
        P = dare(A, H, q * jnp.eye(1), R).P_inf[0, 0]
        return a**2 * P + q

    p = _scipy_filtered(
        jnp.array([[a]]), jnp.array([[1.0]]), q * jnp.eye(1), jnp.array([[r]])
    )[0, 0]
    p = a**2 * p + q
    expected = 1.0 / (1.0 - a**2 * r**2 / (p + r) ** 2)
    np.testing.assert_allclose(jax.jit(jax.grad(p_pred))(q), expected, rtol=1e-10)
