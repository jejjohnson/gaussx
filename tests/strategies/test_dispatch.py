"""Tests for solver dispatch: recipes/sugar accept solver= parameter."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._strategies import DenseSolver
from gaussx._testing import tree_allclose


def _make_pd_operator(key, n=5):
    A = jr.normal(key, (n, n))
    M = A @ A.T + n * jnp.eye(n)
    return lx.MatrixLinearOperator(M, lx.positive_semidefinite_tag)


# ── Sugar: gaussian_log_prob ───────────────────────────────────────


def test_gaussian_log_prob_with_solver(getkey):
    """gaussian_log_prob(solver=DenseSolver()) should match default."""
    from gaussx._sugar._gaussian import gaussian_log_prob

    op = _make_pd_operator(getkey())
    mu = jnp.zeros(5)
    x = jr.normal(getkey(), (5,))
    ref = gaussian_log_prob(mu, op, x)
    got = gaussian_log_prob(mu, op, x, solver=DenseSolver())
    assert tree_allclose(got, ref)


def test_gaussian_entropy_with_solver(getkey):
    """gaussian_entropy(solver=DenseSolver()) should match default."""
    from gaussx._sugar._gaussian import gaussian_entropy

    op = _make_pd_operator(getkey())
    ref = gaussian_entropy(op)
    got = gaussian_entropy(op, solver=DenseSolver())
    assert tree_allclose(got, ref)


def test_quadratic_form_with_solver(getkey):
    """quadratic_form(solver=DenseSolver()) should match default."""
    from gaussx._sugar._gaussian import quadratic_form

    op = _make_pd_operator(getkey())
    x = jr.normal(getkey(), (5,))
    ref = quadratic_form(op, x)
    got = quadratic_form(op, x, solver=DenseSolver())
    assert tree_allclose(got, ref)


def test_kl_standard_normal_with_solver(getkey):
    """kl_standard_normal(solver=DenseSolver()) should match default."""
    from gaussx._sugar._gaussian import kl_standard_normal

    op = _make_pd_operator(getkey())
    m = jr.normal(getkey(), (5,))
    ref = kl_standard_normal(m, op)
    got = kl_standard_normal(m, op, solver=DenseSolver())
    assert tree_allclose(got, ref)


# ── Sugar: log_marginal_likelihood ─────────────────────────────────


def test_log_mll_with_solver(getkey):
    """log_marginal_likelihood(solver=DenseSolver()) should match default."""
    from gaussx._sugar._inference import log_marginal_likelihood

    op = _make_pd_operator(getkey())
    mu = jnp.zeros(5)
    y = jr.normal(getkey(), (5,))
    ref = log_marginal_likelihood(mu, op, y)
    got = log_marginal_likelihood(mu, op, y, solver=DenseSolver())
    assert tree_allclose(got, ref)


def test_trace_correction_with_solver(getkey):
    """trace_correction(solver=DenseSolver()) should match default."""
    from gaussx._sugar._inference import trace_correction

    N, M = 6, 3
    K_xx = _make_pd_operator(getkey(), N)
    K_zz = _make_pd_operator(getkey(), M)
    K_xz = jr.normal(getkey(), (N, M))
    ref = trace_correction(K_xx, K_xz, K_zz)
    got = trace_correction(K_xx, K_xz, K_zz, solver=DenseSolver())
    assert tree_allclose(got, ref)


# ── Recipes: natural params ────────────────────────────────────────


def test_natural_to_mean_cov_with_solver(getkey):
    """natural_to_mean_cov(solver=DenseSolver()) should match default."""
    from gaussx._recipes._natural import mean_cov_to_natural, natural_to_mean_cov

    op = _make_pd_operator(getkey())
    mu = jr.normal(getkey(), (5,))
    eta1, eta2 = mean_cov_to_natural(mu, op)
    mu_ref, Sigma_ref = natural_to_mean_cov(eta1, eta2)
    mu_got, Sigma_got = natural_to_mean_cov(eta1, eta2, solver=DenseSolver())
    assert tree_allclose(mu_got, mu_ref, rtol=1e-4)
    assert tree_allclose(Sigma_got.as_matrix(), Sigma_ref.as_matrix(), rtol=1e-4)


def test_mean_cov_to_natural_with_solver(getkey):
    """mean_cov_to_natural(solver=DenseSolver()) should match default."""
    from gaussx._recipes._natural import mean_cov_to_natural

    op = _make_pd_operator(getkey())
    mu = jr.normal(getkey(), (5,))
    eta1_ref, eta2_ref = mean_cov_to_natural(mu, op)
    eta1_got, eta2_got = mean_cov_to_natural(mu, op, solver=DenseSolver())
    assert tree_allclose(eta1_got, eta1_ref, rtol=1e-4)
    assert tree_allclose(eta2_got.as_matrix(), eta2_ref.as_matrix(), rtol=1e-4)


# ── Recipes: Kalman filter ─────────────────────────────────────────


def test_kalman_filter_with_solver(getkey):
    """kalman_filter(solver=DenseSolver()) should match default."""
    from gaussx._recipes._kalman import kalman_filter

    N_state, M_obs, T = 3, 2, 5
    A = jnp.eye(N_state) + 0.01 * jr.normal(getkey(), (N_state, N_state))
    H = jr.normal(getkey(), (M_obs, N_state))
    Q = 0.1 * jnp.eye(N_state)
    R = 0.5 * jnp.eye(M_obs)
    y = jr.normal(getkey(), (T, M_obs))
    x0 = jnp.zeros(N_state)
    P0 = jnp.eye(N_state)

    ref = kalman_filter(A, H, Q, R, y, x0, P0)
    got = kalman_filter(A, H, Q, R, y, x0, P0, solver=DenseSolver())
    assert tree_allclose(got.log_likelihood, ref.log_likelihood, rtol=1e-4)
    assert tree_allclose(got.filtered_means, ref.filtered_means, rtol=1e-4)


def test_kalman_gain_with_solver(getkey):
    """kalman_gain(solver=DenseSolver()) should match default."""
    from gaussx._recipes._kalman import kalman_gain

    N_state, M_obs = 3, 2
    P_mat = jr.normal(getkey(), (N_state, N_state))
    P_mat = P_mat @ P_mat.T + N_state * jnp.eye(N_state)
    P = lx.MatrixLinearOperator(P_mat, lx.positive_semidefinite_tag)
    H = lx.MatrixLinearOperator(jr.normal(getkey(), (M_obs, N_state)))
    R_mat = 0.5 * jnp.eye(M_obs)
    R = lx.MatrixLinearOperator(R_mat, lx.positive_semidefinite_tag)

    ref = kalman_gain(P, H, R)
    got = kalman_gain(P, H, R, solver=DenseSolver())
    assert tree_allclose(got, ref, rtol=1e-4)
