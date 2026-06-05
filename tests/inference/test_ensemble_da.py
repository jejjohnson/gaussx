"""Tests for ensemble DA primitives: localization, inflation, ETKF."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx import (
    ensemble_kalman_gain,
    etkf_transform,
    euclidean_distance,
    gaspari_cohn,
    haversine_distance,
    inflate_multiplicative,
    inflate_rtpp,
    inflate_rtps,
    localization_matrix,
    localized_kalman_gain,
)
from gaussx._testing import random_pd_matrix, tree_allclose


# ---------------------------------------------------------------------------
# Gaspari-Cohn taper
# ---------------------------------------------------------------------------


def test_gaspari_cohn_endpoints():
    r = jnp.array([0.0, 1.0, 2.0, 2.5])
    rho = gaspari_cohn(r, c=2.0)  # support radius 2 -> zero at |r| >= 2
    assert jnp.isclose(rho[0], 1.0)
    assert rho[2] == 0.0
    assert rho[3] == 0.0
    # monotone non-increasing on the support, all in [0, 1]
    assert jnp.all(rho >= 0.0) and jnp.all(rho <= 1.0)
    assert jnp.all(jnp.diff(rho) <= 1e-6)


def test_gaspari_cohn_continuous_at_knots():
    c = 2.0
    # knots are at z = 1 (|r| = c/2) and z = 2 (|r| = c)
    eps = 1e-4
    for r0 in (c / 2, c):
        lo = gaspari_cohn(jnp.array(r0 - eps), c)
        hi = gaspari_cohn(jnp.array(r0 + eps), c)
        assert jnp.abs(lo - hi) < 1e-2


def test_gaspari_cohn_gradient_finite_at_zero():
    g = jax.grad(lambda x: gaspari_cohn(x, 2.0))(0.0)
    assert jnp.isfinite(g)
    # also finite across the support including the knots
    grads = jax.vmap(jax.grad(lambda x: gaspari_cohn(x, 2.0)))(
        jnp.linspace(0.0, 3.0, 50)
    )
    assert jnp.all(jnp.isfinite(grads))


def test_gaspari_cohn_reference_value():
    # midpoint z = 1 (|r| = c/2) evaluates to 25/120 = 0.2083... in both branches
    val = gaspari_cohn(jnp.array(1.0), c=2.0)
    assert jnp.isclose(val, 5.0 / 24.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------


def test_euclidean_distance_values():
    a = jnp.array([[0.0, 0.0]])
    b = jnp.array([[3.0, 4.0], [0.0, 0.0]])
    d = euclidean_distance(a, b)
    assert d.shape == (1, 2)
    assert jnp.isclose(d[0, 0], 5.0, atol=1e-5)
    assert jnp.isclose(d[0, 1], 0.0, atol=1e-6)


def test_euclidean_distance_gradient_safe_at_zero():
    pt = jnp.array([[1.0, 2.0]])
    g = jax.grad(lambda x: euclidean_distance(x, x)[0, 0])(pt)
    assert jnp.all(jnp.isfinite(g))


def test_haversine_quarter_circle():
    # equator points 90 deg apart -> quarter great circle = pi/2 * radius
    a = jnp.array([[0.0, 0.0]])
    b = jnp.array([[0.0, jnp.pi / 2]])
    d = haversine_distance(a, b, radius=1.0)
    assert jnp.isclose(d[0, 0], jnp.pi / 2, atol=1e-6)


# ---------------------------------------------------------------------------
# Localization matrix + localized gain
# ---------------------------------------------------------------------------


def test_localization_matrix_self_diagonal(getkey):
    coords = jr.normal(getkey(), (6, 2))
    rho = localization_matrix(coords, coords, c=10.0)
    assert rho.shape == (6, 6)
    # zero distance -> taper 1 on the diagonal
    assert tree_allclose(jnp.diag(rho), jnp.ones(6), atol=1e-6)


def test_localized_gain_reduces_to_unlocalized(getkey):
    """rho == 1 everywhere (c -> inf) recovers ensemble_kalman_gain."""
    J, N, M = 12, 5, 3
    particles = jr.normal(getkey(), (J, N))
    obs_particles = jr.normal(getkey(), (J, M))
    R = random_pd_matrix(getkey(), M)
    R_op = lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag)

    rho_xy = jnp.ones((N, M))
    rho_yy = jnp.ones((M, M))
    k_loc = localized_kalman_gain(particles, obs_particles, R_op, rho_xy, rho_yy)
    k_ref = ensemble_kalman_gain(particles, obs_particles, R_op)
    assert tree_allclose(k_loc, k_ref, rtol=1e-4, atol=1e-5)


def test_localized_gain_suppresses_distant_updates(getkey):
    """Tapering zeros the gain where rho_xy is zero."""
    J, N, M = 16, 8, 2
    particles = jr.normal(getkey(), (J, N))
    obs_particles = jr.normal(getkey(), (J, M))
    R = random_pd_matrix(getkey(), M)
    R_op = lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag)

    rho_xy = jnp.ones((N, M)).at[0, :].set(0.0)  # state 0 fully localized away
    rho_yy = jnp.ones((M, M))
    k = localized_kalman_gain(particles, obs_particles, R_op, rho_xy, rho_yy)
    assert tree_allclose(k[0, :], jnp.zeros(M), atol=1e-8)


# ---------------------------------------------------------------------------
# Inflation
# ---------------------------------------------------------------------------


def test_inflate_multiplicative(getkey):
    ens = jr.normal(getkey(), (20, 4))
    out = inflate_multiplicative(ens, factor=2.0)
    assert tree_allclose(out.mean(0), ens.mean(0), atol=1e-6)
    assert tree_allclose(out.std(0), 2.0 * ens.std(0), rtol=1e-5)


def test_inflate_rtpp_limits(getkey):
    post = jr.normal(getkey(), (20, 4))
    prior = jr.normal(getkey(), (20, 4))
    # alpha = 0 -> unchanged posterior
    assert tree_allclose(inflate_rtpp(post, prior, 0.0), post, atol=1e-6)
    # mean is always the posterior mean
    out = inflate_rtpp(post, prior, 0.5)
    assert tree_allclose(out.mean(0), post.mean(0), atol=1e-6)
    # alpha = 1 -> posterior mean with prior perturbations
    full = inflate_rtpp(post, prior, 1.0)
    expected = post.mean(0) + (prior - prior.mean(0))
    assert tree_allclose(full, expected, atol=1e-5)


def test_inflate_rtps_limits(getkey):
    post = 0.5 * jr.normal(getkey(), (30, 4))  # deliberately under-spread
    prior = jr.normal(getkey(), (30, 4))
    # beta = 0 -> unchanged
    assert tree_allclose(inflate_rtps(post, prior, 0.0), post, atol=1e-6)
    # mean preserved
    out = inflate_rtps(post, prior, 0.7)
    assert tree_allclose(out.mean(0), post.mean(0), atol=1e-6)
    # beta = 1 -> posterior spread matches prior spread per coordinate
    full = inflate_rtps(post, prior, 1.0)
    assert tree_allclose(full.std(0), prior.std(0), rtol=1e-4)


# ---------------------------------------------------------------------------
# ETKF transform
# ---------------------------------------------------------------------------


def test_etkf_preserves_mean(getkey):
    J, M = 14, 3
    obs_particles = jr.normal(getkey(), (J, M))
    y = jr.normal(getkey(), (M,))
    R = random_pd_matrix(getkey(), M)
    R_op = lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag)
    _, transform = etkf_transform(obs_particles, y, R_op)

    # mean-preservation: transformed zero-mean perturbations stay zero-mean
    state_pert = jr.normal(getkey(), (J, 5))
    state_pert = state_pert - state_pert.mean(0)
    analysis_pert = transform @ state_pert
    assert tree_allclose(analysis_pert.mean(0), jnp.zeros(5), atol=1e-6)


def test_etkf_matches_kalman_filter(getkey):
    """ETKF analysis mean/cov equal the KF update for the sample prior."""
    J, N, M = 16, 4, 2
    Xf = jr.normal(getkey(), (J, N))
    H = jr.normal(getkey(), (M, N))
    R = random_pd_matrix(getkey(), M)
    R_op = lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag)
    y = jr.normal(getkey(), (M,))

    # ETKF
    obs_particles = Xf @ H.T
    w_mean, transform = etkf_transform(obs_particles, y, R_op)
    xbar_f = Xf.mean(0)
    Xp = Xf - xbar_f
    xbar_a = xbar_f + w_mean @ Xp
    Xa = xbar_a[None, :] + transform @ Xp

    # Kalman filter reference using the sample prior covariance
    Pf = jnp.cov(Xf, rowvar=False)
    S = H @ Pf @ H.T + R
    K = Pf @ H.T @ jnp.linalg.inv(S)
    xbar_kf = xbar_f + K @ (y - H @ xbar_f)
    Pa_kf = (jnp.eye(N) - K @ H) @ Pf

    assert tree_allclose(xbar_a, xbar_kf, rtol=1e-4, atol=1e-5)
    assert tree_allclose(jnp.cov(Xa, rowvar=False), Pa_kf, rtol=1e-3, atol=1e-5)


def test_etkf_jit(getkey):
    J, M = 10, 2
    obs_particles = jr.normal(getkey(), (J, M))
    y = jr.normal(getkey(), (M,))
    R = random_pd_matrix(getkey(), M)
    R_op = lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag)
    w, t = jax.jit(lambda o, yy: etkf_transform(o, yy, R_op))(obs_particles, y)
    assert w.shape == (J,) and t.shape == (J, J)
