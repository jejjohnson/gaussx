"""Tests for ensemble DA primitives: localization, inflation, ETKF."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx import (
    enkf_analysis,
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
from gaussx._einx import einsum, rearrange, reduce
from gaussx._testing import (
    assert_sample_moments,
    empirical_moments,
    key_sequence,
    psd_operator,
    random_pd_matrix,
    tree_allclose,
)


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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


def _etkf_analysis(Xf, H, y, R_op, *, inflation=1.0):
    """Apply `etkf_transform` to a state ensemble under a linear ``H``."""
    obs_particles = einsum(Xf, H, "J N, M N -> J M")
    w_mean, transform = etkf_transform(obs_particles, y, R_op, inflation=inflation)
    xbar_f = reduce(Xf, "J N -> N", "mean")
    Xp = einx.subtract("J N, N -> J N", Xf, xbar_f)
    xbar_a = xbar_f + einsum(w_mean, Xp, "J, J N -> N")
    return einx.add("J N, N -> J N", einsum(transform, Xp, "J K, K N -> J N"), xbar_a)


def _kf_update(mean, cov, H, R, y):
    """Dense Kalman update, the reference for the ETKF identities."""
    S = einsum(H, cov, H, "M N, N K, L K -> M L") + R
    K = rearrange(
        jnp.linalg.solve(S, einsum(H, cov, "M N, N K -> M K")), "M N -> N M"
    )  # P H^T S^{-1}, S symmetric
    post_mean = mean + einsum(K, y - einsum(H, mean, "M N, N -> M"), "N M, M -> N")
    post_cov = cov - einsum(K, H, cov, "N M, M L, L K -> N K")
    return post_mean, post_cov


@pytest.mark.slow
@pytest.mark.x64_only(reason="exact identity checked to float64 round-off")
def test_etkf_matches_kalman_filter():
    """ETKF analysis mean/cov equal the KF update for the sample prior.

    An exact algebraic identity, so the model is pinned (the randomness is
    incidental) and the tolerance is round-off (gh-401).
    """
    nextkey = key_sequence(0)
    J, N, M = 16, 4, 2
    Xf = jr.normal(nextkey(), (J, N))
    H = jr.normal(nextkey(), (M, N))
    R = random_pd_matrix(nextkey(), M, jitter=0.5)
    y = jr.normal(nextkey(), (M,))

    Xa = _etkf_analysis(Xf, H, y, psd_operator(R))
    mean_f, cov_f = empirical_moments(Xf)
    mean_kf, cov_kf = _kf_update(mean_f, cov_f, H, R, y)

    mean_a, cov_a = empirical_moments(Xa)
    assert tree_allclose(mean_a, mean_kf, rtol=1e-10, atol=1e-10)
    assert tree_allclose(cov_a, cov_kf, rtol=1e-10, atol=1e-10)


@pytest.mark.x64_only(reason="exact identity checked to float64 round-off")
@pytest.mark.parametrize("inflation", [1.0, 1.5])
def test_etkf_inflation_matches_inflated_kalman_prior(inflation):
    """``inflation=lambda`` is the KF update of the prior ``lambda * P_f``.

    Pins where the inflation enters the ensemble-space precision -- the
    ``(J - 1) / lambda`` prior term (gh-401). The tolerance is round-off.
    """
    nextkey = key_sequence(1)
    J, N, M = 16, 4, 2
    Xf = jr.normal(nextkey(), (J, N))
    H = jr.normal(nextkey(), (M, N))
    R = random_pd_matrix(nextkey(), M, jitter=0.5)
    y = jr.normal(nextkey(), (M,))

    Xa = _etkf_analysis(Xf, H, y, psd_operator(R), inflation=inflation)
    mean_f, cov_f = empirical_moments(Xf)
    mean_kf, cov_kf = _kf_update(mean_f, inflation * cov_f, H, R, y)

    mean_a, cov_a = empirical_moments(Xa)
    assert jnp.allclose(mean_a, mean_kf, atol=1e-10, rtol=0.0)
    assert jnp.allclose(cov_a, cov_kf, atol=1e-10, rtol=0.0)


def _linear_dynamics(nextkey, N, M, T):
    """A stable linear model with a fixed observation record."""
    A = 0.9 * jnp.linalg.qr(jr.normal(nextkey(), (N, N)))[0]  # spectral radius 0.9
    H = jr.normal(nextkey(), (M, N))
    R = random_pd_matrix(nextkey(), M, jitter=0.5)
    ys = jr.normal(nextkey(), (T, M))
    return A, H, R, ys


@pytest.mark.x64_only(reason="exact identity checked to float64 round-off")
def test_etkf_multi_cycle_matches_kalman_filter():
    """Six ETKF cycles reproduce `gaussx.kalman_filter` exactly.

    With linear dynamics and no process noise, propagating the ensemble keeps
    its sample moments equal to the KF prediction, and each ETKF analysis is
    the KF update of those moments, so the identity holds at every cycle to
    round-off (gh-401). `kalman_filter` predicts before the first update, so
    the ensemble is propagated before each analysis, the first included.
    """
    nextkey = key_sequence(2)
    J, N, M, T = 12, 3, 2, 6
    A, H, R, ys = _linear_dynamics(nextkey, N, M, T)
    X = jr.normal(nextkey(), (J, N))
    init_mean, init_cov = empirical_moments(X)

    kf = gaussx.kalman_filter(A, H, jnp.zeros((N, N)), R, ys, init_mean, init_cov)

    R_op = psd_operator(R)
    for t in range(T):
        X = einsum(X, A, "J K, N K -> J N")
        X = _etkf_analysis(X, H, ys[t], R_op)
        mean_a, cov_a = empirical_moments(X)
        assert jnp.allclose(mean_a, kf.filtered_means[t], atol=1e-10, rtol=0.0)
        assert jnp.allclose(cov_a, kf.filtered_covs[t], atol=1e-10, rtol=0.0)


@pytest.mark.slow
def test_stochastic_enkf_multi_cycle_matches_kalman_filter():
    """Six stochastic EnKF cycles track `gaussx.kalman_filter` in distribution.

    Process noise is added to the ensemble after each propagation, and the
    analysis ensemble is compared to the KF filtered moments with
    `assert_sample_moments`. The analysis members share the sample gain, so
    they are not exactly i.i.d. and the 7-sigma band is approximate; at
    J = 20 000 the gain's sampling error is O(1 / sqrt(J)) relative and the
    band is dominated by the members' own spread (gh-401).
    """
    nextkey = key_sequence(3)
    J, N, M, T = 20_000, 3, 2, 6
    A, H, R, ys = _linear_dynamics(nextkey, N, M, T)
    Q = 0.1 * random_pd_matrix(nextkey(), N, jitter=0.5)
    init_mean = jr.normal(nextkey(), (N,))
    init_cov = random_pd_matrix(nextkey(), N, jitter=0.5)

    kf = gaussx.kalman_filter(A, H, Q, R, ys, init_mean, init_cov)

    chol_p, chol_q = jnp.linalg.cholesky(init_cov), jnp.linalg.cholesky(Q)
    X = einx.add(
        "J N, N -> J N",
        einsum(jr.normal(nextkey(), (J, N)), chol_p, "J K, N K -> J N"),
        init_mean,
    )
    R_op = psd_operator(R)
    for t in range(T):
        noise = einsum(jr.normal(nextkey(), (J, N)), chol_q, "J K, N K -> J N")
        X = einsum(X, A, "J K, N K -> J N") + noise
        obs = einsum(X, H, "J N, M N -> J M")
        X = enkf_analysis(X, obs, ys[t], R_op, key=nextkey())
        assert_sample_moments(X, kf.filtered_means[t], kf.filtered_covs[t])


def _forbid_as_matrix(monkeypatch, *classes):
    def _explode(self):
        raise AssertionError(f"{type(self).__name__}.as_matrix() was called")

    for cls in classes:
        monkeypatch.setattr(cls, "as_matrix", _explode)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="gh-367: etkf_transform densifies obs_noise via as_matrix()",
)
def test_etkf_does_not_materialise_structured_noise(monkeypatch):
    """A diagonal / block-diagonal R must stay structured (gh-282, gh-367)."""
    nextkey = key_sequence(4)
    J, M = 5, 8
    obs_particles = jr.normal(nextkey(), (J, M))
    y = jr.normal(nextkey(), (M,))
    variances = 0.5 + jnp.arange(M) / 10.0
    diag = lx.DiagonalLinearOperator(variances)
    block = gaussx.BlockDiag(
        lx.DiagonalLinearOperator(variances[:3]),
        lx.MatrixLinearOperator(jnp.diag(variances[3:]), lx.positive_semidefinite_tag),
    )
    _forbid_as_matrix(monkeypatch, lx.DiagonalLinearOperator, gaussx.BlockDiag)
    for R_op in (diag, block):
        w_mean, transform = etkf_transform(obs_particles, y, R_op)
        assert jnp.all(jnp.isfinite(w_mean)) and jnp.all(jnp.isfinite(transform))


def test_etkf_jit(getkey):
    J, M = 10, 2
    obs_particles = jr.normal(getkey(), (J, M))
    y = jr.normal(getkey(), (M,))
    R = random_pd_matrix(getkey(), M)
    R_op = lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag)
    w, t = jax.jit(lambda o, yy: etkf_transform(o, yy, R_op))(obs_particles, y)
    assert w.shape == (J,) and t.shape == (J, J)


# gh-341: etkf_transform validates its inputs like its siblings.
_R2 = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
_Y2 = jnp.array([0.5, -0.2])


def test_etkf_rejects_single_member():
    with pytest.raises(ValueError, match="J >= 2"):
        etkf_transform(jr.normal(jr.key(0), (1, 2)), _Y2, _R2)


def test_etkf_rejects_wrong_observation_length():
    H = jr.normal(jr.key(1), (4, 2))
    with pytest.raises(ValueError, match=r"y must have shape \(2,\).*\(3,\)"):
        etkf_transform(H, jnp.ones(3), _R2)


def test_etkf_rejects_wrong_noise_size():
    H = jr.normal(jr.key(1), (4, 2))
    with pytest.raises(ValueError, match=r"obs_noise must be \(2, 2\).*\(3, 3\)"):
        etkf_transform(H, _Y2, lx.DiagonalLinearOperator(jnp.ones(3)))


def test_etkf_rejects_non_positive_inflation():
    H = jr.normal(jr.key(1), (4, 2))
    with pytest.raises(ValueError, match="inflation must be positive"):
        etkf_transform(H, _Y2, _R2, inflation=0.0)
