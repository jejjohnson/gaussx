"""Tests for theta_design: eb / grid / ccd integration designs over theta."""

import functools as ft
import itertools

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
import numpy as np
import pytest

import gaussx
from gaussx._einx import einsum, rearrange, reduce
from gaussx._quadrature._theta_design import _ccd_design


def _gaussian(m):
    """A correlated Gaussian log-posterior with known mean and covariance."""
    k1, k2 = jr.split(jr.key(0))
    a = jr.normal(k1, (m, m))
    cov = einsum(a, a, "i k, j k -> i j") / m + 0.5 * jnp.eye(m)
    mean = jr.normal(k2, (m,))
    prec = jnp.linalg.inv(cov)

    def log_post(theta):
        r = theta - mean
        return -0.5 * r @ prec @ r

    return log_post, mean, cov


def _moments(pts, logw):
    w = jnp.exp(logw)
    mean = w @ pts
    r = pts - mean
    return mean, einsum(einx.multiply("k, k i -> k i", w, r), r, "k i, k j -> i j")


def _std_normal(theta):
    return -0.5 * jnp.sum(theta**2)


class TestCCD:
    @pytest.mark.parametrize(("m", "k"), [(3, 15), (4, 25), (5, 27), (6, 45)])
    def test_point_count(self, m, k):
        pts, logw = gaussx.theta_design(_std_normal, jnp.zeros(m), method="ccd")
        assert pts.shape == (k, m)
        assert logw.shape == (k,)

    @pytest.mark.parametrize("m", [3, 4, 5, 6])
    def test_symmetry(self, m):
        """Shell points on radius f0 sqrt(m), balanced and rotatable."""
        f0 = 1.3
        pts, logw = gaussx.theta_design(
            _std_normal, jnp.zeros(m), method="ccd", ccd_f0=f0
        )
        np.testing.assert_allclose(pts[0], 0.0, atol=1e-12)
        shell = pts[1:]
        np.testing.assert_allclose(
            jnp.sqrt(reduce(shell**2, "k m -> k", "sum")), f0 * np.sqrt(m), rtol=1e-12
        )
        np.testing.assert_allclose(reduce(shell, "k m -> m", "sum"), 0.0, atol=1e-10)
        second = einsum(shell, shell, "k i, k j -> i j")
        np.testing.assert_allclose(second, second[0, 0] * np.eye(m), atol=1e-10)
        # Every shell point has the same density, so the same weight.
        np.testing.assert_allclose(logw[1:], logw[1], rtol=1e-12)

    @pytest.mark.parametrize("m", [3, 4, 5, 6, 7, 8])
    def test_factorial_is_resolution_v(self, m):
        """Main effects and two-factor interactions are pairwise unaliased."""
        z = _ccd_design(m)
        fac = z[1 + 2 * m :]
        effects = [fac[:, i] for i in range(m)]
        effects += [
            fac[:, i] * fac[:, j] for i, j in itertools.combinations(range(m), 2)
        ]
        e = np.stack(effects)
        np.testing.assert_allclose(
            einsum(e, e, "a k, b k -> a b"), fac.shape[0] * np.eye(len(effects))
        )

    @pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
    def test_recovers_gaussian_moments(self, m):
        log_post, mean, cov = _gaussian(m)
        pts, logw = gaussx.theta_design(log_post, mean, method="ccd")
        np.testing.assert_allclose(jsp.logsumexp(logw), 0.0, atol=1e-12)
        mu, sigma = _moments(pts, logw)
        np.testing.assert_allclose(mu, mean, atol=1e-10)
        np.testing.assert_allclose(sigma, cov, atol=1e-10)

    def test_jit_and_float32(self):
        log_post, mean, _ = _gaussian(3)
        mean32 = mean.astype(jnp.float32)
        pts, logw = jax.jit(
            lambda th: gaussx.theta_design(
                lambda t: log_post(t).astype(jnp.float32), th, method="ccd"
            )
        )(mean32)
        assert pts.dtype == jnp.float32
        assert logw.dtype == jnp.float32

    def test_rejects_f0_le_one(self):
        with pytest.raises(ValueError, match="ccd_f0"):
            gaussx.theta_design(_std_normal, jnp.zeros(3), method="ccd", ccd_f0=1.0)


class TestGrid:
    def test_threshold_respected(self):
        log_post, mean, _ = _gaussian(2)
        threshold = 2.3  # off the z-lattice radii, so no ties at the boundary
        pts, logw = gaussx.theta_design(
            log_post, mean, method="grid", grid_step=0.5, grid_threshold=threshold
        )
        lp = jax.vmap(log_post)(pts)
        assert jnp.all(lp >= log_post(mean) - threshold)
        # Points come close to the threshold, so it is what truncated the grid.
        assert jnp.min(lp) < log_post(mean) - threshold + 0.5
        np.testing.assert_allclose(pts[0], mean)
        np.testing.assert_allclose(logw, lp - jsp.logsumexp(lp), atol=1e-12)

    def test_axis_walk_count(self):
        """Standard normal, step 1, threshold 2.3: |z_j| <= 2, |z|^2 <= 4.6."""
        pts, _ = gaussx.theta_design(
            _std_normal, jnp.zeros(2), method="grid", grid_threshold=2.3
        )
        expected = sum(
            1 for a in range(-2, 3) for b in range(-2, 3) if a * a + b * b <= 4.6
        )
        assert pts.shape == (expected, 2)

    def test_recovers_gaussian_moments(self):
        """A fine, wide grid integrates a Gaussian; truncation sets the error."""
        log_post, mean, cov = _gaussian(2)
        pts, logw = gaussx.theta_design(
            log_post, mean, method="grid", grid_step=0.5, grid_threshold=12.3
        )
        mu, sigma = _moments(pts, logw)
        np.testing.assert_allclose(mu, mean, atol=1e-10)
        np.testing.assert_allclose(sigma, cov, atol=1e-3)


class TestDesign:
    def test_eb_is_the_mode(self):
        log_post, mean, _ = _gaussian(3)
        pts, logw = gaussx.theta_design(log_post, mean, method="eb")
        np.testing.assert_allclose(pts, rearrange(mean, "m -> 1 m"))
        np.testing.assert_allclose(logw, [0.0])

    @pytest.mark.parametrize(("m", "method"), [(1, "grid"), (2, "grid"), (3, "ccd")])
    def test_default_method(self, m, method):
        log_post, mean, _ = _gaussian(m)
        got = gaussx.theta_design(log_post, mean)
        want = gaussx.theta_design(log_post, mean, method=method)
        np.testing.assert_allclose(got[0], want[0])

    def test_given_hessian_matches_autodiff(self):
        log_post, mean, cov = _gaussian(3)
        hess = -jnp.linalg.inv(cov)
        got = gaussx.theta_design(log_post, mean, method="ccd", hessian=hess)
        want = gaussx.theta_design(log_post, mean, method="ccd")
        np.testing.assert_allclose(got[0], want[0], atol=1e-10)
        np.testing.assert_allclose(got[1], want[1], atol=1e-10)

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            gaussx.theta_design(_std_normal, jnp.zeros(2), method="bogus")


def _sparse_log_post():
    """θ log-posterior with a sparse log|Q(θ)| and a sparse solve, as in INLA."""
    n = 8
    rows = np.r_[np.arange(n), np.arange(1, n)]
    cols = np.r_[np.arange(n), np.arange(n - 1)]
    degree = jnp.r_[jnp.array([1.0]), jnp.full(n - 2, 2.0), jnp.array([1.0])]
    R = gaussx.SparseOperator.from_coo(
        rows, cols, jnp.r_[degree, -jnp.ones(n - 1)], (n, n), symmetric=True
    )
    y = jnp.linspace(-1.0, 1.0, n)

    def precision(theta):
        Q = eqx.tree_at(lambda op: op.values, R, jnp.exp(theta[0]) * R.values)
        return Q.add_diagonal(jnp.full(n, jnp.exp(theta[1])))

    def log_post(theta, *, sparse):
        Q = precision(theta)
        if sparse:
            solver = gaussx.SparseCholeskySolver()
            ld, x = solver.logdet(Q), solver.solve(Q, y)
        else:
            ld = jnp.linalg.slogdet(Q.as_matrix())[1]
            x = jnp.linalg.solve(Q.as_matrix(), y)
        return 0.5 * ld - 0.5 * y @ x - 2.0 * theta @ theta

    return log_post


@pytest.mark.slow  # seconds: compiles second-order sparse Cholesky / Takahashi scans
def test_default_hessian_through_sparse_cholesky():
    # G4's sparse logdet / solve define custom VJPs only, so the default
    # Hessian must be reverse-over-reverse; it must also see how the
    # Takahashi / adjoint cotangents of their backward passes depend on θ.
    # Reference: jax.hessian of the same log-posterior through dense algebra.
    log_post = _sparse_log_post()
    sparse = jax.jit(ft.partial(log_post, sparse=True))
    mode = jnp.array([0.3, -0.2])
    expected = jax.jit(jax.hessian(ft.partial(log_post, sparse=False)))(mode)
    assert jnp.allclose(jax.jit(jax.jacrev(jax.jacrev(sparse)))(mode), expected)

    pts, logw = gaussx.theta_design(sparse, mode, method="ccd")
    ref_pts, ref_logw = gaussx.theta_design(
        sparse, mode, hessian=expected, method="ccd"
    )
    assert jnp.allclose(pts, ref_pts)
    assert jnp.allclose(logw, ref_logw)
