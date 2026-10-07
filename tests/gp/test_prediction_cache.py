"""Tests for prediction cache."""

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gaussx import (
    CGSolver,
    PredictionCache,
    build_prediction_cache,
    predict_mean,
    predict_variance,
)
from gaussx._testing import (
    psd_operator,
    random_block_diag_pd,
    random_kronecker_pd,
    random_pd_matrix,
)


def _problem(N=10, Nt=4):
    """Fixed dense K_y (jitter 1: cond ~1e2), targets and cross-covariance."""
    keys = jr.split(jr.key(0), 3)
    K = random_pd_matrix(keys[0], N, jitter=1.0)
    y = jr.normal(keys[1], (N,))
    K_cross = jr.normal(keys[2], (Nt, N))
    return K, y, K_cross


def _dense_variance(K, K_cross, K_test_diag):
    return K_test_diag - jnp.diag(K_cross @ jnp.linalg.solve(K, K_cross.T))


class TestPredictionCache:
    def test_alpha_matches_solve(self):
        """Cached alpha matches direct solve."""
        K, y, _ = _problem()
        cache = build_prediction_cache(psd_operator(K), y)
        assert jnp.allclose(cache.alpha, jnp.linalg.solve(K, y), rtol=1e-10)

    def test_predict_mean(self):
        """Predictive mean matches K_cross @ K_inv @ y."""
        K, y, K_cross = _problem()
        cache = build_prediction_cache(psd_operator(K), y)
        mu = predict_mean(cache, K_cross)
        expected = K_cross @ jnp.linalg.solve(K, y)
        assert jnp.allclose(mu, expected, rtol=1e-10)
        assert mu.shape == (4,)

    def test_predict_variance(self):
        """Predictive variance matches exact computation."""
        K, y, K_cross = _problem()
        K_test_diag = 2.0 * jnp.ones(4)
        cache = build_prediction_cache(psd_operator(K), y)
        var = predict_variance(cache, K_cross, K_test_diag)
        expected = _dense_variance(K, K_cross, K_test_diag)
        assert jnp.allclose(var, expected, rtol=1e-10)
        assert var.shape == (4,)

    def test_pytree_compatible(self):
        """Cache is a valid JAX pytree."""
        K, y, _ = _problem()
        cache = build_prediction_cache(psd_operator(K), y)
        leaves, treedef = jax.tree_util.tree_flatten(cache)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, PredictionCache)
        assert cache.factor is not None


class TestCachedFactor:
    """gh-338: one factorisation in build, none per predict_variance call."""

    @staticmethod
    def _count_cholesky(fn, *args):
        return str(jax.make_jaxpr(fn)(*args)).count("= cholesky")

    def test_no_cholesky_per_call(self):
        K, y, K_cross = _problem()
        op = psd_operator(K)
        cache = build_prediction_cache(op, y)
        kd = jnp.ones(4)
        assert self._count_cholesky(lambda yy: build_prediction_cache(op, yy), y) == 1
        n = self._count_cholesky(lambda Kc: predict_variance(cache, Kc, kd), K_cross)
        assert n == 0

    @pytest.mark.parametrize(
        "make_op",
        [
            pytest.param(
                lambda: psd_operator(random_pd_matrix(jr.key(1), 6)), id="dense"
            ),
            pytest.param(
                lambda: random_kronecker_pd(jr.key(1), (2, 3)), id="kronecker"
            ),
            pytest.param(
                lambda: random_block_diag_pd(jr.key(1), (2, 4)), id="blockdiag"
            ),
        ],
    )
    def test_new_path_matches_old(self, make_op):
        op = make_op()
        N = op.in_size()
        y = jr.normal(jr.key(2), (N,))
        K_cross = jr.normal(jr.key(3), (3, N))
        kd = 5.0 * jnp.ones(3)
        cache = build_prediction_cache(op, y)
        assert cache.factor is not None
        new = predict_variance(cache, K_cross, kd)
        with pytest.warns(DeprecationWarning, match="predict_variance"):
            old = predict_variance(K_cross, kd, op)
        assert jnp.allclose(new, old, rtol=1e-12, atol=1e-12)
        assert jnp.allclose(
            new, _dense_variance(op.as_matrix(), K_cross, kd), rtol=1e-10
        )

    def test_iterative_solver_has_no_factor(self):
        K, y, K_cross = _problem()
        solver = CGSolver(rtol=1e-10, atol=1e-10)
        cache = build_prediction_cache(psd_operator(K), y, solver=solver)
        assert cache.factor is None
        var = predict_variance(cache, K_cross, jnp.ones(4), solver=solver)
        expected = _dense_variance(K, K_cross, jnp.ones(4))
        assert jnp.allclose(var, expected, rtol=1e-6, atol=1e-6)

    def test_alpha_only_cache_raises(self):
        with pytest.raises(ValueError, match="holds only alpha"):
            predict_variance(
                PredictionCache(alpha=jnp.ones(3)), jnp.ones((2, 3)), jnp.ones(2)
            )

    def test_old_call_forms_warn(self):
        K, _, K_cross = _problem()
        op, kd = psd_operator(K), jnp.ones(4)
        expected = _dense_variance(K, K_cross, kd)
        with pytest.warns(DeprecationWarning):
            v1 = predict_variance(K_cross, kd, op)
        with pytest.warns(DeprecationWarning):
            v2 = predict_variance(K_cross, kd, operator=op)
        with pytest.warns(DeprecationWarning):
            v3 = predict_variance(K_cross=K_cross, K_test_diag=kd, operator=op)
        for v in (v1, v2, v3):
            assert jnp.allclose(v, expected, rtol=1e-10)

    @pytest.mark.slow  # a jit + grad + vmap sweep: ~4-5 s in CI
    def test_jit_vmap_grad(self):
        x = jnp.linspace(0.0, 3.0, 8)
        xt = jnp.array([0.5, 1.7])
        y = jnp.sin(x)

        def rbf(a, b, ls):
            return jnp.exp(-0.5 * einx.subtract("i, j -> i j", a, b) ** 2 / ls**2)

        def new(ls):
            op = psd_operator(rbf(x, x, ls) + 0.1 * jnp.eye(8))
            cache = build_prediction_cache(op, y)
            return predict_variance(cache, rbf(xt, x, ls), jnp.ones(2)).sum()

        def old(ls):
            op = psd_operator(rbf(x, x, ls) + 0.1 * jnp.eye(8))
            return predict_variance(rbf(xt, x, ls), jnp.ones(2), op).sum()

        g_new = jax.jit(jax.grad(new))(0.8)
        with pytest.warns(DeprecationWarning):
            g_old = jax.grad(old)(0.8)
        assert jnp.allclose(g_new, g_old, rtol=1e-10)
        batched = jax.vmap(new)(jnp.array([0.6, 0.8]))
        assert jnp.allclose(batched[1], new(0.8), rtol=1e-12)
