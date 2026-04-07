"""Tests for ImplicitKernelOperator with explicit params + custom JVP."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import ImplicitKernelOperator
from gaussx._testing import tree_allclose


def _rbf_kernel_params(params, x, y):
    """RBF kernel with explicit params: k(params, x, y)."""
    diff = x - y
    sq_dist = jnp.sum(diff**2) / params["lengthscale"] ** 2
    return params["variance"] * jnp.exp(-0.5 * sq_dist)


def _rbf_kernel_no_params(x, y):
    """RBF kernel without params: k(x, y)."""
    diff = x - y
    return jnp.exp(-0.5 * jnp.sum(diff**2))


def _make_params(key):
    k1, k2 = jr.split(key)
    return {
        "variance": jnp.abs(jr.normal(k1, ())) + 0.5,
        "lengthscale": jnp.abs(jr.normal(k2, ())) + 0.5,
    }


def _build_dense(kernel_fn, params, X):
    return jax.vmap(lambda x_i: jax.vmap(lambda x_j: kernel_fn(params, x_i, x_j))(X))(X)


# ---------------------------------------------------------------------------
# Backward compatibility — no params
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_no_params_mv(self, getkey):
        """No-params mode should work identically to before."""
        N, D = 8, 2
        X = jr.normal(getkey(), (N, D))
        v = jr.normal(getkey(), (N,))
        noise_var = 0.1

        op = ImplicitKernelOperator(_rbf_kernel_no_params, X, noise_var=noise_var)
        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel_no_params(x, y))(X))(X)
        K = K + noise_var * jnp.eye(N)
        assert tree_allclose(op.mv(v), K @ v, rtol=1e-5)

    def test_no_params_as_matrix(self, getkey):
        N, D = 6, 3
        X = jr.normal(getkey(), (N, D))
        op = ImplicitKernelOperator(_rbf_kernel_no_params, X, noise_var=0.05)
        K = jax.vmap(lambda x: jax.vmap(lambda y: _rbf_kernel_no_params(x, y))(X))(X)
        K = K + 0.05 * jnp.eye(N)
        assert tree_allclose(op.as_matrix(), K, rtol=1e-5)

    def test_no_params_transpose(self, getkey):
        N, D = 5, 2
        X = jr.normal(getkey(), (N, D))
        op = ImplicitKernelOperator(
            _rbf_kernel_no_params,
            X,
            tags=frozenset({lx.symmetric_tag}),
        )
        assert op.transpose() is op


# ---------------------------------------------------------------------------
# With params — mv correctness
# ---------------------------------------------------------------------------


class TestParamsMv:
    def test_mv_matches_dense(self, getkey):
        N, D = 8, 3
        X = jr.normal(getkey(), (N, D))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (N,))

        op = ImplicitKernelOperator(_rbf_kernel_params, X, params=params)
        K = _build_dense(_rbf_kernel_params, params, X)
        assert tree_allclose(op.mv(v), K @ v, rtol=1e-5)

    def test_mv_with_noise(self, getkey):
        N, D = 8, 3
        X = jr.normal(getkey(), (N, D))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (N,))

        op = ImplicitKernelOperator(_rbf_kernel_params, X, noise_var=0.1, params=params)
        K = _build_dense(_rbf_kernel_params, params, X) + 0.1 * jnp.eye(N)
        assert tree_allclose(op.mv(v), K @ v, rtol=1e-5)

    def test_as_matrix(self, getkey):
        N, D = 6, 2
        X = jr.normal(getkey(), (N, D))
        params = _make_params(getkey())
        op = ImplicitKernelOperator(
            _rbf_kernel_params, X, noise_var=0.05, params=params
        )
        K = _build_dense(_rbf_kernel_params, params, X) + 0.05 * jnp.eye(N)
        assert tree_allclose(op.as_matrix(), K, rtol=1e-5)


# ---------------------------------------------------------------------------
# With params — gradients
# ---------------------------------------------------------------------------


class TestParamsGradients:
    def test_grad_params(self, getkey):
        """Gradient via custom JVP matches naive dense autodiff."""
        N = 6
        X = jr.normal(getkey(), (N, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (N,))
        u = jr.normal(getkey(), (N,))

        def loss_custom(params):
            op = ImplicitKernelOperator(_rbf_kernel_params, X, params=params)
            return jnp.dot(u, op.mv(v))

        def loss_dense(params):
            K = _build_dense(_rbf_kernel_params, params, X)
            return jnp.dot(u, K @ v)

        assert tree_allclose(
            jax.grad(loss_custom)(params),
            jax.grad(loss_dense)(params),
            rtol=1e-4,
        )

    def test_grad_x(self, getkey):
        """Gradient w.r.t. X matches dense autodiff."""
        N = 6
        params = _make_params(getkey())
        v = jr.normal(getkey(), (N,))
        u = jr.normal(getkey(), (N,))

        def loss_custom(X):
            op = ImplicitKernelOperator(_rbf_kernel_params, X, params=params)
            return jnp.dot(u, op.mv(v))

        def loss_dense(X):
            K = _build_dense(_rbf_kernel_params, params, X)
            return jnp.dot(u, K @ v)

        X = jr.normal(getkey(), (N, 2))
        assert tree_allclose(
            jax.grad(loss_custom)(X),
            jax.grad(loss_dense)(X),
            rtol=1e-4,
        )

    def test_grad_v(self, getkey):
        N = 6
        X = jr.normal(getkey(), (N, 2))
        params = _make_params(getkey())

        def loss_custom(v):
            op = ImplicitKernelOperator(_rbf_kernel_params, X, params=params)
            return jnp.sum(op.mv(v) ** 2)

        def loss_dense(v):
            K = _build_dense(_rbf_kernel_params, params, X)
            return jnp.sum((K @ v) ** 2)

        v = jr.normal(getkey(), (N,))
        assert tree_allclose(
            jax.grad(loss_custom)(v),
            jax.grad(loss_dense)(v),
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


class TestJAXTransforms:
    def test_jit(self, getkey):
        N = 6
        X = jr.normal(getkey(), (N, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (N,))

        op = ImplicitKernelOperator(_rbf_kernel_params, X, params=params)

        @eqx.filter_jit
        def f(op, v):
            return op.mv(v)

        assert tree_allclose(f(op, v), op.as_matrix() @ v, rtol=1e-5)

    def test_vmap(self, getkey):
        N = 6
        X = jr.normal(getkey(), (N, 2))
        params = _make_params(getkey())
        op = ImplicitKernelOperator(_rbf_kernel_params, X, params=params)
        vs = jr.normal(getkey(), (4, N))
        results = jax.vmap(op.mv)(vs)
        assert results.shape == (4, N)
        assert tree_allclose(results[0], op.as_matrix() @ vs[0], rtol=1e-5)

    def test_jvp(self, getkey):
        N = 5
        X = jr.normal(getkey(), (N, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (N,))
        params_tangent = {
            "variance": jnp.array(0.2),
            "lengthscale": jnp.array(-0.1),
        }

        def loss_custom(p):
            op = ImplicitKernelOperator(_rbf_kernel_params, X, params=p)
            return op.mv(v)

        def loss_dense(p):
            return _build_dense(_rbf_kernel_params, p, X) @ v

        custom_p, custom_t = jax.jvp(loss_custom, (params,), (params_tangent,))
        dense_p, dense_t = jax.jvp(loss_dense, (params,), (params_tangent,))
        assert tree_allclose(custom_p, dense_p, rtol=1e-5)
        assert tree_allclose(custom_t, dense_t, rtol=1e-4)

    def test_transpose_with_params(self, getkey):
        N = 5
        X = jr.normal(getkey(), (N, 2))
        params = _make_params(getkey())
        op = ImplicitKernelOperator(_rbf_kernel_params, X, params=params)
        K = op.as_matrix()
        K_T = op.transpose().as_matrix()
        assert tree_allclose(K_T, K.T, rtol=1e-5)
