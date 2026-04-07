"""Tests for the KernelOperator with custom VJP."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import KernelOperator
from gaussx._testing import tree_allclose


def _rbf_kernel(params, x, y):
    """Simple RBF kernel: variance * exp(-0.5 * ||x - y||^2 / lengthscale^2)."""
    diff = x - y
    sq_dist = jnp.sum(diff**2) / params["lengthscale"] ** 2
    return params["variance"] * jnp.exp(-0.5 * sq_dist)


def _make_params(key):
    k1, k2 = jr.split(key)
    return {
        "variance": jnp.abs(jr.normal(k1, ())) + 0.5,
        "lengthscale": jnp.abs(jr.normal(k2, ())) + 0.5,
    }


def _build_dense(kernel_fn, params, X1, X2):
    return jax.vmap(lambda x_i: jax.vmap(lambda x_j: kernel_fn(params, x_i, x_j))(X2))(
        X1
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_square(self, getkey):
        X = jr.normal(getkey(), (10, 3))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X, X, params)
        assert op.in_size() == 10
        assert op.out_size() == 10

    def test_rectangular(self, getkey):
        X1 = jr.normal(getkey(), (10, 3))
        X2 = jr.normal(getkey(), (5, 3))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X1, X2, params)
        assert op.in_size() == 5
        assert op.out_size() == 10

    def test_tags(self, getkey):
        X = jr.normal(getkey(), (5, 2))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X, X, params, tags=lx.symmetric_tag)
        assert lx.is_symmetric(op) is True

    def test_psd_implies_symmetric(self, getkey):
        X = jr.normal(getkey(), (5, 2))
        params = _make_params(getkey())
        op = KernelOperator(
            _rbf_kernel, X, X, params, tags=lx.positive_semidefinite_tag
        )
        assert lx.is_symmetric(op) is True
        assert lx.is_positive_semidefinite(op) is True


# ---------------------------------------------------------------------------
# mv correctness
# ---------------------------------------------------------------------------


class TestMv:
    def test_mv_matches_dense_square(self, getkey):
        X = jr.normal(getkey(), (8, 3))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X, X, params)
        v = jr.normal(getkey(), (8,))
        K_dense = _build_dense(_rbf_kernel, params, X, X)
        assert tree_allclose(op.mv(v), K_dense @ v, rtol=1e-5)

    def test_mv_matches_dense_rectangular(self, getkey):
        X1 = jr.normal(getkey(), (10, 3))
        X2 = jr.normal(getkey(), (6, 3))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X1, X2, params)
        v = jr.normal(getkey(), (6,))
        K_dense = _build_dense(_rbf_kernel, params, X1, X2)
        assert tree_allclose(op.mv(v), K_dense @ v, rtol=1e-5)


# ---------------------------------------------------------------------------
# as_matrix
# ---------------------------------------------------------------------------


class TestAsMatrix:
    def test_as_matrix_matches_manual(self, getkey):
        X1 = jr.normal(getkey(), (8, 3))
        X2 = jr.normal(getkey(), (5, 3))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X1, X2, params)
        K_dense = _build_dense(_rbf_kernel, params, X1, X2)
        assert tree_allclose(op.as_matrix(), K_dense, rtol=1e-5)

    def test_symmetric_square(self, getkey):
        X = jr.normal(getkey(), (8, 3))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X, X, params)
        M = op.as_matrix()
        assert tree_allclose(M, M.T, rtol=1e-5)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


class TestTranspose:
    def test_transpose_matches_dense(self, getkey):
        X1 = jr.normal(getkey(), (8, 3))
        X2 = jr.normal(getkey(), (5, 3))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X1, X2, params)
        assert tree_allclose(op.T.as_matrix(), op.as_matrix().T, rtol=1e-5)

    def test_transpose_mv(self, getkey):
        X1 = jr.normal(getkey(), (8, 3))
        X2 = jr.normal(getkey(), (5, 3))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X1, X2, params)
        u = jr.normal(getkey(), (8,))
        K_dense = _build_dense(_rbf_kernel, params, X1, X2)
        assert tree_allclose(op.T.mv(u), K_dense.T @ u, rtol=1e-5)

    def test_symmetric_transpose_is_self(self, getkey):
        X = jr.normal(getkey(), (8, 3))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X, X, params, tags=lx.symmetric_tag)
        assert op.T is op


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_not_symmetric_by_default(self, getkey):
        X = jr.normal(getkey(), (5, 2))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X, X, params)
        assert lx.is_symmetric(op) is False

    def test_not_diagonal(self, getkey):
        X = jr.normal(getkey(), (5, 2))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X, X, params)
        assert lx.is_diagonal(op) is False


# ---------------------------------------------------------------------------
# Custom VJP / Gradients
# ---------------------------------------------------------------------------


class TestGradients:
    def test_grad_params_matches_autodiff(self, getkey):
        """Gradient via custom VJP should match naive dense autodiff."""
        X = jr.normal(getkey(), (6, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (6,))
        u = jr.normal(getkey(), (6,))

        # Custom VJP path
        def loss_custom(params):
            op = KernelOperator(_rbf_kernel, X, X, params)
            return jnp.dot(u, op.mv(v))

        grad_custom = jax.grad(loss_custom)(params)

        # Dense autodiff path
        def loss_dense(params):
            K = _build_dense(_rbf_kernel, params, X, X)
            return jnp.dot(u, K @ v)

        grad_dense = jax.grad(loss_dense)(params)

        assert tree_allclose(grad_custom, grad_dense, rtol=1e-4)

    def test_grad_v_matches_dense(self, getkey):
        """Gradient w.r.t. input vector should match dense."""
        X = jr.normal(getkey(), (6, 2))
        params = _make_params(getkey())

        def loss_custom(v):
            op = KernelOperator(_rbf_kernel, X, X, params)
            return jnp.sum(op.mv(v) ** 2)

        def loss_dense(v):
            K = _build_dense(_rbf_kernel, params, X, X)
            return jnp.sum((K @ v) ** 2)

        v = jr.normal(getkey(), (6,))
        g_custom = jax.grad(loss_custom)(v)
        g_dense = jax.grad(loss_dense)(v)
        assert tree_allclose(g_custom, g_dense, rtol=1e-4)

    def test_grad_params_rectangular(self, getkey):
        """Gradient through rectangular operator."""
        X1 = jr.normal(getkey(), (8, 2))
        X2 = jr.normal(getkey(), (5, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (5,))
        u = jr.normal(getkey(), (8,))

        def loss_custom(params):
            op = KernelOperator(_rbf_kernel, X1, X2, params)
            return jnp.dot(u, op.mv(v))

        def loss_dense(params):
            K = _build_dense(_rbf_kernel, params, X1, X2)
            return jnp.dot(u, K @ v)

        assert tree_allclose(
            jax.grad(loss_custom)(params),
            jax.grad(loss_dense)(params),
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


class TestJAX:
    def test_jit(self, getkey):
        X = jr.normal(getkey(), (8, 2))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X, X, params)
        v = jr.normal(getkey(), (8,))

        @eqx.filter_jit
        def f(op, v):
            return op.mv(v)

        assert tree_allclose(f(op, v), op.as_matrix() @ v, rtol=1e-5)

    def test_vmap(self, getkey):
        X = jr.normal(getkey(), (8, 2))
        params = _make_params(getkey())
        op = KernelOperator(_rbf_kernel, X, X, params)
        vs = jr.normal(getkey(), (4, 8))
        results = jax.vmap(op.mv)(vs)
        assert results.shape == (4, 8)
        assert tree_allclose(results[0], op.as_matrix() @ vs[0], rtol=1e-5)

    def test_grad_jit(self, getkey):
        """Gradient computation works under JIT."""
        X = jr.normal(getkey(), (6, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (6,))

        @jax.jit
        def loss_and_grad(params):
            op = KernelOperator(_rbf_kernel, X, X, params)
            return jnp.sum(op.mv(v) ** 2)

        g = jax.grad(loss_and_grad)(params)
        assert jnp.all(jnp.isfinite(g["variance"]))
        assert jnp.all(jnp.isfinite(g["lengthscale"]))


# ---------------------------------------------------------------------------
# Comparison with ImplicitKernelOperator
# ---------------------------------------------------------------------------


class TestImplicitKernelComparison:
    def test_mv_matches_implicit(self, getkey):
        """KernelOperator mv should match ImplicitKernelOperator."""
        from gaussx._operators import ImplicitKernelOperator

        X = jr.normal(getkey(), (8, 2))
        params = _make_params(getkey())
        v = jr.normal(getkey(), (8,))

        # ImplicitKernelOperator uses k(x, x') without params
        def k_fixed(x, y):
            return _rbf_kernel(params, x, y)

        implicit_op = ImplicitKernelOperator(k_fixed, X)
        kernel_op = KernelOperator(_rbf_kernel, X, X, params)

        assert tree_allclose(kernel_op.mv(v), implicit_op.mv(v), rtol=1e-5)
