"""Tests for the Toeplitz operator."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import Toeplitz
from gaussx._testing import tree_allclose


def _toeplitz_dense(column):
    """Build a dense symmetric Toeplitz matrix from first column."""
    n = column.shape[0]
    indices = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
    return column[indices]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self, getkey):
        c = jr.normal(getkey(), (5,))
        T = Toeplitz(c)
        assert T.in_size() == 5
        assert T.out_size() == 5

    def test_always_has_symmetric_tag(self, getkey):
        c = jr.normal(getkey(), (4,))
        T = Toeplitz(c)
        assert lx.symmetric_tag in T.tags

    def test_size_one(self):
        T = Toeplitz(jnp.array([3.0]))
        assert T.in_size() == 1
        assert T.out_size() == 1

    def test_size_two(self, getkey):
        c = jr.normal(getkey(), (2,))
        T = Toeplitz(c)
        assert T.in_size() == 2


# ---------------------------------------------------------------------------
# mv correctness
# ---------------------------------------------------------------------------


class TestMv:
    def test_mv_matches_dense(self, getkey):
        c = jr.normal(getkey(), (6,))
        T = Toeplitz(c)
        v = jr.normal(getkey(), (6,))
        assert tree_allclose(T.mv(v), T.as_matrix() @ v, rtol=1e-5)

    def test_mv_small(self):
        c = jnp.array([2.0, 1.0, 0.5])
        T = Toeplitz(c)
        v = jnp.array([1.0, 0.0, 0.0])
        expected = jnp.array([2.0, 1.0, 0.5])
        assert tree_allclose(T.mv(v), expected)

    def test_mv_identity_like(self):
        """Toeplitz with column [1, 0, 0, ...] is the identity."""
        n = 5
        c = jnp.zeros(n, dtype=jnp.float32).at[0].set(1.0)
        T = Toeplitz(c)
        v = jnp.arange(n, dtype=jnp.float32)
        assert tree_allclose(T.mv(v), v, atol=1e-5)

    def test_mv_constant_diagonal(self, getkey):
        """Toeplitz with column [a, 0, 0, ...] is a * I."""
        n = 4
        a = 3.5
        c = jnp.zeros(n).at[0].set(a)
        T = Toeplitz(c)
        v = jr.normal(getkey(), (n,))
        assert tree_allclose(T.mv(v), a * v)

    def test_mv_large(self, getkey):
        n = 128
        c = jr.normal(getkey(), (n,))
        T = Toeplitz(c)
        v = jr.normal(getkey(), (n,))
        assert tree_allclose(T.mv(v), _toeplitz_dense(c) @ v, rtol=1e-4)

    def test_mv_size_one(self):
        T = Toeplitz(jnp.array([3.0]))
        v = jnp.array([2.0])
        assert tree_allclose(T.mv(v), jnp.array([6.0]))

    def test_mv_size_two(self, getkey):
        c = jr.normal(getkey(), (2,))
        T = Toeplitz(c)
        v = jr.normal(getkey(), (2,))
        assert tree_allclose(T.mv(v), T.as_matrix() @ v, rtol=1e-5)


# ---------------------------------------------------------------------------
# as_matrix
# ---------------------------------------------------------------------------


class TestAsMatrix:
    def test_as_matrix_matches_manual(self):
        c = jnp.array([4.0, 1.0, 0.5])
        T = Toeplitz(c)
        expected = jnp.array([[4.0, 1.0, 0.5], [1.0, 4.0, 1.0], [0.5, 1.0, 4.0]])
        assert tree_allclose(T.as_matrix(), expected)

    def test_as_matrix_is_symmetric(self, getkey):
        c = jr.normal(getkey(), (5,))
        T = Toeplitz(c)
        M = T.as_matrix()
        assert tree_allclose(M, M.T)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


class TestTranspose:
    def test_transpose_is_self(self, getkey):
        c = jr.normal(getkey(), (5,))
        T = Toeplitz(c)
        # Symmetric Toeplitz: T.T is T
        assert tree_allclose(T.T.as_matrix(), T.as_matrix())

    def test_transpose_mv(self, getkey):
        c = jr.normal(getkey(), (5,))
        T = Toeplitz(c)
        v = jr.normal(getkey(), (5,))
        assert tree_allclose(T.T.mv(v), T.mv(v))


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestTags:
    def test_is_symmetric(self, getkey):
        T = Toeplitz(jr.normal(getkey(), (4,)))
        assert lx.is_symmetric(T) is True

    def test_is_not_diagonal(self, getkey):
        T = Toeplitz(jr.normal(getkey(), (4,)))
        assert lx.is_diagonal(T) is False

    def test_psd_when_tagged(self, getkey):
        c = jr.normal(getkey(), (4,))
        T = Toeplitz(c, tags=lx.positive_semidefinite_tag)
        assert lx.is_positive_semidefinite(T) is True

    def test_not_psd_by_default(self, getkey):
        T = Toeplitz(jr.normal(getkey(), (4,)))
        assert lx.is_positive_semidefinite(T) is False


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------


class TestJAX:
    def test_jit(self, getkey):
        c = jr.normal(getkey(), (5,))
        T = Toeplitz(c)
        v = jr.normal(getkey(), (5,))

        @eqx.filter_jit
        def f(op, v):
            return op.mv(v)

        assert tree_allclose(f(T, v), T.as_matrix() @ v, rtol=1e-5)

    def test_vmap(self, getkey):
        c = jr.normal(getkey(), (5,))
        T = Toeplitz(c)
        vs = jr.normal(getkey(), (8, 5))
        results = jax.vmap(T.mv)(vs)
        assert results.shape == (8, 5)
        assert tree_allclose(results[0], T.as_matrix() @ vs[0], rtol=1e-5)

    def test_grad_through_column(self, getkey):
        v = jr.normal(getkey(), (5,))

        def loss(c):
            T = Toeplitz(c)
            return jnp.sum(T.mv(v) ** 2)

        c = jr.normal(getkey(), (5,))
        g = jax.grad(loss)(c)
        assert g.shape == (5,)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_vector(self, getkey):
        c = jr.normal(getkey(), (5,))

        def loss(v):
            T = Toeplitz(c)
            return jnp.sum(T.mv(v) ** 2)

        v = jr.normal(getkey(), (5,))
        g = jax.grad(loss)(v)
        assert g.shape == (5,)
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Logdet (via dense fallback)
# ---------------------------------------------------------------------------


class TestLogdet:
    def test_logdet_matches_dense(self, getkey):
        from gaussx._primitives import logdet

        # Use a PSD Toeplitz for a valid logdet
        c = jnp.array([5.0, 1.0, 0.5, 0.1])
        T = Toeplitz(c)
        ld = logdet(T)
        expected = jnp.linalg.slogdet(T.as_matrix())[1]
        assert tree_allclose(ld, expected, rtol=1e-4)
