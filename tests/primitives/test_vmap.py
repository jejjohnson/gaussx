"""Tests for vmap compatibility of gaussx primitives.

Verifies that solve, logdet, cholesky, diag, trace, and inv all work
correctly under jax.vmap with dense and structured operators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._operators import Kronecker
from gaussx._primitives import cholesky, diag, inv, logdet, solve, trace
from gaussx._testing import tree_allclose


def _make_psd(key, n):
    A = jr.normal(key, (n, n))
    return A @ A.T + 0.5 * jnp.eye(n)


class TestVmapSolve:
    def test_vmap_over_vectors(self, getkey):
        n = 5
        K = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        B = jr.normal(getkey(), (8, n))

        xs = jax.vmap(lambda b: solve(op, b))(B)
        expected = jnp.linalg.solve(K, B.T).T

        assert xs.shape == (8, n)
        assert tree_allclose(xs, expected, rtol=1e-5)

    def test_vmap_over_operators(self, getkey):
        n = 4
        Ks = jnp.stack([_make_psd(getkey(), n) for _ in range(5)])
        bs = jr.normal(getkey(), (5, n))

        def solve_one(K_i, b_i):
            op = lx.MatrixLinearOperator(K_i, lx.positive_semidefinite_tag)
            return solve(op, b_i)

        xs = jax.vmap(solve_one)(Ks, bs)
        expected = jax.vmap(jnp.linalg.solve)(Ks, bs)

        assert xs.shape == (5, n)
        assert tree_allclose(xs, expected, rtol=1e-5)

    def test_vmap_solve_columns(self, getkey):
        """Matrix RHS via vmap over columns."""
        n = 5
        K = _make_psd(getkey(), n)
        op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        B = jr.normal(getkey(), (n, 3))

        X = jax.vmap(lambda col: solve(op, col), in_axes=1, out_axes=1)(B)
        expected = jnp.linalg.solve(K, B)

        assert X.shape == (n, 3)
        assert tree_allclose(X, expected, rtol=1e-5)

    def test_vmap_solve_kronecker(self, getkey):
        A = _make_psd(getkey(), 2)
        B = _make_psd(getkey(), 3)
        A_op = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
        B_op = lx.MatrixLinearOperator(B, lx.positive_semidefinite_tag)
        kron = Kronecker(A_op, B_op)
        bs = jr.normal(getkey(), (10, 6))

        xs = jax.vmap(lambda b: solve(kron, b))(bs)
        expected = jnp.linalg.solve(kron.as_matrix(), bs.T).T

        assert xs.shape == (10, 6)
        assert tree_allclose(xs, expected, rtol=1e-4)


class TestVmapLogdet:
    def test_vmap_over_operators(self, getkey):
        n = 4
        Ks = jnp.stack([_make_psd(getkey(), n) for _ in range(5)])

        def ld_one(K_i):
            op = lx.MatrixLinearOperator(K_i, lx.positive_semidefinite_tag)
            return logdet(op)

        lds = jax.vmap(ld_one)(Ks)
        expected = jax.vmap(lambda K: jnp.linalg.slogdet(K)[1])(Ks)

        assert lds.shape == (5,)
        assert tree_allclose(lds, expected, rtol=1e-5)


class TestVmapCholesky:
    def test_vmap_cholesky_mv(self, getkey):
        n = 4
        Ks = jnp.stack([_make_psd(getkey(), n) for _ in range(5)])
        bs = jr.normal(getkey(), (5, n))

        def chol_mv(K_i, b_i):
            op = lx.MatrixLinearOperator(K_i, lx.positive_semidefinite_tag)
            L = cholesky(op)
            return L.mv(b_i)

        results = jax.vmap(chol_mv)(Ks, bs)
        expected = jax.vmap(lambda K, b: jnp.linalg.cholesky(K) @ b)(Ks, bs)

        assert results.shape == (5, n)
        assert tree_allclose(results, expected, rtol=1e-5)


class TestVmapDiag:
    def test_vmap_diag(self, getkey):
        n = 4
        Ks = jnp.stack([_make_psd(getkey(), n) for _ in range(5)])

        def diag_one(K_i):
            op = lx.MatrixLinearOperator(K_i, lx.positive_semidefinite_tag)
            return diag(op)

        ds = jax.vmap(diag_one)(Ks)
        expected = jax.vmap(jnp.diag)(Ks)

        assert ds.shape == (5, n)
        assert tree_allclose(ds, expected)


class TestVmapTrace:
    def test_vmap_trace(self, getkey):
        n = 4
        Ks = jnp.stack([_make_psd(getkey(), n) for _ in range(5)])

        def trace_one(K_i):
            op = lx.MatrixLinearOperator(K_i, lx.positive_semidefinite_tag)
            return trace(op)

        ts = jax.vmap(trace_one)(Ks)
        expected = jax.vmap(jnp.trace)(Ks)

        assert ts.shape == (5,)
        assert tree_allclose(ts, expected)


class TestVmapInv:
    def test_vmap_inv_mv(self, getkey):
        n = 4
        Ks = jnp.stack([_make_psd(getkey(), n) for _ in range(5)])
        bs = jr.normal(getkey(), (5, n))

        def inv_mv(K_i, b_i):
            op = lx.MatrixLinearOperator(K_i, lx.positive_semidefinite_tag)
            return inv(op).mv(b_i)

        results = jax.vmap(inv_mv)(Ks, bs)
        expected = jax.vmap(jnp.linalg.solve)(Ks, bs)

        assert results.shape == (5, n)
        assert tree_allclose(results, expected, rtol=1e-4)
