"""Tests for MINRESSolver strategy."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._strategies import MINRESSolver
from gaussx._testing import random_pd_matrix, tree_allclose


# -------------------------------------------------------------------
# Solve — PSD systems (should match CG/direct)
# -------------------------------------------------------------------


class TestSolvePSD:
    def test_solve_psd(self, getkey):
        """MINRES should converge on PSD systems."""
        solver = MINRESSolver(rtol=1e-8, atol=1e-8)
        mat = random_pd_matrix(getkey(), 5)
        op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
        v = jr.normal(getkey(), (5,))
        expected = jnp.linalg.solve(mat, v)
        assert tree_allclose(solver.solve(op, v), expected, rtol=1e-3)

    def test_solve_diagonal(self, getkey):
        solver = MINRESSolver(rtol=1e-8, atol=1e-8)
        d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
        op = lx.TaggedLinearOperator(
            lx.DiagonalLinearOperator(d), lx.positive_semidefinite_tag
        )
        v = jr.normal(getkey(), (4,))
        expected = v / d
        assert tree_allclose(solver.solve(op, v), expected, rtol=1e-3)

    def test_solve_larger_psd(self, getkey):
        """Convergence on a larger PSD system."""
        solver = MINRESSolver(rtol=1e-6, atol=1e-6, max_steps=500)
        mat = random_pd_matrix(getkey(), 20)
        op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
        v = jr.normal(getkey(), (20,))
        expected = jnp.linalg.solve(mat, v)
        assert tree_allclose(solver.solve(op, v), expected, rtol=1e-3)


# -------------------------------------------------------------------
# Solve — symmetric indefinite (CG would fail here)
# -------------------------------------------------------------------


class TestSolveIndefinite:
    def test_indefinite_symmetric(self, getkey):
        """MINRES should handle symmetric indefinite systems."""
        solver = MINRESSolver(rtol=1e-8, atol=1e-8, max_steps=500)
        N = 6
        A = jr.normal(getkey(), (N, N))
        mat = A + A.T  # symmetric but not necessarily PD
        # Ensure it's actually indefinite by forcing eigenvalues
        eigvals = jnp.linalg.eigvalsh(mat)
        # Add a shift to make it invertible if singular
        mat = mat + 0.1 * jnp.eye(N) * jnp.sign(jnp.mean(eigvals))
        # Make it indefinite: flip sign of some eigenvalues
        mat = mat - 2.0 * jnp.median(jnp.abs(eigvals)) * jnp.eye(N)

        op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
        v = jr.normal(getkey(), (N,))
        expected = jnp.linalg.solve(mat, v)
        result = solver.solve(op, v)
        assert tree_allclose(result, expected, rtol=1e-2, atol=1e-4)

    def test_negative_definite(self, getkey):
        """MINRES should work on negative definite systems."""
        solver = MINRESSolver(rtol=1e-8, atol=1e-8)
        mat = -random_pd_matrix(getkey(), 5)  # negative definite
        op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
        v = jr.normal(getkey(), (5,))
        expected = jnp.linalg.solve(mat, v)
        assert tree_allclose(solver.solve(op, v), expected, rtol=1e-3)


# -------------------------------------------------------------------
# Shifted MINRES
# -------------------------------------------------------------------


class TestShiftedMINRES:
    def test_shifted_matches_direct(self, getkey):
        """Shifted MINRES: (A + shift I) x = b."""
        shift = 2.0
        solver = MINRESSolver(rtol=1e-8, atol=1e-8, shift=shift)
        mat = random_pd_matrix(getkey(), 5)
        op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
        v = jr.normal(getkey(), (5,))

        shifted_mat = mat + shift * jnp.eye(5)
        expected = jnp.linalg.solve(shifted_mat, v)
        assert tree_allclose(solver.solve(op, v), expected, rtol=1e-3)


# -------------------------------------------------------------------
# Logdet
# -------------------------------------------------------------------


class TestLogdet:
    def test_logdet_psd(self, getkey):
        """Stochastic logdet should be reasonable for PSD."""
        solver = MINRESSolver(num_probes=50, lanczos_order=20)
        mat = random_pd_matrix(getkey(), 20)
        op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
        key = jr.PRNGKey(42)
        estimated = solver.logdet(op, key=key)
        true_ld = jnp.linalg.slogdet(mat)[1]
        assert jnp.abs(estimated - true_ld) < 0.1 * jnp.abs(true_ld) + 1.0

    def test_logdet_respects_shift(self, getkey):
        """Shifted solve/logdet pair should target the same matrix."""
        shift = 1.5
        solver = MINRESSolver(shift=shift, num_probes=50, lanczos_order=20)
        mat = random_pd_matrix(getkey(), 20)
        op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)

        estimated = solver.logdet(op, key=jr.PRNGKey(0))
        shifted_mat = mat + shift * jnp.eye(mat.shape[0])
        true_ld = jnp.linalg.slogdet(shifted_mat)[1]
        assert jnp.abs(estimated - true_ld) < 0.1 * jnp.abs(true_ld) + 1.0

    def test_logdet_indefinite_uses_logabsdet(self, getkey):
        """Indefinite symmetric matrices should return log|det(A)|."""
        solver = MINRESSolver(num_probes=100, lanczos_order=6)
        diag = jnp.array([-4.0, -2.0, 3.0, 5.0, 7.0, 11.0])
        q, _ = jnp.linalg.qr(jr.normal(getkey(), (diag.shape[0], diag.shape[0])))
        mat = q @ jnp.diag(diag) @ q.T
        op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)

        estimated = solver.logdet(op, key=jr.PRNGKey(0))
        true_ld = jnp.linalg.slogdet(mat)[1]
        assert jnp.isfinite(estimated)
        assert jnp.abs(estimated - true_ld) < 0.1 * jnp.abs(true_ld) + 1.0


# -------------------------------------------------------------------
# JIT
# -------------------------------------------------------------------


class TestJIT:
    def test_filter_jit_solve(self, getkey):
        solver = MINRESSolver(rtol=1e-6, atol=1e-6)
        mat = random_pd_matrix(getkey(), 4)
        op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
        v = jr.normal(getkey(), (4,))

        @eqx.filter_jit
        def f(op, v):
            return solver.solve(op, v)

        expected = jnp.linalg.solve(mat, v)
        assert tree_allclose(f(op, v), expected, rtol=1e-3)

    def test_zero_rhs(self, getkey):
        """Zero RHS should return zero solution."""
        solver = MINRESSolver(rtol=1e-8, atol=1e-8)
        mat = random_pd_matrix(getkey(), 4)
        op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
        v = jnp.zeros(4)
        result = solver.solve(op, v)
        assert tree_allclose(result, jnp.zeros(4), atol=1e-8)


# -------------------------------------------------------------------
# Gradient
# -------------------------------------------------------------------


class TestGradient:
    def test_grad_through_solve(self, getkey):
        """Gradients should flow through MINRES solve."""
        solver = MINRESSolver(rtol=1e-6, atol=1e-6)

        def loss(v):
            mat = random_pd_matrix(jr.PRNGKey(0), 4)
            op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
            return jnp.sum(solver.solve(op, v) ** 2)

        v = jr.normal(getkey(), (4,))
        g = jax.grad(loss)(v)
        assert jnp.all(jnp.isfinite(g))
        assert g.shape == (4,)
