"""Tests for MINRESSolver strategy."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._einx import einsum
from gaussx._linalg._symmetrize import symmetrize
from gaussx._strategies import IndefiniteSLQLogdet, MINRESSolver
from gaussx._testing import (
    dense_logdet,
    random_pd_matrix,
    random_pd_operator,
    tree_allclose,
)


# Stochastic logdet tests bound |est − exact| by the estimator's own standard
# error (gh-409, as gh-303 did for test_slq_logdet.py): the matrix and the
# probes are pinned, the strategy's logdet is checked to be that SLQ estimate,
# and lanczos_order >= n makes the quadrature exact, so there is no Lanczos
# bias term. Over 1000 random matrices |err| / SEM peaked at 3.1 for the
# indefinite estimator (gh-303).
K_SEM = 5.0


# -------------------------------------------------------------------
# Solve — PSD systems (should match CG/direct)
# -------------------------------------------------------------------


class TestSolvePSD:
    @pytest.mark.slow
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
    @pytest.mark.slow
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


def _assert_logdet_within_sem(solver, op, exact, key):
    """``solver.logdet`` is its SLQ estimate, within K_SEM SEMs of *exact*."""
    est, sem = IndefiniteSLQLogdet(
        num_probes=solver.num_probes,
        lanczos_order=solver.lanczos_order,
        shift=solver.shift,
    ).logdet_and_error(op, key=key)
    assert tree_allclose(solver.logdet(op, key=key), est)
    assert jnp.abs(est - exact) <= K_SEM * sem, (est, exact, sem)


class TestLogdet:
    @pytest.mark.slow
    def test_logdet_psd(self):
        """Stochastic logdet should be reasonable for PSD."""
        solver = MINRESSolver(num_probes=50, lanczos_order=20)
        op = random_pd_operator(jr.key(0), 20)
        _assert_logdet_within_sem(solver, op, dense_logdet(op), jr.PRNGKey(42))

    def test_logdet_respects_shift(self):
        """Shifted solve/logdet pair should target the same matrix."""
        shift = 1.5
        solver = MINRESSolver(shift=shift, num_probes=50, lanczos_order=20)
        mat = random_pd_matrix(jr.key(0), 20)
        op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
        exact = jnp.linalg.slogdet(mat + shift * jnp.eye(mat.shape[0]))[1]
        _assert_logdet_within_sem(solver, op, exact, jr.PRNGKey(0))

    @pytest.mark.slow
    def test_logdet_indefinite_uses_logabsdet(self):
        """Indefinite symmetric matrices should return log|det(A)|."""
        solver = MINRESSolver(num_probes=100, lanczos_order=6)
        diag = jnp.array([-4.0, -2.0, 3.0, 5.0, 7.0, 11.0])
        q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (diag.shape[0], diag.shape[0])))
        mat = einsum(q * diag, q, "i k, j k -> i j")
        op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
        _assert_logdet_within_sem(solver, op, dense_logdet(op), jr.PRNGKey(0))


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
    @pytest.mark.slow
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


# -------------------------------------------------------------------
# Early exit, implicit gradients and exhaustion (gh-336)
# -------------------------------------------------------------------

_M5 = jnp.array(
    [
        [4.0, 1, 0, 0, 0],
        [1, 3, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 3, 1],
        [0, 0, 0, 1, 4],
    ]
)


def _counting_operator(mat, counter):
    def mv(v):
        jax.debug.callback(lambda: counter.__setitem__(0, counter[0] + 1))
        return mat @ v

    return lx.FunctionLinearOperator(
        mv, jax.ShapeDtypeStruct((mat.shape[0],), mat.dtype), lx.symmetric_tag
    )


def _indefinite(n):
    """Symmetric indefinite, |eigenvalues| in [1, 1e4] (the gh-336 system)."""
    q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (n, n)))
    ev = jnp.concatenate([-jnp.logspace(0, 4, n // 2), jnp.logspace(0, 4, n // 2)])
    return symmetrize(einsum(q * ev, q, "i k, j k -> i j"))


def test_stops_when_converged():
    # gh-336: a 1000-step scan applied the operator 1000 times on this 5x5
    # system; MINRES converges in at most 5 steps here.
    count = [0]
    b = jnp.arange(1.0, 6.0, dtype=_M5.dtype)
    x = MINRESSolver().solve(_counting_operator(_M5, count), b)
    jax.effects_barrier()
    assert count[0] <= 10
    assert tree_allclose(x, jnp.linalg.solve(_M5, b), rtol=1e-4)


@pytest.mark.x64_only(reason="rtol=1e-6 gradient check against a dense solve")
def test_grad_matches_dense_on_indefinite_system():
    # gh-336: gradients are implicit (lineax), so they match the dense solve's
    # and do not depend on max_steps.
    A = _indefinite(20)
    b = jr.normal(jr.key(1), (20,), dtype=A.dtype)
    solver = MINRESSolver(rtol=1e-12, atol=1e-12, max_steps=200)

    def loss(A, b, iterative):
        if iterative:
            x = solver.solve(lx.MatrixLinearOperator(A, lx.symmetric_tag), b)
        else:
            x = jnp.linalg.solve(A, b)
        return jnp.sum(x**2)

    got = jax.grad(loss, argnums=(0, 1))(A, b, True)
    expected = jax.grad(loss, argnums=(0, 1))(A, b, False)
    assert tree_allclose(got, expected, rtol=1e-6, atol=1e-10)


def test_exhausting_max_steps_raises_unless_throw_false():
    # gh-336: on this system MINRES(max_steps=5) returned a relative residual
    # of 0.92 with no error.
    A = _indefinite(200)
    b = jr.normal(jr.key(1), (200,), dtype=A.dtype)
    op = lx.MatrixLinearOperator(A, lx.symmetric_tag)
    with pytest.raises(eqx.EquinoxRuntimeError):
        MINRESSolver(max_steps=5).solve(op, b)
    x = MINRESSolver(max_steps=5, throw=False).solve(op, b)
    assert jnp.all(jnp.isfinite(x))


def test_seed_changes_the_probes():
    # gh-384: MINRESSolver had no seed.
    op = random_pd_operator(jr.key(0), 10, jitter=10.0)
    s0 = MINRESSolver(num_probes=4, lanczos_order=4)
    s1 = MINRESSolver(num_probes=4, lanczos_order=4, seed=1)
    assert tree_allclose(s0.logdet(op), s0.logdet(op))
    assert not tree_allclose(s0.logdet(op), s1.logdet(op))
    assert s0.logdet(op) == IndefiniteSLQLogdet(num_probes=4, lanczos_order=4).logdet(
        op
    )
