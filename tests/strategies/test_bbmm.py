"""Tests for BBMMSolver strategy."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._strategies import BBMMSolver
from gaussx._testing import random_pd_matrix, tree_allclose


def test_solve_psd(getkey):
    bbmm = BBMMSolver(cg_tolerance=1e-8, cg_max_iter=2000)
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(bbmm.solve(op, v), expected, rtol=1e-4)


def test_solve_diagonal(getkey):
    bbmm = BBMMSolver(cg_tolerance=1e-8)
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(d), lx.positive_semidefinite_tag
    )
    v = jr.normal(getkey(), (4,))
    expected = v / d
    assert tree_allclose(bbmm.solve(op, v), expected, rtol=1e-4)


def test_logdet_psd(getkey):
    """Stochastic logdet should be within ~10% for moderate-size PSD."""
    bbmm = BBMMSolver(num_probes=50, lanczos_iter=20)
    mat = random_pd_matrix(getkey(), 20)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    estimated = bbmm.logdet(op)
    true_ld = jnp.linalg.slogdet(mat)[1]
    # Stochastic estimate — allow generous tolerance
    assert jnp.abs(estimated - true_ld) < 0.1 * jnp.abs(true_ld) + 1.0


def test_logdet_diagonal(getkey):
    """Stochastic logdet on diagonal should be reasonably accurate."""
    bbmm = BBMMSolver(num_probes=50, lanczos_iter=10)
    d = jnp.abs(jr.normal(getkey(), (10,))) + 0.5
    op = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(d), lx.positive_semidefinite_tag
    )
    estimated = bbmm.logdet(op)
    true_ld = jnp.sum(jnp.log(d))
    assert jnp.abs(estimated - true_ld) < 0.1 * jnp.abs(true_ld) + 1.0


def test_solve_and_logdet(getkey):
    """Joint solve + logdet should match individual calls."""
    bbmm = BBMMSolver(cg_tolerance=1e-8, cg_max_iter=2000, num_probes=50)
    mat = random_pd_matrix(getkey(), 8)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (8,))

    sol, ld = bbmm.solve_and_logdet(op, v)
    expected_ld = bbmm.logdet(op)
    expected_sol = jnp.linalg.solve(mat, v)

    assert tree_allclose(sol, expected_sol, rtol=1e-4)
    assert tree_allclose(ld, expected_ld)


def test_deterministic_logdet(getkey):
    """logdet should be deterministic (same seed -> same result)."""
    bbmm = BBMMSolver(seed=42, num_probes=20, lanczos_iter=15)
    mat = random_pd_matrix(getkey(), 10)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)

    ld1 = bbmm.logdet(op)
    ld2 = bbmm.logdet(op)
    assert tree_allclose(ld1, ld2)


def test_filter_jit_solve(getkey):
    bbmm = BBMMSolver(cg_tolerance=1e-6)
    mat = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (4,))

    @eqx.filter_jit
    def f(op, v):
        return bbmm.solve(op, v)

    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(f(op, v), expected, rtol=1e-4)
