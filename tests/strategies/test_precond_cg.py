"""Tests for PreconditionedCGSolver strategy."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._strategies import PreconditionedCGSolver
from gaussx._testing import random_pd_matrix, tree_allclose


def test_solve_psd_no_precond(getkey):
    """Without preconditioning (rank=0), should still solve correctly."""
    solver = PreconditionedCGSolver(preconditioner_rank=0, rtol=1e-8, atol=1e-8)
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(solver.solve(op, v), expected, rtol=1e-4)


def test_logdet_psd(getkey):
    """Stochastic logdet should approximate true logdet."""
    solver = PreconditionedCGSolver(num_probes=50, lanczos_order=20)
    mat = random_pd_matrix(getkey(), 15)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    estimated = solver.logdet(op)
    true_ld = jnp.linalg.slogdet(mat)[1]
    assert jnp.abs(estimated - true_ld) < 0.1 * jnp.abs(true_ld) + 1.0
