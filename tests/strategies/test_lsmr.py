"""Tests for LSMRSolver strategy."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from gaussx._strategies import LSMRSolver
from gaussx._testing import random_pd_matrix, tree_allclose


def test_solve_square(getkey):
    """LSMR should solve square well-conditioned systems."""
    solver = LSMRSolver(atol=1e-10, btol=1e-10, maxiter=500)
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(solver.solve(op, v), expected, rtol=1e-3)


def test_solve_with_damping(getkey):
    """LSMR with damping should solve regularized system."""
    damp = 0.5
    solver = LSMRSolver(atol=1e-10, btol=1e-10, maxiter=500, damp=damp)
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat)
    v = jr.normal(getkey(), (5,))

    # Damped solution: (A^T A + damp^2 I)^{-1} A^T b
    AtA = mat.T @ mat + damp**2 * jnp.eye(5)
    expected = jnp.linalg.solve(AtA, mat.T @ v)
    assert tree_allclose(solver.solve(op, v), expected, rtol=1e-2)


def test_solve_rectangular(getkey):
    """LSMR should solve rectangular least-squares systems."""
    solver = LSMRSolver(atol=1e-10, btol=1e-10, maxiter=500)
    mat = jr.normal(getkey(), (6, 4)) + 0.5 * jnp.ones((6, 4))
    op = lx.MatrixLinearOperator(mat)
    b = jr.normal(getkey(), (6,))

    result = solver.solve(op, b)

    # Least-squares solution: (A^T A)^{-1} A^T b
    expected = jnp.linalg.lstsq(mat, b, rcond=None)[0]
    assert result.shape == (4,)
    assert tree_allclose(result, expected, rtol=1e-2)


def test_logdet_psd(getkey):
    """Stochastic logdet should approximate true logdet."""
    solver = LSMRSolver(num_probes=50, lanczos_order=20)
    mat = random_pd_matrix(getkey(), 15)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    estimated = solver.logdet(op)
    true_ld = jnp.linalg.slogdet(mat)[1]
    assert jnp.abs(estimated - true_ld) < 0.1 * jnp.abs(true_ld) + 1.0


def test_logdet_respects_explicit_key(getkey):
    """Passing different keys should change the stochastic estimate."""
    solver = LSMRSolver(seed=42, num_probes=5, lanczos_order=8)
    mat = random_pd_matrix(getkey(), 20)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)

    ld1 = solver.logdet(op, key=jr.PRNGKey(1))
    ld2 = solver.logdet(op, key=jr.PRNGKey(2))
    assert not tree_allclose(ld1, ld2)
