"""Tests for LSMRSolver strategy."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._strategies import LSMRSolver, SLQLogdet
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
# bias term. Over 1000 random matrices |err| / SEM peaked at 3.4 (gh-303).
K_SEM = 5.0


@pytest.mark.slow
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


@pytest.mark.slow
def test_logdet_psd():
    """Stochastic logdet is within K_SEM standard errors of the exact one."""
    solver = LSMRSolver(num_probes=50, lanczos_order=20)
    op = random_pd_operator(jr.key(0), 15)
    # LSMRSolver.logdet with no key is SLQ seeded from solver.seed.
    est, sem = SLQLogdet(num_probes=50, lanczos_order=20).logdet_and_error(op)
    assert tree_allclose(solver.logdet(op), est)
    assert jnp.abs(est - dense_logdet(op)) <= K_SEM * sem


@pytest.mark.slow
def test_logdet_respects_explicit_key(getkey):
    """Passing different keys should change the stochastic estimate."""
    solver = LSMRSolver(seed=42, num_probes=5, lanczos_order=8)
    mat = random_pd_matrix(getkey(), 20)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)

    ld1 = solver.logdet(op, key=jr.PRNGKey(1))
    ld2 = solver.logdet(op, key=jr.PRNGKey(2))
    assert not tree_allclose(ld1, ld2)
