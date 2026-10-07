"""Tests for BBMMSolver strategy."""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._strategies import BBMMSolver, SLQLogdet
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


@pytest.mark.slow
def test_logdet_psd():
    """Stochastic logdet is within K_SEM standard errors of the exact one."""
    bbmm = BBMMSolver(num_probes=50, lanczos_iter=20)
    op = random_pd_operator(jr.key(0), 20)
    # BBMMSolver.logdet with no key is SLQ seeded from bbmm.seed.
    est, sem = SLQLogdet(num_probes=50, lanczos_order=20).logdet_and_error(op)
    assert tree_allclose(bbmm.logdet(op), est)
    assert jnp.abs(est - dense_logdet(op)) <= K_SEM * sem


@pytest.mark.slow
def test_logdet_diagonal():
    """On a diagonal operator SLQ with sign probes is exact, not stochastic.

    Each probe gives zᵀ log(D) z = Σ log dᵢ when zᵢ² = 1, and full-order
    Lanczos is exact quadrature, so only round-off is left.
    """
    bbmm = BBMMSolver(num_probes=50, lanczos_iter=10)
    d = jnp.abs(jr.normal(jr.key(0), (10,))) + 0.5
    op = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(d), lx.positive_semidefinite_tag
    )
    assert tree_allclose(bbmm.logdet(op), jnp.sum(jnp.log(d)), rtol=1e-8)


@pytest.mark.slow
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


@pytest.mark.slow
def test_deterministic_logdet(getkey):
    """logdet should be deterministic (same seed -> same result)."""
    bbmm = BBMMSolver(seed=42, num_probes=20, lanczos_iter=15)
    mat = random_pd_matrix(getkey(), 10)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)

    ld1 = bbmm.logdet(op)
    ld2 = bbmm.logdet(op)
    assert tree_allclose(ld1, ld2)


@pytest.mark.slow
def test_logdet_respects_explicit_key(getkey):
    """Passing different keys should change the stochastic estimate."""
    bbmm = BBMMSolver(seed=42, num_probes=5, lanczos_iter=8)
    mat = random_pd_matrix(getkey(), 20)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)

    ld1 = bbmm.logdet(op, key=jr.PRNGKey(1))
    ld2 = bbmm.logdet(op, key=jr.PRNGKey(2))
    assert not tree_allclose(ld1, ld2)


@pytest.mark.slow
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


def test_solve_and_logdet_costs_solve_plus_logdet():
    # gh-396: solve_and_logdet is documented as solve then logdet, with no
    # shared matvecs, so it applies the operator to exactly as many columns
    # as the two separately (pinned data: an RBF Gram plus noise).
    n = 40
    x = jnp.linspace(0.0, 8.0, n)
    K = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2) + 0.1 * jnp.eye(n)
    columns = [0]

    def mv(v):
        jax.debug.callback(
            lambda v: columns.__setitem__(0, columns[0] + v.size // n), v
        )
        return K @ v

    op = lx.FunctionLinearOperator(
        mv, jax.ShapeDtypeStruct((n,), K.dtype), lx.positive_semidefinite_tag
    )
    y = jr.normal(jr.key(0), (n,), dtype=K.dtype)
    bbmm = BBMMSolver(lanczos_iter=10, num_probes=4)

    def count(f):
        columns[0] = 0
        jax.block_until_ready(f())
        jax.effects_barrier()
        return columns[0]

    solve = count(lambda: bbmm.solve(op, y))
    logdet = count(lambda: bbmm.logdet(op))
    assert solve > 0 and logdet > 0
    assert count(lambda: bbmm.solve_and_logdet(op, y)) == solve + logdet
