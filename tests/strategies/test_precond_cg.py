"""Tests for PreconditionedCGSolver strategy."""

from __future__ import annotations

import einx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import PartialCholeskyPreconditioner
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


def test_solve_psd_with_precond(getkey):
    """The preconditioned branch should still converge to the correct solve."""
    solver = PreconditionedCGSolver(
        preconditioner_rank=3,
        rtol=1e-8,
        atol=1e-8,
        max_steps=2000,
    )
    mat = random_pd_matrix(getkey(), 6)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (6,))
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


def test_logdet_respects_explicit_key(getkey):
    """Passing different keys should change the stochastic estimate."""
    solver = PreconditionedCGSolver(seed=42, num_probes=5, lanczos_order=8)
    mat = random_pd_matrix(getkey(), 20)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)

    ld1 = solver.logdet(op, key=jr.PRNGKey(1))
    ld2 = solver.logdet(op, key=jr.PRNGKey(2))
    assert not tree_allclose(ld1, ld2)


def _rbf_system():
    """The #345 reproduction: an RBF kernel on 200 points plus 1e-2 noise."""
    n, noise = 200, 1e-2
    x = jnp.sort(jr.uniform(jr.key(0), (n,)) * 10)
    kernel = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2)
    return kernel, noise


def test_full_rank_preconditioner_is_exactly_the_inverse():
    # #345: the noise was counted twice, giving inv(K + 2σ²I) at full rank.
    kernel, noise = _rbf_system()
    n = kernel.shape[0]
    psd = lx.positive_semidefinite_tag
    system = kernel + noise * jnp.eye(n)
    expected = jnp.linalg.inv(system)

    lazy = PartialCholeskyPreconditioner(rank=n, shift=noise).as_operator(
        lx.MatrixLinearOperator(system, psd)
    )
    built = PartialCholeskyPreconditioner.from_operator(
        lx.MatrixLinearOperator(kernel, psd), rank=n, shift=noise
    ).as_operator()

    assert lazy is not None
    for pre in (lazy, built):
        P = pre.as_matrix()
        assert float(jnp.max(jnp.abs(P - expected)) / jnp.max(jnp.abs(P))) < 1e-10


@pytest.mark.parametrize("built", [False, True])
def test_rank_20_preconditioner_takes_few_cg_steps(built):
    # #345 measured 19 steps here with the double-counted noise; 4 without.
    kernel, noise = _rbf_system()
    n = kernel.shape[0]
    psd = lx.positive_semidefinite_tag
    system = lx.MatrixLinearOperator(kernel + noise * jnp.eye(n), psd)
    if built:
        pre = PartialCholeskyPreconditioner.from_operator(
            lx.MatrixLinearOperator(kernel, psd), rank=20, shift=noise
        ).as_operator()
    else:
        pre = PartialCholeskyPreconditioner(rank=20, shift=noise).as_operator(system)
    b = jr.normal(jr.key(1), (n,))
    sol = lx.linear_solve(
        system,
        b,
        lx.CG(rtol=1e-8, atol=1e-8, max_steps=5000),
        options={"preconditioner": pre},
        throw=False,
    )
    assert int(sol.stats["num_steps"]) <= 6


def test_solver_uses_a_prebuilt_preconditioner():
    kernel, noise = _rbf_system()
    n = kernel.shape[0]
    psd = lx.positive_semidefinite_tag
    system = kernel + noise * jnp.eye(n)
    pre = PartialCholeskyPreconditioner.from_operator(
        lx.MatrixLinearOperator(kernel, psd), rank=20, shift=noise
    )
    solver = PreconditionedCGSolver(preconditioner=pre, rtol=1e-10, atol=1e-10)
    b = jr.normal(jr.key(2), (n,))
    x = solver.solve(lx.MatrixLinearOperator(system, psd), b)
    assert tree_allclose(x, jnp.linalg.solve(system, b), rtol=1e-6)
