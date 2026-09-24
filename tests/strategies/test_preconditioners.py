"""Tests for the preconditioner protocol and concrete preconditioners."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx import (
    CGSolver,
    JacobiPreconditioner,
    NystromPreconditioner,
    OperatorPreconditioner,
    PartialCholeskyPreconditioner,
    linear_solve,
)
from gaussx._testing import random_pd_matrix, tree_allclose


def _psd_operator(key, n):
    mat = random_pd_matrix(key, n)
    return mat, lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)


def test_jacobi_explicit_diagonal(getkey):
    mat, op = _psd_operator(getkey(), 6)
    pre = JacobiPreconditioner(diagonal=jnp.diag(mat))
    minv = pre.as_operator(op)
    assert lx.is_positive_semidefinite(minv)
    v = jr.normal(getkey(), (6,))
    assert tree_allclose(minv.mv(v), v / jnp.diag(mat), rtol=1e-5)


def test_jacobi_extracts_diagonal_from_operator(getkey):
    mat, op = _psd_operator(getkey(), 5)
    pre = JacobiPreconditioner()  # no explicit diagonal
    minv = pre.as_operator(op)
    v = jr.normal(getkey(), (5,))
    assert tree_allclose(minv.mv(v), v / jnp.diag(mat), rtol=1e-5)


def test_jacobi_needs_diagonal_or_operator():
    with pytest.raises(ValueError, match="diagonal"):
        JacobiPreconditioner().as_operator(None)


def test_solve_with_jacobi(getkey):
    mat, op = _psd_operator(getkey(), 12)
    b = jr.normal(getkey(), (12,))
    x = linear_solve(
        op,
        b,
        solver=CGSolver(rtol=1e-8, atol=1e-8),
        preconditioner=JacobiPreconditioner(diagonal=jnp.diag(mat)),
    )
    assert tree_allclose(x, jnp.linalg.solve(mat, b), rtol=1e-4)


def test_nystrom_from_operator_solves(getkey):
    mat, op = _psd_operator(getkey(), 40)
    b = jr.normal(getkey(), (40,))
    pre = NystromPreconditioner.from_operator(op, rank=20, key=getkey())
    assert lx.is_positive_semidefinite(pre.as_operator(op))
    x = linear_solve(op, b, solver=CGSolver(rtol=1e-8, atol=1e-8), preconditioner=pre)
    assert tree_allclose(x, jnp.linalg.solve(mat, b), rtol=1e-4)


def test_nystrom_reduces_iterations():
    """A (near-)full-rank Nyström preconditioner slashes CG iterations.

    Deterministic by construction (fixed keys). A full-rank Nyström sketch of an
    SPD operator is an essentially exact inverse, so the preconditioned system
    is ``~ I`` and CG converges in a handful of steps regardless of the original
    conditioning. (CG-iteration *counts* on a partially-captured spectrum are a
    noisy proxy and were previously flaky; full rank gives a guaranteed margin.)
    """
    n = 40
    q, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(0), (n, n)))
    # Geometrically spread spectrum -> ill-conditioned (kappa ~ 1e3).
    eigs = jnp.logspace(0, 3, n)
    mat = (q * eigs) @ q.T
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    b = jr.normal(jr.PRNGKey(1), (n,))

    def cg_steps(preconditioner):
        solver = lx.CG(rtol=1e-6, atol=1e-6, max_steps=2000)
        options = {}
        if preconditioner is not None:
            options["preconditioner"] = preconditioner.as_operator(op)
        sol = lx.linear_solve(op, b, solver, options=options, throw=False)
        return sol.stats["num_steps"]

    plain = cg_steps(None)
    pre = NystromPreconditioner.from_operator(op, rank=n, key=jr.PRNGKey(2))
    preconditioned = cg_steps(pre)
    assert preconditioned < plain
    assert preconditioned <= 10


def test_partial_cholesky_disabled_returns_none(getkey):
    _, op = _psd_operator(getkey(), 5)
    pre = PartialCholeskyPreconditioner(rank=0)
    assert pre.as_operator(op) is None


def test_partial_cholesky_matches_the_woodbury_inverse_at_full_rank():
    # At full rank L Lᵀ = A exactly, so the preconditioner is (sI + A)⁻¹.
    mat = random_pd_matrix(jr.key(0), 8)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    pre = PartialCholeskyPreconditioner(rank=8, shift=0.7).as_operator(op)

    expected = jnp.linalg.inv(0.7 * jnp.eye(8) + mat)
    assert tree_allclose(pre.as_matrix(), expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("jitter", [0.0, 1e-12])
def test_partial_cholesky_rank_beyond_numerical_rank_is_finite(jitter: float):
    # gh-237: a rank-3 operator factored at rank 8. With no jitter the surplus
    # pivots are exactly zero (0/0 -> NaN); with a tiny one they are rounding
    # noise (tiny/tiny -> huge columns). The guard zeroes both, so the factor
    # captures the operator exactly and the preconditioner is (sI + K)⁻¹.
    n, r = 12, 3
    w = jr.normal(jr.key(1), (n, r))
    kernel = w @ w.T + jitter * jnp.eye(n)
    op = lx.MatrixLinearOperator(kernel, lx.positive_semidefinite_tag)

    pre = PartialCholeskyPreconditioner(rank=8, shift=1.0).as_operator(op)

    applied = pre.as_matrix()
    assert jnp.all(jnp.isfinite(applied))
    expected = jnp.linalg.inv(jnp.eye(n) + kernel)
    assert tree_allclose(applied, expected, rtol=1e-8, atol=1e-8)


def test_partial_cholesky_of_a_noiseless_kernel_preconditions_its_noisy_solve():
    # The issue's workflow: factor the noiseless K, shift by sigma^2, and use
    # the result to precondition CG on K + sigma^2 I.
    n, r, noise = 12, 3, 0.5
    w = jr.normal(jr.key(2), (n, r))
    kernel = w @ w.T
    system = lx.MatrixLinearOperator(
        kernel + noise * jnp.eye(n), lx.positive_semidefinite_tag
    )
    factor_of_kernel = PartialCholeskyPreconditioner(rank=8, shift=noise).as_operator(
        lx.MatrixLinearOperator(kernel, lx.positive_semidefinite_tag)
    )
    b = jr.normal(jr.key(3), (n,))

    x = linear_solve(
        system,
        b,
        solver=CGSolver(rtol=1e-10, atol=1e-10),
        preconditioner=OperatorPreconditioner(factor_of_kernel),
    )

    assert jnp.all(jnp.isfinite(x))
    assert tree_allclose(x, jnp.linalg.solve(kernel + noise * jnp.eye(n), b), rtol=1e-8)


def test_operator_preconditioner_callable(getkey):
    mat, op = _psd_operator(getkey(), 15)
    b = jr.normal(getkey(), (15,))
    inv_diag = 1.0 / jnp.diag(mat)
    pre = OperatorPreconditioner(lambda v: inv_diag * v)
    x = linear_solve(op, b, solver=CGSolver(rtol=1e-8, atol=1e-8), preconditioner=pre)
    assert tree_allclose(x, jnp.linalg.solve(mat, b), rtol=1e-4)


def test_operator_preconditioner_operator_tags_psd(getkey):
    """An untagged operator approximate-inverse is tagged PSD for lineax CG."""
    mat, op = _psd_operator(getkey(), 10)
    untagged = lx.DiagonalLinearOperator(1.0 / jnp.diag(mat))
    pre = OperatorPreconditioner(untagged)
    assert lx.is_positive_semidefinite(pre.as_operator(op))


def test_preconditioner_non_cg_solver_raises(getkey):
    """Preconditioning with a non-CG solver is rejected clearly."""
    from gaussx import MINRESSolver

    a = jr.normal(getkey(), (8, 8))
    sym = 0.5 * (a + a.T)
    op = lx.MatrixLinearOperator(sym, lx.symmetric_tag)
    b = jr.normal(getkey(), (8,))
    with pytest.raises(ValueError, match="only with CGSolver"):
        linear_solve(
            op,
            b,
            solver=MINRESSolver(),
            preconditioner=JacobiPreconditioner(diagonal=jnp.diag(sym)),
        )
