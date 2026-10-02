"""Tests for the preconditioner protocol and concrete preconditioners."""

from __future__ import annotations

import einx
import jax
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
    # At full rank F Fᵀ = K exactly, so the preconditioner is (σ²I + K)⁻¹,
    # whether built once from K or lazily from the system K + σ²I (#345).
    mat = random_pd_matrix(jr.key(0), 8)
    expected = jnp.linalg.inv(0.7 * jnp.eye(8) + mat)
    psd = lx.positive_semidefinite_tag

    built = PartialCholeskyPreconditioner.from_operator(
        lx.MatrixLinearOperator(mat, psd), rank=8, shift=0.7
    ).as_operator()
    lazy = PartialCholeskyPreconditioner(rank=8, shift=0.7).as_operator(
        lx.MatrixLinearOperator(mat + 0.7 * jnp.eye(8), psd)
    )

    assert tree_allclose(built.as_matrix(), expected, rtol=1e-8, atol=1e-10)
    assert lazy is not None
    assert tree_allclose(lazy.as_matrix(), expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("pivoting", ["greedy", "random"])
@pytest.mark.parametrize("jitter", [0.0, 1e-12])
def test_partial_cholesky_rank_beyond_numerical_rank_is_finite(jitter, pivoting):
    # gh-237: a rank-3 operator factored at rank 8. With no jitter the surplus
    # pivots are exactly zero (0/0 -> NaN); with a tiny one they are rounding
    # noise (tiny/tiny -> huge columns). The guard zeroes both, so the factor
    # captures the operator exactly and the preconditioner is (sI + K)⁻¹.
    n, r = 12, 3
    w = jr.normal(jr.key(1), (n, r))
    kernel = w @ w.T + jitter * jnp.eye(n)
    psd = lx.positive_semidefinite_tag
    expected = jnp.linalg.inv(jnp.eye(n) + kernel)

    built = PartialCholeskyPreconditioner.from_operator(
        lx.MatrixLinearOperator(kernel, psd),
        rank=8,
        shift=1.0,
        pivoting=pivoting,
        key=jr.key(0),
    ).as_operator()
    lazy = PartialCholeskyPreconditioner(
        rank=8, shift=1.0, pivoting=pivoting, key=jr.key(0)
    ).as_operator(lx.MatrixLinearOperator(kernel + jnp.eye(n), psd))

    for pre in (built, lazy):
        assert pre is not None
        applied = pre.as_matrix()
        assert jnp.all(jnp.isfinite(applied))
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
    pre = PartialCholeskyPreconditioner.from_operator(
        lx.MatrixLinearOperator(kernel, lx.positive_semidefinite_tag),
        rank=8,
        shift=noise,
    )
    b = jr.normal(jr.key(3), (n,))

    x = linear_solve(
        system, b, solver=CGSolver(rtol=1e-10, atol=1e-10), preconditioner=pre
    )

    assert jnp.all(jnp.isfinite(x))
    assert tree_allclose(x, jnp.linalg.solve(kernel + noise * jnp.eye(n), b), rtol=1e-8)


def _counting_operator(mat, counter):
    """A PSD matrix-free operator that counts the columns it is applied to."""
    n = mat.shape[0]

    def mv(v):
        jax.debug.callback(
            lambda v: counter.__setitem__(0, counter[0] + v.size // n), v
        )
        return mat @ v

    return lx.FunctionLinearOperator(
        mv, jax.ShapeDtypeStruct((n,), mat.dtype), lx.positive_semidefinite_tag
    )


def test_partial_cholesky_from_operator_builds_once():
    # #371: the build applies K at construction; as_operator never again.
    n, rank, noise = 30, 10, 1e-2
    x = jnp.linspace(0.0, 15.0, n)
    kernel = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2)
    count = [0]
    K = _counting_operator(kernel, count)

    pre = PartialCholeskyPreconditioner.from_operator(K, rank=rank, shift=noise)
    build = count[0]
    assert build >= rank

    count[0] = 0
    pre.as_operator(K)
    assert count[0] == 0  # the built preconditioner ignores its argument


@pytest.mark.slow
def test_partial_cholesky_built_solves_skip_the_rebuild():
    # #371: two solves with a prebuilt preconditioner never touch K again
    # beyond CG's own matvecs; the lazy one pays its rank-20 build per solve.
    n, rank, noise = 60, 20, 1e-2
    x = jnp.linspace(0.0, 15.0, n)
    kernel = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2)
    count = [0]
    K = _counting_operator(kernel, count)
    A = lx.TaggedLinearOperator(
        K + lx.DiagonalLinearOperator(jnp.full(n, noise)),
        lx.positive_semidefinite_tag,
    )
    b1, b2 = jr.normal(jr.key(0), (2, n))

    pre = PartialCholeskyPreconditioner.from_operator(K, rank=rank, shift=noise)
    build = count[0]

    count[0] = 0
    solver = CGSolver(rtol=1e-8, atol=1e-8, preconditioner=pre)
    solver.solve(A, b1)
    solver.solve(A, b2)
    built_solves = count[0]

    count[0] = 0
    lazy = CGSolver(
        rtol=1e-8,
        atol=1e-8,
        preconditioner=PartialCholeskyPreconditioner(rank=rank, shift=noise),
    )
    lazy.solve(A, b1)
    lazy.solve(A, b2)
    lazy_solves = count[0]

    assert built_solves < build
    assert lazy_solves - built_solves >= 2 * rank


def test_partial_cholesky_built_is_a_jittable_pytree():
    mat = random_pd_matrix(jr.key(4), 10)
    psd = lx.positive_semidefinite_tag
    system = lx.MatrixLinearOperator(mat + 0.1 * jnp.eye(10), psd)
    b = jr.normal(jr.key(5), (10,))

    @jax.jit
    def build_and_solve(m, b):
        pre = PartialCholeskyPreconditioner.from_operator(
            lx.MatrixLinearOperator(m, psd), rank=5, shift=0.1, pivoting="random"
        )
        return CGSolver(rtol=1e-10, atol=1e-10, preconditioner=pre).solve(system, b)

    x = build_and_solve(mat, b)
    assert tree_allclose(x, jnp.linalg.solve(mat + 0.1 * jnp.eye(10), b), rtol=1e-8)


def test_partial_cholesky_from_operator_needs_positive_rank():
    _, op = _psd_operator(jr.key(6), 5)
    with pytest.raises(ValueError, match="rank"):
        PartialCholeskyPreconditioner.from_operator(op, rank=0, shift=1.0)


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
