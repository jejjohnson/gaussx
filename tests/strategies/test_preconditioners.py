"""Tests for the preconditioner protocol and concrete preconditioners."""

from __future__ import annotations

import itertools

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
    randomized_nystrom,
)
from gaussx._einx import einsum
from gaussx._linalg._symmetrize import symmetrize
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_nystrom_from_operator_solves():
    mat = random_pd_matrix(jr.key(0), 40)
    noise = 0.1
    system = lx.MatrixLinearOperator(
        mat + noise * jnp.eye(40), lx.positive_semidefinite_tag
    )
    b = jr.normal(jr.key(1), (40,))
    pre = NystromPreconditioner.from_operator(
        lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag),
        rank=20,
        shift=noise,
        key=jr.key(2),
    )
    assert lx.is_positive_semidefinite(pre.as_operator())
    x = linear_solve(
        system, b, solver=CGSolver(rtol=1e-10, atol=1e-10), preconditioner=pre
    )
    assert tree_allclose(x, jnp.linalg.solve(mat + noise * jnp.eye(40), b), rtol=1e-6)


@pytest.mark.slow
def test_nystrom_matches_the_ftu_formula():
    # P⁻¹ = (λ̂_l + μ) U (Λ̂ + μI)⁻¹ Uᵀ + (I − UUᵀ), with U, Λ̂ from
    # randomized_nystrom on the PSD part only.
    mat = random_pd_matrix(jr.key(0), 30)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    pre = NystromPreconditioner.from_operator(op, rank=10, shift=0.5, key=jr.key(1))
    approx = randomized_nystrom(op, 10, key=jr.key(1))
    U, lam = approx.U, approx.d
    expected = (lam[-1] + 0.5) * einsum(U / (lam + 0.5), U, "i k, j k -> i j") + (
        jnp.eye(30) - einsum(U, U, "i k, j k -> i j")
    )
    assert tree_allclose(pre.basis, U) and tree_allclose(pre.eigenvalues, lam)
    assert tree_allclose(pre.as_operator().as_matrix(), expected, rtol=1e-10)


def test_nystrom_full_rank_is_a_scaled_exact_inverse():
    # #345: built from K with shift σ², at full rank P⁻¹ (K + σ²I) is
    # (λ̂_n + σ²) I, so the noise is not counted twice.
    n, noise = 40, 1e-2
    q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (n, n)))
    kernel = einsum(q * jnp.logspace(-3, 3, n), q, "i k, j k -> i j")
    pre = NystromPreconditioner.from_operator(
        lx.MatrixLinearOperator(kernel, lx.positive_semidefinite_tag),
        rank=n,
        shift=noise,
        key=jr.key(1),
    )
    product = pre.as_operator().as_matrix() @ (kernel + noise * jnp.eye(n))
    scale = pre.eigenvalues[-1] + noise
    assert tree_allclose(product / scale, jnp.eye(n), atol=1e-8)


def _cg_steps(system, b, pre, tol=1e-8):
    options = {} if pre is None else {"preconditioner": pre.as_operator()}
    sol = lx.linear_solve(
        system,
        b,
        lx.CG(rtol=tol, atol=tol, max_steps=20000),
        options=options,
        throw=False,
    )
    return int(sol.stats["num_steps"])


def _issue_354_system(name):
    """The three spectra of the #354 reproduction, split as (K, σ²)."""
    n = 200
    q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (n, n)))

    def spectral(ev):
        return symmetrize(einsum(q * ev, q, "i k, j k -> i j"))

    if name == "logspace":
        return spectral(jnp.logspace(0, 4, n) - 1.0), 1.0
    if name == "exp":
        return spectral(1e3 * jnp.exp(-0.25 * jnp.arange(n))), 1e-2
    x = jnp.sort(jr.uniform(jr.key(3), (n,)) * 10)
    return jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2), 1e-2


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "rank", "bound"),
    # #354 acceptance criteria (None: at most the unpreconditioned count).
    # The old Rayleigh-Ritz version took 776-826, 2251-2640, 9231-10026 and
    # 199-203 steps on these spectra (to 1e-8, keys 0-2). The rewrite takes
    # 567-585 (plain 678), 121-127 (plain 465), 24-26 and 5 (plain 43).
    [("logspace", 20, None), ("exp", 20, 150), ("exp", 40, 60), ("rbf", 20, 40)],
)
def test_nystrom_below_full_rank_never_slows_cg(name, rank, bound):
    kernel, noise = _issue_354_system(name)
    n = kernel.shape[0]
    psd = lx.positive_semidefinite_tag
    system = lx.MatrixLinearOperator(kernel + noise * jnp.eye(n), psd)
    b = jr.normal(jr.key(1), (n,))
    plain = _cg_steps(system, b, None)
    for seed in range(3):
        pre = NystromPreconditioner.from_operator(
            lx.MatrixLinearOperator(kernel, psd), rank, shift=noise, key=jr.key(seed)
        )
        assert _cg_steps(system, b, pre) <= (plain if bound is None else bound)


def test_nystrom_builds_once():
    # The build applies K at construction; as_operator never again.
    n, rank = 30, 10
    x = jnp.linspace(0.0, 15.0, n)
    kernel = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2)
    count = [0]
    K = _counting_operator(kernel, count)
    pre = NystromPreconditioner.from_operator(K, rank=rank, shift=1e-2)
    assert count[0] == rank
    count[0] = 0
    pre.as_operator(K).mv(jnp.ones(n))
    assert count[0] == 0


def test_nystrom_jit_and_float32():
    x = jnp.linspace(0.0, 10.0, 200, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2)
    op = lx.MatrixLinearOperator(kernel, lx.positive_semidefinite_tag)
    pre = jax.jit(
        lambda o: NystromPreconditioner.from_operator(
            o, 40, shift=jnp.float32(1e-2), key=jr.key(0)
        )
    )(op)
    assert pre.basis.dtype == jnp.float32 and pre.shift.dtype == jnp.float32
    applied = pre.as_operator().mv(jnp.ones(200, dtype=jnp.float32))
    assert applied.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(applied)))


@pytest.mark.parametrize("shift", [0.0, -1.0])
def test_nystrom_rejects_a_non_positive_shift(shift):
    op = lx.MatrixLinearOperator(jnp.eye(4), lx.positive_semidefinite_tag)
    with pytest.raises(ValueError, match="shift"):
        NystromPreconditioner.from_operator(op, 2, shift=shift)


def _matern32(n, lengthscale=1.0):
    x = jnp.sort(jr.uniform(jr.key(0), (n,)) * 10)
    r = jnp.sqrt(3.0) * jnp.abs(einx.subtract("i, j -> i j", x, x)) / lengthscale
    return (1 + r) * jnp.exp(-r)


@pytest.mark.slow
def test_nystrom_cg_iterations_on_matern32_do_not_grow_below_full_rank():
    # #354 regression (roadmap §6, G13): on K + σ²I for a Matérn-3/2 kernel
    # with n = 5000, CG iterations are non-increasing in rank and below the
    # unpreconditioned count at every rank (d_eff(σ²) ≈ 113). To 1e-6, plain
    # CG takes 469 steps; ranks 25-400 take 102, 41, 16, 8, 5. The old
    # Rayleigh-Ritz version (built from the system K + σ²I) took 3211, 4999,
    # 7397, 10653, 12694, so it fails both assertions.
    n, noise = 5000, 1e-2
    kernel = _matern32(n)
    psd = lx.positive_semidefinite_tag
    system = lx.MatrixLinearOperator(kernel + noise * jnp.eye(n), psd)
    b = jr.normal(jr.key(1), (n,))
    plain = _cg_steps(system, b, None, tol=1e-6)
    steps = [
        _cg_steps(
            system,
            b,
            NystromPreconditioner.from_operator(
                lx.MatrixLinearOperator(kernel, psd), rank, shift=noise, key=jr.key(0)
            ),
            tol=1e-6,
        )
        for rank in (25, 50, 100, 200, 400)
    ]
    assert all(s < plain for s in steps)
    assert all(a >= b for a, b in itertools.pairwise(steps))


def test_partial_cholesky_disabled_returns_none(getkey):
    _, op = _psd_operator(getkey(), 5)
    pre = PartialCholeskyPreconditioner(rank=0)
    assert pre.as_operator(op) is None


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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
