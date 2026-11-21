"""linear_solve_with_info reports how a solve ended instead of raising (gh-858).

On a 2-D Poisson problem the solution is `linear_solve`'s and the step count
is lineax's; too few steps and a CG breakdown on an indefinite matrix are
reported as unconverged without an error; a direct strategy takes no steps;
and the report survives ``jit`` and ``vmap`` without changing the gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx


def _poisson_2d(m: int = 16) -> np.ndarray:
    a = 2.0 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)
    return np.kron(a, np.eye(m)) + np.kron(np.eye(m), a)


def _system(sign: float = 1.0):
    matrix = sign * _poisson_2d()
    dtype = jnp.result_type(float)
    tag = lx.positive_semidefinite_tag if sign > 0 else lx.negative_semidefinite_tag
    operator = lx.MatrixLinearOperator(
        jnp.asarray(matrix, dtype=dtype), (lx.symmetric_tag, tag)
    )
    b = jnp.asarray(np.random.default_rng(0).standard_normal(256), dtype=dtype)
    return matrix, operator, b


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive", "negative"])
def test_solution_and_steps_are_linear_solves(sign):
    matrix, operator, b = _system(sign)
    solver = gaussx.CGSolver()
    x, info = gaussx.linear_solve_with_info(operator, b, solver=solver)
    assert np.array_equal(x, gaussx.linear_solve(operator, b, solver=solver))
    # lineax's own count, on the system linear_solve hands it (−A for sign −1)
    positive = lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(jnp.asarray(sign * matrix, dtype=b.dtype)),
        lx.positive_semidefinite_tag,
    )
    if b.dtype == jnp.float64:
        # CGSolver's float64 defaults: rtol = atol = 1e-5, no rescaling
        cg = lx.CG(rtol=1e-5, atol=1e-5, max_steps=1000)
        lineax = lx.linear_solve(positive, sign * b, cg)
        assert int(info.iterations) == int(lineax.stats["num_steps"])
    assert bool(info.converged)
    residual = np.linalg.norm(np.asarray(b) - matrix @ np.asarray(x))
    assert float(info.residual_norm) == pytest.approx(residual, rel=1e-3)


def test_too_few_steps_are_reported_not_raised():
    _, operator, b = _system()
    solver = gaussx.CGSolver(max_steps=5)
    with pytest.raises(Exception, match="maximum number of solver steps"):
        gaussx.linear_solve(operator, b, solver=solver)
    _, info = gaussx.linear_solve_with_info(operator, b, solver=solver)
    assert int(info.iterations) == 5
    assert not bool(info.converged)


@pytest.mark.parametrize(
    "solver",
    [
        gaussx.CGSolver(max_steps=50),
        gaussx.AutoSolver(size_threshold=1),
        gaussx.ComposedSolver(gaussx.CGSolver(max_steps=50), gaussx.SLQLogdet()),
    ],
    ids=["cg", "auto", "composed"],
)
def test_cg_breakdown_on_an_indefinite_matrix_is_unconverged(solver):
    # bᵀ A b = 0 for this indefinite A, so the first CG step divides by zero;
    # tagged PSD, it is the mis-tagged operator jejjohnson/gridax#307 found
    a = lx.MatrixLinearOperator(
        jnp.diag(jnp.asarray([1.0, -1.0, 2.0, -2.0])), lx.positive_semidefinite_tag
    )
    x, info = gaussx.linear_solve_with_info(a, jnp.ones(4), solver=solver)
    assert not bool(info.converged)
    assert not bool(jnp.all(jnp.isfinite(x)))


def test_minres_reports_its_steps():
    _, operator, b = _system()
    _, info = gaussx.linear_solve_with_info(
        operator, b, solver=gaussx.MINRESSolver(max_steps=3)
    )
    assert int(info.iterations) == 3
    assert not bool(info.converged)


def test_direct_strategy_takes_no_steps():
    _, operator, b = _system()
    _, info = gaussx.linear_solve_with_info(operator, b, solver=gaussx.DenseSolver())
    assert int(info.iterations) == 0
    assert bool(info.converged)


def test_jit_and_vmap_over_rhs():
    _, operator, b = _system()
    batch = jnp.stack([b, 2 * b, -b])
    solver = gaussx.CGSolver()

    def solve(rhs):
        return gaussx.linear_solve_with_info(operator, rhs, solver=solver)

    xs, infos = jax.vmap(solve)(batch)
    x0, info0 = jax.jit(solve)(b)
    tol = 1e-4 if b.dtype == jnp.float32 else 1e-10
    assert jnp.allclose(xs[0], x0, rtol=tol, atol=tol)
    assert infos.converged.shape == (3,)
    assert bool(jnp.all(infos.converged))
    assert int(info0.iterations) == int(infos.iterations[0])


def test_gradient_is_unchanged_by_the_report():
    _, operator, b = _system()
    solver = gaussx.CGSolver()
    weights = jnp.asarray(np.random.default_rng(1).standard_normal(256), b.dtype)

    def with_info(rhs):
        return weights @ gaussx.linear_solve_with_info(operator, rhs, solver=solver)[0]

    def plain(rhs):
        return weights @ gaussx.linear_solve(operator, rhs, solver=solver)

    assert np.array_equal(jax.grad(with_info)(b), jax.grad(plain)(b))
