"""Multigrid over a caller-built hierarchy (gh-856).

Poisson problems with linear-interpolation prolongations and Galerkin coarse
operators: the V-cycle converges at a grid-independent rate, is a symmetric
preconditioner, cuts the CG steps by an order of magnitude, works for a
negative-definite operator and for a singular (Neumann) one, and keeps the
dtype, ``vmap`` and gradient contracts.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx


def _laplacian_1d(n: int, *, neumann: bool = False) -> np.ndarray:
    a = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    if neumann:
        a[0, 0] = a[-1, -1] = 1.0
    return a


def _interpolation(n_coarse: int) -> np.ndarray:
    """Linear interpolation from n_c points to the nested 2 n_c + 1 points."""
    p = np.zeros((2 * n_coarse + 1, n_coarse))
    for j in range(n_coarse):
        p[2 * j + 1, j] = 1.0
        p[2 * j, j] += 0.5
        p[2 * j + 2, j] += 0.5
    return p


def _poisson_2d(m: int) -> np.ndarray:
    a, eye = _laplacian_1d(m), np.eye(m)
    return np.kron(a, eye) + np.kron(eye, a)


def _levels(matrix, prolongations, scale, *, sign=1.0, dtype=None, **smoother):
    dtype = jnp.result_type(float) if dtype is None else dtype
    tag = lx.positive_semidefinite_tag if sign > 0 else lx.negative_semidefinite_tag
    operator = lx.MatrixLinearOperator(jnp.asarray(sign * matrix, dtype=dtype), tag)
    transfers = [
        lx.MatrixLinearOperator(jnp.asarray(p, dtype=dtype)) for p in prolongations
    ]

    def make(a):
        return gaussx.WeightedJacobi.from_operator(a, **smoother)

    return operator, gaussx.galerkin_levels(
        operator, transfers, restriction_scale=scale, smoother=make
    )


def _problem_1d():
    return _laplacian_1d(127), [_interpolation(n) for n in (63, 31, 15)], 0.5


def _problem_2d():
    return (
        _poisson_2d(31),
        [np.kron(p, p) for p in map(_interpolation, (15, 7, 3))],
        0.25,
    )


def _residual_factors(operator, pre, b, n_cycles):
    u, norms = jnp.zeros_like(b), [jnp.linalg.norm(b)]
    for _ in range(n_cycles):
        u = pre.v_cycle(b, u)
        norms.append(jnp.linalg.norm(b - operator.mv(u)))
    return [float(norms[i + 1] / norms[i]) for i in range(n_cycles)]


@pytest.mark.x64_only(reason="convergence-rate check down to 1e-8")
@pytest.mark.parametrize(
    ("problem", "smoother", "sweeps"),
    [(_problem_1d, {}, 2), (_problem_2d, {"omega": 0.8}, 3)],
    ids=["1d", "2d"],
)
def test_v_cycle_converges_at_a_grid_independent_rate(problem, smoother, sweeps):
    matrix, prolongations, scale = problem()
    operator, levels = _levels(matrix, prolongations, scale, **smoother)
    pre = gaussx.MultigridPreconditioner(levels, n_pre=sweeps, n_post=sweeps)
    b = jnp.asarray(np.random.default_rng(0).standard_normal(matrix.shape[0]))
    # measured: 1-D 0.054 … 0.089, 2-D (ω = 0.8, 3 sweeps) 0.022 … 0.067
    factors = _residual_factors(operator, pre, b, 6)
    assert max(factors[1:]) < 0.15, factors
    u = gaussx.MultigridSolver(levels, n_pre=sweeps, n_post=sweeps).cycles(b, 12)
    assert jnp.linalg.norm(b - operator.mv(u)) <= 1e-8 * jnp.linalg.norm(b)


@pytest.mark.x64_only(reason="symmetry to 1e-12")
def test_v_cycle_is_symmetric():
    matrix, prolongations, scale = _problem_2d()
    _, levels = _levels(matrix, prolongations, scale)
    pre = gaussx.MultigridPreconditioner(levels)
    x, y = (jnp.asarray(v) for v in np.random.default_rng(1).standard_normal((2, 961)))
    left, right = x @ pre.v_cycle(y), pre.v_cycle(x) @ y
    assert abs(float(left - right)) <= 1e-12 * abs(float(left))


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive", "negative"])
def test_preconditioned_cg_converges_in_few_steps(sign):
    matrix, prolongations, scale = _problem_2d()
    operator, levels = _levels(matrix, prolongations, scale, sign=sign)
    b = jnp.asarray(
        np.random.default_rng(2).standard_normal(961),
        dtype=operator.in_structure().dtype,
    )
    pre = gaussx.MultigridPreconditioner(levels).as_operator(operator)
    rtol = 1e-6 if b.dtype == jnp.float32 else 1e-10
    plain = lx.linear_solve(operator, b, lx.CG(rtol=rtol, atol=rtol, max_steps=2000))
    with_mg = lx.linear_solve(
        operator,
        b,
        lx.CG(rtol=rtol, atol=rtol, max_steps=2000),
        options={"preconditioner": pre},
    )
    # measured in float64: 123 steps without, 11 with one V-cycle
    assert int(with_mg.stats["num_steps"]) <= 15
    assert int(with_mg.stats["num_steps"]) * 5 <= int(plain.stats["num_steps"])
    x = gaussx.MultigridSolver(levels, rtol=rtol, atol=rtol).solve(operator, b)
    assert x.dtype == b.dtype
    assert jnp.allclose(x, with_mg.value, rtol=1e-3, atol=1e-5)


@pytest.mark.x64_only(reason="reference solve to 1e-10")
def test_negative_definite_levels_solve_the_negated_system():
    # linear_solve hands CG −A for a negative-definite A; the levels still
    # describe A, and the V-cycle must stay a positive-definite M⁻¹ ≈ (−A)⁻¹
    matrix, prolongations, scale = _problem_1d()
    operator, levels = _levels(matrix, prolongations, scale, sign=-1.0)
    b = jnp.asarray(np.random.default_rng(9).standard_normal(127))
    solver = gaussx.MultigridSolver(levels, rtol=1e-12, atol=1e-12)
    x = gaussx.linear_solve(operator, b, solver=solver)
    assert jnp.allclose(x, jnp.linalg.solve(-matrix, b), rtol=1e-10, atol=1e-10)


@pytest.mark.x64_only(reason="reference solve to 1e-10")
def test_solve_equals_the_dense_solve():
    matrix, prolongations, scale = _problem_1d()
    operator, levels = _levels(matrix, prolongations, scale)
    b = jnp.asarray(np.random.default_rng(3).standard_normal(127))
    x = gaussx.MultigridSolver(levels, rtol=1e-12, atol=1e-12).solve(operator, b)
    assert jnp.allclose(x, jnp.linalg.solve(matrix, b), rtol=1e-10, atol=1e-10)


@pytest.mark.x64_only(reason="null-space projection to 1e-10")
def test_neumann_problem_with_its_null_space():
    n = 31
    matrix = _laplacian_1d(n, neumann=True)
    operator = lx.MatrixLinearOperator(
        jnp.asarray(matrix), lx.positive_semidefinite_tag
    )
    # the null space of the Neumann Laplacian: constants
    ones = jnp.ones((n, 1))
    prolongation = lx.MatrixLinearOperator(jnp.asarray(_interpolation(15)))
    levels = gaussx.galerkin_levels(operator, [prolongation], restriction_scale=0.5)
    # the coarse Galerkin operator inherits the singularity: pin its solve
    b = jnp.asarray(np.random.default_rng(4).standard_normal(n))
    b = b - jnp.mean(b)
    solver = gaussx.MultigridSolver(
        levels,
        null_space=ones,
        rtol=1e-10,
        atol=1e-10,
        bottom_solve=lambda r: jnp.linalg.lstsq(levels[-1].operator.as_matrix(), r)[0],
    )
    x = solver.solve(operator, b)
    assert abs(float(jnp.mean(x))) <= 1e-10
    assert jnp.linalg.norm(operator.mv(x) - b) <= 1e-8 * jnp.linalg.norm(b)


def test_vmap_and_jit_over_rhs_match_a_loop():
    matrix, prolongations, scale = _problem_1d()
    operator, levels = _levels(matrix, prolongations, scale)
    solver = gaussx.MultigridSolver(levels)
    batch = jnp.asarray(
        np.random.default_rng(5).standard_normal((3, 127)),
        dtype=operator.in_structure().dtype,
    )
    looped = jnp.stack([solver.solve(operator, b) for b in batch])
    batched = jax.vmap(solver.solve, in_axes=(None, 0))(operator, batch)
    jitted = jax.jit(lambda b: solver.solve(operator, b))(batch[0])
    tol = 1e-4 if batch.dtype == jnp.float32 else 1e-8
    assert jnp.allclose(batched, looped, rtol=tol, atol=tol)
    assert jnp.allclose(jitted, looped[0], rtol=tol, atol=tol)


@pytest.mark.x64_only(reason="finite differences at 1e-6")
def test_gradient_with_respect_to_the_rhs():
    matrix, prolongations, scale = _problem_1d()
    operator, levels = _levels(matrix, prolongations, scale)
    solver = gaussx.MultigridSolver(levels, rtol=1e-12, atol=1e-12)
    b = jnp.asarray(np.random.default_rng(6).standard_normal(127))
    weights = jnp.asarray(np.random.default_rng(7).standard_normal(127))
    grad = jax.grad(lambda v: weights @ solver.solve(operator, v))(b)
    # x = A⁻¹ b is linear, so the gradient of wᵀ A⁻¹ b is A⁻ᵀ w
    assert jnp.allclose(
        grad, jnp.linalg.solve(matrix.T, weights), rtol=1e-8, atol=1e-10
    )


def test_float32_stays_float32():
    matrix, prolongations, scale = _problem_1d()
    operator, levels = _levels(matrix, prolongations, scale, dtype=jnp.float32)
    b = jnp.asarray(np.random.default_rng(8).standard_normal(127), dtype=jnp.float32)
    x = gaussx.MultigridSolver(levels).solve(operator, b)
    assert x.dtype == jnp.float32
    assert jnp.linalg.norm(operator.mv(x) - b) <= 1e-3 * jnp.linalg.norm(b)


def test_weighted_jacobi_estimate_is_stable_on_the_1d_laplacian():
    n = 63
    operator = lx.MatrixLinearOperator(jnp.asarray(_laplacian_1d(n)))
    omega = float(gaussx.WeightedJacobi.from_operator(operator).omega)
    rho = 1.0 + np.cos(np.pi / (n + 1))  # λ_max(D⁻¹ A), exact
    # Jacobi converges for ω ρ < 2; the target is ω = 4 / (3 · 1.1 ρ). Power
    # iteration underestimates ρ by 2 % here (the top eigenvalues cluster).
    assert omega * rho < 2.0
    assert omega == pytest.approx(4.0 / (3.0 * 1.1 * rho), rel=0.05)
