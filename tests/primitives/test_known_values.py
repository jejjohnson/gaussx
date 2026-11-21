"""Known-value elimination (gh-857).

Against the dense block solve x_E = A_EE⁻¹ (b_E − A_EK g_K) on random
symmetric definite matrices of both signs, with dense, CG and multigrid
strategies; the prescribed entries are exact, an empty known set is a plain
solve, and the gradient, ``jit`` and ``vmap`` contracts hold.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import gaussx


N = 40


def _spd(seed: int, n: int = N) -> np.ndarray:
    a = np.random.default_rng(seed).standard_normal((n, n))
    return a @ a.T / n + np.eye(n)


def _problem(seed: int, sign: float = 1.0):
    rng = np.random.default_rng(seed + 100)
    matrix = sign * _spd(seed)
    known = rng.random(N) < 0.3
    known[0] = True
    values = rng.standard_normal(N)
    b = rng.standard_normal(N)
    return matrix, known, values, b


def _dense_reference(matrix, known, values, b):
    e, k = ~known, known
    x = np.array(values, dtype=float)
    rhs = b[e] - matrix[np.ix_(e, k)] @ values[k]
    x[e] = np.linalg.solve(matrix[np.ix_(e, e)], rhs)
    return x


def _operator(matrix, sign):
    tag = lx.positive_semidefinite_tag if sign > 0 else lx.negative_semidefinite_tag
    return lx.MatrixLinearOperator(jnp.asarray(matrix), (lx.symmetric_tag, tag))


def _interpolation(n_fine: int, n_coarse: int) -> np.ndarray:
    """Linear interpolation from n_coarse points onto n_fine points on [0, 1]."""
    fine, coarse = np.linspace(0, 1, n_fine), np.linspace(0, 1, n_coarse)
    columns = [np.interp(fine, coarse, e) for e in np.eye(n_coarse)]
    return np.column_stack(columns)


def _multigrid(matrix, known, sign):
    # levels of the reduced operator P A P ± (I − P), the one actually solved
    unknown = (~known).astype(float)
    reduced = np.outer(unknown, unknown) * matrix + sign * np.diag(known)
    operator = _operator(reduced, sign)
    transfer = lx.MatrixLinearOperator(jnp.asarray(_interpolation(N, N // 2)))
    levels = gaussx.galerkin_levels(operator, [transfer], restriction_scale=0.5)
    return gaussx.MultigridSolver(levels, rtol=1e-13, atol=1e-13)


_STRATEGIES = {
    "dense": lambda *_: gaussx.DenseSolver(),
    "cg": lambda *_: gaussx.CGSolver(rtol=1e-13, atol=1e-13),
    "multigrid": _multigrid,
}


@pytest.mark.x64_only(reason="dense block solve to 1e-12")
@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive", "negative"])
@pytest.mark.parametrize("strategy", list(_STRATEGIES))
def test_equals_the_dense_block_solve(strategy, sign):
    matrix, known, values, b = _problem(0, sign)
    solver = _STRATEGIES[strategy](matrix, known, sign)
    x = gaussx.solve_with_known_values(
        _operator(matrix, sign),
        jnp.asarray(b),
        jnp.asarray(known),
        jnp.asarray(values),
        solver=solver,
    )
    reference = _dense_reference(matrix, known, values, b)
    assert np.linalg.norm(x - reference) <= 1e-12 * np.linalg.norm(reference)
    # the prescribed entries are copied, not solved for
    assert np.array_equal(np.asarray(x)[known], values[known])


def test_empty_known_set_is_a_plain_solve():
    matrix, _, values, b = _problem(1)
    operator = _operator(matrix, 1.0)
    rhs = jnp.asarray(b, dtype=jnp.result_type(float))
    known = jnp.zeros(N, bool)
    x = gaussx.solve_with_known_values(operator, rhs, known, jnp.asarray(values))
    tol = 1e-4 if rhs.dtype == jnp.float32 else 1e-10
    assert jnp.allclose(x, gaussx.linear_solve(operator, rhs), rtol=tol, atol=tol)


def test_lift_is_zero_off_its_set():
    matrix, known, values, b = _problem(2)
    rhs, lift = gaussx.lift_known_values(
        _operator(matrix, 1.0), jnp.asarray(b), jnp.asarray(known), jnp.asarray(values)
    )
    assert not np.any(np.asarray(rhs)[known])
    assert not np.any(np.asarray(lift)[~known])


@pytest.mark.x64_only(reason="finite differences at 1e-6")
def test_gradients_match_finite_differences():
    matrix, known, values, b = _problem(3)
    operator = _operator(matrix, 1.0)
    mask = jnp.asarray(known)
    solver = gaussx.CGSolver(rtol=1e-13, atol=1e-13)
    weights = jnp.asarray(np.random.default_rng(4).standard_normal(N))

    def loss(g, rhs):
        x = gaussx.solve_with_known_values(operator, rhs, mask, g, solver=solver)
        return weights @ x

    g0, b0 = jnp.asarray(values), jnp.asarray(b)
    grad_g, grad_b = jax.grad(loss, argnums=(0, 1))(g0, b0)
    direction = jnp.asarray(np.random.default_rng(5).standard_normal(N))
    eps = 1e-6
    for grad, shift in ((grad_g, (direction, 0)), (grad_b, (0, direction))):
        plus = loss(g0 + eps * shift[0], b0 + eps * shift[1])
        minus = loss(g0 - eps * shift[0], b0 - eps * shift[1])
        assert float(grad @ direction) == pytest.approx(
            float((plus - minus) / (2 * eps)), rel=1e-6
        )


def test_jit_and_vmap_match_a_loop():
    matrix, known, values, _ = _problem(6)
    operator = _operator(matrix, 1.0)
    dtype = operator.in_structure().dtype
    mask, g = jnp.asarray(known), jnp.asarray(values, dtype=dtype)
    batch = jnp.asarray(np.random.default_rng(7).standard_normal((3, N)), dtype=dtype)
    solver = gaussx.CGSolver()

    def solve(rhs):
        return gaussx.solve_with_known_values(operator, rhs, mask, g, solver=solver)

    looped = jnp.stack([solve(rhs) for rhs in batch])
    tol = 1e-4 if dtype == jnp.float32 else 1e-8
    assert jnp.allclose(jax.vmap(solve)(batch), looped, rtol=tol, atol=tol)
    assert jnp.allclose(jax.jit(solve)(batch[0]), looped[0], rtol=tol, atol=tol)


def test_mask_may_be_traced():
    # one compiled program for every mask
    matrix, known, values, b = _problem(8)
    operator = _operator(matrix, 1.0)
    dtype = operator.in_structure().dtype

    @jax.jit
    def solve(mask):
        return gaussx.solve_with_known_values(
            operator,
            jnp.asarray(b, dtype=dtype),
            mask,
            jnp.asarray(values, dtype=dtype),
            solver=gaussx.CGSolver(),
        )

    x = solve(jnp.asarray(known))
    assert np.array_equal(np.asarray(x)[known], np.asarray(values, dtype=dtype)[known])
