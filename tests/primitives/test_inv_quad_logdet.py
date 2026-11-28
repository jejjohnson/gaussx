"""Tests for the joint inverse-quadratic / log-determinant primitive."""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
import gaussx._primitives._inv_quad_logdet
from gaussx import LowRankUpdate

# Private helpers whose own behaviour these tests check.
from gaussx._primitives._inv_quad_logdet import _lanczos_coefficients, _mbcg
from gaussx._testing import (
    default_tolerances,
    random_pd_matrix,
    random_pd_operator,
)


def _dense_inv_quad(operator: lx.AbstractLinearOperator, rhs):
    """Reference per-column ``r_c^T A^{-1} r_c``."""
    return jnp.sum(rhs * jnp.linalg.solve(operator.as_matrix(), rhs), axis=0)


def _exact_strategy(n: int, seed: int = 0) -> gaussx.BBMMSolver:
    """A BBMM strategy converged far enough to compare against dense linalg.

    Without reorthogonalisation, mBCG needs more than ``n`` steps to squeeze
    the residual down to rounding level; ``2n`` is comfortably enough for the
    condition numbers used here.
    """
    return gaussx.BBMMSolver(
        max_steps=4 * n,
        rtol=1e-12,
        atol=1e-12,
        lanczos_order=n,
        num_probes=64,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Inverse-quadratic form
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_inv_quad_matches_dense_solve() -> None:
    operator = random_pd_operator(jr.key(0), 40)
    rhs = jr.normal(jr.key(1), (40, 1))

    inv_quad, _ = gaussx.inv_quad_logdet(operator, rhs, strategy=_exact_strategy(40))

    # Dtype-aware: rtol=1e-6 held in float64 but not in the float32 lane.
    rtol, atol = default_tolerances(inv_quad)
    expected = jnp.sum(_dense_inv_quad(operator, rhs))
    assert jnp.allclose(inv_quad, expected, rtol=rtol, atol=atol)


def test_multi_column_inv_quad_is_the_trace() -> None:
    operator = random_pd_operator(jr.key(2), 40)
    rhs = jr.normal(jr.key(3), (40, 4))

    inv_quad, _ = gaussx.inv_quad_logdet(operator, rhs, strategy=_exact_strategy(40))

    # Dtype-aware: rtol=1e-6 held in float64 but not in the float32 lane.
    rtol, atol = default_tolerances(inv_quad)
    expected = jnp.sum(_dense_inv_quad(operator, rhs))
    assert jnp.allclose(inv_quad, expected, rtol=rtol, atol=atol)


@pytest.mark.x64_only(reason="dense-reference tolerance below float32 round-off")
def test_unreduced_inv_quad_is_per_column() -> None:
    operator = random_pd_operator(jr.key(4), 40)
    rhs = jr.normal(jr.key(5), (40, 4))

    inv_quad, _ = gaussx.inv_quad_logdet(
        operator, rhs, strategy=_exact_strategy(40), reduce_inv_quad=False
    )

    assert inv_quad.shape == (4,)
    assert jnp.allclose(inv_quad, _dense_inv_quad(operator, rhs), rtol=1e-6)


# ---------------------------------------------------------------------------
# Log-determinant
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_logdet_is_exact_for_a_scaled_identity() -> None:
    # Every probe gives z^T log(cI) z = log(c) ||z||^2, so the Hutchinson
    # estimator has zero variance here and the comparison is exact.
    scale = 2.5
    operator = lx.DiagonalLinearOperator(jnp.full((12,), scale))
    rhs = jnp.ones((12, 1))

    _, logdet = gaussx.inv_quad_logdet(operator, rhs, strategy=_exact_strategy(12))

    assert jnp.allclose(logdet, 12 * jnp.log(scale), rtol=1e-8)


@pytest.mark.slow
def test_lanczos_quadrature_reproduces_a_single_probe() -> None:
    r"""One probe, full-length Lanczos: the estimate is $z^T \log(A) z$.

    Isolates the mBCG-to-Lanczos correspondence from Hutchinson noise. The
    probe is rebuilt the same way `inv_quad_logdet` builds it, so this also
    pins down the sign-probe convention. mBCG does not reorthogonalise, so the
    recovered tridiagonal drifts from the exact Lanczos one by roughly the
    same relative amount as the CG residual.
    """
    n = 30
    operator = random_pd_operator(jr.key(6), n)
    rhs = jr.normal(jr.key(7), (n, 1))
    strategy = gaussx.BBMMSolver(
        max_steps=2 * n, rtol=1e-12, atol=1e-12, lanczos_order=n, num_probes=1, seed=11
    )

    probe = 2.0 * jr.bernoulli(jr.PRNGKey(11), 0.5, (n, 1)) - 1.0
    values, vectors = jnp.linalg.eigh(operator.as_matrix())
    log_operator = (vectors * jnp.log(values)[None, :]) @ vectors.T
    expected = probe[:, 0] @ log_operator @ probe[:, 0]

    _, logdet = gaussx.inv_quad_logdet(operator, rhs, strategy=strategy)

    assert jnp.allclose(logdet, expected, rtol=1e-4)


@pytest.mark.slow
def test_lanczos_order_may_exceed_the_operator_size() -> None:
    # Without reorthogonalisation, Lanczos past n adds ghost Ritz values whose
    # Gauss weights split between copies, so e1^T log(T) e1 stays accurate --
    # and the extra steps recover what lost orthogonality cost. Measured at
    # kappa = 1e8: 3e-2 relative error with the order clamped to n, 3e-11
    # with 10n.
    n = 12
    rotation, _ = jnp.linalg.qr(jr.normal(jr.key(21), (n, n)))
    values = jnp.geomspace(1e-8, 1.0, n)
    matrix = (rotation * values) @ rotation.T
    operator = lx.MatrixLinearOperator(
        0.5 * (matrix + matrix.T), lx.positive_semidefinite_tag
    )
    strategy = gaussx.BBMMSolver(
        max_steps=10 * n, rtol=1e-12, atol=1e-12, lanczos_order=10 * n, num_probes=1
    )

    probe = 2.0 * jr.bernoulli(jr.PRNGKey(0), 0.5, (n, 1)) - 1.0
    log_operator = (rotation * jnp.log(values)) @ rotation.T
    expected = probe[:, 0] @ log_operator @ probe[:, 0]

    _, logdet = gaussx.inv_quad_logdet(operator, jnp.ones((n, 1)), strategy=strategy)

    assert jnp.allclose(logdet, expected, rtol=1e-8)


@pytest.mark.slow
def test_logdet_tracks_the_dense_value() -> None:
    operator = random_pd_operator(jr.key(8), 40)
    rhs = jr.normal(jr.key(9), (40, 1))

    _, logdet = gaussx.inv_quad_logdet(
        operator,
        rhs,
        strategy=gaussx.BBMMSolver(
            max_steps=80, lanczos_order=40, num_probes=256, seed=2
        ),
    )

    # 256 sign probes; the seed is pinned, so the tolerance is a fixed
    # multiple of the estimator's own spread rather than a lucky draw.
    expected = jnp.linalg.slogdet(operator.as_matrix())[1]
    assert jnp.abs(logdet - expected) < 0.1 * jnp.abs(expected)


def test_a_frozen_probe_is_disconnected_from_its_lanczos_prefix() -> None:
    # Step 1 is the last active step, so its beta was computed before the
    # freeze; it must not couple the valid 2x2 prefix to the frozen steps.
    alphas = jnp.array([[0.5], [0.25], [0.9], [0.9]])
    betas = jnp.array([[0.3], [0.2], [0.7], [0.7]])
    active = jnp.array([[True], [True], [False], [False]])

    diagonal, off_diagonal = _lanczos_coefficients(alphas, betas, active, order=4)

    assert jnp.allclose(off_diagonal[0], jnp.array([jnp.sqrt(0.3) / 0.5, 0.0, 0.0]))
    assert jnp.allclose(diagonal[0, :2], jnp.array([2.0, 4.0 + 0.3 / 0.5]))
    # The trailing block keeps distinct entries so eigh's VJP stays finite.
    assert diagonal[0, 2] != diagonal[0, 3]


def test_a_probe_that_breaks_down_stays_frozen() -> None:
    # A breakdown-frozen column can report active again once its direction
    # resets; everything after the first freeze is not part of the run.
    alphas = jnp.array([[0.5], [0.9], [0.4]])
    betas = jnp.array([[0.3], [0.7], [0.1]])
    active = jnp.array([[True], [False], [True]])

    _, off_diagonal = _lanczos_coefficients(alphas, betas, active, order=3)

    assert jnp.allclose(off_diagonal[0], 0.0)


def test_mbcg_stops_once_every_column_has_frozen() -> None:
    # 2I converges in one step; a fixed-length loop would still run all
    # max_iter batched matvecs.
    calls = []

    def matvec(vector):
        jax.debug.callback(lambda _: calls.append(None), vector[0])
        return 2.0 * vector

    operator = lx.FunctionLinearOperator(
        matvec,
        jax.ShapeDtypeStruct((10,), jnp.float64),
        lx.positive_semidefinite_tag,
    )
    block = jr.normal(jr.key(23), (10, 3))

    solutions, _, _, _, active = _mbcg(
        operator, block, None, max_iter=1000, floors=jnp.full((3,), 1e-24)
    )

    assert jnp.allclose(solutions, block / 2.0)
    assert active[0].all() and not active[1:].any()
    # One step that converges and one that finds nothing left to do, per column.
    assert len(calls) == 2 * 3


# ---------------------------------------------------------------------------
# Preconditioned variance reduction
# ---------------------------------------------------------------------------


def _kernel_system(n: int, rank: int, noise: float = 0.1):
    """An RBF kernel system plus the rank-``rank`` operator that preconditions it."""
    inputs = jr.normal(jr.key(20), (n, 2))
    squared = jnp.sum((inputs[:, None, :] - inputs[None, :, :]) ** 2, axis=-1)
    kernel = jnp.exp(-0.5 * squared)
    operator = lx.MatrixLinearOperator(
        kernel + noise * jnp.eye(n), lx.positive_semidefinite_tag
    )

    values, vectors = jnp.linalg.eigh(kernel)
    top = jnp.argsort(values)[::-1][:rank]
    preconditioner = LowRankUpdate(
        lx.DiagonalLinearOperator(jnp.full((n,), noise)),
        vectors[:, top],
        jnp.maximum(values[top], 0.0),
        vectors[:, top],
    )
    return operator, preconditioner


@pytest.mark.slow
def test_preconditioning_reduces_logdet_variance() -> None:
    n = 400
    operator, preconditioner = _kernel_system(n, rank=30)
    rhs = jr.normal(jr.key(21), (n, 1))
    expected = jnp.linalg.slogdet(operator.as_matrix())[1]

    plain, preconditioned = [], []
    for seed in range(6):
        strategy = gaussx.BBMMSolver(
            max_steps=120, lanczos_order=30, num_probes=10, seed=seed
        )
        _, raw = gaussx.inv_quad_logdet(operator, rhs, strategy=strategy)
        _, reduced = gaussx.inv_quad_logdet(
            operator, rhs, strategy=strategy, logdet_preconditioner=preconditioner
        )
        plain.append(raw)
        preconditioned.append(reduced)

    plain = jnp.stack(plain)
    preconditioned = jnp.stack(preconditioned)

    assert jnp.std(preconditioned) < 0.5 * jnp.std(plain)
    assert jnp.abs(jnp.mean(preconditioned) - expected) < jnp.abs(
        jnp.mean(plain) - expected
    )


@pytest.mark.slow
def test_preconditioned_logdet_stays_accurate() -> None:
    operator, preconditioner = _kernel_system(60, rank=20)
    rhs = jr.normal(jr.key(22), (60, 1))
    expected = jnp.linalg.slogdet(operator.as_matrix())[1]

    inv_quad, logdet = gaussx.inv_quad_logdet(
        operator,
        rhs,
        strategy=gaussx.BBMMSolver(
            max_steps=120, lanczos_order=30, num_probes=16, seed=4
        ),
        logdet_preconditioner=preconditioner,
    )

    assert jnp.allclose(inv_quad, jnp.sum(_dense_inv_quad(operator, rhs)), rtol=1e-5)
    assert jnp.abs(logdet - expected) < 0.02 * jnp.abs(expected)


# ---------------------------------------------------------------------------
# Alternative strategies
# ---------------------------------------------------------------------------


@pytest.mark.x64_only(reason="exactness check at float64 tolerance")
def test_dense_strategy_is_exact() -> None:
    operator = random_pd_operator(jr.key(10), 20)
    rhs = jr.normal(jr.key(11), (20, 2))

    inv_quad, logdet = gaussx.inv_quad_logdet(
        operator, rhs, strategy=gaussx.DenseSolver()
    )

    assert jnp.allclose(inv_quad, jnp.sum(_dense_inv_quad(operator, rhs)), rtol=1e-8)
    assert jnp.allclose(logdet, jnp.linalg.slogdet(operator.as_matrix())[1], rtol=1e-8)


@pytest.mark.parametrize(
    "strategy",
    [
        gaussx.DenseSolver(),
        gaussx.AutoSolver(),
        gaussx.ComposedSolver(
            solve_strategy=gaussx.CGSolver(), logdet_strategy=gaussx.DenseSolver()
        ),
        gaussx.ComposedSolver(
            solve_strategy=gaussx.CGSolver(), logdet_strategy=gaussx.DenseLogdet()
        ),
    ],
    ids=["dense", "auto", "composed", "composed-dense-logdet"],
)
def test_exact_strategy_ignores_the_preconditioner(strategy) -> None:
    # Whitening an exact log-determinant would only add contour-quadrature
    # error, so the preconditioner must not change the result.
    operator, preconditioner = _kernel_system(30, rank=10)
    rhs = jr.normal(jr.key(17), (30, 1))

    _, logdet = gaussx.inv_quad_logdet(
        operator, rhs, strategy=strategy, logdet_preconditioner=preconditioner
    )

    expected = jnp.linalg.slogdet(operator.as_matrix())[1]
    assert jnp.allclose(logdet, expected, rtol=1e-10)


@pytest.mark.slow
def test_cg_strategy_matches_dense_inv_quad() -> None:
    operator = random_pd_operator(jr.key(12), 25)
    rhs = jr.normal(jr.key(13), (25, 2))

    inv_quad, logdet = gaussx.inv_quad_logdet(
        operator, rhs, strategy=gaussx.CGSolver(num_probes=64, lanczos_order=25)
    )

    expected_logdet = jnp.linalg.slogdet(operator.as_matrix())[1]
    assert jnp.allclose(inv_quad, jnp.sum(_dense_inv_quad(operator, rhs)), rtol=1e-5)
    assert jnp.abs(logdet - expected_logdet) < 0.15 * jnp.abs(expected_logdet)


@pytest.mark.slow
def test_non_bbmm_strategy_applies_the_preconditioner_identity() -> None:
    operator, preconditioner = _kernel_system(40, rank=15)
    rhs = jr.normal(jr.key(14), (40, 1))

    _, logdet = gaussx.inv_quad_logdet(
        operator,
        rhs,
        strategy=gaussx.CGSolver(num_probes=16, lanczos_order=20),
        logdet_preconditioner=preconditioner,
    )

    expected = jnp.linalg.slogdet(operator.as_matrix())[1]
    assert jnp.abs(logdet - expected) < 0.02 * jnp.abs(expected)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_function_operator_is_supported_and_differentiable() -> None:
    # jax.custom_vjp rejects the matvec closure a FunctionLinearOperator
    # carries; the shared-work VJP has to accept arbitrary operator pytrees.
    n = 12
    rhs = jr.normal(jr.key(24), (n, 2))

    def loss(scale):
        operator = lx.FunctionLinearOperator(
            lambda vector: scale * vector,
            jax.ShapeDtypeStruct((n,), jnp.float64),
            lx.positive_semidefinite_tag,
        )
        inv_quad, logdet = gaussx.inv_quad_logdet(
            operator, rhs, strategy=gaussx.BBMMSolver(num_probes=4)
        )
        return inv_quad + logdet

    scale = 3.0
    value, gradient = jax.value_and_grad(loss)(scale)

    # For sI both terms are exact: sign probes make Hutchinson exact on tr(I/s).
    squared = jnp.sum(rhs**2)
    assert jnp.allclose(value, squared / scale + n * jnp.log(scale))
    assert jnp.allclose(gradient, -squared / scale**2 + n / scale)


@pytest.mark.slow
def test_jit_matches_eager() -> None:
    operator = random_pd_operator(jr.key(15), 20)
    rhs = jr.normal(jr.key(16), (20, 2))
    strategy = gaussx.BBMMSolver(max_steps=40, lanczos_order=20, num_probes=8, seed=5)

    def call(op, vectors):
        return gaussx.inv_quad_logdet(op, vectors, strategy=strategy)

    jitted = jax.jit(call)(operator, rhs)
    eager = call(operator, rhs)

    assert jnp.allclose(jitted[0], eager[0], rtol=1e-6)
    assert jnp.allclose(jitted[1], eager[1], rtol=1e-6)


def _symmetric(matrix):
    """Only the symmetric part is observable when the operator is symmetric."""
    return (matrix + matrix.T) / 2


@pytest.mark.slow
def test_grad_of_inv_quad_matches_the_dense_derivative() -> None:
    n = 25
    matrix = random_pd_operator(jr.key(17), n).as_matrix()
    rhs = jr.normal(jr.key(18), (n, 2))
    strategy = _exact_strategy(n, seed=6)

    def objective(mat):
        operator = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
        return gaussx.inv_quad_logdet(operator, rhs, strategy=strategy)[0]

    def reference(mat):
        return jnp.sum(rhs * jnp.linalg.solve(mat, rhs))

    gradient = jax.grad(objective)(matrix)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(
        _symmetric(gradient), _symmetric(jax.grad(reference)(matrix)), atol=1e-6
    )


@pytest.mark.x64_only(reason="exactness check at float64 tolerance")
def test_grad_with_respect_to_the_right_hand_side_is_exact() -> None:
    n = 25
    operator = random_pd_operator(jr.key(19), n)
    rhs = jr.normal(jr.key(20), (n, 2))
    strategy = _exact_strategy(n, seed=7)

    gradient = jax.grad(
        lambda vectors: gaussx.inv_quad_logdet(operator, vectors, strategy=strategy)[0]
    )(rhs)

    assert jnp.allclose(
        gradient, 2 * jnp.linalg.solve(operator.as_matrix(), rhs), atol=1e-7
    )


@pytest.mark.slow
def test_grad_of_logdet_is_an_unbiased_inverse_estimate() -> None:
    r"""$\partial \log|A| / \partial A = A^{-1}$, estimated stochastically.

    The backward pass reuses the probe solves as a Hutchinson estimate of
    $A^{-1}$, so a single call is noisy by construction; averaging over
    independent probe seeds has to converge on the dense inverse.
    """
    n = 15
    matrix = random_pd_operator(jr.key(21), n).as_matrix()
    rhs = jr.normal(jr.key(22), (n, 1))

    def objective(mat, seed):
        operator = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
        strategy = _exact_strategy(n, seed=seed)
        return gaussx.inv_quad_logdet(operator, rhs, strategy=strategy)[1]

    gradients = jnp.stack([jax.grad(objective)(matrix, seed) for seed in range(24)])
    averaged = _symmetric(jnp.mean(gradients, axis=0))
    expected = jnp.linalg.inv(matrix)

    assert jnp.all(jnp.isfinite(gradients))
    relative = jnp.linalg.norm(averaged - expected) / jnp.linalg.norm(expected)
    assert relative < 0.1


@pytest.mark.slow
def test_grad_flows_through_a_kernel_parameterisation_with_a_preconditioner() -> None:
    """The headline use case: one marginal-likelihood gradient step."""
    n = 40
    inputs = jr.normal(jr.key(23), (n, 2))
    squared = jnp.sum((inputs[:, None, :] - inputs[None, :, :]) ** 2, axis=-1)
    noise = 0.1
    rhs = jr.normal(jr.key(24), (n, 1))
    _, preconditioner = _kernel_system(n, rank=15, noise=noise)
    strategy = gaussx.BBMMSolver(max_steps=80, lanczos_order=30, num_probes=32, seed=8)

    def kernel(lengthscale):
        return jnp.exp(-0.5 * squared / lengthscale**2) + noise * jnp.eye(n)

    def negative_log_marginal(lengthscale):
        operator = lx.MatrixLinearOperator(
            kernel(lengthscale), lx.positive_semidefinite_tag
        )
        inv_quad, logdet = gaussx.inv_quad_logdet(
            operator, rhs, strategy=strategy, logdet_preconditioner=preconditioner
        )
        return 0.5 * inv_quad + 0.5 * logdet

    def reference(lengthscale):
        matrix = kernel(lengthscale)
        return (
            0.5 * jnp.sum(rhs * jnp.linalg.solve(matrix, rhs))
            + 0.5 * jnp.linalg.slogdet(matrix)[1]
        )

    gradient = jax.grad(negative_log_marginal)(1.0)
    expected = jax.grad(reference)(1.0)

    assert jnp.isfinite(gradient)
    # The log-determinant half of the gradient is a Hutchinson estimate, so the
    # tolerance tracks the estimator, not machine precision.
    assert jnp.abs(gradient - expected) < 0.1 * jnp.abs(expected)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_rectangular_operator() -> None:
    operator = lx.MatrixLinearOperator(jnp.ones((3, 4)))
    with pytest.raises(ValueError, match="square operator"):
        gaussx.inv_quad_logdet(operator, jnp.ones((4, 1)))


def test_rejects_non_matrix_right_hand_side() -> None:
    operator = random_pd_operator(jr.key(23), 5)
    with pytest.raises(ValueError, match=r"shape \(N, C\)"):
        gaussx.inv_quad_logdet(operator, jnp.ones(5))


def test_rejects_mismatched_right_hand_side() -> None:
    operator = random_pd_operator(jr.key(24), 5)
    with pytest.raises(ValueError, match="rows but operator has size"):
        gaussx.inv_quad_logdet(operator, jnp.ones((4, 1)))


# -- gh-340: strategy=None takes the exact structural path when the operator
# has one. Keys pinned: the property is about the operator type, not a draw.

_PSD = lx.positive_semidefinite_tag


def _structured(scale):
    """``name -> operator`` with every leaf multiplied by ``scale``."""
    A = lx.MatrixLinearOperator(scale * random_pd_matrix(jr.key(0), 3, jitter=3), _PSD)
    B = lx.MatrixLinearOperator(scale * random_pd_matrix(jr.key(1), 4, jitter=4), _PSD)
    U = jr.normal(jr.key(2), (12, 2))
    return {
        "kronecker": gaussx.Kronecker(A, B),
        "block_diag": gaussx.BlockDiag(A, B),
        "low_rank": LowRankUpdate(
            lx.MatrixLinearOperator(
                scale * random_pd_matrix(jr.key(3), 12, jitter=12), _PSD
            ),
            U,
            jnp.ones(2),
            U,
        ),
        "block_tridiag": gaussx.BlockTriDiag(
            scale * jnp.tile(4.0 * jnp.eye(2), (6, 1, 1)),
            scale * jnp.tile(-1.0 * jnp.eye(2), (5, 1, 1)),
            tags=_PSD,
        ),
        "kronecker_sum": gaussx.KroneckerSum(A, B, tags=_PSD),
    }


_WRAP = {
    "bare": lambda op: op,
    "tagged": lambda op: lx.TaggedLinearOperator(op, _PSD),
    "scaled": lambda op: 2.0 * op,
}


@pytest.mark.parametrize("wrap", list(_WRAP))
@pytest.mark.parametrize("name", list(_structured(1.0)))
def test_default_is_exact_for_structured_operators(name, wrap) -> None:
    op = _WRAP[wrap](_structured(1.0)[name])
    rhs = jr.normal(jr.key(4), (op.in_size(), 2))
    M = op.as_matrix()
    inv_quad, ld = gaussx.inv_quad_logdet(op, rhs)
    assert jnp.allclose(inv_quad, jnp.sum(rhs * jnp.linalg.solve(M, rhs)), atol=1e-10)
    assert jnp.allclose(ld, jnp.linalg.slogdet(M)[1], atol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_structured(1.0)))
def test_default_gradients_match_dense(name) -> None:
    rhs = jr.normal(jr.key(5), (_structured(1.0)[name].in_size(), 2))

    def structured(scale):
        quad, ld = gaussx.inv_quad_logdet(_structured(scale)[name], rhs)
        return quad + ld

    def dense(scale):
        M = _structured(scale)[name].as_matrix()
        return jnp.sum(rhs * jnp.linalg.solve(M, rhs)) + jnp.linalg.slogdet(M)[1]

    assert jnp.allclose(jax.grad(structured)(1.3), jax.grad(dense)(1.3), rtol=1e-8)


@pytest.mark.slow
def test_unstructured_operators_still_take_bbmm() -> None:
    op = lx.MatrixLinearOperator(random_pd_matrix(jr.key(6), 20, jitter=20), _PSD)
    rhs = jr.normal(jr.key(7), (20, 2))
    default = gaussx.inv_quad_logdet(op, rhs)
    bbmm = gaussx.inv_quad_logdet(op, rhs, strategy=gaussx.BBMMSolver())
    assert jnp.array_equal(default[1], bbmm[1])
    assert jnp.allclose(default[0], bbmm[0])


def test_shared_pass_honours_throw():
    # The mBCG pass bypasses BBMMSolver.solve, so it checks the right-hand
    # sides itself: out of budget raises unless throw=False.
    op = lx.MatrixLinearOperator(
        random_pd_matrix(jr.key(0), 20), lx.positive_semidefinite_tag
    )
    rhs = jr.normal(jr.key(1), (20, 1), dtype=op.as_matrix().dtype)
    with pytest.raises(eqx.EquinoxRuntimeError):
        gaussx.inv_quad_logdet(op, rhs, strategy=gaussx.BBMMSolver(max_steps=1))
    inv_quad, _ = gaussx.inv_quad_logdet(
        op, rhs, strategy=gaussx.BBMMSolver(max_steps=1, throw=False)
    )
    assert jnp.isfinite(inv_quad)


# ---------------------------------------------------------------------------
# BBMMSolver(preconditioner=...) and key= (gh-879)
# ---------------------------------------------------------------------------

_SYMMETRIC_PSD = (lx.symmetric_tag, _PSD)


def _matern32_site_system(size: int, key, lengthscale=0.3):
    """The smooth site system of gh-879: Matérn-3/2 in the unit square,
    variance 1, d on [4, 10.9]; returns (A, dense A, K, d)."""
    input_key, shift_key = jr.split(key)
    inputs = jr.uniform(input_key, (size, 2))
    gaps = einx.subtract("row dim, col dim -> row col dim", inputs, inputs)
    distance = jnp.sqrt(einx.sum("row col dim -> row col", gaps**2) + 1e-30)
    scaled = jnp.sqrt(3.0) * distance / lengthscale
    kernel = (1.0 + scaled) * jnp.exp(-scaled) + 1e-6 * jnp.eye(size)
    shift = 4.0 + 6.9 * jr.uniform(shift_key, (size,), dtype=kernel.dtype)
    system = lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(kernel) + lx.DiagonalLinearOperator(shift),
        _SYMMETRIC_PSD,
    )
    return system, kernel + jnp.diag(shift), kernel, shift


class _CountingCore:
    """Counts the calls of `_shared_work_core`, the shared mBCG pass."""

    def __init__(self, monkeypatch):
        module = gaussx._primitives._inv_quad_logdet
        original = module._shared_work_core
        self.calls = 0

        def counting(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "_shared_work_core", counting)


def test_bbmm_key_none_is_the_seed_key() -> None:
    system, _, _, _ = _matern32_site_system(30, jr.key(0))
    rhs = jr.normal(jr.key(1), (30, 1))
    strategy = gaussx.BBMMSolver(num_probes=4, seed=3)
    default = gaussx.inv_quad_logdet(system, rhs, strategy=strategy)
    keyed = gaussx.inv_quad_logdet(system, rhs, strategy=strategy, key=jr.PRNGKey(3))
    assert jnp.array_equal(default[0], keyed[0])
    assert jnp.array_equal(default[1], keyed[1])


def test_bbmm_preconditioned_solve_takes_fewer_steps() -> None:
    system, dense, _, _ = _matern32_site_system(200, jr.key(0))
    rhs = jnp.ones(200)
    pre = gaussx.PartialCholeskyPreconditioner(rank=30, shift=None)

    def steps(preconditioner):
        solver = gaussx.BBMMSolver(rtol=1e-6, atol=1e-6, preconditioner=preconditioner)
        solution, info = gaussx.linear_solve_with_info(system, rhs, solver=solver)
        return solution, int(info.iterations)

    plain_solution, plain = steps(None)
    solution, preconditioned = steps(pre)
    assert preconditioned < plain
    rtol = 1e-4 if solution.dtype == jnp.float64 else 1e-2
    assert jnp.allclose(solution, jnp.linalg.solve(dense, rhs), rtol=rtol, atol=rtol)
    assert jnp.allclose(solution, plain_solution, rtol=rtol, atol=rtol)


@pytest.mark.slow
@pytest.mark.x64_only(reason="the issue's measurement is float64")
def test_bbmm_preconditioned_logdet_rms_is_a_quarter_of_the_plain_one() -> None:
    system, dense, _, _ = _matern32_site_system(400, jr.key(1))
    rhs = jr.normal(jr.key(2), (400, 1))
    exact = jnp.linalg.slogdet(dense)[1]
    keys = jr.split(jr.key(3), 20)
    pre = gaussx.PartialCholeskyPreconditioner(rank=60, shift=None)

    def rms(preconditioner):
        strategy = gaussx.BBMMSolver(num_probes=50, preconditioner=preconditioner)
        _, logdets = jax.vmap(
            lambda key: gaussx.inv_quad_logdet(system, rhs, strategy=strategy, key=key)
        )(keys)
        return jnp.sqrt(jnp.mean((logdets - exact) ** 2))

    # the issue's bar: below 25 % (measured there with a FITC P: 0.109
    # against 0.995)
    assert rms(pre) < 0.25 * rms(None)


def test_keyed_bbmm_takes_the_shared_pass(monkeypatch) -> None:
    system, _, _, _ = _matern32_site_system(20, jr.key(0))
    rhs = jr.normal(jr.key(1), (20, 1))
    pre = gaussx.JacobiPreconditioner()
    bbmm = gaussx.BBMMSolver(num_probes=8, preconditioner=pre)
    counter = _CountingCore(monkeypatch)
    first = gaussx.inv_quad_logdet(
        system, rhs, strategy=gaussx.KeyedSolver(bbmm, jr.key(5))
    )
    assert counter.calls == 1
    second = gaussx.inv_quad_logdet(
        system, rhs, strategy=gaussx.KeyedSolver(bbmm, jr.key(6))
    )
    again = gaussx.inv_quad_logdet(
        system, rhs, strategy=gaussx.KeyedSolver(bbmm, jr.key(5))
    )
    assert counter.calls == 3
    # two keys give two values; one key repeats
    assert not jnp.array_equal(first[1], second[1])
    assert jnp.array_equal(first[1], again[1])
    # an explicit key wins over the bound one
    explicit = gaussx.inv_quad_logdet(
        system, rhs, strategy=gaussx.KeyedSolver(bbmm, jr.key(6)), key=jr.key(5)
    )
    assert jnp.array_equal(first[1], explicit[1])


@pytest.mark.slow
def test_bbmm_logdet_with_a_preconditioner_runs_the_shared_pass(monkeypatch) -> None:
    system, dense, _, _ = _matern32_site_system(40, jr.key(0))
    counter = _CountingCore(monkeypatch)
    pre = gaussx.PartialCholeskyPreconditioner(rank=10, shift=None)
    estimate = gaussx.BBMMSolver(num_probes=16, preconditioner=pre).logdet(
        system, key=jr.key(1)
    )
    assert counter.calls == 1
    exact = jnp.linalg.slogdet(dense)[1]
    assert jnp.abs(estimate - exact) < 0.05 * jnp.abs(exact)
    # without one it stays plain SLQ
    gaussx.BBMMSolver(num_probes=4).logdet(system)
    assert counter.calls == 1


def test_bbmm_preconditioner_and_logdet_preconditioner_both_raise() -> None:
    system, _, _, _ = _matern32_site_system(10, jr.key(0))
    rhs = jnp.ones((10, 1))
    strategy = gaussx.BBMMSolver(preconditioner=gaussx.JacobiPreconditioner())
    P = lx.DiagonalLinearOperator(jnp.ones(10))
    with pytest.raises(ValueError, match=r"logdet_preconditioner.*preconditioner="):
        gaussx.inv_quad_logdet(system, rhs, strategy=strategy, logdet_preconditioner=P)


def test_bbmm_opaque_preconditioner_cannot_precondition_the_logdet() -> None:
    system, dense, _, _ = _matern32_site_system(10, jr.key(0))
    opaque = gaussx.OperatorPreconditioner(lambda v: v / 5.0)
    strategy = gaussx.BBMMSolver(preconditioner=opaque)
    # the solve is fine with M⁻¹ alone
    solution = strategy.solve(system, jnp.ones(10))
    assert jnp.allclose(dense @ solution, jnp.ones(10), atol=1e-2)
    with pytest.raises(ValueError, match="supports_slq"):
        strategy.logdet(system)


class _NoDensify(lx.FunctionLinearOperator):
    """A matrix-free kernel that fails the test if anything densifies it."""

    def as_matrix(self):
        raise AssertionError("as_matrix called on the matrix-free system")


def test_bbmm_preconditioned_pass_never_densifies() -> None:
    _, dense, kernel, shift = _matern32_site_system(40, jr.key(0))
    structure = jax.ShapeDtypeStruct((40,), kernel.dtype)
    matrix_free = _NoDensify(lambda v: kernel @ v, structure, _PSD)
    system = lx.TaggedLinearOperator(
        matrix_free + lx.DiagonalLinearOperator(shift), _SYMMETRIC_PSD
    )
    rhs = jr.normal(jr.key(1), (40, 1))
    pre = gaussx.PartialCholeskyPreconditioner(rank=10, shift=None)
    strategy = gaussx.BBMMSolver(num_probes=8, preconditioner=pre)
    inv_quad, logdet = gaussx.inv_quad_logdet(system, rhs, strategy=strategy)
    expected = rhs[:, 0] @ jnp.linalg.solve(dense, rhs[:, 0])
    assert jnp.allclose(inv_quad, expected, rtol=1e-2)
    assert jnp.isfinite(logdet)


@pytest.mark.slow
def test_bbmm_preconditioned_jit_vmap_grad() -> None:
    size = 60
    inputs_key, shift_key, rhs_key = jr.split(jr.key(0), 3)
    inputs = jnp.sort(jr.uniform(inputs_key, (size,)))
    shift = 0.5 + jr.uniform(shift_key, (size,))
    rhs = jr.normal(rhs_key, (size, 1))
    pre = gaussx.PartialCholeskyPreconditioner(rank=15, shift=None)
    strategy = gaussx.BBMMSolver(num_probes=32, preconditioner=pre)

    def kernel(lengthscale):
        gaps = jnp.abs(einx.subtract("row, col -> row col", inputs, inputs))
        scaled = jnp.sqrt(3.0) * gaps / lengthscale
        return (1.0 + scaled) * jnp.exp(-scaled) + 1e-6 * jnp.eye(size)

    def objective(lengthscale, key):
        system = lx.TaggedLinearOperator(
            lx.MatrixLinearOperator(kernel(lengthscale))
            + lx.DiagonalLinearOperator(shift),
            _SYMMETRIC_PSD,
        )
        inv_quad, logdet = gaussx.inv_quad_logdet(
            system, rhs, strategy=strategy, key=key
        )
        return inv_quad + logdet

    def dense_objective(lengthscale):
        matrix = kernel(lengthscale) + jnp.diag(shift)
        quadratic = rhs[:, 0] @ jnp.linalg.solve(matrix, rhs[:, 0])
        return quadratic + jnp.linalg.slogdet(matrix)[1]

    lengthscale = jnp.asarray(0.3)
    keys = jr.split(jr.key(1), 20)
    jitted = jax.jit(objective)(lengthscale, keys[0])
    assert jnp.allclose(jitted, objective(lengthscale, keys[0]))
    gradients = jax.vmap(jax.jit(jax.grad(objective)), in_axes=(None, 0))(
        lengthscale, keys
    )
    expected = jax.grad(dense_objective)(lengthscale)
    # unbiased, so the mean over 20 keys is within 3 standard errors
    standard_error = jnp.std(gradients) / jnp.sqrt(20.0)
    assert jnp.abs(jnp.mean(gradients) - expected) < 3.0 * standard_error + 1e-6
