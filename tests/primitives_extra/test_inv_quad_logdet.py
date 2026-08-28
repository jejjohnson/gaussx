"""Tests for the joint inverse-quadratic / log-determinant primitive."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._operators import LowRankUpdate
from gaussx._testing import random_pd_operator


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
        cg_max_iter=4 * n,
        cg_tolerance=1e-12,
        lanczos_iter=n,
        num_probes=64,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Inverse-quadratic form
# ---------------------------------------------------------------------------


def test_inv_quad_matches_dense_solve() -> None:
    operator = random_pd_operator(jr.key(0), 40)
    rhs = jr.normal(jr.key(1), (40, 1))

    inv_quad, _ = gaussx.inv_quad_logdet(operator, rhs, strategy=_exact_strategy(40))

    assert jnp.allclose(inv_quad, jnp.sum(_dense_inv_quad(operator, rhs)), rtol=1e-6)


def test_multi_column_inv_quad_is_the_trace() -> None:
    operator = random_pd_operator(jr.key(2), 40)
    rhs = jr.normal(jr.key(3), (40, 4))

    inv_quad, _ = gaussx.inv_quad_logdet(operator, rhs, strategy=_exact_strategy(40))

    assert jnp.allclose(inv_quad, jnp.sum(_dense_inv_quad(operator, rhs)), rtol=1e-6)


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


def test_logdet_is_exact_for_a_scaled_identity() -> None:
    # Every probe gives z^T log(cI) z = log(c) ||z||^2, so the Hutchinson
    # estimator has zero variance here and the comparison is exact.
    scale = 2.5
    operator = lx.DiagonalLinearOperator(jnp.full((12,), scale))
    rhs = jnp.ones((12, 1))

    _, logdet = gaussx.inv_quad_logdet(operator, rhs, strategy=_exact_strategy(12))

    assert jnp.allclose(logdet, 12 * jnp.log(scale), rtol=1e-8)


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
        cg_max_iter=2 * n, cg_tolerance=1e-12, lanczos_iter=n, num_probes=1, seed=11
    )

    probe = 2.0 * jr.bernoulli(jr.PRNGKey(11), 0.5, (n, 1)) - 1.0
    values, vectors = jnp.linalg.eigh(operator.as_matrix())
    log_operator = (vectors * jnp.log(values)[None, :]) @ vectors.T
    expected = probe[:, 0] @ log_operator @ probe[:, 0]

    _, logdet = gaussx.inv_quad_logdet(operator, rhs, strategy=strategy)

    assert jnp.allclose(logdet, expected, rtol=1e-4)


def test_logdet_tracks_the_dense_value() -> None:
    operator = random_pd_operator(jr.key(8), 40)
    rhs = jr.normal(jr.key(9), (40, 1))

    _, logdet = gaussx.inv_quad_logdet(
        operator,
        rhs,
        strategy=gaussx.BBMMSolver(
            cg_max_iter=80, lanczos_iter=40, num_probes=256, seed=2
        ),
    )

    # 256 sign probes; the seed is pinned, so the tolerance is a fixed
    # multiple of the estimator's own spread rather than a lucky draw.
    expected = jnp.linalg.slogdet(operator.as_matrix())[1]
    assert jnp.abs(logdet - expected) < 0.1 * jnp.abs(expected)


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
            cg_max_iter=120, lanczos_iter=30, num_probes=10, seed=seed
        )
        _, raw = gaussx.inv_quad_logdet(operator, rhs, strategy=strategy)
        _, reduced = gaussx.inv_quad_logdet(
            operator, rhs, strategy=strategy, preconditioner=preconditioner
        )
        plain.append(raw)
        preconditioned.append(reduced)

    plain = jnp.stack(plain)
    preconditioned = jnp.stack(preconditioned)

    assert jnp.std(preconditioned) < 0.5 * jnp.std(plain)
    assert jnp.abs(jnp.mean(preconditioned) - expected) < jnp.abs(
        jnp.mean(plain) - expected
    )


def test_preconditioned_logdet_stays_accurate() -> None:
    operator, preconditioner = _kernel_system(60, rank=20)
    rhs = jr.normal(jr.key(22), (60, 1))
    expected = jnp.linalg.slogdet(operator.as_matrix())[1]

    inv_quad, logdet = gaussx.inv_quad_logdet(
        operator,
        rhs,
        strategy=gaussx.BBMMSolver(
            cg_max_iter=120, lanczos_iter=30, num_probes=16, seed=4
        ),
        preconditioner=preconditioner,
    )

    assert jnp.allclose(inv_quad, jnp.sum(_dense_inv_quad(operator, rhs)), rtol=1e-5)
    assert jnp.abs(logdet - expected) < 0.02 * jnp.abs(expected)


# ---------------------------------------------------------------------------
# Alternative strategies
# ---------------------------------------------------------------------------


def test_dense_strategy_is_exact() -> None:
    operator = random_pd_operator(jr.key(10), 20)
    rhs = jr.normal(jr.key(11), (20, 2))

    inv_quad, logdet = gaussx.inv_quad_logdet(
        operator, rhs, strategy=gaussx.DenseSolver()
    )

    assert jnp.allclose(inv_quad, jnp.sum(_dense_inv_quad(operator, rhs)), rtol=1e-8)
    assert jnp.allclose(logdet, jnp.linalg.slogdet(operator.as_matrix())[1], rtol=1e-8)


def test_cg_strategy_matches_dense_inv_quad() -> None:
    operator = random_pd_operator(jr.key(12), 25)
    rhs = jr.normal(jr.key(13), (25, 2))

    inv_quad, logdet = gaussx.inv_quad_logdet(
        operator, rhs, strategy=gaussx.CGSolver(num_probes=64, lanczos_order=25)
    )

    expected_logdet = jnp.linalg.slogdet(operator.as_matrix())[1]
    assert jnp.allclose(inv_quad, jnp.sum(_dense_inv_quad(operator, rhs)), rtol=1e-5)
    assert jnp.abs(logdet - expected_logdet) < 0.15 * jnp.abs(expected_logdet)


def test_non_bbmm_strategy_applies_the_preconditioner_identity() -> None:
    operator, preconditioner = _kernel_system(40, rank=15)
    rhs = jr.normal(jr.key(14), (40, 1))

    _, logdet = gaussx.inv_quad_logdet(
        operator,
        rhs,
        strategy=gaussx.CGSolver(num_probes=16, lanczos_order=20),
        preconditioner=preconditioner,
    )

    expected = jnp.linalg.slogdet(operator.as_matrix())[1]
    assert jnp.abs(logdet - expected) < 0.02 * jnp.abs(expected)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


def test_jit_matches_eager() -> None:
    operator = random_pd_operator(jr.key(15), 20)
    rhs = jr.normal(jr.key(16), (20, 2))
    strategy = gaussx.BBMMSolver(cg_max_iter=40, lanczos_iter=20, num_probes=8, seed=5)

    def call(op, vectors):
        return gaussx.inv_quad_logdet(op, vectors, strategy=strategy)

    jitted = jax.jit(call)(operator, rhs)
    eager = call(operator, rhs)

    assert jnp.allclose(jitted[0], eager[0], rtol=1e-6)
    assert jnp.allclose(jitted[1], eager[1], rtol=1e-6)


def _symmetric(matrix):
    """Only the symmetric part is observable when the operator is symmetric."""
    return (matrix + matrix.T) / 2


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
    strategy = gaussx.BBMMSolver(cg_max_iter=80, lanczos_iter=30, num_probes=32, seed=8)

    def kernel(lengthscale):
        return jnp.exp(-0.5 * squared / lengthscale**2) + noise * jnp.eye(n)

    def negative_log_marginal(lengthscale):
        operator = lx.MatrixLinearOperator(
            kernel(lengthscale), lx.positive_semidefinite_tag
        )
        inv_quad, logdet = gaussx.inv_quad_logdet(
            operator, rhs, strategy=strategy, preconditioner=preconditioner
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
