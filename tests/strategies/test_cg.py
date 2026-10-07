"""Tests for CGSolver strategy."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._operators import Kronecker
from gaussx._strategies import CGSolver, SLQLogdet
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


@pytest.mark.slow
def test_solve_psd(getkey):
    cg = CGSolver(rtol=1e-8, atol=1e-8)
    mat = random_pd_matrix(getkey(), 5)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (5,))
    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(cg.solve(op, v), expected, rtol=1e-4)


def test_solve_diagonal(getkey):
    cg = CGSolver()
    d = jnp.abs(jr.normal(getkey(), (4,))) + 0.1
    op = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(d), lx.positive_semidefinite_tag
    )
    v = jr.normal(getkey(), (4,))
    expected = v / d
    assert tree_allclose(cg.solve(op, v), expected, rtol=1e-4)


@pytest.mark.slow
def test_logdet_psd():
    """Stochastic logdet is within K_SEM standard errors of the exact one."""
    cg = CGSolver(num_probes=50, lanczos_order=20)
    op = random_pd_operator(jr.key(0), 20)
    key = jr.PRNGKey(42)
    est, sem = SLQLogdet(num_probes=50, lanczos_order=20).logdet_and_error(op, key=key)
    assert tree_allclose(cg.logdet(op, key=key), est)
    assert jnp.abs(est - dense_logdet(op)) <= K_SEM * sem


@pytest.mark.slow
def test_logdet_diagonal():
    """On a diagonal operator SLQ with sign probes is exact, not stochastic.

    Each probe gives zᵀ log(D) z = Σ log dᵢ when zᵢ² = 1, and full-order
    Lanczos is exact quadrature, so only round-off is left.
    """
    cg = CGSolver(num_probes=50, lanczos_order=10)
    d = jnp.abs(jr.normal(jr.key(0), (10,))) + 0.5
    op = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(d), lx.positive_semidefinite_tag
    )
    estimated = cg.logdet(op, key=jr.PRNGKey(123))
    assert tree_allclose(estimated, jnp.sum(jnp.log(d)), rtol=1e-8)


def test_filter_jit_solve(getkey):
    cg = CGSolver(rtol=1e-6, atol=1e-6)
    mat = random_pd_matrix(getkey(), 4)
    op = lx.MatrixLinearOperator(mat, lx.positive_semidefinite_tag)
    v = jr.normal(getkey(), (4,))

    @eqx.filter_jit
    def f(op, v):
        return cg.solve(op, v)

    expected = jnp.linalg.solve(mat, v)
    assert tree_allclose(f(op, v), expected, rtol=1e-4)


def test_solve_structured_operator():
    """CG drives a gaussx operator's matrix-free ``mv``.

    ``lineax.CG`` calls ``lineax.linearise``, which lineax registers only
    for its own operator classes — so this raised ``NotImplementedError``
    on every gaussx operator until the registration in
    ``gaussx._operators``. It is the route for structured covariances with
    no closed-form solve, e.g. a `SumOfKroneckers` of three or more terms.
    """
    factor = lx.MatrixLinearOperator(
        random_pd_matrix(jr.key(0), 3),
        (lx.symmetric_tag, lx.positive_semidefinite_tag),
    )
    op = Kronecker(factor, factor, tags=lx.positive_semidefinite_tag)
    v = jr.normal(jr.key(1), (9,))
    expected = jnp.linalg.solve(op.as_matrix(), v)
    assert tree_allclose(
        CGSolver(rtol=1e-10, atol=1e-10).solve(op, v), expected, rtol=1e-5
    )
