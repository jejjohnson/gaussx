"""Tests for standalone logdet strategies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from gaussx._einx import einsum
from gaussx._strategies import (
    AbstractLogdetStrategy,
    ComposedSolver,
    DenseLogdet,
    DenseSolver,
    IndefiniteSLQLogdet,
    SLQLogdet,
)
from gaussx._testing import tree_allclose


def _make_pd_operator(key, n=8):
    A = jr.normal(key, (n, n))
    M = A @ A.T + n * jnp.eye(n)
    return lx.MatrixLinearOperator(M, lx.positive_semidefinite_tag), M


# The SLQ estimators report their own standard error of the mean (SEM), so
# their tests bound |est - ref| by K_SEM standard errors instead of a fixed
# rtol (gh-303). Over 1000 random matrices |err| / SEM peaked at 3.1
# (indefinite) and 3.4 (PSD), so 5 SEM is a real bound; with the matrix and
# the probes pinned, each test is also deterministic.
K_SEM = 5.0


def _assert_within_sem(strategy, op, ref):
    est, sem = strategy.logdet_and_error(op)
    assert jnp.abs(est - ref) <= K_SEM * sem, (est, ref, sem)


# ── SLQLogdet ──────────────────────────────────────────────────────


@pytest.mark.slow
def test_slq_logdet_psd():
    """SLQLogdet should approximate logdet of a PSD matrix."""
    op, M = _make_pd_operator(jr.key(0))
    _, ref = jnp.linalg.slogdet(M)
    _assert_within_sem(SLQLogdet(num_probes=40, lanczos_order=8), op, ref)


def test_slq_logdet_is_abstract_logdet():
    """SLQLogdet should be an AbstractLogdetStrategy."""
    assert isinstance(SLQLogdet(), AbstractLogdetStrategy)


@pytest.mark.slow
def test_slq_logdet_jit(getkey):
    """SLQLogdet.logdet should be JIT-compatible."""
    op, _ = _make_pd_operator(getkey())
    slq = SLQLogdet(num_probes=10, lanczos_order=8)
    eager = slq.logdet(op)
    jitted = jax.jit(slq.logdet)(op)
    assert tree_allclose(eager, jitted)


def test_slq_logdet_with_key(getkey):
    """Passing an explicit key should work and produce a result."""
    op, _ = _make_pd_operator(getkey())
    slq = SLQLogdet(num_probes=10, lanczos_order=8)
    key = jr.PRNGKey(42)
    result = slq.logdet(op, key=key)
    assert jnp.isfinite(result)


# ── IndefiniteSLQLogdet ────────────────────────────────────────────


def test_indefinite_slq_logdet_psd():
    """IndefiniteSLQLogdet on PSD should match |logdet|."""
    op, M = _make_pd_operator(jr.key(0))
    _, ref = jnp.linalg.slogdet(M)
    _assert_within_sem(IndefiniteSLQLogdet(num_probes=40, lanczos_order=8), op, ref)


def test_indefinite_slq_logdet_indefinite():
    """IndefiniteSLQLogdet should handle indefinite symmetric matrices.

    The spectrum is set by construction, Q diag(lambda) Q^T with mixed signs
    and |lambda| >= 0.5, so log|det M| is not near zero. A random A + A^T + 2I
    reached min |lambda| = 7e-5 over 1000 seeds, which is what made the old
    relative tolerance flaky (gh-303).
    """
    n = 8
    Q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (n, n)))
    eigvals = jnp.array([-3.0, -1.5, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0])
    M = einsum(Q * eigvals, Q, "i k, j k -> i j")
    op = lx.MatrixLinearOperator(M, lx.symmetric_tag)
    ref = jnp.sum(jnp.log(jnp.abs(eigvals)))
    _assert_within_sem(IndefiniteSLQLogdet(num_probes=80, lanczos_order=8), op, ref)


def test_indefinite_slq_logdet_shift():
    """Shift parameter should be applied correctly."""
    op, M = _make_pd_operator(jr.key(0))
    shift = 2.0
    M_shifted = M + shift * jnp.eye(M.shape[0])
    _, ref = jnp.linalg.slogdet(M_shifted)
    strategy = IndefiniteSLQLogdet(num_probes=40, lanczos_order=8, shift=shift)
    _assert_within_sem(strategy, op, ref)


def test_indefinite_slq_logdet_is_abstract_logdet():
    """IndefiniteSLQLogdet should be an AbstractLogdetStrategy."""
    assert isinstance(IndefiniteSLQLogdet(), AbstractLogdetStrategy)


# ── DenseLogdet ────────────────────────────────────────────────────


def test_dense_logdet_matches_primitive(getkey):
    """DenseLogdet should match the gaussx.logdet primitive exactly."""
    op, _M = _make_pd_operator(getkey())
    from gaussx._primitives._logdet import logdet as _logdet

    ref = _logdet(op)
    est = DenseLogdet().logdet(op)
    assert tree_allclose(est, ref)


def test_dense_logdet_is_abstract_logdet():
    """DenseLogdet should be an AbstractLogdetStrategy."""
    assert isinstance(DenseLogdet(), AbstractLogdetStrategy)


def test_dense_logdet_jit(getkey):
    """DenseLogdet.logdet should be JIT-compatible."""
    op, _ = _make_pd_operator(getkey())
    dl = DenseLogdet()
    eager = dl.logdet(op)
    jitted = jax.jit(dl.logdet)(op)
    assert tree_allclose(eager, jitted)


# ── Composition tests ──────────────────────────────────────────────


@pytest.mark.slow
def test_composed_with_slq_logdet(getkey):
    """ComposedSolver should accept SLQLogdet as logdet_strategy."""
    op, _M = _make_pd_operator(getkey())
    v = jr.normal(getkey(), (8,))
    composed = ComposedSolver(
        solve_strategy=DenseSolver(),
        logdet_strategy=SLQLogdet(num_probes=30, lanczos_order=8),
    )
    sol = composed.solve(op, v)
    ld = composed.logdet(op)
    assert jnp.isfinite(sol).all()
    assert jnp.isfinite(ld)


def test_composed_with_dense_logdet(getkey):
    """ComposedSolver(Dense, DenseLogdet) should match DenseSolver."""
    op, _ = _make_pd_operator(getkey())
    v = jr.normal(getkey(), (8,))
    ref = DenseSolver()
    composed = ComposedSolver(
        solve_strategy=DenseSolver(),
        logdet_strategy=DenseLogdet(),
    )
    assert tree_allclose(composed.solve(op, v), ref.solve(op, v))
    assert tree_allclose(composed.logdet(op), ref.logdet(op))
