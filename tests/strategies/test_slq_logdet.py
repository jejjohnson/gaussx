"""Tests for standalone logdet strategies."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

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


# ── SLQLogdet ──────────────────────────────────────────────────────


def test_slq_logdet_psd(getkey):
    """SLQLogdet should approximate logdet of a PSD matrix."""
    op, M = _make_pd_operator(getkey())
    _, ref = jnp.linalg.slogdet(M)
    slq = SLQLogdet(num_probes=40, lanczos_order=8)
    est = slq.logdet(op)
    assert tree_allclose(est, ref, rtol=0.1)


def test_slq_logdet_is_abstract_logdet():
    """SLQLogdet should be an AbstractLogdetStrategy."""
    assert isinstance(SLQLogdet(), AbstractLogdetStrategy)


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


def test_indefinite_slq_logdet_psd(getkey):
    """IndefiniteSLQLogdet on PSD should match |logdet|."""
    op, M = _make_pd_operator(getkey())
    _, ref = jnp.linalg.slogdet(M)
    est = IndefiniteSLQLogdet(num_probes=40, lanczos_order=8).logdet(op)
    assert tree_allclose(est, ref, rtol=0.1)


def test_indefinite_slq_logdet_indefinite(getkey):
    """IndefiniteSLQLogdet should handle indefinite symmetric matrices."""
    n = 8
    A = jr.normal(getkey(), (n, n))
    M = A + A.T  # symmetric but not necessarily PSD
    M = M + 0.1 * jnp.eye(n)  # slight shift to avoid zero eigenvalues
    op = lx.MatrixLinearOperator(M)
    ref = jnp.sum(jnp.log(jnp.abs(jnp.linalg.eigvalsh(M))))
    est = IndefiniteSLQLogdet(num_probes=40, lanczos_order=8).logdet(op)
    assert tree_allclose(est, ref, rtol=0.2)


def test_indefinite_slq_logdet_shift(getkey):
    """Shift parameter should be applied correctly."""
    op, M = _make_pd_operator(getkey())
    shift = 2.0
    M_shifted = M + shift * jnp.eye(M.shape[0])
    _, ref = jnp.linalg.slogdet(M_shifted)
    est = IndefiniteSLQLogdet(num_probes=40, lanczos_order=8, shift=shift).logdet(op)
    assert tree_allclose(est, ref, rtol=0.1)


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
