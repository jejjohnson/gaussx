"""Tests for KeyedSolver (gh-384)."""

from __future__ import annotations

import jax.random as jr
import pytest

from gaussx import CGSolver, DenseSolver, KeyedSolver, SLQLogdet
from gaussx._testing import random_pd_operator, tree_allclose


def _op():
    return random_pd_operator(jr.key(0), 10, jitter=10.0)


def test_logdet_uses_the_bound_key_and_an_explicit_one_wins():
    op = _op()
    slq = SLQLogdet(num_probes=4, lanczos_order=4)
    keyed = KeyedSolver(slq, jr.key(1))
    assert tree_allclose(keyed.logdet(op), slq.logdet(op, key=jr.key(1)))
    assert tree_allclose(keyed.logdet(op, key=jr.key(2)), slq.logdet(op, key=jr.key(2)))


def test_solve_delegates():
    op = _op()
    b = jr.normal(jr.key(3), (10,), dtype=op.as_matrix().dtype)
    keyed = KeyedSolver(DenseSolver(), jr.key(1))
    assert tree_allclose(keyed.solve(op, b), DenseSolver().solve(op, b))


def test_solve_needs_a_solve_strategy():
    keyed = KeyedSolver(SLQLogdet(), jr.key(1))
    op = _op()
    with pytest.raises(TypeError, match="logdet-only"):
        keyed.solve(op, op.as_matrix()[0])


def test_cg_seed_changes_the_probes():
    # gh-384: CGSolver had no seed, so two CGSolvers could never decorrelate.
    op = _op()
    s0 = CGSolver(num_probes=4, lanczos_order=4)
    s1 = CGSolver(num_probes=4, lanczos_order=4, seed=1)
    assert tree_allclose(s0.logdet(op), s0.logdet(op))
    assert not tree_allclose(s0.logdet(op), s1.logdet(op))
    # seed=0 is the old seedless behaviour, bit for bit.
    assert s0.logdet(op) == SLQLogdet(num_probes=4, lanczos_order=4).logdet(op)
