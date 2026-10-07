"""Canonical option names and their deprecated aliases (gh-405)."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx
from gaussx._testing import random_pd_operator, tree_allclose


_CANONICAL = {"rtol": 1e-6, "atol": 1e-7, "max_steps": 50}
_PROBES = {"num_probes": 4, "lanczos_order": 5, "seed": 3}


@pytest.mark.parametrize(
    "cls",
    [
        gaussx.CGSolver,
        gaussx.PreconditionedCGSolver,
        gaussx.MINRESSolver,
        gaussx.BBMMSolver,
    ],
)
def test_iterative_strategies_take_the_canonical_names(cls):
    strategy = cls(**_CANONICAL, **_PROBES)
    for name, value in {**_CANONICAL, **_PROBES}.items():
        assert getattr(strategy, name) == value


def test_lsmr_and_slq_take_the_canonical_names():
    # LSMR's relative tolerance stays btol (algorithm-specific).
    lsmr = gaussx.LSMRSolver(atol=1e-7, btol=1e-6, max_steps=50, **_PROBES)
    assert lsmr.max_steps == 50 and lsmr.lanczos_order == 5 and lsmr.seed == 3
    slq = gaussx.SLQLogdet(**_PROBES)
    assert slq.num_probes == 4 and slq.lanczos_order == 5 and slq.seed == 3


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            lambda: gaussx.BBMMSolver(cg_tolerance=1e-6),
            lambda: gaussx.BBMMSolver(rtol=1e-6, atol=1e-6),
        ),
        (
            lambda: gaussx.BBMMSolver(cg_max_iter=50),
            lambda: gaussx.BBMMSolver(max_steps=50),
        ),
        (
            lambda: gaussx.BBMMSolver(lanczos_iter=7),
            lambda: gaussx.BBMMSolver(lanczos_order=7),
        ),
        (
            lambda: gaussx.LSMRSolver(maxiter=50),
            lambda: gaussx.LSMRSolver(max_steps=50),
        ),
    ],
    ids=["cg_tolerance", "cg_max_iter", "lanczos_iter", "maxiter"],
)
def test_old_keyword_warns_once_and_builds_the_same_pytree(old, new):
    with pytest.warns(DeprecationWarning) as record:
        built = old()
    assert len(record) == 1
    assert "gh-405" in str(record[0].message)
    assert eqx.tree_equal(built, new())


def test_old_and_new_keyword_together_raise():
    with pytest.raises(TypeError, match="max_steps"):
        gaussx.BBMMSolver(max_steps=5, cg_max_iter=5)


def test_old_attribute_names_warn():
    bbmm = gaussx.BBMMSolver(rtol=1e-6, max_steps=50, lanczos_order=7)
    with pytest.warns(DeprecationWarning):
        assert bbmm.cg_tolerance == 1e-6
    with pytest.warns(DeprecationWarning):
        assert bbmm.cg_max_iter == 50
    with pytest.warns(DeprecationWarning):
        assert bbmm.lanczos_iter == 7
    with pytest.warns(DeprecationWarning):
        assert gaussx.LSMRSolver(max_steps=50).maxiter == 50


def test_canonical_names_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gaussx.BBMMSolver(rtol=1e-6, atol=1e-6, max_steps=50, lanczos_order=7)
        gaussx.LSMRSolver(max_steps=50)


def test_inv_quad_logdet_preconditioner_alias():
    # gh-405: inv_quad_logdet's P ≈ A is now logdet_preconditioner=; the old
    # keyword warns and behaves the same.
    op = random_pd_operator(jr.key(0), 8, jitter=8.0)
    rhs = jr.normal(jr.key(1), (8, 2), dtype=op.as_matrix().dtype)
    P = lx.TaggedLinearOperator(
        lx.DiagonalLinearOperator(jnp.diag(op.as_matrix())),
        lx.positive_semidefinite_tag,
    )
    strategy = gaussx.BBMMSolver(num_probes=4, lanczos_order=8)
    new = gaussx.inv_quad_logdet(op, rhs, strategy=strategy, logdet_preconditioner=P)
    with pytest.warns(DeprecationWarning, match="logdet_preconditioner"):
        old = gaussx.inv_quad_logdet(op, rhs, strategy=strategy, preconditioner=P)
    assert tree_allclose(old, new)
