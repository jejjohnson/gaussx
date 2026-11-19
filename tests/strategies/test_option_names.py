"""Canonical option names of the iterative strategies (gh-405)."""

from __future__ import annotations

import warnings

import pytest

import gaussx


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


def test_canonical_names_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gaussx.BBMMSolver(rtol=1e-6, atol=1e-6, max_steps=50, lanczos_order=7)
        gaussx.LSMRSolver(max_steps=50)
