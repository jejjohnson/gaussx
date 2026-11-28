"""compose_solver: one strategy from a solver, a preconditioner and a logdet (gh-881).

Every accepted combination is checked against a dense solve and slogdet of
one PD system, at a tolerance stated per log-determinant; every rejected one
checks that its message names the classes involved. The file is in the
float32 lane (Makefile ``NO_X64_TESTS``).
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

import gaussx


_SYMMETRIC_PSD = (lx.symmetric_tag, lx.positive_semidefinite_tag)


def _x64() -> bool:
    return jnp.zeros(()).dtype == jnp.float64


def _system(size=24):
    """A = K + diag(d), a Matérn-3/2 kernel plus a split diagonal."""
    inputs = jnp.linspace(0.0, 1.0, size)
    gaps = jnp.abs(einx.subtract("row, col -> row col", inputs, inputs))
    scaled = jnp.sqrt(3.0) * gaps / 0.3
    kernel = (1.0 + scaled) * jnp.exp(-scaled) + 1e-6 * jnp.eye(size)
    shift = jnp.linspace(0.2, 1.0, size)
    system = lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(kernel) + lx.DiagonalLinearOperator(shift),
        _SYMMETRIC_PSD,
    )
    return system, kernel + jnp.diag(shift)


_PRECONDITIONERS = {
    "none": lambda: None,
    "jacobi": gaussx.JacobiPreconditioner,
    "partial_cholesky": lambda: gaussx.PartialCholeskyPreconditioner(
        rank=6, shift=None
    ),
    "nystrom": lambda: gaussx.NystromPreconditioner.lazy(6, key=jr.key(0)),
}


def _tight():
    tolerance = 1e-10 if _x64() else 1e-6
    return {"rtol": tolerance, "atol": tolerance}


# (solver, logdet, logdet tolerance): the log-determinant is exact (1e-8 in
# float64), or a probe estimate held to 1.5 nats (log det A ≈ −3.8 here; 40
# unpreconditioned probes measured errors up to 0.73 nats on this system)
_ACCEPTED = {
    "cg-none": (lambda: gaussx.CGSolver(**_tight()), lambda: None, "stochastic"),
    "cg-dense": (
        lambda: gaussx.CGSolver(**_tight()),
        gaussx.DenseLogdet,
        "exact",
    ),
    "cg-slq": (
        lambda: gaussx.CGSolver(**_tight()),
        lambda: gaussx.SLQLogdet(num_probes=40),
        "stochastic",
    ),
    "cg-bbmm": (
        lambda: gaussx.CGSolver(**_tight()),
        lambda: gaussx.BBMMSolver(num_probes=40),
        "stochastic",
    ),
    "bbmm-none": (
        lambda: gaussx.BBMMSolver(num_probes=40, **_tight()),
        lambda: None,
        "stochastic",
    ),
    "keyed_bbmm-none": (
        lambda: gaussx.KeyedSolver(
            gaussx.BBMMSolver(num_probes=40, **_tight()), jr.key(1)
        ),
        lambda: None,
        "stochastic",
    ),
    "auto-none": (
        lambda: gaussx.AutoSolver(size_threshold=10),
        lambda: None,
        "stochastic",
    ),
    "preconditioned_cg-dense": (
        lambda: gaussx.PreconditionedCGSolver(
            preconditioner_rank=4, shift=0.2, **_tight()
        ),
        gaussx.DenseLogdet,
        "exact",
    ),
}

# The fast lane keeps one case per route; the rest are slow (each traces a
# Krylov solve and a stochastic logdet).
_FAST = {
    ("cg-dense", "jacobi"),
    ("bbmm-none", "partial_cholesky"),
    ("cg-bbmm", "nystrom"),
    ("keyed_bbmm-none", "jacobi"),
}


def _accepted_cases():
    for combination in _ACCEPTED:
        for preconditioner in _PRECONDITIONERS:
            if combination.startswith("auto") and preconditioner != "jacobi":
                # AutoSolver hands its preconditioner to its CG unchanged
                continue
            marks = () if (combination, preconditioner) in _FAST else pytest.mark.slow
            yield pytest.param(
                combination,
                preconditioner,
                id=f"{combination}-{preconditioner}",
                marks=marks,
            )


@pytest.mark.parametrize(("combination", "preconditioner"), list(_accepted_cases()))
def test_accepted_combinations_match_the_dense_reference(combination, preconditioner):
    make_solver, make_logdet, kind = _ACCEPTED[combination]
    system, dense = _system()
    strategy = gaussx.compose_solver(
        make_solver(),
        preconditioner=_PRECONDITIONERS[preconditioner](),
        logdet=make_logdet(),
    )
    rhs = jnp.ones(system.in_size())
    # AutoSolver's CG runs at its default rtol (1e-5 in float64)
    solve_tolerance = 1e-4 if _x64() else 1e-2
    expected = jnp.linalg.solve(dense, rhs)
    solution = strategy.solve(system, rhs)
    assert jnp.allclose(solution, expected, rtol=solve_tolerance, atol=solve_tolerance)
    exact = jnp.linalg.slogdet(dense)[1]
    logdet = strategy.logdet(system, key=jr.key(2))
    if kind == "exact":
        tolerance = 1e-8 if _x64() else 1e-4
        assert jnp.allclose(logdet, exact, rtol=tolerance, atol=tolerance)
    else:
        assert jnp.abs(logdet - exact) < 1.5


def test_all_none_is_the_dense_solver():
    system, _ = _system()
    strategy = gaussx.compose_solver()
    assert strategy == gaussx.DenseSolver()
    rhs = jnp.ones(system.in_size())
    reference = gaussx.DenseSolver()
    assert jnp.array_equal(strategy.solve(system, rhs), reference.solve(system, rhs))
    assert jnp.array_equal(strategy.logdet(system), reference.logdet(system))


def test_a_solve_only_strategy_gets_the_auto_logdet():
    system, dense = _system()
    strategy = gaussx.compose_solver(gaussx.LineaxSolver(lx.CG(rtol=1e-6, atol=1e-6)))
    assert isinstance(strategy, gaussx.ComposedSolver)
    assert isinstance(strategy.logdet_strategy, gaussx.AutoSolver)
    tolerance = 1e-8 if _x64() else 1e-4
    exact = jnp.linalg.slogdet(dense)[1]
    assert jnp.allclose(strategy.logdet(system), exact, rtol=tolerance)


def test_one_preconditioner_reaches_both_parts():
    pre = gaussx.PartialCholeskyPreconditioner(rank=6, shift=None)
    shared = gaussx.compose_solver(gaussx.BBMMSolver(), preconditioner=pre)
    assert shared.preconditioner is pre
    composed = gaussx.compose_solver(
        gaussx.CGSolver(), preconditioner=pre, logdet=gaussx.BBMMSolver()
    )
    assert composed.solve_strategy.preconditioner is pre
    assert composed.logdet_strategy.preconditioner is pre
    keyed = gaussx.compose_solver(
        gaussx.CGSolver(),
        preconditioner=pre,
        logdet=gaussx.KeyedSolver(gaussx.BBMMSolver(), jr.key(3)),
    )
    assert keyed.logdet_strategy.strategy.preconditioner is pre
    # the key is kept
    assert jnp.array_equal(
        jr.key_data(keyed.logdet_strategy.key), jr.key_data(jr.key(3))
    )


def test_keyed_solver_is_looked_through_and_keeps_its_key():
    pre = gaussx.JacobiPreconditioner()
    key = jr.key(4)
    strategy = gaussx.compose_solver(
        gaussx.KeyedSolver(gaussx.CGSolver(), key), preconditioner=pre
    )
    assert isinstance(strategy, gaussx.KeyedSolver)
    assert strategy.strategy.preconditioner is pre
    assert jnp.array_equal(jr.key_data(strategy.key), jr.key_data(key))


def test_composed_solver_keeps_one_object_in_both_places():
    bbmm = gaussx.BBMMSolver()
    pre = gaussx.JacobiPreconditioner()
    composed = gaussx.ComposedSolver(bbmm, bbmm).with_preconditioner(pre)
    assert composed.solve_strategy is composed.logdet_strategy
    assert composed.solve_strategy.preconditioner is pre


def test_an_opaque_preconditioner_is_accepted_for_the_solve_alone():
    system, dense = _system()
    opaque = gaussx.OperatorPreconditioner(lambda v: v / 2.0)
    strategy = gaussx.compose_solver(
        gaussx.CGSolver(), preconditioner=opaque, logdet=gaussx.DenseLogdet()
    )
    rhs = jnp.ones(system.in_size())
    tolerance = 1e-3 if _x64() else 1e-2
    assert jnp.allclose(
        strategy.solve(system, rhs), jnp.linalg.solve(dense, rhs), atol=tolerance
    )


_REJECTED = {
    "dense": (
        lambda: {"solver": gaussx.DenseSolver()},
        ("DenseSolver", "JacobiPreconditioner"),
    ),
    "sparse_cholesky": (
        lambda: {"solver": gaussx.SparseCholeskySolver()},
        ("SparseCholeskySolver", "JacobiPreconditioner"),
    ),
    "minres": (
        lambda: {"solver": gaussx.MINRESSolver()},
        ("MINRESSolver", "JacobiPreconditioner"),
    ),
    "lsmr": (
        lambda: {"solver": gaussx.LSMRSolver()},
        ("LSMRSolver", "JacobiPreconditioner"),
    ),
    "lineax": (
        lambda: {"solver": gaussx.LineaxSolver(lx.CG(rtol=1e-6, atol=1e-6))},
        ("LineaxSolver", "JacobiPreconditioner"),
    ),
    "solver_carries_one": (
        lambda: {
            "solver": gaussx.CGSolver(
                preconditioner=gaussx.PartialCholeskyPreconditioner(rank=2)
            )
        },
        ("CGSolver", "PartialCholeskyPreconditioner", "JacobiPreconditioner"),
    ),
    "logdet_carries_one": (
        lambda: {
            "solver": gaussx.CGSolver(),
            "logdet": gaussx.BBMMSolver(
                preconditioner=gaussx.PartialCholeskyPreconditioner(rank=2)
            ),
        },
        ("BBMMSolver", "PartialCholeskyPreconditioner", "JacobiPreconditioner"),
    ),
    "nystrom_logdet": (
        lambda: {"solver": gaussx.CGSolver(), "logdet": gaussx.NystromLogdet(0.1)},
        ("NystromLogdet", "JacobiPreconditioner"),
    ),
}


@pytest.mark.parametrize("name", list(_REJECTED))
def test_rejected_combinations_name_the_classes(name):
    make, names = _REJECTED[name]
    with pytest.raises(ValueError) as error:
        gaussx.compose_solver(preconditioner=gaussx.JacobiPreconditioner(), **make())
    for class_name in names:
        assert class_name in str(error.value)


def test_a_full_solver_as_logdet_carrying_an_ignored_preconditioner_raises():
    ignored = gaussx.CGSolver(preconditioner=gaussx.JacobiPreconditioner())
    with pytest.raises(ValueError) as error:
        gaussx.compose_solver(gaussx.CGSolver(), logdet=ignored)
    assert "CGSolver" in str(error.value)
    assert "JacobiPreconditioner" in str(error.value)


def test_an_opaque_preconditioner_with_a_logdet_that_needs_p_raises():
    opaque = gaussx.OperatorPreconditioner(lambda v: v / 2.0)
    with pytest.raises(ValueError) as error:
        gaussx.compose_solver(gaussx.BBMMSolver(), preconditioner=opaque)
    assert "OperatorPreconditioner" in str(error.value)
    assert "BBMMSolver" in str(error.value)


def test_slq_logdet_refuses_a_preconditioner_until_825():
    with pytest.raises(ValueError, match="825"):
        gaussx.SLQLogdet().with_preconditioner(gaussx.JacobiPreconditioner())


def test_jit_does_not_retrace_when_the_preconditioner_arrays_change():
    system, dense = _system()
    rhs = jnp.ones(system.in_size())
    traces = []

    @eqx.filter_jit
    def solve(strategy, rhs):
        traces.append(None)
        return strategy.solve(system, rhs)

    results = []
    for scale in (1.0, 2.0):
        diagonal = scale * jnp.diag(dense)
        pre = gaussx.JacobiPreconditioner(diagonal=diagonal)
        strategy = gaussx.compose_solver(
            gaussx.CGSolver(**_tight()), preconditioner=pre, logdet=gaussx.DenseLogdet()
        )
        results.append(solve(strategy, rhs))
    assert len(traces) == 1
    tolerance = 1e-5 if _x64() else 1e-2
    for result in results:
        assert jnp.allclose(result, jnp.linalg.solve(dense, rhs), atol=tolerance)


def test_compose_solver_runs_under_jit():
    system, dense = _system()
    rhs = jnp.ones(system.in_size())

    @jax.jit
    def solve(diagonal):
        pre = gaussx.JacobiPreconditioner(diagonal=diagonal)
        strategy = gaussx.compose_solver(
            gaussx.CGSolver(**_tight()), preconditioner=pre, logdet=gaussx.DenseLogdet()
        )
        return strategy.solve(system, rhs)

    tolerance = 1e-5 if _x64() else 1e-2
    assert jnp.allclose(
        solve(jnp.diag(dense)), jnp.linalg.solve(dense, rhs), atol=tolerance
    )
