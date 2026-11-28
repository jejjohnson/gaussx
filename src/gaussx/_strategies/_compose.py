"""One solver strategy from three choices: a solve, a preconditioner, a logdet.

A model that exposes a solver, a preconditioner and a log-determinant as three
fields builds its strategy with `compose_solver`, so it holds no linear algebra
of its own. Every check here is on types and presence, never on values.
"""

from __future__ import annotations

from gaussx._preconditioners import AbstractPreconditioner
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import (
    AbstractLogdetStrategy,
    AbstractSolverStrategy,
    AbstractSolveStrategy,
    raise_two_preconditioners,
)
from gaussx._strategies._composed import ComposedSolver, logdet_takes_preconditioner
from gaussx._strategies._dense import DenseSolver
from gaussx._strategies._keyed import KeyedSolver
from gaussx._strategies._nystrom_logdet import NystromLogdet


def compose_solver(
    solver: AbstractSolveStrategy | None = None,
    *,
    preconditioner: AbstractPreconditioner | None = None,
    logdet: AbstractLogdetStrategy | None = None,
) -> AbstractSolverStrategy:
    """One strategy from a solver, a preconditioner and a log-determinant.

        b ↦ A⁻¹ b,   A ↦ log det A

    A preconditioner changes neither map, only how they are computed. The
    three choices resolve as follows:

    1. ``solver=None`` is `gaussx.DenseSolver`.
    2. ``preconditioner`` is attached to the solver with
       ``solver.with_preconditioner``.
    3. ``logdet=None`` keeps the resolved solver's own log-determinant, so a
       `gaussx.BBMMSolver` keeps its shared pass. A solve-only strategy
       (`gaussx.LineaxSolver`, `gaussx.MultigridSolver`) then gets the
       log-determinant of `gaussx.AutoSolver`: exact for a structured or
       small operator, stochastic Lanczos for a large PSD one.
    4. ``logdet`` given: ``preconditioner`` is attached to it too when its
       log-determinant uses one (a `gaussx.BBMMSolver`), and the result is
       ``ComposedSolver(solver, logdet)``. Other log-determinants
       (`gaussx.DenseLogdet`, `gaussx.SLQLogdet` until gaussx#825) keep their
       own estimate, and the preconditioner serves the solve.
    5. A `gaussx.KeyedSolver` is looked through, and its key is kept.

    Rejected combinations raise ``ValueError`` naming the classes involved:
    a preconditioner with an exact factorisation (`gaussx.DenseSolver`,
    `gaussx.SparseCholeskySolver`) or a solver with no preconditioned solve
    (`gaussx.MINRESSolver`, `gaussx.LSMRSolver`, `gaussx.MultigridSolver`,
    `gaussx.LineaxSolver`); a preconditioner when the solver or ``logdet``
    already carries one (or ``logdet`` is a `gaussx.NystromLogdet`, which
    builds its own); a full solver passed as ``logdet`` that carries a
    preconditioner its log-determinant ignores; and a preconditioner that
    does not know P itself (``supports_slq()`` is ``False``, as for
    `gaussx.OperatorPreconditioner`) with a log-determinant that needs P.
    That last one is accepted for the solve alone, next to an exact
    ``logdet``.

    Every check is on types and presence, so the function runs under
    ``jit`` and its result has the same pytree structure for every value of
    the preconditioner's arrays.

    Args:
        solver: The solve strategy; ``None`` is `gaussx.DenseSolver`.
        preconditioner: A preconditioner for the solve and, where it is used,
            the log-determinant.
        logdet: A log-determinant strategy; ``None`` keeps the solver's own.

    Returns:
        One `AbstractSolverStrategy` with both maps.

    Raises:
        ValueError: For the rejected combinations above.

    Examples:
        All three ``None`` is the default exact strategy:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> gaussx.compose_solver()
        DenseSolver()

        CG with an exact log-determinant, one Jacobi preconditioner for the
        solve:

        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> strategy = gaussx.compose_solver(
        ...     gaussx.CGSolver(),
        ...     preconditioner=gaussx.JacobiPreconditioner(),
        ...     logdet=gaussx.DenseLogdet(),
        ... )
        >>> type(strategy).__name__
        'ComposedSolver'
        >>> round(float(strategy.logdet(A)), 4)
        2.3979

        BBMM, whose log-determinant is its own, with one preconditioner for
        both maps:

        >>> pre = gaussx.JacobiPreconditioner()
        >>> bbmm = gaussx.compose_solver(gaussx.BBMMSolver(), preconditioner=pre)
        >>> bbmm.preconditioner is pre
        True

        A preconditioner has nothing to do in an exact factorisation:

        >>> gaussx.compose_solver(
        ...     gaussx.DenseSolver(), preconditioner=pre
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Preconditioning is not implemented for DenseSolver: ...
    """
    if solver is None:
        solver = DenseSolver()
    if preconditioner is not None:
        solver = solver.with_preconditioner(preconditioner)
    if logdet is None:
        if isinstance(solver, AbstractSolverStrategy):
            composed = solver
            logdet_part: AbstractLogdetStrategy = solver
        else:
            # a solve-only strategy takes AutoSolver's structural logdet
            logdet_part = AutoSolver()
            composed = ComposedSolver(solver, logdet_part)
    else:
        logdet_part = _with_logdet_preconditioner(logdet, preconditioner)
        composed = ComposedSolver(solver, logdet_part)
    if (
        preconditioner is not None
        and not preconditioner.supports_slq()
        and logdet_takes_preconditioner(logdet_part)
    ):
        raise ValueError(
            f"{type(preconditioner).__name__} does not know P itself "
            "(supports_slq() is False), so the log-determinant of "
            f"{type(logdet_part).__name__} cannot use it. Pass "
            "logdet=gaussx.DenseLogdet() to use it for the solve alone."
        )
    return composed


def _carried_preconditioner(
    strategy: AbstractLogdetStrategy,
) -> AbstractPreconditioner | None:
    """The preconditioner a strategy already carries, looking through a key.

    Args:
        strategy: A log-determinant strategy.

    Returns:
        Its ``preconditioner`` field, or ``None``.
    """
    if isinstance(strategy, KeyedSolver):
        return _carried_preconditioner(strategy.strategy)
    carried = getattr(strategy, "preconditioner", None)
    return carried if isinstance(carried, AbstractPreconditioner) else None


def _with_logdet_preconditioner(
    logdet: AbstractLogdetStrategy,
    preconditioner: AbstractPreconditioner | None,
) -> AbstractLogdetStrategy:
    """``logdet``, with ``preconditioner`` attached where its logdet uses one.

    Args:
        logdet: The log-determinant strategy.
        preconditioner: The preconditioner of the composition, or ``None``.

    Returns:
        The log-determinant strategy to compose.

    Raises:
        ValueError: If ``logdet`` already carries a preconditioner and one is
            given, carries one its log-determinant ignores, or is a
            `gaussx.NystromLogdet` and one is given.
    """
    carried = _carried_preconditioner(logdet)
    if carried is not None and preconditioner is not None:
        raise_two_preconditioners(logdet, carried, preconditioner)
    if carried is not None and not logdet_takes_preconditioner(logdet):
        raise ValueError(
            f"{type(logdet).__name__} passed as logdet carries a "
            f"{type(carried).__name__}, but only its solve would use it and "
            "that solve is never called here, so the preconditioner would be "
            "ignored. Pass it to compose_solver(preconditioner=...) instead."
        )
    if preconditioner is None:
        return logdet
    if isinstance(logdet, NystromLogdet):
        raise ValueError(
            "NystromLogdet builds its own Nyström preconditioner; attaching "
            f"{type(preconditioner).__name__} too would give two answers to "
            "one question."
        )
    if logdet_takes_preconditioner(logdet):
        return logdet.with_preconditioner(preconditioner)
    return logdet
