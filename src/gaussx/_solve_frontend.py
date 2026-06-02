"""Front door for the unified solver layer.

This module provides two things:

* :func:`as_linear_operator` -- a thin adapter that wraps a raw ``matvec``
  callable (and an optional shape) into a :class:`lineax.AbstractLinearOperator`
  with the appropriate structural tags. This is the entry point used by
  matrix-free callers (e.g. finite-volume / spectral PDE solvers) that hand
  gaussx a bare ``v -> A @ v`` function rather than a built operator object.

* :func:`linear_solve` -- a unified solve entry point that normalises its
  operator argument (operator *or* ``(matvec, shape)``), handles the
  negative-definite sign convention used by elliptic operators, selects a
  sensible default :class:`AbstractSolveStrategy`, and optionally applies a
  preconditioner.

The design goal is that callers in other packages pre-transform their problem
into an ``(operator, rhs)`` pair, call :func:`linear_solve`, and post-transform
the result -- without gaussx ever needing to know about grids, boundary
conditions, or transforms.
"""

from __future__ import annotations

from collections.abc import Callable

import lineax as lx
from jaxtyping import Array, Float

from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._strategies._cg import CGSolver
from gaussx._strategies._minres import MINRESSolver
from gaussx._tags import (
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
)


MatvecLike = Callable[[Float[Array, " n"]], Float[Array, " n"]]
"""A raw matrix-vector product ``v -> A @ v``."""

OperatorLike = lx.AbstractLinearOperator | tuple[MatvecLike, tuple[int, int]]
"""Either a built operator or a ``(matvec, shape)`` pair."""

PreconditionerLike = lx.AbstractLinearOperator | MatvecLike | object
"""A preconditioner: a lineax operator, a callable applying ``M^{-1}``, or an
object exposing ``.as_operator()`` (the Phase 1 ``AbstractPreconditioner``)."""


def as_linear_operator(
    matvec: MatvecLike,
    *,
    shape: tuple[int, int] | None = None,
    in_structure: object | None = None,
    dtype: object = float,
    symmetric: bool = False,
    positive_semidefinite: bool = False,
    negative_definite: bool = False,
) -> lx.AbstractLinearOperator:
    """Wrap a raw ``matvec`` callable as a tagged lineax operator.

    This is the matrix-free front door: callers that only have a
    ``v -> A @ v`` function (rather than a structured operator object) use this
    to obtain an operator that gaussx's solvers and primitives understand.

    Args:
        matvec: The matrix-vector product ``v -> A @ v``.
        shape: Matrix shape ``(out, in)``. The operator's input structure is
            inferred as a length-``in`` vector of ``dtype``. Ignored when
            ``in_structure`` is given.
        in_structure: An explicit input structure -- either a
            ``jax.ShapeDtypeStruct``, an ``int`` (interpreted as a vector
            length), or a PyTree thereof. Takes precedence over ``shape``.
        dtype: Dtype used when building the input structure from ``shape``.
        symmetric: Tag the operator symmetric (``A == A^T``).
        positive_semidefinite: Tag the operator PSD (implies symmetric).
        negative_definite: Tag the operator negative semidefinite (implies
            symmetric). Use this for elliptic operators such as a discrete
            Laplacian; :func:`linear_solve` will route the solve through the
            equivalent positive-definite system.

    Returns:
        A :class:`lineax.FunctionLinearOperator` carrying the requested tags.

    Raises:
        ValueError: If neither ``shape`` nor ``in_structure`` is provided.
    """
    import jax

    if in_structure is not None:
        structure = _normalise_in_structure(in_structure, dtype)
    elif shape is not None:
        structure = jax.ShapeDtypeStruct((shape[1],), dtype)
    else:
        raise ValueError("Provide either `shape=(out, in)` or `in_structure`.")

    tags: list[object] = []
    if symmetric or positive_semidefinite or negative_definite:
        tags.append(lx.symmetric_tag)
    if positive_semidefinite:
        tags.append(lx.positive_semidefinite_tag)
    if negative_definite:
        tags.append(lx.negative_semidefinite_tag)

    return lx.FunctionLinearOperator(matvec, structure, tags=tuple(tags))


def linear_solve(
    operator: OperatorLike,
    vector: Float[Array, " n"],
    *,
    solver: AbstractSolveStrategy | None = None,
    preconditioner: PreconditionerLike | None = None,
) -> Float[Array, " n"]:
    """Solve ``A x = b`` through the unified front door.

    Accepts either a built operator or a ``(matvec, shape)`` pair, handles the
    negative-definite sign convention, picks a default solver when none is
    given, and optionally applies a preconditioner.

    Sign handling: an operator tagged *negative* semidefinite (and not PSD) is
    solved via the equivalent positive-definite system ``(-A) x = -b``, so that
    a CG-style solver can be used directly. This is the common case for elliptic
    PDE operators (e.g. a discrete Laplacian), which finite-volume / spectral
    callers hand over as negative-definite matvecs.

    Default solver selection (when ``solver is None``):

    * positive semidefinite operator -> :class:`CGSolver`
    * symmetric (possibly indefinite) operator -> :class:`MINRESSolver`
    * otherwise -> a :class:`ValueError` asking for an explicit solver

    Args:
        operator: The linear operator ``A``, or a ``(matvec, shape)`` pair.
        vector: Right-hand side ``b``, shape ``(n,)``.
        solver: Solve strategy to use. When ``None`` a default is selected from
            the operator's structural tags.
        preconditioner: Optional preconditioner. May be a lineax operator
            applying ``M^{-1}``, a callable ``v -> M^{-1} v``, or an object with
            an ``.as_operator()`` method (the Phase 1 preconditioner protocol).

    Returns:
        The solution ``x``, shape ``(n,)``.
    """
    op = _coerce_operator(operator, vector)

    # Negative-definite -> solve the equivalent positive-definite system.
    if is_negative_semidefinite(op) and not is_positive_semidefinite(op):
        op = _negate(op)
        vector = -vector

    if solver is None:
        solver = _default_solver(op)

    if preconditioner is not None:
        return _solve_preconditioned(op, vector, solver, preconditioner)

    return solver.solve(op, vector)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_in_structure(in_structure: object, dtype: object) -> object:
    """Normalise an ``in_structure`` argument into a ShapeDtypeStruct PyTree."""
    import jax

    if isinstance(in_structure, int):
        return jax.ShapeDtypeStruct((in_structure,), dtype)
    if isinstance(in_structure, tuple) and all(
        isinstance(d, int) for d in in_structure
    ):
        return jax.ShapeDtypeStruct(in_structure, dtype)
    return in_structure


def _coerce_operator(
    operator: OperatorLike,
    vector: Float[Array, " n"],
) -> lx.AbstractLinearOperator:
    """Normalise *operator* into a lineax operator.

    A bare ``(matvec, shape)`` pair is wrapped with no structural assumptions;
    callers wanting tags should build the operator via :func:`as_linear_operator`
    and pass it directly.
    """
    if isinstance(operator, lx.AbstractLinearOperator):
        return operator
    if isinstance(operator, tuple) and len(operator) == 2:
        matvec, shape = operator
        return as_linear_operator(matvec, shape=shape, dtype=vector.dtype)
    raise TypeError(
        "operator must be a lineax AbstractLinearOperator or a "
        "(matvec, shape) tuple; got "
        f"{type(operator).__name__}."
    )


def _negate(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Return ``-A`` as a symmetric PSD operator.

    Negating a symmetric *negative*-semidefinite operator yields a symmetric
    *positive*-semidefinite one, which CG can solve directly.
    """
    return lx.FunctionLinearOperator(
        lambda v: -operator.mv(v),
        operator.in_structure(),
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
    )


def _default_solver(operator: lx.AbstractLinearOperator) -> AbstractSolveStrategy:
    """Pick a default solve strategy from the operator's structural tags."""
    if is_positive_semidefinite(operator):
        return CGSolver()
    if is_symmetric(operator):
        return MINRESSolver()
    raise ValueError(
        "Could not infer a default solver for a non-symmetric operator. "
        "Pass an explicit `solver=` or tag the operator via "
        "`as_linear_operator(..., symmetric=/positive_semidefinite=)`."
    )


def _coerce_preconditioner(
    preconditioner: PreconditionerLike,
    operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Normalise a preconditioner into a lineax operator applying ``M^{-1}``."""
    as_operator = getattr(preconditioner, "as_operator", None)
    if callable(as_operator) and not isinstance(
        preconditioner, lx.AbstractLinearOperator
    ):
        preconditioner = as_operator()
    if isinstance(preconditioner, lx.AbstractLinearOperator):
        # lineax CG requires the preconditioner to be tagged PSD.
        if is_positive_semidefinite(preconditioner):
            return preconditioner
        return lx.TaggedLinearOperator(preconditioner, lx.positive_semidefinite_tag)
    if callable(preconditioner):
        return lx.FunctionLinearOperator(
            preconditioner,
            operator.out_structure(),
            lx.positive_semidefinite_tag,
        )
    raise TypeError(
        "preconditioner must be a lineax operator, a callable applying "
        "M^{-1}, or an object with an `.as_operator()` method."
    )


def _solve_preconditioned(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " n"],
    solver: AbstractSolveStrategy,
    preconditioner: PreconditionerLike,
) -> Float[Array, " n"]:
    """Solve with a preconditioner via lineax CG.

    Preconditioning currently routes through lineax's CG with the
    ``preconditioner`` option, reusing the tolerances of a :class:`CGSolver`
    when one is supplied. Richer per-strategy preconditioner support arrives
    with the Phase 1 preconditioner protocol.
    """
    precond_op = _coerce_preconditioner(preconditioner, operator)
    if isinstance(solver, CGSolver):
        cg = lx.CG(rtol=solver.rtol, atol=solver.atol, max_steps=solver.max_steps)
    else:
        cg = lx.CG(rtol=1e-5, atol=1e-5, max_steps=1000)
    return lx.linear_solve(
        operator,
        vector,
        cg,
        options={"preconditioner": precond_op},
    ).value
