"""Front door for the unified solver layer: solve A x = b from anything.

A is a square N × N operator, b ∈ ℝᴺ the right-hand side and x ∈ ℝᴺ the
solution; an optional preconditioner applies M⁻¹ ≈ A⁻¹ to each CG residual.
This module provides two things:

- `as_linear_operator`, a thin adapter that wraps a raw ``matvec`` callable
  v ↦ A v (and an optional shape) into a `lineax.AbstractLinearOperator`
  with the appropriate structural tags. This is the entry point used by
  matrix-free callers (e.g. finite-volume or spectral PDE solvers) that hand
  gaussx a bare function rather than a built operator object.
- `linear_solve`, a unified solve entry point that normalises its operator
  argument (an operator *or* ``(matvec, shape)``), handles the
  negative-definite sign convention of elliptic operators, selects a
  sensible default `AbstractSolveStrategy`, and optionally attaches a
  preconditioner.

The design goal is that callers in other packages pre-transform their problem
into an ``(operator, rhs)`` pair, call `linear_solve`, and post-transform
the result -- without gaussx ever needing to know about grids, boundary
conditions, or transforms.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import lineax as lx
from jaxtyping import Array, Float

from gaussx._preconditioners import AbstractPreconditioner, OperatorPreconditioner
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import AbstractLogdetStrategy, AbstractSolveStrategy
from gaussx._strategies._cg import CGSolver
from gaussx._strategies._composed import ComposedSolver
from gaussx._strategies._keyed import KeyedSolver
from gaussx._strategies._lineax import as_solve_strategy
from gaussx._strategies._minres import MINRESSolver
from gaussx._strategies._precond_cg import PreconditionedCGSolver
from gaussx._tags import (
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
)


MatvecLike = Callable[[Float[Array, " n"]], Float[Array, " n"]]
"""A raw matrix-vector product v ↦ A v."""

OperatorLike = lx.AbstractLinearOperator | tuple[MatvecLike, tuple[int, int]]
"""Either a built operator or a ``(matvec, shape)`` pair."""

PreconditionerLike = AbstractPreconditioner | lx.AbstractLinearOperator | MatvecLike
"""A preconditioner: an `AbstractPreconditioner`, a lineax operator applying
M⁻¹, or a callable v ↦ M⁻¹ v (gh-405)."""


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

        v ↦ A v

    This is the matrix-free front door: callers that only have the product
    v ↦ A v (rather than a structured operator object) use this to obtain
    an operator that gaussx's solvers and primitives understand.

    Args:
        matvec: The matrix-vector product v ↦ A v.
        shape: Matrix shape ``(out, in)``. The operator's input structure is
            inferred as a length-``in`` vector of ``dtype``. Ignored when
            ``in_structure`` is given.
        in_structure: An explicit input structure: a
            ``jax.ShapeDtypeStruct``, an ``int`` (a vector length), a tuple
            of ints (a shape), or a PyTree of structures. Takes precedence
            over ``shape``.
        dtype: Dtype used when building the input structure from ``shape``.
        symmetric: Tag the operator symmetric (A = Aᵀ).
        positive_semidefinite: Tag the operator PSD (implies symmetric).
        negative_definite: Tag the operator negative semidefinite (implies
            symmetric). Use this for elliptic operators such as a discrete
            Laplacian; `linear_solve` will route the solve through the
            equivalent positive-definite system.

    Returns:
        A `lineax.FunctionLinearOperator` carrying the requested tags.

    Raises:
        ValueError: If neither ``shape`` nor ``in_structure`` is provided.

    Examples:
        A matrix-free negative-definite Laplacian, solved by `linear_solve`:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> laplacian = jnp.array([[-2.0, 1.0], [1.0, -2.0]])
        >>> A = gaussx.as_linear_operator(
        ...     lambda v: laplacian @ v,
        ...     shape=(2, 2),
        ...     dtype=laplacian.dtype,
        ...     negative_definite=True,
        ... )
        >>> A.in_structure().shape
        (2,)
        >>> x = gaussx.linear_solve(A, jnp.array([1.0, 0.0]))
        >>> bool(jnp.allclose(laplacian @ x, jnp.array([1.0, 0.0]), atol=1e-4))
        True
    """
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
    solver: AbstractSolveStrategy | lx.AbstractLinearSolver | None = None,
    preconditioner: PreconditionerLike | None = None,
) -> Float[Array, " n"]:
    """Solve A x = b through the unified front door.

        x = A⁻¹ b

    Accepts either a built operator or a ``(matvec, shape)`` pair, handles the
    negative-definite sign convention, picks a default solver when none is
    given, and optionally attaches a preconditioner.

    Sign handling: an operator tagged *negative* semidefinite (and not PSD)
    is solved through the equivalent positive-definite system (−A) x = −b, so
    that a CG-style solver can be used directly. This is the common case for
    elliptic PDE operators (e.g. a discrete Laplacian), which finite-volume
    and spectral callers hand over as negative-definite matvecs.

    Default solver selection (when ``solver is None``):

    - positive semidefinite operator: `gaussx.AutoSolver`, an exact
      structural or dense solve for structured or small operators and
      `gaussx.CGSolver` for large ones (gh-390);
    - symmetric (possibly indefinite) operator: `gaussx.MINRESSolver`;
    - otherwise a ``ValueError`` asking for an explicit solver.

    Algorithm:
        1. Wrap a ``(matvec, shape)`` pair as an untagged operator.
        2. If A is negative semidefinite and not PSD, solve (−A) x = −b.
        3. Take ``solver``, or the default from A's tags.
        4. Attach the preconditioner M⁻¹ to the CG solve, if one is given.

    Args:
        operator: The linear operator A, or a ``(matvec, shape)`` pair.
        vector: Right-hand side b, of length N.
        solver: Solve strategy to use, or a lineax solver (wrapped in
            `gaussx.LineaxSolver`). When ``None`` a default is selected from
            the operator's structural tags.
        preconditioner: Optional preconditioner. May be an
            `AbstractPreconditioner`, a lineax operator applying M⁻¹, or a
            callable v ↦ M⁻¹ v. It is attached to
            the CG solve: a `CGSolver` directly, the solve half of a
            `ComposedSolver` or a `KeyedSolver`, and the CG that an
            `AutoSolver` builds for a large operator (a direct solve
            ignores it). Other strategies raise ``ValueError``.

    Returns:
        The solution x, of length N.

    Raises:
        TypeError: If ``operator`` or ``preconditioner`` has an unsupported
            type.
        ValueError: If no default solver fits A, or ``solver`` cannot take
            a preconditioner.

    Examples:
        A PSD system with the default solver and with a preconditioned CG:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> b = jnp.array([1.0, 2.0])
        >>> x = gaussx.linear_solve(A, b)
        >>> bool(jnp.allclose(matrix @ x, b, atol=1e-4))
        True
        >>> x_cg = gaussx.linear_solve(
        ...     A,
        ...     b,
        ...     solver=gaussx.CGSolver(),
        ...     preconditioner=gaussx.JacobiPreconditioner(),
        ... )
        >>> bool(jnp.allclose(x_cg, x, atol=1e-3))
        True

        A non-symmetric operator needs an explicit solver:

        >>> gaussx.linear_solve(
        ...     lx.MatrixLinearOperator(jnp.array([[1.0, 2.0], [0.0, 1.0]])), b
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Could not infer a default solver for a non-symmetric operator.
    """
    op = _coerce_operator(operator, vector)

    if is_negative_semidefinite(op) and not is_positive_semidefinite(op):
        # (−A) x = −b, the equivalent positive-definite system
        op = _negate(op)
        vector = -vector

    solver = as_solve_strategy(solver)
    if solver is None:
        solver = _default_solver(op)

    if preconditioner is not None:
        solver = _attach_preconditioner(solver, _as_preconditioner(preconditioner))

    return solver.solve(op, vector)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_in_structure(in_structure: object, dtype: object) -> object:
    """Normalise an ``in_structure`` argument into a ShapeDtypeStruct PyTree.

    Args:
        in_structure: An ``int`` (vector length), a tuple of ints (shape),
            or a structure PyTree, returned unchanged.
        dtype: Dtype for the int and tuple forms.

    Returns:
        The input structure of the operator.
    """
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
    """Normalise ``operator`` into a lineax operator.

    A bare ``(matvec, shape)`` pair is wrapped with no structural assumptions;
    callers wanting tags should build the operator via `as_linear_operator`
    and pass it directly.

    Args:
        operator: Operator A, or a ``(matvec, shape)`` pair.
        vector: Right-hand side b, whose dtype the wrapped operator takes.

    Returns:
        A as a lineax operator.

    Raises:
        TypeError: If ``operator`` is neither form.
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
    """Return −A as a symmetric PSD operator.

    Negating a symmetric *negative*-semidefinite operator yields a symmetric
    *positive*-semidefinite one, which CG can solve directly. Plain negation
    keeps the structure (a lineax ``NegLinearOperator``, which every gaussx
    primitive unwraps, or an operator that negates in place such as a
    `gaussx.DiagonalizedOperator`), so structural dispatch still sees (and
    solves exactly) a diagonal or Kronecker A (gh-390). The tags are set
    explicitly because they do not always follow the negation (gh-391).

    Args:
        operator: Symmetric negative-semidefinite operator A.

    Returns:
        −A, tagged symmetric and positive semidefinite.
    """
    # −A
    return lx.TaggedLinearOperator(
        -operator, (lx.symmetric_tag, lx.positive_semidefinite_tag)
    )


def _default_solver(operator: lx.AbstractLinearOperator) -> AbstractSolveStrategy:
    """Pick a default solve strategy from the operator's structural tags.

    Args:
        operator: Operator A.

    Returns:
        `gaussx.AutoSolver` for a PSD A, `gaussx.MINRESSolver` for a
        symmetric one.

    Raises:
        ValueError: If A is tagged neither PSD nor symmetric.
    """
    if is_positive_semidefinite(operator):
        return AutoSolver()
    if is_symmetric(operator):
        return MINRESSolver()
    raise ValueError(
        "Could not infer a default solver for a non-symmetric operator. "
        "Pass an explicit `solver=` or tag the operator via "
        "`as_linear_operator(..., symmetric=/positive_semidefinite=)`."
    )


def _as_preconditioner(
    preconditioner: PreconditionerLike,
) -> AbstractPreconditioner:
    """Normalise ``preconditioner`` into an `AbstractPreconditioner`.

    A raw lineax operator or callable applying M⁻¹ is wrapped in an
    `OperatorPreconditioner`; an existing preconditioner is returned
    unchanged.

    Args:
        preconditioner: A preconditioner, an operator M⁻¹ or a callable
            v ↦ M⁻¹ v.

    Returns:
        The preconditioner as an `AbstractPreconditioner`.

    Raises:
        TypeError: If ``preconditioner`` is none of these.
    """
    if isinstance(preconditioner, AbstractPreconditioner):
        return preconditioner
    if isinstance(preconditioner, lx.AbstractLinearOperator) or callable(
        preconditioner
    ):
        return OperatorPreconditioner(preconditioner)
    raise TypeError(
        "preconditioner must be an AbstractPreconditioner, a lineax operator, "
        "or a callable applying M^{-1}."
    )


def _attach_preconditioner(
    solver: AbstractSolveStrategy,
    preconditioner: AbstractPreconditioner,
) -> AbstractSolveStrategy:
    """Attach ``preconditioner`` to the CG solve inside ``solver`` (gh-390).

    `CGSolver` and `AutoSolver` take it as a field; `ComposedSolver` and
    `KeyedSolver` pass it on to the strategy that solves. A
    `PreconditionedCGSolver` already has its own; other strategies have no
    preconditioned solve.

    Args:
        solver: Solve strategy.
        preconditioner: Preconditioner M⁻¹ to attach.

    Returns:
        The strategy with M⁻¹ attached to its CG solve.

    Raises:
        ValueError: If ``solver`` is a `PreconditionedCGSolver` or has no CG
            solve to precondition.
    """
    if isinstance(solver, CGSolver | AutoSolver):
        return dataclasses.replace(solver, preconditioner=preconditioner)
    if isinstance(solver, ComposedSolver):
        return dataclasses.replace(
            solver,
            solve_strategy=_attach_preconditioner(
                solver.solve_strategy, preconditioner
            ),
        )
    if isinstance(solver, KeyedSolver) and isinstance(
        solver.strategy, AbstractSolveStrategy
    ):
        attached = _attach_preconditioner(solver.strategy, preconditioner)
        assert isinstance(attached, AbstractLogdetStrategy)
        return dataclasses.replace(solver, strategy=attached)
    if isinstance(solver, PreconditionedCGSolver):
        raise ValueError(
            "PreconditionedCGSolver has its own preconditioner; pass it as "
            "PreconditionedCGSolver(preconditioner=...) instead of "
            "linear_solve(..., preconditioner=...)."
        )
    raise ValueError(
        f"Preconditioning is not implemented for {type(solver).__name__}. "
        "Use CGSolver, AutoSolver, or a ComposedSolver whose solve is one."
    )
