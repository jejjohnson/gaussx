"""Known-value elimination: solve A x = b with x prescribed on an index set.

A is a square N × N operator, b ∈ ℝᴺ the right-hand side and x ∈ ℝᴺ the
solution. A boolean mask splits the indices into the known set K, where
x_K = g_K is prescribed, and the unknown set E, its complement. The rows of
the known entries are dropped, and the unknowns solve the reduced system

    A_EE x_E = b_E − A_EK g_K.

With the lift g̃ (g_K on K, zero on E), the right-hand side is (b − A g̃)_E,
which needs only products with A. The reduced operator is applied at full
size, as Ã = P A P ± (I − P) with P = diag(1_E): Ã agrees with A_EE on E,
is the identity (negated for a negative-semidefinite A) on K, and keeps A's
symmetry and definiteness, so any gaussx strategy, CG included, solves it.
The mask may be a traced array, so the same program serves every mask.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._solve_frontend import (
    OperatorLike,
    PreconditionerLike,
    _coerce_operator,
    linear_solve,
)
from gaussx._strategies._base import AbstractSolveStrategy
from gaussx._tags import (
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
)


def lift_known_values(
    operator: OperatorLike,
    vector: Float[Array, " N"],
    known: Bool[Array, " N"],
    values: Float[Array, " N"],
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    """Lift prescribed values to the right-hand side of A x = b.

        g̃ = g on K, 0 on E,   r = (b − A g̃) on E, 0 on K

    Args:
        operator: Square operator A, N × N, or a ``(matvec, shape)`` pair.
        vector: Right-hand side b, of length N.
        known: Mask of the known set K (``True`` where x is prescribed).
        values: Prescribed values g, of length N; read only on K.

    Returns:
        The lifted right-hand side r, zero on K, and the lift g̃, zero on E.

    Examples:
        The two ends of a 3-point Laplacian on 5 points prescribed:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> L = 2 * jnp.eye(5) - jnp.eye(5, k=1) - jnp.eye(5, k=-1)
        >>> A = lx.MatrixLinearOperator(L, lx.positive_semidefinite_tag)
        >>> known = jnp.array([True, False, False, False, True])
        >>> values = jnp.array([1.0, 0.0, 0.0, 0.0, 3.0])
        >>> rhs, lift = gaussx.lift_known_values(A, jnp.zeros(5), known, values)
        >>> rhs.tolist()
        [0.0, 1.0, 0.0, 3.0, 0.0]
        >>> lift.tolist()
        [1.0, 0.0, 0.0, 0.0, 3.0]
    """
    op = _coerce_operator(operator, vector)
    zero = jnp.zeros((), vector.dtype)
    # g̃ = g on K, 0 on E
    lift = jnp.where(known, values.astype(vector.dtype), zero)
    # r = (b − A g̃) on E, 0 on K
    rhs = jnp.where(known, zero, vector - op.mv(lift))
    return rhs, lift


def solve_with_known_values(
    operator: OperatorLike,
    vector: Float[Array, " N"],
    known: Bool[Array, " N"],
    values: Float[Array, " N"],
    *,
    solver: AbstractSolveStrategy | lx.AbstractLinearSolver | None = None,
    preconditioner: PreconditionerLike | None = None,
) -> Float[Array, " N"]:
    """Solve A x = b on the unknowns, with x prescribed on the known set.

        x_K = g_K,   A_EE x_E = b_E − A_EK g_K

    The rows of A on K are dropped: b is read only on E. The reduced system
    is solved at full size through `gaussx.linear_solve` (so ``solver`` and
    ``preconditioner`` mean what they mean there) on
    Ã = P A P ± (I − P), P = diag(1_E), which is symmetric (positive or
    negative) definite when A is.

    Algorithm:
        1. Lift: g̃ = g on K, r = (b − A g̃) on E (`lift_known_values`).
        2. Solve Ã y = r; y vanishes on K because r does.
        3. x = g on K, y on E.

    Args:
        operator: Square operator A, N × N, or a ``(matvec, shape)`` pair.
        vector: Right-hand side b, of length N.
        known: Mask of the known set K (``True`` where x is prescribed).
        values: Prescribed values g, of length N; read only on K.
        solver: Solve strategy for Ã, as in `gaussx.linear_solve`; ``None``
            picks one from A's tags.
        preconditioner: Optional preconditioner for the CG solve of Ã, as
            in `gaussx.linear_solve`.

    Returns:
        The solution x, of length N, equal to g on K.

    Examples:
        The ends of a 1-D Laplacian set to 1 and 3, with no source: the
        solution is the straight line between them.

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> L = 2 * jnp.eye(9) - jnp.eye(9, k=1) - jnp.eye(9, k=-1)
        >>> A = lx.MatrixLinearOperator(L, lx.positive_semidefinite_tag)
        >>> known = jnp.zeros(9, bool).at[jnp.array([0, 8])].set(True)
        >>> values = jnp.zeros(9).at[0].set(1.0).at[8].set(3.0)
        >>> x = gaussx.solve_with_known_values(
        ...     A, jnp.zeros(9), known, values, solver=gaussx.CGSolver()
        ... )
        >>> [round(v, 2) for v in x.tolist()]
        [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    """
    op = _coerce_operator(operator, vector)
    rhs, _ = lift_known_values(op, vector, known, values)
    reduced = _reduced_operator(op, known, vector.dtype)
    # y = Ã⁻¹ r, zero on K
    y = linear_solve(reduced, rhs, solver=solver, preconditioner=preconditioner)
    # x = g on K, y on E
    return jnp.where(known, values.astype(y.dtype), y)


def _reduced_operator(
    operator: lx.AbstractLinearOperator,
    known: Bool[Array, " N"],
    dtype: jnp.dtype,
) -> lx.AbstractLinearOperator:
    """Ã = P A P ± (I − P), P = diag(1_E), with A's tags.

    The sign of the identity block on K follows A's definiteness, so Ã is
    definite whenever A is: −1 for a negative-semidefinite A, +1 otherwise.

    Args:
        operator: Square operator A, N × N.
        known: Mask of the known set K.
        dtype: Dtype of P.

    Returns:
        Ã, tagged symmetric, PSD or NSD as A is.
    """
    negative = is_negative_semidefinite(operator) and not is_positive_semidefinite(
        operator
    )
    # 1_E and ±1_K
    unknown = jnp.where(known, 0.0, 1.0).astype(dtype)
    pad = jnp.where(known, -1.0 if negative else 1.0, 0.0).astype(dtype)
    project = lx.DiagonalLinearOperator(unknown)
    # P A P ± (I − P)
    reduced = project @ operator @ project + lx.DiagonalLinearOperator(pad)
    tags: list[object] = []
    if is_symmetric(operator):
        tags.append(lx.symmetric_tag)
    if negative:
        tags.append(lx.negative_semidefinite_tag)
    elif is_positive_semidefinite(operator):
        tags.append(lx.positive_semidefinite_tag)
    return lx.TaggedLinearOperator(reduced, tuple(tags))
