"""Exact discretisation of stationary linear SDEs."""

from __future__ import annotations

from typing import overload

import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._linalg import cov_transform


@overload
def process_noise_covariance(
    A: Float[Array, "N N"],
    Pinf: Float[Array, "N N"],
) -> Float[Array, "N N"]: ...


@overload
def process_noise_covariance(
    A: lx.AbstractLinearOperator,
    Pinf: Float[Array, "N N"] | lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator: ...


@overload
def process_noise_covariance(
    A: Float[Array, "N N"] | lx.AbstractLinearOperator,
    Pinf: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator: ...


def process_noise_covariance(
    A: Float[Array, "N N"] | lx.AbstractLinearOperator,
    Pinf: Float[Array, "N N"] | lx.AbstractLinearOperator,
) -> Float[Array, "N N"] | lx.AbstractLinearOperator:
    r"""Compute process noise from stationary covariance.

    Computes:

        Q = Pinf - A @ Pinf @ A^T

    For a discrete-time state-space model with stationary covariance
    ``Pinf`` and transition matrix ``A``. This is the standard
    "steady-state trick": it follows from stationarity of the discretised
    recursion ``Pinf = A Pinf A^T + Q``, and avoids matrix-fraction
    integration (Särkkä & Solin 2019, §6.4).

    It is the forward direction of the discrete Lyapunov equation that
    `gaussx.discrete_lyapunov_solve` inverts: this maps
    ``Pinf -> Q``, that one maps ``Q -> Pinf`` for the same ``A``.

    The ``A Pinf A^T`` term is delegated to
    `gaussx.cov_transform`, so structure is exploited rather than
    re-derived: a diagonal ``Pinf`` skips its ``(N, N)`` materialization, and
    matched `gaussx.Kronecker` / `gaussx.BlockDiag`
    operands stay factorised through `gaussx.sandwich`.

    Passing **operators** returns a lazy operator — nothing is materialized,
    and the structural class of the result follows `gaussx.sandwich`.
    Passing **arrays** returns an array, as before.

    The result is not symmetrised. Callers that need a covariance clean of
    floating-point asymmetry should wrap it in `gaussx.symmetrize`,
    as `gaussx.SDEKernel.discretise` does.

    Args:
        A: State transition matrix, shape ``(N, N)`` — array or operator.
        Pinf: Stationary covariance, shape ``(N, N)`` — array or operator.

    Returns:
        Process noise covariance Q, shape ``(N, N)``. An operator when
        either argument is an operator, otherwise an array.
    """
    Pinf_op = (
        Pinf
        if isinstance(Pinf, lx.AbstractLinearOperator)
        else lx.MatrixLinearOperator(Pinf, lx.symmetric_tag)
    )
    # One implementation of the congruence, for both paths. Wrapping a dense
    # ``Pinf`` costs nothing at runtime — it is pytree bookkeeping at trace
    # time, and the dense branch of ``cov_transform`` evaluates the identical
    # ``A @ Pinf @ A.T`` expression.
    congruence = cov_transform(A, Pinf_op)

    if isinstance(A, lx.AbstractLinearOperator) or isinstance(
        Pinf, lx.AbstractLinearOperator
    ):
        return Pinf_op - congruence
    return Pinf - congruence.as_matrix()
