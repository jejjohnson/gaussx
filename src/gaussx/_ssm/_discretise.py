"""Exact discretisation of stationary linear SDEs.

Leaf module (no gaussx imports) so both `gaussx._ssm._sde_kernel` and the
backwards-compatible `gaussx._inference` re-export can depend on it without
an import cycle.
"""

from __future__ import annotations

from jaxtyping import Array, Float


def process_noise_covariance(
    A: Float[Array, "N N"],
    Pinf: Float[Array, "N N"],
) -> Float[Array, "N N"]:
    """Compute process noise from stationary covariance.

    Computes:

        Q = Pinf - A @ Pinf @ A^T

    For a discrete-time state-space model with stationary covariance
    ``Pinf`` and transition matrix ``A``. This is the standard
    "steady-state trick": it follows from stationarity of the discretised
    recursion ``Pinf = A Pinf A^T + Q``, and avoids matrix-fraction
    integration (Särkkä & Solin 2019, §6.4).

    The result is not symmetrised — callers that need a covariance clean of
    floating-point asymmetry should wrap the result in
    `gaussx.symmetrize`, as `gaussx.SDEKernel.discretise` does.

    Args:
        A: State transition matrix, shape ``(N, N)``.
        Pinf: Stationary covariance, shape ``(N, N)``.

    Returns:
        Process noise covariance Q, shape ``(N, N)``.
    """
    return Pinf - A @ Pinf @ A.T
