"""Core types for uncertainty propagation."""

from __future__ import annotations

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float


class GaussianState(eqx.Module):
    """Gaussian distribution as (mean, covariance operator) pair.

    Attributes:
        mean: Mean vector, shape ``(N,)``.
        cov: Covariance operator, shape ``(N, N)``.
    """

    mean: Float[Array, " N"]
    cov: lx.AbstractLinearOperator


class PropagationResult(eqx.Module):
    """Output of uncertainty propagation through a nonlinear function.

    Attributes:
        state: Output Gaussian distribution.
        cross_cov: Input-output cross-covariance, shape ``(N_in, N_out)``.
            Used for downstream Kalman updates. ``None`` if not computed.
    """

    state: GaussianState
    cross_cov: Float[Array, "N_in N_out"] | None
