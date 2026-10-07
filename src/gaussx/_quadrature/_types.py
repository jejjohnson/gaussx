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

    Tag convention for ``state.cov``: `lineax.positive_semidefinite_tag`
    when the rule guarantees a PSD covariance at the input dimension
    (``integrator.guarantees_psd(N)``: Gauss-Hermite, spherical cubature,
    unscented with non-negative weights, Monte Carlo, ADF, Taylor), else
    `lineax.symmetric_tag` (negative-weight rules such as the default
    scaled unscented transform or the fifth-order cubature rule).
    Dispatch on `lineax.is_positive_semidefinite` therefore depends only
    on whether PSD is guaranteed, not on which rule produced the state.

    Attributes:
        state: Output Gaussian distribution.
        cross_cov: Input-output cross-covariance, shape ``(N_in, N_out)``.
            Used for downstream Kalman updates. ``None`` if not computed.
    """

    state: GaussianState
    cross_cov: Float[Array, "N_in N_out"] | None
