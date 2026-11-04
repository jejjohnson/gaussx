"""Core types for uncertainty propagation: Gaussian belief and propagation result."""

from __future__ import annotations

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float


class GaussianState(eqx.Module):
    """Gaussian belief as a pair of mean and covariance operator.

        x ~ 𝒩(μ_x, Σ_x),    x ∈ ℝ^D_x

    Attributes:
        mean: Mean μ_x.
        cov: Covariance Σ_x, as a lineax operator.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> state = GaussianState(mean=jnp.zeros(2), cov=covariance)
        >>> state.mean.shape
        (2,)
    """

    mean: Float[Array, " N"]
    cov: lx.AbstractLinearOperator


class PropagationResult(eqx.Module):
    """Output moments of a Gaussian pushed through a nonlinear map g.

        y = g(x),    x ~ 𝒩(μ_x, Σ_x),    y ≈ 𝒩(μ_y, Σ_y),    Σ_xy = Cov[x, y]

    Tag convention for ``state.cov``: `lineax.positive_semidefinite_tag`
    when the rule guarantees a PSD covariance at the input dimension
    (``integrator.guarantees_psd(N)``: Gauss-Hermite, spherical cubature,
    unscented with non-negative weights, Monte Carlo, ADF, Taylor), else
    `lineax.symmetric_tag` (negative-weight rules such as the default
    scaled unscented transform or the fifth-order cubature rule).
    Dispatch on `lineax.is_positive_semidefinite` therefore depends only
    on whether PSD is guaranteed, not on which rule produced the state.

    Attributes:
        state: Output Gaussian 𝒩(μ_y, Σ_y).
        cross_cov: Input-output cross-covariance Σ_xy, used for downstream
            Kalman updates. ``None`` if not computed.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(1))
        >>> state = GaussianState(mean=jnp.zeros(1), cov=covariance)
        >>> result = PropagationResult(state=state, cross_cov=None)
        >>> result.cross_cov is None
        True
    """

    state: GaussianState
    cross_cov: Float[Array, "N_in N_out"] | None
