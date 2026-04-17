"""Shared assembly for integrator output moments."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._quadrature._types import GaussianState, PropagationResult


def assemble_propagation_result(
    chi: Float[Array, "P N"],
    Y: Float[Array, "P M"],
    mu: Float[Array, " N"],
    w_m: Float[Array, " P"],
    w_c: Float[Array, " P"] | None = None,
) -> PropagationResult:
    """Assemble output distribution from sigma points and function values.

    Shared helper used by all sigma-point integrators (Gauss-Hermite,
    unscented, Monte Carlo, ADF) to compute output moments from
    weighted point evaluations.

    Args:
        chi: Sigma/quadrature points in input space, shape ``(P, N)``.
        Y: Function evaluations at sigma points, shape ``(P, M)``.
        mu: Input mean, shape ``(N,)``.
        w_m: Mean weights, shape ``(P,)``.
        w_c: Covariance weights, shape ``(P,)``. Defaults to ``w_m``.

    Returns:
        ``PropagationResult`` with output Gaussian and cross-covariance.
    """
    if w_c is None:
        w_c = w_m

    # Output mean: μ_y = Σᵢ wᵢᵐ yᵢ
    mu_y = jnp.sum(w_m[:, None] * Y, axis=0)  # (M,)

    # Residuals
    dy = Y - mu_y[None, :]  # (P, M)
    dx = chi - mu[None, :]  # (P, N)

    # Output covariance: Σ_y = Σᵢ wᵢᶜ (yᵢ − μ_y)(yᵢ − μ_y)ᵀ
    Sigma_y = jnp.sum(
        w_c[:, None, None] * (dy[:, :, None] * dy[:, None, :]),
        axis=0,
    )
    Sigma_y = 0.5 * (Sigma_y + Sigma_y.T)  # enforce symmetry

    # Cross-covariance: C_xy = Σᵢ wᵢᶜ (xᵢ − μ)(yᵢ − μ_y)ᵀ
    cross_cov = jnp.sum(
        w_c[:, None, None] * (dx[:, :, None] * dy[:, None, :]),
        axis=0,
    )

    cov_y = lx.MatrixLinearOperator(Sigma_y, lx.symmetric_tag)
    out_state = GaussianState(mean=mu_y, cov=cov_y)
    return PropagationResult(state=out_state, cross_cov=cross_cov)
