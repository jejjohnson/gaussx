"""Shared assembly for integrator output moments."""

from __future__ import annotations

import einx
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._linalg._symmetrize import symmetrize
from gaussx._quadrature._types import GaussianState, PropagationResult


def assemble_propagation_result(
    chi: Float[Array, "P N"],
    Y: Float[Array, "P M"],
    mu: Float[Array, " N"],
    w_m: Float[Array, " P"],
    w_c: Float[Array, " P"] | None = None,
    *,
    psd: bool = False,
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
        psd: Tag the output covariance ``positive_semidefinite_tag``
            (pass the rule's ``guarantees_psd(N)``); otherwise it is tagged
            ``symmetric_tag``. See `PropagationResult`.

    Returns:
        ``PropagationResult`` with output Gaussian and cross-covariance.
    """
    if w_c is None:
        w_c = w_m

    # Output mean: μ_y = Σᵢ wᵢᵐ yᵢ
    mu_y = einsum(w_m, Y, "p, p m -> m")  # (M,)

    # Residuals
    dy = einx.subtract("p m, m -> p m", Y, mu_y)  # (P, M)
    dx = einx.subtract("p n, n -> p n", chi, mu)  # (P, N)

    # Output covariance: Σ_y = Σᵢ wᵢᶜ (yᵢ − μ_y)(yᵢ − μ_y)ᵀ, contracted over
    # the points without a (P, M, M) temporary.
    w_dy = einx.multiply("p, p m -> p m", w_c, dy)
    Sigma_y = symmetrize(einsum(w_dy, dy, "p i, p j -> i j"))

    # Cross-covariance: C_xy = Σᵢ wᵢᶜ (xᵢ − μ)(yᵢ − μ_y)ᵀ
    cross_cov = einsum(dx, w_dy, "p n, p m -> n m")

    tag = lx.positive_semidefinite_tag if psd else lx.symmetric_tag
    cov_y = lx.MatrixLinearOperator(Sigma_y, tag)
    out_state = GaussianState(mean=mu_y, cov=cov_y)
    return PropagationResult(state=out_state, cross_cov=cross_cov)
