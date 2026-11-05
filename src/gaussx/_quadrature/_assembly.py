"""Shared assembly of output moments from weighted point evaluations."""

from __future__ import annotations

import einx
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._linalg._symmetrize import symmetrize
from gaussx._quadrature._types import GaussianState, PropagationResult


def assemble_propagation_result(
    points: Float[Array, "P N"],
    outputs: Float[Array, "P M"],
    input_mean: Float[Array, " N"],
    mean_weights: Float[Array, " P"],
    covariance_weights: Float[Array, " P"] | None = None,
    *,
    psd: bool = False,
) -> PropagationResult:
    """Assemble the moments of y = g(x) from weighted point evaluations.

        μ_y = ∑ᵢ w_m,i yᵢ
        Σ_y = ∑ᵢ w_c,i (yᵢ − μ_y)(yᵢ − μ_y)ᵀ
        Σ_xy = ∑ᵢ w_c,i (χᵢ − μ_x)(yᵢ − μ_y)ᵀ

    with yᵢ = g(χᵢ). Shared by the point-based integrators (Gauss-Hermite,
    unscented, cubature, Monte Carlo, ADF).

    Algorithm:
        1. Average the evaluations for μ_y.
        2. Centre the outputs on μ_y and the points on μ_x.
        3. Contract the weighted anomalies for Σ_y and Σ_xy, and symmetrise
           Σ_y.

    Args:
        points: Points χᵢ in the input space.
        outputs: Evaluations yᵢ = g(χᵢ).
        input_mean: Input mean μ_x.
        mean_weights: Mean weights w_m,i.
        covariance_weights: Covariance weights w_c,i. Defaults to
            ``mean_weights``.
        psd: Tag Σ_y ``positive_semidefinite_tag`` (pass the rule's
            ``guarantees_psd(D_x)``); otherwise it is tagged ``symmetric_tag``.
            See `PropagationResult`.

    Returns:
        `PropagationResult` holding 𝒩(μ_y, Σ_y) and Σ_xy.

    Examples:
        The rule (−1, 1) with equal weights on the map y = 2 x:

        >>> import jax.numpy as jnp
        >>> points = jnp.array([[-1.0], [1.0]])
        >>> weights = jnp.array([0.5, 0.5])
        >>> result = assemble_propagation_result(
        ...     points, 2.0 * points, jnp.zeros(1), weights
        ... )
        >>> bool(jnp.allclose(result.state.cov.as_matrix(), jnp.array([[4.0]])))
        True
        >>> bool(jnp.allclose(result.cross_cov, jnp.array([[2.0]])))
        True
    """
    if covariance_weights is None:
        covariance_weights = mean_weights

    # μ_y = ∑ᵢ w_m,i yᵢ
    output_mean = einsum(mean_weights, outputs, "point, point out -> out")

    # yᵢ − μ_y
    output_anomalies = einx.subtract(
        "point out, out -> point out", outputs, output_mean
    )
    # χᵢ − μ_x
    input_anomalies = einx.subtract(
        "point state, state -> point state", points, input_mean
    )

    # w_c,i (yᵢ − μ_y), contracted over the points without a (P, M, M) temporary
    weighted_anomalies = einx.multiply(
        "point, point out -> point out", covariance_weights, output_anomalies
    )
    # ∑ᵢ w_c,i (yᵢ − μ_y)(yᵢ − μ_y)ᵀ
    output_scatter = einsum(
        weighted_anomalies, output_anomalies, "point row, point col -> row col"
    )
    # Σ_y, symmetrised against round-off
    output_covariance = symmetrize(output_scatter)

    # Σ_xy = ∑ᵢ w_c,i (χᵢ − μ_x)(yᵢ − μ_y)ᵀ
    cross_covariance = einsum(
        input_anomalies, weighted_anomalies, "point state, point out -> state out"
    )

    tag = lx.positive_semidefinite_tag if psd else lx.symmetric_tag
    output_covariance_op = lx.MatrixLinearOperator(output_covariance, tag)
    output_state = GaussianState(mean=output_mean, cov=output_covariance_op)
    return PropagationResult(state=output_state, cross_cov=cross_covariance)
