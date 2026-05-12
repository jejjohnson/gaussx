"""Matheron's-rule pathwise Gaussian conditioning."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._solve import solve


def matheron_update(
    prior_sample_target: Float[Array, "S N_star"],
    prior_sample_conditioning: Float[Array, "S M"],
    observed_value: Float[Array, " M"],
    cross_covariance: lx.AbstractLinearOperator,
    conditioning_covariance: lx.AbstractLinearOperator,
) -> Float[Array, "S N_star"]:
    r"""Posterior samples via Matheron's-rule correction.

    Given joint prior draws ``(a, b)`` and an observed conditioning value
    ``β``, Matheron's rule samples from ``a | b = β`` by applying

    .. math::

        a + \operatorname{Cov}(a, b)\operatorname{Cov}(b, b)^{-1}(β - b).

    This helper keeps both covariance arguments as lineax operators, so the
    conditioning solve uses the existing GaussX structural dispatch and the
    target correction is a rectangular matvec.

    Args:
        prior_sample_target: Prior samples at target points, shape
            ``(S, N_star)``.
        prior_sample_conditioning: Joint prior samples at conditioning
            points, shape ``(S, M)``.
        observed_value: Observed conditioning value, shape ``(M,)``.
        cross_covariance: Cross-covariance operator ``Cov(a, b)``, shape
            ``(N_star, M)``.
        conditioning_covariance: Conditioning covariance operator
            ``Cov(b, b)``, shape ``(M, M)``.

    Returns:
        Corrected posterior samples, shape ``(S, N_star)``.
    """
    prior_sample_target = jnp.asarray(prior_sample_target)
    prior_sample_conditioning = jnp.asarray(prior_sample_conditioning)
    observed_value = jnp.asarray(observed_value)
    dtype = jnp.result_type(
        prior_sample_target,
        prior_sample_conditioning,
        observed_value,
    )
    prior_sample_target = prior_sample_target.astype(dtype)
    prior_sample_conditioning = prior_sample_conditioning.astype(dtype)
    observed_value = observed_value.astype(dtype)

    if prior_sample_target.ndim != 2:
        raise ValueError("prior_sample_target must have shape (S, N_star).")
    if prior_sample_conditioning.ndim != 2:
        raise ValueError("prior_sample_conditioning must have shape (S, M).")
    if observed_value.ndim != 1:
        raise ValueError("observed_value must have shape (M,).")
    if prior_sample_target.shape[0] != prior_sample_conditioning.shape[0]:
        raise ValueError("prior samples must have the same sample dimension S.")

    num_conditioning = prior_sample_conditioning.shape[1]
    num_target = prior_sample_target.shape[1]
    if observed_value.shape[0] != num_conditioning:
        raise ValueError("observed_value must match the conditioning dimension M.")
    if conditioning_covariance.in_size() != num_conditioning:
        raise ValueError(
            "conditioning_covariance input size must match the conditioning "
            "dimension M."
        )
    if conditioning_covariance.out_size() != num_conditioning:
        raise ValueError(
            "conditioning_covariance output size must match the conditioning "
            "dimension M."
        )
    if cross_covariance.in_size() != num_conditioning:
        raise ValueError(
            "cross_covariance input size must match the conditioning dimension M."
        )
    if cross_covariance.out_size() != num_target:
        raise ValueError(
            "cross_covariance output size must match the target dimension N_star."
        )

    residuals = observed_value[None, :] - prior_sample_conditioning
    solves = jax.vmap(lambda residual: solve(conditioning_covariance, residual))(
        residuals
    )
    corrections = jax.vmap(cross_covariance.mv)(solves)
    return prior_sample_target + corrections
