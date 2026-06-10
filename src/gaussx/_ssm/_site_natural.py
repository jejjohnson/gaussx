"""Per-site (scalar/diagonal) natural parameter conversions.

These functions implement expectation-propagation (EP) site operations on
scalar or diagonal Gaussians. They are not general-purpose Gaussian
parameterization conversions — for those see `gaussx._expfam._natural`.

For block-tridiagonal (SSM) parameterizations see
`gaussx._ssm._ssm_natural`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def site_natural_from_tilted(
    tilted_mean: Float[Array, " *batch"],
    tilted_var: Float[Array, " *batch"],
    cav_mean: Float[Array, " *batch"],
    cav_var: Float[Array, " *batch"],
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
    """Compute site natural parameters from tilted and cavity moments.

    Args:
        tilted_mean: Tilted distribution means.
        tilted_var: Tilted distribution variances (positive).
        cav_mean: Cavity distribution means.
        cav_var: Cavity distribution variances (positive).

    Returns:
        Tuple ``(site_nat1, site_nat2)``.
    """
    site_nat2 = jnp.reciprocal(tilted_var) - jnp.reciprocal(cav_var)
    site_nat1 = tilted_mean * jnp.reciprocal(tilted_var) - cav_mean * jnp.reciprocal(
        cav_var
    )
    return site_nat1, site_nat2


def site_mean_var_from_natural(
    site_nat1: Float[Array, " *batch"],
    site_nat2: Float[Array, " *batch"],
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
    """Convert per-site natural parameters to mean/variance.

    Args:
        site_nat1: Site precision-weighted means.
        site_nat2: Site precisions (positive for valid Gaussians).

    Returns:
        Tuple ``(mean, var)`` of the equivalent Gaussian site.
    """
    var = jnp.reciprocal(site_nat2)
    mean = site_nat1 * var
    return mean, var


def cavity_from_marginal(
    marg_mean: Float[Array, " *batch"],
    marg_var: Float[Array, " *batch"],
    site_nat1: Float[Array, " *batch"],
    site_nat2: Float[Array, " *batch"],
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
    """Compute cavity distribution by removing a site from the marginal.

    Args:
        marg_mean: Marginal distribution means.
        marg_var: Marginal distribution variances (positive).
        site_nat1: Site precision-weighted means to remove.
        site_nat2: Site precisions to remove.

    Returns:
        Tuple ``(cav_mean, cav_var)`` of the cavity distribution.
    """
    cav_prec = jnp.reciprocal(marg_var) - site_nat2
    cav_var = jnp.reciprocal(cav_prec)
    cav_mean = (marg_mean * jnp.reciprocal(marg_var) - site_nat1) * cav_var
    return cav_mean, cav_var
