"""Per-site natural parameters for expectation propagation on scalar Gaussians.

An EP site is a Gaussian factor in natural form, written with the site
precision Λ_site and the precision-weighted mean Λμ_site (one symbol, since
Λ_site may be zero or negative):

    t_site(x) ∝ exp(−½ Λ_site x² + Λμ_site x)

Every function works elementwise on a batch of scalar sites, or on the
diagonal of a mean-field Gaussian. Unlike `gaussx.mean_cov_to_natural`, the
second natural parameter here is the precision Λ_site itself, not −½ Λ_site.
Dense Gaussian conversions are in ``_expfam/_natural.py`` and the
block-tridiagonal ones of a state-space model in ``_ssm_natural.py``.
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
    """Site natural parameters that turn the cavity into the tilted moments.

        Λ_site = 1 / σ_tilt² − 1 / σ_cav²
        Λμ_site = μ_tilt / σ_tilt² − μ_cav / σ_cav²

    This is the EP site update: the site whose product with the cavity
    𝒩(μ_cav, σ_cav²) matches the tilted moments 𝒩(μ_tilt, σ_tilt²).

    Args:
        tilted_mean: Tilted means μ_tilt.
        tilted_var: Tilted variances σ_tilt² (positive).
        cav_mean: Cavity means μ_cav.
        cav_var: Cavity variances σ_cav² (positive).

    Returns:
        Tuple ``(site_nat1, site_nat2)`` holding Λμ_site and Λ_site.

    Examples:
        Matching moments give a flat site:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> mean, var = jnp.array([0.5, -1.0]), jnp.array([2.0, 0.5])
        >>> nat1, nat2 = gaussx.site_natural_from_tilted(mean, var, mean, var)
        >>> bool(jnp.allclose(nat1, 0.0)), bool(jnp.allclose(nat2, 0.0))
        (True, True)

        A tilted distribution tighter than the cavity gives a positive
        site precision:

        >>> nat1, nat2 = gaussx.site_natural_from_tilted(
        ...     jnp.array([1.0]), jnp.array([0.5]), jnp.array([0.0]), jnp.array([1.0])
        ... )
        >>> round(float(nat1[0]), 4), round(float(nat2[0]), 4)
        (2.0, 1.0)
    """
    # Each use takes its own reciprocal: sharing one changes the order in
    # which the gradient's cotangents are summed.
    # Λ_site = 1 / σ_tilt² − 1 / σ_cav²
    site_nat2 = jnp.reciprocal(tilted_var) - jnp.reciprocal(cav_var)
    # μ_tilt / σ_tilt²
    tilted_precision_mean = tilted_mean * jnp.reciprocal(tilted_var)
    # μ_cav / σ_cav²
    cavity_precision_mean = cav_mean * jnp.reciprocal(cav_var)
    # Λμ_site = μ_tilt / σ_tilt² − μ_cav / σ_cav²
    site_nat1 = tilted_precision_mean - cavity_precision_mean
    return site_nat1, site_nat2


def site_mean_var_from_natural(
    site_nat1: Float[Array, " *batch"],
    site_nat2: Float[Array, " *batch"],
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
    """Mean and variance of a site from its natural parameters.

        σ_site² = 1 / Λ_site,   μ_site = Λμ_site σ_site²

    Args:
        site_nat1: Site precision-weighted means Λμ_site.
        site_nat2: Site precisions Λ_site (positive for a proper Gaussian).

    Returns:
        Tuple ``(mean, var)`` holding μ_site and σ_site².

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> mean, var = gaussx.site_mean_var_from_natural(
        ...     jnp.array([2.0, -1.0]), jnp.array([1.0, 4.0])
        ... )
        >>> [round(float(v), 4) for v in mean], [round(float(v), 4) for v in var]
        ([2.0, -0.25], [1.0, 0.25])

        With a nearly flat cavity, the site carries the tilted moments:

        >>> nat1, nat2 = gaussx.site_natural_from_tilted(
        ...     jnp.array([0.3]), jnp.array([0.5]), jnp.array([0.0]), jnp.array([1e6])
        ... )
        >>> mean, var = gaussx.site_mean_var_from_natural(nat1, nat2)
        >>> bool(jnp.allclose(mean, 0.3, atol=1e-4))
        True
    """
    # σ_site² = 1 / Λ_site
    var = jnp.reciprocal(site_nat2)
    # μ_site = Λμ_site σ_site²
    mean = site_nat1 * var
    return mean, var


def cavity_from_marginal(
    marg_mean: Float[Array, " *batch"],
    marg_var: Float[Array, " *batch"],
    site_nat1: Float[Array, " *batch"],
    site_nat2: Float[Array, " *batch"],
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
    """Cavity distribution: the marginal with one site divided out.

        1 / σ_cav² = 1 / σ_marg² − Λ_site
        μ_cav = (μ_marg / σ_marg² − Λμ_site) σ_cav²

    Args:
        marg_mean: Marginal means μ_marg.
        marg_var: Marginal variances σ_marg² (positive).
        site_nat1: Precision-weighted means Λμ_site of the site to remove.
        site_nat2: Precisions Λ_site of the site to remove.

    Returns:
        Tuple ``(cav_mean, cav_var)`` holding μ_cav and σ_cav².

    Examples:
        Removing a flat site leaves the marginal:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> mean, var = jnp.array([0.5, -1.0]), jnp.array([2.0, 0.5])
        >>> cav_mean, cav_var = gaussx.cavity_from_marginal(
        ...     mean, var, jnp.zeros(2), jnp.zeros(2)
        ... )
        >>> bool(jnp.allclose(cav_mean, mean)), bool(jnp.allclose(cav_var, var))
        (True, True)

        It undoes `site_natural_from_tilted` when the marginal is the tilted
        distribution:

        >>> tilted = (jnp.array([1.0]), jnp.array([0.5]))
        >>> cavity = (jnp.array([0.0]), jnp.array([1.0]))
        >>> nat1, nat2 = gaussx.site_natural_from_tilted(*tilted, *cavity)
        >>> cav_mean, cav_var = gaussx.cavity_from_marginal(*tilted, nat1, nat2)
        >>> bool(jnp.allclose(cav_mean, 0.0, atol=1e-6))
        True
    """
    # Each use takes its own reciprocal, as in `site_natural_from_tilted`.
    # 1 / σ_cav² = 1 / σ_marg² − Λ_site
    cavity_precision = jnp.reciprocal(marg_var) - site_nat2
    # σ_cav² = (1 / σ_cav²)⁻¹
    cav_var = jnp.reciprocal(cavity_precision)
    # μ_marg / σ_marg²
    marginal_precision_mean = marg_mean * jnp.reciprocal(marg_var)
    # μ_cav / σ_cav² = μ_marg / σ_marg² − Λμ_site
    cavity_precision_mean = marginal_precision_mean - site_nat1
    # μ_cav = (μ_marg / σ_marg² − Λμ_site) σ_cav²
    cav_mean = cavity_precision_mean * cav_var
    return cav_mean, cav_var
