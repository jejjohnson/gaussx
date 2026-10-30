"""Gaussian likelihood sites for conjugate-computation variational inference.

CVI (Khan & Lin, 2017) approximates each non-conjugate likelihood term
p(y_t ∣ x_t) by a Gaussian site in natural form,

    t_site,t(x_t) ∝ exp(x_tᵀ Λμ_site,t − ½ x_tᵀ Λ_site,t x_t),

so the approximate posterior q(x_1:T; ν) ∝ p(x_1:T) ∏_t t_site,t(x_t) is a
Gauss-Markov chain. The variational parameters are the site natural
parameters ν_t = (Λμ_site,t, −½ Λ_site,t), in the convention of
`gaussx.mean_cov_to_natural`; Λμ_site,t is one symbol because the site
precision Λ_site,t is singular whenever a site sees fewer than D_x
directions.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._operators._block_tridiag import BlockTriDiag


class GaussianSites(eqx.Module):
    """Time-varying Gaussian likelihood sites in natural parameterization.

        ν_t = (Λμ_site,t, −½ Λ_site,t),   t = 1, …, T

    One site per time step on a state of length D_x, in the convention of
    `gaussx.mean_cov_to_natural`.

    Attributes:
        nat1: First natural parameters Λμ_site,t, the precision-weighted
            site means.
        nat2: Second natural parameters −½ Λ_site,t.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> identity = jnp.broadcast_to(jnp.eye(2), (4, 2, 2))
        >>> sites = gaussx.GaussianSites(nat1=jnp.zeros((4, 2)), nat2=-0.5 * identity)
        >>> sites.nat1.shape, sites.nat2.shape
        ((4, 2), (4, 2, 2))
    """

    nat1: Float[Array, "N d"]
    nat2: Float[Array, "N d d"]


def cvi_update_sites(
    sites: GaussianSites,
    grad_nat1: Float[Array, "N d"],
    grad_nat2: Float[Array, "N d d"],
    rho: float,
) -> GaussianSites:
    """Damped natural-gradient update of CVI sites.

        ν_t ← (1 − γ) ν_t + γ g_t

    applied to both natural parameters, with ν_t = (Λμ_site,t, −½ Λ_site,t)
    and g_t = (g_1,t, g_2,t) the natural-gradient targets in the same
    convention. γ = 1 replaces the sites by the targets; γ = 0 keeps them.

    Args:
        sites: Current sites ν_t.
        grad_nat1: Natural-gradient target g_1,t for Λμ_site,t.
        grad_nat2: Natural-gradient target g_2,t for −½ Λ_site,t.
        rho: Step size γ ∈ [0, 1].

    Returns:
        The updated `GaussianSites`.

    Examples:
        A full step replaces the sites by the targets:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> identity = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
        >>> sites = gaussx.GaussianSites(nat1=jnp.zeros((3, 2)), nat2=-0.5 * identity)
        >>> target1, target2 = jnp.ones((3, 2)), -identity
        >>> updated = gaussx.cvi_update_sites(sites, target1, target2, 1.0)
        >>> bool(jnp.allclose(updated.nat1, target1))
        True

        A half step lands midway:

        >>> half = gaussx.cvi_update_sites(sites, target1, target2, 0.5)
        >>> round(float(half.nat2[0, 0, 0]), 4)
        -0.75
    """
    # (1 − γ) Λμ_site,t
    kept_nat1 = (1.0 - rho) * sites.nat1
    # γ g_1,t
    step_nat1 = rho * grad_nat1
    # Λμ_site,t ← (1 − γ) Λμ_site,t + γ g_1,t
    new_nat1 = kept_nat1 + step_nat1
    # (1 − γ) (−½ Λ_site,t)
    kept_nat2 = (1.0 - rho) * sites.nat2
    # γ g_2,t
    step_nat2 = rho * grad_nat2
    # −½ Λ_site,t ← (1 − γ) (−½ Λ_site,t) + γ g_2,t
    new_nat2 = kept_nat2 + step_nat2
    return GaussianSites(nat1=new_nat1, nat2=new_nat2)


def sites_to_precision(sites: GaussianSites) -> BlockTriDiag:
    """Block-diagonal precision of the sites, as a `BlockTriDiag`.

        Λ_site = blockdiag(Λ_site,1, …, Λ_site,T),   Λ_site,t = −2 (−½ Λ_site,t)

    The result is a `BlockTriDiag` with zero sub-diagonal blocks, so it adds
    to a Gauss-Markov prior precision Λ_x with ``+`` to give the CVI
    posterior precision Λ_x + Λ_site without leaving the banded structure.

    Args:
        sites: Sites whose ``nat2`` holds −½ Λ_site,t.

    Returns:
        The block-diagonal site precision Λ_site.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> identity = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
        >>> sites = gaussx.GaussianSites(nat1=jnp.zeros((3, 2)), nat2=-0.5 * identity)
        >>> precision = gaussx.sites_to_precision(sites)
        >>> precision.as_matrix().shape
        (6, 6)
        >>> bool(jnp.allclose(precision.as_matrix(), jnp.eye(6)))
        True
    """
    num_steps, state_size = sites.nat1.shape
    # Λ_site,t = −2 (−½ Λ_site,t)
    diag_blocks = -2.0 * sites.nat2
    sub_diag_blocks = jnp.zeros(
        (num_steps - 1, state_size, state_size), dtype=diag_blocks.dtype
    )
    return BlockTriDiag(diag_blocks, sub_diag_blocks)
