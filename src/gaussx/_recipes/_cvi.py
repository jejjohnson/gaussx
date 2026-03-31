"""CVI — Conjugate-computation Variational Inference sites."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._operators._block_tridiag import BlockTriDiag


class GaussianSites(eqx.Module):
    r"""Time-varying Gaussian likelihood sites in natural parameterization.

    Stores per-timestep natural parameters for ``N`` Gaussian sites,
    following the ``\eta_2 = -\tfrac{1}{2}\Lambda`` convention
    (consistent with :func:`gaussx.expectation_to_natural`).

    Attributes:
        nat1: Natural location parameters, shape ``(N, d)``.
        nat2: Natural precision parameters, shape ``(N, d, d)``.
            Stores ``-\tfrac{1}{2}\Lambda_k`` at each time step.
    """

    nat1: Float[Array, "N d"]
    nat2: Float[Array, "N d d"]


def cvi_update_sites(
    sites: GaussianSites,
    grad_nat1: Float[Array, "N d"],
    grad_nat2: Float[Array, "N d d"],
    rho: float,
) -> GaussianSites:
    r"""Natural gradient update for CVI sites.

    Performs a damped update in natural parameter space::

        \theta \leftarrow (1 - \rho) \theta + \rho \nabla

    Args:
        sites: Current Gaussian sites.
        grad_nat1: Natural gradient for location, shape ``(N, d)``.
        grad_nat2: Natural gradient for precision, shape ``(N, d, d)``.
        rho: Step size / damping factor in ``[0, 1]``.

    Returns:
        Updated :class:`GaussianSites`.
    """
    new_nat1 = (1.0 - rho) * sites.nat1 + rho * grad_nat1
    new_nat2 = (1.0 - rho) * sites.nat2 + rho * grad_nat2
    return GaussianSites(nat1=new_nat1, nat2=new_nat2)


def sites_to_precision(sites: GaussianSites) -> BlockTriDiag:
    r"""Convert Gaussian sites to a block-tridiagonal precision.

    Returns a block-diagonal :class:`~gaussx.BlockTriDiag` (zero
    sub-diagonals) representing the precision contribution of the
    sites. This can be added to a prior precision via ``.add()``
    or ``+`` to form the posterior precision::

        \Lambda_{post} = \Lambda_{prior} + \Lambda_{sites}

    Since ``nat2`` stores ``-\tfrac{1}{2}\Lambda``, the precision
    blocks are ``-2 \cdot nat2``.

    Args:
        sites: Gaussian sites with ``nat2`` in eta2 convention.

    Returns:
        Block-diagonal :class:`~gaussx.BlockTriDiag` precision.
    """
    N, d = sites.nat1.shape
    diag_blocks = -2.0 * sites.nat2  # (N, d, d)
    sub_diag_blocks = jnp.zeros((N - 1, d, d), dtype=diag_blocks.dtype)
    return BlockTriDiag(diag_blocks, sub_diag_blocks)
