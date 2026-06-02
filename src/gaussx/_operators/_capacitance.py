"""Capacitance-matrix solver (Sherman-Morrison / Woodbury) for masked domains.

The capacitance-matrix method extends a fast *base* solver ``B^{-1}`` (for which
an efficient inverse exists -- e.g. an FFT/DST/DCT Helmholtz solve on a
rectangle) to a problem with a small number of additional point constraints
(e.g. enforcing ``u = 0`` on the irregular boundary of a masked sub-domain). It
is a low-rank (Woodbury) correction around the base solve.

This module owns only the *generic linear algebra*. The caller supplies:

* ``base_solve`` -- a callable applying ``B^{-1}`` to a flat right-hand side;
* ``boundary_indices`` -- the flat indices of the constrained degrees of freedom.

Which degrees of freedom are constrained (mask / boundary extraction) and which
base solver to use (the spectral transform) stay in the calling package
(finitevolX / spectraldiffx).

Reference: Buzbee, Golub & Nielson (1970), "On Direct Methods for Solving
Poisson's Equations", SIAM J. Numer. Anal.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class CapacitanceSolver(eqx.Module):
    r"""Solve a base system subject to homogeneous point constraints.

    Given a fast base solver ``B^{-1}`` and a set of ``N_b`` constrained indices,
    this enforces ``u = 0`` at those indices via the capacitance-matrix
    correction:

    1. Base solve:        ``u = B^{-1} f``
    2. Sample boundary:   ``u_b = u[boundary]``
    3. Correction:        ``alpha = C^{-1} u_b``
    4. Subtract:          ``x = u - G^T alpha``

    where ``G[k] = B^{-1} e_{b_k}`` are the Green's functions of the base solver
    for unit sources at the constrained indices, and ``C[k, l] = G[l][b_k]`` is
    the capacitance matrix. ``C^{-1}`` and ``G`` are precomputed at construction.

    The solver operates on **flat** vectors. Any reshaping between fields and
    flat vectors, and any masking of the exterior, is the caller's
    responsibility -- keeping grid/mask concepts out of this class.

    Args:
        base_solve: Callable applying the base inverse ``B^{-1}`` to a flat
            right-hand side of length ``n``.
        boundary_indices: Flat indices of the constrained degrees of freedom,
            shape ``(N_b,)``.
        n: Length of the flat solution vector.

    Attributes:
        base_solve: The base inverse callable.
        boundary_indices: The constrained indices.
        green: Green's functions ``G``, shape ``(N_b, n)``.
        capacitance_inv: Inverse capacitance matrix ``C^{-1}``, shape
            ``(N_b, N_b)``.
    """

    base_solve: Callable[[Float[Array, " n"]], Float[Array, " n"]]
    boundary_indices: Int[Array, " Nb"]
    green: Float[Array, "Nb n"]
    capacitance_inv: Float[Array, "Nb Nb"]

    def __init__(
        self,
        base_solve: Callable[[Float[Array, " n"]], Float[Array, " n"]],
        boundary_indices: Int[Array, " Nb"],
        n: int,
    ):
        indices = jnp.asarray(boundary_indices)
        n_b = indices.shape[0]
        unit_sources = jnp.zeros((n_b, n)).at[jnp.arange(n_b), indices].set(1.0)
        green = jax.vmap(base_solve)(unit_sources)  # [Nb, n]
        capacitance = green[:, indices].T  # C[k, l] = green[l][b_k]

        self.base_solve = base_solve
        self.boundary_indices = indices
        self.green = green
        self.capacitance_inv = jnp.linalg.inv(capacitance)

    def __call__(self, rhs: Float[Array, " n"]) -> Float[Array, " n"]:
        """Solve the constrained system for a flat right-hand side ``rhs``."""
        u = self.base_solve(rhs)
        u_b = u[self.boundary_indices]
        alpha = self.capacitance_inv @ u_b
        correction = self.green.T @ alpha
        return u - correction
