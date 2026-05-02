"""Discrete Lyapunov equation solver via per-factor eigendecomposition.

Solves ``P - G P G^T = Q`` for symmetric ``P`` without materializing
the ``(N², N²)`` Kronecker matrix ``I - kron(G, G)``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def discrete_lyapunov_solve(
    G: Float[Array, "N N"],
    Q: Float[Array, "N N"],
) -> Float[Array, "N N"]:
    r"""Solve the discrete Lyapunov equation ``P - G P G^T = Q``.

    Uses the eigendecomposition ``G = V Λ V^{-1}`` so that

    .. math::

        \tilde{P} - \Lambda \tilde{P} \Lambda^T
            = V^{-1} Q V^{-T}, \quad
        \tilde{P}_{ij} = \frac{(V^{-1} Q V^{-T})_{ij}}{1 - \lambda_i \lambda_j},

    and then ``P = V \tilde{P} V^T``. This avoids materializing the
    ``(N², N²)`` Kronecker matrix the vectorized formulation requires.

    Cost is ``O(N³)`` (one general eigendecomposition + a couple of
    matrix multiplies) versus ``O(N⁶)`` for the vectorized solve, and
    the memory footprint drops from ``O(N⁴)`` to ``O(N²)``.

    The discrete Lyapunov equation has a unique solution iff
    ``λ_i λ_j ≠ 1`` for all eigenvalue pairs ``(λ_i, λ_j)`` of ``G``.
    The standard sufficient condition — ``G`` is stable
    (spectral radius < 1) — guarantees this.

    Args:
        G: Square matrix, shape ``(N, N)``. Should be stable for a
            unique steady-state solution.
        Q: Right-hand side, shape ``(N, N)``. Typically symmetric PSD
            in Kalman-smoother applications.

    Returns:
        Symmetric matrix ``P`` of shape ``(N, N)`` solving
        ``P - G P G^T = Q``.
    """
    eigs, V = jnp.linalg.eig(G)
    Vinv = jnp.linalg.inv(V)
    # Q in the eigenbasis: V^{-1} Q V^{-T}
    Q_tilde = Vinv @ Q.astype(V.dtype) @ Vinv.T
    denom = 1.0 - eigs[:, None] * eigs[None, :]
    P_tilde = Q_tilde / denom
    P = V @ P_tilde @ V.T
    return jnp.real(P)
