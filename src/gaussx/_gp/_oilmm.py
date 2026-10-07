"""OILMM projection for multi-output Gaussian processes."""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import einsum


def oilmm_project(
    Y: Float[Array, "N P"],
    W: Float[Array, "P L"],
    noise_var: Float[Array, " P"] | float,
    *,
    check_orthonormal: bool = False,
    orthonormal_tol: float | None = None,
) -> tuple[Float[Array, "N L"], Float[Array, " L"]]:
    """Project multi-output data to independent latent GPs via OILMM.

    Given an orthonormal mixing matrix W ∈ ℝᴾˣᴸ with WᵀW = I_L, projects
    P-output observations to L independent latent channels:

        Y_latent    = Y W                      (N, L)
        σ²_latent   = diag(Wᵀ D W) = (W ⊙ W)ᵀ σ²   (L,)

    with ``D = diag(σ²)``. For isotropic noise ``Wᵀ D W = σ² I`` is exactly
    diagonal. For heteroscedastic noise ``Wᵀ D W`` has off-diagonal terms;
    only its diagonal is returned, which is the OILMM independence
    approximation (the latent channels are treated as independent).

    ``WᵀW = I`` is a precondition and is **not** checked by default: with a
    non-orthonormal ``W`` the projection ``Y W`` is not the least-squares
    one (that would be ``Y W (WᵀW)⁻¹``) and the latent noise is wrong.

    Args:
        Y: Observations, shape ``(N, P)``.
        W: Orthonormal mixing matrix, shape ``(P, L)`` with WᵀW = I_L.
        noise_var: Observation noise variance. Scalar for isotropic noise,
            or shape ``(P,)`` for heteroscedastic noise.
        check_orthonormal: If ``True``, raise (jit-safe, via
            `equinox.error_if`) when ``max |WᵀW − I| > orthonormal_tol``.
        orthonormal_tol: Tolerance for ``check_orthonormal``. Default
            ``sqrt(eps)`` of ``W``'s dtype.

    Returns:
        Tuple ``(Y_latent, noise_latent)`` with shapes ``(N, L)``
        and ``(L,)``.
    """
    if check_orthonormal:
        gram = einsum(W, W, "p i, p j -> i j")
        tol = (
            math.sqrt(float(jnp.finfo(W.dtype).eps))
            if orthonormal_tol is None
            else orthonormal_tol
        )
        err = jnp.max(jnp.abs(gram - jnp.eye(W.shape[1], dtype=W.dtype)))
        W = eqx.error_if(
            W, err > tol, "oilmm_project: W is not orthonormal (WᵀW != I)."
        )
    Y_latent = Y @ W  # (N, L)
    noise_var = jnp.broadcast_to(jnp.asarray(noise_var), (W.shape[0],))  # (P,)
    noise_latent = einsum(W**2, noise_var, "p l, p -> l")  # (L,)
    return Y_latent, noise_latent


def oilmm_back_project(
    f_means: Float[Array, "N L"],
    f_vars: Float[Array, "N L"],
    W: Float[Array, "P L"],
) -> tuple[Float[Array, "N P"], Float[Array, "N P"]]:
    """Back-project latent GP predictions to the observation space.

    Reconstructs observation-space predictions via:

        y_means = f_means Wᵀ              (N, P)
        y_vars  = f_vars (W ⊙ W)ᵀ        (N, P)

    Args:
        f_means: Latent predictive means, shape ``(N, L)``.
        f_vars: Latent predictive variances, shape ``(N, L)``.
        W: Orthogonal mixing matrix, shape ``(P, L)`` with WᵀW = I_L.

    Returns:
        Tuple ``(y_means, y_vars)`` with shapes ``(N, P)`` and ``(N, P)``.
    """
    y_means = f_means @ W.T  # (N, P)
    y_vars = f_vars @ (W**2).T  # (N, P)
    return y_means, y_vars
