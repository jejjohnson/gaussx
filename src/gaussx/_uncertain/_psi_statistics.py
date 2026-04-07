"""Analytical Psi statistics protocol and dispatch."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float

from gaussx._uncertain._integrator import AbstractIntegrator
from gaussx._uncertain._types import GaussianState


@runtime_checkable
class AnalyticalPsiStatistics(Protocol):
    """Protocol for kernels with closed-form Ψ statistics.

    Ψ statistics are required for uncertain-input GP models
    (e.g., BGPLVM). A kernel implementing this protocol provides
    analytical formulae instead of requiring numerical integration.
    """

    def psi0(self, state: GaussianState) -> jnp.ndarray:
        """Compute Ψ₀ = E[k(x, x)] (scalar)."""
        ...

    def psi1(
        self,
        state: GaussianState,
        X_train: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Ψ₁ᵢ = E[k(x, xᵢ)], shape ``(M,)``."""
        ...

    def psi2(
        self,
        state: GaussianState,
        X_train: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Ψ₂ᵢⱼ = E[k(x, xᵢ) k(x, xⱼ)], shape ``(M, M)``."""
        ...


def compute_psi_statistics(
    kernel: object,
    state: GaussianState,
    X_train: Float[Array, "M D"],
    *,
    integrator: AbstractIntegrator | None = None,
) -> tuple[Float[Array, ""], Float[Array, " M"], Float[Array, "M M"]]:
    """Compute Ψ statistics, dispatching to analytical or numerical.

    If ``kernel`` implements :class:`AnalyticalPsiStatistics`, uses
    the closed-form methods. Otherwise, falls back to numerical
    integration via the provided integrator::

        Ψ₀   = E[k(x, x)]                   scalar
        Ψ₁ᵢ  = E[k(x, xᵢ)]                 (M,)
        Ψ₂ᵢⱼ = E[k(x, xᵢ) k(x, xⱼ)]       (M, M)

    Args:
        kernel: Kernel object, optionally implementing
            :class:`AnalyticalPsiStatistics`.
        state: Input Gaussian distribution x ~ 𝒩(μ, Σ).
        X_train: Training/inducing points, shape ``(M, D)``.
        integrator: Numerical integrator for fallback. Required if
            ``kernel`` does not implement analytical Ψ statistics.

    Returns:
        Tuple ``(Ψ₀, Ψ₁, Ψ₂)`` of Psi statistics.

    Raises:
        ValueError: If ``kernel`` has no analytical Ψ statistics
            and no integrator is provided.
    """
    if isinstance(kernel, AnalyticalPsiStatistics):
        psi0 = kernel.psi0(state)
        psi1 = kernel.psi1(state, X_train)
        psi2 = kernel.psi2(state, X_train)
        return psi0, psi1, psi2

    if integrator is None:
        msg = (
            "Kernel does not implement AnalyticalPsiStatistics and no "
            "integrator was provided. Either implement the protocol on "
            "the kernel or pass an integrator for numerical computation."
        )
        raise ValueError(msg)

    import jax

    # ── Numerical fallback ────────────────────────────────────────

    # Ψ₀ = E[k(x, x)]
    def _k_self(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.atleast_1d(kernel(x, x))  # type: ignore[operator]

    psi0_result = integrator.integrate(_k_self, state)
    psi0 = psi0_result.state.mean[0]  # scalar

    # Ψ₁ᵢ = E[k(x, xᵢ)]
    def _k_cross(x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda xj: kernel(x, xj))(X_train)  # type: ignore[operator]

    psi1_result = integrator.integrate(_k_cross, state)
    psi1 = psi1_result.state.mean  # (M,)

    # Ψ₂ᵢⱼ = E[k(x, xᵢ) k(x, xⱼ)]
    M = X_train.shape[0]

    def _k_outer(x: jnp.ndarray) -> jnp.ndarray:
        kx = jax.vmap(lambda xj: kernel(x, xj))(X_train)  # type: ignore[operator]
        return rearrange(jnp.outer(kx, kx), "i j -> (i j)")  # (M²,)

    psi2_result = integrator.integrate(_k_outer, state)
    psi2 = rearrange(
        psi2_result.state.mean,
        "(i j) -> i j",
        i=M,
        j=M,
    )  # (M, M)

    return psi0, psi1, psi2
