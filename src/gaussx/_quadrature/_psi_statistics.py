"""Analytical Psi statistics protocol and dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast, runtime_checkable

from jaxtyping import Array, Float

from gaussx._quadrature._gp_predict import kernel_expectations
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._types import GaussianState


@runtime_checkable
class AnalyticalPsiStatistics(Protocol):
    """Protocol for kernels with closed-form Ψ statistics.

    Ψ statistics are required for uncertain-input GP models
    (e.g., BGPLVM). A kernel implementing this protocol provides
    analytical formulae instead of requiring numerical integration.
    """

    def psi0(self, state: GaussianState) -> Float[Array, ""]:
        """Compute Ψ₀ = E[k(x, x)] (scalar)."""
        ...

    def psi1(
        self,
        state: GaussianState,
        X_train: Float[Array, "M D"],
    ) -> Float[Array, " M"]:
        """Compute Ψ₁ᵢ = E[k(x, xᵢ)], shape ``(M,)``."""
        ...

    def psi2(
        self,
        state: GaussianState,
        X_train: Float[Array, "M D"],
    ) -> Float[Array, "M M"]:
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

    If ``kernel`` implements `AnalyticalPsiStatistics`, uses
    the closed-form methods. Otherwise, falls back to numerical
    integration via the provided integrator:

        Ψ₀   = E[k(x, x)]                   scalar
        Ψ₁ᵢ  = E[k(x, xᵢ)]                 (M,)
        Ψ₂ᵢⱼ = E[k(x, xᵢ) k(x, xⱼ)]       (M, M)

    Args:
        kernel: Kernel object, optionally implementing
            `AnalyticalPsiStatistics`.
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

    # ── Numerical fallback: one implementation, `kernel_expectations` ──

    return kernel_expectations(cast(Callable, kernel), state, X_train, integrator)
