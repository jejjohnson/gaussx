"""Ψ statistics of a kernel: the analytical protocol and its numerical fallback."""

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

        Ψ₀ = 𝔼[k(x, x)]
        Ψ₁ᵢ = 𝔼[k(x, xᵢ)]
        Ψ₂ᵢⱼ = 𝔼[k(x, xᵢ) k(x, xⱼ)],    x ~ 𝒩(μ_x, Σ_x)

    Ψ statistics are required for uncertain-input GP models (e.g., BGPLVM). A
    kernel implementing this protocol provides analytical formulae instead of
    requiring numerical integration.

    Examples:
        The check is structural, so a class needs all three methods:

        >>> import jax.numpy as jnp
        >>> class ConstantKernel:
        ...     def psi0(self, state):
        ...         return jnp.ones(())
        ...
        ...     def psi1(self, state, X_train):
        ...         return jnp.ones(X_train.shape[0])
        ...
        ...     def psi2(self, state, X_train):
        ...         return jnp.ones((X_train.shape[0], X_train.shape[0]))
        >>> isinstance(ConstantKernel(), AnalyticalPsiStatistics)
        True
        >>> isinstance(lambda x, x_prime: 1.0, AnalyticalPsiStatistics)
        False
    """

    def psi0(self, state: GaussianState) -> Float[Array, ""]:
        """Compute Ψ₀ = 𝔼[k(x, x)].

        Args:
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            The scalar statistic Ψ₀.
        """
        ...

    def psi1(
        self,
        state: GaussianState,
        X_train: Float[Array, "M D"],
    ) -> Float[Array, " M"]:
        """Compute Ψ₁ᵢ = 𝔼[k(x, xᵢ)].

        Args:
            state: Input Gaussian 𝒩(μ_x, Σ_x).
            X_train: Training or inducing inputs, whose rows are the xᵢ.

        Returns:
            The vector statistic Ψ₁.
        """
        ...

    def psi2(
        self,
        state: GaussianState,
        X_train: Float[Array, "M D"],
    ) -> Float[Array, "M M"]:
        """Compute Ψ₂ᵢⱼ = 𝔼[k(x, xᵢ) k(x, xⱼ)].

        Args:
            state: Input Gaussian 𝒩(μ_x, Σ_x).
            X_train: Training or inducing inputs, whose rows are the xᵢ.

        Returns:
            The matrix statistic Ψ₂.
        """
        ...


def compute_psi_statistics(
    kernel: object,
    state: GaussianState,
    X_train: Float[Array, "M D"],
    *,
    integrator: AbstractIntegrator | None = None,
) -> tuple[Float[Array, ""], Float[Array, " M"], Float[Array, "M M"]]:
    """Compute Ψ statistics, dispatching to analytical or numerical.

        Ψ₀ = 𝔼[k(x, x)]
        Ψ₁ᵢ = 𝔼[k(x, xᵢ)]
        Ψ₂ᵢⱼ = 𝔼[k(x, xᵢ) k(x, xⱼ)],    x ~ 𝒩(μ_x, Σ_x)

    If ``kernel`` implements `AnalyticalPsiStatistics`, uses the closed-form
    methods and ignores ``integrator``. Otherwise, falls back to numerical
    integration via the provided integrator, with ``kernel`` called as k(x, x′).

    Args:
        kernel: Kernel object, optionally implementing
            `AnalyticalPsiStatistics`.
        state: Input Gaussian 𝒩(μ_x, Σ_x).
        X_train: Training or inducing inputs, whose rows are the xᵢ.
        integrator: Numerical integrator for fallback. Required if ``kernel``
            does not implement analytical Ψ statistics.

    Returns:
        Tuple ``(Ψ₀, Ψ₁, Ψ₂)`` of Ψ statistics.

    Raises:
        ValueError: If ``kernel`` has no analytical Ψ statistics and no
            integrator is provided.

    Examples:
        A plain callable falls back to the integrator:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> X_train = jnp.zeros((3, 2))
        >>> kernel = lambda x, x_prime: jnp.ones(())
        >>> rule = gaussx.CubatureIntegrator()
        >>> _, psi_1, psi_2 = compute_psi_statistics(
        ...     kernel, state, X_train, integrator=rule
        ... )
        >>> psi_1.shape, psi_2.shape
        ((3,), (3, 3))

        And fails without one:

        >>> compute_psi_statistics(kernel, state, X_train)
        ... # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: Kernel does not implement AnalyticalPsiStatistics and no
        integrator was provided. Either implement the protocol on the kernel or
        pass an integrator for numerical computation.
    """
    if isinstance(kernel, AnalyticalPsiStatistics):
        # Ψ₀ = 𝔼[k(x, x)]
        psi0 = kernel.psi0(state)
        # Ψ₁ᵢ = 𝔼[k(x, xᵢ)]
        psi1 = kernel.psi1(state, X_train)
        # Ψ₂ᵢⱼ = 𝔼[k(x, xᵢ) k(x, xⱼ)]
        psi2 = kernel.psi2(state, X_train)
        return psi0, psi1, psi2

    if integrator is None:
        msg = (
            "Kernel does not implement AnalyticalPsiStatistics and no "
            "integrator was provided. Either implement the protocol on "
            "the kernel or pass an integrator for numerical computation."
        )
        raise ValueError(msg)

    # --- numerical fallback: one implementation, `kernel_expectations` ---
    kernel_fn = cast(Callable, kernel)
    return kernel_expectations(kernel_fn, state, X_train, integrator)
