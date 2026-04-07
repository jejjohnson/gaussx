"""Analytical Psi statistics protocol and dispatch."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp

from gaussx._uncertain._integrator import AbstractIntegrator
from gaussx._uncertain._types import GaussianState


@runtime_checkable
class AnalyticalPsiStatistics(Protocol):
    """Protocol for kernels with closed-form Psi statistics.

    Psi statistics are required for uncertain-input GP models
    (e.g., BGPLVM). A kernel implementing this protocol provides
    analytical formulae instead of requiring numerical integration.
    """

    def psi0(self, state: GaussianState) -> jnp.ndarray:
        """Compute Psi0 statistic: ``E[k(x, x)]``.

        Args:
            state: Input Gaussian distribution.

        Returns:
            Scalar or shape ``(N,)`` expected self-kernel values.
        """
        ...

    def psi1(
        self,
        state: GaussianState,
        X_train: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Psi1 statistic: ``E[k(x, X_train)]``.

        Args:
            state: Input Gaussian distribution.
            X_train: Training/inducing points, shape ``(M, D)``.

        Returns:
            Shape ``(N, M)`` expected cross-kernel values.
        """
        ...

    def psi2(
        self,
        state: GaussianState,
        X_train: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Psi2 statistic: ``E[k(x, X_i) k(x, X_j)]``.

        Args:
            state: Input Gaussian distribution.
            X_train: Training/inducing points, shape ``(M, D)``.

        Returns:
            Shape ``(M, M)`` or ``(N, M, M)`` expected outer products.
        """
        ...


def compute_psi_statistics(
    kernel: object,
    state: GaussianState,
    X_train: jnp.ndarray,
    *,
    integrator: AbstractIntegrator | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Psi statistics, dispatching to analytical or numerical.

    If ``kernel`` implements :class:`AnalyticalPsiStatistics`, uses
    the closed-form methods. Otherwise, falls back to numerical
    integration via the provided integrator.

    Args:
        kernel: Kernel object, optionally implementing
            :class:`AnalyticalPsiStatistics`.
        state: Input Gaussian distribution.
        X_train: Training/inducing points, shape ``(M, D)``.
        integrator: Numerical integrator for fallback. Required if
            ``kernel`` does not implement analytical Psi statistics.

    Returns:
        Tuple ``(psi0, psi1, psi2)`` of Psi statistics.

    Raises:
        ValueError: If ``kernel`` has no analytical Psi statistics
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

    # Numerical fallback: integrate kernel evaluations over the input distribution
    # psi0: E[k(x, x)]
    def _k_self(x: jnp.ndarray) -> jnp.ndarray:
        return kernel(x, x)  # type: ignore[operator]

    psi0_result = integrator.integrate(_k_self, state)
    psi0 = psi0_result.state.mean

    # psi1: E[k(x, X_j)] for each inducing point j
    def _k_cross(x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda xj: kernel(x, xj))(X_train)  # type: ignore[operator]

    psi1_result = integrator.integrate(_k_cross, state)
    psi1 = psi1_result.state.mean

    # psi2: E[k(x, X_i) k(x, X_j)]
    def _k_outer(x: jnp.ndarray) -> jnp.ndarray:
        kx = jax.vmap(lambda xj: kernel(x, xj))(X_train)  # type: ignore[operator]
        return jnp.outer(kx, kx).ravel()

    psi2_result = integrator.integrate(_k_outer, state)
    M = X_train.shape[0]
    psi2 = psi2_result.state.mean.reshape(M, M)

    return psi0, psi1, psi2
