"""Constant (bias) kernel k(τ) = σ_k² as a one-dimensional SDE."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


class ConstantSDE(SDEKernel):
    """State-space form of the constant kernel k(τ) = σ_k².

    A degenerate SDE with one state, no drift and no diffusion: the state
    keeps the value it starts with, drawn from the stationary variance σ_k².
    Discretised over any step, the transition is 1 and the model error 0.

    Attributes:
        variance: Kernel variance σ_k².

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> kernel = gaussx.ConstantSDE(variance=jnp.asarray(2.0))
        >>> kernel.state_dim, kernel.sde_params().P_inf.shape
        (1, (1, 1))
        >>> bool(jnp.allclose(kernel.sde_params().P_inf, 2.0))
        True

        Any step leaves the state unchanged:

        >>> transition, process_noise = kernel.discretise(jnp.asarray(0.5))
        >>> bool(jnp.allclose(transition, 1.0)), bool(jnp.allclose(process_noise, 0.0))
        (True, True)
    """

    variance: Float[Array, ""]

    @property
    def state_dim(self) -> int:
        """State length D_x = 1."""
        return 1

    def sde_params(self) -> SDEParams:
        """Continuous-time parameters: zero drift and diffusion, variance σ_k².

        Returns:
            `SDEParams` with ``F = 0``, ``L = 0``, ``H = 1``, ``Q_c = 0`` and
            the stationary covariance ``P_inf = σ_k²``.
        """
        # Constant blocks follow the hyperparameter dtype; untyped
        # ``jnp.zeros``/``jnp.array`` are float64 under x64 (gh-224).
        dtype = jnp.result_type(self.variance)
        # drift 0
        F = jnp.zeros((1, 1), dtype=dtype)
        # diffusion 0
        L = jnp.zeros((1, 1), dtype=dtype)
        # observation 1
        H = jnp.array([[1.0]], dtype=dtype)
        # spectral density 0
        Q_c = jnp.zeros((1, 1), dtype=dtype)
        # stationary covariance σ_k²
        P_inf = jnp.array([[self.variance]])
        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        """Closed-form discretisation: transition 1 and model error 0.

        Args:
            dt: Time step Δt; unused, since nothing changes over time.

        Returns:
            Tuple ``(A, Q)`` of the discrete transition and the model-error
            covariance Σ_η.
        """
        dtype = jnp.result_type(self.variance)
        # transition exp(0 · Δt) = 1
        A = jnp.eye(1, dtype=dtype)
        # Σ_η = 0
        Q = jnp.zeros((1, 1), dtype=dtype)
        return A, Q
