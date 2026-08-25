"""Constant (bias) SDE kernel."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


class ConstantSDE(SDEKernel):
    r"""State-space representation of a constant kernel.

    Models $k(\tau) = \sigma^2$ — a degenerate kernel with zero
    dynamics and zero diffusion. State dimension is 1.

    Attributes:
        variance: Signal variance $\sigma^2$.
    """

    variance: Float[Array, ""]

    @property
    def state_dim(self) -> int:
        return 1

    def sde_params(self) -> SDEParams:
        """Return SDE parameters for the constant kernel."""
        # Constant blocks follow the hyperparameter dtype; untyped
        # ``jnp.zeros``/``jnp.array`` are float64 under x64 (gh-224).
        dtype = jnp.result_type(self.variance)
        F = jnp.zeros((1, 1), dtype=dtype)
        L = jnp.zeros((1, 1), dtype=dtype)
        H = jnp.array([[1.0]], dtype=dtype)
        Q_c = jnp.zeros((1, 1), dtype=dtype)
        P_inf = jnp.array([[self.variance]])
        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        """Closed-form: A = I, Q = 0 (no dynamics)."""
        dtype = jnp.result_type(self.variance)
        A = jnp.eye(1, dtype=dtype)
        Q = jnp.zeros((1, 1), dtype=dtype)
        return A, Q
