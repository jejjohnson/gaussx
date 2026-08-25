"""Matern SDE kernels (orders 0, 1, 2)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


class MaternSDE(SDEKernel):
    r"""State-space representation of the Matern kernel.

    Supports orders 0 (Matern-1/2), 1 (Matern-3/2), and 2 (Matern-5/2).
    The state dimension is ``order + 1``.

    Attributes:
        variance: Signal variance $\sigma^2$.
        lengthscale: Lengthscale $\ell$.
        order: Matern order (0, 1, or 2).
    """

    variance: Float[Array, ""]
    lengthscale: Float[Array, ""]
    order: int = eqx.field(static=True)

    @property
    def state_dim(self) -> int:
        return self.order + 1

    def sde_params(self) -> SDEParams:
        """Compute SDE parameters for the Matern kernel."""
        if self.order == 0:
            return self._matern12()
        elif self.order == 1:
            return self._matern32()
        elif self.order == 2:
            return self._matern52()
        else:
            msg = f"Unsupported Matern order {self.order}; must be 0, 1, or 2"
            raise ValueError(msg)

    def _matern12(self) -> SDEParams:
        # Constant blocks follow the hyperparameter dtype; an untyped
        # ``jnp.array`` of Python floats is float64 under x64 (gh-224).
        dtype = jnp.result_type(self.variance, self.lengthscale)
        lam = 1.0 / self.lengthscale
        F = jnp.array([[-lam]])
        L = jnp.array([[1.0]], dtype=dtype)
        H = jnp.array([[1.0]], dtype=dtype)
        Q_c = jnp.array([[2.0 * lam * self.variance]])
        P_inf = jnp.array([[self.variance]])
        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)

    def _matern32(self) -> SDEParams:
        dtype = jnp.result_type(self.variance, self.lengthscale)
        lam = jnp.sqrt(3.0) / self.lengthscale
        F = jnp.array([[0.0, 1.0], [-(lam**2), -2.0 * lam]])
        L = jnp.array([[0.0], [1.0]], dtype=dtype)
        H = jnp.array([[1.0, 0.0]], dtype=dtype)
        q = 4.0 * lam**3 * self.variance
        Q_c = jnp.array([[q]])
        P_inf = jnp.array([[self.variance, 0.0], [0.0, lam**2 * self.variance]])
        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)

    def _matern52(self) -> SDEParams:
        dtype = jnp.result_type(self.variance, self.lengthscale)
        lam = jnp.sqrt(5.0) / self.lengthscale
        F = jnp.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-(lam**3), -3.0 * lam**2, -3.0 * lam],
            ]
        )
        L = jnp.array([[0.0], [0.0], [1.0]], dtype=dtype)
        H = jnp.array([[1.0, 0.0, 0.0]], dtype=dtype)
        kappa = 5.0 / 3.0 * self.variance / self.lengthscale**2
        q = 16.0 / 3.0 * lam**5 * self.variance
        Q_c = jnp.array([[q]])
        P_inf = jnp.array(
            [
                [self.variance, 0.0, -kappa],
                [0.0, kappa, 0.0],
                [-kappa, 0.0, lam**4 * self.variance],
            ]
        )
        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)
