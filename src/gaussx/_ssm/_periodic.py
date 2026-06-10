"""Cosine and periodic SDE kernels."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jss
from jaxtyping import Array, Float

from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


class CosineSDE(SDEKernel):
    r"""State-space representation of the cosine kernel.

    Models $k(\tau) = \sigma^2 \cos(\omega_0 \tau)$ via a 2-D
    rotation SDE. State dimension is 2.

    Attributes:
        variance: Signal variance $\sigma^2$.
        frequency: Angular frequency $\omega_0$.
    """

    variance: Float[Array, ""]
    frequency: Float[Array, ""]

    @property
    def state_dim(self) -> int:
        return 2

    def sde_params(self) -> SDEParams:
        """Return SDE parameters for the cosine kernel."""
        w = self.frequency
        F = jnp.array([[0.0, -w], [w, 0.0]])
        L = jnp.zeros((2, 1))
        H = jnp.array([[1.0, 0.0]])
        Q_c = jnp.zeros((1, 1))
        P_inf = self.variance * jnp.eye(2)
        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        """Closed-form rotation matrix discretization."""
        w = self.frequency
        cos_wdt = jnp.cos(w * dt)
        sin_wdt = jnp.sin(w * dt)
        A = jnp.array([[cos_wdt, -sin_wdt], [sin_wdt, cos_wdt]])
        Q = jnp.zeros((2, 2))
        return A, Q


class PeriodicSDE(SDEKernel):
    r"""State-space representation of the periodic (MacKay) kernel.

    Approximates the periodic kernel via Fourier series truncation
    to ``n_harmonics`` terms. State dimension is ``2 * n_harmonics``.

    Attributes:
        variance: Signal variance $\sigma^2$.
        lengthscale: Lengthscale $\ell$.
        period: Period $T$.
        n_harmonics: Number of Fourier harmonics (truncation order).
    """

    variance: Float[Array, ""]
    lengthscale: Float[Array, ""]
    period: Float[Array, ""]
    n_harmonics: int = eqx.field(static=True, default=6)

    @property
    def state_dim(self) -> int:
        return 2 * self.n_harmonics

    def sde_params(self) -> SDEParams:
        """Return SDE parameters for the periodic kernel."""
        J = self.n_harmonics
        d = 2 * J
        w0 = 2.0 * jnp.pi / self.period

        inv_ell_sq = 1.0 / self.lengthscale**2
        js = jnp.arange(1, J + 1)
        log_ij = self._log_bessel_i(js, inv_ell_sq)
        log_q = jnp.log(2.0) + log_ij - inv_ell_sq
        q_j = self.variance * jnp.exp(log_q)

        F = jnp.zeros((d, d))
        P_inf = jnp.zeros((d, d))
        for j_idx in range(J):
            freq = (j_idx + 1) * w0
            block_start = 2 * j_idx
            F = F.at[block_start, block_start + 1].set(-freq)
            F = F.at[block_start + 1, block_start].set(freq)
            P_inf = P_inf.at[block_start, block_start].set(q_j[j_idx])
            P_inf = P_inf.at[block_start + 1, block_start + 1].set(q_j[j_idx])

        L = jnp.zeros((d, 1))
        H = jnp.zeros((1, d))
        for j_idx in range(J):
            H = H.at[0, 2 * j_idx].set(1.0)

        Q_c = jnp.zeros((1, 1))
        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        """Closed-form: block-diagonal rotation matrices."""
        J = self.n_harmonics
        d = 2 * J
        w0 = 2.0 * jnp.pi / self.period

        A = jnp.zeros((d, d))
        for j_idx in range(J):
            freq = (j_idx + 1) * w0
            cos_val = jnp.cos(freq * dt)
            sin_val = jnp.sin(freq * dt)
            block_start = 2 * j_idx
            A = A.at[block_start, block_start].set(cos_val)
            A = A.at[block_start, block_start + 1].set(-sin_val)
            A = A.at[block_start + 1, block_start].set(sin_val)
            A = A.at[block_start + 1, block_start + 1].set(cos_val)

        Q = jnp.zeros((d, d))
        return A, Q

    @staticmethod
    def _log_bessel_i(
        order: Float[Array, " J"],
        x: Float[Array, ""],
    ) -> Float[Array, " J"]:
        """Log of modified Bessel function I_n(x) via series."""
        half_x = x / 2.0
        log_half_x = jnp.log(half_x)

        log_leading = order * log_half_x - jss.gammaln(order + 1.0)

        x2_over_4 = x**2 / 4.0
        K = 20
        log_sum = jnp.zeros_like(order)
        log_term = jnp.zeros_like(order)
        for k in range(1, K + 1):
            log_term = (
                log_term
                + jnp.log(x2_over_4)
                - jnp.log(jnp.array(k, dtype=order.dtype))
                - jnp.log(order + k)
            )
            log_sum = jnp.logaddexp(log_sum, log_term)

        return log_leading + log_sum
