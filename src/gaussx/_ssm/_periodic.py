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
        # Constant blocks follow the hyperparameter dtype; untyped
        # ``jnp.zeros``/``jnp.eye`` are float64 under x64 (gh-224).
        dtype = jnp.result_type(self.variance, self.frequency)
        w = self.frequency
        F = jnp.array([[0.0, -w], [w, 0.0]])
        L = jnp.zeros((2, 1), dtype=dtype)
        H = jnp.array([[1.0, 0.0]], dtype=dtype)
        Q_c = jnp.zeros((1, 1), dtype=dtype)
        P_inf = self.variance * jnp.eye(2, dtype=dtype)
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
        Q = jnp.zeros((2, 2), dtype=A.dtype)
        return A, Q


class PeriodicSDE(SDEKernel):
    r"""State-space representation of the periodic (MacKay) kernel.

    Approximates the periodic kernel via Fourier series truncation
    to ``n_harmonics`` terms. State dimension is ``2 * n_harmonics``.

    Short lengthscales put more of the variance into high harmonics, so
    they need more terms: use ``n_harmonics ≳ 3 / ℓ`` (with ``ℓ`` in units
    of the period). At ``ℓ = 0.2`` the default 6 harmonics carry about 81%
    of the variance, and 20 carry 99.99%.

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
        dtype = jnp.result_type(self.variance, self.lengthscale, self.period)
        J = self.n_harmonics
        d = 2 * J
        w0 = 2.0 * jnp.pi / self.period

        inv_ell_sq = 1.0 / self.lengthscale**2
        # q_j = 2 σ² I_j(x) e^{-x}, from the scaled Bessel values directly.
        q_j = 2.0 * self.variance * _scaled_bessel_i(J, inv_ell_sq)[1:]

        F = jnp.zeros((d, d), dtype=dtype)
        P_inf = jnp.zeros((d, d), dtype=dtype)
        for j_idx in range(J):
            freq = (j_idx + 1) * w0
            block_start = 2 * j_idx
            F = F.at[block_start, block_start + 1].set(-freq)
            F = F.at[block_start + 1, block_start].set(freq)
            P_inf = P_inf.at[block_start, block_start].set(q_j[j_idx])
            P_inf = P_inf.at[block_start + 1, block_start + 1].set(q_j[j_idx])

        L = jnp.zeros((d, 1), dtype=dtype)
        H = jnp.zeros((1, d), dtype=dtype)
        for j_idx in range(J):
            H = H.at[0, 2 * j_idx].set(1.0)

        Q_c = jnp.zeros((1, 1), dtype=dtype)
        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=P_inf)

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        """Closed-form: block-diagonal rotation matrices."""
        J = self.n_harmonics
        d = 2 * J
        w0 = 2.0 * jnp.pi / self.period

        A = jnp.zeros((d, d), dtype=jnp.result_type(w0, dt))
        for j_idx in range(J):
            freq = (j_idx + 1) * w0
            cos_val = jnp.cos(freq * dt)
            sin_val = jnp.sin(freq * dt)
            block_start = 2 * j_idx
            A = A.at[block_start, block_start].set(cos_val)
            A = A.at[block_start, block_start + 1].set(-sin_val)
            A = A.at[block_start + 1, block_start].set(sin_val)
            A = A.at[block_start + 1, block_start + 1].set(cos_val)

        Q = jnp.zeros((d, d), dtype=A.dtype)
        return A, Q


def _scaled_bessel_i(n_max: int, x: Float[Array, ""]) -> Float[Array, " n"]:
    r"""Exponentially scaled Bessel values $I_j(x) e^{-x}$ for $j = 0..n$.

    Two static-length recurrences, selected per ``x`` (gh-291):

    - ``x < x₀``: Miller's backward recurrence, carried as the ratios
      $r_k = I_k / I_{k-1} = 1 / (2k/x + r_{k+1})$ from order ``n + 40``
      and anchored at ``i0e(x)``. Ratios cannot overflow for small ``x``.
    - ``x ≥ x₀``: the upward recurrence
      $\tilde I_{j+1} = \tilde I_{j-1} - (2j/x) \tilde I_j$ from ``i0e`` /
      ``i1e``, which is stable while ``j ≲ x``.

    With ``x₀ = max(12, 2n)`` this agrees with ``scipy.special.ive`` to
    1e-10 relative for ``x`` in ``[1e-2, 1e4]`` and ``n ≤ 30``. The branch
    not taken sees a clamped input, so gradients stay finite.
    """
    x0 = max(12.0, 2.0 * n_max)
    small = x < x0

    x_small = jnp.where(small, x, 1.0)
    ratio = jnp.zeros_like(x_small)
    ratios = []
    for k in range(n_max + 40, 0, -1):
        ratio = 1.0 / (2.0 * k / x_small + ratio)
        if k <= n_max:
            ratios.append(ratio)
    miller = [jss.i0e(x_small)]
    for r_k in reversed(ratios):
        miller.append(miller[-1] * r_k)

    x_large = jnp.where(small, x0, x)
    upward = [jss.i0e(x_large), jss.i1e(x_large)]
    for j in range(1, n_max):
        upward.append(upward[j - 1] - (2.0 * j / x_large) * upward[j])

    return jnp.where(small, jnp.stack(miller), jnp.stack(upward[: n_max + 1]))
