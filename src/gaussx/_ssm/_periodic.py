"""Cosine and periodic kernels k(τ) as linear SDEs of rotation blocks.

Each is dx(t) = F_c x(t) dt + L_c db(t), f(t) = H x(t), with drift matrix
F_c, diffusion (noise-effect) matrix L_c, white-noise spectral density Q_c,
stationary covariance Σ_x,∞ and measurement row H. Both are noise-free
(L_c = 0, Q_c = 0): the state rotates deterministically from a random start
drawn from Σ_x,∞. Discretised over a step Δt_t, the transition
F_t = exp(F_c Δt_t) is a rotation and the model-noise covariance Σ_η,t is 0.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jss
from jaxtyping import Array, Float

from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


class CosineSDE(SDEKernel):
    """State-space form of the cosine kernel k(τ) = σ_k² cos(ω₀ τ).

        F_c = [[0, −ω₀], [ω₀, 0]],  L_c = 0,  H = [1, 0],  Q_c = 0,
        Σ_x,∞ = σ_k² I₂

    A two-state rotation SDE, D_x = 2.

    Attributes:
        variance: Kernel variance σ_k².
        frequency: Angular frequency ω₀.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> kernel = gaussx.CosineSDE(
        ...     variance=jnp.asarray(2.0), frequency=jnp.asarray(3.0)
        ... )
        >>> lags = jnp.array([0.0, 0.4])
        >>> got = gaussx.sde_autocovariance(kernel, lags)
        >>> bool(jnp.allclose(got, 2.0 * jnp.cos(3.0 * lags), atol=1e-5))
        True

        A step rotates the state and adds no noise:

        >>> transition, noise = kernel.discretise(jnp.asarray(0.4))
        >>> bool(jnp.allclose(transition @ transition.T, jnp.eye(2), atol=1e-6))
        True
        >>> bool(jnp.all(noise == 0.0))
        True
    """

    variance: Float[Array, ""]
    frequency: Float[Array, ""]

    @property
    def state_dim(self) -> int:
        """State length D_x = 2."""
        return 2

    def sde_params(self) -> SDEParams:
        """Continuous-time parameters (F_c, L_c, H, Q_c, Σ_x,∞).

        Returns:
            `SDEParams` with F_c = [[0, −ω₀], [ω₀, 0]], L_c = 0, H = [1, 0],
            Q_c = 0 and Σ_x,∞ = σ_k² I₂.
        """
        # Constant blocks follow the hyperparameter dtype; untyped
        # ``jnp.zeros``/``jnp.eye`` are float64 under x64 (gh-224).
        dtype = jnp.result_type(self.variance, self.frequency)
        angular_frequency = self.frequency
        # F_c = [[0, −ω₀], [ω₀, 0]]
        drift = jnp.array([[0.0, -angular_frequency], [angular_frequency, 0.0]])
        # L_c = 0
        noise_effect = jnp.zeros((2, 1), dtype=dtype)
        # H = [1, 0]
        obs_operator = jnp.array([[1.0, 0.0]], dtype=dtype)
        # Q_c = 0
        spectral_density = jnp.zeros((1, 1), dtype=dtype)
        # Σ_x,∞ = σ_k² I₂
        stationary_covariance = self.variance * jnp.eye(2, dtype=dtype)
        return SDEParams(
            F=drift,
            L=noise_effect,
            H=obs_operator,
            Q_c=spectral_density,
            P_inf=stationary_covariance,
        )

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        """Closed-form discretisation: a rotation by ω₀ Δt and no noise.

            F_t = [[cos ω₀Δt, −sin ω₀Δt], [sin ω₀Δt, cos ω₀Δt]],  Σ_η,t = 0

        Args:
            dt: Time step Δt.

        Returns:
            Tuple ``(A, Q)`` holding the transition F_t and the
            model-noise covariance Σ_η,t = 0.
        """
        angular_frequency = self.frequency
        # cos ω₀Δt
        cos_step = jnp.cos(angular_frequency * dt)
        # sin ω₀Δt
        sin_step = jnp.sin(angular_frequency * dt)
        # F_t = [[cos ω₀Δt, −sin ω₀Δt], [sin ω₀Δt, cos ω₀Δt]]
        transition = jnp.array([[cos_step, -sin_step], [sin_step, cos_step]])
        # Σ_η,t = 0
        model_noise_covariance = jnp.zeros((2, 2), dtype=transition.dtype)
        return transition, model_noise_covariance


class PeriodicSDE(SDEKernel):
    """State-space form of the periodic (MacKay) kernel, as J + 1 harmonics.

        k(τ) = σ_k² exp(−2 sin²(ω₀τ/2) / ℓ²),    ω₀ = 2π / T_period
             ≈ ∑_{j=0}^{J} q_j cos(j ω₀ τ)
        q₀ = σ_k² e^{−ℓ⁻²} I₀(ℓ⁻²),    q_j = 2σ_k² e^{−ℓ⁻²} I_j(ℓ⁻²),  j ≥ 1

    with I_j the modified Bessel function of the first kind and J the
    number of harmonics (Solin & Särkkä, 2014). Each harmonic j = 0, …, J
    is a two-state rotation block at frequency j ω₀ with stationary
    covariance q_j I₂; the constant j = 0 block has zero drift. The state
    length is D_x = 2(J + 1).

    Short lengthscales put more of the variance into high harmonics, so
    they need more terms: use J ≳ 3/ℓ, with ℓ in units of the period. At
    ℓ = 0.2 the default 6 harmonics carry about 81% of the variance, and
    20 carry 99.99%.

    Attributes:
        variance: Kernel variance σ_k².
        lengthscale: Lengthscale ℓ.
        period: Period T_period.
        n_harmonics: Number of harmonics J (truncation order).

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> kernel = gaussx.PeriodicSDE(
        ...     variance=jnp.asarray(1.5),
        ...     lengthscale=jnp.asarray(1.0),
        ...     period=jnp.asarray(2.0),
        ...     n_harmonics=8,
        ... )
        >>> kernel.state_dim
        18

        The truncated series matches the kernel:

        >>> lags = jnp.array([0.0, 0.3, 1.1])
        >>> exact = 1.5 * jnp.exp(-2.0 * jnp.sin(jnp.pi * lags / 2.0) ** 2)
        >>> got = gaussx.sde_autocovariance(kernel, lags)
        >>> bool(jnp.allclose(got, exact, atol=1e-5))
        True

        One full period returns every block to its start:

        >>> transition, _ = kernel.discretise(jnp.asarray(2.0))
        >>> bool(jnp.allclose(transition, jnp.eye(18), atol=1e-5))
        True

    References:
        Solin, A. & Särkkä, S. (2014). Explicit link between periodic
        covariance functions and state space models. AISTATS.
    """

    variance: Float[Array, ""]
    lengthscale: Float[Array, ""]
    period: Float[Array, ""]
    n_harmonics: int = eqx.field(static=True, default=6)

    @property
    def state_dim(self) -> int:
        """State length D_x = 2(J + 1)."""
        return 2 * (self.n_harmonics + 1)

    def sde_params(self) -> SDEParams:
        """Continuous-time parameters (F_c, L_c, H, Q_c, Σ_x,∞).

            F_c = blockdiag_j([[0, −jω₀], [jω₀, 0]]),  L_c = 0,  Q_c = 0
            H = [1, 0, 1, 0, …, 1, 0],  Σ_x,∞ = blockdiag_j(q_j I₂)

        Returns:
            The `SDEParams` of the J + 1 rotation blocks.
        """
        dtype = jnp.result_type(self.variance, self.lengthscale, self.period)
        n_blocks = self.n_harmonics + 1
        state_size = 2 * n_blocks
        # ω₀ = 2π / T_period
        base_frequency = 2.0 * jnp.pi / self.period

        # ℓ⁻²
        inv_lengthscale_sq = 1.0 / self.lengthscale**2
        # Dropping j = 0 left every covariance short by q₀ (gh-289).
        # weights 1 for j = 0, 2 for j ≥ 1
        weights = jnp.ones(n_blocks, dtype=dtype).at[1:].set(2.0)
        # e^{−ℓ⁻²} I_j(ℓ⁻²)
        scaled_bessel = _scaled_bessel_i(self.n_harmonics, inv_lengthscale_sq)
        # q_j = weight_j σ_k² e^{−ℓ⁻²} I_j(ℓ⁻²)
        harmonic_variances = self.variance * weights * scaled_bessel

        drift = jnp.zeros((state_size, state_size), dtype=dtype)
        stationary_covariance = jnp.zeros((state_size, state_size), dtype=dtype)
        for harmonic in range(n_blocks):
            # j ω₀
            frequency = harmonic * base_frequency
            block_start = 2 * harmonic
            # F_c block j = [[0, −jω₀], [jω₀, 0]]
            drift = drift.at[block_start, block_start + 1].set(-frequency)
            drift = drift.at[block_start + 1, block_start].set(frequency)
            # Σ_x,∞ block j = q_j I₂
            stationary_covariance = stationary_covariance.at[
                block_start, block_start
            ].set(harmonic_variances[harmonic])
            stationary_covariance = stationary_covariance.at[
                block_start + 1, block_start + 1
            ].set(harmonic_variances[harmonic])

        # L_c = 0
        noise_effect = jnp.zeros((state_size, 1), dtype=dtype)
        obs_operator = jnp.zeros((1, state_size), dtype=dtype)
        for harmonic in range(n_blocks):
            # H = [1, 0, 1, 0, …]: f(t) sums the first state of every block
            obs_operator = obs_operator.at[0, 2 * harmonic].set(1.0)

        # Q_c = 0
        spectral_density = jnp.zeros((1, 1), dtype=dtype)
        return SDEParams(
            F=drift,
            L=noise_effect,
            H=obs_operator,
            Q_c=spectral_density,
            P_inf=stationary_covariance,
        )

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        """Closed form: block-diagonal rotations, the identity for j = 0.

            F_t = blockdiag_j([[cos jω₀Δt, −sin jω₀Δt],
                               [sin jω₀Δt, cos jω₀Δt]]),    Σ_η,t = 0

        Args:
            dt: Time step Δt.

        Returns:
            Tuple ``(A, Q)`` holding the transition F_t and the
            model-noise covariance Σ_η,t = 0.
        """
        n_blocks = self.n_harmonics + 1
        state_size = 2 * n_blocks
        # ω₀ = 2π / T_period
        base_frequency = 2.0 * jnp.pi / self.period

        transition = jnp.zeros(
            (state_size, state_size), dtype=jnp.result_type(base_frequency, dt)
        )
        for harmonic in range(n_blocks):
            # j ω₀
            frequency = harmonic * base_frequency
            # cos jω₀Δt
            cos_step = jnp.cos(frequency * dt)
            # sin jω₀Δt
            sin_step = jnp.sin(frequency * dt)
            block_start = 2 * harmonic
            # F_t block j = [[cos jω₀Δt, −sin jω₀Δt], [sin jω₀Δt, cos jω₀Δt]]
            transition = transition.at[block_start, block_start].set(cos_step)
            transition = transition.at[block_start, block_start + 1].set(-sin_step)
            transition = transition.at[block_start + 1, block_start].set(sin_step)
            transition = transition.at[block_start + 1, block_start + 1].set(cos_step)

        # Σ_η,t = 0
        model_noise_covariance = jnp.zeros(
            (state_size, state_size), dtype=transition.dtype
        )
        return transition, model_noise_covariance


def _scaled_bessel_i(n_max: int, argument: Float[Array, ""]) -> Float[Array, " n"]:
    """Exponentially scaled Bessel values e^{−a} I_j(a) for j = 0, …, n_max.

        Ĩ_j(a) := e^{−a} I_j(a),    a = ``argument``

    Two static-length recurrences, selected per a (gh-291):

    - a < a₀: Miller's backward recurrence, carried as the ratios
      r_j = I_j / I_{j−1} = 1 / (2j/a + r_{j+1}) from order n_max + 40
      and anchored at Ĩ₀(a). Ratios cannot overflow for small a.
    - a ≥ a₀: the upward recurrence Ĩ_{j+1} = Ĩ_{j−1} − (2j/a) Ĩ_j from
      Ĩ₀ and Ĩ₁, which is stable while j ≲ a.

    With a₀ = max(12, 2 n_max) this agrees with ``scipy.special.ive`` to
    1e-10 relative for a in [1e-2, 1e4] and n_max ≤ 30. The branch not
    taken sees a clamped input, so gradients stay finite.

    Args:
        n_max: Highest order n_max.
        argument: Argument a > 0, ℓ⁻² for `PeriodicSDE`.

    Returns:
        Ĩ_0(a), …, Ĩ_{n_max}(a).
    """
    # a₀ = max(12, 2 n_max)
    switch_point = max(12.0, 2.0 * n_max)
    small = argument < switch_point

    # --- backward (Miller) recurrence, a < a₀ ---
    argument_small = jnp.where(small, argument, 1.0)
    ratio = jnp.zeros_like(argument_small)
    ratios = []
    for order in range(n_max + 40, 0, -1):
        # r_j = 1 / (2j/a + r_{j+1})
        ratio = 1.0 / (2.0 * order / argument_small + ratio)
        if order <= n_max:
            ratios.append(ratio)
    # Ĩ₀(a)
    miller = [jss.i0e(argument_small)]
    for ratio_order in reversed(ratios):
        # Ĩ_j = r_j Ĩ_{j−1}
        miller.append(miller[-1] * ratio_order)

    # --- upward recurrence, a ≥ a₀ ---
    argument_large = jnp.where(small, switch_point, argument)
    # Ĩ₀(a), Ĩ₁(a)
    upward = [jss.i0e(argument_large), jss.i1e(argument_large)]
    for order in range(1, n_max):
        # Ĩ_{j+1} = Ĩ_{j−1} − (2j/a) Ĩ_j
        upward.append(
            upward[order - 1] - (2.0 * order / argument_large) * upward[order]
        )

    return jnp.where(small, jnp.stack(miller), jnp.stack(upward[: n_max + 1]))
