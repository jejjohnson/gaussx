"""Abstract SDE kernel base class and its `SDEParams` container.

A stationary kernel k(τ) is the measurement of a linear time-invariant SDE

    dx(t) = F_c x(t) dt + L_c db(t),    f(t) = H x(t),

with drift matrix F_c, diffusion (noise-effect) matrix L_c, white-noise
spectral density Q_c of the Brownian motion b(t) (db ~ 𝒩(0, Q_c dt)),
stationary covariance Σ_x,∞ and measurement row H. Over a step Δt_t the SDE
discretises to the transition F_t = exp(F_c Δt_t) and the model-noise
covariance Σ_η,t, the inputs of the Kalman recursions in ``_kalman.py``.
"""

from __future__ import annotations

import abc
from typing import NamedTuple

import equinox as eqx
import jax
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._linalg._symmetrize import symmetrize
from gaussx._ssm._discretise import discretize_mfd, process_noise_covariance


class SDEParams(NamedTuple):
    """Continuous-time matrices (F_c, L_c, H, Q_c, Σ_x,∞) of a linear SDE.

        dx(t) = F_c x(t) dt + L_c db(t),    db(t) ~ 𝒩(0, Q_c dt),
        f(t) = H x(t),
        F_c Σ_x,∞ + Σ_x,∞ F_cᵀ + L_c Q_c L_cᵀ = 0

    Attributes:
        F: Drift matrix F_c.
        L: Diffusion (noise-effect) matrix L_c.
        H: Measurement row H.
        Q_c: White-noise spectral density Q_c.
        P_inf: Stationary covariance Σ_x,∞, or ``None`` when the kernel
            has no closed form for it: a learned drift matrix, or a
            non-stationary kernel such as `gaussx.IntegratedWienerSDE`,
            which has no stationary covariance at all.
            `SDEKernel.discretise` then falls back to
            `gaussx.discretize_mfd`, which needs no Σ_x,∞, and the filter
            is started from `SDEKernel.initial_covariance` instead.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> kernel = gaussx.MaternSDE(
        ...     variance=jnp.asarray(1.0), lengthscale=jnp.asarray(1.0), order=1
        ... )
        >>> params = kernel.sde_params()
        >>> params.F.shape, params.L.shape, params.H.shape, params.Q_c.shape
        ((2, 2), (2, 1), (1, 2), (1, 1))

        The stationary covariance solves the Lyapunov equation:

        >>> lyapunov = (
        ...     params.F @ params.P_inf
        ...     + params.P_inf @ params.F.T
        ...     + params.L @ params.Q_c @ params.L.T
        ... )
        >>> bool(jnp.allclose(lyapunov, 0.0, atol=1e-4))
        True
    """

    F: Float[Array, "d d"]
    L: Float[Array, "d s"]
    H: Float[Array, "1 d"]
    Q_c: Float[Array, "s s"]
    P_inf: Float[Array, "d d"] | None = None


class SDEKernel(eqx.Module):
    """Abstract state-space form of a kernel k(τ): a linear SDE.

        k(τ) = H exp(F_c τ) Σ_x,∞ Hᵀ,    τ ≥ 0

    Subclasses implement `sde_params`, which returns (F_c, L_c, H, Q_c,
    Σ_x,∞). The default `discretise` takes the matrix exponential of the
    drift; subclasses may override it with a closed form.

    Examples:
        Every kernel in the zoo is an `SDEKernel`:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> kernel = gaussx.MaternSDE(
        ...     variance=jnp.asarray(1.0), lengthscale=jnp.asarray(0.5), order=2
        ... )
        >>> isinstance(kernel, gaussx.SDEKernel), kernel.state_dim
        (True, 3)

        Discretise over a stack of steps Δt_1:T:

        >>> transitions, noises = kernel.discretise_sequence(jnp.array([0.1, 0.2]))
        >>> transitions.shape, noises.shape
        ((2, 3, 3), (2, 3, 3))
    """

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """State length D_x."""
        ...

    @property
    def stationary(self) -> bool:
        """Whether the process has a stationary distribution.

        ``True`` for every kernel in the zoo, which is why that is the
        default; a non-stationary kernel such as
        `gaussx.IntegratedWienerSDE` overrides it. Consumers should
        branch on this rather than on ``sde_params().P_inf is None``:
        the two are different questions, since a stationary kernel may
        report ``P_inf=None`` when it has no *closed form* for Σ_x,∞ (a
        learned drift, say).
        """
        return True

    @abc.abstractmethod
    def sde_params(self) -> SDEParams:
        """Continuous-time parameters (F_c, L_c, H, Q_c, Σ_x,∞) of the SDE.

        Returns:
            The `SDEParams` of this kernel.
        """
        ...

    def initial_covariance(self) -> Float[Array, "d d"]:
        """Covariance Σ_x,0 of the state at the first time point.

            Σ_x,0 = Σ_x,∞

        A stationary kernel is assumed started in its stationary
        distribution, so the default returns Σ_x,∞ and existing kernels
        need no change. A non-stationary kernel has no such limit to start
        from and must override this with an explicit choice.

        Returns:
            Initial state covariance Σ_x,0.

        Raises:
            ValueError: If ``sde_params().P_inf`` is ``None``, so there
                is no stationary covariance to fall back on.
        """
        stationary_covariance = self.sde_params().P_inf
        if stationary_covariance is None:
            msg = (
                f"{type(self).__name__} has no initial covariance: it "
                f"reports P_inf=None, and the default initial covariance "
                f"is the stationary one. Override initial_covariance() "
                f"with an explicit choice (a diffuse prior, typically), "
                f"or give the kernel a P_inf."
            )
            raise ValueError(msg)
        return stationary_covariance

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        """Discretise the SDE over one step Δt.

            F_t = exp(F_c Δt)
            Σ_η,t = Σ_x,∞ − F_t Σ_x,∞ F_tᵀ

        When ``sde_params()`` returns ``P_inf=None`` (a kernel with no
        closed-form stationary covariance, such as one whose drift is a
        learned parameter) this falls back to `gaussx.discretize_mfd`,
        which recovers both F_t and Σ_η,t from one matrix exponential and
        is well defined for every F_c. The fallback is chosen at trace
        time from a static ``None`` check, so kernels that do supply
        Σ_x,∞ keep the stationary route and its precision exactly.
        Subclasses may override with closed-form expressions.

        Algorithm:
            1. Without Σ_x,∞, form B_c = L_c Q_c L_cᵀ and return
               `gaussx.discretize_mfd` (F_c, B_c, Δt).
            2. Otherwise exponentiate the drift, F_t = exp(F_c Δt).
            3. Take Σ_η,t = Σ_x,∞ − F_t Σ_x,∞ F_tᵀ and symmetrise it.

        Args:
            dt: Time step Δt ≥ 0.

        Returns:
            Tuple ``(A, Q)`` holding the transition F_t and the
            model-noise covariance Σ_η,t.
        """
        params = self.sde_params()
        if params.P_inf is None:
            # L_c Q_c
            noise_effect_density = einsum(
                params.L, params.Q_c, "state noise, noise col -> state col"
            )
            # B_c = L_c Q_c L_cᵀ
            diffusion_covariance = einsum(
                noise_effect_density, params.L, "state col, other col -> state other"
            )
            return discretize_mfd(params.F, diffusion_covariance, dt)
        # F_t = exp(F_c Δt)
        transition = jsl.expm(params.F * dt)
        # Σ_η,t = Σ_x,∞ − F_t Σ_x,∞ F_tᵀ
        model_noise_covariance = process_noise_covariance(transition, params.P_inf)
        model_noise_covariance = symmetrize(model_noise_covariance)
        return transition, model_noise_covariance

    def discretise_sequence(
        self,
        dt: Float[Array, " N"],
    ) -> tuple[Float[Array, "N d d"], Float[Array, "N d d"]]:
        """Discretise the SDE over a stack of steps Δt_1:T, one per step.

            F_t = exp(F_c Δt_t),    Σ_η,t,    t = 1, …, T

        Args:
            dt: Time steps Δt_1:T.

        Returns:
            Tuple ``(A_seq, Q_seq)`` stacking F_t and Σ_η,t over time.
        """
        return jax.vmap(self.discretise)(dt)
