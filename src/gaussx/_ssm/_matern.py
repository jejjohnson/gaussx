"""Matérn-1/2, -3/2 and -5/2 kernels k(τ) as linear SDEs.

Each is dx(t) = F_c x(t) dt + L_c db(t), f(t) = H x(t), with drift matrix
F_c, diffusion (noise-effect) matrix L_c, white-noise spectral density Q_c,
stationary covariance Σ_x,∞ and measurement row H. Discretised over a step
Δt_t, the transition is F_t = exp(F_c Δt_t) and the model-noise covariance
Σ_η,t (see ``_sde_kernel.py``).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


class MaternSDE(SDEKernel):
    """State-space form of the Matérn kernel of order p, smoothness p + ½.

        k_Matérn-1/2(τ) = σ_k² exp(−λ|τ|),                       λ = 1/ℓ
        k_Matérn-3/2(τ) = σ_k² (1 + λ|τ|) exp(−λ|τ|),            λ = √3/ℓ
        k_Matérn-5/2(τ) = σ_k² (1 + λ|τ| + λ²τ²/3) exp(−λ|τ|),   λ = √5/ℓ

    The state x(t) = [f(t), f^(1)(t), …, f^(p)(t)] stacks f and its first p
    derivatives, so D_x = p + 1. The drift F_c is a companion matrix whose
    p + 1 eigenvalues all equal −λ, the noise enters the top derivative and
    Q_c = σ_k² (2λ)^(2p+1) (p!)² / (2p)!.

    Attributes:
        variance: Kernel variance σ_k².
        lengthscale: Lengthscale ℓ.
        order: Order p ∈ {0, 1, 2}, for Matérn-1/2, -3/2 and -5/2.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> kernel = gaussx.MaternSDE(
        ...     variance=jnp.asarray(2.0), lengthscale=jnp.asarray(0.5), order=1
        ... )
        >>> kernel.state_dim, kernel.sde_params().F.shape
        (2, (2, 2))

        The SDE reproduces the kernel, here at τ = 0 and τ = ℓ:

        >>> lags = jnp.array([0.0, 0.5])
        >>> rate = jnp.sqrt(3.0) / 0.5
        >>> expected = 2.0 * (1.0 + rate * lags) * jnp.exp(-rate * lags)
        >>> got = gaussx.sde_autocovariance(kernel, lags)
        >>> bool(jnp.allclose(got, expected, atol=1e-5))
        True

        Only orders 0, 1 and 2 have a state-space form here:

        >>> gaussx.MaternSDE(
        ...     variance=jnp.asarray(1.0), lengthscale=jnp.asarray(1.0), order=3
        ... ).sde_params()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Unsupported Matern order 3; must be 0, 1, or 2
    """

    variance: Float[Array, ""]
    lengthscale: Float[Array, ""]
    order: int = eqx.field(static=True)

    @property
    def state_dim(self) -> int:
        """State length D_x = p + 1."""
        return self.order + 1

    def sde_params(self) -> SDEParams:
        """Continuous-time parameters (F_c, L_c, H, Q_c, Σ_x,∞) for order p.

        Returns:
            The `SDEParams` of the Matérn-(p + ½) SDE.

        Raises:
            ValueError: If ``order`` is not 0, 1 or 2.
        """
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
        """Matérn-1/2 (Ornstein-Uhlenbeck) SDE, λ = 1/ℓ.

            F_c = −λ,  L_c = 1,  H = 1,  Q_c = 2λσ_k²,  Σ_x,∞ = σ_k²

        Returns:
            The `SDEParams` of the one-state SDE.
        """
        # Constant blocks follow the hyperparameter dtype; an untyped
        # ``jnp.array`` of Python floats is float64 under x64 (gh-224).
        dtype = jnp.result_type(self.variance, self.lengthscale)
        # λ = 1/ℓ
        rate = 1.0 / self.lengthscale
        # F_c = −λ
        drift = jnp.array([[-rate]])
        # L_c = 1
        noise_effect = jnp.array([[1.0]], dtype=dtype)
        # H = 1
        obs_operator = jnp.array([[1.0]], dtype=dtype)
        # Q_c = 2λσ_k²
        spectral_density = jnp.array([[2.0 * rate * self.variance]])
        # Σ_x,∞ = σ_k²
        stationary_covariance = jnp.array([[self.variance]])
        return SDEParams(
            F=drift,
            L=noise_effect,
            H=obs_operator,
            Q_c=spectral_density,
            P_inf=stationary_covariance,
        )

    def _matern32(self) -> SDEParams:
        """Matérn-3/2 SDE on x = [f, f^(1)], λ = √3/ℓ.

            F_c = [[0, 1], [−λ², −2λ]],  L_c = [0, 1]ᵀ,  H = [1, 0]
            Q_c = 4λ³σ_k²,  Σ_x,∞ = diag(σ_k², λ²σ_k²)

        Returns:
            The `SDEParams` of the two-state SDE.
        """
        dtype = jnp.result_type(self.variance, self.lengthscale)
        # λ = √3/ℓ
        rate = jnp.sqrt(3.0) / self.lengthscale
        # F_c = [[0, 1], [−λ², −2λ]]
        drift = jnp.array([[0.0, 1.0], [-(rate**2), -2.0 * rate]])
        # L_c = [0, 1]ᵀ
        noise_effect = jnp.array([[0.0], [1.0]], dtype=dtype)
        # H = [1, 0]
        obs_operator = jnp.array([[1.0, 0.0]], dtype=dtype)
        # q = 4λ³σ_k²
        density = 4.0 * rate**3 * self.variance
        # Q_c = q
        spectral_density = jnp.array([[density]])
        # Σ_x,∞ = diag(σ_k², λ²σ_k²)
        stationary_covariance = jnp.array(
            [[self.variance, 0.0], [0.0, rate**2 * self.variance]]
        )
        return SDEParams(
            F=drift,
            L=noise_effect,
            H=obs_operator,
            Q_c=spectral_density,
            P_inf=stationary_covariance,
        )

    def _matern52(self) -> SDEParams:
        """Matérn-5/2 SDE on x = [f, f^(1), f^(2)], λ = √5/ℓ.

            F_c = [[0, 1, 0], [0, 0, 1], [−λ³, −3λ², −3λ]]
            L_c = [0, 0, 1]ᵀ,  H = [1, 0, 0],  Q_c = 16λ⁵σ_k²/3
            Σ_x,∞ = [[σ_k², 0, −v₁], [0, v₁, 0], [−v₁, 0, λ⁴σ_k²]]

        with v₁ = 5σ_k²/(3ℓ²) the variance of f^(1).

        Returns:
            The `SDEParams` of the three-state SDE.
        """
        dtype = jnp.result_type(self.variance, self.lengthscale)
        # λ = √5/ℓ
        rate = jnp.sqrt(5.0) / self.lengthscale
        # F_c = [[0, 1, 0], [0, 0, 1], [−λ³, −3λ², −3λ]]
        drift = jnp.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-(rate**3), -3.0 * rate**2, -3.0 * rate],
            ]
        )
        # L_c = [0, 0, 1]ᵀ
        noise_effect = jnp.array([[0.0], [0.0], [1.0]], dtype=dtype)
        # H = [1, 0, 0]
        obs_operator = jnp.array([[1.0, 0.0, 0.0]], dtype=dtype)
        # v₁ = 5σ_k²/(3ℓ²), the variance of f^(1)
        derivative_variance = 5.0 / 3.0 * self.variance / self.lengthscale**2
        # q = 16λ⁵σ_k²/3
        density = 16.0 / 3.0 * rate**5 * self.variance
        # Q_c = q
        spectral_density = jnp.array([[density]])
        # Σ_x,∞ = [[σ_k², 0, −v₁], [0, v₁, 0], [−v₁, 0, λ⁴σ_k²]]
        stationary_covariance = jnp.array(
            [
                [self.variance, 0.0, -derivative_variance],
                [0.0, derivative_variance, 0.0],
                [-derivative_variance, 0.0, rate**4 * self.variance],
            ]
        )
        return SDEParams(
            F=drift,
            L=noise_effect,
            H=obs_operator,
            Q_c=spectral_density,
            P_inf=stationary_covariance,
        )
