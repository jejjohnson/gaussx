"""Sums and products of SDE kernels, and the quasi-periodic product.

Every kernel is dx(t) = F_c x(t) dt + L_c db(t), f(t) = H x(t), with drift
matrix F_c, diffusion (noise-effect) matrix L_c, white-noise spectral
density Q_c, stationary covariance Σ_x,∞, measurement row H and diffusion
covariance B_c = L_c Q_c L_cᵀ. Component i of a composition carries the
superscript ^(i), as in F_c^(i). Discretised over a step Δt_t, the
transition is F_t = exp(F_c Δt_t) and the model-noise covariance Σ_η,t.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._kronecker import Kronecker
from gaussx._ssm._discretise import process_noise_covariance
from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


class SumSDE(SDEKernel):
    """Sum of independent SDE kernels, stacked block-diagonally.

        k(τ) = ∑_i k^(i)(τ)
        F_c = blockdiag_i(F_c^(i)),  L_c = blockdiag_i(L_c^(i)),
        Q_c = blockdiag_i(Q_c^(i)),  Σ_x,∞ = blockdiag_i(Σ_x,∞^(i)),
        H = [H^(1), …, H^(n)]

    Attributes:
        kernels: Component kernels k^(1), …, k^(n).

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> matern = gaussx.MaternSDE(
        ...     variance=jnp.asarray(1.0), lengthscale=jnp.asarray(0.5), order=1
        ... )
        >>> cosine = gaussx.CosineSDE(
        ...     variance=jnp.asarray(0.5), frequency=jnp.asarray(2.0)
        ... )
        >>> kernel = gaussx.SumSDE(kernels=(matern, cosine))
        >>> kernel.state_dim
        4

        The sum's kernel is the sum of the kernels:

        >>> lags = jnp.array([0.0, 0.7])
        >>> total = gaussx.sde_autocovariance(kernel, lags)
        >>> parts = gaussx.sde_autocovariance(
        ...     matern, lags
        ... ) + gaussx.sde_autocovariance(cosine, lags)
        >>> bool(jnp.allclose(total, parts, atol=1e-5))
        True

        A non-stationary component makes the sum non-stationary:

        >>> trend = gaussx.IntegratedWienerSDE(diffusion=jnp.asarray(1.0))
        >>> gaussx.SumSDE(kernels=(trend, matern)).stationary
        False
    """

    kernels: tuple[SDEKernel, ...] = eqx.field()

    @property
    def state_dim(self) -> int:
        """State length D_x = ∑_i D_x^(i)."""
        return sum(k.state_dim for k in self.kernels)

    @property
    def stationary(self) -> bool:
        """Stationary only if every component is."""
        return all(k.stationary for k in self.kernels)

    def initial_covariance(self) -> Float[Array, "d d"]:
        """Block-diagonal initial covariance Σ_x,0 = blockdiag_i(Σ_x,0^(i)).

        The components are independent, so their initial covariances
        stack block-diagonally, which lets a sum mix stationary and
        non-stationary components (a local linear trend plus a Matérn
        seasonal, say), each started from its own.

        Returns:
            Initial state covariance Σ_x,0.
        """
        return jsl.block_diag(*[k.initial_covariance() for k in self.kernels])

    def sde_params(self) -> SDEParams:
        """Block-diagonal continuous-time parameters of the sum.

            F_c = blockdiag_i(F_c^(i)),  L_c = blockdiag_i(L_c^(i)),
            Q_c = blockdiag_i(Q_c^(i)),  H = [H^(1), …, H^(n)],
            Σ_x,∞ = blockdiag_i(Σ_x,∞^(i)), or ``None`` if any is ``None``

        Returns:
            The `SDEParams` of the sum.
        """
        component_params = [k.sde_params() for k in self.kernels]

        # F_c = blockdiag_i(F_c^(i))
        drift = jsl.block_diag(*[p.F for p in component_params])
        # A component with no closed-form stationary covariance leaves the
        # sum without one either; propagating ``None`` routes the composite
        # through ``discretize_mfd`` rather than fabricating a Σ_x,∞.
        component_stationary = [p.P_inf for p in component_params]
        # Σ_x,∞ = blockdiag_i(Σ_x,∞^(i))
        stationary_covariance = (
            None
            if any(block is None for block in component_stationary)
            else jsl.block_diag(*component_stationary)
        )

        # ``block_diag`` takes rectangular blocks and keeps their dtype; an
        # untyped ``jnp.zeros`` accumulator would be float64 under x64.
        # L_c = blockdiag_i(L_c^(i))
        noise_effect = jsl.block_diag(*[p.L for p in component_params])

        # Q_c = blockdiag_i(Q_c^(i))
        spectral_density = jsl.block_diag(*[p.Q_c for p in component_params])
        # H = [H^(1), …, H^(n)]
        obs_operator = jnp.concatenate([p.H for p in component_params], axis=1)

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
        """Block-diagonal stack of each component's own discretisation.

            F_t = blockdiag_i(F_t^(i)),    Σ_η,t = blockdiag_i(Σ_η,t^(i))

        The components are independent, so F_t and Σ_η,t are
        block-diagonal. Each block keeps its component's exact route
        (closed form, stationary Σ_x,∞ − F_t Σ_x,∞ F_tᵀ, or MFD only where
        needed) instead of exponentiating the whole sum, which, with one
        non-stationary component, sent everything through
        ``discretize_mfd`` at ~27× the cost (gh-318).

        Args:
            dt: Time step Δt.

        Returns:
            Tuple ``(A, Q)`` holding the transition F_t and the
            model-noise covariance Σ_η,t.
        """
        component_steps = [k.discretise(dt) for k in self.kernels]
        # F_t = blockdiag_i(F_t^(i))
        transition = jsl.block_diag(*(a for a, _ in component_steps))
        # Σ_η,t = blockdiag_i(Σ_η,t^(i))
        model_noise_covariance = jsl.block_diag(*(q for _, q in component_steps))
        return transition, model_noise_covariance


class ProductSDE(SDEKernel):
    """Product of two SDE kernels, by Kronecker composition.

        k(τ) = k^(1)(τ) k^(2)(τ)
        F_c = F_c^(1) ⊕ F_c^(2) = F_c^(1) ⊗ I + I ⊗ F_c^(2)
        Σ_x,∞ = Σ_x,∞^(1) ⊗ Σ_x,∞^(2),    H = H^(1) ⊗ H^(2)

    Both factors need a stationary covariance; see `sde_params`.

    Attributes:
        kernel1: First factor k^(1).
        kernel2: Second factor k^(2).

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> matern = gaussx.MaternSDE(
        ...     variance=jnp.asarray(1.0), lengthscale=jnp.asarray(0.5), order=1
        ... )
        >>> cosine = gaussx.CosineSDE(
        ...     variance=jnp.asarray(2.0), frequency=jnp.asarray(3.0)
        ... )
        >>> kernel = gaussx.ProductSDE(kernel1=matern, kernel2=cosine)
        >>> kernel.state_dim
        4

        The product's kernel is the product of the kernels:

        >>> lags = jnp.array([0.0, 0.4])
        >>> got = gaussx.sde_autocovariance(kernel, lags)
        >>> expected = gaussx.sde_autocovariance(
        ...     matern, lags
        ... ) * gaussx.sde_autocovariance(cosine, lags)
        >>> bool(jnp.allclose(got, expected, atol=1e-5))
        True

        A factor without a stationary covariance cannot be composed:

        >>> trend = gaussx.IntegratedWienerSDE(diffusion=jnp.asarray(1.0))
        >>> gaussx.ProductSDE(
        ...     kernel1=trend, kernel2=cosine
        ... ).sde_params()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        NotImplementedError: ProductSDE cannot report SDE parameters
    """

    kernel1: SDEKernel
    kernel2: SDEKernel

    @property
    def state_dim(self) -> int:
        """State length D_x = D_x^(1) D_x^(2)."""
        return self.kernel1.state_dim * self.kernel2.state_dim

    @property
    def stationary(self) -> bool:
        """Stationary only if both factors are."""
        return self.kernel1.stationary and self.kernel2.stationary

    def sde_params(self) -> SDEParams:
        """Kronecker-structured continuous-time parameters of the product.

            F_c = F_c^(1) ⊗ I + I ⊗ F_c^(2),    Σ_x,∞ = Σ_x,∞^(1) ⊗ Σ_x,∞^(2)
            B_c = B_c^(1) ⊗ Σ_x,∞^(2) + Σ_x,∞^(1) ⊗ B_c^(2)
            L_c = [L_c^(1) ⊗ I, I ⊗ L_c^(2)]
            Q_c = blockdiag(Q_c^(1) ⊗ Σ_x,∞^(2), Σ_x,∞^(1) ⊗ Q_c^(2))

        The drift is the Kronecker **sum** and the stationary covariance
        the Kronecker **product**. Substituting those into the Lyapunov
        equation F_c Σ_x,∞ + Σ_x,∞ F_cᵀ + B_c = 0 fixes the composite
        diffusion covariance B_c above, which is **not** B_c^(1) ⊗ B_c^(2),
        the value a naive L_c^(1) ⊗ L_c^(2), Q_c^(1) ⊗ Q_c^(2) pair would
        imply. Reporting the latter used to hand out a tuple that failed
        its own Lyapunov equation (gh-219); for a Matérn ⊗ Cosine product
        it was identically zero, since `CosineSDE` has Q_c = 0.

        The sum of two Kronecker products is still expressible in the
        (L_c, Q_c) form above, by widening the noise dimension and carrying
        each factor's stationary covariance in the spectral density, so
        that L_c Q_c L_cᵀ telescopes to exactly B_c by the mixed-product
        property. Writing it this way rather than through square roots
        S Sᵀ = Σ_x,∞ keeps the result exact for singular or zero Σ_x,∞
        (where a Cholesky would need jitter, and would then violate the
        very Lyapunov equation this enforces) and keeps ``sde_params``
        reverse-mode differentiable.

        Note:
            ``SDEParams`` types its fields as dense
            ``jaxtyping.Float[Array, ...]``. The Kronecker products below
            are dense materialisations of size D_x × D_x with
            D_x = D_x^(1) D_x^(2); for typical SSM kernels (Matérn-3/2,
            periodic) this is ≤ 32, so the materialisation is bounded and
            cheap. A future refactor could expose a parallel
            ``sde_operators()`` method that returns `gaussx.Kronecker`
            operators for downstream filters that can exploit the
            structure (issue #153).

        Returns:
            The `SDEParams` of the product.

        Raises:
            NotImplementedError: If either factor lacks a stationary
                covariance. The composite diffusion needs both, so the
                tuple cannot be built, and reporting the inconsistent
                Kronecker product instead is what gh-219 was about.
        """
        params_1 = self.kernel1.sde_params()
        params_2 = self.kernel2.sde_params()

        state_size_1 = self.kernel1.state_dim
        state_size_2 = self.kernel2.state_dim

        if params_1.P_inf is None or params_2.P_inf is None:
            msg = (
                f"ProductSDE cannot report SDE parameters when a factor has "
                f"no stationary covariance: "
                f"{type(self.kernel1).__name__}.P_inf is "
                f"{'None' if params_1.P_inf is None else 'set'} and "
                f"{type(self.kernel2).__name__}.P_inf is "
                f"{'None' if params_2.P_inf is None else 'set'}. The composite "
                f"diffusion of a product kernel is B1 (x) P2 + P1 (x) B2, "
                f"which needs both. Use the factors' own parameters, or "
                f"give the factor a P_inf."
            )
            raise NotImplementedError(msg)

        # Identities carry the factor dtypes: an untyped ``jnp.eye`` is
        # float64 under x64 and would promote float32 kernels.
        identity_1 = jnp.eye(state_size_1, dtype=params_1.F.dtype)
        identity_2 = jnp.eye(state_size_2, dtype=params_2.F.dtype)

        # F_c = F_c^(1) ⊗ I + I ⊗ F_c^(2)
        drift = jnp.kron(params_1.F, identity_2) + jnp.kron(identity_1, params_2.F)
        # H = H^(1) ⊗ H^(2)
        obs_operator = jnp.kron(params_1.H, params_2.H)
        # Σ_x,∞ = Σ_x,∞^(1) ⊗ Σ_x,∞^(2)
        stationary_covariance = jnp.kron(params_1.P_inf, params_2.P_inf)

        # B_c = B_c^(1) ⊗ Σ_x,∞^(2) + Σ_x,∞^(1) ⊗ B_c^(2), kept in the
        # (L_c, Q_c) pair by putting each factor's Σ_x,∞ in the *spectral
        # density* rather than taking its square root:
        #
        #     B_c^(1) ⊗ Σ_x,∞^(2)
        #         = (L_c^(1) ⊗ I) (Q_c^(1) ⊗ Σ_x,∞^(2)) (L_c^(1) ⊗ I)ᵀ
        #     Σ_x,∞^(1) ⊗ B_c^(2)
        #         = (I ⊗ L_c^(2)) (Σ_x,∞^(1) ⊗ Q_c^(2)) (I ⊗ L_c^(2))ᵀ
        #
        # by the mixed-product property. Exact for any PSD Σ_x,∞, including
        # singular or zero ones, where a Cholesky would need jitter and stop
        # satisfying the Lyapunov equation this is here to enforce. The
        # noise dimension widens from D_b^(1) D_b^(2) to
        # D_b^(1) D_x^(2) + D_x^(1) D_b^(2).
        # L_c = [L_c^(1) ⊗ I, I ⊗ L_c^(2)]
        noise_effect = jnp.concatenate(
            [jnp.kron(params_1.L, identity_2), jnp.kron(identity_1, params_2.L)],
            axis=1,
        )
        # Q_c = blockdiag(Q_c^(1) ⊗ Σ_x,∞^(2), Σ_x,∞^(1) ⊗ Q_c^(2))
        spectral_density = jsl.block_diag(
            jnp.kron(params_1.Q_c, params_2.P_inf),
            jnp.kron(params_1.P_inf, params_2.Q_c),
        )

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
        """Discretise per factor, through the Kronecker exponential identity.

            F_t = exp(F_c Δt) = exp(F_c^(1) Δt) ⊗ exp(F_c^(2) Δt)
                = F_t^(1) ⊗ F_t^(2)
            Σ_η,t = Σ_x,∞ − (F_t^(1) Σ_x,∞^(1) F_t^(1)ᵀ)
                            ⊗ (F_t^(2) Σ_x,∞^(2) F_t^(2)ᵀ)

        The two terms of the Kronecker sum F_c^(1) ⊗ I and I ⊗ F_c^(2)
        commute, so the exponential factorises: two exponentials of size
        D_x^(1) and D_x^(2) plus one Kronecker product, instead of one of
        size D_x^(1) D_x^(2). By the mixed-product property the congruence
        F_t Σ_x,∞ F_tᵀ factorises too, so `gaussx.process_noise_covariance`
        evaluates it per factor on `gaussx.Kronecker` operands, in
        O((D_x^(1))³ + (D_x^(2))³) instead of O((D_x^(1) D_x^(2))³). Only
        the final Σ_η,t is materialised, to keep the consumer-facing
        ``(A, Q)`` interface unchanged.

        Algorithm:
            1. Exponentiate each factor's drift, F_t^(i) = exp(F_c^(i) Δt).
            2. Form F_t = F_t^(1) ⊗ F_t^(2).
            3. Take Σ_η,t = Σ_x,∞ − F_t Σ_x,∞ F_tᵀ on Kronecker operands,
               then materialise and symmetrise it.

        Args:
            dt: Time step Δt > 0.

        Returns:
            Tuple ``(A, Q)`` holding the transition F_t and the
            model-noise covariance Σ_η,t, as in `SDEKernel.discretise`.

        Raises:
            NotImplementedError: If either factor lacks a stationary
                covariance.
        """
        params_1 = self.kernel1.sde_params()
        params_2 = self.kernel2.sde_params()
        # F_t^(1) = exp(F_c^(1) Δt)
        transition_1 = jsl.expm(params_1.F * dt)
        # F_t^(2) = exp(F_c^(2) Δt)
        transition_2 = jsl.expm(params_2.F * dt)
        # F_t = F_t^(1) ⊗ F_t^(2)
        transition = jnp.kron(transition_1, transition_2)

        # Use the per-factor stationary covariances directly; building
        # the full F_c via ``self.sde_params()`` would defeat the
        # whole point of this override. Keeping both operands as
        # ``Kronecker`` lets the shared helper contract each factor
        # separately instead of forming the full-size triple product.
        if params_1.P_inf is None or params_2.P_inf is None:
            # Deferring to the base MFD path here would be *wrong*, not
            # merely slower: the composite diffusion covariance
            #
            #     B_c = B_c^(1) ⊗ Σ_x,∞^(2) + Σ_x,∞^(1) ⊗ B_c^(2)
            #
            # needs both factor stationary covariances, exactly what is
            # missing, so MFD has nothing correct to consume. Checked
            # against the factors directly rather than via
            # ``self.sde_params()`` (which raises for the same reason)
            # so the message names the offending factor, and so the
            # happy path never builds the full-size drift it would
            # otherwise have to discard.
            msg = (
                f"ProductSDE cannot be discretised when a factor has no "
                f"stationary covariance: "
                f"{type(self.kernel1).__name__}.P_inf is "
                f"{'None' if params_1.P_inf is None else 'set'} and "
                f"{type(self.kernel2).__name__}.P_inf is "
                f"{'None' if params_2.P_inf is None else 'set'}. The composite "
                f"diffusion of a product kernel is B1 (x) P2 + P1 (x) B2, "
                f"which needs both. Discretise the factors separately, or "
                f"give the factor a P_inf."
            )
            raise NotImplementedError(msg)

        transition_op = Kronecker(
            lx.MatrixLinearOperator(transition_1),
            lx.MatrixLinearOperator(transition_2),
        )
        stationary_covariance_op = Kronecker(
            lx.MatrixLinearOperator(params_1.P_inf, lx.symmetric_tag),
            lx.MatrixLinearOperator(params_2.P_inf, lx.symmetric_tag),
        )
        # Σ_η,t = Σ_x,∞ − F_t Σ_x,∞ F_tᵀ, factor by factor
        model_noise_op = process_noise_covariance(
            transition_op, stationary_covariance_op
        )
        model_noise_covariance = symmetrize(model_noise_op.as_matrix())
        return transition, model_noise_covariance


class QuasiPeriodicSDE(ProductSDE):
    """Quasi-periodic kernel: a Matérn factor times a periodic factor.

        k(τ) = k_Matérn(τ) k_periodic(τ)

    The state-space form is that of `ProductSDE`.

    Attributes:
        kernel1: Modulating factor k^(1), typically a `MaternSDE`.
        kernel2: Periodic factor k^(2), typically a `PeriodicSDE`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> kernel = gaussx.QuasiPeriodicSDE(
        ...     kernel1=gaussx.MaternSDE(
        ...         variance=jnp.asarray(1.0), lengthscale=jnp.asarray(2.0), order=0
        ...     ),
        ...     kernel2=gaussx.PeriodicSDE(
        ...         variance=jnp.asarray(1.0),
        ...         lengthscale=jnp.asarray(1.0),
        ...         period=jnp.asarray(1.0),
        ...         n_harmonics=3,
        ...     ),
        ... )
        >>> kernel.state_dim
        8
        >>> transition, noise = kernel.discretise(jnp.asarray(0.1))
        >>> transition.shape, noise.shape
        ((8, 8), (8, 8))
    """

    pass
