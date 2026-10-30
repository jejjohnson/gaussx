"""Integrated Wiener (local linear trend) prior as a linear SDE.

It is dx(t) = F_c x(t) dt + L_c db(t), f(t) = H x(t), with drift matrix
F_c, diffusion (noise-effect) matrix L_c, white-noise spectral density Q_c
and measurement row H. F_c is nilpotent, so the process is non-stationary
and has no stationary covariance Σ_x,∞; the filter starts from an explicit
Σ_x,0 instead. Discretised over a step Δt_t, the transition
F_t = exp(F_c Δt_t) and the model-noise covariance Σ_η,t are polynomials
in Δt_t.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


def _inexact_dtype(*args) -> jnp.dtype:
    """Promotion of the hyperparameter dtypes, never to an integer.

    Every matrix this kernel builds is a *fraction* of a power of the
    step, so an integer promotion is not merely unusual but wrong: it
    truncates the coefficients to zero, and ``jnp.finfo``, which the
    diffuse default needs, rejects the dtype outright. An integer
    ``diffusion`` or ``dt`` therefore promotes to the default float
    dtype rather than being carried through.

    Args:
        *args: The hyperparameters and step whose dtypes are promoted.

    Returns:
        The promoted dtype, or the default float dtype if that is not
        inexact.
    """
    dtype = jnp.result_type(*args)
    if jnp.issubdtype(dtype, jnp.inexact):
        return dtype
    return jnp.result_type(float)


def _default_diffuse_variance(dtype) -> float:
    """Variance κ of the default diffuse initial covariance Σ_x,0 = κ I.

        κ = 1/√ε_mach

    A diffuse prior is pulled two ways. It must be vague relative to the
    observation-noise variance Σ_ε, since the prior's pull on the first
    estimate is Σ_ε / (Σ_x,0 + Σ_ε); but the Kalman update forms
    Σ_x − G Σ_y Gᵀ, and once Σ_x,0 / Σ_ε exceeds the dtype's precision
    that subtraction cancels to *exactly* zero: the filter then reports
    the first observation as noiseless and stays overconfident
    thereafter. In float32 the usual ``1e6`` does exactly that for any
    Σ_ε ≲ 0.1.

    So the default splits the difference in log space at 1/√ε_mach, with
    ε_mach the machine epsilon: about ``2.9e3`` in float32 and ``6.7e7``
    in float64, spending half the mantissa on vagueness and keeping half
    for the update. It is a *default*, not a recommendation: with a known
    data scale, and especially with a large Σ_ε, pass an explicit Σ_x,0.

    Args:
        dtype: Floating dtype whose machine epsilon ε_mach sets κ.

    Returns:
        The diffuse variance κ.
    """
    return float(1.0 / jnp.sqrt(jnp.finfo(dtype).eps))


class IntegratedWienerSDE(SDEKernel):
    """State-space form of the p-times integrated Wiener process.

        df^(p)(t) = db(t),    db(t) ~ 𝒩(0, Q_c dt)
        F_c = [[0, I_p], [0, 0]],  L_c = e_p,  H = e₀ᵀ

    with state x(t) = [f(t), f^(1)(t), …, f^(p)(t)], D_x = p + 1, and e_i the
    i-th unit vector. The top derivative is white noise and every lower
    component is its integral; at p = 1 this is the local linear trend
    prior. F_c is nilpotent, which is what makes the process
    **non-stationary**: its marginal variance grows without bound, so no
    Σ_x,∞ exists and `sde_params` reports ``P_inf=None``. `discretise` is
    overridden with the exact closed form and never needs one, but the
    filter has to be started from an explicit `initial_covariance` Σ_x,0.

    That non-stationarity is the point: unlike a Matérn prior, the local
    linear trend commits to no lengthscale. It is smooth and linear unless
    the data push back, so a long record may drift without the model
    asserting a scale on which it must revert.

    Attributes:
        diffusion: White-noise spectral density Q_c driving the top
            derivative. Must be non-negative: Σ_η,t is linear in it, so a
            negative value returns something that is not a covariance.
            Like every hyperparameter in the kernel zoo this is assumed
            rather than checked; constrain it with a positive transform
            when it is learned.
        order: Number of integrations p. ``0`` is a Brownian random walk;
            ``1`` (the default) is the local linear trend, which is also
            the cubic-spline-equivalent prior: f^(2) is white noise there,
            and the smoothed posterior mean is the cubic smoothing spline.
            Each further order raises the spline by two degrees, so ``2``
            is the quintic-spline prior.
        P_0: Initial state covariance Σ_x,0. A modelling choice rather
            than something the kernel can derive; ``None`` (the default)
            means a diffuse κ I with κ = 1/√ε_mach for the dtype.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> kernel = gaussx.IntegratedWienerSDE(diffusion=jnp.asarray(2.0))
        >>> kernel.state_dim, kernel.stationary, kernel.sde_params().P_inf
        (2, False, None)

        The local linear trend at Δt = 1: F_t = [[1, 1], [0, 1]] and
        Σ_η,t = Q_c [[1/3, 1/2], [1/2, 1]]:

        >>> transition, noise = kernel.discretise(jnp.asarray(1.0))
        >>> bool(jnp.allclose(transition, jnp.array([[1.0, 1.0], [0.0, 1.0]])))
        True
        >>> bool(jnp.allclose(noise, 2.0 * jnp.array([[1 / 3, 0.5], [0.5, 1.0]])))
        True

        A negative order is rejected at construction:

        >>> gaussx.IntegratedWienerSDE(
        ...     diffusion=jnp.asarray(1.0), order=-1
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: IntegratedWienerSDE order must be non-negative
    """

    diffusion: Float[Array, ""]
    order: int = eqx.field(static=True, default=1)
    P_0: Float[Array, "d d"] | None = None

    def __check_init__(self) -> None:
        """Reject a negative order at construction.

        The state length is D_x = p + 1, so a negative order gives an
        empty state and fails later, deep inside whichever method is
        called first, with an index error about an axis of size zero.

        Raises:
            ValueError: If ``order`` is negative.
        """
        if self.order < 0:
            msg = (
                f"IntegratedWienerSDE order must be non-negative "
                f"(the state is [f, ..., f^(order)], of dimension "
                f"order + 1), got {self.order}."
            )
            raise ValueError(msg)

    @property
    def state_dim(self) -> int:
        """State length D_x = p + 1."""
        return self.order + 1

    @property
    def stationary(self) -> bool:
        """``False``: the marginal variance grows without bound."""
        return False

    def sde_params(self) -> SDEParams:
        """Continuous-time parameters (F_c, L_c, H, Q_c), with no Σ_x,∞.

            F_c = [[0, I_p], [0, 0]],  L_c = e_p,  H = e₀ᵀ

        The drift is nilpotent, so no stationary covariance exists; see
        `initial_covariance` for what starts the filter instead.

        Returns:
            `SDEParams` with ``P_inf=None``.
        """
        # Constant blocks follow the hyperparameter dtype; untyped
        # ``jnp.zeros``/``jnp.eye`` are float64 under x64 (gh-224).
        dtype = _inexact_dtype(self.diffusion)
        state_size = self.state_dim
        # F_c = [[0, I_p], [0, 0]], the upward shift
        drift = jnp.eye(state_size, k=1, dtype=dtype)
        # L_c = e_p
        noise_effect = (
            jnp.zeros((state_size, 1), dtype=dtype).at[state_size - 1, 0].set(1.0)
        )
        # H = e₀ᵀ
        obs_operator = jnp.zeros((1, state_size), dtype=dtype).at[0, 0].set(1.0)
        # Q_c
        spectral_density = jnp.full((1, 1), self.diffusion, dtype=dtype)
        return SDEParams(
            F=drift,
            L=noise_effect,
            H=obs_operator,
            Q_c=spectral_density,
            P_inf=None,
        )

    def initial_covariance(self) -> Float[Array, "d d"]:
        """Initial state covariance Σ_x,0: the ``P_0`` field, or κ I.

            Σ_x,0 = κ I,    κ = 1/√ε_mach    (when ``P_0`` is ``None``)

        The diffuse default sets κ from the dtype's precision through
        `_default_diffuse_variance`; a vaguer prior than that is not merely
        wasteful but actively wrong, since the Kalman update cancels it to
        a zero-variance first estimate.

        Pass a ``P_0`` to encode what is actually known about the level
        and its derivatives at the first time point, e.g. diag(κ, s²) for
        a vague level and a slope of scale s. Do so in particular when the
        observation noise is large: the default is diffuse relative to an
        observation-noise variance of order one, not relative to every
        scale.

        Returns:
            Initial state covariance Σ_x,0.
        """
        dtype = _inexact_dtype(self.diffusion)
        if self.P_0 is None:
            # κ = 1/√ε_mach
            diffuse_variance = _default_diffuse_variance(dtype)
            # Σ_x,0 = κ I
            return diffuse_variance * jnp.eye(self.state_dim, dtype=dtype)
        return jnp.asarray(self.P_0, dtype=dtype)

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        """Closed-form discretisation, with no matrix exponential and no Σ_x,∞.

            [F_t]_ij = Δt^(j−i) / (j−i)!,    j ≥ i,  else 0
            [Σ_η,t]_ij = Q_c Δt^(2p+1−i−j) / ((p−i)! (p−j)! (2p+1−i−j))

        with i, j = 0, …, p. The drift is nilpotent, so its exponential
        terminates; at p = 1 this is the familiar F_t = [[1, Δt], [0, 1]]
        and Σ_η,t = Q_c [[Δt³/3, Δt²/2], [Δt²/2, Δt]]. Both are exact, so
        this route is cheaper *and* more accurate than the
        `gaussx.discretize_mfd` fallback the ``P_inf=None`` base
        implementation would otherwise take.

        Algorithm:
            1. Check Δt ≥ 0 with `equinox.error_if`.
            2. Form the 2p + 2 distinct powers Δt⁰, …, Δt^(2p+1) once.
            3. Build the static coefficient tables 1/(j−i)! and
               1/((p−i)! (p−j)! (2p+1−i−j)) on the host.
            4. Gather the powers by a static index table and scale.

        Note:
            The coefficients carry 1/(p−i)!, which outruns float64 from
            roughly ``order=90`` up and underflows to zero there (XLA
            flushes the subnormals in between). Nothing is silently wrong,
            since the entries are zero rather than garbage, but a prior
            that integrates white noise ninety times is not a numerically
            meaningful object in any dtype.

        Args:
            dt: Time step Δt. Must be non-negative: a negative step would
                return a Σ_η,t that is not a covariance (at p = 1 it is
                negative definite), rather than the harmless reverse-time
                transition the sign might suggest. Checked with
                `equinox.error_if`, matching `gaussx.discretize_mfd`, so
                under ``jit`` the error fires at evaluation rather than
                trace time.

        Returns:
            Tuple ``(A, Q)`` holding the transition F_t and the
            model-noise covariance Σ_η,t.
        """
        dtype = _inexact_dtype(self.diffusion, dt)
        order = self.order
        state_size = self.state_dim

        # The closed form below is a polynomial in dt with no guard of
        # its own, unlike the ``expm`` routes; a negative step would run
        # it happily and hand back an indefinite Σ_η.
        dt = eqx.error_if(
            dt, dt < 0, "IntegratedWienerSDE.discretise requires dt >= 0."
        )

        # Powers are taken with *static* Python exponents so JAX lowers
        # them to ``integer_pow``, whose derivative is exact at zero. A
        # traced exponent would go through the generic ``y * x**(y-1)``
        # rule instead, and dt = 0 -- which the natural
        # ``diff(times, prepend=times[0])`` produces at the first step --
        # would differentiate to 0 * inf = NaN.
        def power(exponent: int) -> Float[Array, ""]:
            if exponent == 0:
                return jnp.ones_like(dt)
            # Δt^m
            return dt**exponent

        # Only 2p+2 distinct powers appear across both matrices, so they
        # are formed once and gathered by a static index table. That is
        # O(D_x) traced operations rather than one per entry: at order 40
        # the per-entry version took over five seconds to trace.
        # Δt⁰, …, Δt^(2p+1)
        powers = jnp.stack([power(m) for m in range(2 * order + 2)]).astype(dtype)

        # Host-side index grids i (rows) and j (columns).
        row, col = np.indices((state_size, state_size))
        upper = col >= row
        # Zeroed through the *coefficient* rather than by masking a
        # negative power, so no entry raises dt to a negative exponent.
        # j − i, clamped at 0 below the diagonal
        transition_power_index = np.maximum(col - row, 0)
        # 2p + 1 − i − j
        noise_power_index = 2 * order + 1 - row - col

        # ``1 / n`` keeps both operands Python ints, which are unbounded
        # and divide to a correctly rounded float. ``1.0 / n`` would
        # convert first and raise OverflowError from order 98 up, where
        # the denominator exceeds the float range even though its
        # reciprocal is perfectly representable (down to a subnormal,
        # and to zero beyond that).
        factorials = [math.factorial(n) for n in range(state_size)]
        # 1/(j−i)! for j ≥ i, else 0
        transition_coefficients = np.array(
            [
                [1 / factorials[c - r] if c >= r else 0.0 for c in range(state_size)]
                for r in range(state_size)
            ]
        )
        # 1/((p−i)! (p−j)! (2p+1−i−j))
        noise_coefficients = np.array(
            [
                [
                    1
                    / (
                        factorials[order - r]
                        * factorials[order - c]
                        * (2 * order + 1 - r - c)
                    )
                    for c in range(state_size)
                ]
                for r in range(state_size)
            ]
        )

        # [F_t]_ij = Δt^(j−i) / (j−i)!
        transition = (
            jnp.asarray(transition_coefficients * upper, dtype=dtype)
            * powers[transition_power_index]
        )
        # [Σ_η,t]_ij = Q_c Δt^(2p+1−i−j) / ((p−i)! (p−j)! (2p+1−i−j))
        model_noise_covariance = (
            self.diffusion
            * jnp.asarray(noise_coefficients, dtype=dtype)
            * powers[noise_power_index]
        )
        return transition.astype(dtype), model_noise_covariance.astype(dtype)
