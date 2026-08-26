"""Integrated Wiener (local linear trend) SDE kernel."""

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
    truncates the coefficients to zero, and ``jnp.finfo`` — which the
    diffuse default needs — rejects the dtype outright. An integer
    ``diffusion`` or ``dt`` therefore promotes to the default float
    dtype rather than being carried through.
    """
    dtype = jnp.result_type(*args)
    if jnp.issubdtype(dtype, jnp.inexact):
        return dtype
    return jnp.result_type(float)


def _default_diffuse_variance(dtype) -> float:
    r"""Variance of the default diffuse initial covariance.

    A diffuse prior is pulled two ways. It must be vague relative to the
    observation noise $R$ — the prior's pull on the first estimate is
    $R / (P_0 + R)$ — but the Kalman update forms $P - K S K^\top$, and
    once $P_0 / R$ exceeds the dtype's precision that subtraction
    cancels to *exactly* zero: the filter then reports the first
    observation as noiseless and stays overconfident thereafter. In
    float32 the usual ``1e6`` does exactly that for any $R \lesssim
    0.1$.

    So the default splits the difference in log space at
    $1/\sqrt{\varepsilon}$ — about ``2.9e3`` in float32 and ``6.7e7``
    in float64 — spending half the mantissa on vagueness and keeping
    half for the update. It is a *default*, not a recommendation: with
    a known data scale, and especially with large $R$, pass an explicit
    ``P_0``.
    """
    return float(1.0 / jnp.sqrt(jnp.finfo(dtype).eps))


class IntegratedWienerSDE(SDEKernel):
    r"""State-space representation of an integrated Wiener process.

    The $p$-times integrated Wiener process — the local linear trend
    prior at ``order=1`` — with state
    $x(t) = [f(t), f'(t), \dots, f^{(p)}(t)]$ and

    $$
    \mathrm{d} f^{(p)}(t) = \sqrt{q} \, \mathrm{d} W(t),
    $$

    so the top derivative is white noise and every lower component is its
    integral. State dimension is ``order + 1``. The SDE matrices are

    $$
    F = \begin{bmatrix} 0 & I_p \\ 0 & 0 \end{bmatrix}, \quad
    L = e_p, \quad
    H = e_0^\top, \quad
    Q_c = q ,
    $$

    with $F$ nilpotent, which is what makes the process **non-stationary**:
    its marginal variance grows without bound, so no $P_\infty$ exists and
    `sde_params` reports ``P_inf=None``. `discretise` is overridden with
    the exact closed form (below) and never needs one, but the filter has
    to be started from an explicit `initial_covariance` instead.

    That non-stationarity is the point: unlike a Matérn prior, the local
    linear trend commits to no lengthscale — it is smooth and linear
    unless the data push back — so a long record may drift without the
    model asserting a scale on which it must revert.

    Attributes:
        diffusion: Diffusion intensity $q$, the spectral density of the
            white noise driving the top derivative. Must be
            non-negative — $Q$ is linear in it, so a negative value
            returns something that is not a covariance. Like every
            hyperparameter in the kernel zoo this is assumed rather
            than checked; constrain it with a positive transform when
            it is learned.
        order: Number of integrations $p$. ``0`` is a Brownian random
            walk; ``1`` (the default) is the local linear trend, which
            is also the cubic-spline-equivalent prior — it is $f''$ that
            is white noise there, and the smoothed posterior mean is the
            cubic smoothing spline. Each further order raises the spline
            by two degrees, so ``2`` is the quintic-spline prior.
        P_0: Initial state covariance, shape ``(order + 1, order + 1)``.
            A modelling choice rather than something the kernel can
            derive; ``None`` (the default) means a diffuse
            ``_default_diffuse_variance(dtype) * I``.
    """

    diffusion: Float[Array, ""]
    order: int = eqx.field(static=True, default=1)
    P_0: Float[Array, "d d"] | None = None

    def __check_init__(self) -> None:
        """Reject a negative order at construction.

        The state dimension is ``order + 1``, so a negative order gives
        an empty state and fails later, deep inside whichever method is
        called first, with an index error about an axis of size zero.
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
        return self.order + 1

    @property
    def stationary(self) -> bool:
        """``False`` — the marginal variance grows without bound."""
        return False

    def sde_params(self) -> SDEParams:
        """Return SDE parameters, with ``P_inf=None``.

        The drift is nilpotent, so no stationary covariance exists; see
        `initial_covariance` for what starts the filter instead.
        """
        # Constant blocks follow the hyperparameter dtype; untyped
        # ``jnp.zeros``/``jnp.eye`` are float64 under x64 (gh-224).
        dtype = _inexact_dtype(self.diffusion)
        d = self.state_dim
        F = jnp.eye(d, k=1, dtype=dtype)
        L = jnp.zeros((d, 1), dtype=dtype).at[d - 1, 0].set(1.0)
        H = jnp.zeros((1, d), dtype=dtype).at[0, 0].set(1.0)
        Q_c = jnp.reshape(self.diffusion, (1, 1)).astype(dtype)
        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=None)

    def initial_covariance(self) -> Float[Array, "d d"]:
        r"""Return the initial state covariance $P_0$.

        Defaults to a diffuse ``kappa * I`` when the ``P_0`` field is
        ``None``, with ``kappa`` set from the dtype's precision by
        `_default_diffuse_variance` — a vaguer prior than that is not
        merely wasteful but actively wrong, since the Kalman update
        cancels it to a zero-variance first estimate.

        Pass a ``P_0`` to encode what is actually known about the level
        and its derivatives at the first time point — e.g.
        ``diag(kappa, s^2)`` for a vague level and a slope of scale
        ``s``. Do so in particular when the observation noise is large:
        the default is diffuse relative to a noise variance of order
        one, not relative to every scale.
        """
        dtype = _inexact_dtype(self.diffusion)
        if self.P_0 is None:
            kappa = _default_diffuse_variance(dtype)
            return kappa * jnp.eye(self.state_dim, dtype=dtype)
        return jnp.asarray(self.P_0, dtype=dtype)

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        r"""Closed-form discretisation — no ``expm``, no $P_\infty$.

        The drift is nilpotent, so its exponential terminates:

        $$
        A(\Delta t)_{ij} = \frac{\Delta t^{\,j-i}}{(j-i)!} \;\; (j \ge i),
        \qquad
        Q(\Delta t)_{ij} = q \,
            \frac{\Delta t^{\,2p+1-i-j}}
                 {(p-i)!\,(p-j)!\,(2p+1-i-j)} ,
        $$

        which at ``order=1`` is the familiar

        $$
        A = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix},
        \qquad
        Q = q \begin{bmatrix}
            \Delta t^3/3 & \Delta t^2/2 \\
            \Delta t^2/2 & \Delta t
        \end{bmatrix}.
        $$

        Both are exact, so this route is cheaper *and* more accurate than
        the `gaussx.discretise_mfd` fallback the ``P_inf=None`` base
        implementation would otherwise take.

        Note:
            The coefficients carry $1/(p-i)!$, which outruns float64
            from roughly ``order=90`` up and underflows to zero there
            (XLA flushes the subnormals in between). Nothing is
            silently wrong — the entries are zero rather than garbage —
            but a prior that integrates white noise ninety times is not
            a numerically meaningful object in any dtype.

        Args:
            dt: Time step. Must be non-negative — a negative step would
                return a ``Q`` that is not a covariance (at ``order=1``
                it is negative definite), rather than the harmless
                reverse-time transition the sign might suggest. Checked
                with `equinox.error_if`, matching
                `gaussx.discretise_mfd`, so under ``jit`` the error
                fires at evaluation rather than trace time.

        Returns:
            Tuple ``(A, Q)``, both shape ``(order + 1, order + 1)``.
        """
        dtype = _inexact_dtype(self.diffusion, dt)
        p = self.order
        d = self.state_dim

        # The closed form below is a polynomial in dt with no guard of
        # its own, unlike the ``expm`` routes; a negative step would run
        # it happily and hand back an indefinite Q.
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
            return dt**exponent

        # Only 2p+2 distinct powers appear across both matrices, so they
        # are formed once and gathered by a static index table. That is
        # O(d) traced operations rather than one per entry: at order 40
        # the per-entry version took over five seconds to trace.
        powers = jnp.stack([power(k) for k in range(2 * p + 2)]).astype(dtype)

        i = np.arange(d)[:, None]
        j = np.arange(d)[None, :]
        upper = j >= i
        # Zeroed through the *coefficient* rather than by masking a
        # negative power, so no entry raises dt to a negative exponent.
        a_index = np.maximum(j - i, 0)
        q_index = 2 * p + 1 - i - j

        # ``1 / n`` keeps both operands Python ints, which are unbounded
        # and divide to a correctly rounded float. ``1.0 / n`` would
        # convert first and raise OverflowError from order 98 up, where
        # the denominator exceeds the float range even though its
        # reciprocal is perfectly representable (down to a subnormal,
        # and to zero beyond that).
        factorial = [math.factorial(n) for n in range(d)]
        a_coeff = np.array(
            [
                [1 / factorial[j - i] if j >= i else 0.0 for j in range(d)]
                for i in range(d)
            ]
        )
        q_coeff = np.array(
            [
                [
                    1 / (factorial[p - i] * factorial[p - j] * (2 * p + 1 - i - j))
                    for j in range(d)
                ]
                for i in range(d)
            ]
        )

        A = jnp.asarray(a_coeff * upper, dtype=dtype) * powers[a_index]
        Q = self.diffusion * jnp.asarray(q_coeff, dtype=dtype) * powers[q_index]
        return A.astype(dtype), Q.astype(dtype)
