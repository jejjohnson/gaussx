"""Integrated Wiener (local linear trend) SDE kernel."""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from gaussx._ssm._sde_kernel import SDEKernel, SDEParams


# Variance of the default diffuse initial covariance, in units of the
# state. Large enough that the first few observations, not the prior,
# fix the level and slope; small enough to stay well inside float32.
_DEFAULT_DIFFUSE_VARIANCE = 1e6


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
            white noise driving the top derivative.
        order: Number of integrations $p$. ``0`` is a Brownian random
            walk, ``1`` (the default) the local linear trend, ``2`` the
            cubic-spline-equivalent prior.
        P_0: Initial state covariance, shape ``(order + 1, order + 1)``.
            A modelling choice rather than something the kernel can
            derive; ``None`` (the default) means a diffuse
            ``1e6 * I``.
    """

    diffusion: Float[Array, ""]
    order: int = eqx.field(static=True, default=1)
    P_0: Float[Array, "d d"] | None = None

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
        dtype = jnp.result_type(self.diffusion)
        d = self.state_dim
        F = jnp.eye(d, k=1, dtype=dtype)
        L = jnp.zeros((d, 1), dtype=dtype).at[d - 1, 0].set(1.0)
        H = jnp.zeros((1, d), dtype=dtype).at[0, 0].set(1.0)
        Q_c = jnp.reshape(self.diffusion, (1, 1)).astype(dtype)
        return SDEParams(F=F, L=L, H=H, Q_c=Q_c, P_inf=None)

    def initial_covariance(self) -> Float[Array, "d d"]:
        r"""Return the initial state covariance $P_0$.

        Defaults to a diffuse ``1e6 * I`` when the ``P_0`` field is
        ``None``. Pass a ``P_0`` to encode what is actually known about
        the level and its derivatives at the first time point — e.g.
        ``diag(kappa, s^2)`` for a vague level and a slope of scale
        ``s``.
        """
        dtype = jnp.result_type(self.diffusion)
        if self.P_0 is None:
            return _DEFAULT_DIFFUSE_VARIANCE * jnp.eye(self.state_dim, dtype=dtype)
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

        Args:
            dt: Time step (scalar, non-negative).

        Returns:
            Tuple ``(A, Q)``, both shape ``(order + 1, order + 1)``.
        """
        dtype = jnp.result_type(self.diffusion, dt)
        p = self.order
        d = self.state_dim

        # Exponent and coefficient tables depend only on the static
        # order, so they are built once in NumPy at trace time.
        i = np.arange(d)[:, None]
        j = np.arange(d)[None, :]
        factorial = np.vectorize(math.factorial)

        upper = j >= i
        # The lower triangle is zeroed through the *coefficient*, and its
        # exponent clipped at zero, so no entry raises dt to a negative
        # power: dt ** -1 is inf at dt = 0, and a zero coefficient times
        # inf is NaN rather than the zero intended.
        a_exponent = jnp.asarray(np.maximum(j - i, 0), dtype=dtype)
        a_coeff = jnp.asarray(
            np.where(upper, 1.0 / factorial(np.maximum(j - i, 0)), 0.0),
            dtype=dtype,
        )
        A = a_coeff * dt**a_exponent

        q_exponent = jnp.asarray(2 * p + 1 - i - j, dtype=dtype)
        q_coeff = jnp.asarray(
            1.0 / (factorial(p - i) * factorial(p - j) * (2 * p + 1 - i - j)),
            dtype=dtype,
        )
        Q = self.diffusion * q_coeff * dt**q_exponent
        return A, Q
