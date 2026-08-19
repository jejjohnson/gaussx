"""Abstract SDE kernel base class and SDEParams container."""

from __future__ import annotations

import abc
from typing import NamedTuple

import equinox as eqx
import jax
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float

from gaussx._linalg._symmetrize import symmetrize
from gaussx._ssm._discretise import discretise_mfd, process_noise_covariance


class SDEParams(NamedTuple):
    """Continuous-time SDE parameters for a stationary kernel.

    Defines the linear time-invariant SDE:

        dx = F x dt + L dW,   W ~ N(0, Q_c dt)

    with observation model ``y = H x``.

    Attributes:
        F: Drift matrix, shape ``(d, d)``.
        L: Diffusion matrix, shape ``(d, s)``.
        H: Observation matrix, shape ``(1, d)``.
        Q_c: Spectral density, shape ``(s, s)``.
        P_inf: Stationary covariance, shape ``(d, d)``, or ``None`` when
            the kernel has no closed-form stationary covariance — as for a
            learned drift matrix. `SDEKernel.discretise` then falls back to
            `gaussx.discretise_mfd`, which needs no ``P_inf``.
    """

    F: Float[Array, "d d"]
    L: Float[Array, "d s"]
    H: Float[Array, "1 d"]
    Q_c: Float[Array, "s s"]
    P_inf: Float[Array, "d d"] | None = None


class SDEKernel(eqx.Module):
    """Abstract base class for state-space kernel representations.

    Subclasses implement `sde_params` to provide the continuous-time
    SDE matrices ``(F, L, H, Q_c, P_inf)``. The default `discretise`
    uses the matrix exponential for discretization; subclasses may override
    with closed-form solutions.
    """

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """Dimension of the latent state vector."""
        ...

    @abc.abstractmethod
    def sde_params(self) -> SDEParams:
        """Return continuous-time SDE parameters."""
        ...

    def discretise(
        self,
        dt: Float[Array, ""],
    ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
        """Discretise the SDE at time step ``dt``.

        Default implementation computes:

            A = expm(F * dt)
            Q = P_inf - A @ P_inf @ A^T

        When ``sde_params()`` returns ``P_inf=None`` — a kernel with no
        closed-form stationary covariance, such as one whose drift is a
        learned parameter — this falls back to `gaussx.discretise_mfd`,
        which recovers both ``A`` and ``Q`` from one matrix exponential and
        is well defined for every ``F``. The fallback is chosen at trace
        time from a static ``None`` check, so kernels that do supply
        ``P_inf`` keep the stationary route and its precision exactly.

        Subclasses may override with closed-form expressions.

        Args:
            dt: Time step (scalar, non-negative).

        Returns:
            Tuple ``(A, Q)`` where A is the transition matrix and
            Q is the process noise covariance.
        """
        params = self.sde_params()
        if params.P_inf is None:
            diffusion = params.L @ params.Q_c @ params.L.T
            return discretise_mfd(params.F, diffusion, dt)
        A = jsl.expm(params.F * dt)
        Q = symmetrize(process_noise_covariance(A, params.P_inf))
        return A, Q

    def discretise_sequence(
        self,
        dt: Float[Array, " N"],
    ) -> tuple[Float[Array, "N d d"], Float[Array, "N d d"]]:
        """Discretise the SDE at multiple time steps.

        Args:
            dt: Time steps, shape ``(N,)``.

        Returns:
            Tuple ``(A_seq, Q_seq)`` with shapes ``(N, d, d)``.
        """
        return jax.vmap(self.discretise)(dt)
