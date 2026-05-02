"""Abstract SDE kernel base class and SDEParams container."""

from __future__ import annotations

import abc
from typing import NamedTuple

import equinox as eqx
import jax
import jax.scipy.linalg as jsl
from jaxtyping import Array, Float


class SDEParams(NamedTuple):
    """Continuous-time SDE parameters for a stationary kernel.

    Defines the linear time-invariant SDE::

        dx = F x dt + L dW,   W ~ N(0, Q_c dt)

    with observation model ``y = H x``.

    Attributes:
        F: Drift matrix, shape ``(d, d)``.
        L: Diffusion matrix, shape ``(d, s)``.
        H: Observation matrix, shape ``(1, d)``.
        Q_c: Spectral density, shape ``(s, s)``.
        P_inf: Stationary covariance, shape ``(d, d)``.
    """

    F: Float[Array, "d d"]
    L: Float[Array, "d s"]
    H: Float[Array, "1 d"]
    Q_c: Float[Array, "s s"]
    P_inf: Float[Array, "d d"]


class SDEKernel(eqx.Module):
    """Abstract base class for state-space kernel representations.

    Subclasses implement :meth:`sde_params` to provide the continuous-time
    SDE matrices ``(F, L, H, Q_c, P_inf)``. The default :meth:`discretise`
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

        Default implementation computes::

            A = expm(F * dt)
            Q = P_inf - A @ P_inf @ A^T

        Subclasses may override with closed-form expressions.

        Args:
            dt: Time step (scalar, positive).

        Returns:
            Tuple ``(A, Q)`` where A is the transition matrix and
            Q is the process noise covariance.
        """
        params = self.sde_params()
        A = jsl.expm(params.F * dt)
        Q = params.P_inf - A @ params.P_inf @ A.T
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
