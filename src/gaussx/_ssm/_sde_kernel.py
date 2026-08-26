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
    """Continuous-time SDE parameters for a linear SDE.

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
            learned drift matrix, or a non-stationary kernel such as
            `gaussx.IntegratedWienerSDE`, which has no stationary
            covariance at all. `SDEKernel.discretise` then falls back to
            `gaussx.discretise_mfd`, which needs no ``P_inf``, and the
            filter is started from `SDEKernel.initial_covariance`
            instead.
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

    @property
    def stationary(self) -> bool:
        """Whether the process has a stationary distribution.

        ``True`` for every kernel in the zoo, which is why that is the
        default; a non-stationary kernel such as
        `gaussx.IntegratedWienerSDE` overrides it. Consumers should
        branch on this rather than on ``sde_params().P_inf is None``:
        the two are different questions, since a stationary kernel may
        report ``P_inf=None`` when it has no *closed form* for it (a
        learned drift, say).
        """
        return True

    @abc.abstractmethod
    def sde_params(self) -> SDEParams:
        """Return continuous-time SDE parameters."""
        ...

    def initial_covariance(self) -> Float[Array, "d d"]:
        r"""Return the covariance of the state at the first time point.

        For a stationary kernel the process is assumed started in its
        stationary distribution, so this is $P_\infty$ — the default
        implementation returns it and existing kernels need no change.
        A non-stationary kernel has no such limit to start from and must
        override this with an explicit choice.

        Returns:
            Initial state covariance, shape ``(d, d)``.

        Raises:
            ValueError: If ``sde_params().P_inf`` is ``None``, so there
                is no stationary covariance to fall back on.
        """
        P_inf = self.sde_params().P_inf
        if P_inf is None:
            msg = (
                f"{type(self).__name__} has no initial covariance: it "
                f"reports P_inf=None, and the default initial covariance "
                f"is the stationary one. Override initial_covariance() "
                f"with an explicit choice (a diffuse prior, typically), "
                f"or give the kernel a P_inf."
            )
            raise ValueError(msg)
        return P_inf

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
