"""Abstract integrator protocol for Gaussian integral approximation."""

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array, Float

from gaussx._quadrature._types import GaussianState, PropagationResult


class AbstractIntegrator(eqx.Module):
    """Protocol for Gaussian integral approximation.

    Subclasses implement ``integrate`` to propagate a Gaussian through
    a nonlinear function, returning an approximate output distribution
    and (optionally) input-output cross-covariance.
    """

    @abc.abstractmethod
    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate a Gaussian through ``fn``, returning output moments.

        Args:
            fn: Nonlinear function mapping ``(N,) -> (M,)``.
            state: Input Gaussian distribution.

        Returns:
            ``PropagationResult`` with output distribution and optional
            cross-covariance.
        """
        ...

    def points_and_weights(
        self,
        state: GaussianState,
    ) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
        """Return the raw quadrature points and weights for ``state``.

        Point-based rules override this so that consumers needing the raw
        evaluations — `gaussx.moment_match`,
        `gaussx.statistical_linear_regression` — can reuse a single pass
        over the points instead of re-deriving the rule.

        Args:
            state: Input Gaussian distribution.

        Returns:
            Tuple ``(points, w_m, w_c)`` where ``points`` has shape
            ``(P, N)`` in the input space, ``w_m`` are the mean weights and
            ``w_c`` the covariance weights, both shape ``(P,)``.

        Raises:
            NotImplementedError: If the integrator is not point-based, e.g.
                `TaylorIntegrator`, which linearises instead of sampling.
        """
        msg = (
            f"{type(self).__name__} is not a point-based rule and does not "
            f"expose quadrature points and weights."
        )
        raise NotImplementedError(msg)
