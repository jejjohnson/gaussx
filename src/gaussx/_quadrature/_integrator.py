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
