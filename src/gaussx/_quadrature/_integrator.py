"""Abstract integrator protocol for Gaussian integral approximation."""

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import lineax as lx
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


def moment_transform(
    fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
    mean: Float[Array, " N"],
    cov: Float[Array, "N N"],
    *,
    integrator: AbstractIntegrator | None = None,
) -> tuple[Float[Array, " M"], Float[Array, "M M"], Float[Array, "N M"]]:
    r"""Moment triple of ``(x, fn(x))`` for ``x`` Gaussian.

    The Gaussian moment transform,

    $$
    \mathcal{T}[g; \mu, \Sigma] = (\mu_g,\ \Sigma_g,\ \Sigma_{xg}),
    \qquad x \sim \mathcal{N}(\mu, \Sigma),
    $$

    with $\mu_g = \mathbb{E}[g(x)]$, $\Sigma_g = \mathrm{Cov}[g(x)]$ and
    $\Sigma_{xg} = \mathrm{Cov}[x, g(x)]$.

    This is the one method-specific step of every Gaussian filter: the
    filter loop, the smoother backward pass and the log-likelihood are
    identical across methods, and only $\mathcal{T}$ changes. Taking
    dense arrays and returning dense arrays, it is the convenient form for
    building a filter — `gaussx.nonlinear_kalman_filter` is built on it —
    while `AbstractIntegrator.integrate` is the operator-typed form.

    Unlike calling ``integrate`` directly, this **requires** the
    cross-covariance and says so if it is missing, rather than letting a
    ``None`` propagate into a silently zero Kalman gain.

    Args:
        fn: The map ``(N,) -> (M,)`` to push the Gaussian through.
        mean: Input mean $\mu$, shape ``(N,)``.
        cov: Input covariance $\Sigma$, shape ``(N, N)``. Assumed PSD.
        integrator: Rule realising $\mathcal{T}$. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``. Note this is *not*
            `gaussx.UnscentedIntegrator`'s own ``alpha=1e-3`` default,
            which is not safe in float32.

    Returns:
        Tuple ``(mean_out, cov_out, cross_cov)`` with shapes ``(M,)``,
        ``(M, M)`` and ``(N, M)``.

    Raises:
        TypeError: If the integrator does not supply a cross-covariance.
            Raised at trace time, so it costs nothing per step.
    """
    if integrator is None:
        from gaussx._quadrature._unscented import UnscentedIntegrator

        # alpha=1.0, not UnscentedIntegrator's own 1e-3: that spread
        # recovers even affine moments by cancellation between weights of
        # order 1e6, which costs ~7 digits and is ruinous in float32.
        integrator = UnscentedIntegrator(alpha=1.0)

    state = GaussianState(
        mean=mean, cov=lx.MatrixLinearOperator(cov, lx.positive_semidefinite_tag)
    )
    result = integrator.integrate(fn, state)
    if result.cross_cov is None:
        msg = (
            f"{type(integrator).__name__} returned cross_cov=None, but the "
            f"moment transform is defined by all three moments. Downstream "
            f"consumers build a Kalman gain as cross_cov @ S^-1, so a "
            f"missing cross-covariance would silently give a zero gain."
        )
        raise TypeError(msg)
    return result.state.mean, result.state.cov.as_matrix(), result.cross_cov
