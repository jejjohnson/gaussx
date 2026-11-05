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

        y = g(x),    x ~ 𝒩(μ_x, Σ_x),    y ≈ 𝒩(μ_y, Σ_y),    Σ_xy = Cov[x, y]

    Subclasses implement ``integrate`` to propagate a Gaussian through a
    nonlinear map g, returning an approximate output distribution and
    (optionally) the input-output cross-covariance.

    Examples:
        A custom rule only has to implement ``integrate``:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> class Linearised(AbstractIntegrator):
        ...     def integrate(self, fn, state):
        ...         return gaussx.TaylorIntegrator().integrate(fn, state)
        >>> Linearised().guarantees_psd(2)
        False
    """

    @abc.abstractmethod
    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate a Gaussian through ``fn``, returning output moments.

        Args:
            fn: Nonlinear map g.
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            `PropagationResult` with the output distribution 𝒩(μ_y, Σ_y) and
            optional cross-covariance Σ_xy.
        """
        ...

    def guarantees_psd(self, dim: int) -> bool:
        """Whether the rule's moment triple is PSD by construction at ``dim``.

        A rule whose covariance weights are all non-negative (or a
        linearisation, which propagates the covariance exactly) returns a
        consistent joint over (x, g(x)), so its output covariance and the
        implied conditional are PSD. Consumers such as
        `gaussx.nonlinear_kalman_filter` skip their definiteness checks for
        such rules (gh-331). The default is ``False``, so a custom rule is
        always checked; the built-in positive-weight rules override it.

        Args:
            dim: Input dimension D_x the rule is applied at.

        Returns:
            ``True`` if every covariance weight is non-negative at ``dim``.

        Examples:
            >>> import gaussx
            >>> gaussx.CubatureIntegrator().guarantees_psd(3)
            True
        """
        del dim
        return False

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
            state: Input Gaussian 𝒩(μ_x, Σ_x).

        Returns:
            Tuple ``(points, w_m, w_c)`` holding the points χᵢ in the input
            space, the mean weights w_m,i and the covariance weights w_c,i.

        Raises:
            NotImplementedError: If the integrator is not point-based, e.g.
                `TaylorIntegrator`, which linearises instead of sampling.

        Examples:
            >>> import jax.numpy as jnp
            >>> import lineax as lx
            >>> import gaussx
            >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
            >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
            >>> rule = gaussx.CubatureIntegrator()
            >>> rule.points_and_weights(state)[0].shape
            (4, 2)
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
    """Moment triple of (x, g(x)) for a Gaussian x.

        (𝔼[g(x)], Cov[g(x)], Cov[x, g(x)]),    x ~ 𝒩(μ_x, Σ_x)

    This is the one method-specific step of every Gaussian filter: the
    filter loop, the smoother backward pass and the log-likelihood are
    identical across methods, and only this transform changes. Taking
    dense arrays and returning dense arrays, it is the convenient form for
    building a filter — `gaussx.nonlinear_kalman_filter` is built on it —
    while `AbstractIntegrator.integrate` is the operator-typed form.

    Unlike calling ``integrate`` directly, this **requires** the
    cross-covariance and says so if it is missing, rather than letting a
    ``None`` propagate into a silently zero Kalman gain.

    Args:
        fn: The map g to push the Gaussian through.
        mean: Input mean μ_x.
        cov: Input covariance Σ_x. Assumed PSD.
        integrator: Rule realising the transform. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``. Note this is *not*
            `gaussx.UnscentedIntegrator`'s own ``alpha=1e-3`` default,
            which is not safe in float32.

    Returns:
        Tuple ``(mean_out, cov_out, cross_cov)`` holding 𝔼[g(x)], Cov[g(x)]
        and Cov[x, g(x)].

    Raises:
        TypeError: If the integrator does not supply a cross-covariance.
            Raised at trace time, so it costs nothing per step.

    Examples:
        An affine map under the default rule:

        >>> import jax.numpy as jnp
        >>> mean_out, cov_out, cross_cov = moment_transform(
        ...     lambda x: 2.0 * x, jnp.ones(2), jnp.eye(2)
        ... )
        >>> bool(jnp.allclose(cov_out, 4.0 * jnp.eye(2)))
        True
        >>> bool(jnp.allclose(cross_cov, 2.0 * jnp.eye(2)))
        True
    """
    if integrator is None:
        # lazy import, cycle: _quadrature._unscented -> _quadrature._integrator
        from gaussx._quadrature._unscented import UnscentedIntegrator

        # alpha=1.0, not UnscentedIntegrator's own 1e-3: that spread
        # recovers even affine moments by cancellation between weights of
        # order 1e6, which costs ~7 digits and is ruinous in float32.
        integrator = UnscentedIntegrator(alpha=1.0)

    covariance_op = lx.MatrixLinearOperator(cov, lx.positive_semidefinite_tag)
    state = GaussianState(mean=mean, cov=covariance_op)
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
