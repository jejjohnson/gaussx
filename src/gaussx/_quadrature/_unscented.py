"""Unscented integrator for uncertainty propagation (UKF-style)."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float

from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._quadrature import sigma_points
from gaussx._quadrature._types import GaussianState, PropagationResult


class UnscentedIntegrator(AbstractIntegrator):
    r"""Unscented transform: deterministic sigma points.

    Generates ``2N+1`` sigma points around the mean, propagates them
    through the nonlinear function, and reconstructs output moments:

        chi_i = mu + sqrt((N + lambda) * Sigma) @ xi_i
        y_i = f(chi_i)
        mu_y = sum(w_m * y_i)
        Sigma_y = sum(w_c * (y_i - mu_y)(y_i - mu_y)^T)
        cross_cov = sum(w_c * (chi_i - mu)(y_i - mu_y)^T)

    where ``lambda = alpha^2 * (N + kappa) - N``.

    Warning:
        The default ``alpha=1e-3`` (the classic Wan-van der Merwe scaled
        transform) puts a centre mean weight of about ``-1e6`` on the
        points and recovers the moments by cancellation, losing about six
        digits: in float32 a smooth nonlinearity is wrong in the second
        digit (gh-310). Prefer ``UnscentedIntegrator(alpha=1.0)``, the
        symmetric ``2N+1`` rule with a zero centre mean weight (exact for
        affine maps and positive covariance weights), which is what
        `moment_transform` and the nonlinear Kalman filters use by
        default. A `UserWarning` is emitted when the centre
        mean weight reaches magnitude ``1e3`` with a float32 state.

    Attributes:
        alpha: Spread parameter. Default ``1e-3``; ``1.0`` recommended.
        beta: Prior knowledge parameter (2.0 optimal for Gaussian).
        kappa: Secondary scaling. Default ``0.0``.
    """

    alpha: float = eqx.field(static=True, default=1e-3)
    beta: float = eqx.field(static=True, default=2.0)
    kappa: float = eqx.field(static=True, default=0.0)

    def integrate(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " M"]],
        state: GaussianState,
    ) -> PropagationResult:
        """Propagate Gaussian via unscented transform."""
        chi, w_m, w_c = self.points_and_weights(state)
        Y = jax.vmap(fn)(chi)
        return assemble_propagation_result(chi, Y, state.mean, w_m, w_c)

    def guarantees_psd(self, dim: int) -> bool:
        """Whether every scaled-unscented covariance weight is non-negative.

        With ``lambda = alpha^2 (N + kappa) - N`` the weights are
        ``1 / (2 (N + lambda))`` off-centre and
        ``lambda / (N + lambda) + 1 - alpha^2 + beta`` at the centre, so
        small ``alpha`` (the ``1e-3`` default) makes the centre weight
        large and negative; ``alpha = 1`` does not.
        """
        lam = self.alpha**2 * (dim + self.kappa) - dim
        spread = dim + lam
        if spread <= 0:
            return False
        return lam / spread + 1.0 - self.alpha**2 + self.beta >= 0

    def points_and_weights(
        self,
        state: GaussianState,
    ) -> tuple[Float[Array, "P N"], Float[Array, " P"], Float[Array, " P"]]:
        """Return the scaled unscented sigma points and weights.

        Args:
            state: Input Gaussian distribution.

        Returns:
            Tuple ``(points, w_m, w_c)`` with ``P = 2N + 1``. Mean and
            covariance weights differ in the centre point.
        """
        return sigma_points(
            state.mean,
            state.cov,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
        )
