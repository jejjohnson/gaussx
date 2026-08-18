"""Statistical linear regression: moment-matched linear-Gaussian surrogate."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._symmetrize import symmetrize
from gaussx._quadrature._assembly import assemble_propagation_result
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._types import GaussianState


class SLRResult(eqx.Module):
    r"""Linear-Gaussian surrogate produced by statistical linear regression.

    Encodes ``p(y | f) ≈ N(y | A f + b, Ω)``, the moment-matched
    approximation of a nonlinear conditional under a Gaussian state.

    Attributes:
        A: Linearisation gain, shape ``(M, D)``.
        b: Linearisation offset, shape ``(M,)``.
        omega: Residual noise covariance, shape ``(M, M)``.
        mu: Expected conditional mean ``E_q[E[y | f]]``, shape ``(M,)``.
    """

    A: Float[Array, "M D"]
    b: Float[Array, " M"]
    omega: Float[Array, "M M"]
    mu: Float[Array, " M"]


def statistical_linear_regression(
    conditional_mean_fn: Callable[[Float[Array, " D"]], Float[Array, " M"]],
    conditional_var_fn: Callable[[Float[Array, " D"]], Float[Array, "M M"]],
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> SLRResult:
    r"""Moment-match a nonlinear conditional to a linear-Gaussian surrogate.

    Given a conditional ``p(y | f)`` described by its first two moments and
    a Gaussian state ``q(f) = N(m_q, C_ff)``, returns the linear-Gaussian
    model ``p(y | f) ≈ N(y | A f + b, Ω)`` whose moments match those of the
    true conditional under ``q``.

    With

    $$
    \mu = \mathbb{E}_q[\mathbb{E}[y \mid f]], \quad
    C_{xf} = \mathrm{Cov}_q(f, \mathbb{E}[y \mid f]), \quad
    S = \mathrm{Var}_q(\mathbb{E}[y \mid f])
        + \mathbb{E}_q[\mathrm{Var}[y \mid f]],
    $$

    the surrogate is

    $$
    A = C_{xf}^\top C_{ff}^{-1}, \quad
    b = \mu - A m_q, \quad
    \Omega = S - A C_{xf}.
    $$

    This is the shared primitive behind posterior linearisation, the
    iterated Kalman smoother (IPLF / IPLS), and the conjugate Gaussian site
    that an EP cavity meets. Feeding ``(A, b, omega)`` into a standard
    linear-Gaussian Kalman update reproduces posterior-linearisation
    inference exactly.

    All expectations are evaluated with ``integrator``, so the accuracy /
    cost trade-off is chosen by the caller. Only the conditional moments
    are needed — no autodiff runs through the quadrature.

    Args:
        conditional_mean_fn: Maps a latent ``f`` of shape ``(D,)`` to the
            conditional mean ``E[y | f]`` of shape ``(M,)``.
        conditional_var_fn: Maps a latent ``f`` of shape ``(D,)`` to the
            conditional covariance ``Var[y | f]``. Either the full
            ``(M, M)`` covariance or, for conditionally independent
            outputs, an ``(M,)`` vector of variances.
        state: Gaussian state ``q(f)``. Its covariance must be symmetric.
        integrator: Point-based rule used for the expectations, e.g.
            `gaussx.FifthOrderCubatureIntegrator`.

    Returns:
        `SLRResult` with the gain, offset, residual covariance, and
        expected conditional mean.

    Raises:
        NotImplementedError: If ``integrator`` is not point-based.
        ValueError: If ``conditional_var_fn`` returns neither an ``(M,)``
            nor an ``(M, M)`` array.
    """
    chi, w_m, w_c = integrator.points_and_weights(state)

    # First pass: moments of the conditional mean under q(f).
    G = jax.vmap(conditional_mean_fn)(chi)  # (P, M)
    moments = assemble_propagation_result(chi, G, state.mean, w_m, w_c)
    mu = moments.state.mean  # (M,)
    var_of_mean = moments.state.cov.as_matrix()  # Var_q(E[y|f])
    cross_cov = moments.cross_cov  # C_xf, (D, M)
    assert cross_cov is not None, "Integrator must return cross_cov"

    # Second pass: E_q[Var[y|f]], reusing the same points.
    V = jax.vmap(conditional_var_fn)(chi)
    if V.ndim == 2:  # (P, M) diagonal variances
        mean_of_var = jnp.diag(jnp.sum(w_m[:, None] * V, axis=0))
    elif V.ndim == 3:  # (P, M, M) full covariances
        mean_of_var = jnp.sum(w_m[:, None, None] * V, axis=0)
    else:
        msg = (
            f"conditional_var_fn must return an (M,) or (M, M) array per "
            f"point, got trailing shape {V.shape[1:]}."
        )
        raise ValueError(msg)

    S = var_of_mean + mean_of_var

    # A = C_xf^T C_ff^{-1}; solve_rows solves C_ff x = row, and C_ff is
    # symmetric, so stacking the rows of C_xf^T gives A directly.
    A = solve_rows(state.cov, cross_cov.T)  # (M, D)

    b = mu - A @ state.mean
    omega = symmetrize(S - A @ cross_cov)

    return SLRResult(A=A, b=b, omega=omega, mu=mu)
