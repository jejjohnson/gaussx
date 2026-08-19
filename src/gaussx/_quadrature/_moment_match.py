"""EP moment matching: tilted log-normaliser and its cavity-mean derivatives."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._symmetrize import symmetrize
from gaussx._primitives._inv import inv
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._types import GaussianState


class MomentMatchResult(eqx.Module):
    r"""Tilted log-normaliser and its derivatives w.r.t. the cavity mean.

    Attributes:
        log_Z: Scalar tilted log-normaliser ``log Z``.
        d_log_Z: Gradient ``∂ log Z / ∂m``, shape ``(D,)``.
        d2_log_Z: Hessian ``∂² log Z / ∂m²``, shape ``(D, D)``.
    """

    log_Z: Float[Array, ""]
    d_log_Z: Float[Array, " D"]
    d2_log_Z: Float[Array, "D D"]


def moment_match(
    log_lik_fn: Callable[[Float[Array, " D"]], Float[Array, ""]],
    state: GaussianState,
    integrator: AbstractIntegrator,
    *,
    power: float = 1.0,
) -> MomentMatchResult:
    r"""EP moment matching: tilted log-normaliser and its first two derivatives.

    Computes the log normaliser of the tilted distribution,

    $$
    \log Z = \log \int p(y \mid f)^{a}\, \mathcal{N}(f \mid m, C)\, df,
    $$

    together with

    $$
    \partial_m \log Z = C^{-1} \mathbb{E}_{\mathrm{tilt}}[f - m],
    \qquad
    \partial^2_m \log Z
        = C^{-1}\big(\mathrm{Cov}_{\mathrm{tilt}}[f] - C\big) C^{-1}.
    $$

    This is the EP quantity, *not* ``E_q[log p(y|f)]`` (which is what
    variational inference needs — see `gaussx.expected_log_likelihood`).

    Writing ``g = d_log_Z`` and ``H = d2_log_Z``, the tilted moments follow
    directly from the two derivatives:

    $$
    m_{\mathrm{tilt}} = m + C g, \qquad
    C_{\mathrm{tilt}} = C + C H C.
    $$

    The Gaussian site is the natural-parameter *difference* between the
    tilted distribution and the cavity, which reduces to

    $$
    \Lambda_{\mathrm{site}} = C_{\mathrm{tilt}}^{-1} - C^{-1}
        = -(I + H C)^{-1} H,
    \qquad
    \lambda_{\mathrm{site}} = C_{\mathrm{tilt}}^{-1} m_{\mathrm{tilt}}
        - C^{-1} m = (I + H C)^{-1} (g - H m),
    $$

    in the same ``nat2 = +Λ`` convention `gaussx.cavity_distribution` uses.
    That site approximates ``p(y | f) ** power``, so divide both naturals by
    ``power`` to recover the site for ``p(y | f)`` itself.

    Do **not** feed ``(g, -H)`` in as the site: it drops the ``-H m`` term
    and the ``(I + H C)^{-1}`` factor. Nor is `gaussx.newton_update` the
    right conversion here — it yields ``(g - H m, -H)``, which is the
    local-Newton site, i.e. only the first-order approximation to the EP
    site above.

    All three outputs come from a **single** pass over the quadrature
    points: applying Stein's lemma to the derivatives of the Gaussian
    density turns them into reweightings of the same likelihood
    evaluations, so no autodiff runs through the integrator and no extra
    likelihood evaluations are needed.

    The sum over points is evaluated with the log-sum-exp shift, so
    likelihoods that underflow in the raw scale are handled. Note that
    quadrature weights may be negative (the scaled unscented transform and
    the degree-5 cubature rule both have negative weights in places), so
    ``Z`` is not guaranteed positive for a badly matched rule; prefer a
    rule whose points cover the likelihood's mass.

    Args:
        log_lik_fn: Maps a latent ``f`` of shape ``(D,)`` to the scalar
            log-likelihood ``log p(y | f)``.
        state: Cavity distribution ``N(m, C)``. Its covariance must be
            symmetric.
        integrator: Point-based rule supplying the points and weights,
            e.g. `gaussx.FifthOrderCubatureIntegrator`.
        power: Power-EP exponent ``a``. ``1.0`` is standard EP; ``a < 1``
            corresponds to fractional EP / alpha-divergence minimisation.
            Documented for ``a`` in ``(0, 1]``.

    Returns:
        `MomentMatchResult` with ``log_Z``, ``d_log_Z``, ``d2_log_Z``.

    Raises:
        NotImplementedError: If ``integrator`` is not point-based.
    """
    chi, w_m, _ = integrator.points_and_weights(state)

    # Tilted weights, shifted for numerical stability. The shift cancels in
    # every ratio below and is differentiated as a constant.
    log_p = power * jax.vmap(log_lik_fn)(chi)  # (P,)
    shift = jax.lax.stop_gradient(jnp.max(log_p))
    p_tilde = w_m * jnp.exp(log_p - shift)  # (P,), signed
    Z_tilde = jnp.sum(p_tilde)

    log_Z = shift + jnp.log(Z_tilde)

    # u_i = C^{-1} (f_i − m); C is symmetric so a row-wise solve suffices.
    dx = chi - state.mean[None, :]  # (P, D)
    U = solve_rows(state.cov, dx)  # (P, D)

    # ∂_m log Z = Z^{-1} Σ w_i u_i p_i
    d_log_Z = jnp.sum(p_tilde[:, None] * U, axis=0) / Z_tilde

    # Z^{-1} ∂²_m Z = Z^{-1} Σ w_i (u_i u_iᵀ − C^{-1}) p_i
    second = (
        jnp.sum(
            p_tilde[:, None, None] * (U[:, :, None] * U[:, None, :]),
            axis=0,
        )
        / Z_tilde
    )
    cov_inv = inv(state.cov).as_matrix()

    # ∂²_m log Z = Z^{-1} ∂²_m Z − (∂_m log Z)(∂_m log Z)ᵀ
    d2_log_Z = symmetrize(second - cov_inv - jnp.outer(d_log_Z, d_log_Z))

    return MomentMatchResult(log_Z=log_Z, d_log_Z=d_log_Z, d2_log_Z=d2_log_Z)
