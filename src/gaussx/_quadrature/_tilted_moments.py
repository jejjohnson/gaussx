"""EP tilted moments of a scalar site under any point-based rule."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._quadrature._gauss_hermite import GaussHermiteIntegrator
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._moment_match import _tilted_weights
from gaussx._quadrature._types import GaussianState


def ep_tilted_moments(
    log_lik_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    cav_mean: Float[Array, " *batch"],
    cav_var: Float[Array, " *batch"],
    *,
    order: int | None = None,
    integrator: AbstractIntegrator | None = None,
    power: float = 1.0,
) -> tuple[Float[Array, " *batch"], Float[Array, " *batch"]]:
    r"""Moments of the scalar EP tilted distribution, elementwise over a batch.

    For each cavity ``N(f | m, v)`` computes the mean and variance of the
    tilted distribution

    $$
    \tilde p(f) \propto p(y \mid f)^{a}\, \mathcal{N}(f \mid m, v)
    $$

    with the points ``f_i`` and mean weights ``w_i`` of a point-based rule
    applied to the cavity:

    $$
    \tilde w_i = w_i\, p(y \mid f_i)^{a}, \quad
    m_{\mathrm{tilt}} = \frac{\sum_i \tilde w_i f_i}{\sum_i \tilde w_i}, \quad
    v_{\mathrm{tilt}} = \frac{\sum_i \tilde w_i (f_i - m_{\mathrm{tilt}})^2}
        {\sum_i \tilde w_i}.
    $$

    These are the same tilted moments as ``m + v g`` and ``v + v H v`` from
    `gaussx.moment_match` (Stein's lemma), and share its log-sum-exp shifted
    weights. The default rule is Gauss-Hermite of ``order`` points, the
    behaviour before ``integrator`` existed.

    Pseudocode (per batch element):

    ```text
    if v <= 0: return m, v                      # invalid site, passed through
    f, w = integrator.points_and_weights(N(m, v))
    w~ = w * exp(a * log_lik(f) - max(a * log_lik(f)))
    m_t = sum(w~ f) / sum(w~);  v_t = sum(w~ (f - m_t)^2) / sum(w~)
    return m_t, max(v_t, eps * v)
    ```

    A site whose cavity variance is not positive -- EP's usual failure mode,
    when the site holds more precision than the current posterior -- has no
    tilted distribution. Its cavity moments are returned unchanged, so the
    moment-matched site update ``1/t_var - 1/cav_var`` is zero and nothing
    ``NaN`` reaches the site parameters; skip or damp such sites in the
    caller. Valid sites are unaffected, and gradients stay finite for both.

    Args:
        log_lik_fn: Scalar function mapping latent value ``f`` to scalar
            log-likelihood ``log p(y|f)``.
        cav_mean: Cavity means, shape ``(*batch,)``.
        cav_var: Cavity variances, shape ``(*batch,)``. Non-positive entries
            are passed through as described above.
        order: Number of Gauss-Hermite points of the default rule. Default
            ``20``. Only valid when ``integrator`` is ``None``.
        integrator: Point-based rule (e.g. `gaussx.UnscentedIntegrator`,
            `gaussx.CubatureIntegrator`) applied to each 1-D cavity. Default
            ``GaussHermiteIntegrator(order)``.
        power: Power-EP exponent ``a``; ``1.0`` is standard EP.

    Returns:
        Tuple ``(tilted_mean, tilted_var)``, each of shape ``(*batch,)``.

    Raises:
        ValueError: If both ``integrator`` and ``order`` are given.
        NotImplementedError: If ``integrator`` is not point-based.
    """
    if integrator is None:
        integrator = GaussHermiteIntegrator(order=20 if order is None else order)
    elif order is not None:
        msg = "Pass either `integrator` or `order` (Gauss-Hermite), not both."
        raise ValueError(msg)

    # Cavity (and so the rule's points) in at least float32, without
    # promoting a float32 cavity to float64 under x64. promote_types (not
    # result_type with a float32 operand) so a weakly typed float64 cavity,
    # e.g. jnp.array(0.3), stays float64.
    dtype = jnp.promote_types(jnp.result_type(cav_mean, cav_var), jnp.float32)
    cav_mean = jnp.asarray(cav_mean, dtype=dtype)
    cav_var = jnp.asarray(cav_var, dtype=dtype)

    def _quadrature_moments(mean_i: Float[Array, ""], var_i: Float[Array, ""]):
        state = GaussianState(
            mean=jnp.atleast_1d(mean_i),
            cov=lx.DiagonalLinearOperator(jnp.atleast_1d(var_i)),
        )
        chi, w_m, _ = integrator.points_and_weights(state)  # (P, 1), (P,)
        p_tilde, _ = _tilted_weights(lambda x: log_lik_fn(x[0]), chi, w_m, power)
        f_nodes = chi[:, 0]
        Z = jnp.sum(p_tilde)
        t_mean = jnp.sum(p_tilde * f_nodes) / Z
        t_var = jnp.sum(p_tilde * (f_nodes - t_mean) ** 2) / Z
        # Floor relative to the cavity, not in absolute units: a tilted
        # variance below eps * cavity variance is quadrature round-off, and
        # an absolute floor would inflate small-scale problems.
        t_var = jnp.maximum(t_var, jnp.finfo(t_var.dtype).eps * var_i)
        return t_mean, t_var

    def _compute_moments(mean_i: Float[Array, ""], var_i: Float[Array, ""]):
        # Double where: the quadrature never sees a negative variance, so
        # neither the value nor the gradient of an invalid site is NaN.
        valid = var_i > 0
        t_mean, t_var = _quadrature_moments(mean_i, jnp.where(valid, var_i, 1.0))
        return jnp.where(valid, t_mean, mean_i), jnp.where(valid, t_var, var_i)

    return jnp.vectorize(_compute_moments)(cav_mean, cav_var)
