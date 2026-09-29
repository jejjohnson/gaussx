"""Natural gradient primitives: damped updates, PSD correction, Gauss-Newton."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._low_rank_update import LowRankUpdate


def damped_natural_update(
    nat1_old: Float[Array, " d"],
    nat2_old: lx.AbstractLinearOperator | Float[Array, "d d"],
    nat1_target: Float[Array, " d"],
    nat2_target: lx.AbstractLinearOperator | Float[Array, "d d"],
    lr: float = 1.0,
) -> tuple[Float[Array, " d"], lx.AbstractLinearOperator | Float[Array, "d d"]]:
    r"""Damped update in natural parameter space.

    The universal primitive for iterative approximate inference
    (EP, VI, Newton, PL). Every method reduces to computing target
    natural parameters and applying this damped update:

        nat1_{new} = (1 - lr) \cdot nat1_{old} + lr \cdot nat1_{target}
        nat2_{new} = (1 - lr) \cdot nat2_{old} + lr \cdot nat2_{target}

    Args:
        nat1_old: Current natural location parameter.
        nat2_old: Current natural precision-like parameter.
            Can be an array, ``BlockTriDiag``, or any linear operator.
        nat1_target: Target natural location parameter.
        nat2_target: Target natural precision-like parameter.
        lr: Learning rate / damping factor. ``lr=1`` gives the
            undamped update. Default ``1.0``.

    Returns:
        Tuple ``(nat1_new, nat2_new)`` with same types as inputs.
    """
    nat1_new = (1.0 - lr) * nat1_old + lr * nat1_target

    if isinstance(nat2_old, jax.Array) and isinstance(nat2_target, jax.Array):
        nat2_new: lx.AbstractLinearOperator | Float[Array, "d d"] = (
            1.0 - lr
        ) * nat2_old + lr * nat2_target
    elif isinstance(nat2_old, BlockTriDiag) and isinstance(nat2_target, BlockTriDiag):
        nat2_new = (1.0 - lr) * nat2_old + lr * nat2_target
    elif isinstance(nat2_old, lx.AbstractLinearOperator) and isinstance(
        nat2_target, lx.AbstractLinearOperator
    ):
        nat2_new_mat = (1.0 - lr) * nat2_old.as_matrix() + lr * nat2_target.as_matrix()
        nat2_new = lx.MatrixLinearOperator(nat2_new_mat)
    else:
        msg = "nat2_old and nat2_target must be the same type"
        raise TypeError(msg)

    return nat1_new, nat2_new


def riemannian_psd_correction(
    hessian: Float[Array, "d d"],
    site_precision: Float[Array, "d d"],
    site_covariance: Float[Array, "d d"],
    lr: float = 1.0,
) -> Float[Array, "d d"]:
    r"""Riemannian gradient correction for PSD precision updates.

    Ensures the corrected Hessian remains negative semi-definite,
    stabilizing Newton/EP/VI when the raw Hessian is indefinite:

        G = site\_precision + hessian
        H_{psd} = hessian - 0.5 \cdot lr \cdot G \cdot S \cdot G

    where ``S`` is the site covariance.

    Args:
        hessian: Raw second derivative, shape ``(d, d)``.
        site_precision: Current site precision, shape ``(d, d)``.
        site_covariance: Current site covariance, shape ``(d, d)``.
        lr: Learning rate. Default ``1.0``.

    Returns:
        Corrected Hessian, shape ``(d, d)``.
    """
    G = site_precision + hessian
    correction = G @ site_covariance @ G
    return hessian - 0.5 * lr * correction


def gauss_newton_precision(
    jacobian: Float[Array, "D_obs D_latent"],
    *,
    base: lx.AbstractLinearOperator | None = None,
) -> lx.AbstractLinearOperator:
    r"""Gauss-Newton precision matrix ``J^T J``, optionally plus a prior.

    For likelihoods with residual structure ``r(f)``, the Gauss-Newton
    Hessian approximation is ``-J_r^T J_r`` which gives precision
    ``\Lambda = J^T J`` (always PSD).

    With ``base`` (a prior precision ``\Lambda_0``), returns the posterior
    precision ``\Lambda_0 + J^T J`` as a `LowRankUpdate` on ``base``, for
    any ``D_obs``. `gaussx.solve` and `gaussx.logdet` then use the Woodbury
    identity and ``base``'s own structured solve. This is the form to use
    whenever the precision will be solved against.

    Without ``base``, ``J^T J`` has rank at most ``D_obs``. When
    ``D_{obs} < D_{latent}`` it is returned as a `LowRankUpdate` on a zero
    base, which is **singular**: it is meant as a summand or for ``mv``,
    and `gaussx.solve` / `gaussx.logdet` on it are undefined (``NaN``).
    Adding a prior afterwards (``gauss_newton_precision(J) + prior``) gives
    a plain lineax sum that is solved densely; pass ``base=prior`` instead.

    Args:
        jacobian: Jacobian of the residual, shape ``(D_obs, D_latent)``.
        base: Optional prior precision of shape ``(D_latent, D_latent)``.
            Its symmetry and positive-semidefiniteness tags carry over to
            the result.

    Returns:
        PSD precision operator of shape ``(D_latent, D_latent)``.
    """
    D_obs, D_latent = jacobian.shape
    ones = jnp.ones(D_obs, dtype=jacobian.dtype)

    if base is not None:
        if base.in_size() != D_latent or base.out_size() != D_latent:
            raise ValueError(
                f"base must be ({D_latent}, {D_latent}) to match the Jacobian's "
                f"D_latent, got ({base.out_size()}, {base.in_size()})."
            )
        # J^T J is PSD, so the sum keeps whatever the prior can claim.
        tags = frozenset(
            tag
            for query, tag in (
                (lx.is_symmetric, lx.symmetric_tag),
                (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
            )
            if query(base)
        )
        return LowRankUpdate(base=base, U=jacobian.T, d=ones, tags=tags)

    if D_obs < D_latent:
        base = lx.DiagonalLinearOperator(jnp.zeros(D_latent, dtype=jacobian.dtype))
        return LowRankUpdate(
            base=base,
            U=jacobian.T,
            d=ones,
            tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
        )

    return lx.MatrixLinearOperator(
        jacobian.T @ jacobian,
        lx.positive_semidefinite_tag,
    )
