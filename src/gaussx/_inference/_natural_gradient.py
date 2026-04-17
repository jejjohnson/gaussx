"""Natural gradient primitives: damped updates, PSD correction, Gauss-Newton."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._low_rank_update import LowRankUpdate


def damped_natural_update(
    nat1_old: Float[Array, " d"],
    nat2_old: lx.AbstractLinearOperator | Float[Array, ...],
    nat1_target: Float[Array, " d"],
    nat2_target: lx.AbstractLinearOperator | Float[Array, ...],
    lr: float = 1.0,
) -> tuple[Float[Array, " d"], lx.AbstractLinearOperator | Float[Array, ...]]:
    r"""Damped update in natural parameter space.

    The universal primitive for iterative approximate inference
    (EP, VI, Newton, PL). Every method reduces to computing target
    natural parameters and applying this damped update::

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

    if isinstance(nat2_old, jnp.ndarray) and isinstance(nat2_target, jnp.ndarray):
        nat2_new: lx.AbstractLinearOperator | jnp.ndarray = (
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
    stabilizing Newton/EP/VI when the raw Hessian is indefinite::

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
) -> lx.AbstractLinearOperator:
    r"""Gauss-Newton precision matrix ``J^T J``.

    For likelihoods with residual structure ``r(f)``, the Gauss-Newton
    Hessian approximation is ``-J_r^T J_r`` which gives precision
    ``\Lambda = J^T J`` (always PSD).

    When ``D_{obs} < D_{latent}``, returns a :class:`~gaussx.LowRankUpdate`
    to enable efficient Woodbury-based solves downstream.

    Args:
        jacobian: Jacobian of the residual, shape ``(D_obs, D_latent)``.

    Returns:
        PSD precision operator of shape ``(D_latent, D_latent)``.
    """
    D_obs, D_latent = jacobian.shape

    if D_obs < D_latent:
        base = lx.DiagonalLinearOperator(jnp.zeros(D_latent))
        return LowRankUpdate(
            base=base,
            U=jacobian.T,
            d=jnp.ones(D_obs),
            tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
        )

    return lx.MatrixLinearOperator(
        jacobian.T @ jacobian,
        lx.positive_semidefinite_tag,
    )
