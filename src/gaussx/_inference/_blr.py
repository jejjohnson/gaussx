"""Bayesian Learning Rule (BLR) update primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import lineax as lx
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from gaussx._einx import reduce
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_solve


NaturalConvention = Literal["expfam", "precision"]


def _to_expfam(nat2, convention: NaturalConvention):
    """``eta2 = -Lambda / 2`` from either convention."""
    if convention == "expfam":
        return nat2
    if convention == "precision":
        return -0.5 * nat2
    raise ValueError(
        f"convention must be 'expfam' (eta2 = -Lambda/2) or 'precision' "
        f"(nat2 = +Lambda), got {convention!r}."
    )


def _from_expfam(eta2, convention: NaturalConvention):
    """Back to the caller's convention; inverse of `_to_expfam`."""
    return eta2 if convention == "expfam" else -2.0 * eta2


def blr_diag_update(
    nat1: Float[Array, " d"],
    nat2_diag: Float[Array, " d"],
    grad: Float[Array, " d"],
    hessian_diag: Float[Array, " d"],
    lr: float | Float[Array, ""],
    *,
    convention: NaturalConvention = "expfam",
) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
    r"""Diagonal natural parameter BLR update step.

    Computes the damped update for diagonal variational parameters:

        \mu = nat1 / (-2 \cdot nat2)
        eta2_{target} = -\tfrac{1}{2}(-hessian\_diag) = 0.5 \cdot hessian\_diag
        eta1_{target} = grad - hessian\_diag \cdot \mu
        nat1_{new} = (1 - lr) \cdot nat1 + lr \cdot eta1_{target}
        nat2_{new} = (1 - lr) \cdot nat2 + lr \cdot eta2_{target}

    where ``nat2`` (eta2) stores ``-\tfrac{1}{2} \lambda`` with
    ``\lambda = -hessian\_diag`` (diagonal precision).

    Args:
        nat1: Current natural location, shape ``(d,)``.
        nat2_diag: Current diagonal natural precision (eta2), shape ``(d,)``.
        grad: Gradient of log-likelihood, shape ``(d,)``.
        hessian_diag: Diagonal of Hessian (negative for log-concave),
            shape ``(d,)``.
        lr: Learning rate / damping factor.
        convention: Convention of ``nat2_diag`` and of the returned
            ``nat2_new``. ``"expfam"`` (default) is ``eta2 = -Lambda/2``;
            ``"precision"`` is ``nat2 = +Lambda``, the convention of
            `gaussx.newton_update` and `gaussx.cavity_distribution`.
            ``nat1`` is ``Lambda mu`` in both.

    Returns:
        Tuple ``(nat1_new, nat2_new)`` — updated natural parameters.

    Note:
        By default ``nat2`` is the exponential-family ``eta2 = -Lambda/2``,
        matching `gaussx.mean_cov_to_natural`. `gaussx.newton_update` and
        `gaussx.cavity_distribution` use ``nat2 = +Lambda`` instead; convert
        with ``nat2_plus = -2 * eta2``, or pass ``convention="precision"`` to
        read and return ``+Lambda`` directly. `gaussx.damped_natural_update`
        is linear, so it works in either convention as long as both of its
        arguments share it.
    """
    nat2_diag = _to_expfam(nat2_diag, convention)
    # Current mean from natural parameters
    mu = nat1 / (-2.0 * nat2_diag)

    # Target natural parameters from Newton step
    nat1_target = grad - hessian_diag * mu
    nat2_target = -0.5 * (-hessian_diag)  # eta2 = -0.5 * (-H) = 0.5 * H

    # Damped update
    nat1_new = (1.0 - lr) * nat1 + lr * nat1_target
    nat2_new = (1.0 - lr) * nat2_diag + lr * nat2_target

    return nat1_new, _from_expfam(nat2_new, convention)


def blr_full_update(
    nat1: Float[Array, " d"],
    nat2: Float[Array, "d d"],
    grad: Float[Array, " d"],
    hessian: Float[Array, "d d"],
    lr: float | Float[Array, ""],
    *,
    solver: AbstractSolverStrategy | None = None,
    convention: NaturalConvention = "expfam",
) -> tuple[Float[Array, " d"], Float[Array, "d d"]]:
    r"""Full-rank natural parameter BLR update step.

    Computes the damped update for full-rank variational parameters:

        nat2_{new} = (1 - lr) \cdot nat2 + lr \cdot (-\tfrac{1}{2}(-H))
        \mu = solve(-2 \cdot nat2, nat1)
        nat1_{new} = (1 - lr) \cdot nat1 + lr \cdot (grad - H \mu)

    Args:
        nat1: Current natural location, shape ``(d,)``.
        nat2: Current natural precision matrix (eta2), shape ``(d, d)``.
        grad: Gradient of log-likelihood, shape ``(d,)``.
        hessian: Hessian of log-likelihood (negative for log-concave),
            shape ``(d, d)``.
        lr: Learning rate / damping factor.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.
        convention: Convention of ``nat2`` and of the returned ``nat2_new``.
            ``"expfam"`` (default) is ``eta2 = -Lambda/2``; ``"precision"``
            is ``nat2 = +Lambda``, the convention of `gaussx.newton_update`
            and `gaussx.cavity_distribution`. ``nat1`` is ``Lambda mu`` in
            both.

    Returns:
        Tuple ``(nat1_new, nat2_new)`` — updated natural parameters.

    Note:
        By default ``nat2`` is the exponential-family ``eta2 = -Lambda/2``,
        matching `gaussx.mean_cov_to_natural`. `gaussx.newton_update` and
        `gaussx.cavity_distribution` use ``nat2 = +Lambda`` instead; convert
        with ``nat2_plus = -2 * eta2``, or pass ``convention="precision"`` to
        read and return ``+Lambda`` directly. `gaussx.damped_natural_update`
        is linear, so it works in either convention as long as both of its
        arguments share it.
    """
    nat2 = _to_expfam(nat2, convention)
    # Current mean from natural parameters: mu = solve(-2*eta2, eta1)
    Lambda = -2.0 * nat2
    Lambda_op = lx.MatrixLinearOperator(Lambda, lx.positive_semidefinite_tag)
    mu = dispatch_solve(Lambda_op, nat1, solver)

    # Target natural parameters from Newton step
    nat1_target = grad - hessian @ mu
    nat2_target = 0.5 * hessian  # eta2 = -0.5 * (-H) = 0.5 * H

    # Damped update
    nat1_new = (1.0 - lr) * nat1 + lr * nat1_target
    nat2_new = (1.0 - lr) * nat2 + lr * nat2_target

    return nat1_new, _from_expfam(nat2_new, convention)


def ggn_diagonal(
    jacobian: Float[Array, "N d"],
) -> Float[Array, " d"]:
    r"""Generalized Gauss-Newton diagonal approximation.

    Computes ``\mathrm{diag}(J^T J) = \sum_i J_{i,:}^2``, the diagonal
    of the Gauss-Newton Hessian approximation. Always non-negative,
    guaranteeing PSD precision updates.

    Args:
        jacobian: Jacobian matrix, shape ``(N, d)`` where N is the
            number of observations and d is the parameter dimension.

    Returns:
        Diagonal of ``J^T J``, shape ``(d,)``.
    """
    return reduce(jacobian**2, "K D -> D", "sum")


def hutchinson_hessian_diag(
    hvp_fn: Callable[[Float[Array, " d"]], Float[Array, " d"]],
    key: jax.Array,
    d: int,
    n_samples: int = 1,
    dtype: DTypeLike | None = None,
) -> Float[Array, " d"]:
    r"""Stochastic Hessian diagonal via Hutchinson with Rademacher probes.

    Estimates ``\mathrm{diag}(H)`` using the identity
    ``\mathrm{diag}(H) = E[z \odot (H z)]`` where ``z`` is a
    Rademacher random vector (entries ``\pm 1`` with equal probability).

    Args:
        hvp_fn: Hessian-vector product function ``v -> H @ v``.
        key: PRNG key for random probe generation.
        d: Dimension of the Hessian.
        n_samples: Number of random probes. More samples give better
            estimates. Default ``1``.
        dtype: Floating-point dtype for the Rademacher probes. Defaults to the
            current JAX default floating dtype.

    Returns:
        Estimated diagonal of the Hessian, shape ``(d,)``.
    """

    probe_dtype = jnp.dtype(jnp.asarray(0.0).dtype if dtype is None else dtype)

    def _single_probe(k):
        z = jnp.where(
            jax.random.bernoulli(k, shape=(d,)),
            jnp.array(1.0, dtype=probe_dtype),
            jnp.array(-1.0, dtype=probe_dtype),
        )
        return z * hvp_fn(z)

    keys = jax.random.split(key, n_samples)
    estimates = jax.vmap(_single_probe)(keys)
    return jnp.mean(estimates, axis=0)
