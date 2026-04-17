"""Bayesian Learning Rule (BLR) update primitives."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from einops import reduce
from jax.typing import DTypeLike
from jaxtyping import Array, Float


def blr_diag_update(
    nat1: Float[Array, " d"],
    nat2_diag: Float[Array, " d"],
    grad: Float[Array, " d"],
    hessian_diag: Float[Array, " d"],
    lr: float,
) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
    r"""Diagonal natural parameter BLR update step.

    Computes the damped update for diagonal variational parameters::

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

    Returns:
        Tuple ``(nat1_new, nat2_new)`` — updated natural parameters.
    """
    # Current mean from natural parameters
    mu = nat1 / (-2.0 * nat2_diag)

    # Target natural parameters from Newton step
    nat1_target = grad - hessian_diag * mu
    nat2_target = -0.5 * (-hessian_diag)  # eta2 = -0.5 * (-H) = 0.5 * H

    # Damped update
    nat1_new = (1.0 - lr) * nat1 + lr * nat1_target
    nat2_new = (1.0 - lr) * nat2_diag + lr * nat2_target

    return nat1_new, nat2_new


def blr_full_update(
    nat1: Float[Array, " d"],
    nat2: Float[Array, "d d"],
    grad: Float[Array, " d"],
    hessian: Float[Array, "d d"],
    lr: float,
) -> tuple[Float[Array, " d"], Float[Array, "d d"]]:
    r"""Full-rank natural parameter BLR update step.

    Computes the damped update for full-rank variational parameters::

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

    Returns:
        Tuple ``(nat1_new, nat2_new)`` — updated natural parameters.
    """
    # Current mean from natural parameters: mu = solve(-2*eta2, eta1)
    Lambda = -2.0 * nat2
    mu = jnp.linalg.solve(Lambda, nat1)

    # Target natural parameters from Newton step
    nat1_target = grad - hessian @ mu
    nat2_target = 0.5 * hessian  # eta2 = -0.5 * (-H) = 0.5 * H

    # Damped update
    nat1_new = (1.0 - lr) * nat1 + lr * nat1_target
    nat2_new = (1.0 - lr) * nat2 + lr * nat2_target

    return nat1_new, nat2_new


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
