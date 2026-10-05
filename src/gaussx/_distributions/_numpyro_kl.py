"""numpyro ``kl_divergence`` registrations for the gaussx MVN classes.

Registering them lets ``numpyro.distributions.kl_divergence`` (and so
``numpyro.infer.TraceMeanField_ELBO``, which falls back to a Monte-Carlo KL
on ``NotImplementedError``) use the closed form, via
`gaussx.dist_kl_divergence` and its structural dispatch.

Importing this module performs the registrations; ``gaussx._distributions``
imports it alongside the classes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as nd
from jaxtyping import Array, Float
from numpyro.distributions.kl import kl_divergence

from gaussx._distributions._kl import dist_kl_divergence
from gaussx._distributions._mvn import MultivariateNormal
from gaussx._distributions._mvn_prec import MultivariateNormalPrecision
from gaussx._distributions._utils import _reshape_batch
from gaussx._einx import rearrange
from gaussx._primitives._inv import inv
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._dense import DenseSolver


_Gaussian = MultivariateNormal | MultivariateNormalPrecision | nd.MultivariateNormal


def _shared_covariance(d: _Gaussian) -> lx.AbstractLinearOperator | None:
    """The batch-shared covariance operator of a gaussx class, else ``None``."""
    if isinstance(d, MultivariateNormal):
        return d.cov_operator
    if isinstance(d, MultivariateNormalPrecision):
        # Lazy: the KL's structural dispatch sees the inverse of the precision.
        return inv(d.prec_operator)
    return None


def _has_exact_kl(d: _Gaussian) -> bool:
    """Whether ``d``'s solver strategy solves exactly (gh-313 review).

    `dist_kl_divergence` uses exact structural dispatch, densifying where
    there is no structural rule. That matches a distribution whose strategy
    is `DenseSolver` (or an `AutoSolver` choosing it), but not one that asked
    for a matrix-free iterative or stochastic strategy.
    """
    if isinstance(d, nd.MultivariateNormal):
        return True
    operator = d.cov_operator if isinstance(d, MultivariateNormal) else d.prec_operator
    if isinstance(d.solver, DenseSolver):
        return True
    if isinstance(d.solver, AutoSolver):
        return isinstance(d.solver._get_strategy(operator), DenseSolver)
    return False


def _flat_dense_covariance(
    d: _Gaussian, batch_shape: tuple[int, ...], n: int
) -> Float[Array, "B N N"]:
    """numpyro's (possibly batched) covariance, broadcast and flattened."""
    matrix = jnp.asarray(getattr(d, "covariance_matrix"))  # noqa: B009
    cov = jnp.broadcast_to(matrix, (*batch_shape, n, n))
    return rearrange(cov, "... i j -> (...) i j")


def _gaussian_kl(p: _Gaussian, q: _Gaussian) -> Float[Array, "*batch"]:
    """``KL(p || q)`` with numpyro's batch semantics.

    Mirrors numpyro's own MVN-MVN KL: equal event shapes, batch shapes
    broadcast together, and the result has the broadcast batch shape.
    """
    if p.event_shape != q.event_shape:
        raise ValueError(
            "Distributions must have the same event shape, but are"
            f" {p.event_shape} and {q.event_shape} for p and q, respectively."
        )
    if not (_has_exact_kl(p) and _has_exact_kl(q)):
        # numpyro's callers (TraceMeanField_ELBO) catch this and estimate the
        # KL by Monte Carlo through each distribution's own log_prob, which
        # honours a matrix-free CG/SLQ strategy; the closed form would not.
        raise NotImplementedError(
            "The closed-form KL needs exact solves: both distributions must use "
            "DenseSolver, or an AutoSolver that routes to it."
        )
    batch_shape = jnp.broadcast_shapes(p.batch_shape, q.batch_shape)
    (n,) = p.event_shape

    def flat_loc(d: _Gaussian) -> Float[Array, "B N"]:
        loc = jnp.broadcast_to(d.loc, (*batch_shape, n))
        return rearrange(loc, "... n -> (...) n")

    p_shared, q_shared = _shared_covariance(p), _shared_covariance(q)
    p_dense = (
        None if p_shared is not None else _flat_dense_covariance(p, batch_shape, n)
    )
    q_dense = (
        None if q_shared is not None else _flat_dense_covariance(q, batch_shape, n)
    )

    def covariance(shared, matrix) -> lx.AbstractLinearOperator:
        if shared is not None:
            return shared
        return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)

    def one(p_loc, q_loc, p_mat, q_mat):
        p_cov = covariance(p_shared, p_mat)
        q_cov = covariance(q_shared, q_mat)
        return dist_kl_divergence(p_loc, p_cov, q_loc, q_cov)

    in_axes = (0, 0, None if p_dense is None else 0, None if q_dense is None else 0)
    flat = jax.vmap(one, in_axes=in_axes)(flat_loc(p), flat_loc(q), p_dense, q_dense)
    return _reshape_batch(flat, batch_shape)


_GAUSSX = (MultivariateNormal, MultivariateNormalPrecision)

for _p in _GAUSSX:
    for _q in (*_GAUSSX, nd.MultivariateNormal):
        kl_divergence.register(_p, _q)(_gaussian_kl)
    kl_divergence.register(nd.MultivariateNormal, _p)(_gaussian_kl)
