"""numpyro ``kl_divergence`` registrations for the gaussx MVN classes.

Registering them lets ``numpyro.distributions.kl_divergence`` (and so
``numpyro.infer.TraceMeanField_ELBO``, which falls back to a Monte-Carlo KL
on ``NotImplementedError``) use the closed form KL[p ‖ q] of
`gaussx.gaussian_kl` and its structural dispatch, with p = 𝒩(μ_p, Σ_p) and
q = 𝒩(μ_q, Σ_q) per batch element.

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

from gaussx._distributions._kl import gaussian_kl
from gaussx._distributions._mvn import MultivariateNormal
from gaussx._distributions._mvn_base import AbstractMultivariateNormal
from gaussx._distributions._utils import _reshape_batch
from gaussx._einx import rearrange
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._dense import DenseSolver


_Gaussian = AbstractMultivariateNormal | nd.MultivariateNormal


def _shared_covariance(d: _Gaussian) -> lx.AbstractLinearOperator | None:
    """The batch-shared covariance Σ of a gaussx class, else ``None``.

    For a precision-parameterised class this is the lazy inverse Λ⁻¹, so the
    KL's structural dispatch sees the inverse of the precision.

    Args:
        d: A gaussx or numpyro Gaussian.

    Returns:
        The covariance operator Σ, or ``None`` for a numpyro Gaussian, whose
        covariance may vary over the batch.
    """
    if isinstance(d, AbstractMultivariateNormal):
        return d.covariance_operator
    return None


def _has_exact_kl(d: _Gaussian) -> bool:
    """Whether ``d``'s solver strategy solves exactly (gh-313 review).

    `gaussian_kl` uses exact structural dispatch, densifying where
    there is no structural rule. That matches a distribution whose strategy
    is `DenseSolver` (or an `AutoSolver` choosing it), but not one that asked
    for a matrix-free iterative or stochastic strategy.

    Args:
        d: A gaussx or numpyro Gaussian.

    Returns:
        ``True`` when the native operator of ``d`` is solved exactly.
    """
    if isinstance(d, nd.MultivariateNormal):
        return True
    operator = (
        d.covariance_operator
        if isinstance(d, MultivariateNormal)
        else d.precision_operator
    )
    if isinstance(d.solver, DenseSolver):
        return True
    if isinstance(d.solver, AutoSolver):
        return isinstance(d.solver._get_strategy(operator), DenseSolver)
    return False


def _flat_dense_covariance(
    d: _Gaussian, batch_shape: tuple[int, ...], n: int
) -> Float[Array, "B N N"]:
    """numpyro's (possibly batched) dense Σ, broadcast and flattened.

    Args:
        d: A numpyro Gaussian.
        batch_shape: The broadcast batch shape of the KL.
        n: Event size D_x.

    Returns:
        One dense Σ per flattened batch element.
    """
    matrix = jnp.asarray(getattr(d, "covariance_matrix"))  # noqa: B009
    covariance = jnp.broadcast_to(matrix, (*batch_shape, n, n))
    return rearrange(covariance, "... row col -> (...) row col")


def _check_event_shapes(p: _Gaussian, q: _Gaussian) -> None:
    """Check that p and q share one event shape.

    Args:
        p: The first Gaussian p.
        q: The second Gaussian q.

    Raises:
        ValueError: If the event shapes differ.
    """
    if p.event_shape != q.event_shape:
        raise ValueError(
            "Distributions must have the same event shape, but are"
            f" {p.event_shape} and {q.event_shape} for p and q, respectively."
        )


def _gaussian_kl(p: _Gaussian, q: _Gaussian) -> Float[Array, "*batch"]:
    """KL(first ‖ second): KL[p ‖ q] with numpyro's batch semantics.

    Mirrors numpyro's own MVN-MVN KL: equal event shapes, batch shapes
    broadcast together, and the result has the broadcast batch shape.

    Args:
        p: The first Gaussian p.
        q: The second Gaussian q.

    Returns:
        The KL divergence, with the broadcast batch shape.

    Raises:
        ValueError: If the event shapes differ.
        NotImplementedError: If either strategy is not an exact solve, so
            numpyro's callers fall back to a Monte-Carlo estimate.
    """
    _check_event_shapes(p, q)
    if not (_has_exact_kl(p) and _has_exact_kl(q)):
        # numpyro's callers (TraceMeanField_ELBO) catch this and estimate the
        # KL by Monte Carlo through each distribution's own log_prob, which
        # honours a matrix-free CG/SLQ strategy; the closed form would not.
        raise NotImplementedError(
            "The closed-form KL needs exact solves: both distributions must use "
            "DenseSolver, or an AutoSolver that routes to it."
        )
    return closed_form_kl(p, q)


def closed_form_kl(p: _Gaussian, q: _Gaussian) -> Float[Array, "*batch"]:
    """KL(first ‖ second): KL[p ‖ q] in closed form, whatever the solver strategies.

        KL[p ‖ q] = ½ (tr(Σ_q⁻¹ Σ_p) + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) − D_x
                       + log det Σ_q − log det Σ_p)

    Batch shapes broadcast as in numpyro's MVN-MVN KL. Backs
    `AbstractMultivariateNormal.kl` and, behind the exact-solver check, the
    numpyro ``kl_divergence`` registrations.

    Algorithm:
        1. Broadcast both means to the common batch shape and flatten it.
        2. Keep a gaussx covariance shared across the batch; broadcast and
           flatten a numpyro one.
        3. vmap `gaussx.gaussian_kl` over the flattened batch and restore its
           shape.

    Args:
        p: The first Gaussian p.
        q: The second Gaussian q.

    Returns:
        The KL divergence, with the broadcast batch shape.

    Raises:
        ValueError: If the event shapes differ.
    """
    _check_event_shapes(p, q)
    batch_shape = jnp.broadcast_shapes(p.batch_shape, q.batch_shape)
    (event_size,) = p.event_shape

    def flat_loc(d: _Gaussian) -> Float[Array, "B N"]:
        # μ, one per broadcast batch element
        mean = jnp.broadcast_to(d.loc, (*batch_shape, event_size))
        return rearrange(mean, "... n -> (...) n")

    p_shared, q_shared = _shared_covariance(p), _shared_covariance(q)
    p_dense = (
        None
        if p_shared is not None
        else _flat_dense_covariance(p, batch_shape, event_size)
    )
    q_dense = (
        None
        if q_shared is not None
        else _flat_dense_covariance(q, batch_shape, event_size)
    )

    def covariance(shared, matrix) -> lx.AbstractLinearOperator:
        if shared is not None:
            return shared
        return lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)

    def one(p_loc, q_loc, p_mat, q_mat):
        p_cov = covariance(p_shared, p_mat)
        q_cov = covariance(q_shared, q_mat)
        # KL[p ‖ q] for one batch element
        return gaussian_kl(p_loc, p_cov, q_loc, q_cov)

    in_axes = (0, 0, None if p_dense is None else 0, None if q_dense is None else 0)
    flat = jax.vmap(one, in_axes=in_axes)(flat_loc(p), flat_loc(q), p_dense, q_dense)
    return _reshape_batch(flat, batch_shape)


# One registration per pair of families: subclasses of the shared base
# (gh-360) dispatch through it.
kl_divergence.register(AbstractMultivariateNormal, AbstractMultivariateNormal)(
    _gaussian_kl
)
kl_divergence.register(AbstractMultivariateNormal, nd.MultivariateNormal)(_gaussian_kl)
kl_divergence.register(nd.MultivariateNormal, AbstractMultivariateNormal)(_gaussian_kl)
