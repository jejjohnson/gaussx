"""Bayesian Learning Rule (BLR) updates of a Gaussian's natural parameters.

A Gaussian q(x) = 𝒩(μ_x, Λ_x⁻¹) has natural parameters (Λ_x μ_x, −½ Λ_x), the
``nat1`` and ``nat2`` below. With a step size γ and the derivatives g and ∇²_x of
the log-likelihood at μ_x, one BLR step moves them towards the Newton target
(g − ∇²_x μ_x, ½ ∇²_x). ``nat2`` may also be stored as +Λ_x (``convention``).
"""

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
    """Read −½ Λ_x from either convention of the second natural parameter.

    Args:
        nat2: Second natural parameter, −½ Λ_x or +Λ_x.
        convention: ``"expfam"`` when ``nat2`` is −½ Λ_x, ``"precision"`` when
            it is +Λ_x.

    Returns:
        The exponential-family parameter −½ Λ_x.

    Raises:
        ValueError: If ``convention`` is neither of the two.
    """
    if convention == "expfam":
        return nat2
    if convention == "precision":
        # −½ Λ_x
        return -0.5 * nat2
    raise ValueError(
        f"convention must be 'expfam' (eta2 = -Lambda/2) or 'precision' "
        f"(nat2 = +Lambda), got {convention!r}."
    )


def _from_expfam(eta2, convention: NaturalConvention):
    """Write −½ Λ_x in the caller's convention; the inverse of `_to_expfam`.

    Args:
        eta2: Exponential-family parameter −½ Λ_x.
        convention: ``"expfam"`` returns it as is; anything else returns +Λ_x.

    Returns:
        The second natural parameter in the caller's convention.
    """
    # Λ_x = −2 (−½ Λ_x)
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
    """Damped BLR step on the natural parameters of a diagonal Gaussian.

        μ_x = (Λ_x μ_x) / Λ_x
        Λ_x μ_x ← (1 − γ) Λ_x μ_x + γ (g − h ⊙ μ_x)
        −½ Λ_x ← (1 − γ) (−½ Λ_x) + γ (½ h)

    with g the gradient and h the diagonal of ∇²_x of the log-likelihood.

    Algorithm:
        1. Convert ``nat2_diag`` to −½ Λ_x and recover the mean μ_x.
        2. Form the Newton target (g − h ⊙ μ_x, ½ h).
        3. Interpolate both parameters towards it with step size γ.

    Args:
        nat1: Current first natural parameter Λ_x μ_x, in both conventions.
        nat2_diag: Current diagonal second natural parameter, −½ Λ_x or +Λ_x.
        grad: Gradient g of the log-likelihood.
        hessian_diag: Diagonal h of ∇²_x of the log-likelihood, negative where
            it is log-concave.
        lr: Step size γ.
        convention: Convention of ``nat2_diag`` and of the returned parameter.
            ``"expfam"`` (default) is −½ Λ_x, matching
            `gaussx.mean_cov_to_natural`; ``"precision"`` is +Λ_x, matching
            `gaussx.newton_update` and `gaussx.cavity_distribution`.
            `gaussx.damped_natural_update` is linear, so it works in either as
            long as both of its arguments share it.

    Returns:
        Tuple ``(nat1_new, nat2_new)``, the updated natural parameters.

    Raises:
        ValueError: If ``convention`` is unknown.

    Examples:
        A unit Gaussian at its Newton target does not move:

        >>> import jax.numpy as jnp
        >>> mean = jnp.array([1.0, 2.0])
        >>> nat1, nat2 = blr_diag_update(
        ...     mean, -0.5 * jnp.ones(2), jnp.zeros(2), -jnp.ones(2), 0.5
        ... )
        >>> bool(jnp.allclose(nat1, mean)), bool(jnp.allclose(nat2, -0.5))
        (True, True)

        The precision convention reads and returns +Λ_x:

        >>> _, precision = blr_diag_update(
        ...     mean, jnp.ones(2), jnp.zeros(2), -jnp.ones(2), 0.5,
        ...     convention="precision",
        ... )
        >>> bool(jnp.allclose(precision, 1.0))
        True

        An unknown convention is rejected:

        >>> blr_diag_update(
        ...     mean, jnp.ones(2), mean, mean, 0.5, convention="other"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: convention must be 'expfam' (eta2 = -Lambda/2) or 'precision' ...
    """
    nat2_expfam = _to_expfam(nat2_diag, convention)
    # μ_x = (Λ_x μ_x) / Λ_x
    mean = nat1 / (-2.0 * nat2_expfam)

    # --- Newton target ---
    # g − h ⊙ μ_x
    nat1_target = grad - hessian_diag * mean
    # ½ h = −½ (−h)
    nat2_target = -0.5 * (-hessian_diag)

    # --- damped update ---
    # (1 − γ) Λ_x μ_x + γ (g − h ⊙ μ_x)
    nat1_new = (1.0 - lr) * nat1 + lr * nat1_target
    # (1 − γ) (−½ Λ_x) + γ (½ h)
    nat2_new = (1.0 - lr) * nat2_expfam + lr * nat2_target

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
    """Damped BLR step on the natural parameters of a full-rank Gaussian.

        μ_x = Λ_x⁻¹ (Λ_x μ_x)
        Λ_x μ_x ← (1 − γ) Λ_x μ_x + γ (g − ∇²_x μ_x)
        −½ Λ_x ← (1 − γ) (−½ Λ_x) + γ (½ ∇²_x)

    with g the gradient and ∇²_x the Hessian of the log-likelihood.

    Algorithm:
        1. Convert ``nat2`` to −½ Λ_x and solve Λ_x μ_x for the mean.
        2. Form the Newton target (g − ∇²_x μ_x, ½ ∇²_x).
        3. Interpolate both parameters towards it with step size γ.

    Args:
        nat1: Current first natural parameter Λ_x μ_x, in both conventions.
        nat2: Current second natural parameter, −½ Λ_x or +Λ_x.
        grad: Gradient g of the log-likelihood.
        hessian: Hessian ∇²_x of the log-likelihood, negative definite where it
            is log-concave.
        lr: Step size γ.
        solver: Solver strategy for Λ_x; ``None`` uses structural dispatch.
        convention: Convention of ``nat2`` and of the returned parameter.
            ``"expfam"`` (default) is −½ Λ_x, matching
            `gaussx.mean_cov_to_natural`; ``"precision"`` is +Λ_x, matching
            `gaussx.newton_update` and `gaussx.cavity_distribution`.
            `gaussx.damped_natural_update` is linear, so it works in either as
            long as both of its arguments share it.

    Returns:
        Tuple ``(nat1_new, nat2_new)``, the updated natural parameters.

    Raises:
        ValueError: If ``convention`` is unknown.

    Examples:
        A unit Gaussian at its Newton target does not move:

        >>> import jax.numpy as jnp
        >>> mean, eye = jnp.array([1.0, 2.0]), jnp.eye(2)
        >>> nat1, nat2 = blr_full_update(
        ...     mean, -0.5 * eye, jnp.zeros(2), -eye, 0.5
        ... )
        >>> bool(jnp.allclose(nat1, mean)), bool(jnp.allclose(nat2, -0.5 * eye))
        (True, True)

        The precision convention reads and returns +Λ_x:

        >>> _, precision = blr_full_update(
        ...     mean, eye, jnp.zeros(2), -eye, 0.5, convention="precision"
        ... )
        >>> bool(jnp.allclose(precision, eye))
        True
    """
    nat2_expfam = _to_expfam(nat2, convention)
    # Λ_x = −2 (−½ Λ_x)
    precision = -2.0 * nat2_expfam
    precision_op = lx.MatrixLinearOperator(precision, lx.positive_semidefinite_tag)
    # μ_x = Λ_x⁻¹ (Λ_x μ_x)
    mean = dispatch_solve(precision_op, nat1, solver)

    # --- Newton target ---
    # g − ∇²_x μ_x
    nat1_target = grad - hessian @ mean
    # ½ ∇²_x
    nat2_target = 0.5 * hessian

    # --- damped update ---
    # (1 − γ) Λ_x μ_x + γ (g − ∇²_x μ_x)
    nat1_new = (1.0 - lr) * nat1 + lr * nat1_target
    # (1 − γ) (−½ Λ_x) + γ (½ ∇²_x)
    nat2_new = (1.0 - lr) * nat2_expfam + lr * nat2_target

    return nat1_new, _from_expfam(nat2_new, convention)


def ggn_diagonal(
    jacobian: Float[Array, "N d"],
) -> Float[Array, " d"]:
    """Diagonal of the generalized Gauss-Newton matrix J_hᵀ J_h.

        [diag(J_hᵀ J_h)]_d = ∑ₙ [J_h]²_n,d

    Always non-negative, so a precision update built on it stays PSD.

    Args:
        jacobian: Jacobian J_h with one row per observation.

    Returns:
        The diagonal of J_hᵀ J_h.

    Examples:
        >>> import jax.numpy as jnp
        >>> jacobian = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> bool(jnp.allclose(ggn_diagonal(jacobian), jnp.array([10.0, 20.0])))
        True
    """
    # [J_h]²_n,d
    squared = jacobian**2
    # ∑ₙ [J_h]²_n,d
    return reduce(squared, "obs state -> state", "sum")


def hutchinson_hessian_diag(
    hvp_fn: Callable[[Float[Array, " d"]], Float[Array, " d"]],
    key: jax.Array,
    d: int,
    n_samples: int = 1,
    dtype: DTypeLike | None = None,
) -> Float[Array, " d"]:
    """Stochastic Hessian diagonal by Hutchinson with Rademacher probes.

        diag(∇²_x) = 𝔼[v ⊙ (∇²_x v)],   v_d = ±1 with equal probability

    Algorithm:
        1. Split ``key`` into one key per probe.
        2. Draw each probe v⁽ᵉ⁾ and form v⁽ᵉ⁾ ⊙ (∇²_x v⁽ᵉ⁾).
        3. Average over the probes.

    Args:
        hvp_fn: Hessian-vector product v ↦ ∇²_x v.
        key: PRNG key for the probes.
        d: Length D_x of the probes.
        n_samples: Number of probes N_ℰ.
        dtype: Floating-point dtype of the probes; ``None`` is the current JAX
            default.

    Returns:
        The estimated diagonal of ∇²_x.

    Examples:
        A single probe is already exact on a diagonal Hessian:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> diagonal = jnp.array([1.0, 2.0, 3.0])
        >>> estimate = hutchinson_hessian_diag(
        ...     lambda v: diagonal * v, jr.key(0), 3
        ... )
        >>> bool(jnp.allclose(estimate, diagonal))
        True

        Several probes average to the same diagonal:

        >>> estimate = hutchinson_hessian_diag(
        ...     lambda v: diagonal * v, jr.key(1), 3, n_samples=4
        ... )
        >>> estimate.shape
        (3,)
    """
    probe_dtype = jnp.dtype(jnp.asarray(0.0).dtype if dtype is None else dtype)

    def _single_probe(probe_key: jax.Array) -> Float[Array, " d"]:
        # v_d = ±1
        probe = jnp.where(
            jax.random.bernoulli(probe_key, shape=(d,)),
            jnp.array(1.0, dtype=probe_dtype),
            jnp.array(-1.0, dtype=probe_dtype),
        )
        # v ⊙ (∇²_x v)
        return probe * hvp_fn(probe)

    keys = jax.random.split(key, n_samples)
    estimates = jax.vmap(_single_probe)(keys)
    # jnp, not einx: einx cannot resolve an empty ensemble (N_ℰ = 0).
    # 𝔼[v ⊙ (∇²_x v)] ≈ (1 / N_ℰ) ∑ₑ v⁽ᵉ⁾ ⊙ (∇²_x v⁽ᵉ⁾)
    return jnp.mean(estimates, axis=0)
