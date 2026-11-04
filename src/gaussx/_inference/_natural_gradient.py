"""Natural-gradient primitives: damped updates, PSD correction, Gauss-Newton.

The damped update interpolates natural parameters with a step size γ; the
Riemannian correction keeps the damped precision positive definite; and the two
Gauss-Newton constructors build Jᵀ Λ J with or without a stored Jacobian.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.flatten_util
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PyTree

from gaussx._einx import einsum
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._solve_frontend import as_linear_operator


def damped_natural_update(
    nat1_old: Float[Array, " d"],
    nat2_old: lx.AbstractLinearOperator | Float[Array, "d d"],
    nat1_target: Float[Array, " d"],
    nat2_target: lx.AbstractLinearOperator | Float[Array, "d d"],
    lr: float | Float[Array, ""] = 1.0,
) -> tuple[Float[Array, " d"], lx.AbstractLinearOperator | Float[Array, "d d"]]:
    """Damped update in natural-parameter space.

        nat1 ← (1 − γ) nat1 + γ nat1_target
        nat2 ← (1 − γ) nat2 + γ nat2_target

    The universal primitive of iterative approximate inference (EP, VI, Newton,
    PL): every method reduces to computing target natural parameters and applying
    this update.

    The update is linear, so it is convention-agnostic: it works with the +Λ_x of
    `gaussx.newton_update` and `gaussx.cavity_distribution` and with the −½ Λ_x
    of `gaussx.blr_diag_update` and `gaussx.mean_cov_to_natural` alike. Both
    arguments must use the same one; mixing them silently flips the sign and
    halves the precision.

    Args:
        nat1_old: Current first natural parameter.
        nat2_old: Current second natural parameter: an array, a `BlockTriDiag`
            or any linear operator.
        nat1_target: Target first natural parameter.
        nat2_target: Target second natural parameter.
        lr: Step size γ; 1 is the undamped update.

    Returns:
        Tuple ``(nat1_new, nat2_new)``. Arrays stay arrays and two `BlockTriDiag`
        stay one; any other pair of operators comes back as a dense operator.

    Raises:
        TypeError: If one of ``nat2_old`` and ``nat2_target`` is an operator and
            the other an array.

    Examples:
        Halfway between two array parameters:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> nat1, nat2 = damped_natural_update(
        ...     jnp.zeros(2), jnp.eye(2), jnp.ones(2), 3.0 * jnp.eye(2), lr=0.5
        ... )
        >>> bool(jnp.allclose(nat1, 0.5)), bool(jnp.allclose(nat2, 2.0 * jnp.eye(2)))
        (True, True)

        Operators come back as an operator:

        >>> unit = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> _, nat2 = damped_natural_update(jnp.zeros(2), unit, jnp.ones(2), unit)
        >>> bool(jnp.allclose(nat2.as_matrix(), jnp.eye(2)))
        True

        Mixing an array with an operator is rejected:

        >>> damped_natural_update(
        ...     jnp.zeros(2), jnp.eye(2), jnp.ones(2), unit
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TypeError: nat2_old and nat2_target must both be arrays or both be linear ...
    """
    # (1 − γ) nat1 + γ nat1_target
    nat1_new = (1.0 - lr) * nat1_old + lr * nat1_target

    # Anything that is not an operator is an array-like (NumPy arrays and
    # nested lists included); normalise those before dispatching on type.
    if not isinstance(nat2_old, lx.AbstractLinearOperator):
        nat2_old = jnp.asarray(nat2_old)
    if not isinstance(nat2_target, lx.AbstractLinearOperator):
        nat2_target = jnp.asarray(nat2_target)

    if isinstance(nat2_old, jax.Array) and isinstance(nat2_target, jax.Array):
        # (1 − γ) nat2 + γ nat2_target
        nat2_new: lx.AbstractLinearOperator | Float[Array, "d d"] = (
            1.0 - lr
        ) * nat2_old + lr * nat2_target
    elif isinstance(nat2_old, BlockTriDiag) and isinstance(nat2_target, BlockTriDiag):
        # (1 − γ) nat2 + γ nat2_target, keeping the block structure
        nat2_new = (1.0 - lr) * nat2_old + lr * nat2_target
    elif isinstance(nat2_old, lx.AbstractLinearOperator) and isinstance(
        nat2_target, lx.AbstractLinearOperator
    ):
        # (1 − γ) nat2 + γ nat2_target
        nat2_matrix = (1.0 - lr) * nat2_old.as_matrix() + lr * nat2_target.as_matrix()
        nat2_new = lx.MatrixLinearOperator(nat2_matrix)
    else:
        msg = (
            "nat2_old and nat2_target must both be arrays or both be linear "
            f"operators, got {type(nat2_old).__name__} and "
            f"{type(nat2_target).__name__}."
        )
        raise TypeError(msg)

    return nat1_new, nat2_new


def riemannian_psd_correction(
    hessian: Float[Array, "d d"],
    site_precision: Float[Array, "d d"],
    site_covariance: Float[Array, "d d"],
    lr: float | Float[Array, ""] = 1.0,
) -> Float[Array, "d d"]:
    """Riemannian gradient correction for PSD precision updates.

        C = Λ_site + ∇²_x
        ∇²_x,psd = ∇²_x − ½ γ C Σ_site C

    Stabilises Newton, EP and VI when the raw Hessian ∇²_x is indefinite.

    The corrected Hessian itself is **not** guaranteed to be negative
    semi-definite: with ∇²_x = diag(−1, 2), Λ_site = I and γ = 0.1 it has
    eigenvalues [−1, 1.55]. What the correction guarantees is that the damped
    precision update stays positive definite. With Σ_site = Λ_site⁻¹,

        (1 − γ) Λ_site − γ ∇²_x,psd
            = ½ Λ_site + ½ (I − γ C Σ_site) Λ_site (I − γ Σ_site C)

    which follows by expanding both sides to Λ_site − γ C + (γ² / 2) C Σ_site C.
    The right-hand side is a positive definite matrix plus a positive
    semi-definite one.

    Args:
        hessian: Raw second derivative ∇²_x.
        site_precision: Current site precision Λ_site.
        site_covariance: Current site covariance Σ_site.
        lr: Step size γ.

    Returns:
        The corrected Hessian ∇²_x,psd.

    Examples:
        The case above:

        >>> import jax.numpy as jnp
        >>> hessian = jnp.diag(jnp.array([-1.0, 2.0]))
        >>> corrected = riemannian_psd_correction(
        ...     hessian, jnp.eye(2), jnp.eye(2), lr=0.1
        ... )
        >>> bool(jnp.allclose(jnp.diag(corrected), jnp.array([-1.0, 1.55])))
        True

        At Λ_site = −∇²_x there is nothing to correct:

        >>> corrected = riemannian_psd_correction(-jnp.eye(2), jnp.eye(2), jnp.eye(2))
        >>> bool(jnp.allclose(corrected, -jnp.eye(2)))
        True
    """
    # C = Λ_site + ∇²_x
    gap = site_precision + hessian
    # C Σ_site
    gap_covariance = gap @ site_covariance
    # C Σ_site C
    correction = gap_covariance @ gap
    # ∇²_x − ½ γ C Σ_site C
    return hessian - 0.5 * lr * correction


def gauss_newton_operator(
    residual_fn: Callable[[PyTree], Float[Array, "*obs"]],
    point: PyTree,
    *,
    noise_precision: lx.AbstractLinearOperator
    | Callable[[Float[Array, "*obs"]], Float[Array, "*obs"]]
    | None = None,
    prior_precision: lx.AbstractLinearOperator
    | Callable[[PyTree], PyTree]
    | None = None,
    jac: Literal["fwd", "bwd"] = "fwd",
) -> lx.AbstractLinearOperator:
    r"""Matrix-free Gauss-Newton precision of a residual function.

        Λ_GN = Λ_θ + J_rᵀ Λ_ε J_r,   Λ_GN v = Λ_θ v + J_rᵀ (Λ_ε (J_r v))

    J_r = ∂r/∂θ at θ\* is the Jacobian of the residual r, Λ_ε the noise precision
    and Λ_θ the prior precision. Λ_GN is the Gauss-Newton approximation of the
    Hessian of ½ rᵀ Λ_ε r + ½ θᵀ Λ_θ θ (Nocedal & Wright 2006, *Numerical
    Optimization*, ch. 10): it drops the second-order residual term, so it is
    PSD whenever Λ_ε and Λ_θ are. This is the sibling of
    `gauss_newton_precision` for a state too large for a dense Jacobian.

    Algorithm:
        1. With ``jac="fwd"``, linearise r at θ\* with `lineax.linearise` of a
           `lineax.JacobianLinearOperator`, so each matvec costs one J_r and one
           J_rᵀ product and one Λ_ε apply, and no Jacobian is formed.
        2. With ``jac="bwd"``, for an r that defines only a VJP (a ``custom_vjp``
           ODE solver, say), form J_r once by `jax.jacrev`, at one reverse pass
           per residual entry. J_r v cannot come from transposing the VJP, as
           that fails on a reverse pass that runs a ``while_loop``.
        3. Wrap v ↦ Λ_θ v + J_rᵀ (Λ_ε (J_r v)) as a PSD-tagged operator.

    Args:
        residual_fn: The residual θ ↦ r(θ).
        point: The linearisation point θ\*, an array or a pytree of arrays.
        noise_precision: Λ_ε, as an operator on r's structure or a callable
            r ↦ Λ_ε r. Default ``None`` is the identity.
        prior_precision: Λ_θ, as an operator on θ's structure or a callable
            θ ↦ Λ_θ θ. Default ``None`` is zero, which leaves Λ_GN singular when
            J_r does not have full column rank.
        jac: How to differentiate r, ``"fwd"`` or ``"bwd"``.

    Returns:
        The lazy PSD operator Λ_GN, whose ``in_structure`` is that of ``point``.

    Raises:
        ValueError: If ``jac`` is neither ``"fwd"`` nor ``"bwd"``.

    Examples:
        For a linear residual r(θ) = H θ − y, Λ_GN is exactly Λ_θ + Hᵀ Λ_ε H:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> obs_operator = jnp.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
        >>> noise = lx.DiagonalLinearOperator(jnp.array([2.0, 1.0, 1.0]))
        >>> prior = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> precision = gaussx.gauss_newton_operator(
        ...     lambda params: obs_operator @ params,
        ...     jnp.zeros(2),
        ...     noise_precision=noise,
        ...     prior_precision=prior,
        ... )
        >>> [round(float(v), 4) for v in precision.mv(jnp.array([1.0, 0.0]))]
        [4.0, 4.0]
        >>> solver = gaussx.CGSolver(rtol=1e-4, atol=1e-4)
        >>> solution = solver.solve(precision, jnp.array([4.0, 4.0]))
        >>> bool(jnp.allclose(solution, jnp.array([1.0, 0.0]), atol=1e-3))
        True
    """
    if jac not in ("fwd", "bwd"):
        raise ValueError(f"jac must be 'fwd' or 'bwd', got {jac!r}.")
    tangent, adjoint = (
        _matrix_free_jacobian(residual_fn, point)
        if jac == "fwd"
        else _dense_jacobian(residual_fn, point)
    )

    def matvec(direction: PyTree) -> PyTree:
        # J_r v
        observed = tangent(direction)
        # Λ_ε J_r v
        weighted = _apply_precision(noise_precision, observed)
        # J_rᵀ Λ_ε J_r v
        data_term = adjoint(weighted)
        if prior_precision is None:
            return data_term
        # Λ_θ v
        prior_term = _apply_precision(prior_precision, direction)
        # Λ_θ v + J_rᵀ Λ_ε J_r v
        return jax.tree.map(jnp.add, prior_term, data_term)

    structure = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), point
    )
    return as_linear_operator(
        matvec, in_structure=structure, positive_semidefinite=True
    )


def _matrix_free_jacobian(
    residual_fn: Callable[[PyTree], Float[Array, "*obs"]],
    point: PyTree,
) -> tuple[Callable[[PyTree], Array], Callable[[Array], PyTree]]:
    """The tangent v ↦ J_r v and adjoint w ↦ J_rᵀ w, by forward-mode linearisation.

    Args:
        residual_fn: The residual θ ↦ r(θ).
        point: The linearisation point θ\\*.

    Returns:
        The pair ``(tangent, adjoint)``, which lineax caches at θ\\*.
    """
    jacobian = lx.linearise(
        lx.JacobianLinearOperator(lambda theta, _: residual_fn(theta), point, jac="fwd")
    )
    return jacobian.mv, jacobian.T.mv


def _dense_jacobian(
    residual_fn: Callable[[PyTree], Float[Array, "*obs"]],
    point: PyTree,
) -> tuple[Callable[[PyTree], Array], Callable[[Array], PyTree]]:
    """The tangent v ↦ J_r v and adjoint w ↦ J_rᵀ w, from a J_r by `jax.jacrev`.

    Args:
        residual_fn: The residual θ ↦ r(θ).
        point: The linearisation point θ\\*.

    Returns:
        The pair ``(tangent, adjoint)``.
    """
    flat_point, unravel_point = jax.flatten_util.ravel_pytree(point)
    _, unravel_residual = jax.flatten_util.ravel_pytree(residual_fn(point))

    def flat_residual(flat: Float[Array, " D"]) -> Float[Array, " M"]:
        # vec r(θ)
        return jnp.ravel(residual_fn(unravel_point(flat)))

    # J_r = ∂ vec r / ∂ vec θ, one reverse pass per residual entry
    jacobian = jax.jacrev(flat_residual)(flat_point)

    def tangent(direction: PyTree) -> Array:
        flat_direction, _ = jax.flatten_util.ravel_pytree(direction)
        # J_r v
        return unravel_residual(jacobian @ flat_direction)

    def adjoint(weighted: Array) -> PyTree:
        # J_rᵀ w
        return unravel_point(einsum(jacobian, jnp.ravel(weighted), "m d, m -> d"))

    return tangent, adjoint


def _apply_precision(
    precision: lx.AbstractLinearOperator | Callable[[PyTree], PyTree] | None,
    vector: PyTree,
) -> PyTree:
    """Apply a precision Λ to a vector v; ``None`` is the identity.

    Args:
        precision: The precision Λ: an operator, a callable matvec or ``None``.
        vector: The vector v.

    Returns:
        The product Λ v.
    """
    if precision is None:
        return vector
    if isinstance(precision, lx.AbstractLinearOperator):
        return precision.mv(vector)
    return precision(vector)


def gauss_newton_precision(
    jacobian: Float[Array, "D_obs D_latent"],
    *,
    base: lx.AbstractLinearOperator | None = None,
) -> lx.AbstractLinearOperator:
    """Gauss-Newton precision J_rᵀ J_r, optionally plus a prior precision.

        Λ_z = Λ_base + J_rᵀ J_r

    For a likelihood with residual structure r(z), the Gauss-Newton approximation
    of the Hessian ∇²_z is −J_rᵀ J_r, which gives a precision that is always PSD.

    With ``base`` (a prior precision Λ_base), returns Λ_base + J_rᵀ J_r as a
    `LowRankUpdate` on ``base``, for any D_y. `gaussx.solve` and `gaussx.logdet`
    then use the Woodbury identity and ``base``'s own structured solve. This is
    the form to use whenever the precision will be solved against.

    Without ``base``, J_rᵀ J_r has rank at most D_y. When D_y < D_z it is returned
    as a `LowRankUpdate` on a zero base, which is **singular**: it is meant as a
    summand or for ``mv``, and `gaussx.solve` and `gaussx.logdet` on it are
    undefined (``NaN``). Adding a prior afterwards
    (``gauss_newton_precision(J) + prior``) gives a plain lineax sum that is
    solved densely; pass ``base=prior`` instead.

    Args:
        jacobian: Jacobian J_r of the residual, of shape (D_y, D_z).
        base: Optional prior precision Λ_base of shape (D_z, D_z). Its symmetry
            and positive-semidefiniteness tags carry over to the result.

    Returns:
        The PSD precision operator Λ_z of shape (D_z, D_z).

    Raises:
        ValueError: If ``base`` is not of shape (D_z, D_z).

    Examples:
        More residuals than latents gives a dense matrix:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> jacobian = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> precision = gauss_newton_precision(jacobian)
        >>> expected = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        >>> bool(jnp.allclose(precision.as_matrix(), expected))
        True

        A prior keeps the low-rank structure, so the solve is by Woodbury:

        >>> prior = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> precision = gauss_newton_precision(jacobian, base=prior)
        >>> type(precision).__name__
        'LowRankUpdate'
        >>> bool(jnp.allclose(precision.as_matrix(), expected + jnp.eye(2)))
        True

        A prior of the wrong size is rejected:

        >>> gauss_newton_precision(
        ...     jacobian, base=lx.DiagonalLinearOperator(jnp.ones(3))
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: base must be (2, 2) to match the Jacobian's D_latent
    """
    obs_size, latent_size = jacobian.shape
    ones = jnp.ones(obs_size, dtype=jacobian.dtype)

    if base is not None:
        if base.in_size() != latent_size or base.out_size() != latent_size:
            raise ValueError(
                f"base must be ({latent_size}, {latent_size}) to match the "
                f"Jacobian's D_latent, got ({base.out_size()}, {base.in_size()})."
            )
        # J_rᵀ J_r is PSD, so the sum keeps whatever the prior can claim.
        tags = frozenset(
            tag
            for query, tag in (
                (lx.is_symmetric, lx.symmetric_tag),
                (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
            )
            if query(base)
        )
        # jnp, not einx: einx cannot resolve an empty Jacobian (D_y = 0).
        # J_rᵀ
        jacobian_adjoint = jnp.matrix_transpose(jacobian)
        # Λ_base + J_rᵀ J_r
        return LowRankUpdate(base=base, U=jacobian_adjoint, d=ones, tags=tags)

    if obs_size < latent_size:
        zero_base = lx.DiagonalLinearOperator(
            jnp.zeros(latent_size, dtype=jacobian.dtype)
        )
        # jnp, not einx: einx cannot resolve an empty Jacobian (D_y = 0).
        # J_rᵀ
        jacobian_adjoint = jnp.matrix_transpose(jacobian)
        # 0 + J_rᵀ J_r
        return LowRankUpdate(
            base=zero_base,
            U=jacobian_adjoint,
            d=ones,
            tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
        )

    # jnp, not einx: einx cannot resolve an empty Jacobian (D_z = 0), and its
    # contraction differs from this product in the last bit.
    # J_rᵀ
    jacobian_adjoint = jnp.matrix_transpose(jacobian)
    # J_rᵀ J_r
    gauss_newton = jacobian_adjoint @ jacobian
    return lx.MatrixLinearOperator(gauss_newton, lx.positive_semidefinite_tag)
