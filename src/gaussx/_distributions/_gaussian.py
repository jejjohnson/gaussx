"""Operator-level Gaussian helpers: log-density, entropy, KL and quadratic form.

A Gaussian x ~ 𝒩(μ, Σ) has mean μ ∈ ℝ^{D_x} and covariance Σ, a lineax
operator whose structure the solves and log-determinants dispatch on. The
residual is r = x − μ. A is a generic non-singular operator and β ≥ 0 a
diagonal jitter.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._primitives._inv_quad_logdet import inv_quad_logdet
from gaussx._primitives._logdet import _cholesky_logdet, _dense_psd_matrix
from gaussx._primitives._trace import trace
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import (
    AbstractLogdetStrategy,
    AbstractSolverStrategy,
    AbstractSolveStrategy,
)
from gaussx._strategies._bbmm import BBMMSolver
from gaussx._strategies._composed import ComposedSolver
from gaussx._strategies._dense import DenseSolver
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve
from gaussx._strategies._keyed import KeyedSolver


# log 2π. A Python float, not a jnp array: it is weakly typed, so it takes the
# dtype of the array it meets, and it does not freeze the default dtype that
# was active at import (x64 is often enabled after `import gaussx`).
_LOG_2PI = math.log(2.0 * math.pi)


def quadratic_form(
    operator: lx.AbstractLinearOperator,
    x: Float[Array, " N"],
    *,
    solver: AbstractSolveStrategy | None = None,
) -> Float[Array, ""]:
    """Quadratic form xᵀ A⁻¹ x through a single solve.

        q(x) = xᵀ A⁻¹ x

    Args:
        operator: Non-singular operator A.
        x: Vector x.
        solver: Solve strategy for A; ``None`` uses structural dispatch.

    Returns:
        The scalar xᵀ A⁻¹ x.

    Examples:
        A diagonal operator:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> operator = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> x = jnp.array([2.0, 2.0])
        >>> round(float(gaussx.quadratic_form(operator, x)), 4)
        3.0

        A dense PSD operator agrees with the dense inverse:

        >>> matrix = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        >>> dense = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> expected = x @ jnp.linalg.solve(matrix, x)
        >>> bool(jnp.allclose(gaussx.quadratic_form(dense, x), expected))
        True
    """
    # A⁻¹ x
    solution = dispatch_solve(operator, x, solver)
    # xᵀ A⁻¹ x
    return x @ solution


def _gaussian_log_prob_residual(
    residual: Float[Array, " N"],
    cov_operator: lx.AbstractLinearOperator,
    *,
    solver: AbstractSolverStrategy | None = None,
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """Gaussian log-density from a precomputed residual r = x − μ.

        log 𝒩(x; μ, Σ) = −½ (D_x log 2π + log det Σ + rᵀ Σ⁻¹ r)

    Algorithm:
        1. For a dense PSD Σ under structural dispatch, factor Σ = L Lᵀ once
           and read both rᵀ Σ⁻¹ r = ‖L⁻¹ r‖² and log det Σ from L.
        2. Otherwise take rᵀ Σ⁻¹ r and log det Σ from
           `_quadratic_and_logdet`: BBMM's shared pass, or the strategy's
           solve and logdet.
        3. Combine the terms.

    Args:
        residual: Residual r = x − μ.
        cov_operator: Covariance Σ.
        solver: Solver strategy for Σ; ``None`` uses structural dispatch.
        key: PRNG key for a stochastic logdet strategy's probes.

    Returns:
        The scalar log 𝒩(x; μ, Σ).
    """
    event_size = residual.shape[-1]
    matrix = _dense_psd_matrix(cov_operator)
    if matrix is not None and _uses_structural_dispatch(cov_operator, solver):
        # Factor once (gh-329): one Cholesky gives both the quadratic form
        # and the log-determinant, where a solve plus a logdet factor twice.
        # L Lᵀ = Σ
        factor = jnp.linalg.cholesky(matrix)
        # L⁻¹ r
        whitened_residual = jax.scipy.linalg.solve_triangular(
            factor, residual, lower=True
        )
        # log det Σ = 2 ∑_i log L_ii
        logdet = _cholesky_logdet(factor)
        # rᵀ Σ⁻¹ r = ‖L⁻¹ r‖²
        mahalanobis = whitened_residual @ whitened_residual
        # log 𝒩(x; μ, Σ) = −½ (D_x log 2π + log det Σ + rᵀ Σ⁻¹ r)
        return -0.5 * (event_size * _LOG_2PI + logdet + mahalanobis)
    # rᵀ Σ⁻¹ r and log det Σ
    mahalanobis, logdet = _quadratic_and_logdet(
        cov_operator, residual, solver=solver, key=key
    )
    # log 𝒩(x; μ, Σ) = −½ (D_x log 2π + log det Σ + rᵀ Σ⁻¹ r)
    return -0.5 * (event_size * _LOG_2PI + logdet + mahalanobis)


def _quadratic_and_logdet(
    operator: lx.AbstractLinearOperator,
    residual: Float[Array, " N"],
    *,
    solver: AbstractSolverStrategy | None = None,
    key: jax.Array | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """rᵀ Σ⁻¹ r and log det Σ, from one BBMM pass when the strategy has one.

        (rᵀ Σ⁻¹ r, log det Σ)

    Routes, in order (gh-880):

    1. A `gaussx.BBMMSolver`, a `gaussx.KeyedSolver` around one, or a
       `gaussx.ComposedSolver` whose two parts are the same `BBMMSolver`:
       `gaussx.inv_quad_logdet` with ``key``, one modified-batched-CG pass
       over [r ∣ probes]. Its ``rtol`` controls the quadratic form only; the
       probes run to breakdown.
    2. Everything else: the strategy's solve, then its logdet, as before.

    Args:
        operator: Covariance Σ.
        residual: Residual r.
        solver: Solver strategy for Σ; ``None`` uses structural dispatch.
        key: PRNG key for a stochastic logdet strategy's probes.

    Returns:
        Tuple ``(rᵀ Σ⁻¹ r, log det Σ)``.
    """
    shared = _shared_pass_strategy(solver)
    if shared is not None:
        # [r], N × 1
        column = rearrange(residual, "row -> row 1")
        # rᵀ Σ⁻¹ r and log det Σ, in one block-CG pass
        return inv_quad_logdet(operator, column, strategy=shared, key=key)
    # Σ⁻¹ r
    precision_residual = dispatch_solve(operator, residual, solver)
    # rᵀ Σ⁻¹ r
    mahalanobis = residual @ precision_residual
    # log det Σ
    logdet = dispatch_logdet(operator, solver, key=key)
    return mahalanobis, logdet


def _shared_pass_strategy(
    solver: AbstractSolverStrategy | None,
) -> AbstractSolverStrategy | None:
    """The strategy to give `gaussx.inv_quad_logdet`, if ``solver`` shares a pass.

    Args:
        solver: The strategy, or ``None``.

    Returns:
        ``solver`` for a `gaussx.BBMMSolver` or a `gaussx.KeyedSolver` around
        one; the `BBMMSolver` of a `gaussx.ComposedSolver` whose solve and
        logdet parts are the same one; ``None`` otherwise.
    """
    if isinstance(solver, BBMMSolver):
        return solver
    if isinstance(solver, KeyedSolver) and isinstance(solver.strategy, BBMMSolver):
        return solver
    if (
        isinstance(solver, ComposedSolver)
        and isinstance(solver.logdet_strategy, BBMMSolver)
        and _same_strategy(solver.solve_strategy, solver.logdet_strategy)
    ):
        return solver.logdet_strategy
    return None


def _same_strategy(
    first: AbstractSolveStrategy,
    second: AbstractLogdetStrategy,
) -> bool:
    """Whether two strategies are the same object or equal pytrees.

    A static check: a comparison that depends on traced arrays counts as
    different, so under ``jit`` a composed pair built from two copies takes
    the separate route unless it is one object.

    Args:
        first: The solve strategy.
        second: The logdet strategy.

    Returns:
        ``True`` when they are the same strategy.
    """
    if first is second:
        return True
    equal = eqx.tree_equal(first, second)
    if isinstance(equal, bool):
        return equal
    try:
        return bool(equal)
    except jax.errors.TracerBoolConversionError:
        return False


def _uses_structural_dispatch(
    operator: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy | None,
) -> bool:
    """Whether ``solver`` would solve ``operator`` by `gaussx.solve` and `logdet`.

    Only then may the factor-once path stand in for it: an explicit
    iterative or stochastic strategy must keep its own semantics.

    Args:
        operator: The covariance Σ the strategy would see.
        solver: The strategy, or ``None`` for structural dispatch.

    Returns:
        ``True`` when the strategy resolves to exact dense dispatch.
    """
    if solver is None or isinstance(solver, DenseSolver):
        return True
    if isinstance(solver, KeyedSolver) and isinstance(
        solver.strategy, AbstractSolverStrategy
    ):
        return _uses_structural_dispatch(operator, solver.strategy)
    if isinstance(solver, AutoSolver):
        return isinstance(solver._get_strategy(operator), DenseSolver)
    return False


def gaussian_log_prob(
    loc: Float[Array, " N"],
    cov_operator: lx.AbstractLinearOperator,
    value: Float[Array, " N"],
    *,
    solver: AbstractSolverStrategy | None = None,
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """Multivariate normal log-density log 𝒩(x; μ, Σ) of a structured covariance.

        log 𝒩(x; μ, Σ) = −½ (D_x log 2π + log det Σ + (x − μ)ᵀ Σ⁻¹ (x − μ))

    The solve and the log-determinant dispatch on the structure of Σ, or go
    through an explicit ``solver``.

    Algorithm:
        1. Form the residual r = x − μ.
        2. For a dense PSD Σ under structural dispatch, factor Σ = L Lᵀ once
           and read rᵀ Σ⁻¹ r and log det Σ from L; otherwise solve Σ⁻¹ r and
           take log det Σ through the strategy.
        3. Combine the terms.

    Args:
        loc: Mean μ.
        cov_operator: Covariance Σ.
        value: Point x at which the density is evaluated.
        solver: Solver strategy for Σ (it needs both solve and logdet);
            ``None`` uses structural dispatch. A `gaussx.BBMMSolver` (or a
            `gaussx.KeyedSolver` around one) takes the quadratic form and
            the log-determinant from one shared pass of
            `gaussx.inv_quad_logdet`, whose ``rtol`` controls the quadratic
            form only.
        key: PRNG key for a stochastic logdet strategy's probes. ``None``
            uses the strategy's own seed, i.e. the same probes on every call
            (see `gaussx.KeyedSolver`). Ignored by exact strategies.

    Returns:
        The scalar log 𝒩(x; μ, Σ).

    Examples:
        A standard normal in three dimensions, at x = 1:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.MatrixLinearOperator(
        ...     jnp.eye(3), lx.positive_semidefinite_tag
        ... )
        >>> log_density = gaussx.gaussian_log_prob(
        ...     jnp.zeros(3), covariance, jnp.ones(3)
        ... )
        >>> round(float(log_density), 4)
        -4.2568

        A diagonal operator matches its dense matrix:

        >>> diagonal = lx.DiagonalLinearOperator(jnp.ones(3))
        >>> structured = gaussx.gaussian_log_prob(jnp.zeros(3), diagonal, jnp.ones(3))
        >>> bool(jnp.allclose(structured, log_density))
        True

        BBMM gives both terms from one shared pass, with a key:

        >>> import jax.random as jr
        >>> matrix = jnp.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 2.0]])
        >>> dense = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> exact = gaussx.gaussian_log_prob(jnp.zeros(3), dense, jnp.ones(3))
        >>> shared = gaussx.gaussian_log_prob(
        ...     jnp.zeros(3), dense, jnp.ones(3),
        ...     solver=gaussx.BBMMSolver(num_probes=16), key=jr.key(0),
        ... )
        >>> bool(jnp.abs(shared - exact) < 0.3)
        True
    """
    # r = x − μ
    residual = value - loc
    return _gaussian_log_prob_residual(residual, cov_operator, solver=solver, key=key)


def gaussian_entropy(
    cov_operator: lx.AbstractLinearOperator,
    *,
    solver: AbstractLogdetStrategy | None = None,
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """Entropy of a multivariate normal 𝒩(μ, Σ), independent of the mean.

        ℍ[𝒩(μ, Σ)] = ½ (D_x (1 + log 2π) + log det Σ)

    Args:
        cov_operator: Covariance Σ.
        solver: Logdet strategy for Σ; ``None`` uses structural dispatch.
        key: PRNG key for a stochastic logdet strategy's probes. ``None``
            uses the strategy's own seed, i.e. the same probes on every call
            (see `gaussx.KeyedSolver`). Ignored by exact strategies.

    Returns:
        The scalar entropy ℍ[𝒩(μ, Σ)].

    Examples:
        A standard normal in two dimensions:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> identity = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> round(float(gaussx.gaussian_entropy(identity)), 4)
        2.8379

        Scaling Σ by c adds ½ D_x log c:

        >>> scaled = lx.DiagonalLinearOperator(4.0 * jnp.ones(2))
        >>> difference = gaussx.gaussian_entropy(scaled) - gaussx.gaussian_entropy(
        ...     identity
        ... )
        >>> bool(jnp.allclose(difference, jnp.log(4.0)))
        True
    """
    event_size = cov_operator.in_size()
    # log det Σ
    logdet = dispatch_logdet(cov_operator, solver, key=key)
    # ℍ[𝒩(μ, Σ)] = ½ (D_x (1 + log 2π) + log det Σ)
    return 0.5 * (event_size * (1.0 + _LOG_2PI) + logdet)


def kl_standard_normal(
    m: Float[Array, " N"],
    S: lx.AbstractLinearOperator,
    *,
    solver: AbstractLogdetStrategy | None = None,
    key: jax.Array | None = None,
) -> Float[Array, ""]:
    """KL(first ‖ second): KL[𝒩(μ, Σ) ‖ 𝒩(0, I)] from the standard normal.

        KL[𝒩(μ, Σ) ‖ 𝒩(0, I)] = ½ (tr(Σ) + μᵀ μ − D_x − log det Σ)

    The special case of `gaussian_kl` with the second Gaussian standard.
    The identity needs no solve, so this is cheaper than the general form.
    Ubiquitous in variational inference as the prior KL term.

    Args:
        m: Mean μ of the first Gaussian.
        S: Covariance Σ of the first Gaussian.
        solver: Logdet strategy for Σ; ``None`` uses structural dispatch.
        key: PRNG key for a stochastic logdet strategy's probes. ``None``
            uses the strategy's own seed, i.e. the same probes on every call
            (see `gaussx.KeyedSolver`). Ignored by exact strategies.

    Returns:
        The scalar KL divergence.

    Examples:
        A unit shift of the mean costs ½:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> mean = jnp.array([1.0, 0.0])
        >>> identity = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> round(float(gaussx.kl_standard_normal(mean, identity)), 4)
        0.5

        It agrees with the general form against 𝒩(0, I):

        >>> covariance = lx.DiagonalLinearOperator(jnp.array([2.0, 0.5]))
        >>> special = gaussx.kl_standard_normal(mean, covariance)
        >>> general = gaussx.gaussian_kl(mean, covariance, jnp.zeros(2), identity)
        >>> bool(jnp.allclose(special, general))
        True

    See Also:
        `gaussian_kl`: General KL between two multivariate normals with
        arbitrary lineax covariance operators.
    """
    event_size = m.shape[-1]
    # tr(Σ)
    covariance_trace = trace(S)
    # μᵀ μ
    mean_norm_squared = m @ m
    # log det Σ
    logdet = dispatch_logdet(S, solver, key=key)
    # KL = ½ (tr(Σ) + μᵀ μ − D_x − log det Σ)
    return 0.5 * (covariance_trace + mean_norm_squared - event_size - logdet)


def add_jitter(
    operator: lx.AbstractLinearOperator,
    jitter: float = 1e-6,
) -> lx.AbstractLinearOperator:
    """Diagonal jitter A + β I for numerical stability.

        A + β I

    Args:
        operator: Square operator A.
        jitter: Jitter β added to every diagonal entry. Default ``1e-6``.

    Returns:
        A + β I as a lineax ``AddLinearOperator``, in the dtype of A.

    Examples:
        Jitter a singular matrix:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> singular = lx.MatrixLinearOperator(jnp.ones((2, 2)))
        >>> jittered = gaussx.add_jitter(singular, jitter=0.5)
        >>> expected = jnp.ones((2, 2)) + 0.5 * jnp.eye(2)
        >>> bool(jnp.allclose(jittered.as_matrix(), expected))
        True

        The jitter takes the operator's dtype:

        >>> single = lx.MatrixLinearOperator(jnp.eye(2, dtype=jnp.float32))
        >>> gaussx.add_jitter(single).as_matrix().dtype
        dtype('float32')
    """
    size = operator.in_size()
    dtype = operator.out_structure().dtype
    # β I, in the dtype of A
    jitter_operator = lx.DiagonalLinearOperator(jnp.full(size, jitter, dtype=dtype))
    # A + β I
    return operator + jitter_operator
