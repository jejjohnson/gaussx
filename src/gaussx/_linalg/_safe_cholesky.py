"""Cholesky factor L Lᵀ = A + β I with an adaptive diagonal jitter β.

A is a square positive-definite N × N operator, L its lower Cholesky factor
and β ≥ 0 the jitter added to the diagonal when the plain factorisation
fails.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._cholesky import cholesky


def _chol_matrix(operator: lx.AbstractLinearOperator) -> Float[Array, "N N"]:
    """Dense Cholesky factor L of A, through the structured primitive.

    Args:
        operator: The operator A.

    Returns:
        The lower-triangular factor L as a dense matrix.
    """
    return cholesky(operator).as_matrix()


def safe_cholesky(
    operator: lx.AbstractLinearOperator,
    *,
    initial_jitter: float = 1e-8,
    max_jitter: float = 1e-2,
    max_retries: int = 5,
    growth_factor: float = 10.0,
) -> Float[Array, "N N"]:
    """Cholesky factor with adaptive diagonal jitter for near-singular matrices.

        L Lᵀ = A + β I,   β ∈ {0, β₀, g β₀, g² β₀, …},   β ≤ β_max

    The first attempt (β = 0) routes through `gaussx.cholesky`, so
    structured operators (``DiagonalLinearOperator``, ``BlockDiag``,
    ``Kronecker``, ``BlockTriDiag``) keep their structure on the happy path.
    Jitter destroys structure in general, so the retries factor the dense
    matrix.

    The retry loop is a ``jax.lax.fori_loop`` with static bounds
    (``max_retries`` is a Python int), so the function is both
    JIT-compatible and reverse-mode differentiable; ``lax.while_loop``
    would forbid the latter. Each iteration is a ``lax.cond`` that becomes
    a no-op once the factor is clean, so a matrix that succeeds on attempt
    k still pays for exactly k factorisations.

    Algorithm:
        1. Factor A through the structured primitive.
        2. While the factor holds a ``NaN`` and retries remain, factor the
           dense A + β I and grow the jitter, β ← min(g β, β_max).

    Args:
        operator: Square positive-definite operator A whose Cholesky factor
            is wanted.
        initial_jitter: First jitter β₀ added to the diagonal.
        max_jitter: Upper bound β_max on the jitter, applied after growth.
        max_retries: Maximum number of jittered retries after the first
            attempt. It must be a static Python int, since it sets the
            loop's trip count; a traced value (a jitted function's argument,
            say) raises.
        growth_factor: Factor g that multiplies the jitter on each retry.

    Returns:
        The lower-triangular factor L as a dense matrix. If every attempt
        fails it holds ``NaN``s, on purpose: JAX cannot raise inside
        ``jit``, so callers check for ``NaN`` when robustness matters.

    Examples:
        A positive-definite matrix factors on the first attempt:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = jnp.array([[4.0, 2.0], [2.0, 3.0]])
        >>> operator = lx.MatrixLinearOperator(A, lx.positive_semidefinite_tag)
        >>> L = gaussx.safe_cholesky(operator)
        >>> bool(jnp.allclose(L @ L.T, A))
        True

        A singular matrix gets jitter instead of ``NaN``:

        >>> singular = lx.MatrixLinearOperator(
        ...     jnp.ones((2, 2)), lx.positive_semidefinite_tag
        ... )
        >>> L = gaussx.safe_cholesky(singular)
        >>> bool(jnp.all(jnp.isfinite(L)))
        True
    """
    # A traced max_retries would silently turn the fori_loop into a
    # dynamic-bound loop, which reverse-mode AD forbids; fail loudly
    # instead (int() on a tracer raises ConcretizationTypeError).
    max_retries = int(max_retries)

    # L Lᵀ = A, structured where the operator allows
    first_factor = _chol_matrix(operator)
    first_failed = jnp.any(jnp.isnan(first_factor))

    # State: (L, β, still failing). The jitter must be a concrete array so
    # both cond branches carry identical avals.
    initial_state = (
        first_factor,
        jnp.asarray(initial_jitter, dtype=first_factor.dtype),
        first_failed,
    )

    def _retry(state):
        _, jitter, _ = state
        # The dense operator is built only on the (rare) retry branch, so the
        # happy path of a structured operator never materialises it (gh-365).
        dense = operator.as_matrix()
        identity = jnp.eye(dense.shape[0], dtype=dense.dtype)
        # A + β I
        jittered = lx.MatrixLinearOperator(
            dense + jitter * identity, lx.positive_semidefinite_tag
        )
        # L Lᵀ = A + β I
        factor = _chol_matrix(jittered)
        failed = jnp.any(jnp.isnan(factor))
        # β ← min(g β, β_max)
        next_jitter = jnp.minimum(jitter * growth_factor, max_jitter)
        return (factor, next_jitter, failed)

    def _body(_, state):
        _, _, still_failing = state
        return jax.lax.cond(still_failing, _retry, lambda s: s, state)

    final_factor, _, _ = jax.lax.fori_loop(0, max_retries, _body, initial_state)
    return final_factor
