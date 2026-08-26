"""Safe Cholesky decomposition with adaptive jitter retry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._primitives._cholesky import cholesky


def _chol_matrix(operator: lx.AbstractLinearOperator) -> Float[Array, "N N"]:
    """Route through the structured primitive and materialise the factor."""
    return cholesky(operator).as_matrix()


def safe_cholesky(
    operator: lx.AbstractLinearOperator,
    *,
    initial_jitter: float = 1e-8,
    max_jitter: float = 1e-2,
    max_retries: int = 5,
    growth_factor: float = 10.0,
) -> Float[Array, "N N"]:
    """Cholesky decomposition with adaptive jitter for near-singular matrices.

    The first attempt routes through `gaussx._primitives.cholesky`, so
    structured operators (``DiagonalLinearOperator``, ``BlockDiag``,
    ``Kronecker``, ``BlockTriDiag``) keep their structure on the happy path.
    If the result contains NaNs (the matrix is not numerically
    positive-definite), retries with geometrically increasing diagonal jitter:
    ``cholesky(A + eps * I)`` where *eps* starts at ``initial_jitter`` and
    grows by ``growth_factor`` each retry, up to ``max_jitter``. Jittering in
    general destroys structure, so retries operate on the dense matrix form.

    The retry loop is a ``jax.lax.fori_loop`` with static bounds
    (``max_retries`` is a Python int), so the function is both
    JIT-compatible and reverse-mode differentiable — ``lax.while_loop``
    would forbid the latter. Each iteration is a ``lax.cond`` that becomes
    a no-op once the factor is clean, so a matrix that succeeds on attempt
    *k* still pays for exactly *k* factorisations.

    Args:
        operator: A lineax linear operator whose Cholesky factor is
            required. Must be square and positive-definite.
        initial_jitter: Starting jitter magnitude added to the diagonal.
        max_jitter: Upper bound on jitter (clamped after growth).
        max_retries: Maximum number of jittered retries after the initial
            attempt. Must be a static Python int — it sets the loop's trip
            count, so it cannot be a traced value (e.g. a jitted function
            argument).
        growth_factor: Multiplicative factor applied to jitter each retry.

    Returns:
        Lower-triangular Cholesky factor as a dense array.
        If all attempts fail the result will contain NaNs — this is
        intentional: JAX cannot raise exceptions inside ``jit``-traced
        code, so callers should check for NaNs when robustness matters.
    """
    # A traced max_retries would silently turn the fori_loop into a
    # dynamic-bound loop, which reverse-mode AD forbids — fail loudly
    # instead (int() on a tracer raises ConcretizationTypeError).
    max_retries = int(max_retries)

    # Structured first attempt — preserves operator structure where possible.
    L0 = _chol_matrix(operator)
    has_nan0 = jnp.any(jnp.isnan(L0))

    A = operator.as_matrix()
    n = A.shape[0]
    eye = jnp.eye(n, dtype=A.dtype)

    # State: (L, jitter, still_bad). The jitter must be a concrete array so
    # both cond branches carry identical avals.
    init_state = (L0, jnp.asarray(initial_jitter, dtype=A.dtype), has_nan0)

    def _retry(state):
        _, eps, _ = state
        jittered = lx.MatrixLinearOperator(A + eps * eye, lx.positive_semidefinite_tag)
        L = _chol_matrix(jittered)
        has_nan = jnp.any(jnp.isnan(L))
        next_eps = jnp.minimum(eps * growth_factor, max_jitter)
        return (L, next_eps, has_nan)

    def _body(_, state):
        _, _, still_bad = state
        return jax.lax.cond(still_bad, _retry, lambda s: s, state)

    L_final, _, _ = jax.lax.fori_loop(0, max_retries, _body, init_state)
    return L_final
