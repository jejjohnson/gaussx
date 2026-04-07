"""Safe Cholesky decomposition with adaptive jitter retry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx


def safe_cholesky(
    operator: lx.AbstractLinearOperator,
    *,
    initial_jitter: float = 1e-8,
    max_jitter: float = 1e-2,
    max_retries: int = 5,
    growth_factor: float = 10.0,
) -> jnp.ndarray:
    """Cholesky decomposition with adaptive jitter for near-singular matrices.

    Attempts ``jnp.linalg.cholesky(A)`` first. If the result contains NaNs
    (indicating the matrix is not numerically positive-definite), retries with
    geometrically increasing diagonal jitter: ``cholesky(A + eps * I)`` where
    *eps* starts at ``initial_jitter`` and grows by ``growth_factor`` each
    retry, up to ``max_jitter``.

    Uses ``jax.lax.while_loop`` internally so the function is fully
    JIT-compatible.

    Args:
        operator: A lineax linear operator whose dense matrix should be
            factored. Must be square.
        initial_jitter: Starting jitter magnitude added to the diagonal.
        max_jitter: Upper bound on jitter (clamped after growth).
        max_retries: Maximum number of jittered retries after the initial
            attempt.
        growth_factor: Multiplicative factor applied to jitter each retry.

    Returns:
        Lower-triangular Cholesky factor as a dense ``jnp.ndarray``.
        If all attempts fail the result will contain NaNs — this is
        intentional: JAX cannot raise exceptions inside ``jit``-traced
        code, so callers should check for NaNs when robustness matters.
    """
    A = operator.as_matrix()
    n = A.shape[0]
    eye = jnp.eye(n, dtype=A.dtype)

    # Initial (unjittered) attempt.
    L0 = jnp.linalg.cholesky(A)
    has_nan0 = jnp.any(jnp.isnan(L0))

    # State: (L, jitter, retry_count, still_bad)
    init_state = (L0, initial_jitter, 0, has_nan0)

    def _cond(state):
        _, _, count, still_bad = state
        return still_bad & (count < max_retries)

    def _body(state):
        _, eps, count, _ = state
        L = jnp.linalg.cholesky(A + eps * eye)
        has_nan = jnp.any(jnp.isnan(L))
        next_eps = jnp.minimum(eps * growth_factor, max_jitter)
        return (L, next_eps, count + 1, has_nan)

    L_final, _, _, _ = jax.lax.while_loop(_cond, _body, init_state)
    return L_final
