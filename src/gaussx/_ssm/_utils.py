"""Shared internal helpers for SSM operators.

These let the Kalman family accept ``Float[Array, ...]`` *or*
``lineax.AbstractLinearOperator`` for ``transition``, ``obs_model``,
``process_noise``, ``obs_noise`` while keeping the structural matvec hot
path on the operator side and dropping to dense at the (one-shot)
sandwich sites.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float


def _materialise(
    op_or_array: Float[Array, ...] | lx.AbstractLinearOperator,
) -> Float[Array, ...]:
    """Return a dense array view of an operator or pass arrays through.

    Used at sandwich / additive sites (``A P A^T``, ``H P H^T``,
    ``+ Q``, ``+ R``) where lineax operators don't compose without one
    side being materialised. Structural advantages are preserved on
    *linear* operations (matvec, downstream solve / logdet through
    ``solver=``).
    """
    if isinstance(op_or_array, lx.AbstractLinearOperator):
        return op_or_array.as_matrix()
    return op_or_array


def _matvec(
    op_or_array: Float[Array, "M N"] | lx.AbstractLinearOperator,
    vector: Float[Array, " N"],
) -> Float[Array, " M"]:
    """Apply ``A x`` whether ``A`` is a raw array or a lineax operator.

    Preserves structural matvec for ``BlockDiag`` / ``Kronecker`` /
    ``MaskedOperator`` / etc.
    """
    if isinstance(op_or_array, lx.AbstractLinearOperator):
        return op_or_array.mv(vector)
    return op_or_array @ vector


def _is_operator_input(value: object) -> bool:
    """True iff *value* is a lineax operator (and so signals TI mode)."""
    return isinstance(value, lx.AbstractLinearOperator)


def _normalise_tv_inputs(
    transition: Float[Array, ...] | lx.AbstractLinearOperator,
    obs_model: Float[Array, ...] | lx.AbstractLinearOperator,
    process_noise: Float[Array, ...] | lx.AbstractLinearOperator,
    obs_noise: Float[Array, ...] | lx.AbstractLinearOperator,
    *,
    T: int,
    mask: Bool[Array, " T"] | None,
) -> tuple[
    Float[Array, "T N N"],
    Float[Array, "T M N"],
    Float[Array, "T N N"],
    Float[Array, "T M M"],
    Bool[Array, " T"],
    bool,
]:
    """Normalise the four shape-bearing args + ``mask`` for the scan body.

    Returns
    -------
    A_seq, H_seq, Q_seq, R_seq : ``(T, …)`` arrays ready to scan over.
    mask_seq : ``(T,)`` boolean array (defaults to all-True).
    is_time_invariant : True iff every shape-bearing input is either an
        operator or a 2D array (and was broadcast to ``(T, …)`` here).

    Raises
    ------
    TypeError when operator-form inputs are mixed with 3D ``(T, …)``
    arrays — that path is intentionally not supported.
    """
    has_operator = any(
        _is_operator_input(x) for x in (transition, obs_model, process_noise, obs_noise)
    )
    # Detect 3D inputs across any array-like (jax.Array, numpy.ndarray,
    # lists/tuples coerced through ``hasattr``) — not just ``jax.Array``.
    has_3d_array = any(
        not _is_operator_input(x) and getattr(x, "ndim", None) == 3
        for x in (transition, obs_model, process_noise, obs_noise)
    )
    if has_operator and has_3d_array:
        raise TypeError(
            "Time-varying (3D) inputs cannot be mixed with operator-typed "
            "(lineax) inputs. Operator form is supported only in the "
            "time-invariant signature; pass dense (T, ...) stacks for the "
            "TV path."
        )

    A_dense = _materialise(transition)
    H_dense = _materialise(obs_model)
    Q_dense = _materialise(process_noise)
    R_dense = _materialise(obs_noise)

    def _broadcast_to_T(x: Float[Array, ...], expected_ndim: int) -> Float[Array, ...]:
        if x.ndim == expected_ndim - 1:
            # 2D array → broadcast to (T, …)
            return jnp.broadcast_to(x, (T, *x.shape))
        if x.ndim == expected_ndim:
            return x
        raise ValueError(
            f"Expected ndim {expected_ndim - 1} or {expected_ndim}, got "
            f"{x.ndim} (shape={x.shape})."
        )

    A_seq = _broadcast_to_T(A_dense, expected_ndim=3)
    H_seq = _broadcast_to_T(H_dense, expected_ndim=3)
    Q_seq = _broadcast_to_T(Q_dense, expected_ndim=3)
    R_seq = _broadcast_to_T(R_dense, expected_ndim=3)

    if mask is None:
        mask_seq = jnp.ones((T,), dtype=bool)
    else:
        mask_seq = jnp.asarray(mask, dtype=bool)
        # Allow scalar broadcast for ergonomic ``mask=True`` / ``mask=False``;
        # otherwise require a 1D array of length T to give a clear error
        # before the scan rather than a confusing tracing failure.
        if mask_seq.ndim == 0:
            mask_seq = jnp.broadcast_to(mask_seq, (T,))
        elif mask_seq.shape != (T,):
            raise ValueError(
                f"mask must be a scalar or have shape ({T},); got shape "
                f"{mask_seq.shape}."
            )

    return A_seq, H_seq, Q_seq, R_seq, mask_seq, not has_3d_array
