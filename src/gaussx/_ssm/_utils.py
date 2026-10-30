"""Shared helpers for the SSM recipes.

Operator-or-array inputs, the observation covariance, per-channel masking,
time stacking and the smoothing element. They let the Kalman family take a
dense array or a `lineax.AbstractLinearOperator` for the dynamics F, the
observation operator H and the covariances Σ_η and Σ_ε, keeping the
structural ``mv`` on the operator side and materialising only at the
sandwich and addition sites.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import einsum, rearrange, repeat
from gaussx._linalg._linalg import sandwich
from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._low_rank_update import LowRankUpdate


def _materialise(
    op_or_array: Float[Array, ...] | lx.AbstractLinearOperator,
) -> Float[Array, ...]:
    """Dense matrix of an operator; an array passes through unchanged.

    Used at the sandwich and addition sites (F Σ Fᵀ, H Σ Hᵀ, + Σ_η, + Σ_ε),
    where two lineax operators do not compose unless one side is dense. The
    linear operations (``mv``, and solves and log-determinants through
    ``solver=``) keep the operator's structure.

    Args:
        op_or_array: A lineax operator or a dense array.

    Returns:
        The dense matrix.
    """
    if isinstance(op_or_array, lx.AbstractLinearOperator):
        return op_or_array.as_matrix()
    return op_or_array


def _matvec(
    op_or_array: Float[Array, "M N"] | lx.AbstractLinearOperator,
    vector: Float[Array, " N"],
) -> Float[Array, " M"]:
    """Matrix-vector product A v for a dense array or a lineax operator.

    An operator keeps its structural ``mv`` (``BlockDiag``, ``Kronecker``,
    ``MaskedOperator``, …).

    Args:
        op_or_array: The matrix A, as an operator or a dense array.
        vector: The vector v.

    Returns:
        The product A v.
    """
    if isinstance(op_or_array, lx.AbstractLinearOperator):
        # A v, through the operator's structural mv
        return op_or_array.mv(vector)
    # A v
    return op_or_array @ vector


def _as_operator(
    op_or_array: Float[Array, "M N"] | lx.AbstractLinearOperator,
    tags: object | frozenset[object] = frozenset(),
) -> lx.AbstractLinearOperator:
    """Operator view of an array or operator; an operator passes through.

    Args:
        op_or_array: A lineax operator or a dense matrix.
        tags: lineax tags for the wrapper; applied only when wrapping an
            array.

    Returns:
        The operator.
    """
    if isinstance(op_or_array, lx.AbstractLinearOperator):
        return op_or_array
    return lx.MatrixLinearOperator(op_or_array, tags)


def _right_matmul_transpose(
    matrix: Float[Array, "N N"],
    operator: lx.AbstractLinearOperator,
) -> Float[Array, "N M"]:
    """Dense product Σ Bᵀ of a matrix and the transpose of an operator B.

    Algorithm:
        1. Apply Bᵀ to each basis vector of the output space of B, which
           stacks the rows of B without ``as_matrix``.
        2. Contract Σ with those rows.

    Args:
        matrix: Dense square matrix Σ, typically a covariance.
        operator: The operator B.

    Returns:
        The product Σ Bᵀ.
    """
    identity = jnp.eye(operator.out_size(), dtype=matrix.dtype)
    # Bᵀ e_i for each basis vector e_i: the rows of B
    operator_rows = jax.vmap(operator.T.mv)(identity)
    # Σ Bᵀ
    product = einsum(matrix, operator_rows, "row col, other col -> row other")
    return product


def _left_matmul(
    operator: lx.AbstractLinearOperator,
    matrix: Float[Array, "N K"],
) -> Float[Array, "M K"]:
    """Dense product B Σ of an operator B and a matrix, column by column.

    Args:
        operator: The operator B.
        matrix: Dense matrix Σ.

    Returns:
        The product B Σ.
    """
    # B Σ, one structural mv per column of Σ
    return jax.vmap(operator.mv, in_axes=1, out_axes=1)(matrix)


def _innovation_covariance(
    obs_operator: Float[Array, "M N"] | lx.AbstractLinearOperator,
    state_covariance: Float[Array, "N N"] | lx.AbstractLinearOperator,
    obs_noise_covariance: Float[Array, "M M"] | lx.AbstractLinearOperator,
    *,
    woodbury: bool = False,
) -> lx.AbstractLinearOperator:
    """Observation covariance Σ_y = H Σ_x Hᵀ + Σ_ε, dense or as a low-rank update.

    When H is an operator, the dense path goes through `gaussx.sandwich`, so
    matched ``Kronecker``, ``BlockDiag`` and diagonal structure survives the
    construction.

    The Woodbury path makes Σ_ε the base of a `gaussx.LowRankUpdate`, so
    later solves and log-determinants need Σ_ε invertible. If Σ_ε may be
    (numerically) singular, e.g. a ``DiagonalLinearOperator`` with zeros,
    leave ``woodbury=False`` so the full Σ_y is factored directly.

    Args:
        obs_operator: Observation operator H, as an operator or a dense
            array.
        state_covariance: State covariance Σ_x, typically the predicted
            Σ_x,t|t−1.
        obs_noise_covariance: Observation-error covariance Σ_ε.
        woodbury: When ``True``, return Σ_y as a `gaussx.LowRankUpdate`
            on Σ_ε instead of a dense PSD-tagged matrix.

    Returns:
        The observation covariance Σ_y as a PSD-tagged operator.
    """
    state_covariance_dense = _materialise(state_covariance)
    if woodbury:
        # The low-rank factors are dense; materialise H here rather than
        # at the top so the operator path below stays structural.
        obs_operator_dense = _materialise(obs_operator)
        base = (
            obs_noise_covariance
            if isinstance(obs_noise_covariance, lx.AbstractLinearOperator)
            else lx.MatrixLinearOperator(
                _materialise(obs_noise_covariance), lx.positive_semidefinite_tag
            )
        )
        # H Σ_x
        obs_projection = einsum(
            obs_operator_dense,
            state_covariance_dense,
            "obs state, state col -> obs col",
        )
        # Σ_y = Σ_ε + (H Σ_x) Hᵀ, kept as a low-rank update
        return LowRankUpdate(base, obs_projection, V=obs_operator_dense)

    if isinstance(obs_operator, lx.AbstractLinearOperator):
        state_covariance_op = lx.MatrixLinearOperator(
            state_covariance_dense, lx.positive_semidefinite_tag
        )
        # H Σ_x Hᵀ, structure-preserving
        propagated = sandwich(obs_operator, state_covariance_op).as_matrix()
    else:
        # H Σ_x
        obs_projection = einsum(
            obs_operator,
            state_covariance_dense,
            "obs state, state col -> obs col",
        )
        # H Σ_x Hᵀ
        propagated = einsum(
            obs_projection, obs_operator, "obs col, other col -> obs other"
        )
    # Σ_y = H Σ_x Hᵀ + Σ_ε
    obs_covariance = propagated + _materialise(obs_noise_covariance)
    return lx.MatrixLinearOperator(obs_covariance, lx.positive_semidefinite_tag)


def _masked_obs_inputs(
    obs_operator: Float[Array, "M N"],
    obs_noise_covariance: Float[Array, "M M"],
    obs: Float[Array, " M"],
    mask: Bool[Array, " M"],
) -> tuple[
    Float[Array, "M N"],
    Float[Array, "M M"],
    Float[Array, " M"],
    Float[Array, ""],
]:
    """Rewrite (H_t, Σ_ε,t, y_t) so that masked observation channels are inert.

        Σ_y,t = blockdiag(Σ_y,obs, I)

    Marginalising channel i out of the update is the same as keeping it with
    no information: zero row i of H_t, replace row and column i of Σ_ε,t by
    the unit basis vector, and zero y_t,i. Σ_y,t then splits into the block
    of the observed channels and an identity block, so column i of the gain
    G_t = Σ_x,t|t−1 H_tᵀ Σ_y,t⁻¹ vanishes and the masked channel cannot move
    the state. The update is exactly the row-deleted one, with no branching.

    The substitution is a `jax.numpy.where`, not a multiply, so a masked
    entry of y_t may be ``NaN`` (the usual "not measured") without
    poisoning the residual.

    Args:
        obs_operator: Observation operator H_t.
        obs_noise_covariance: Observation-error covariance Σ_ε,t.
        obs: Observation y_t.
        mask: Per-channel mask; ``True`` keeps the channel.

    Returns:
        Tuple ``(masked_obs_operator, masked_obs_noise_covariance,
        masked_obs, n_missing)``, where ``n_missing`` is the number of
        masked channels N_missing as a float, used to strip the identity
        block's term from the log-likelihood.
    """
    obs_size = obs.shape[-1]
    state_size = obs_operator.shape[-1]
    # keep row i of H_t where channel i is observed
    row_mask = repeat(mask, "obs -> obs state", state=state_size)
    masked_obs_operator = jnp.where(
        row_mask, obs_operator, jnp.zeros_like(obs_operator)
    )
    # keep entry (i, j) of Σ_ε,t where channels i and j are both observed
    pair_mask = einx.logical_and("row, col -> row col", mask, mask)
    identity = jnp.eye(obs_size, dtype=obs_noise_covariance.dtype)
    masked_obs_noise_covariance = jnp.where(pair_mask, obs_noise_covariance, identity)
    masked_obs = jnp.where(mask, obs, jnp.zeros_like(obs))
    # N_missing = D_y − ∑_i mask_i
    n_missing = obs_size - jnp.sum(mask.astype(obs.dtype))
    return masked_obs_operator, masked_obs_noise_covariance, masked_obs, n_missing


def _is_operator_input(value: object) -> bool:
    """True when ``value`` is a lineax operator, which signals the TI path.

    Args:
        value: Any model input.

    Returns:
        Whether ``value`` is a `lineax.AbstractLinearOperator`.
    """
    return isinstance(value, lx.AbstractLinearOperator)


def _normalise_tv_inputs(
    transition: Float[Array, ...] | lx.AbstractLinearOperator,
    obs_model: Float[Array, ...] | lx.AbstractLinearOperator,
    process_noise: Float[Array, ...] | lx.AbstractLinearOperator,
    obs_noise: Float[Array, ...] | lx.AbstractLinearOperator,
    *,
    T: int,
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None,
    M: int | None = None,
    materialise_transition: bool = True,
    materialise_obs: bool = True,
    materialise_obs_noise: bool = True,
) -> tuple[
    Float[Array, "T N N"],
    Float[Array, "T M N"],
    Float[Array, "T N N"],
    Float[Array, "T M M"],
    Bool[Array, " T"] | Bool[Array, "T M"],
    bool,
]:
    """Stack the model inputs F_t, H_t, Σ_η,t, Σ_ε,t and the mask over time.

    Time-invariant inputs (an operator or a matrix) are broadcast to a stack
    of length T; time-varying stacks pass through.

    Args:
        transition: Dynamics F_t, one for all steps or a stack over time.
        obs_model: Observation operator H_t, one or a stack.
        process_noise: Model-error covariance Σ_η,t, one or a stack.
        obs_noise: Observation-error covariance Σ_ε,t, one or a stack.
        T: Number of steps T.
        mask: ``None`` (every step observed), a scalar, a per-step mask of
            shape ``(T,)`` or a per-channel mask of shape ``(T, M)``.
        M: Observation length D_y, needed only to validate a per-channel
            mask. ``None`` rejects per-channel masks, which is how filters
            without a per-channel path (e.g. the PSD-projected parallel
            filter) opt out.
        materialise_transition: When ``False`` and ``transition`` is an
            operator, return an empty ``(T, 0, 0)`` placeholder instead of
            a dense stack; the caller applies the operator itself (``.mv``).
        materialise_obs: The same for ``obs_model``.
        materialise_obs_noise: The same for ``obs_noise``, e.g. when it is
            the base of a Woodbury observation covariance.

    Raises:
        TypeError: If operator inputs are mixed with time-stacked arrays;
            the operator form is supported in the time-invariant signature
            only.
        ValueError: If an input is neither a matrix nor a stack, or the
            mask has the wrong shape.

    Returns:
        Tuple ``(transitions, obs_models, process_noises, obs_noises,
        masks, is_time_invariant)``: four ``(T, …)`` stacks ready to scan
        over (or placeholders, see above), the mask as an all-``True``
        ``(T,)`` default or with its rank kept so the caller can dispatch
        on ``masks.ndim``, and whether every input was an operator or a
        matrix broadcast here.
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

    transition_dense = (
        _materialise(transition)
        if materialise_transition or not _is_operator_input(transition)
        else None
    )
    obs_model_dense = (
        _materialise(obs_model)
        if materialise_obs or not _is_operator_input(obs_model)
        else None
    )
    process_noise_dense = _materialise(process_noise)
    obs_noise_dense = (
        _materialise(obs_noise)
        if materialise_obs_noise or not _is_operator_input(obs_noise)
        else None
    )

    def _broadcast_to_T(
        x: Float[Array, ...] | None,
        expected_ndim: int,
        op: lx.AbstractLinearOperator | None = None,
    ) -> Float[Array, ...]:
        if x is None:
            if op is None:
                raise TypeError(
                    "When materialising is disabled for an operator input, "
                    "op must be provided to _broadcast_to_T."
                )
            # Operator-mode placeholder keeps the scan pytree fixed without
            # materializing the operator into a dense (T, …) array. The
            # empty array carries no matrix data; the operator is closed
            # over separately by the caller (e.g. as the base of a
            # Woodbury innovation, or via ``.mv``).
            return jnp.zeros((T, 0, 0), dtype=op.out_structure().dtype)
        if x.ndim == expected_ndim - 1:
            # one matrix for every step
            return repeat(x, "row col -> t row col", t=T)
        if x.ndim == expected_ndim:
            return x
        raise ValueError(
            f"Expected ndim {expected_ndim - 1} or {expected_ndim}, got "
            f"{x.ndim} (shape={x.shape})."
        )

    transitions = _broadcast_to_T(
        transition_dense,
        expected_ndim=3,
        op=transition if isinstance(transition, lx.AbstractLinearOperator) else None,
    )
    obs_models = _broadcast_to_T(
        obs_model_dense,
        expected_ndim=3,
        op=obs_model if isinstance(obs_model, lx.AbstractLinearOperator) else None,
    )
    process_noises = _broadcast_to_T(process_noise_dense, expected_ndim=3)
    obs_noises = _broadcast_to_T(
        obs_noise_dense,
        expected_ndim=3,
        op=obs_noise if isinstance(obs_noise, lx.AbstractLinearOperator) else None,
    )

    if mask is None:
        masks = jnp.ones((T,), dtype=bool)
    else:
        masks = jnp.asarray(mask, dtype=bool)
        # Allow scalar broadcast for ergonomic ``mask=True`` / ``mask=False``;
        # otherwise require ``(T,)`` (per-step gate) or ``(T, M)``
        # (per-channel gate) to give a clear error before the scan rather
        # than a confusing tracing failure.
        if masks.ndim == 0:
            masks = repeat(masks, " -> t", t=T)
        elif masks.ndim == 2:
            if M is None:
                raise ValueError(
                    "mask must be a scalar or have shape "
                    f"({T},); got shape {masks.shape}. Per-channel "
                    "(T, M) masks are not supported by this filter."
                )
            if masks.shape != (T, M):
                raise ValueError(
                    f"mask must be a scalar or have shape ({T},) or "
                    f"({T}, {M}); got shape {masks.shape}."
                )
        elif masks.shape != (T,):
            expected = f"({T},)" if M is None else f"({T},) or ({T}, {M})"
            raise ValueError(
                f"mask must be a scalar or have shape {expected}; got shape "
                f"{masks.shape}."
            )

    return transitions, obs_models, process_noises, obs_noises, masks, not has_3d_array


def _warn_unused_process_noise(function: str, process_noise: object) -> None:
    """Deprecate passing ``process_noise`` to an RTS smoother (gh-364).

    The RTS recurrence never reads Σ_η: the filter's predicted covariances
    Σ_x,t+1|t already include it, so accepting it suggests behaviour that
    does not exist.

    Args:
        function: Name of the smoother, for the warning message.
        process_noise: The value passed; ``None`` does not warn.
    """
    if process_noise is not None:
        warn_deprecated(
            f"{function}(..., process_noise) is deprecated and ignored: the RTS "
            "recurrence never reads it (the filter's predicted covariances "
            "already include Q). Omit it; the argument will be removed in gaussx 0.7.0."
        )


SmootherElement = tuple[
    Float[Array, "*T N N"], Float[Array, "*T N"], Float[Array, "*T N N"]
]


def _transition_stack(
    transition: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    num_steps: int,
) -> Float[Array, "T N N"]:
    """Dense dynamics F_t stacked over time, as the smoothers consume them.

    Args:
        transition: Dynamics F_t, one for all steps or a stack over time,
            as an array or a `lineax.AbstractLinearOperator`.
        num_steps: Number of steps T.

    Raises:
        TypeError: If an operator-typed ``transition`` has a time axis.
        ValueError: If ``transition`` is neither a matrix nor a stack.

    Returns:
        F_1:T, one dense matrix per step.
    """
    transition_dense = _materialise(transition)
    transition_op = (
        transition if isinstance(transition, lx.AbstractLinearOperator) else None
    )
    if transition_dense.ndim == 2:
        return repeat(transition_dense, "row col -> t row col", t=num_steps)
    if transition_dense.ndim == 3:
        if transition_op is not None:
            raise TypeError(
                "Operator-typed transition cannot have a leading time axis."
            )
        return transition_dense
    raise ValueError(f"transition must have ndim 2 or 3, got {transition_dense.ndim}.")


def _smoother_element(
    filtered_mean: Float[Array, " N"],
    filtered_covariance: Float[Array, "N N"],
    predicted_mean_next: Float[Array, " N"],
    predicted_covariance_next: Float[Array, "N N"],
    transition_next: Float[Array, "N N"],
) -> SmootherElement:
    """Smoothing element of step t < T: the backward conditional of x_t.

        G_smooth,t = Σ_x,t|t F_t+1ᵀ Σ_x,t+1|t⁻¹
        μ_back,t = μ_x,t|t − G_smooth,t μ_x,t+1|t
        Σ_back,t = Σ_x,t|t − G_smooth,t Σ_x,t+1|t G_smooth,tᵀ

    Args:
        filtered_mean: Filtered mean μ_x,t|t.
        filtered_covariance: Filtered covariance Σ_x,t|t.
        predicted_mean_next: Predicted mean μ_x,t+1|t.
        predicted_covariance_next: Predicted covariance Σ_x,t+1|t.
        transition_next: Dynamics F_t+1.

    Returns:
        The element ``(G_smooth, μ_back, Σ_back)`` of step t.
    """
    # Σ_x,t|t F_t+1ᵀ
    cross_covariance = einsum(
        filtered_covariance, transition_next, "state col, next col -> state next"
    )
    # F_t+1 Σ_x,t|t
    cross_covariance_transpose = rearrange(cross_covariance, "state next -> next state")
    # G_smooth,tᵀ = Σ_x,t+1|t⁻¹ F_t+1 Σ_x,t|t, using the symmetry of Σ_x,t+1|t
    smoother_gain_transpose = jnp.linalg.solve(
        predicted_covariance_next, cross_covariance_transpose
    )
    # G_smooth,t = Σ_x,t|t F_t+1ᵀ Σ_x,t+1|t⁻¹
    smoother_gain = rearrange(smoother_gain_transpose, "next state -> state next")
    # G_smooth,t μ_x,t+1|t
    predicted_shift = smoother_gain @ predicted_mean_next
    # μ_back,t = μ_x,t|t − G_smooth,t μ_x,t+1|t
    back_mean = filtered_mean - predicted_shift
    # G_smooth,t Σ_x,t+1|t
    gain_covariance = einsum(
        smoother_gain, predicted_covariance_next, "state next, next col -> state col"
    )
    # G_smooth,t Σ_x,t+1|t G_smooth,tᵀ
    correction = einsum(
        gain_covariance, smoother_gain, "state col, other col -> state other"
    )
    # Σ_back,t = Σ_x,t|t − G_smooth,t Σ_x,t+1|t G_smooth,tᵀ
    back_covariance = symmetrize(filtered_covariance - correction)
    return smoother_gain, back_mean, back_covariance
