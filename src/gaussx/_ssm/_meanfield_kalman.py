"""Mean-field (block-diagonal) Kalman filter and RTS smoother.

The state x of length D_x splits into L independent blocks x⁽ℓ⁾ of length
d, so D_x = L d. The mean-field approximation keeps the posterior
covariance block-diagonal, so the joint filter factorises into L
independent d-state filters: O(L d³) per step instead of O(L³ d³). The
factorisation is exact when the cross-block dynamics are zero; otherwise
the off-block entries of the inputs are dropped, which is the mean-field
projection (the posterior precision keeps the same block-diagonal
sparsity).

These recipes ``vmap`` `gaussx.kalman_filter` and `gaussx.rts_smoother`
(or their parallel associative-scan variants) over the diagonal blocks;
there is no forked Kalman implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import BlockDiag
from gaussx._ssm._kalman import FilterState, kalman_filter, rts_smoother
from gaussx._ssm._parallel_kalman import (
    parallel_kalman_filter,
    parallel_rts_smoother,
)
from gaussx._ssm._utils import _materialise, _warn_unused_process_noise
from gaussx._strategies._base import AbstractSolverStrategy


def _split_diag_blocks(
    matrix: Float[Array, ...] | lx.AbstractLinearOperator,
    num_blocks: int,
    rows: int,
    cols: int,
) -> Float[Array, ...]:
    """The L diagonal blocks of a matrix, or of a stack of them over time.

    A `gaussx.BlockDiag` whose sub-operators match the requested blocking
    is split structurally (one ``as_matrix`` per block, never the full
    matrix); anything else is materialised and its diagonal blocks are
    sliced out, discarding the off-block entries. That is the mean-field
    projection.

    Args:
        matrix: A matrix, a stack of matrices over time, or an operator.
        num_blocks: Number of blocks L.
        rows: Rows per block.
        cols: Columns per block.

    Returns:
        The blocks, stacked as ``(L, rows, cols)`` for a single matrix and
        ``(L, T, rows, cols)`` for a stack over time.

    Raises:
        ValueError: If ``matrix`` is neither a matrix nor a stack.
    """
    if (
        isinstance(matrix, BlockDiag)
        and len(matrix.operators) == num_blocks
        and all(
            op.out_size() == rows and op.in_size() == cols for op in matrix.operators
        )
    ):
        return jnp.stack([op.as_matrix() for op in matrix.operators])
    dense = _materialise(matrix)
    if dense.ndim == 2:
        grid = rearrange(
            dense,
            "(block_row row) (block_col col) -> block_row block_col row col",
            block_row=num_blocks,
            block_col=num_blocks,
        )
    elif dense.ndim == 3:
        grid = rearrange(
            dense,
            "t (block_row row) (block_col col) -> block_row block_col t row col",
            block_row=num_blocks,
            block_col=num_blocks,
        )
    else:
        raise ValueError(f"Expected ndim 2 or 3, got {dense.ndim}.")
    block_index = jnp.arange(num_blocks)
    return grid[block_index, block_index]


def _embed_block_diag(
    block_covs: Float[Array, "L T d d"],
) -> Float[Array, "T D D"]:
    """Assemble per-block covariance stacks into block-diagonal matrices.

    Contracting against the identity I_L places block ℓ on the diagonal and
    exact zeros everywhere off-block.

    Args:
        block_covs: Per-block covariances, one stack over time per block.

    Returns:
        The block-diagonal covariances over time.
    """
    num_blocks = block_covs.shape[0]
    block_identity = jnp.eye(num_blocks, dtype=block_covs.dtype)
    # δ_ℓℓ' Σ⁽ℓ⁾, block (ℓ, ℓ') of the block-diagonal matrix
    grid = einsum(
        block_identity,
        block_covs,
        "block_row block_col, block_row t row col -> t block_row row block_col col",
    )
    return rearrange(
        grid, "t block_row row block_col col -> t (block_row row) (block_col col)"
    )


def meanfield_kalman_filter(
    transition: Float[Array, "*T D D"] | lx.AbstractLinearOperator,
    obs_model: Float[Array, "*T M D"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "*T D D"] | lx.AbstractLinearOperator,
    obs_noise: Float[Array, "*T M M"] | lx.AbstractLinearOperator,
    observations: Float[Array, "T M"],
    init_mean: Float[Array, " D"],
    init_cov: Float[Array, "D D"],
    *,
    block_size: int,
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    solver: AbstractSolverStrategy | None = None,
    parallel: bool = False,
) -> FilterState:
    """Mean-field Kalman filter over L = D_x / d independent state blocks.

        log p(y_1:T) ≈ ∑_ℓ log p(y⁽ℓ⁾_1:T)

    for the model of `gaussx.kalman_filter` with every input projected onto
    its diagonal blocks: F_t⁽ℓ⁾, H_t⁽ℓ⁾, Σ_η,t⁽ℓ⁾ and Σ_ε,t⁽ℓ⁾ for
    ℓ = 1, …, L. The L filters run under `jax.vmap`, O(T L d³) work instead
    of O(T L³ d³), at the cost of the posterior cross-block covariances.
    Dropping the off-block entries is the mean-field projection, exact when
    the true cross-block couplings are zero. The observation must split the
    same way: channel block ℓ, of length D_y / L, observes state block ℓ.

    Accepts the time-invariant, time-varying and operator forms of
    `gaussx.kalman_filter`. A `gaussx.BlockDiag` whose sub-operators match
    the blocking is split without materialising the full matrix.

    Algorithm:
        1. Split F_t, H_t, Σ_η,t, Σ_ε,t, μ_x,0, Σ_x,0, y_1:T and the mask
           into their L diagonal blocks.
        2. Run `gaussx.kalman_filter` (or `gaussx.parallel_kalman_filter`)
           on every block under `jax.vmap`.
        3. Concatenate the block means, embed the block covariances in
           block-diagonal matrices and sum the block log-likelihoods.

    Args:
        transition: Linear dynamics F_t, one for all steps, a stack over
            time or an operator.
        obs_model: Observation operator H_t, in the same forms; D_y must be
            divisible by L.
        process_noise: Model-error covariance Σ_η,t, in the same forms.
        obs_noise: Observation-error covariance Σ_ε,t, in the same forms.
        observations: Observations y_1:T, one row per step.
        init_mean: Prior mean μ_x,0 of the initial state x₀.
        init_cov: Prior covariance Σ_x,0 of x₀.
        block_size: State block length d; D_x must be divisible by it.
        mask: Optional observation mask, as in `gaussx.kalman_filter`. A
            per-step mask gates whole steps for every block; a per-channel
            mask is split per block.
        solver: Solver strategy for the per-block Σ_y,t. Sequential mode
            only; the parallel filter does not thread it through.
        parallel: When ``True``, run each block through
            `gaussx.parallel_kalman_filter` (``O(log T)`` depth on
            accelerators) instead of the sequential scan.

    Returns:
        A `gaussx.FilterState` over the full state. Its covariances are
        block-diagonal embeddings of the block covariances, with exact
        zeros off-block.

    Raises:
        ValueError: If D_x is not divisible by ``block_size``, or D_y is not
            divisible by the number of blocks L.

    Examples:
        Two independent random walks, each observed directly:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> model = (jnp.eye(2), jnp.eye(2), 0.1 * jnp.eye(2), 0.5 * jnp.eye(2))
        >>> observations = jnp.array([[0.1, -0.2], [0.3, 0.0], [0.2, 0.1]])
        >>> prior = (jnp.zeros(2), jnp.eye(2))
        >>> state = gaussx.meanfield_kalman_filter(
        ...     *model, observations, *prior, block_size=1
        ... )
        >>> state.filtered_covs.shape
        (3, 2, 2)

        With no cross-block coupling the full filter agrees:

        >>> full = gaussx.kalman_filter(*model, observations, *prior)
        >>> bool(jnp.allclose(state.log_likelihood, full.log_likelihood))
        True
    """
    state_size = init_mean.shape[0]
    obs_size = observations.shape[-1]
    if state_size % block_size != 0:
        raise ValueError(
            f"State dimension {state_size} is not divisible by block_size {block_size}."
        )
    num_blocks = state_size // block_size
    if obs_size % num_blocks != 0:
        raise ValueError(
            f"Observation dimension {obs_size} is not divisible by the number "
            f"of blocks {num_blocks}."
        )
    block_obs_size = obs_size // num_blocks

    # F_t⁽ℓ⁾, H_t⁽ℓ⁾, Σ_η,t⁽ℓ⁾, Σ_ε,t⁽ℓ⁾
    transition_blocks = _split_diag_blocks(
        transition, num_blocks, block_size, block_size
    )
    obs_model_blocks = _split_diag_blocks(
        obs_model, num_blocks, block_obs_size, block_size
    )
    process_noise_blocks = _split_diag_blocks(
        process_noise, num_blocks, block_size, block_size
    )
    obs_noise_blocks = _split_diag_blocks(
        obs_noise, num_blocks, block_obs_size, block_obs_size
    )
    # y⁽ℓ⁾_1:T
    obs_blocks = rearrange(
        observations, "t (block obs) -> block t obs", block=num_blocks
    )
    # μ_x,0⁽ℓ⁾ and Σ_x,0⁽ℓ⁾
    init_mean_blocks = rearrange(
        init_mean, "(block state) -> block state", block=num_blocks
    )
    init_cov_blocks = _split_diag_blocks(init_cov, num_blocks, block_size, block_size)

    base_filter = parallel_kalman_filter if parallel else kalman_filter
    block_inputs = (
        transition_blocks,
        obs_model_blocks,
        process_noise_blocks,
        obs_noise_blocks,
        obs_blocks,
        init_mean_blocks,
        init_cov_blocks,
    )

    channel_mask = mask is not None and jnp.ndim(mask) == 2
    if channel_mask:
        mask_blocks = rearrange(mask, "t (block obs) -> block t obs", block=num_blocks)

        def _run_masked(*inputs):
            *model_inputs, mask_block = inputs
            return base_filter(*model_inputs, mask=mask_block, solver=solver)

        block_states = jax.vmap(_run_masked)(*block_inputs, mask_blocks)
    else:

        def _run(*inputs):
            return base_filter(*inputs, mask=mask, solver=solver)

        block_states = jax.vmap(_run)(*block_inputs)

    filtered_means = rearrange(
        block_states.filtered_means, "block t state -> t (block state)"
    )
    predicted_means = rearrange(
        block_states.predicted_means, "block t state -> t (block state)"
    )
    # log p(y_1:T) ≈ ∑_ℓ log p(y⁽ℓ⁾_1:T)
    log_likelihood = jnp.sum(block_states.log_likelihood)

    return FilterState(
        filtered_means=filtered_means,
        filtered_covs=_embed_block_diag(block_states.filtered_covs),
        predicted_means=predicted_means,
        predicted_covs=_embed_block_diag(block_states.predicted_covs),
        log_likelihood=log_likelihood,
    )


def meanfield_rts_smoother(
    filter_state: FilterState,
    transition: Float[Array, "*T D D"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "*T D D"] | lx.AbstractLinearOperator | None = None,
    *,
    block_size: int,
    solver: AbstractSolverStrategy | None = None,
    parallel: bool = False,
) -> tuple[Float[Array, "T D"], Float[Array, "T D D"]]:
    """Mean-field RTS smoother over L = D_x / d independent state blocks.

        μ_x,t|T⁽ℓ⁾ = μ_x,t|t⁽ℓ⁾ + G_smooth,t⁽ℓ⁾ (μ_x,t+1|T⁽ℓ⁾ − μ_x,t+1|t⁽ℓ⁾)

    with one smoother gain G_smooth,t⁽ℓ⁾ per diagonal block ℓ. The backward
    pass paired with `meanfield_kalman_filter`: L independent
    `gaussx.rts_smoother` passes under `jax.vmap`. ``transition`` accepts
    the same forms as the filter and is projected onto its diagonal blocks.

    Algorithm:
        1. Split F_t and the filtered and predicted moments into their L
           diagonal blocks.
        2. Run `gaussx.rts_smoother` (or `gaussx.parallel_rts_smoother`) on
           every block under `jax.vmap`.
        3. Concatenate the block means and embed the block covariances in
           block-diagonal matrices.

    Args:
        filter_state: Output of `meanfield_kalman_filter`, or any
            `gaussx.FilterState` whose covariances are block-diagonal; the
            off-block entries are discarded.
        transition: Linear dynamics F_t, as in the filter.
        process_noise: Deprecated and ignored, as in `gaussx.rts_smoother`;
            it will be removed in gaussx 0.7.0.
        block_size: State block length d; D_x must be divisible by it.
        solver: Solver strategy for the per-block Σ_x,t+1|t. Sequential
            mode only.
        parallel: When ``True``, run each block through
            `gaussx.parallel_rts_smoother` instead of the sequential scan.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` holding μ_x,t|T and the
        block-diagonal Σ_x,t|T.

    Raises:
        ValueError: If D_x is not divisible by ``block_size``.

    Examples:
        Smooth a mean-field filter run of two independent random walks:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> model = (jnp.eye(2), jnp.eye(2), 0.1 * jnp.eye(2), 0.5 * jnp.eye(2))
        >>> observations = jnp.array([[0.1, -0.2], [0.3, 0.0], [0.2, 0.1]])
        >>> state = gaussx.meanfield_kalman_filter(
        ...     *model, observations, jnp.zeros(2), jnp.eye(2), block_size=1
        ... )
        >>> means, covs = gaussx.meanfield_rts_smoother(
        ...     state, jnp.eye(2), block_size=1
        ... )
        >>> means.shape, covs.shape
        ((3, 2), (3, 2, 2))

        The off-block covariances stay exactly zero:

        >>> bool(jnp.all(covs[:, 0, 1] == 0.0))
        True
    """
    state_size = filter_state.filtered_means.shape[-1]
    if state_size % block_size != 0:
        raise ValueError(
            f"State dimension {state_size} is not divisible by block_size {block_size}."
        )
    num_blocks = state_size // block_size

    _warn_unused_process_noise("meanfield_rts_smoother", process_noise)
    # F_t⁽ℓ⁾
    transition_blocks = _split_diag_blocks(
        transition, num_blocks, block_size, block_size
    )

    filtered_mean_blocks = rearrange(
        filter_state.filtered_means,
        "t (block state) -> block t state",
        block=num_blocks,
    )
    predicted_mean_blocks = rearrange(
        filter_state.predicted_means,
        "t (block state) -> block t state",
        block=num_blocks,
    )
    # The smoother never reads the log-likelihood; broadcast it so the
    # vmapped state has a block axis on every leaf.
    log_likelihood_blocks = jnp.broadcast_to(filter_state.log_likelihood, (num_blocks,))
    block_states = FilterState(
        filtered_means=filtered_mean_blocks,
        filtered_covs=_split_diag_blocks(
            filter_state.filtered_covs, num_blocks, block_size, block_size
        ),
        predicted_means=predicted_mean_blocks,
        predicted_covs=_split_diag_blocks(
            filter_state.predicted_covs, num_blocks, block_size, block_size
        ),
        log_likelihood=log_likelihood_blocks,
    )

    base_smoother = parallel_rts_smoother if parallel else rts_smoother

    def _run(block_state, transition_block):
        return base_smoother(block_state, transition_block, solver=solver)

    smoothed_mean_blocks, smoothed_cov_blocks = jax.vmap(_run)(
        block_states, transition_blocks
    )

    smoothed_means = rearrange(smoothed_mean_blocks, "block t state -> t (block state)")
    smoothed_covs = _embed_block_diag(smoothed_cov_blocks)
    return smoothed_means, smoothed_covs
