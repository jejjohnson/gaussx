"""Mean-field (block-diagonal) Kalman filter and RTS smoother.

For a state that decomposes into ``L`` independent blocks of size ``d``
(total dimension ``D = L * d``), the mean-field approximation keeps the
posterior covariance block-diagonal, so the joint filter factorises into
``L`` independent ``d``-state filters — ``O(L d^3)`` per step instead of
the full filter's ``O((L d)^3)``. The factorisation is exact when the
true cross-block dynamics are zero; otherwise the off-block entries of
the inputs are dropped (the natural mean-field projection: the posterior
precision keeps the same block-diagonal sparsity).

Internally these recipes ``vmap`` the existing `gaussx.kalman_filter` /
`gaussx.rts_smoother` (or their parallel associative-scan variants) over
the diagonal blocks — no forked Kalman implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Bool, Float

from gaussx._einx import rearrange
from gaussx._operators._block_diag import BlockDiag
from gaussx._ssm._kalman import FilterState, kalman_filter, rts_smoother
from gaussx._ssm._parallel_kalman import (
    parallel_kalman_filter,
    parallel_rts_smoother,
)
from gaussx._ssm._utils import _materialise
from gaussx._strategies._base import AbstractSolverStrategy


def _split_diag_blocks(
    x: Float[Array, ...] | lx.AbstractLinearOperator,
    L: int,
    rows: int,
    cols: int,
) -> Float[Array, ...]:
    """Extract the ``L`` diagonal blocks of a (stacked) matrix or operator.

    Returns ``(L, rows, cols)`` for time-invariant inputs and
    ``(L, T, rows, cols)`` for time-varying ``(T, ...)`` stacks. A
    `gaussx.BlockDiag` whose sub-operators align with the requested
    blocking is split structurally (per-block ``as_matrix``, never
    materialising the full matrix); everything else drops to dense and
    slices out the diagonal blocks, discarding off-block entries — the
    mean-field projection.
    """
    if (
        isinstance(x, BlockDiag)
        and len(x.operators) == L
        and all(op.out_size() == rows and op.in_size() == cols for op in x.operators)
    ):
        return jnp.stack([op.as_matrix() for op in x.operators])
    dense = _materialise(x)
    if dense.ndim == 2:
        grid = rearrange(dense, "(l1 r) (l2 c) -> l1 l2 r c", l1=L, l2=L)
    elif dense.ndim == 3:
        grid = rearrange(dense, "t (l1 r) (l2 c) -> l1 l2 t r c", l1=L, l2=L)
    else:
        raise ValueError(f"Expected ndim 2 or 3, got {dense.ndim}.")
    return grid[jnp.arange(L), jnp.arange(L)]


def _embed_block_diag(covs: Float[Array, "L T d d"]) -> Float[Array, "T D D"]:
    """Assemble per-block covariance stacks into ``(T, D, D)`` block-diagonals."""
    per_step = rearrange(covs, "l t d1 d2 -> t l d1 d2")
    return jax.vmap(lambda blocks: jax.scipy.linalg.block_diag(*blocks))(per_step)


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
    r"""Mean-field Kalman filter over ``L = D // block_size`` state blocks.

    Runs ``L`` independent ``block_size``-state Kalman filters under
    `jax.vmap` — ``O(T L d^3)`` total work instead of the full filter's
    ``O(T L^3 d^3)`` — at the cost of dropping posterior cross-block
    covariances. All inputs are projected onto their diagonal blocks;
    off-block entries are discarded, which is the mean-field projection
    (exact when the true cross-block couplings are zero). The returned
    log-likelihood is the sum of the per-block log-likelihoods,

    $$\log p(y_{1:T}) \approx \sum_{\ell=1}^{L}
    \log p^{(\ell)}\bigl(y_{1:T}^{(\ell)}\bigr).$$

    Accepts the same time-invariant / time-varying / operator input
    forms as `gaussx.kalman_filter`. A `gaussx.BlockDiag` whose
    sub-operators match the blocking is split structurally without
    materialising the full ``(D, D)`` matrix. The observation dimension
    must factorise the same way: channel block ``\ell`` (of size
    ``M // L``) is observed through state block ``\ell``.

    Args:
        transition: State transition ``A``. Shape ``(D, D)``,
            ``(T, D, D)``, or `lineax.AbstractLinearOperator`.
        obs_model: Observation matrix ``H``. Shape ``(M, D)``,
            ``(T, M, D)``, or operator. ``M`` must be divisible by
            ``L``.
        process_noise: Process noise covariance ``Q``. Shape ``(D, D)``,
            ``(T, D, D)``, or operator.
        obs_noise: Observation noise covariance ``R``. Shape ``(M, M)``,
            ``(T, M, M)``, or operator.
        observations: Observed data, shape ``(T, M)``.
        init_mean: Initial state mean, shape ``(D,)``.
        init_cov: Initial state covariance, shape ``(D, D)``.
        block_size: State block size ``d``; ``D`` must be divisible by
            it.
        mask: Optional observation mask, as in `gaussx.kalman_filter`.
            A ``(T,)`` mask gates whole steps for every block; a
            ``(T, M)`` mask gates individual channels and is split
            per block.
        solver: Optional solver strategy for the per-block innovation
            solves. Sequential mode only (the parallel filter does not
            thread it through).
        parallel: When ``True``, run each block through
            `gaussx.parallel_kalman_filter` (``O(log T)`` depth on
            accelerators) instead of the sequential scan.

    Raises:
        ValueError: If ``D`` is not divisible by ``block_size``, or the
            observation dimension is not divisible by the number of
            blocks.

    Returns:
        A `gaussx.FilterState` over the full ``D``-dimensional state.
        Covariances are block-diagonal ``(T, D, D)`` embeddings of the
        per-block covariances (exact zeros off-block).
    """
    D = init_mean.shape[0]
    M = observations.shape[-1]
    d = block_size
    if D % d != 0:
        raise ValueError(f"State dimension {D} is not divisible by block_size {d}.")
    L = D // d
    if M % L != 0:
        raise ValueError(
            f"Observation dimension {M} is not divisible by the number of blocks {L}."
        )
    m = M // L

    A_blocks = _split_diag_blocks(transition, L, d, d)
    H_blocks = _split_diag_blocks(obs_model, L, m, d)
    Q_blocks = _split_diag_blocks(process_noise, L, d, d)
    R_blocks = _split_diag_blocks(obs_noise, L, m, m)
    y_blocks = rearrange(observations, "t (l m) -> l t m", l=L)
    m0_blocks = rearrange(init_mean, "(l d) -> l d", l=L)
    P0_blocks = _split_diag_blocks(init_cov, L, d, d)

    base_filter = parallel_kalman_filter if parallel else kalman_filter

    channel_mask = mask is not None and jnp.ndim(mask) == 2
    if channel_mask:
        mask_blocks = rearrange(mask, "t (l m) -> l t m", l=L)

        def _run(A_b, H_b, Q_b, R_b, y_b, m0_b, P0_b, mask_b):
            return base_filter(
                A_b, H_b, Q_b, R_b, y_b, m0_b, P0_b, mask=mask_b, solver=solver
            )

        states = jax.vmap(_run)(
            A_blocks,
            H_blocks,
            Q_blocks,
            R_blocks,
            y_blocks,
            m0_blocks,
            P0_blocks,
            mask_blocks,
        )
    else:

        def _run(A_b, H_b, Q_b, R_b, y_b, m0_b, P0_b):
            return base_filter(
                A_b, H_b, Q_b, R_b, y_b, m0_b, P0_b, mask=mask, solver=solver
            )

        states = jax.vmap(_run)(
            A_blocks, H_blocks, Q_blocks, R_blocks, y_blocks, m0_blocks, P0_blocks
        )

    return FilterState(
        filtered_means=rearrange(states.filtered_means, "l t d -> t (l d)"),
        filtered_covs=_embed_block_diag(states.filtered_covs),
        predicted_means=rearrange(states.predicted_means, "l t d -> t (l d)"),
        predicted_covs=_embed_block_diag(states.predicted_covs),
        log_likelihood=jnp.sum(states.log_likelihood),
    )


def meanfield_rts_smoother(
    filter_state: FilterState,
    transition: Float[Array, "*T D D"] | lx.AbstractLinearOperator,
    process_noise: Float[Array, "*T D D"] | lx.AbstractLinearOperator,
    *,
    block_size: int,
    solver: AbstractSolverStrategy | None = None,
    parallel: bool = False,
) -> tuple[Float[Array, "T D"], Float[Array, "T D D"]]:
    """Mean-field RTS smoother over ``L = D // block_size`` state blocks.

    Backward pass paired with `meanfield_kalman_filter`: runs ``L``
    independent `gaussx.rts_smoother` passes under `jax.vmap`, one per
    diagonal block. ``transition`` / ``process_noise`` accept the same
    forms as the filter and are projected onto their diagonal blocks.

    Args:
        filter_state: Output of `meanfield_kalman_filter` (or any
            `gaussx.FilterState` whose covariances are block-diagonal —
            off-block entries are discarded).
        transition: State transition matrix or operator.
        process_noise: Process noise covariance or operator. (Unused by
            the standard RTS recurrence — kept for API symmetry.)
        block_size: State block size ``d``; ``D`` must be divisible by
            it.
        solver: Optional solver strategy for the per-block smoother
            gains. Sequential mode only.
        parallel: When ``True``, run each block through
            `gaussx.parallel_rts_smoother` instead of the sequential
            scan.

    Raises:
        ValueError: If ``D`` is not divisible by ``block_size``.

    Returns:
        Tuple ``(smoothed_means, smoothed_covs)`` with shapes
        ``(T, D)`` and ``(T, D, D)``; covariances are block-diagonal
        embeddings of the per-block smoothed covariances.
    """
    D = filter_state.filtered_means.shape[-1]
    d = block_size
    if D % d != 0:
        raise ValueError(f"State dimension {D} is not divisible by block_size {d}.")
    L = D // d

    A_blocks = _split_diag_blocks(transition, L, d, d)
    Q_blocks = _split_diag_blocks(process_noise, L, d, d)

    block_states = FilterState(
        filtered_means=rearrange(filter_state.filtered_means, "t (l d) -> l t d", l=L),
        filtered_covs=_split_diag_blocks(filter_state.filtered_covs, L, d, d),
        predicted_means=rearrange(
            filter_state.predicted_means, "t (l d) -> l t d", l=L
        ),
        predicted_covs=_split_diag_blocks(filter_state.predicted_covs, L, d, d),
        log_likelihood=jnp.broadcast_to(filter_state.log_likelihood, (L,)),
    )

    base_smoother = parallel_rts_smoother if parallel else rts_smoother

    def _run(state_b, A_b, Q_b):
        return base_smoother(state_b, A_b, Q_b, solver=solver)

    s_means, s_covs = jax.vmap(_run)(block_states, A_blocks, Q_blocks)

    return (
        rearrange(s_means, "l t d -> t (l d)"),
        _embed_block_diag(s_covs),
    )
