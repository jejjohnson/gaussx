"""Structured trace tr(A), dispatched on the operator type of A.

A is a square N × N operator and tr(A) = ∑_i A_ii its trace. A Kronecker
product is A = A_1 ⊗ … ⊗ A_D with square factors A_d, d = 1, …, D, and a
Kronecker sum is A_1 ⊕ A_2 = A_1 ⊗ I + I ⊗ A_2 with N_1 × N_1 and N_2 × N_2
factors. A diagonalised operator is A = W⁻¹ diag(γ) W for a forward
transform W (an FFT, say) and eigenvalues γ. A low-rank update is
A = A₀ + U D Vᵀ, with base A₀, N × k factors U and V and D = diag(d). The
stochastic estimators average z^(i)ᵀ A z^(i) over probes z^(i),
i = 1, …, N_probe, and the variance-reduced ones deflate an orthonormal
basis Q of A's dominant range first.
"""

from __future__ import annotations

import functools as ft
from typing import Literal

import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
from jaxtyping import Array, Float

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import (
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
)
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sum_kronecker import SumOfKroneckers
from gaussx._operators._toeplitz import Toeplitz
from gaussx._primitives._samplers import SamplerName, resolve_sampler, split_keys
from gaussx._randomized._trace import hutchpp_trace


def trace(
    operator: lx.AbstractLinearOperator,
    *,
    stochastic: bool = False,
    num_probes: int = 20,
    key: jax.Array | None = None,
    sampler: SamplerName | None = None,
    algorithm: Literal["hutchinson", "hutchpp", "xtrace"] = "hutchinson",
) -> Float[Array, ""]:
    """Trace tr(A) of an operator, with structural dispatch.

        tr(A) = ∑_i A_ii

    With ``stochastic=True`` an operator that has no structural rule gets a
    stochastic estimate that needs only matvecs and never materialises A.
    Hutchinson averages z^(i)ᵀ A z^(i) over N_probe probes, with variance
    O(‖A‖_F² / N_probe). The variance-reduced estimators deflate a low-rank
    part first,

        tr(A) = tr(Qᵀ A Q) + tr((I − Q Qᵀ) A (I − Q Qᵀ)),

    with Q from a randomised range finder: the first term is exact and only
    the (small) remainder is probed. For a PSD A with a decaying spectrum
    that reaches relative error ε in O(1/ε) matvecs instead of O(1/ε²).

    - ``"hutchpp"`` (Hutch++, Meyer, Musco, Musco & Woodruff, 2021):
      ``num_probes`` is the total matvec budget, at least 3, a third each
      for the range sketch, the exact projection and Hutchinson on the
      remainder.
    - ``"xtrace"`` (Epperly, Tropp & Webber, 2024; matfree's
      ``leave_one_out_xtrace``): every probe both builds Q and estimates
      the remainder, by leave-one-out; ``num_probes`` probes cost
      ``2 * num_probes`` matvecs. Usually the most accurate per matvec.

    Algorithm:
        1. Take the first structural rule that matches A: the identity has
           trace N, diagonal and diagonalised operators sum their entries or
           eigenvalues, Toeplitz has N c₀, a block-diagonal operator sums
           its blocks' traces, a Kronecker product multiplies its factors'
           traces, a low-rank update adds ∑_k d_k V_kᵀ U_k to tr(A₀), and a
           Kronecker sum gives N_2 tr(A_1) + N_1 tr(A_2).
        2. Tagged, summed, scaled and negated operators recurse into their
           parts, splitting ``key`` so each part gets its own probes.
        3. Otherwise estimate the trace from probes when ``stochastic``,
           else materialise A and sum its diagonal.

    Args:
        operator: The square operator A.
        stochastic: If ``True``, estimate the trace of any operator that has
            no structural rule from probes instead of materialising it.
        num_probes: Number of probes N_probe (the matvec budget for
            ``"hutchpp"``).
        key: PRNG key for the probes. ``None`` means ``PRNGKey(0)``.
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``). Defaults to ``"signs"`` for Hutchinson and
            Hutch++ and ``"sphere"`` for XTrace (which requires a
            rotationally invariant distribution).
        algorithm: ``"hutchinson"`` (Monte Carlo), ``"hutchpp"``
            (Hutch++, low-rank deflation plus Hutchinson) or ``"xtrace"``
            (leave-one-out, Epperly et al. 2024; much lower variance for the
            same number of matvecs).

    Returns:
        The trace tr(A), exact or estimated.

    Raises:
        ValueError: For an unknown ``algorithm``, ``algorithm="xtrace"``
            with ``sampler="signs"``, or ``algorithm="hutchpp"`` with
            ``num_probes < 3``.

    References:
        Meyer, R. A., Musco, C., Musco, C. & Woodruff, D. P. (2021).
        Hutch++: optimal stochastic trace estimation. *SOSA*, 142-155.

        Epperly, E. N., Tropp, J. A. & Webber, R. J. (2024). XTrace: making
        the most of every sample in stochastic trace estimation. *SIAM J.
        Matrix Anal. Appl.*, 45(1), 1-23.

    Examples:
        Structured operators take an exact fast path:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> A_2 = lx.DiagonalLinearOperator(jnp.array([4.0, 5.0]))
        >>> A = gaussx.Kronecker(A_1, A_2)  # diag(4, 5, 8, 10)
        >>> float(gaussx.trace(A))  # tr(A_1) · tr(A_2)
        27.0

        Hutch++ on a dense matrix with a decaying spectrum γ_i = 1 / i²:

        >>> Q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (200, 200)))
        >>> eigenvalues = 1.0 / jnp.arange(1.0, 201.0) ** 2
        >>> scaled = einx.multiply("row col, col -> row col", Q, eigenvalues)
        >>> matrix = einx.dot("row k, col k -> row col", scaled, Q)
        >>> A = lx.MatrixLinearOperator(matrix)
        >>> estimate = gaussx.trace(
        ...     A, stochastic=True, num_probes=30, algorithm="hutchpp"
        ... )
        >>> exact = jnp.sum(eigenvalues)
        >>> bool(jnp.abs(estimate - exact) / exact < 0.05)
        True

        An unknown algorithm is rejected:

        >>> gaussx.trace(
        ...     A, stochastic=True, algorithm="lanczos"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Unknown algorithm 'lanczos'
    """

    # Every recursive call forwards the estimator options, so a wrapped or
    # structured matrix-free operator is never materialised (gh-320).
    def recurse(
        part: lx.AbstractLinearOperator, part_key: jax.Array | None = key
    ) -> Array:
        return trace(
            part,
            stochastic=stochastic,
            num_probes=num_probes,
            key=part_key,
            sampler=sampler,
            algorithm=algorithm,
        )

    def recurse_each(parts) -> list[Array]:
        parts = tuple(parts)
        part_keys = split_keys(key, len(parts))
        return [
            recurse(part, part_key)
            for part, part_key in zip(parts, part_keys, strict=True)
        ]

    if isinstance(operator, lx.IdentityLinearOperator):
        # tr(I) = N
        return jnp.asarray(operator.in_size(), dtype=operator.in_structure().dtype)
    if isinstance(operator, lx.DiagonalLinearOperator):
        # tr(diag(a)) = ∑_i a_i
        return jnp.sum(lx.diagonal(operator))
    if isinstance(operator, DiagonalizedOperator):
        # tr(W⁻¹ diag(γ) W) = ∑_i γ_i
        total = jnp.sum(operator.eigenvalues)
        return jnp.real(total) if operator.real_output else total
    if isinstance(operator, Toeplitz):
        # tr(A) = N c₀, from the constant diagonal c₀ (gh-373).
        return operator.in_size() * operator.column[0]
    if isinstance(operator, BlockDiag):
        # tr(blockdiag(A_1, …, A_D)) = ∑_d tr(A_d)
        return ft.reduce(jnp.add, recurse_each(operator.operators))
    if isinstance(operator, Kronecker):
        # tr(A_1 ⊗ … ⊗ A_D) = ∏_d tr(A_d)
        return ft.reduce(jnp.multiply, recurse_each(operator.operators))
    if isinstance(operator, BlockTriDiag | LowerBlockTriDiag | UpperBlockTriDiag):
        return _trace_block_tridiag(operator)
    if isinstance(operator, LowRankUpdate):
        return _trace_low_rank(operator, recurse(operator.base))
    if isinstance(operator, KroneckerSum):
        trace_a, trace_b = recurse_each((operator.A, operator.B))
        # N_2 tr(A_1)
        first = operator.B.out_size() * trace_a
        # N_1 tr(A_2)
        second = operator.A.out_size() * trace_b
        # tr(A_1 ⊕ A_2) = N_2 tr(A_1) + N_1 tr(A_2)
        return first + second
    if isinstance(operator, SumOfKroneckers):
        # tr(∑_j A_j) = ∑_j tr(A_j)
        return ft.reduce(jnp.add, recurse_each(operator.operators))
    if isinstance(operator, lx.TaggedLinearOperator):
        return recurse(operator.operator)
    if isinstance(operator, lx.AddLinearOperator):
        first, second = recurse_each((operator.operator1, operator.operator2))
        # tr(A + B) = tr(A) + tr(B)
        return first + second
    if isinstance(operator, lx.MulLinearOperator):
        # tr(c A) = c tr(A)
        return operator.scalar * recurse(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        # tr(A / c) = tr(A) / c
        return recurse(operator.operator) / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        # tr(−A) = −tr(A)
        return -recurse(operator.operator)
    if stochastic:
        return _trace_stochastic(operator, num_probes, key, sampler, algorithm)
    # tr(A), dense
    return jnp.trace(operator.as_matrix())


def _trace_block_tridiag(
    operator: BlockTriDiag | LowerBlockTriDiag | UpperBlockTriDiag,
) -> Float[Array, ""]:
    """Trace of a block-tridiagonal or block-bidiagonal operator.

        tr(A) = ∑_b tr(A_bb)

    Only the B diagonal blocks A_bb hold diagonal entries.

    Args:
        operator: Block-tridiagonal or block-bidiagonal operator.

    Returns:
        The trace, summed over the diagonal blocks.
    """
    # tr(A_bb), per block
    block_traces = jax.vmap(jnp.trace)(operator.diagonal)
    # ∑_b tr(A_bb)
    return jnp.sum(block_traces)


def _trace_low_rank(
    operator: LowRankUpdate, base_trace: Float[Array, ""]
) -> Float[Array, ""]:
    """Trace of a low-rank update A₀ + U D Vᵀ.

        tr(A₀ + U D Vᵀ) = tr(A₀) + ∑_k d_k V_kᵀ U_k

    with U_k and V_k the k-th columns of U and V.

    Args:
        operator: Low-rank update A₀ + U D Vᵀ.
        base_trace: The base trace tr(A₀).

    Returns:
        The trace of the update.
    """
    # ∑_k d_k V_kᵀ U_k, a full sum, so k = 0 is fine
    update = jnp.sum(operator.U * operator.d * operator.V)
    # tr(A₀) + ∑_k d_k V_kᵀ U_k
    return base_trace + update


def _trace_stochastic(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jax.Array | None,
    sampler: SamplerName | None,
    algorithm: Literal["hutchinson", "hutchpp", "xtrace"],
) -> Float[Array, ""]:
    """Stochastic trace estimate: matfree Hutchinson or XTrace, or Hutch++.

        tr(A) ≈ (1 / N_probe) ∑_i z^(i)ᵀ A z^(i)

    for Hutchinson; XTrace and Hutch++ deflate a sketched range Q first.

    Args:
        operator: The square operator A, used through matvecs only.
        num_probes: Number of probes N_probe (the matvec budget for
            ``"hutchpp"``).
        key: PRNG key for the probes. ``None`` means ``PRNGKey(0)``.
        sampler: Probe distribution, or ``None`` for the algorithm's
            default.
        algorithm: ``"hutchinson"``, ``"hutchpp"`` or ``"xtrace"``.

    Returns:
        The estimated trace.

    Raises:
        ValueError: For an unknown ``algorithm``, or ``"xtrace"`` with
            ``sampler="signs"``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    size = operator.in_size()
    if algorithm == "xtrace":
        # XTrace requires rotationally invariant probes.
        if sampler == "signs":
            raise ValueError(
                "XTrace requires a rotationally invariant sampler; "
                'use sampler="normal" or sampler="sphere".'
            )
        probe_fn = resolve_sampler(
            sampler or "sphere", size, num_probes, dtype=operator.in_structure().dtype
        )
        integrand = matfree.stochtrace.leave_one_out_xtrace()
        estimate = matfree.stochtrace.estimator_leave_one_out(integrand, probe_fn)
    elif algorithm == "hutchinson":
        probe_fn = resolve_sampler(
            sampler or "signs", size, num_probes, dtype=operator.in_structure().dtype
        )
        # z^(i)ᵀ A z^(i)
        integrand = matfree.stochtrace.monte_carlo_trace()
        estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    elif algorithm == "hutchpp":
        return hutchpp_trace(operator, num_probes, key, sampler or "signs")
    else:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; expected "
            '"hutchinson", "hutchpp" or "xtrace".'
        )
    # tr(A) ≈ (1 / N_probe) ∑_i z^(i)ᵀ A z^(i), or its leave-one-out form
    return estimate(operator.mv, key)


def trace_and_diag(
    operator: lx.AbstractLinearOperator,
    *,
    num_probes: int = 20,
    key: jax.Array | None = None,
    sampler: SamplerName = "signs",
) -> tuple[Float[Array, ""], Float[Array, " n"]]:
    """Joint stochastic estimate of tr(A) and diag(A) from one probe pass.

        tr(A) ≈ (1 / N_probe) ∑_i z^(i)ᵀ A z^(i),
        diag(A) ≈ (1 / N_probe) ∑_i z^(i) ⊙ A z^(i)

    Halves the matvec budget of calling ``trace(..., stochastic=True)`` and
    ``diag(..., stochastic=True)`` separately: both statistics accumulate
    from the same products A z^(i). There is no structural dispatch; A is
    used through matvecs only.

    Args:
        operator: The square operator A.
        num_probes: Number of probes N_probe.
        key: PRNG key for the probes. ``None`` means ``PRNGKey(0)``.
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``).

    Returns:
        Tuple ``(trace_estimate, diagonal_estimate)`` holding the estimates
        of tr(A) and diag(A).

    Examples:
        Sign probes recover a stored diagonal matrix exactly:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.MatrixLinearOperator(jnp.diag(jnp.array([3.0, 1.0, 2.0])))
        >>> trace, diagonal = gaussx.trace_and_diag(A, num_probes=4, key=jr.key(0))
        >>> round(float(trace), 4), [round(float(v), 4) for v in diagonal]
        (6.0, [3.0, 1.0, 2.0])
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    size = operator.in_size()
    probe_fn = resolve_sampler(
        sampler, size, num_probes, dtype=operator.in_structure().dtype
    )
    # z^(i)ᵀ A z^(i) and z^(i) ⊙ A z^(i)
    integrand = matfree.stochtrace.monte_carlo_trace_and_diagonal()
    estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    # averages over the N_probe probes
    result = estimate(operator.mv, key)
    return result["trace"], result["diagonal"]
