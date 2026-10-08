"""Structured diagonal extraction with dispatch on operator type."""

from __future__ import annotations

import functools as ft

import einx
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
from gaussx._operators._diagonalised import (
    DiagonalizedOperator,
    _fftn,
    _ifftn,
)
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._low_rank_update import LowRankUpdate
from gaussx._operators._sparse import SparseOperator
from gaussx._operators._spectral_function import SpectralFunction
from gaussx._operators._sum_kronecker import SumOfKroneckers
from gaussx._operators._toeplitz import Toeplitz
from gaussx._primitives._cholesky import warn_dense_fallback
from gaussx._primitives._samplers import SamplerName, resolve_sampler, split_keys


def diag(
    operator: lx.AbstractLinearOperator,
    *,
    stochastic: bool = False,
    num_probes: int = 20,
    key: jax.Array | None = None,
    sampler: SamplerName = "signs",
) -> Float[Array, " n"]:
    """Extract the diagonal of an operator as a 1D array.

    When ``stochastic=True``, uses Hutchinson's diagonal estimator
    via matfree — only requires matvec access, no materialization.

    Args:
        operator: A linear operator.
        stochastic: If ``True``, use stochastic diagonal estimation.
        num_probes: Number of probe vectors for stochastic mode.
        key: PRNG key for stochastic mode.
        sampler: Probe distribution for stochastic mode (``"signs"``,
            ``"normal"``, ``"sphere"``).

    Returns:
        1D array of diagonal entries (exact or estimated).

    Examples:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> B = lx.DiagonalLinearOperator(jnp.array([4.0, 5.0]))
        >>> K = gaussx.Kronecker(A, B)  # diag(4, 5, 8, 10)
        >>> [float(v) for v in gaussx.diag(K)]  # diag(A) ⊗ diag(B)
        [4.0, 5.0, 8.0, 10.0]
    """

    # Every recursive call forwards the estimator options, so a wrapped or
    # structured matrix-free operator is never materialised (gh-320).
    def rec(op: lx.AbstractLinearOperator, k: jax.Array | None = key) -> Array:
        return diag(
            op, stochastic=stochastic, num_probes=num_probes, key=k, sampler=sampler
        )

    def rec_all(ops) -> list[Array]:
        ops = tuple(ops)
        return [
            rec(op, k) for op, k in zip(ops, split_keys(key, len(ops)), strict=True)
        ]

    if isinstance(operator, lx.IdentityLinearOperator):
        return jnp.ones(operator.in_size(), dtype=operator.in_structure().dtype)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    if isinstance(operator, BlockDiag):
        return jnp.concatenate(rec_all(operator.operators))
    if isinstance(operator, Kronecker):
        # diag(A ⊗ B) = diag(A) ⊗ diag(B).
        return ft.reduce(jnp.kron, rec_all(operator.operators))
    if isinstance(operator, BlockTriDiag | LowerBlockTriDiag | UpperBlockTriDiag):
        return _diag_block_tridiag(operator)
    if isinstance(operator, LowRankUpdate):
        return _diag_low_rank(operator, rec(operator.base))
    if isinstance(operator, KroneckerSum):
        return _diag_kronecker_sum(*rec_all((operator.A, operator.B)))
    if isinstance(operator, SparseOperator | SpectralFunction):
        return operator.diagonal()
    if isinstance(operator, Toeplitz):
        # A symmetric Toeplitz matrix has the constant diagonal c[0] (gh-373).
        return jnp.full(operator.in_size(), operator.column[0])
    if isinstance(operator, DiagonalizedOperator) and _is_fft_pair(operator):
        # F⁻¹ diag(λ) F has the constant diagonal mean(λ) (gh-373).
        mean = jnp.mean(operator.eigenvalues)
        mean = jnp.real(mean) if operator.real_output else mean
        return jnp.full(operator.in_size(), mean)
    if isinstance(operator, SumOfKroneckers):
        return ft.reduce(jnp.add, rec_all(operator.operators))
    if isinstance(operator, lx.TaggedLinearOperator):
        return rec(operator.operator)
    if isinstance(operator, lx.AddLinearOperator):
        first, second = rec_all((operator.operator1, operator.operator2))
        return first + second
    if isinstance(operator, lx.MulLinearOperator):
        return operator.scalar * rec(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        return rec(operator.operator) / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        return -rec(operator.operator)
    if stochastic:
        return _diag_stochastic(operator, num_probes, key, sampler)
    if isinstance(operator, DiagonalizedOperator):
        warn_dense_fallback(
            "diag(DiagonalizedOperator) with a non-FFT transform pair "
            "materialises the operator; diag(..., stochastic=True) estimates "
            "it from matvecs."
        )
    return jnp.diag(operator.as_matrix())


def _is_fft_pair(operator: DiagonalizedOperator) -> bool:
    return operator.forward is _fftn and operator.inverse is _ifftn


def _diag_block_tridiag(
    operator: BlockTriDiag | LowerBlockTriDiag | UpperBlockTriDiag,
) -> Float[Array, " n"]:
    """Extract block-diagonal entries of a block-(tri/bi)diagonal operator."""
    from gaussx._einx import rearrange

    # diagonal blocks contain the diagonal entries
    block_diags = jax.vmap(jnp.diag)(operator.diagonal)  # (N, d)
    return rearrange(block_diags, "N d -> (N d)")


def _diag_low_rank(
    operator: LowRankUpdate, base_diag: Float[Array, " n"]
) -> Float[Array, " n"]:
    """diag(L + U diag(d) V^T) = diag(L) + sum_k U[:, k] d[k] V[:, k]."""
    from gaussx._einx import reduce

    if operator.rank == 0:
        # einx rejects a zero-length axis.
        return base_diag
    update = reduce(operator.U * operator.d * operator.V, "n k -> n", "sum")
    return base_diag + update


def _diag_kronecker_sum(
    diag_a: Float[Array, " a"], diag_b: Float[Array, " b"]
) -> Float[Array, " n"]:
    """diag(A (+) B) = kron(diag(A), 1_b) + kron(1_a, diag(B))."""
    return einx.add("a, b -> (a b)", diag_a, diag_b)


def _diag_stochastic(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jax.Array | None,
    sampler: SamplerName,
) -> Float[Array, " n"]:
    """Hutchinson diagonal estimator via matfree."""
    if key is None:
        key = jax.random.PRNGKey(0)

    n = operator.in_size()
    probe_fn = resolve_sampler(
        sampler, n, num_probes, dtype=operator.in_structure().dtype
    )
    integrand = matfree.stochtrace.monte_carlo_diagonal()
    estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    return estimate(operator.mv, key)


# Operators whose `diag` is exact and needs no matvecs of a matrix-free part:
# a stored matrix or diagonal, or a structural formula over small factors.
_CHEAP_DIAGONAL = (
    lx.IdentityLinearOperator,
    lx.DiagonalLinearOperator,
    lx.MatrixLinearOperator,
    lx.TridiagonalLinearOperator,
    Kronecker,
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
    KroneckerSum,
    SparseOperator,
    SpectralFunction,
    SumOfKroneckers,
)


def matrix_free_diag(
    operator: lx.AbstractLinearOperator,
    *,
    estimate: bool,
    num_probes: int = 20,
    key: jax.Array | None = None,
) -> Float[Array, " n"]:
    """The diagonal of *operator* without materialising it (gh-361).

    Stored and structured parts (`MatrixLinearOperator`, `Kronecker`, ...)
    give their exact diagonal through `diag`, also inside sums, scalings,
    tags, `BlockDiag` and `LowRankUpdate`. A matrix-free part, such as a bare
    `FunctionLinearOperator`, is never formed: with ``estimate=True`` its
    diagonal is a Hutchinson estimate from ``num_probes`` matvecs, otherwise
    it is read off one matvec per unit vector, exact in O(n) memory but
    ``n`` matvecs.

    Args:
        operator: A square linear operator.
        estimate: Estimate matrix-free diagonals instead of reading them.
        num_probes: Probes for the estimate.
        key: PRNG key for the estimate. ``None`` means ``PRNGKey(0)``.

    Returns:
        The diagonal, shape ``(n,)``.
    """

    def recurse(op):
        return matrix_free_diag(op, estimate=estimate, num_probes=num_probes, key=key)

    if isinstance(operator, lx.TaggedLinearOperator):
        return recurse(operator.operator)
    if isinstance(operator, lx.AddLinearOperator):
        return recurse(operator.operator1) + recurse(operator.operator2)
    if isinstance(operator, lx.MulLinearOperator):
        return operator.scalar * recurse(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        return recurse(operator.operator) / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        return -recurse(operator.operator)
    if isinstance(operator, BlockDiag):
        return jnp.concatenate([recurse(op) for op in operator.operators])
    if isinstance(operator, LowRankUpdate):
        from gaussx._einx import reduce

        update = reduce(operator.U * operator.d * operator.V, "n k -> n", "sum")
        return recurse(operator.base) + update
    if isinstance(operator, _CHEAP_DIAGONAL):
        return diag(operator)
    if estimate:
        return _diag_stochastic(operator, num_probes, key, "signs")
    return _diag_by_unit_vectors(operator)


def _diag_by_unit_vectors(operator: lx.AbstractLinearOperator) -> Float[Array, " n"]:
    """``A_ii = (A eᵢ)ᵢ``, one matvec at a time: exact, O(n) memory."""
    n = operator.in_size()
    dtype = operator.in_structure().dtype

    def entry(i):
        unit = jnp.zeros(n, dtype=dtype).at[i].set(1.0)
        return operator.mv(unit)[i]

    return jax.lax.map(entry, jnp.arange(n))
