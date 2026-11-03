"""Structured diagonal diag(A), dispatched on the operator type of A.

A is a square N × N operator and diag(A) ∈ ℝᴺ the vector of its diagonal
entries A_ii. A Kronecker product is A = A_1 ⊗ … ⊗ A_D with square factors
A_d, d = 1, …, D, and a Kronecker sum is A_1 ⊕ A_2 = A_1 ⊗ I + I ⊗ A_2. A
diagonalised operator is A = W⁻¹ diag(γ) W for a forward transform W (an
FFT, say) and eigenvalues γ. A low-rank update is A = A₀ + U D Vᵀ, with base
A₀, N × k factors U and V and D = diag(d). The stochastic estimator averages
z^(i) ⊙ A z^(i) over probes z^(i), i = 1, …, N_probe.
"""

from __future__ import annotations

import functools as ft

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
from jaxtyping import Array, Float

from gaussx._einx import rearrange, reduce
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
    """Diagonal diag(A) of an operator, with structural dispatch.

        diag(A)_i = A_ii,   i = 1, …, N

    With ``stochastic=True`` an operator that has no structural rule gets
    Hutchinson's diagonal estimate through matfree, which needs only
    matvecs and never materialises A:

        diag(A) ≈ (1 / N_probe) ∑_i z^(i) ⊙ A z^(i)

    Algorithm:
        1. Take the first structural rule that matches A: the identity and a
           diagonal read off their entries, a block-diagonal operator
           concatenates its blocks' diagonals, a Kronecker product takes
           diag(A_1) ⊗ … ⊗ diag(A_D), a low-rank update adds ∑_k U_ik d_k
           V_ik to diag(A₀), a Kronecker sum broadcasts its factors'
           diagonals, and Toeplitz and FFT-diagonalised operators have a
           constant diagonal.
        2. Tagged, summed, scaled and negated operators recurse into their
           parts, splitting ``key`` so each part gets its own probes.
        3. Otherwise estimate the diagonal from probes when ``stochastic``,
           else materialise A and read its diagonal.

    Args:
        operator: The square operator A.
        stochastic: If ``True``, estimate the diagonal of any operator that
            has no structural rule from probes instead of materialising it.
        num_probes: Number of probes N_probe for the estimate.
        key: PRNG key for the probes. ``None`` means ``PRNGKey(0)``.
        sampler: Probe distribution for the estimate (``"signs"``,
            ``"normal"``, ``"sphere"``).

    Returns:
        The diagonal diag(A), exact or estimated.

    Examples:
        A Kronecker product of diagonals takes the product of the factors'
        diagonals:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> A_2 = lx.DiagonalLinearOperator(jnp.array([4.0, 5.0]))
        >>> A = gaussx.Kronecker(A_1, A_2)  # diag(4, 5, 8, 10)
        >>> [float(v) for v in gaussx.diag(A)]  # diag(A_1) ⊗ diag(A_2)
        [4.0, 5.0, 8.0, 10.0]

        A low-rank update adds the row sums of U ⊙ V weighted by d:

        >>> base = lx.DiagonalLinearOperator(jnp.array([1.0, 1.0, 1.0]))
        >>> U = jnp.array([[1.0], [2.0], [0.0]])
        >>> [float(v) for v in gaussx.diag(gaussx.LowRankUpdate(base, U))]
        [2.0, 5.0, 1.0]

        Sign probes recover a stored diagonal matrix exactly:

        >>> A = lx.MatrixLinearOperator(jnp.diag(jnp.array([3.0, 1.0, 2.0])))
        >>> estimate = gaussx.diag(A, stochastic=True, num_probes=4, key=jr.key(0))
        >>> [round(float(v), 4) for v in estimate]
        [3.0, 1.0, 2.0]
    """

    # Every recursive call forwards the estimator options, so a wrapped or
    # structured matrix-free operator is never materialised (gh-320).
    def recurse(
        part: lx.AbstractLinearOperator, part_key: jax.Array | None = key
    ) -> Array:
        return diag(
            part,
            stochastic=stochastic,
            num_probes=num_probes,
            key=part_key,
            sampler=sampler,
        )

    def recurse_each(parts) -> list[Array]:
        parts = tuple(parts)
        part_keys = split_keys(key, len(parts))
        return [
            recurse(part, part_key)
            for part, part_key in zip(parts, part_keys, strict=True)
        ]

    if isinstance(operator, lx.IdentityLinearOperator):
        # diag(I) = 1
        return jnp.ones(operator.in_size(), dtype=operator.in_structure().dtype)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator)
    if isinstance(operator, BlockDiag):
        # diag(blockdiag(A_1, …, A_D)) = [diag(A_1); …; diag(A_D)]
        return jnp.concatenate(recurse_each(operator.operators))
    if isinstance(operator, Kronecker):
        # diag(A_1 ⊗ … ⊗ A_D) = diag(A_1) ⊗ … ⊗ diag(A_D)
        return ft.reduce(jnp.kron, recurse_each(operator.operators))
    if isinstance(operator, BlockTriDiag | LowerBlockTriDiag | UpperBlockTriDiag):
        return _diag_block_tridiag(operator)
    if isinstance(operator, LowRankUpdate):
        return _diag_low_rank(operator, recurse(operator.base))
    if isinstance(operator, KroneckerSum):
        return _diag_kronecker_sum(*recurse_each((operator.A, operator.B)))
    if isinstance(operator, SparseOperator | SpectralFunction):
        return operator.diagonal()
    if isinstance(operator, Toeplitz):
        # A symmetric Toeplitz matrix has the constant diagonal c₀ (gh-373).
        return jnp.full(operator.in_size(), operator.column[0])
    if isinstance(operator, DiagonalizedOperator) and _is_fft_pair(operator):
        # W⁻¹ diag(γ) W has the constant diagonal mean(γ) for the FFT (gh-373).
        mean = jnp.mean(operator.eigenvalues)
        mean = jnp.real(mean) if operator.real_output else mean
        return jnp.full(operator.in_size(), mean)
    if isinstance(operator, SumOfKroneckers):
        # diag(∑_j A_j) = ∑_j diag(A_j)
        return ft.reduce(jnp.add, recurse_each(operator.operators))
    if isinstance(operator, lx.TaggedLinearOperator):
        return recurse(operator.operator)
    if isinstance(operator, lx.AddLinearOperator):
        first, second = recurse_each((operator.operator1, operator.operator2))
        # diag(A + B) = diag(A) + diag(B)
        return first + second
    if isinstance(operator, lx.MulLinearOperator):
        # diag(c A) = c diag(A)
        return operator.scalar * recurse(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        # diag(A / c) = diag(A) / c
        return recurse(operator.operator) / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        # diag(−A) = −diag(A)
        return -recurse(operator.operator)
    if stochastic:
        return _diag_stochastic(operator, num_probes, key, sampler)
    if isinstance(operator, DiagonalizedOperator):
        warn_dense_fallback(
            "diag(DiagonalizedOperator) with a non-FFT transform pair "
            "materialises the operator; diag(..., stochastic=True) estimates "
            "it from matvecs."
        )
    # diag(A), dense
    return jnp.diag(operator.as_matrix())


def _is_fft_pair(operator: DiagonalizedOperator) -> bool:
    """Whether the transform pair W, W⁻¹ is gaussx's own FFT pair."""
    return operator.forward is _fftn and operator.inverse is _ifftn


def _diag_block_tridiag(
    operator: BlockTriDiag | LowerBlockTriDiag | UpperBlockTriDiag,
) -> Float[Array, " n"]:
    """Diagonal of a block-tridiagonal or block-bidiagonal operator.

        diag(A) = [diag(A_11); …; diag(A_BB)]

    Only the B diagonal blocks A_bb hold diagonal entries.

    Args:
        operator: Block-tridiagonal or block-bidiagonal operator.

    Returns:
        The diagonal, the diagonal blocks' diagonals stacked in order.
    """
    # diag(A_bb), per block
    block_diagonals = jax.vmap(jnp.diag)(operator.diagonal)
    # [diag(A_11); …; diag(A_BB)]
    return rearrange(block_diagonals, "block row -> (block row)")


def _diag_low_rank(
    operator: LowRankUpdate, base_diag: Float[Array, " n"]
) -> Float[Array, " n"]:
    """Diagonal of a low-rank update A₀ + U D Vᵀ.

        diag(A₀ + U D Vᵀ)_i = diag(A₀)_i + ∑_k U_ik d_k V_ik

    Args:
        operator: Low-rank update A₀ + U D Vᵀ.
        base_diag: The base diagonal diag(A₀).

    Returns:
        The diagonal of the update.
    """
    if operator.rank == 0:
        # einx rejects a zero-length axis.
        return base_diag
    # U_ik d_k V_ik
    weighted_products = operator.U * operator.d * operator.V
    # ∑_k U_ik d_k V_ik
    update = reduce(weighted_products, "row rank -> row", "sum")
    # diag(A₀) + ∑_k U_ik d_k V_ik
    return base_diag + update


def _diag_kronecker_sum(
    diag_a: Float[Array, " a"], diag_b: Float[Array, " b"]
) -> Float[Array, " n"]:
    """Diagonal of a Kronecker sum from its factors' diagonals.

        diag(A_1 ⊕ A_2) = diag(A_1) ⊗ 1 + 1 ⊗ diag(A_2)

    Args:
        diag_a: The first factor's diagonal diag(A_1).
        diag_b: The second factor's diagonal diag(A_2).

    Returns:
        The diagonal of A_1 ⊕ A_2.
    """
    # diag(A_1) ⊗ 1 + 1 ⊗ diag(A_2)
    return einx.add("left, right -> (left right)", diag_a, diag_b)


def _diag_stochastic(
    operator: lx.AbstractLinearOperator,
    num_probes: int,
    key: jax.Array | None,
    sampler: SamplerName,
) -> Float[Array, " n"]:
    """Hutchinson's diagonal estimate through matfree.

        diag(A) ≈ (1 / N_probe) ∑_i z^(i) ⊙ A z^(i)

    Args:
        operator: The square operator A, used through matvecs only.
        num_probes: Number of probes N_probe.
        key: PRNG key for the probes. ``None`` means ``PRNGKey(0)``.
        sampler: Probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``).

    Returns:
        The estimated diagonal.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    size = operator.in_size()
    probe_fn = resolve_sampler(
        sampler, size, num_probes, dtype=operator.in_structure().dtype
    )
    # z^(i) ⊙ A z^(i)
    integrand = matfree.stochtrace.monte_carlo_diagonal()
    estimate = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    # (1 / N_probe) ∑_i z^(i) ⊙ A z^(i)
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
    """Diagonal diag(A) of an operator without materialising it (gh-361).

        diag(A)_i = A_ii = (A eᵢ)ᵢ

    Stored and structured parts (`MatrixLinearOperator`, `Kronecker`, ...)
    give their exact diagonal through `diag`, also inside sums, scalings,
    tags, `BlockDiag` and `LowRankUpdate`. A matrix-free part, such as a bare
    `FunctionLinearOperator`, is never formed: with ``estimate=True`` its
    diagonal is a Hutchinson estimate from N_probe matvecs, otherwise it is
    read off one matvec per unit vector eᵢ, exact in O(N) memory but N
    matvecs.

    Algorithm:
        1. Recurse through tags, sums, scalings, negations, block-diagonal
           operators and the base of a low-rank update.
        2. Read the exact diagonal of a stored or structured part with
           `diag`.
        3. Estimate or read a matrix-free part from matvecs.

    Args:
        operator: The square operator A.
        estimate: Estimate matrix-free diagonals instead of reading them.
        num_probes: Number of probes N_probe for the estimate.
        key: PRNG key for the estimate. ``None`` means ``PRNGKey(0)``.

    Returns:
        The diagonal diag(A).

    Examples:
        A matrix-free operator read one unit vector at a time:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._primitives._diag import matrix_free_diag
        >>> matrix = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        >>> structure = jax.ShapeDtypeStruct((2,), matrix.dtype)
        >>> A = lx.FunctionLinearOperator(lambda v: matrix @ v, structure)
        >>> [float(v) for v in matrix_free_diag(A, estimate=False)]
        [2.0, 3.0]
    """

    def recurse(part):
        return matrix_free_diag(part, estimate=estimate, num_probes=num_probes, key=key)

    if isinstance(operator, lx.TaggedLinearOperator):
        return recurse(operator.operator)
    if isinstance(operator, lx.AddLinearOperator):
        # diag(A + B) = diag(A) + diag(B)
        return recurse(operator.operator1) + recurse(operator.operator2)
    if isinstance(operator, lx.MulLinearOperator):
        # diag(c A) = c diag(A)
        return operator.scalar * recurse(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        # diag(A / c) = diag(A) / c
        return recurse(operator.operator) / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        # diag(−A) = −diag(A)
        return -recurse(operator.operator)
    if isinstance(operator, BlockDiag):
        # diag(blockdiag(A_1, …, A_D)) = [diag(A_1); …; diag(A_D)]
        return jnp.concatenate([recurse(block) for block in operator.operators])
    if isinstance(operator, LowRankUpdate):
        # U_ik d_k V_ik
        weighted_products = operator.U * operator.d * operator.V
        # ∑_k U_ik d_k V_ik
        update = reduce(weighted_products, "row rank -> row", "sum")
        # diag(A₀) + ∑_k U_ik d_k V_ik
        return recurse(operator.base) + update
    if isinstance(operator, _CHEAP_DIAGONAL):
        return diag(operator)
    if estimate:
        return _diag_stochastic(operator, num_probes, key, "signs")
    return _diag_by_unit_vectors(operator)


def _diag_by_unit_vectors(operator: lx.AbstractLinearOperator) -> Float[Array, " n"]:
    """Diagonal read off one matvec per unit vector: exact, O(N) memory.

        A_ii = (A eᵢ)ᵢ

    Args:
        operator: The square operator A, used through matvecs only.

    Returns:
        The exact diagonal diag(A), after N matvecs.
    """
    size = operator.in_size()
    dtype = operator.in_structure().dtype

    def entry(index):
        # eᵢ
        unit = jnp.zeros(size, dtype=dtype).at[index].set(1.0)
        # (A eᵢ)ᵢ
        return operator.mv(unit)[index]

    return jax.lax.map(entry, jnp.arange(size))
