"""Pivoted partial-Cholesky preconditioner.

A = K + σ² I is an N × N symmetric positive-definite system with a PSD part K
and a noise variance σ² (the ``shift``). A pivoted rank-k partial Cholesky
factor L_k (N × k) with K ≈ L_k L_kᵀ gives the preconditioner
M = σ² I + L_k L_kᵀ ≈ A, applied by the Woodbury identity

    M⁻¹ v = σ⁻² (v − L_k C⁻¹ L_kᵀ v),   C = σ² I_k + L_kᵀ L_k,

with the k × k capacitance C held as its lower Cholesky factor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float, Int

from gaussx._einx import einsum
from gaussx._operators import BlockDiag, LowRankUpdate
from gaussx._preconditioners._base import AbstractPreconditioner
from gaussx._primitives._diag import _CHEAP_DIAGONAL, matrix_free_diag
from gaussx._randomized._rpcholesky import _pivoted_cholesky


# Hutchinson probes for the diagonal of a matrix-free operator (gh-361). The
# diagonal only selects pivots; the factor columns are exact either way.
_DIAGONAL_PROBES = 20


class PartialCholeskyPreconditioner(AbstractPreconditioner):
    """Preconditioner from a pivoted partial Cholesky factor.

        M = σ² I + L_k L_kᵀ ≈ A = K + σ² I,
        M⁻¹ v = σ⁻² (v − L_k (σ² I_k + L_kᵀ L_k)⁻¹ L_kᵀ v)

    For a system A = K + σ² I it builds a rank-k pivoted partial Cholesky
    factor L_k of the PSD part, K ≈ L_k L_kᵀ, and applies M⁻¹ through the
    Woodbury identity. At full rank this is exactly A⁻¹. Pivots are greedy
    (``argmax`` of the residual diagonal) or random (`gaussx.rp_cholesky`,
    proportional to it).

    Two ways to build it:

    - **Once**, with `from_operator` on the PSD part K and an explicit
      ``shift=σ²``. The factor and the Woodbury capacitance are stored as
      arrays, so `as_operator` ignores its argument and repeated solves cost
      no further matvecs of K (#371).
    - **Lazily**, as ``PartialCholeskyPreconditioner(rank, shift)``. Then
      `as_operator` receives the *system* operator A at every solve and
      factors K = A − σ² I implicitly, so ``shift`` is the part of A's
      diagonal treated as noise and the noise is never counted twice
      (#345). ``shift`` must not exceed the σ² actually in A.

    The factor is guarded like LAPACK ``?pstrf``: once ``rank`` exceeds the
    numerical rank (small datasets, duplicated inputs, noiseless kernels),
    the surplus columns are exactly zero instead of NaN or inf, and the
    preconditioner degrades gracefully to the lower-rank one (gh-237).

    **Cost.** A build applies the operator to ``rank`` pivot columns (or
    calls ``column``, see `from_operator`) and needs its diagonal. A given
    ``diagonal``, a stored matrix and the structured operators
    (`gaussx.diag`) cost nothing more. For a matrix-free part, such as a bare
    `lineax.FunctionLinearOperator`, the diagonal is a Hutchinson estimate
    from 20 matvecs: it only selects the pivots, and each factor column is
    read exactly from its pivot column, so the operator is never
    materialised (O(N k) memory, gh-361).

    Covariance form only: the factor captures the *top* of K's spectrum,
    which is right for K + σ² I. It is not a good preconditioner for a
    precision-form system Q + Aᵀ W A, whose hard directions are its
    smallest eigenvalues.

    Attributes:
        rank: Rank k of the partial Cholesky. ``<= 0`` disables the lazy
            preconditioner (`as_operator` returns ``None``).
        shift: Noise variance σ² for the lazy path: the part of the
            system's diagonal treated as noise.
        pivoting: ``"greedy"`` (default) or ``"random"``.
        key: PRNG key for ``"random"`` pivoting and for the diagonal
            estimate of a matrix-free operator. ``None`` means
            ``jax.random.PRNGKey(0)``.
        diagonal: The diagonal of the *system* operator A for the lazy path.
            ``None`` takes it from the operator (exact unless it is
            matrix-free; see **Cost**).
        factor: Stored factor L_k, N × k; set by `from_operator`.
        capacitance: Lower Cholesky factor of C = σ² I_k + L_kᵀ L_k,
            k × k; set by `from_operator`.
        factor_shift: The σ² the stored factor was built with; set by
            `from_operator`.

    Examples:
        Build once on the PSD part, then reuse across solves:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> inputs = jnp.linspace(0.0, 10.0, 50)
        >>> gaps = einx.subtract("row, col -> row col", inputs, inputs)
        >>> kernel = jnp.exp(-0.5 * gaps**2)
        >>> psd = lx.positive_semidefinite_tag
        >>> preconditioner = gaussx.PartialCholeskyPreconditioner.from_operator(
        ...     lx.MatrixLinearOperator(kernel, psd), rank=20, shift=0.1
        ... )
        >>> preconditioner.factor.shape
        (50, 20)
        >>> A = lx.MatrixLinearOperator(kernel + 0.1 * jnp.eye(50), psd)
        >>> solver = gaussx.CGSolver(preconditioner=preconditioner)
        >>> x1 = solver.solve(A, jnp.ones(50))
        >>> x2 = solver.solve(A, jnp.arange(50.0))  # no rebuild
        >>> bool(jnp.allclose(A.mv(x1), jnp.ones(50), atol=1e-2))
        True

        Built lazily from the system operator at every solve:

        >>> lazy = gaussx.PartialCholeskyPreconditioner(rank=20, shift=0.1)
        >>> M_inv = lazy.as_operator(A)
        >>> M_inv.in_structure().shape
        (50,)
    """

    rank: int = eqx.field(static=True, default=50)
    shift: float = eqx.field(static=True, default=1.0)
    pivoting: Literal["greedy", "random"] = eqx.field(static=True, default="greedy")
    key: jax.Array | None = None
    diagonal: Float[Array, " n"] | None = None
    factor: Float[Array, "n k"] | None = None
    capacitance: Float[Array, "k k"] | None = None
    factor_shift: Float[Array, ""] | None = None

    @classmethod
    def from_operator(
        cls,
        operator: lx.AbstractLinearOperator,
        rank: int = 50,
        *,
        shift: float | Float[Array, ""],
        pivoting: Literal["greedy", "random"] = "greedy",
        key: jax.Array | None = None,
        diagonal: Float[Array, " n"] | None = None,
        column: Callable[[Int[Array, ""]], Float[Array, " n"]] | None = None,
    ) -> PartialCholeskyPreconditioner:
        """Factor the PSD part K once and store the preconditioner.

            K ≈ L_k L_kᵀ,   C = σ² I_k + L_kᵀ L_k

        Algorithm:
            1. Clamp k to N and run the pivoted partial Cholesky of K.
            2. Factor the capacitance C = σ² I_k + L_kᵀ L_k by Cholesky.
            3. Store L_k, the factor of C and σ².

        Args:
            operator: The PSD part K (e.g. a kernel matrix), **not**
                K + σ² I.
            rank: Number of pivots k, clamped to the operator size N.
            shift: Noise variance σ²; the preconditioner approximates
                (K + σ² I)⁻¹. Must be positive.
            pivoting: ``"greedy"`` or ``"random"`` (see `gaussx.rp_cholesky`).
            key: PRNG key for ``"random"`` pivoting and the diagonal
                estimate. ``None`` means ``jax.random.PRNGKey(0)``.
            diagonal: The diagonal of K. ``None`` takes it from ``operator``
                (see **Cost** in the class docstring).
            column: ``n -> K[:, n]``, used instead of ``operator.mv(e_n)``
                (gh-544). For a matrix-free kernel operator a column is N
                kernel evaluations, where ``mv(e_n)`` is a full matvec. With
                ``column`` and ``diagonal`` both given, the build never
                applies ``operator``.

        Returns:
            A built `PartialCholeskyPreconditioner` whose `as_operator`
            ignores its argument.

        Raises:
            ValueError: If ``rank < 1``.
        """
        if rank < 1:
            raise ValueError("rank must be at least 1")
        rank = min(rank, operator.in_size())
        # K ≈ L_k L_kᵀ
        factor = _factor(operator, rank, 0.0, pivoting, key, diagonal, column)
        noise_variance = jnp.asarray(shift, dtype=factor.dtype)
        return cls(
            rank=rank,
            pivoting=pivoting,
            key=key,
            factor=factor,
            capacitance=_capacitance(factor, noise_variance),
            factor_shift=noise_variance,
        )

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator | None:
        """Return M⁻¹ = (σ² I + L_k L_kᵀ)⁻¹ as a PSD operator.

            M⁻¹ v = σ⁻² (v − L_k (σ² I_k + L_kᵀ L_k)⁻¹ L_kᵀ v)

        A preconditioner built by `from_operator` ignores ``operator``. The
        lazy one factors K = A − σ² I, where ``operator`` is the system A.

        Args:
            operator: The system operator A = K + σ² I, needed by the lazy
                path only.

        Returns:
            The matrix-free operator M⁻¹, tagged positive semidefinite, or
            ``None`` when the lazy ``rank`` is ``<= 0``.

        Raises:
            ValueError: If the lazy path gets no ``operator``.
        """
        if self.factor is not None:
            assert self.capacitance is not None and self.factor_shift is not None
            structure = jax.ShapeDtypeStruct((self.factor.shape[0],), self.factor.dtype)
            return _woodbury(
                self.factor, self.capacitance, self.factor_shift, structure
            )
        if self.rank <= 0:
            return None
        if operator is None:
            raise ValueError(
                "PartialCholeskyPreconditioner.as_operator requires the system "
                "operator to build its factor (or build it once with from_operator)."
            )

        rank = min(self.rank, operator.in_size())
        dtype = operator.in_structure().dtype
        noise_variance = jnp.asarray(self.shift, dtype=dtype)
        # Factor K = A − σ² I, never A itself: the Woodbury step adds σ² back
        # (#345).
        # A − σ² I ≈ L_k L_kᵀ
        factor = _factor(
            operator, rank, noise_variance, self.pivoting, self.key, self.diagonal, None
        )
        return _woodbury(
            factor,
            _capacitance(factor, noise_variance),
            noise_variance,
            operator.out_structure(),
        )


def _factor(
    operator: lx.AbstractLinearOperator,
    rank: int,
    shift: float | Float[Array, ""],
    pivoting: Literal["greedy", "random"],
    key: jax.Array | None,
    diagonal: Float[Array, " n"] | None,
    column: Callable[[Int[Array, ""]], Float[Array, " n"]] | None,
) -> Float[Array, "n k"]:
    """Pivoted partial Cholesky of B − σ² I, one column per pivot.

        B − σ² I ≈ L_k L_kᵀ

    ``diagonal`` and ``column`` describe ``operator`` B itself; the shift
    σ² is subtracted here.

    Args:
        operator: Operator B (K on the built path, A on the lazy path).
        rank: Number of pivots k.
        shift: σ² to subtract from B (0 on the built path).
        pivoting: ``"greedy"`` or ``"random"``.
        key: PRNG key for random pivots and the Hutchinson diagonal.
        diagonal: diag(B), or ``None`` to read or estimate it.
        column: ``n -> B[:, n]``, or ``None`` to use ``operator.mv(e_n)``.

    Returns:
        The factor L_k, N × k, with exactly zero surplus columns past the
        numerical rank.
    """
    size = operator.in_size()
    dtype = operator.in_structure().dtype
    # Only a Hutchinson estimate is approximate; the pivots then take their
    # values from the exact columns.
    approximate = diagonal is None and not _diagonal_is_exact(operator)
    if diagonal is None:
        diagonal = matrix_free_diag(
            operator, estimate=True, num_probes=_DIAGONAL_PROBES, key=key
        )
    # diag(B) − σ²
    diagonal = jnp.asarray(diagonal, dtype=dtype) - shift
    if column is None:
        column = _mv_column(operator, size, dtype)

    def shifted_column(index):
        # B[:, n] − σ² e_n
        return column(index) - shift * jnp.zeros(size, dtype=dtype).at[index].set(1.0)

    factor, _ = _pivoted_cholesky(
        diagonal,
        shifted_column,
        rank,
        pivoting,
        key,
        approximate_diagonal=approximate,
    )
    return factor


def _mv_column(operator, n, dtype):
    """Column reader n ↦ B e_n through one matvec.

    Args:
        operator: Operator B.
        n: Operator size N.
        dtype: Dtype of the unit vectors e_n.

    Returns:
        A function ``index -> B e_index``.
    """

    def column(index):
        # B e_n
        return operator.mv(jnp.zeros(n, dtype=dtype).at[index].set(1.0))

    return column


def _diagonal_is_exact(operator: lx.AbstractLinearOperator) -> bool:
    """Whether `matrix_free_diag` gives the operator's exact diagonal.

    Args:
        operator: Operator B, walked through tags, negations, scalings,
            sums, `BlockDiag` and the base of a `LowRankUpdate`.

    Returns:
        ``True`` when every leaf has a cheap exact diagonal.
    """
    if isinstance(operator, lx.TaggedLinearOperator | lx.NegLinearOperator):
        return _diagonal_is_exact(operator.operator)
    if isinstance(operator, lx.MulLinearOperator | lx.DivLinearOperator):
        return _diagonal_is_exact(operator.operator)
    if isinstance(operator, lx.AddLinearOperator):
        return _diagonal_is_exact(operator.operator1) and _diagonal_is_exact(
            operator.operator2
        )
    if isinstance(operator, BlockDiag):
        return all(_diagonal_is_exact(op) for op in operator.operators)
    if isinstance(operator, LowRankUpdate):
        return _diagonal_is_exact(operator.base)
    return isinstance(operator, _CHEAP_DIAGONAL)


def _capacitance(
    factor: Float[Array, "n k"], shift: Float[Array, ""]
) -> Float[Array, "k k"]:
    """Lower Cholesky factor of the capacitance C = σ² I_k + L_kᵀ L_k.

        C = σ² I_k + L_kᵀ L_k = L_C L_Cᵀ

    Zero surplus columns only add σ² to the diagonal, so C stays positive
    definite.

    Args:
        factor: Partial Cholesky factor L_k, N × k.
        shift: Noise variance σ².

    Returns:
        The lower-triangular L_C, k × k.
    """
    identity = jnp.eye(factor.shape[1], dtype=factor.dtype)
    # L_kᵀ L_k
    gram = einsum(factor, factor, "row rank, row other -> rank other")
    # C = σ² I_k + L_kᵀ L_k = L_C L_Cᵀ
    return jnp.linalg.cholesky(shift * identity + gram)


def _woodbury(
    factor: Float[Array, "n k"],
    capacitance: Float[Array, "k k"],
    shift: Float[Array, ""],
    structure: jax.ShapeDtypeStruct,
) -> lx.AbstractLinearOperator:
    """M⁻¹ = (σ² I + L_k L_kᵀ)⁻¹ by the Woodbury identity, as an operator.

        M⁻¹ v = σ⁻² (v − L_k C⁻¹ L_kᵀ v),   C = σ² I_k + L_kᵀ L_k

    Args:
        factor: Partial Cholesky factor L_k, N × k.
        capacitance: Lower Cholesky factor L_C of C, k × k.
        shift: Noise variance σ².
        structure: Input structure of M⁻¹.

    Returns:
        The matrix-free operator M⁻¹, tagged positive semidefinite.
    """

    def matvec(v: Float[Array, " n"]) -> Float[Array, " n"]:
        # L_kᵀ v
        projection = einsum(factor, v, "row rank, row -> rank")
        # C⁻¹ L_kᵀ v
        coefficients = jax.scipy.linalg.cho_solve((capacitance, True), projection)
        # M⁻¹ v = σ⁻² (v − L_k C⁻¹ L_kᵀ v)
        return (v - factor @ coefficients) / shift

    return lx.FunctionLinearOperator(matvec, structure, lx.positive_semidefinite_tag)
