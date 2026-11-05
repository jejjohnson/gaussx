"""Areal (graph) precision builders: Besag (ICAR), BYM2 and its scaling.

The Besag model (τ / 2) ∑_i~j w_ij (x_i − x_j)² = (τ / 2) xᵀ R x has the graph
Laplacian R as its structure matrix. gaussx never builds graphs: R arrives as an
operator (kernellib's ``Graph.laplacian_operator()``), and so does its null space
V (``graph_null_space``).
"""

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from gaussx._einx import einsum, rearrange
from gaussx._gmrf._temporal import _as_float
from gaussx._linalg._diag_inv import diag_inv
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._sparse import (
    _PLAN_CACHE_SIZE,
    SparseOperator,
    SparsityPattern,
    _canonicalise,
)
from gaussx._operators._spectral_function import SpectralFunction
from gaussx._primitives._diag import diag
from gaussx._primitives._solve import solve
from gaussx._sparse._factor import sparse_cholesky


_PSD = frozenset({lx.positive_semidefinite_tag})
# Row sums of a Laplacian must vanish to this fraction of its largest diagonal.
_ROW_SUM_TOLERANCE = 1e-8


def besag_structure(
    laplacian_op: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Validate a graph Laplacian as a Besag (ICAR) structure matrix.

        R = D − W,   R 1 = 0

    Checks that the operator is square and symmetric and, when its values are
    concrete, that its rows sum to zero (constants are in its null space), then
    tags it positive semidefinite. A `SparseOperator` or `BlockTriDiag` keeps its
    type (and so its sparse dispatch); any other operator is wrapped in a
    `lineax.TaggedLinearOperator`.

    Args:
        laplacian_op: The weighted graph Laplacian R = D − W, for the degree
            matrix D and adjacency W.

    Returns:
        The same matrix R, tagged symmetric and positive semidefinite.

    Raises:
        ValueError: If it is not square or symmetric, or a row sum is not zero.

    Examples:
        The path graph 0 − 1 − 2, each edge once:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import numpy as np
        >>> import gaussx
        >>> laplacian = gaussx.SparseOperator.from_coo(
        ...     np.array([0, 1, 2, 1, 2]),
        ...     np.array([0, 1, 2, 0, 1]),
        ...     jnp.array([1.0, 2.0, 1.0, -1.0, -1.0]),
        ...     (3, 3),
        ...     symmetric=True,
        ... )
        >>> structure = gaussx.besag_structure(laplacian)
        >>> type(structure).__name__, lx.is_positive_semidefinite(structure)
        ('SparseOperator', True)

        A matrix whose rows do not sum to zero is not a Laplacian:

        >>> gaussx.besag_structure(
        ...     lx.MatrixLinearOperator(jnp.eye(3), lx.symmetric_tag)
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        ValueError: A Besag structure matrix is a graph Laplacian, whose rows sum
        to zero; the largest row sum is 1.000e+00.
    """
    size = laplacian_op.in_size()
    if laplacian_op.out_size() != size:
        raise ValueError(
            "A structure matrix must be square, got "
            f"({laplacian_op.out_size()}, {size})."
        )
    if not lx.is_symmetric(laplacian_op):
        raise ValueError(
            "A structure matrix must be symmetric (tag it lx.symmetric_tag)."
        )
    dtype = laplacian_op.in_structure().dtype
    ones = jnp.ones(size, dtype=dtype)
    # R 1
    row_sums = laplacian_op.mv(ones)
    try:
        row_sums = np.asarray(row_sums)
        diagonal = np.asarray(diag(laplacian_op))
        # max |[R]_ii|
        scale = float(np.max(np.abs(diagonal)))
    except jax.errors.TracerArrayConversionError:
        pass  # traced values: the check is skipped
    else:
        tolerance = _ROW_SUM_TOLERANCE * max(scale, 1.0)
        if np.max(np.abs(row_sums), initial=0.0) > tolerance:
            raise ValueError(
                "A Besag structure matrix is a graph Laplacian, whose rows sum "
                f"to zero; the largest row sum is {np.max(np.abs(row_sums)):.3e}."
            )
    if isinstance(laplacian_op, SparseOperator):
        return SparseOperator(
            laplacian_op.values,
            laplacian_op.pattern,
            tags=laplacian_op.tags | _PSD | {lx.symmetric_tag},
        )
    if isinstance(laplacian_op, BlockTriDiag):
        return BlockTriDiag(
            laplacian_op.diagonal,
            laplacian_op.sub_diagonal,
            symmetric=laplacian_op.symmetric,
            tags=laplacian_op.tags | _PSD,
        )
    return lx.TaggedLinearOperator(laplacian_op, _PSD | {lx.symmetric_tag})


def generalized_variance_scale(
    structure: lx.AbstractLinearOperator,
    null_space: Float[ArrayLike, "n k"] | Float[ArrayLike, " n"],
    *,
    eps: float | None = None,
) -> Float[Array, ""]:
    """Generalized variance of an intrinsic GMRF (Sørbye & Rue, 2014).

        s = exp((1 / n) ∑_i log [Σ]_ii),   Σ = R⁺ on range(R)

    s is the geometric mean of the marginal variances under the constraint
    Vᵀ x = 0, for V the null space of R, so that s R has generalized variance one:
    the scaling that makes a precision τ mean the same thing for every graph (BYM2,
    scaled RW1 / RW2).

    Two exact paths:

    - **Eigen-structured** R (a `KroneckerSum` grid Laplacian, a
      `DiagonalizedOperator` or a `SpectralFunction`): the diagonal of the
      pseudo-inverse from the factor eigenvectors (`gaussx.diag_inv` with
      ``pinv=True``). This assumes ``null_space`` spans exactly the zero
      eigenspace.
    - **Anything else** (a `SparseOperator` graph Laplacian, a `BlockTriDiag` random
      walk): as R-INLA's ``inla.scale.model``, the marginal variances of R + β I
      from `gaussx.diag_inv` (the block selected inverse for a `BlockTriDiag`;
      sparse Cholesky / Takahashi for a `SparseOperator`), then the kriging
      correction for the constraint, at D_c solves:

          Σ = (R + β I)⁻¹ − W S⁻¹ Wᵀ,   W = (R + β I)⁻¹ V,   S = Vᵀ W

    **Graphs too large to factor.** The ridge path cancels two O(1 / β) terms, so
    it needs exact variances. Instead, with V orthonormal, (R + V Vᵀ)⁻¹ = R⁺ + V Vᵀ
    is a well-conditioned system, and

        [Σ]_ii = [(R + V Vᵀ)⁻¹]_ii − ∑_c [V]_ic²

    whose first term `gaussx.diag_inv` estimates with ``method="xdiag"`` and a CG
    solve on the matrix-free R + V Vᵀ (the last example below).

    For a disconnected graph scale each connected component separately.
    ``structure`` may have one more row than ``null_space``: the decoupled padding
    node of an odd-size `gaussx.rw2_structure`, which is then left out.

    Algorithm:
        1. Eigen-structured R with no padding row: read diag(R⁺) from the factors.
        2. Otherwise pad V with a zero row if R has a padding node, and set the
           ridge β.
        3. Factor R + β I once, and solve W = (R + β I)⁻¹ V.
        4. Correct: diag(Σ) = diag((R + β I)⁻¹) − diag(W S⁻¹ Wᵀ).
        5. Return the geometric mean of the first n variances.

    Args:
        structure: The structure matrix R (symmetric positive semidefinite).
        null_space: Basis V of the null space of R, shape ``(n, k)`` or ``(n,)``
            (e.g. the constants for a connected graph).
        eps: Ridge β. Defaults to √(machine eps) · max diag(R), the value R-INLA
            uses.

    Returns:
        The scalar s.

    Raises:
        ValueError: If the sizes disagree.

    Examples:
        Scale a first-order random walk, so that s R has generalized variance 1:

        >>> import einx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> structure = gaussx.rw1_structure(20)
        >>> scale = gaussx.generalized_variance_scale(structure, jnp.ones(20))
        >>> scale.shape, bool(scale > 0.0)
        ((), True)

        The sizes must agree, up to one padding row:

        >>> gaussx.generalized_variance_scale(structure, jnp.ones(3))
        Traceback (most recent call last):
        ...
        ValueError: null_space has 3 rows but the structure matrix has size 20.

        A graph too large to factor, through the matrix-free route above:

        >>> basis = jnp.ones(20) / jnp.linalg.norm(jnp.ones(20))
        >>> shifted = lx.TaggedLinearOperator(
        ...     gaussx.LowRankUpdate(structure, einx.id("n -> n 1", basis)),
        ...     lx.positive_semidefinite_tag,
        ... )
        >>> estimate = gaussx.diag_inv(
        ...     shifted,
        ...     method="xdiag",
        ...     num_probes=8,
        ...     key=jax.random.key(0),
        ...     solver=gaussx.CGSolver(),
        ... )
        >>> variances = estimate - basis**2
        >>> jnp.exp(jnp.mean(jnp.log(variances))).shape
        ()
    """
    null_basis = _as_float(null_space)
    if null_basis.ndim == 1:
        # The pattern is quoted by the AxisSizeError of an empty V: keep its text.
        null_basis = rearrange(null_basis, "n -> n 1")
    n = null_basis.shape[0]
    size = structure.in_size()
    if size not in (n, n + 1):
        raise ValueError(
            f"null_space has {n} rows but the structure matrix has size {size}."
        )
    if _has_eigenbasis(structure) and size == n:
        # diag(R⁺), from the factor eigenvectors
        variances = diag_inv(structure, pinv=True)
    else:
        if size == n + 1:
            padding = jnp.zeros((1, null_basis.shape[1]), dtype=null_basis.dtype)
            # V with a zero row on the padding node
            null_basis = jnp.concatenate([null_basis, padding])
        dtype = jnp.result_type(structure.in_structure().dtype, null_basis.dtype)
        if eps is None:
            machine_eps = jnp.finfo(dtype).eps
            root_eps = jnp.sqrt(machine_eps)
            diagonal = diag(structure)
            largest_diagonal = jnp.max(diagonal)
            # β = √eps · max diag(R)
            ridge = root_eps * largest_diagonal
        else:
            ridge = jnp.asarray(eps, dtype=dtype)
        # R + β I
        ridged_structure = _add_ridge(structure, ridge)
        # diag((R + β I)⁻¹) and the correction are both O(1 / β) and cancel to
        # O(1), so they must come from the same factorisation for the rounding
        # to cancel.
        if isinstance(ridged_structure, SparseOperator):
            factor = sparse_cholesky(ridged_structure)
            ridged_solve, ridged_diag_inv = factor.solve, factor.diag_inv
        else:
            ridged_solve = ft.partial(solve, ridged_structure)
            ridged_diag_inv = ft.partial(diag_inv, ridged_structure)
        column_solve = jax.vmap(ridged_solve, in_axes=1, out_axes=1)
        # W = (R + β I)⁻¹ V
        solved_null_space = column_solve(null_basis)
        # S = Vᵀ W. The pattern is quoted by the AxisSizeError of a V with no
        # columns: keep its text.
        constraint_covariance = einsum(null_basis, solved_null_space, "i a, i b -> a b")
        # S⁻¹
        constraint_inverse = jnp.linalg.inv(constraint_covariance)
        # diag(W S⁻¹ Wᵀ)
        correction = einsum(
            solved_null_space,
            constraint_inverse,
            solved_null_space,
            "state row, row col, state col -> state",
        )
        ridged_variances = ridged_diag_inv()
        # diag(Σ) = diag((R + β I)⁻¹) − diag(W S⁻¹ Wᵀ)
        variances = ridged_variances - correction
    # [Σ]_ii on the n nodes, without a padding node
    node_variances = variances[:n]
    log_variances = jnp.log(node_variances)
    # (1 / n) ∑_i log [Σ]_ii
    mean_log_variance = jnp.mean(log_variances)
    # s = exp((1 / n) ∑_i log [Σ]_ii)
    return jnp.exp(mean_log_variance)


def bym2_precision(
    structure_scaled: lx.AbstractLinearOperator,
    tau: Float[ArrayLike, ""],
    phi: Float[ArrayLike, ""],
) -> SparseOperator:
    """Joint precision Λ of the BYM2 pair (b, u*) (Riebler et al., 2016).

        b = (√(1 − φ) v + √φ u*) / √τ,   v ~ 𝒩(0, I)

        Λ = [ τ / (1 − φ) I        −√(τ φ) / (1 − φ) I ]
            [ −√(τ φ) / (1 − φ) I  R* + φ / (1 − φ) I  ]

    u* is the scaled ICAR field with structure R*, and the marginal covariance of
    b is ((1 − φ) I + φ R*⁺) / τ under the sum-to-zero constraint on u*. The
    pattern (that of R*, shifted, plus two diagonals) is built on the host once per
    pattern of R* and does not depend on (τ, φ).

    Algorithm:
        1. Read the stored values of R* (and any scalar multiple) from the operator.
        2. Fetch the cached plan: the joint pattern and where each value lands.
        3. Form the four raw blocks in plan order and sum them into the pattern.

    Args:
        structure_scaled: The scaled structure R* = s R (see
            `generalized_variance_scale`), a `SparseOperator` or a scalar multiple
            of one.
        tau: Precision τ > 0 of b (may be traced).
        phi: Mixing 0 ≤ φ < 1, the spatial share of the variance (may be traced).

    Returns:
        Λ as a symmetric positive-semidefinite `SparseOperator` of size 2n acting
        on the stacked vector (b, u*); singular along the null space of u*, which
        the sum-to-zero constraint removes.

    Raises:
        TypeError: If ``structure_scaled`` is not (a multiple of) a
            `SparseOperator`.

    Examples:
        A path of 3 areas gives a joint precision of size 6:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import numpy as np
        >>> import gaussx
        >>> laplacian = gaussx.SparseOperator.from_coo(
        ...     np.array([0, 1, 2, 1, 2]),
        ...     np.array([0, 1, 2, 0, 1]),
        ...     jnp.array([1.0, 2.0, 1.0, -1.0, -1.0]),
        ...     (3, 3),
        ...     symmetric=True,
        ... )
        >>> scale = gaussx.generalized_variance_scale(laplacian, jnp.ones(3))
        >>> precision = gaussx.bym2_precision(scale * laplacian, tau=1.5, phi=0.7)
        >>> precision.in_size(), type(precision).__name__
        (6, 'SparseOperator')

        The structure must be sparse:

        >>> dense = lx.MatrixLinearOperator(laplacian.as_matrix(), lx.symmetric_tag)
        >>> gaussx.bym2_precision(
        ...     dense, tau=1.5, phi=0.7
        ... )  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
        ...
        TypeError: bym2_precision needs the structure as a SparseOperator (or a
        scalar multiple of one), got MatrixLinearOperator.
    """
    values, pattern = _as_sparse(structure_scaled)
    tau = _as_float(tau)
    phi = _as_float(phi)
    n = pattern.shape[0]
    out_pattern, inverse = _bym2_plan(pattern)
    dtype = jnp.result_type(values, tau, phi)
    ones = jnp.ones(n, dtype=dtype)
    # The three 1 − φ stay separate: sharing one reorders the reverse-mode sum.
    # τ / (1 − φ), the b diagonal
    b_diagonal = tau / (1.0 - phi) * ones
    # √(τ φ)
    root_tau_phi = jnp.sqrt(tau * phi)
    # −√(τ φ) / (1 − φ), the coupling of b and u*
    coupling = -root_tau_phi / (1.0 - phi) * ones
    # R*, on its stored entries
    structure_values = values.astype(dtype)
    # φ / (1 − φ), the extra u* diagonal
    u_diagonal = phi / (1.0 - phi) * ones
    raw = jnp.concatenate([b_diagonal, coupling, structure_values, u_diagonal])
    # Stored values of Λ, summed over the repeated positions
    out = jax.ops.segment_sum(raw, inverse, num_segments=out_pattern.nnz)
    return SparseOperator(out, out_pattern, tags=_PSD)


def _has_eigenbasis(operator: lx.AbstractLinearOperator) -> bool:
    """Whether a structure has factor eigenvectors for diag(R⁺).

    Args:
        operator: The structure R, possibly tagged.

    Returns:
        ``True`` for a `KroneckerSum`, `DiagonalizedOperator` or `SpectralFunction`.
    """
    if isinstance(operator, lx.TaggedLinearOperator):
        return _has_eigenbasis(operator.operator)
    return isinstance(operator, KroneckerSum | DiagonalizedOperator | SpectralFunction)


def _add_ridge(
    operator: lx.AbstractLinearOperator, eps: Float[Array, ""]
) -> lx.AbstractLinearOperator:
    """Ridged structure R + β I, keeping a sparse or block-tridiagonal structure.

    Args:
        operator: The structure R; anything but a (tagged) `SparseOperator` or
            `BlockTriDiag` is densified.
        eps: The ridge β.

    Returns:
        R + β I, tagged positive semidefinite.
    """
    n = operator.in_size()
    if isinstance(operator, lx.TaggedLinearOperator) and isinstance(
        operator.operator, SparseOperator | BlockTriDiag
    ):
        operator = operator.operator
    if isinstance(operator, SparseOperator):
        ridge_diagonal = jnp.full(n, eps, dtype=operator.values.dtype)
        # R + β I on the pattern of R
        return operator.add_diagonal(ridge_diagonal, tags=_PSD)
    if isinstance(operator, BlockTriDiag):
        block_size = operator.diagonal.shape[-1]
        eye = jnp.eye(block_size, dtype=operator.diagonal.dtype)
        # [R]_t,t + β I, block by block
        ridged_diagonal = operator.diagonal + eps * eye
        return BlockTriDiag(
            ridged_diagonal,
            operator.sub_diagonal,
            symmetric=operator.symmetric,
            tags=_PSD,
        )
    matrix = operator.as_matrix()
    eye = jnp.eye(n, dtype=matrix.dtype)
    # R + β I, dense
    ridged_matrix = matrix + eps * eye
    return lx.MatrixLinearOperator(
        ridged_matrix,
        frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


def _as_sparse(
    operator: lx.AbstractLinearOperator,
) -> tuple[Float[Array, " nnz"], SparsityPattern]:
    """Stored lower-triangle values and pattern of a (scaled) `SparseOperator`.

    Args:
        operator: The structure R*: a `SparseOperator`, or one under any nesting of
            tags, scalar multiples and scalar divisions.

    Returns:
        Tuple ``(values, pattern)`` holding the scaled stored entries of R* and its
        symmetric pattern.

    Raises:
        TypeError: If no `SparseOperator` is found under the wrappers.
    """
    scale = None
    while isinstance(
        operator, lx.TaggedLinearOperator | lx.MulLinearOperator | lx.DivLinearOperator
    ):
        if isinstance(operator, lx.MulLinearOperator):
            scale = operator.scalar if scale is None else scale * operator.scalar
        elif isinstance(operator, lx.DivLinearOperator):
            inverse_scalar = 1.0 / operator.scalar
            scale = inverse_scalar if scale is None else scale * inverse_scalar
        operator = operator.operator
    if not isinstance(operator, SparseOperator):
        raise TypeError(
            "bym2_precision needs the structure as a SparseOperator (or a scalar "
            f"multiple of one), got {type(operator).__name__}."
        )
    values, pattern = operator.values, operator.pattern
    if not pattern.symmetric:
        values, pattern = _lower_storage(operator)
    if scale is not None:
        values = scale * values
    return values, pattern


def _lower_storage(
    operator: SparseOperator,
) -> tuple[Float[Array, " nnz"], SparsityPattern]:
    """Lower triangle of a symmetric matrix held in general storage.

    Args:
        operator: The matrix, stored in both triangles.

    Returns:
        Tuple ``(values, pattern)`` of the entries with row ≥ col, with a symmetric
        pattern.
    """
    pattern = operator.pattern
    keep = np.flatnonzero(pattern.rows >= pattern.cols)
    lower = SparsityPattern._from_canonical(
        pattern.rows[keep], pattern.cols[keep], pattern.shape, True
    )
    return operator.values[keep], lower


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _bym2_plan(pattern: SparsityPattern) -> tuple[SparsityPattern, np.ndarray]:
    """Pattern of the BYM2 joint precision and where each raw value lands.

    Raw values are, in order: the b diagonal, the (u*_i, b_i) coupling, the stored
    entries of R*, the extra u* diagonal.

    Args:
        pattern: The symmetric pattern of R*, of size n.

    Returns:
        Tuple ``(out_pattern, inverse)``: the pattern of Λ, of size 2n, and the
        position of each raw value in it.
    """
    n = pattern.shape[0]
    index = np.arange(n)
    rows = np.concatenate([index, n + index, n + pattern.rows, n + index])
    cols = np.concatenate([index, index, n + pattern.cols, n + index])
    out_rows, out_cols, inverse = _canonicalise(rows, cols, (2 * n, 2 * n), True)
    out = SparsityPattern._from_canonical(out_rows, out_cols, (2 * n, 2 * n), True)
    return out, inverse
