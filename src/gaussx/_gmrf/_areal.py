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

from gaussx._deprecation import warn_deprecated
from gaussx._einx import rearrange
from gaussx._gmrf._intrinsic import pinv_diagonal
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


_PSD = frozenset({lx.positive_semidefinite_tag})
# Row sums of a Laplacian must vanish to this fraction of its largest diagonal, or to
# this many machine epsilons of its dtype where that is more (float32).
_ROW_SUM_TOLERANCE = 1e-8
_ROW_SUM_EPSILONS = 64.0


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
        scale = float(np.max(np.abs(diagonal), initial=0.0))
    except jax.errors.TracerArrayConversionError:
        pass  # traced values: the check is skipped
    else:
        machine_eps = float(np.finfo(row_sums.dtype).eps)
        relative = max(_ROW_SUM_TOLERANCE, _ROW_SUM_EPSILONS * machine_eps)
        tolerance = relative * max(scale, 1.0)
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
      walk): the rows and columns of D_c nodes P that determine the null space are
      replaced by a unit diagonal, which leaves a positive-definite matrix on R's
      pattern. Its inverse with zeros on P, G, is a generalised inverse of R, so

          Σ = P⊥ G P⊥,   P⊥ = I − V (Vᵀ V)⁻¹ Vᵀ,

      with no ridge to choose: one sparse or banded Cholesky, its selected inverse,
      and D_c solves. R-INLA's ``inla.scale.model`` adds a ridge of √eps times the
      diagonal instead, which overtakes the smallest eigenvalue of a second-order
      random walk near 200 nodes in float64.

    **Accuracy.** The grounded matrix of a path is conditioned like N² for RW1 and
    like N⁴ for RW2, so float64 carries an RW2 of a few thousand nodes and float32
    one of about fifty.

    **Graphs too large to factor.** With V orthonormal, (R + V Vᵀ)⁻¹ = R⁺ + V Vᵀ is a
    well-conditioned system, and

        [Σ]_ii = [(R + V Vᵀ)⁻¹]_ii − ∑_c [V]_ic²

    whose first term `gaussx.diag_inv` estimates with ``method="xdiag"`` and a CG
    solve on the matrix-free R + V Vᵀ (the last example below).

    For a disconnected graph scale each connected component separately.
    ``structure`` may have one more row than ``null_space``: the decoupled padding
    node of an odd-size `gaussx.rw2_structure`, which is then left out.

    Algorithm:
        1. Eigen-structured R with no padding row: read diag(R⁺) from the factors.
        2. Otherwise pad V with a zero row if R has a padding node.
        3. Ground R at the pivot nodes of V, factor it once, and project its
           inverse: diag(Σ) = diag(P⊥ G P⊥).
        4. Return the geometric mean of the first n variances.

    Args:
        structure: The structure matrix R (symmetric positive semidefinite).
        null_space: Basis V of the null space of R, shape ``(n, k)`` or ``(n,)``
            (e.g. the constants for a connected graph).
        eps: Deprecated and ignored: the scale takes no ridge. Removed in gaussx
            0.9.0.

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
    if eps is not None:
        warn_deprecated(
            "generalized_variance_scale's eps is deprecated, ignored, and will be "
            "removed in gaussx 0.9.0; the scale is exact and takes no ridge."
        )
    null_basis = _as_float(null_space)
    if null_basis.ndim == 1:
        # The pattern is quoted by the AxisSizeError of an empty V: keep its text.
        null_basis = rearrange(null_basis, "n -> n 1")
    weak_basis = null_basis
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
            # V with a zero row on the padding node
            null_basis = jnp.pad(null_basis, ((0, 1), (0, 0)))
        # the array, not its dtype: an integer V is weakly typed
        dtype = jnp.result_type(structure.in_structure().dtype, weak_basis)
        # diag(R⁺) = diag(P⊥ G P⊥)
        variances = pinv_diagonal(structure, null_basis.astype(dtype))
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
