r"""Areal (graph) precision builders: Besag (ICAR), BYM2 and its scaling.

The Besag model $\tfrac{\tau}{2}\sum_{i\sim j}w_{ij}(x_i-x_j)^2 =
\tfrac{\tau}{2}x^\top Rx$ has the graph Laplacian $R$ as its structure
matrix. gaussx never builds graphs: $R$ arrives as an operator (kernellib's
``Graph.laplacian_operator()``), and so does its null space
(``graph_null_space``).
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
    r"""Validate a graph Laplacian as a Besag (ICAR) structure matrix.

    Checks that the operator is square and symmetric and, when its values
    are concrete, that its rows sum to zero (constants are in its null
    space), then tags it positive semidefinite. A `SparseOperator` or
    `BlockTriDiag` keeps its type (and so its sparse dispatch); any other
    operator is wrapped in a `lineax.TaggedLinearOperator`.

    Args:
        laplacian_op: The weighted graph Laplacian ``R = D − W``.

    Returns:
        The same matrix, tagged symmetric and positive semidefinite.

    Raises:
        ValueError: If it is not square or symmetric, or a row sum is not
            zero.

    Examples:
        ```python
        import jax.numpy as jnp
        import numpy as np
        import gaussx

        # Path graph 0 - 1 - 2, each edge once
        R = gaussx.SparseOperator.from_coo(
            np.array([0, 1, 2, 1, 2]),
            np.array([0, 1, 2, 0, 1]),
            jnp.array([1.0, 2.0, 1.0, -1.0, -1.0]),
            (3, 3),
            symmetric=True,
        )
        R = gaussx.besag_structure(R)  # now tagged positive semidefinite
        ```
    """
    n = laplacian_op.in_size()
    if laplacian_op.out_size() != n:
        raise ValueError(
            f"A structure matrix must be square, got ({laplacian_op.out_size()}, {n})."
        )
    if not lx.is_symmetric(laplacian_op):
        raise ValueError(
            "A structure matrix must be symmetric (tag it lx.symmetric_tag)."
        )
    row_sums = laplacian_op.mv(jnp.ones(n, dtype=laplacian_op.in_structure().dtype))
    try:
        row_sums = np.asarray(row_sums)
        scale = float(np.max(np.abs(np.asarray(diag(laplacian_op)))))
    except jax.errors.TracerArrayConversionError:
        pass  # traced values: the check is skipped
    else:
        if np.max(np.abs(row_sums), initial=0.0) > _ROW_SUM_TOLERANCE * max(scale, 1.0):
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
    r"""Generalized variance of an intrinsic GMRF (Sørbye & Rue, 2014).

    The geometric mean of the marginal variances under the constraint
    ``Vᵀx = 0`` (``V`` the null space),

    $$
    s = \exp\Big(\frac1n\sum_i\log\Sigma_{ii}\Big),\qquad
    \Sigma = R^{+}\ \text{on}\ \operatorname{range}(R),
    $$

    so that ``s · R`` has generalized variance one: the scaling that makes a
    precision ``τ`` mean the same thing for every graph (BYM2, scaled
    RW1 / RW2).

    Two exact paths:

    - **Eigen-structured** ``R`` (a `KroneckerSum` grid Laplacian, a
      `DiagonalizedOperator` or a `SpectralFunction`): the diagonal of the
      pseudo-inverse from the factor eigenvectors (`gaussx.diag_inv` with
      ``pinv=True``). This assumes ``null_space`` spans exactly the zero
      eigenspace.
    - **Anything else** (a `SparseOperator` graph Laplacian, a `BlockTriDiag`
      random walk): as R-INLA's ``inla.scale.model``, the marginal variances
      of ``R + εI`` from `gaussx.diag_inv` (the block selected inverse for a
      `BlockTriDiag`; sparse Cholesky / Takahashi for a `SparseOperator` as
      that dispatch lands), then the kriging correction for the constraint,
      ``Σ = S − S V (Vᵀ S V)⁻¹ Vᵀ S`` with ``S = (R + εI)⁻¹``, which needs
      ``k`` solves.

    For a disconnected graph scale each connected component separately.
    ``structure`` may have one more row than ``null_space``: the decoupled
    padding node of an odd-size `gaussx.rw2_structure`, which is then
    left out.

    Args:
        structure: The structure matrix ``R`` (symmetric PSD).
        null_space: Basis of its null space, shape ``(n, k)`` or ``(n,)``
            (e.g. the constants for a connected graph).
        eps: Ridge ``ε``. Defaults to ``√(machine eps) · max diag(R)``, the
            value R-INLA uses.

    Returns:
        The scalar ``s``.

    Raises:
        ValueError: If the sizes disagree.

    Examples:
        ```python
        import jax.numpy as jnp
        import gaussx

        R = gaussx.rw1_structure(20)
        s = gaussx.generalized_variance_scale(R, jnp.ones(20))
        # R_scaled = s * R has generalized variance 1
        ```
    """
    V = _as_float(null_space)
    if V.ndim == 1:
        V = rearrange(V, "n -> n 1")
    n = V.shape[0]
    size = structure.in_size()
    if size not in (n, n + 1):
        raise ValueError(
            f"null_space has {n} rows but the structure matrix has size {size}."
        )
    if _has_eigenbasis(structure) and size == n:
        variances = diag_inv(structure, pinv=True)
    else:
        if size == n + 1:
            V = jnp.concatenate([V, jnp.zeros((1, V.shape[1]), dtype=V.dtype)])
        dtype = jnp.result_type(structure.in_structure().dtype, V.dtype)
        if eps is None:
            eps_value = jnp.sqrt(jnp.finfo(dtype).eps) * jnp.max(diag(structure))
        else:
            eps_value = jnp.asarray(eps, dtype=dtype)
        S = _add_ridge(structure, eps_value)
        # diag(S⁻¹) and the correction are both O(1/ε) and cancel to O(1), so
        # they must come from the same factorisation for the rounding to cancel.
        if isinstance(S, SparseOperator):
            factor = sparse_cholesky(S)
            solve_S, diag_inv_S = factor.solve, factor.diag_inv
        else:
            solve_S, diag_inv_S = ft.partial(solve, S), ft.partial(diag_inv, S)
        W = jax.vmap(solve_S, in_axes=1, out_axes=1)(V)
        M = einsum(V, W, "i a, i b -> a b")
        correction = einsum(W, jnp.linalg.inv(M), W, "i a, a b, i b -> i")
        variances = diag_inv_S() - correction
    return jnp.exp(jnp.mean(jnp.log(variances[:n])))


def bym2_precision(
    structure_scaled: lx.AbstractLinearOperator,
    tau: Float[ArrayLike, ""],
    phi: Float[ArrayLike, ""],
) -> SparseOperator:
    r"""Joint precision of the BYM2 pair ``(b, u*)`` (Riebler et al., 2016).

    ``b = (√(1−φ) v + √φ u*)/√τ`` with ``v ~ N(0, I)`` and ``u*`` the scaled
    ICAR field (structure ``R*``) gives

    $$
    Q = \begin{pmatrix}
    \frac{\tau}{1-\phi}I & -\frac{\sqrt{\tau\phi}}{1-\phi}I\\
    -\frac{\sqrt{\tau\phi}}{1-\phi}I & R^* + \frac{\phi}{1-\phi}I
    \end{pmatrix},
    $$

    and the marginal covariance of ``b`` is
    ``((1−φ)I + φ R*⁺)/τ`` under ``u*``'s sum-to-zero constraint. The
    pattern (``R*``'s, shifted, plus two diagonals) is built on the host
    once per pattern of ``R*`` and does not depend on ``(τ, φ)``.

    Args:
        structure_scaled: The scaled structure ``R* = s R`` (see
            `generalized_variance_scale`), a `SparseOperator` or a scalar
            multiple of one.
        tau: Precision ``τ > 0`` of ``b`` (may be traced).
        phi: Mixing ``0 ≤ φ < 1``, the spatial share of the variance (may be
            traced).

    Returns:
        A symmetric positive-semidefinite `SparseOperator` of size ``2n``
        acting on the stacked vector ``(b, u*)``; singular along ``u*``'s
        null space, which the sum-to-zero constraint removes.

    Raises:
        TypeError: If ``structure_scaled`` is not (a multiple of) a
            `SparseOperator`.

    Examples:
        ```python
        import jax.numpy as jnp
        import numpy as np
        import gaussx

        R = gaussx.SparseOperator.from_coo(
            np.array([0, 1, 2, 1, 2]),
            np.array([0, 1, 2, 0, 1]),
            jnp.array([1.0, 2.0, 1.0, -1.0, -1.0]),
            (3, 3),
            symmetric=True,
        )
        s = gaussx.generalized_variance_scale(R, jnp.ones(3))
        Q = gaussx.bym2_precision(s * R, tau=1.5, phi=0.7)  # (6, 6)
        ```
    """
    values, pattern = _as_sparse(structure_scaled)
    tau = _as_float(tau)
    phi = _as_float(phi)
    n = pattern.shape[0]
    out_pattern, inverse = _bym2_plan(pattern)
    dtype = jnp.result_type(values, tau, phi)
    ones = jnp.ones(n, dtype=dtype)
    raw = jnp.concatenate(
        [
            tau / (1.0 - phi) * ones,
            -jnp.sqrt(tau * phi) / (1.0 - phi) * ones,
            values.astype(dtype),
            phi / (1.0 - phi) * ones,
        ]
    )
    out = jax.ops.segment_sum(raw, inverse, num_segments=out_pattern.nnz)
    return SparseOperator(out, out_pattern, tags=_PSD)


def _has_eigenbasis(operator: lx.AbstractLinearOperator) -> bool:
    if isinstance(operator, lx.TaggedLinearOperator):
        return _has_eigenbasis(operator.operator)
    return isinstance(operator, KroneckerSum | DiagonalizedOperator | SpectralFunction)


def _add_ridge(
    operator: lx.AbstractLinearOperator, eps: Float[Array, ""]
) -> lx.AbstractLinearOperator:
    """``R + εI``, keeping a sparse or block-tridiagonal structure."""
    n = operator.in_size()
    if isinstance(operator, lx.TaggedLinearOperator) and isinstance(
        operator.operator, SparseOperator | BlockTriDiag
    ):
        operator = operator.operator
    if isinstance(operator, SparseOperator):
        return operator.add_diagonal(
            jnp.full(n, eps, dtype=operator.values.dtype), tags=_PSD
        )
    if isinstance(operator, BlockTriDiag):
        d = operator.diagonal.shape[-1]
        eye = jnp.eye(d, dtype=operator.diagonal.dtype)
        return BlockTriDiag(
            operator.diagonal + eps * eye,
            operator.sub_diagonal,
            symmetric=operator.symmetric,
            tags=_PSD,
        )
    matrix = operator.as_matrix()
    return lx.MatrixLinearOperator(
        matrix + eps * jnp.eye(n, dtype=matrix.dtype),
        frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


def _as_sparse(
    operator: lx.AbstractLinearOperator,
) -> tuple[Float[Array, " nnz"], SparsityPattern]:
    """Stored lower-triangle values and pattern of a (scaled) `SparseOperator`."""
    scale = None
    while isinstance(
        operator, lx.TaggedLinearOperator | lx.MulLinearOperator | lx.DivLinearOperator
    ):
        if isinstance(operator, lx.MulLinearOperator):
            scale = operator.scalar if scale is None else scale * operator.scalar
        elif isinstance(operator, lx.DivLinearOperator):
            inv = 1.0 / operator.scalar
            scale = inv if scale is None else scale * inv
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
    """The lower triangle of a symmetric matrix held in general storage."""
    pattern = operator.pattern
    keep = np.flatnonzero(pattern.rows >= pattern.cols)
    lower = SparsityPattern._from_canonical(
        pattern.rows[keep], pattern.cols[keep], pattern.shape, True
    )
    return operator.values[keep], lower


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _bym2_plan(pattern: SparsityPattern) -> tuple[SparsityPattern, np.ndarray]:
    """Pattern of the BYM2 joint precision and where each raw value lands.

    Raw values are, in order: the ``b`` diagonal, the ``(u*_i, b_i)``
    coupling, ``R*``'s stored entries, the extra ``u*`` diagonal.
    """
    n = pattern.shape[0]
    idx = np.arange(n)
    rows = np.concatenate([idx, n + idx, n + pattern.rows, n + idx])
    cols = np.concatenate([idx, idx, n + pattern.cols, n + idx])
    out_rows, out_cols, inverse = _canonicalise(rows, cols, (2 * n, 2 * n), True)
    out = SparsityPattern._from_canonical(out_rows, out_cols, (2 * n, 2 * n), True)
    return out, inverse
