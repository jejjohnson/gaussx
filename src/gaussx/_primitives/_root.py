"""Root and inverse-root decompositions A ≈ R Rᵀ and A⁻¹ ≈ R⁻ (R⁻)ᵀ.

A is a square symmetric positive-definite N × N operator. A root
decomposition is a tall N × K factor R with A ≈ R Rᵀ, and an inverse-root
decomposition a tall factor R⁻ with A⁻¹ ≈ R⁻ (R⁻)ᵀ, both keeping K ≤ N
directions. A symmetric A has the eigendecomposition A = Q diag(γ) Qᵀ, with
orthonormal eigenvectors Q and eigenvalues γ, and the singular value
decomposition A = U diag(s) Vᵀ; L is a lower Cholesky factor, A = L Lᵀ.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._symmetrize import symmetrize
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._eig import eig
from gaussx._primitives._inv import inv
from gaussx._primitives._svd import svd
from gaussx._randomized._rpcholesky import rp_cholesky


RootMethod = Literal["cholesky", "lanczos", "pivoted_cholesky", "svd"]


class RootDecomposition(eqx.Module):
    """Tall factor R of a root decomposition, A ≈ R Rᵀ.

        A ≈ R Rᵀ,   R of size N × K

    The same container holds an inverse root R⁻, with A⁻¹ ≈ R⁻ (R⁻)ᵀ.

    Attributes:
        root: The tall factor R, N × K.

    Examples:
        A rank-one factor and its action on a batch:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> decomposition = gaussx.RootDecomposition(jnp.array([[1.0], [2.0]]))
        >>> decomposition.rank
        1
        >>> decomposition.matmul(jnp.ones((3, 1))).shape
        (3, 2)
    """

    root: Float[Array, "N k"]

    @property
    def rank(self) -> int:
        """Number of retained root directions K."""
        return self.root.shape[1]

    def matmul(self, x: Float[Array, "*B k"]) -> Float[Array, "*B N"]:
        """Right-multiply a batch of coefficients by the root factor, X Rᵀ.

        Args:
            x: Coefficients X, with K entries in the last axis.

        Returns:
            The product X Rᵀ, with N entries in the last axis.
        """
        # jnp, not einx: einx cannot resolve an empty batch (*B with a 0).
        # X Rᵀ
        return x @ self.root.T


def root_decomposition(
    operator: lx.AbstractLinearOperator,
    rank: int = 50,
    method: RootMethod = "lanczos",
    key: jax.Array | None = None,
) -> RootDecomposition:
    """Root decomposition of an SPD operator: a tall factor R with A ≈ R Rᵀ.

        A ≈ R Rᵀ,   R of size N × K

    ``method`` chooses the factor:

    - ``"lanczos"``: R = Q_K diag(√γ_K) from the K largest eigenpairs.
    - ``"svd"``: R = U_K diag(√s_K) from the K leading singular pairs.
    - ``"pivoted_cholesky"``: a greedy pivoted partial Cholesky factor,
      on a dense A.
    - ``"cholesky"``: R = L, exact (``rank`` is ignored).

    A diagonal A takes the matching closed form without a dense
    eigendecomposition.

    Algorithm:
        1. Check that A is square and clamp K to N.
        2. A diagonal A keeps the square roots of its K largest entries.
        3. Otherwise build R from the method's factorisation, scaling each
           direction by the square root of its eigenvalue or singular value.

    Args:
        operator: The square symmetric positive-definite operator A.
        rank: Number of retained directions K. Ignored by ``"cholesky"``,
            which returns the exact full-rank factor.
        method: Decomposition method: ``"lanczos"``, ``"cholesky"``,
            ``"pivoted_cholesky"``, or ``"svd"``.
        key: PRNG key for random-start methods.

    Returns:
        A `RootDecomposition` holding R, N × K.

    Raises:
        ValueError: If A is not square, ``rank < 1`` (except for
            ``"cholesky"``), or ``method`` is unknown.

    Examples:
        A diagonal operator keeps its K largest entries:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([4.0, 1.0, 9.0]))
        >>> R = gaussx.root_decomposition(A, rank=2).root
        >>> R.shape
        (3, 2)
        >>> bool(jnp.allclose(R @ R.T, jnp.diag(jnp.array([4.0, 0.0, 9.0]))))
        True

        ``"cholesky"`` is exact on a dense positive-definite matrix:

        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> R = gaussx.root_decomposition(A, method="cholesky").root
        >>> bool(jnp.allclose(R @ R.T, matrix, atol=1e-5))
        True
    """
    _size, rank = _validate_square_rank(operator, rank, method=method)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return RootDecomposition(_diagonal_root(operator, rank, method, inverse=False))
    if method == "cholesky":
        # R = L
        return RootDecomposition(_cholesky_root(operator))
    if method == "lanczos":
        # R = Q_K diag(√γ_K)
        return RootDecomposition(_eig_root(operator, rank, key, inverse=False))
    if method == "svd":
        # R = U_K diag(√s_K)
        return RootDecomposition(_svd_root(operator, rank, key, inverse=False))
    if method == "pivoted_cholesky":
        matrix = _symmetric_operator_matrix(operator)
        return RootDecomposition(_pivoted_cholesky_root(matrix, rank))
    raise ValueError(f"Unknown root decomposition method: {method!r}")


def root_inv_decomposition(
    operator: lx.AbstractLinearOperator,
    rank: int = 50,
    method: RootMethod = "lanczos",
    key: jax.Array | None = None,
) -> RootDecomposition:
    """Inverse-root decomposition: a tall factor R⁻ with A⁻¹ ≈ R⁻ (R⁻)ᵀ.

        A⁻¹ ≈ R⁻ (R⁻)ᵀ,   R⁻ of size N × K

    ``method`` chooses the approximation target, not only the algorithm
    (gh-413):

    - ``"lanczos"`` / ``"svd"``: R⁻ (R⁻)ᵀ is A⁻¹ restricted to the dominant
      K-dimensional eigenspace of A (its *largest* eigenvalues, inverted),
      R⁻ = Q_K diag(γ_K^−½), GPyTorch's convention, used by
      `gaussx.love_cache`.
    - ``"pivoted_cholesky"``: R⁻ (R⁻)ᵀ is a rank-K approximation of A⁻¹
      itself, which favours the dominant subspace of A⁻¹ (the *smallest*
      eigenvalues of A); on a diagonal A = diag(a) it keeps the K largest
      1 / a_i, the Frobenius-optimal rank-K approximation of A⁻¹.
    - ``"cholesky"``: exact, R⁻ = L⁻ᵀ (``rank`` is ignored).

    Algorithm:
        1. Check that A is square and clamp K to N.
        2. A diagonal A keeps the inverse square roots of the K entries the
           method targets.
        3. Otherwise build R⁻ from the method's factorisation; the dense
           pivoted Cholesky path materialises A⁻¹ first.

    Args:
        operator: The square symmetric positive-definite operator A.
        rank: Number of retained directions K. Ignored by ``"cholesky"``,
            which returns the exact full-rank inverse-root factor.
        method: Decomposition method: ``"lanczos"``, ``"cholesky"``,
            ``"pivoted_cholesky"``, or ``"svd"``.
        key: PRNG key for random-start methods.

    Returns:
        A `RootDecomposition` holding R⁻, N × K.

    Raises:
        ValueError: If A is not square, ``rank < 1`` (except for
            ``"cholesky"``), or ``method`` is unknown.

    Examples:
        ``"pivoted_cholesky"`` keeps the largest 1 / a_i of a diagonal:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([4.0, 1.0, 9.0]))
        >>> R = gaussx.root_inv_decomposition(
        ...     A, rank=2, method="pivoted_cholesky"
        ... ).root
        >>> expected = jnp.diag(jnp.array([0.25, 1.0, 0.0]))
        >>> bool(jnp.allclose(R @ R.T, expected, atol=1e-6))
        True

        ``"cholesky"`` is exact on a dense positive-definite matrix:

        >>> matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> R = gaussx.root_inv_decomposition(A, method="cholesky").root
        >>> bool(jnp.allclose(R @ R.T, jnp.linalg.inv(matrix), atol=1e-5))
        True
    """
    size, rank = _validate_square_rank(operator, rank, method=method)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return RootDecomposition(_diagonal_root(operator, rank, method, inverse=True))
    if method == "cholesky":
        # A = L Lᵀ
        cholesky_factor = _cholesky_root(operator)
        identity = jnp.eye(size, dtype=cholesky_factor.dtype)
        # L⁻¹
        inverse_factor = jax.scipy.linalg.solve_triangular(
            cholesky_factor, identity, lower=True
        )
        # jnp, not einx: einx cannot resolve an empty operator (N = 0).
        # R⁻ = L⁻ᵀ, so R⁻ (R⁻)ᵀ = L⁻ᵀ L⁻¹ = A⁻¹
        return RootDecomposition(inverse_factor.T)
    if method == "lanczos":
        # R⁻ = Q_K diag(γ_K^−½)
        return RootDecomposition(_eig_root(operator, rank, key, inverse=True))
    if method == "svd":
        # R⁻ = U_K diag(s_K^−½)
        return RootDecomposition(_svd_root(operator, rank, key, inverse=True))
    if method == "pivoted_cholesky":
        # Dense fallback: forms A⁻¹ explicitly via the lazy inverse, then runs
        # pivoted Cholesky on it, O(N³) time and O(N²) memory. Prefer
        # ``method="lanczos"`` for large operators where structural solves
        # and matvecs are available.
        inverse_matrix = _symmetric_operator_matrix(inv(operator))
        return RootDecomposition(_pivoted_cholesky_root(inverse_matrix, rank))
    raise ValueError(f"Unknown inverse-root decomposition method: {method!r}")


def _validate_square_rank(
    operator: lx.AbstractLinearOperator,
    rank: int,
    *,
    method: RootMethod = "lanczos",
) -> tuple[int, int]:
    """Validate that A is square and clamp K to its size N.

    ``method="cholesky"`` returns the exact full-rank factor, so ``rank``
    is documented as ignored; in that case the ``rank >= 1`` guard is
    skipped and the returned rank is N, so callers can pass any sentinel
    value.

    Args:
        operator: The operator A.
        rank: Requested number of directions K.
        method: Decomposition method.

    Returns:
        Tuple ``(size, rank)`` holding N and min(K, N), or (N, N) for
        ``"cholesky"``.

    Raises:
        ValueError: If A is not square, or ``rank < 1`` for a method other
            than ``"cholesky"``.
    """
    if operator.in_size() != operator.out_size():
        raise ValueError("root decompositions require a square operator")
    size = operator.in_size()
    if method == "cholesky":
        return size, size
    if rank < 1:
        raise ValueError("rank must be at least 1")
    return size, min(rank, size)


def _diagonal_root(
    operator: lx.DiagonalLinearOperator,
    rank: int,
    method: RootMethod,
    *,
    inverse: bool,
) -> Float[Array, "N k"]:
    """Root or inverse-root factor of a diagonal A = diag(a), with no dense eig.

        R = diag(√a) or diag(a^−½), restricted to K selected columns

    ``"cholesky"`` keeps every column; the other methods keep the K largest
    a_i, except the inverse ``"pivoted_cholesky"``, which keeps the K
    largest 1 / a_i.

    Args:
        operator: Diagonal operator diag(a).
        rank: Number of retained directions K.
        method: Decomposition method.
        inverse: Whether to build R⁻ instead of R.

    Returns:
        The factor, N × K (N × N for ``"cholesky"``).

    Raises:
        ValueError: If ``method`` is unknown.
    """
    # Re a
    diagonal = jnp.real(lx.diagonal(operator))
    if method == "cholesky":
        # diag(√a) or diag(a^−½)
        return jnp.diag(_root_scales(diagonal, inverse=inverse))
    if method not in ("lanczos", "svd", "pivoted_cholesky"):
        target = "inverse-root" if inverse else "root"
        raise ValueError(f"Unknown {target} decomposition method: {method!r}")

    size = diagonal.shape[0]
    if inverse and method == "pivoted_cholesky":
        # 1 / a_i, the diagonal of A⁻¹
        order_scores = _safe_inverse_sqrt(diagonal) ** 2
    else:
        order_scores = diagonal
    ascending = jnp.argsort(order_scores)
    descending = ascending[::-1]
    # The K largest scores, descending
    order = descending[:rank]
    selected = diagonal[order]
    # √a_i or a_i^−½ on the kept entries
    scales = _root_scales(selected, inverse=inverse)
    root = jnp.zeros((size, rank), dtype=scales.dtype)
    # R_{order_j, j} = scale_j
    return root.at[order, jnp.arange(rank)].set(scales)


def _cholesky_root(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, "N N"]:
    """Exact Cholesky root L, structured until the final densification.

        A = L Lᵀ

    Args:
        operator: The SPD operator A.

    Returns:
        The dense real lower factor L, N × N.
    """
    # A = L Lᵀ, through the structured cholesky dispatch
    factor = cholesky(operator)
    return jnp.real(factor.as_matrix())


def _eig_root(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
    *,
    inverse: bool,
) -> Float[Array, "N k"]:
    """Root or inverse root from the K largest real eigenpairs of A.

        R = Q_K diag(√γ_K)   or   R⁻ = Q_K diag(γ_K^−½)

    Args:
        operator: The SPD operator A.
        rank: Number of retained eigenpairs K.
        key: PRNG key for the partial eigendecomposition, or ``None``.
        inverse: Whether to build R⁻ instead of R.

    Returns:
        The factor, N × K.
    """
    # A Q_K = Q_K diag(γ_K)
    eigenvalues, eigenvectors = eig(operator, rank=rank, key=key)
    eigenvalues, eigenvectors = _top_real_eigenpairs(eigenvalues, eigenvectors, rank)
    # √γ_K or γ_K^−½
    scales = _root_scales(eigenvalues, inverse=inverse)
    # jnp, not einx: einx cannot resolve an empty operator (N = K = 0).
    # Q_K diag(√γ_K)
    return eigenvectors * scales[None, :]


def _svd_root(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
    *,
    inverse: bool,
) -> Float[Array, "N k"]:
    """Root or inverse root from the K leading left singular vectors of A.

        R = U_K diag(√s_K)   or   R⁻ = U_K diag(s_K^−½)

    Args:
        operator: The SPD operator A.
        rank: Number of retained singular pairs K.
        key: PRNG key for the partial SVD, or ``None``.
        inverse: Whether to build R⁻ instead of R.

    Returns:
        The factor, N × K.
    """
    # A ≈ U diag(s) Vᵀ
    left_vectors, singular_values, _right_vectors_t = svd(operator, rank=rank, key=key)
    ascending = jnp.argsort(jnp.real(singular_values))
    descending = ascending[::-1]
    # The K largest singular values, descending
    order = descending[:rank]
    # U_K
    left_vectors = jnp.real(left_vectors[:, order])
    # s_K
    singular_values = jnp.real(singular_values[order])
    # √s_K or s_K^−½
    scales = _root_scales(singular_values, inverse=inverse)
    # jnp, not einx: einx cannot resolve an empty operator (N = K = 0).
    # U_K diag(√s_K)
    return left_vectors * scales[None, :]


def _top_real_eigenpairs(
    eigenvalues: Array,
    eigenvectors: Array,
    rank: int,
) -> tuple[Float[Array, " k"], Float[Array, "N k"]]:
    """The K largest real eigenpairs, in descending eigenvalue order.

    Args:
        eigenvalues: Eigenvalues γ.
        eigenvectors: Eigenvectors Q, one per column.
        rank: Number of eigenpairs K to keep.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding Re γ_K and Re Q_K.
    """
    eigenvalues = jnp.real(eigenvalues)
    eigenvectors = jnp.real(eigenvectors)
    ascending = jnp.argsort(eigenvalues)
    descending = ascending[::-1]
    # The K largest eigenvalues, descending
    order = descending[:rank]
    return eigenvalues[order], eigenvectors[:, order]


def _safe_psd(values: Array) -> Array:
    """Clamp eigenvalues or singular values to the smallest positive normal.

        max(γ, tiny)

    Args:
        values: Eigenvalues or singular values.

    Returns:
        The values, floored at the dtype's smallest positive normal number.
    """
    floor = jnp.finfo(values.dtype).tiny
    return jnp.maximum(values, floor)


def _root_scales(values: Array, *, inverse: bool) -> Array:
    """Root or inverse-root column scales from spectral values.

        √max(γ, tiny)   or   max(γ, tiny)^−½

    Args:
        values: Eigenvalues or singular values.
        inverse: Whether to return inverse-root scales.

    Returns:
        The column scales.
    """
    if inverse:
        return _safe_inverse_sqrt(values)
    # √max(γ, tiny)
    return jnp.sqrt(_safe_psd(values))


def _safe_inverse_sqrt(values: Array) -> Array:
    """Finite inverse square root of a non-negative spectrum.

        max(γ, tiny)^−½

    Args:
        values: Eigenvalues or singular values.

    Returns:
        The inverse square roots, finite even at zero.
    """
    # max(γ, tiny)^−½
    return jax.lax.rsqrt(_safe_psd(values))


def _symmetric_matrix(matrix: Array) -> Array:
    """Symmetrise a dense covariance-like matrix after numerical operations.

        ½ (M + Mᵀ)

    Args:
        matrix: Dense square matrix M.

    Returns:
        The symmetric part of M.
    """
    return symmetrize(matrix)


def _symmetric_operator_matrix(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, "N N"]:
    """Materialise A, symmetrised, for the dense fallback algorithms only.

    Args:
        operator: The operator A, densified here.

    Returns:
        The symmetric dense matrix ½ (A + Aᵀ).
    """
    return _symmetric_matrix(operator.as_matrix())


def _pivoted_cholesky_root(
    matrix: Float[Array, "N N"],
    rank: int,
) -> Float[Array, "N k"]:
    """Pivoted-Cholesky root of a dense matrix via `guarded_pivoted_cholesky`.

        M ≈ L_K L_Kᵀ

    Args:
        matrix: Dense symmetric PSD matrix M.
        rank: Number of pivots K.

    Returns:
        The factor L_K, N × K.
    """
    diagonal = jnp.diag(matrix)
    return guarded_pivoted_cholesky(diagonal, lambda pivot: matrix[:, pivot], rank)


def guarded_pivoted_cholesky(
    diagonal: Float[Array, " N"],
    column,
    rank: int,
) -> Float[Array, "N k"]:
    """Greedy pivoted partial Cholesky with a numerical-rank guard.

        M ≈ L_K L_Kᵀ,   L_K of size N × K

    A thin wrapper around `gaussx.rp_cholesky` with ``pivoting="greedy"``
    that drops the pivots.

    Needs only the diagonal and a ``column(k)`` callable returning column
    k of the matrix, so it serves a dense matrix and a matrix-free
    operator (one matvec per pivot) alike.

    matfree's ``cholesky_partial_pivot`` divides by the pivot without a
    positivity guard, so once the numerical rank is exhausted the
    residual pivot is rounding noise: an exactly nonpositive value
    yields NaN columns (which ``nan_to_num`` used to zero), but a tiny
    positive one yields huge or infinite columns that ``nan_to_num``
    passed through, the gh-236 failure. Guard each step instead: when
    the best remaining pivot falls below a pstrf-style tolerance, the
    surplus columns are exactly zero.

    Args:
        diagonal: Diagonal of the PSD matrix M, of length N.
        column: Callable returning column k of M, of length N.
        rank: Number of pivots K.

    Returns:
        The factor L_K, N × K, with L_K L_Kᵀ approximating M; columns past
        its numerical rank are exactly zero.

    Examples:
        A rank-one matrix leaves the surplus column exactly zero:

        >>> import jax.numpy as jnp
        >>> from gaussx._primitives._root import guarded_pivoted_cholesky
        >>> u = jnp.array([1.0, 2.0, 3.0])
        >>> matrix = jnp.outer(u, u)
        >>> L = guarded_pivoted_cholesky(jnp.diag(matrix), lambda k: matrix[:, k], 2)
        >>> L.shape, bool(jnp.all(L[:, 1] == 0.0))
        ((3, 2), True)
        >>> bool(jnp.allclose(L @ L.T, matrix, atol=1e-5))
        True
    """
    factor, _ = rp_cholesky(diagonal, column, rank, pivoting="greedy")
    return factor
