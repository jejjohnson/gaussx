"""Structured log-determinant with dispatch on operator type."""

from __future__ import annotations

import functools as ft
from typing import TYPE_CHECKING, Literal

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._block_tridiag import (
    BlockTriDiag,
    LowerBlockTriDiag,
    UpperBlockTriDiag,
)
from gaussx._operators._diagonalised import DiagonalizedOperator, as_diagonalized
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum, _eigh_factor
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    orthonormal_scaled_identity,
)
from gaussx._operators._sparse import _PLAN_CACHE_SIZE, SparseOperator, SparsityPattern
from gaussx._operators._spectral_function import SpectralFunction
from gaussx._operators._sum_kronecker import (
    SumOfKroneckers,
    _sum_of_kroneckers_eigen,
)
from gaussx._primitives._cholesky import warn_dense_fallback


if TYPE_CHECKING:
    from gaussx._strategies._base import AbstractLogdetStrategy


def cholesky_logdet(L: Float[Array, "N N"]) -> Float[Array, ""]:
    """Compute log|A| from Cholesky factor L where A = L Lᵀ.

    Args:
        L: Lower-triangular Cholesky factor, shape ``(N, N)``.

    Returns:
        Scalar log-determinant.
    """
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def logdet(operator: lx.AbstractLinearOperator) -> Float[Array, ""]:
    """Compute log |det(A)| with structural dispatch.

    Args:
        operator: The linear operator A.

    Returns:
        Scalar log |det(A)|.
    """
    if isinstance(operator, lx.IdentityLinearOperator):
        return jnp.zeros((), dtype=operator.in_structure().dtype)
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _logdet_diagonal(operator)
    if isinstance(operator, DiagonalizedOperator):
        return _logdet_diagonalised(operator)
    if isinstance(operator, BlockDiag):
        return _logdet_block_diag(operator)
    if isinstance(operator, Kronecker):
        return _logdet_kronecker(operator)
    if isinstance(operator, LowRankUpdate):
        if operator.rank == 0:
            return logdet(operator.base)
        c = orthonormal_scaled_identity(operator)
        if c is not None:
            return _logdet_low_rank_orthonormal(operator, c)
        return _logdet_low_rank(operator)
    if isinstance(operator, SumOfKroneckers):
        return _logdet_sum_of_kroneckers(operator)
    if isinstance(operator, KroneckerSum):
        return _logdet_kronecker_sum(operator)
    if isinstance(operator, SpectralFunction):
        return operator.logdet()
    if isinstance(operator, BlockTriDiag) and operator.symmetric:
        # Non-symmetric diagonal blocks take the dense slogdet (gh-344).
        return _logdet_block_tridiag(operator)
    if isinstance(operator, LowerBlockTriDiag | UpperBlockTriDiag):
        return _logdet_block_bidiagonal(operator)
    if isinstance(operator, SparseOperator):
        return _logdet_sparse(operator)
    if isinstance(operator, lx.TaggedLinearOperator):
        return logdet(operator.operator)
    if isinstance(operator, lx.MulLinearOperator):
        n = operator.out_size()
        return n * jnp.log(jnp.abs(operator.scalar)) + logdet(operator.operator)
    if isinstance(operator, lx.DivLinearOperator):
        n = operator.out_size()
        return logdet(operator.operator) - n * jnp.log(jnp.abs(operator.scalar))
    if isinstance(operator, lx.NegLinearOperator):
        # log|det(-A)| = log|det(A)| since |(-1)^n| = 1.
        return logdet(operator.operator)
    if isinstance(operator, lx.ComposedLinearOperator) and (
        operator.operator1.in_size() == operator.operator1.out_size()
        and operator.operator2.in_size() == operator.operator2.out_size()
    ):
        return logdet(operator.operator1) + logdet(operator.operator2)
    if isinstance(operator, lx.AddLinearOperator):
        factorization = _sum_of_kroneckers_eigen(operator)
        if factorization is not None:
            return factorization.logdet()
    return _logdet_dense(operator)


def _has_structural_logdet(operator: lx.AbstractLinearOperator) -> bool:
    """Whether `logdet` (and `gaussx.solve`) take an exact structural path.

    Mirrors `logdet`'s dispatch: ``True`` when the operator, after unwrapping
    `lineax.TaggedLinearOperator` / ``MulLinearOperator`` /
    ``DivLinearOperator`` / ``NegLinearOperator`` (and composing square
    factors), is one whose log-determinant is computed without materialising
    it and without a stochastic estimate. A large `SparseOperator`'s default
    logdet is a stochastic SLQ estimate, and a `Toeplitz` or a non-reducible
    sum of Kronecker products is materialised, so those are ``False``.

    Kept beside `logdet` so the two lists cannot drift apart; `AutoSolver`
    and `inv_quad_logdet` route on it.

    Args:
        operator: The linear operator.

    Returns:
        Whether the structural path applies.
    """
    if isinstance(
        operator,
        lx.IdentityLinearOperator
        | lx.DiagonalLinearOperator
        | DiagonalizedOperator
        | BlockDiag
        | Kronecker
        | LowRankUpdate
        | KroneckerSum
        | SpectralFunction
        | LowerBlockTriDiag
        | UpperBlockTriDiag,
    ):
        return True
    if isinstance(operator, BlockTriDiag):
        return operator.symmetric
    if isinstance(operator, SumOfKroneckers | lx.AddLinearOperator):
        return _sum_of_kroneckers_eigen(operator) is not None
    if isinstance(
        operator,
        lx.TaggedLinearOperator
        | lx.MulLinearOperator
        | lx.DivLinearOperator
        | lx.NegLinearOperator,
    ):
        return _has_structural_logdet(operator.operator)
    if isinstance(operator, lx.ComposedLinearOperator):
        first, second = operator.operator1, operator.operator2
        return (
            first.in_size() == first.out_size()
            and second.in_size() == second.out_size()
            and _has_structural_logdet(first)
            and _has_structural_logdet(second)
        )
    return False


def _logdet_sparse(operator: SparseOperator) -> Float[Array, ""]:
    """`SLQLogdet` when large and PSD (`AutoSolver` threshold), dense otherwise.

    The SLQ estimate uses the strategy's default fixed key. The exact sparse
    Cholesky path is the `SparseCholeskySolver` strategy, passed explicitly.
    """
    from gaussx._strategies._auto import AutoSolver
    from gaussx._strategies._slq_logdet import SLQLogdet

    if operator.in_size() > AutoSolver().size_threshold and (
        lx.is_positive_semidefinite(operator)
    ):
        return SLQLogdet().logdet(operator)
    return _logdet_dense(operator)


def _logdet_diagonal(operator: lx.DiagonalLinearOperator) -> Float[Array, ""]:
    diag = lx.diagonal(operator)
    return jnp.sum(jnp.log(jnp.abs(diag)))


def _logdet_block_diag(operator: BlockDiag) -> Float[Array, ""]:
    return ft.reduce(jnp.add, (logdet(op) for op in operator.operators))


def _logdet_kronecker(operator: Kronecker) -> Float[Array, ""]:
    """logdet(A1 kron A2 kron ... kron Ak).

    For two factors: logdet(A kron B) = n_B * logdet(A) + n_A * logdet(B).
    Generalizes to k factors.
    """
    total_size = operator.out_size()
    result = jnp.array(0.0)
    for op in operator.operators:
        n_i = op.out_size()
        # This factor's logdet is scaled by total_size / n_i
        result = result + (total_size // n_i) * logdet(op)
    return result


def _logdet_low_rank(operator: LowRankUpdate) -> Float[Array, ""]:
    """Matrix determinant lemma: det(L + U D V^T) = det(L) det(K).

    where K = I + D V^T L^{-1} U is the k x k capacitance scaled by D, so
    a zero weight needs no log(0) (gh-307).
    """
    from gaussx._primitives._solve import _low_rank_capacitance

    ld_base = logdet(operator.base)
    _, K = _low_rank_capacitance(operator, solver=None)
    _, ld_K = jnp.linalg.slogdet(K)
    return ld_base + ld_K


def _logdet_low_rank_orthonormal(
    operator: LowRankUpdate, c: Float[Array, ""]
) -> Float[Array, ""]:
    """``log|det(cI + U D Uᵀ)| = (n − k) log|c| + Σᵢ log|c + dᵢ|`` for ``UᵀU = I``.

    The eigenvalues are ``c + dᵢ`` on the span of ``U`` and ``c`` on its
    ``n − k``-dimensional complement; ``O(k)`` (gh-333).
    """
    n, k = operator.in_size(), operator.rank
    return (n - k) * jnp.log(jnp.abs(c)) + jnp.sum(jnp.log(jnp.abs(c + operator.d)))


def _logdet_sum_of_kroneckers(operator: SumOfKroneckers) -> Float[Array, ""]:
    r"""``logdet(Σ_k A_k ⊗ B_k)`` from the simultaneous diagonalization.

    For two terms with one positive definite this is
    ``Σ_ij log|λ_a[i] λ_b[j] + 1|`` plus the anchor's own scaled logdets —
    the same reduction `solve` uses, so the two agree by construction.
    Three or more terms keep the dense fallback; `SLQLogdet` estimates that
    case matrix-free.
    """
    factorization = _sum_of_kroneckers_eigen(operator)
    if factorization is None:
        warn_dense_fallback(
            "logdet(SumOfKroneckers) has no closed form here (three or more "
            "terms, or no positive-definite anchor) and materialises the "
            "operator; SLQLogdet estimates it matrix-free."
        )
        return _logdet_dense(operator)
    return factorization.logdet()


def _logdet_kronecker_sum(operator: KroneckerSum) -> Float[Array, ""]:
    """logdet(A (+) B) = sum(log(lambda_A_i + lambda_B_j)).

    The eigenvalues of ``A ⊕ B`` are ``λ_i(A) + μ_j(B)`` for any square
    factors. Each factor is handled on its own: a symmetric one (tagged,
    or diagonal) uses the shared ``_eigh_factor`` helper — the same
    routine the KroneckerSum solve path uses, ``O(n)`` for a diagonal.
    ``eigh`` reads only one triangle, so any other factor takes the
    general ``eigvals`` instead (still ``O(n³)`` in that factor alone;
    gh-308), mirroring the symmetry guard in `solve`. The general
    eigensolver runs on CPU (LAPACK ``geev``) and on CUDA GPUs
    (cuSOLVER / MAGMA ``geev``).
    """
    diagonalised = as_diagonalized(operator)
    if diagonalised is not None:
        return _logdet_diagonalised(diagonalised)
    evals_a = _factor_eigvals(operator.A)
    evals_b = _factor_eigvals(operator.B)
    eig_mat = einx.add("a, b -> a b", evals_a, evals_b)
    return jnp.sum(jnp.log(jnp.abs(eig_mat)))


def _factor_eigvals(operator: lx.AbstractLinearOperator) -> Array:
    """Eigenvalues of one Kronecker-sum factor: ``eigh`` iff symmetric."""
    if lx.is_symmetric(operator):
        return _eigh_factor(operator)[0]
    return jnp.linalg.eigvals(operator.as_matrix())


def _logdet_diagonalised(operator: DiagonalizedOperator) -> Float[Array, ""]:
    """``log|det A| = Σ log|λ|`` (``slogdet`` convention; ``−inf`` if singular)."""
    return jnp.sum(jnp.log(jnp.abs(operator.eigenvalues)))


def _logdet_block_tridiag(operator: BlockTriDiag) -> Float[Array, ""]:
    """logdet via banded Cholesky: logdet(A) = 2 * logdet(L)."""
    from gaussx._primitives._cholesky import cholesky

    L = cholesky(operator)
    return 2.0 * logdet(L)


def _logdet_block_bidiagonal(
    operator: LowerBlockTriDiag | UpperBlockTriDiag,
) -> Float[Array, ""]:
    """logdet of a block-bidiagonal operator with triangular diagonal blocks.

    Valid for both lower and upper variants: the determinant is the
    product of the diagonal blocks' determinants, and the blocks are
    triangular (Cholesky factors), so each reduces to its diagonal.
    """
    return jnp.sum(
        jax.vmap(lambda L: jnp.sum(jnp.log(jnp.abs(jnp.diag(L)))))(operator.diagonal)
    )


def _logdet_dense(operator: lx.AbstractLinearOperator) -> Float[Array, ""]:
    mat = operator.as_matrix()
    if _is_psd(operator):
        # A Cholesky is half an LU. A singular PSD matrix gives nan here (as
        # numpyro does), not slogdet's -inf: non-finite either way (gh-329).
        return _cholesky_logdet(jnp.linalg.cholesky(mat))
    _, ld = jnp.linalg.slogdet(mat)
    return ld


def _is_psd(operator: lx.AbstractLinearOperator) -> bool:
    """``lx.is_positive_semidefinite``, False where lineax does not know."""
    try:
        return bool(lx.is_positive_semidefinite(operator))
    except NotImplementedError:
        return False


def _cholesky_logdet(factor: Float[Array, "N N"]) -> Float[Array, ""]:
    """``log det(L Lᵀ)`` from a Cholesky factor ``L``."""
    return 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))


def _dense_psd_matrix(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, "N N"] | None:
    """The matrix of a PSD-tagged dense operator, looking through tags.

    ``None`` for anything structured, so callers keep its dispatch.
    """
    inner = operator
    while isinstance(inner, lx.TaggedLinearOperator):
        inner = inner.operator
    if not isinstance(inner, lx.MatrixLinearOperator) or not _is_psd(operator):
        return None
    return inner.matrix


def pseudo_logdet(
    operator: lx.AbstractLinearOperator,
    *,
    null_space: Float[ArrayLike, "N c"] | Float[ArrayLike, " N"] | None = None,
    structure: Literal["laplacian"] | None = None,
    rcond: float | None = None,
    strategy: AbstractLogdetStrategy | None = None,
) -> Float[Array, ""]:
    r"""Log pseudo-determinant: the log of the product of the non-zero eigenvalues.

    For a symmetric positive-semidefinite ``A``,
    $\log|A|_+ = \sum_{\lambda_i > 0}\log\lambda_i$. Intrinsic GMRFs (Besag /
    ICAR, RW1, RW2) have singular structure matrices ``R``, and
    $\tfrac12\log|R|_+$ is their normalising constant. Paths, cheapest first:

    - ``structure="laplacian"`` (a weighted graph Laplacian, e.g. Besag or
      RW1): the matrix-tree theorem. Every principal $(n_c-1)$-minor of a
      connected component's Laplacian $L_c$ equals its weighted spanning-tree
      count, and $\operatorname{pdet}(L_c) = n_c$ times it, so
      $\log|L|_+ = \sum_c (\log n_c + \log|L_c^{(-k_c)}|)$. The minors are
      taken by replacing one node per component with a unit diagonal, which
      keeps the sparsity pattern, so it is **one** sparse Cholesky
      (`SparseCholeskySolver` by default; G4) for a `SparseOperator`, or one
      banded Cholesky for a `BlockTriDiag` with ``1 × 1`` blocks (a path, as
      from `rw1_structure`). Components are found from the sparsity pattern
      on the host (once per pattern), so every stored off-diagonal weight
      must be non-zero. Gradients are exact. A `KroneckerSum` (a grid
      Laplacian) takes the eigenvalue path below instead.
    - ``null_space`` given: $\operatorname{pdet}(A) = \det(A + NN^\top)$
      for orthonormal $N$ spanning $\ker A$ (each zero eigenvalue becomes
      1). Any basis $B$ of the kernel works, since
      $\det(A + BB^\top) = \operatorname{pdet}(A)\det(B^\top B)$ and the
      second factor is subtracted. ``A + BBᵀ`` is a matvec-only lineax sum,
      so ``strategy=gaussx.SLQLogdet()`` estimates it matrix-free; the
      default is `gaussx.logdet` of it (dense). The matrix determinant lemma
      does not apply because ``A`` is singular.
    - A `KroneckerSum` (recursively, with dense, diagonal or
      `DiagonalizedOperator` factors) or a `DiagonalizedOperator`: all
      pairwise sums of the factor eigenvalues, with no factorisation of the
      full operator.
    - Anything else: dense ``eigvalsh``.

    The eigenvalue paths drop eigenvalues ``≤ rcond · λ_max``.

    **Constants are constant.** ``log|τR|_+ = rank(R) log τ + log|R|_+``, so
    ``log|R|_+`` never depends on hyperparameters: compute it once, outside
    any θ loop, and keep it.

    Args:
        operator: Symmetric positive-semidefinite operator ``A``.
        null_space: A basis of ``ker A``, shape ``(N, c)`` (or ``(N,)`` for
            ``c = 1``); for a connected graph Laplacian, ``1/√N``. kernellib's
            ``graph_null_space`` gives one column per connected component.
        structure: ``"laplacian"`` to use the matrix-tree theorem on a
            weighted graph Laplacian (`SparseOperator`, ``1 × 1``-block
            `BlockTriDiag` or `KroneckerSum`).
        rcond: Relative cut-off for the eigenvalue paths. Defaults to
            ``n · eps`` of the dtype, with ``n`` the largest dense
            eigenproblem solved (a factor's size for a `KroneckerSum`), as in
            ``numpy.linalg.matrix_rank``.
        strategy: Log-determinant strategy for the null-space path (of
            ``A + BBᵀ``) and the Laplacian path (of the minor). Not used by
            the eigenvalue paths.

    Returns:
        Scalar $\log|A|_+$.

    Raises:
        ValueError: If both ``null_space`` and ``structure`` are given, if
            ``strategy`` is given for an eigenvalue path, if ``structure`` is
            unknown, or if ``null_space`` has the wrong number of rows.
        TypeError: If ``structure="laplacian"`` is given an operator it has
            no Laplacian path for.

    Examples:
        ```python
        import jax.numpy as jnp
        import lineax as lx
        import numpy as np
        import gaussx

        # ICAR normalising constant on the path graph 0 - 1 - 2 - 3 (computed once)
        n = 4
        R = gaussx.SparseOperator.from_coo(
            np.r_[np.arange(n), np.arange(1, n)],
            np.r_[np.arange(n), np.arange(n - 1)],
            jnp.r_[jnp.array([1.0, 2.0, 2.0, 1.0]), -jnp.ones(n - 1)],
            (n, n),
            symmetric=True,
        )
        half_log_pdet = 0.5 * gaussx.pseudo_logdet(R, structure="laplacian")
        # pdet(R) = n · (one spanning tree) = 4
        assert jnp.allclose(2 * half_log_pdet, jnp.log(4.0))

        # The same through its null space, constants / √n
        ones = jnp.ones(n) / jnp.sqrt(n)
        assert jnp.allclose(gaussx.pseudo_logdet(R, null_space=ones), jnp.log(4.0))

        # On a grid there is no factorisation: Σ log of the non-zero λ^H_i + λ^W_j
        L = gaussx.rw1_structure(5).as_matrix()
        grid = gaussx.KroneckerSum(
            lx.MatrixLinearOperator(L, lx.symmetric_tag),
            lx.MatrixLinearOperator(L, lx.symmetric_tag),
        )
        half_log_pdet_grid = 0.5 * gaussx.pseudo_logdet(grid)
        ```
    """
    if structure not in (None, "laplacian"):
        raise ValueError(f"structure must be None or 'laplacian', got {structure!r}.")
    if null_space is not None and structure is not None:
        raise ValueError("Pass either null_space or structure, not both.")
    if isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    if null_space is not None:
        return _pseudo_logdet_null_space(operator, null_space, strategy)
    if structure == "laplacian":
        if isinstance(operator, SparseOperator):
            return _pseudo_logdet_laplacian_sparse(operator, strategy)
        if isinstance(operator, BlockTriDiag):
            return _pseudo_logdet_laplacian_banded(operator, strategy)
        if not isinstance(operator, KroneckerSum | DiagonalizedOperator):
            raise TypeError(
                "structure='laplacian' needs a SparseOperator, a BlockTriDiag "
                "with 1 x 1 blocks or a KroneckerSum, got "
                f"{type(operator).__name__}; pass null_space instead."
            )
    if strategy is not None:
        raise ValueError(
            "strategy is used by the null_space and structure='laplacian' "
            "paths only; the eigenvalue paths are exact."
        )
    eigenvalues, size = _spectrum(operator)
    if rcond is None:
        rcond = size * float(jnp.finfo(eigenvalues.dtype).eps)
    keep = eigenvalues > rcond * jnp.max(eigenvalues)
    logs = jnp.log(jnp.where(keep, eigenvalues, 1))
    return jnp.sum(jnp.where(keep, logs, 0))


def _spectrum(operator: lx.AbstractLinearOperator) -> tuple[Float[Array, " n"], int]:
    """Real eigenvalues of a symmetric operator, and the largest ``eigh`` size."""
    if isinstance(operator, lx.TaggedLinearOperator):
        return _spectrum(operator.operator)
    if isinstance(operator, KroneckerSum):
        a, n_a = _spectrum(operator.A)
        b, n_b = _spectrum(operator.B)
        pairs = einx.add("i, j -> i j", a, b)
        return rearrange(pairs, "i j -> (i j)"), max(n_a, n_b)
    if isinstance(operator, DiagonalizedOperator):
        return jnp.real(operator.eigenvalues_flat()), 1
    if isinstance(operator, lx.DiagonalLinearOperator):
        return lx.diagonal(operator), 1
    return jnp.linalg.eigvalsh(operator.as_matrix()), operator.in_size()


def _pseudo_logdet_null_space(
    operator: lx.AbstractLinearOperator,
    null_space: Float[ArrayLike, "N c"] | Float[ArrayLike, " N"],
    strategy: AbstractLogdetStrategy | None,
) -> Float[Array, ""]:
    """``log det(A + B Bᵀ) − log det(Bᵀ B)`` for a kernel basis ``B``."""
    dtype = operator.in_structure().dtype
    B = jnp.asarray(null_space, dtype=dtype)
    if B.ndim == 1:
        B = rearrange(B, "n -> n 1")
    if B.ndim != 2 or B.shape[0] != operator.in_size():
        raise ValueError(
            f"null_space must have shape ({operator.in_size()}, c), got {B.shape}."
        )

    def project(v: Float[Array, " N"]) -> Float[Array, " N"]:
        return einsum(B, einsum(B, v, "n c, n -> c"), "n c, c -> n")

    tags = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    low_rank = lx.FunctionLinearOperator(project, operator.in_structure(), tags)
    shifted = lx.TaggedLinearOperator(lx.AddLinearOperator(operator, low_rank), tags)
    ld = logdet(shifted) if strategy is None else strategy.logdet(shifted)
    _, ld_gram = jnp.linalg.slogdet(einsum(B, B, "n c, n d -> c d"))
    return ld - ld_gram


@ft.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _laplacian_plan(
    pattern: SparsityPattern,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per stored value: keep it, or the minor's unit diagonal; and ``Σ log n_c``.

    Connected components by hooking and pointer jumping over the pattern
    (each node ends up pointing at the smallest index in its component, its
    root); the roots are the nodes removed from each component.
    """
    n = pattern.shape[0]
    rows, cols = pattern.rows, pattern.cols
    parent = np.arange(n)
    while True:
        hooked = parent.copy()
        np.minimum.at(hooked, parent[rows], parent[cols])
        np.minimum.at(hooked, parent[cols], parent[rows])
        while not np.array_equal(hooked[hooked], hooked):
            hooked = hooked[hooked]
        if np.array_equal(hooked, parent):
            break
        parent = hooked
    is_root = parent == np.arange(n)
    touches_root = is_root[rows] | is_root[cols]
    unit = touches_root & (rows == cols)
    log_sizes = float(np.sum(np.log(np.bincount(parent, minlength=n)[is_root])))
    return ~touches_root, unit, log_sizes


def _pseudo_logdet_laplacian_sparse(
    operator: SparseOperator, strategy: AbstractLogdetStrategy | None
) -> Float[Array, ""]:
    """Matrix-tree theorem: ``Σ_c log n_c + log|minor|`` by one sparse Cholesky."""
    from gaussx._strategies._sparse_cholesky import SparseCholeskySolver

    if operator.pattern.shape[0] != operator.pattern.shape[1]:
        raise ValueError(f"A Laplacian must be square, got {operator.pattern.shape}.")
    keep, unit, log_sizes = _laplacian_plan(operator.pattern)
    values = operator.values
    minor_values = jnp.where(
        jnp.asarray(keep), values, jnp.asarray(unit, dtype=values.dtype)
    )
    minor = SparseOperator(
        minor_values,
        operator.pattern,
        tags=operator.tags | {lx.positive_semidefinite_tag},
    )
    solver = SparseCholeskySolver() if strategy is None else strategy
    return solver.logdet(minor) + jnp.asarray(log_sizes, dtype=values.dtype)


def _pseudo_logdet_laplacian_banded(
    operator: BlockTriDiag, strategy: AbstractLogdetStrategy | None
) -> Float[Array, ""]:
    """Matrix-tree theorem on a path: ``log n + log|L with the last node removed|``.

    The minor is the band with the last node replaced by a unit diagonal, so
    it is one banded Cholesky.
    """
    n, d, _ = operator.diagonal.shape
    if d != 1:
        raise ValueError(
            "structure='laplacian' on a BlockTriDiag needs 1 x 1 blocks (a "
            f"weighted path graph), got {d} x {d}; use a SparseOperator or "
            "null_space."
        )
    diagonal = operator.diagonal.at[-1].set(1)
    sub_diagonal = operator.sub_diagonal.at[n - 2 :].set(0)
    minor = BlockTriDiag(
        diagonal,
        sub_diagonal,
        symmetric=operator.symmetric,
        tags=operator.tags | {lx.positive_semidefinite_tag},
    )
    ld = logdet(minor) if strategy is None else strategy.logdet(minor)
    return ld + jnp.log(jnp.asarray(n, dtype=diagonal.dtype))
