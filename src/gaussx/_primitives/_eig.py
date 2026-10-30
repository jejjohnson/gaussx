"""Structured eigendecomposition A = Q diag(γ) Q⁻¹, dispatched on the type of A.

A is a square N × N operator with eigenvalues γ_i and eigenvectors in the
columns of Q; a symmetric A has orthonormal Q and A = Q diag(γ) Qᵀ. A
Kronecker product is A = A_1 ⊗ … ⊗ A_D with square factors A_d,
d = 1, …, D, and a Kronecker sum is A_1 ⊕ A_2 = A_1 ⊗ I + I ⊗ A_2. K ≤ N is
the number of eigenpairs kept. The generalised problem A v = γ B v pairs a
symmetric A with a symmetric positive-semidefinite B; its eigenvectors are
B-orthonormal, Qᵀ B Q = I. B has its own eigendecomposition
B = Q_B diag(γ_B) Q_Bᵀ, and L is a lower Cholesky factor, B = L Lᵀ.
"""

from __future__ import annotations

import functools as ft
import warnings
from typing import Literal

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import matfree.decomp
import matfree.eig
import numpy as np
from jax.core import Tracer
from jax.scipy.linalg import block_diag as _block_diag, cho_solve, solve_triangular
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._randomized._svd import randomized_eigh


def eig(
    operator: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    method: Literal["lanczos", "randomized"] = "lanczos",
    key: jax.Array | None = None,
) -> tuple[Array, Array]:
    """Eigenvalues and eigenvectors of A, with structural dispatch.

        A Q = Q diag(γ)

    A symmetric operator returns real eigenvalues through ``eigh``.

    ``rank`` follows one rule shared with `eigvals` and `gaussx.svd`: an
    operator with a structured decomposition (diagonal, `gaussx.BlockDiag`,
    `gaussx.Kronecker`, `gaussx.KroneckerSum`) is decomposed exactly and
    truncated to its K = ``rank`` largest eigenvalues (by real part,
    descending). Any other operator gets a partial eigendecomposition of a
    symmetric operator without materialising A, by ``method``:

    - ``"lanczos"`` (default): matfree Lanczos.
    - ``"randomized"``: `randomized_eigh` with its defaults
      (``oversample=10``, ``n_power_iter=2``, ``which="largest"``); call it
      directly to tune them. Use ``n_power_iter >= 2`` for slowly decaying
      spectra.

    Randomized methods target the **top** of the spectrum (the eigenvalues
    of largest magnitude).

    Algorithm:
        1. Peel tags off A; a PSD or NSD tag marks a real symmetric A.
        2. Decompose a diagonal, block-diagonal, Kronecker or Kronecker-sum
           A exactly from its blocks or factors, then keep the top K.
        3. Otherwise, with ``rank`` given, run a partial Lanczos or
           randomized eigendecomposition.
        4. Otherwise densify A and call ``eigh`` (symmetric) or ``eig``.

    Args:
        operator: The square operator A.
        rank: Number of eigenpairs K. If ``None``, computes the full
            eigendecomposition.
        method: Partial-eig algorithm, ``"lanczos"`` or ``"randomized"``.
            Only used when ``rank`` is given.
        key: PRNG key for the initial random vector (Lanczos) or the
            Gaussian test matrix (randomized) when using partial eig.
            If ``None``, uses ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ, of length K, and
        Q, N × K.

    Raises:
        ValueError: If ``method`` is invalid, or ``"randomized"`` without
            ``rank``.

    Examples:
        A Kronecker product of diagonals, from its factors' spectra:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> A_2 = lx.DiagonalLinearOperator(jnp.array([3.0, 5.0]))
        >>> A = gaussx.Kronecker(A_1, A_2)  # diag(3, 5, 6, 10)
        >>> eigenvalues, eigenvectors = gaussx.eig(A)
        >>> [round(float(g), 4) for g in eigenvalues]
        [3.0, 5.0, 6.0, 10.0]

        ``rank`` keeps the K largest, in descending order:

        >>> eigenvalues, eigenvectors = gaussx.eig(A, rank=2)
        >>> [round(float(g), 4) for g in eigenvalues], eigenvectors.shape
        ([10.0, 6.0], (4, 2))

        A PSD-tagged dense matrix takes ``eigh``:

        >>> matrix = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        >>> A = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> eigenvalues, Q = gaussx.eig(A)
        >>> [round(float(g), 4) for g in eigenvalues]
        [1.0, 3.0]
        >>> bool(jnp.allclose(Q @ jnp.diag(eigenvalues) @ Q.T, matrix, atol=1e-5))
        True
    """
    if method not in ("lanczos", "randomized"):
        raise ValueError(f"method must be 'lanczos' or 'randomized', got {method!r}.")
    if method == "randomized" and rank is None:
        raise ValueError("method='randomized' needs a rank.")
    return _eig(operator, rank, method, key, symmetric=False)


def _eig(
    operator: lx.AbstractLinearOperator,
    rank: int | None,
    method: Literal["lanczos", "randomized"],
    key: jax.Array | None,
    *,
    symmetric: bool,
) -> tuple[Array, Array]:
    """`eig` dispatch; ``symmetric`` carries an outer wrapper's tags (gh-314).

        A Q = Q diag(γ)

    Args:
        operator: The square operator A.
        rank: Number of eigenpairs K, or ``None`` for all of them.
        method: Partial-eig algorithm, ``"lanczos"`` or ``"randomized"``.
        key: PRNG key for a partial eigendecomposition, or ``None``.
        symmetric: Whether an enclosing tag already marked A as real
            symmetric.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ and Q.
    """
    if isinstance(operator, lx.TaggedLinearOperator):
        # Tags cannot change the spectrum, but a PSD/NSD tag implies a real
        # symmetric operator, so the inner one takes ``eigh`` (gh-314).
        return _eig(
            operator.operator,
            rank,
            method,
            key,
            symmetric=symmetric or _is_real_symmetric(operator),
        )
    structured = _eig_structured(operator, symmetric=symmetric)
    if structured is not None:
        # Exact structured decomposition first, then the top ``rank`` (gh-383).
        eigenvalues, eigenvectors = structured
        if rank is None:
            return eigenvalues, eigenvectors
        top_indices = _top_order(eigenvalues, rank)
        # γ_1:K and Q_:,1:K, the K largest by real part
        return eigenvalues[top_indices], eigenvectors[:, top_indices]
    if rank is not None:
        if method == "randomized":
            return randomized_eigh(operator, rank, key=key)
        return _eig_partial(operator, rank, key)
    return _eig_dense(operator, symmetric=symmetric)


def _eig_structured(
    operator: lx.AbstractLinearOperator,
    *,
    symmetric: bool,
) -> tuple[Array, Array] | None:
    """Exact eigendecomposition of a structured A, or ``None`` if unstructured.

    Args:
        operator: The square operator A.
        symmetric: Whether A is known to be real symmetric.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ and Q, or ``None``
        when A has no structured rule.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _eig_diagonal(operator)
    # Per-factor paths need square factors; a square product of rectangular
    # ones (e.g. ``A ⊗ Aᵀ``) is decomposed whole. The outer symmetry carries
    # into the blocks/factors, so they take ``eigh`` too.
    if isinstance(operator, BlockDiag) and _square_factors(operator):
        return _eig_block_diag(operator, symmetric=symmetric)
    if isinstance(operator, Kronecker) and _square_factors(operator):
        return _eig_kronecker(operator, symmetric=symmetric)
    if isinstance(operator, KroneckerSum):
        return _eig_kronecker_sum(operator, symmetric=symmetric)
    return None


def _top_order(eigenvalues: Array, rank: int) -> Array:
    """Indices of the K = ``rank`` largest eigenvalues (by real part), descending.

    Args:
        eigenvalues: Eigenvalues γ.
        rank: Number of indices K to keep.

    Returns:
        The K indices i that sort Re γ_i in descending order.
    """
    # −Re γ, so an ascending sort puts the largest first
    negated = -jnp.real(eigenvalues)
    sorted_indices = jnp.argsort(negated)
    return sorted_indices[:rank]


def eigvals(
    operator: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    key: jax.Array | None = None,
) -> Array:
    """Eigenvalues of A only, with structural dispatch.

        γ_i,   A q_i = γ_i q_i

    When ``rank`` is given, returns the top K eigenvalues: exactly and
    truncated for an operator with a structured decomposition (as in
    `eig`), via matfree Lanczos without materialising A otherwise.

    Algorithm:
        1. Peel tags off A; a PSD or NSD tag marks a real symmetric A.
        2. Take a diagonal, block-diagonal, Kronecker or Kronecker-sum A's
           eigenvalues from its blocks or factors, then keep the top K.
        3. Otherwise, with ``rank`` given, run a partial Lanczos
           eigendecomposition.
        4. Otherwise densify A and call ``eigvalsh`` or ``eigvals``.

    Args:
        operator: The square operator A.
        rank: Number of eigenvalues K to compute.
        key: PRNG key for partial eigendecomposition.

    Returns:
        The eigenvalues γ, of length K.

    Examples:
        A Kronecker sum has the pairwise sums of its factors' eigenvalues:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> A_2 = lx.DiagonalLinearOperator(jnp.array([10.0, 20.0]))
        >>> A = gaussx.KroneckerSum(A_1, A_2)
        >>> [round(float(g), 4) for g in gaussx.eigvals(A)]
        [11.0, 21.0, 12.0, 22.0]

        ``rank`` keeps the K largest, in descending order:

        >>> [round(float(g), 4) for g in gaussx.eigvals(A, rank=2)]
        [22.0, 21.0]
    """
    return _eigvals(operator, rank, key, symmetric=False)


def _eigvals(
    operator: lx.AbstractLinearOperator,
    rank: int | None,
    key: jax.Array | None,
    *,
    symmetric: bool,
) -> Array:
    """`eigvals` dispatch; ``symmetric`` as in ``_eig`` (gh-314).

    Args:
        operator: The square operator A.
        rank: Number of eigenvalues K, or ``None`` for all of them.
        key: PRNG key for a partial eigendecomposition, or ``None``.
        symmetric: Whether an enclosing tag already marked A as real
            symmetric.

    Returns:
        The eigenvalues γ.
    """
    if isinstance(operator, lx.TaggedLinearOperator):
        return _eigvals(
            operator.operator,
            rank,
            key,
            symmetric=symmetric or _is_real_symmetric(operator),
        )
    structured = _eigvals_structured(operator, symmetric=symmetric)
    if structured is not None:
        # Exact structured eigenvalues first, then the top ``rank`` (gh-383).
        if rank is None:
            return structured
        top_indices = _top_order(structured, rank)
        # γ_1:K, the K largest by real part
        return structured[top_indices]
    if rank is not None:
        eigenvalues, _ = _eig_partial(operator, rank, key)
        return eigenvalues
    return _eigvals_dense(operator, symmetric=symmetric)


def _eigvals_structured(
    operator: lx.AbstractLinearOperator, *, symmetric: bool
) -> Array | None:
    """Exact eigenvalues of a structured A, or ``None`` if unstructured.

    Args:
        operator: The square operator A.
        symmetric: Whether A is known to be real symmetric.

    Returns:
        The eigenvalues γ, or ``None`` when A has no structured rule.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        # γ = diag(A)
        return lx.diagonal(operator)
    if isinstance(operator, BlockDiag) and _square_factors(operator):
        # γ = (γ(A_1), …, γ(A_D)), block by block
        return jnp.concatenate(
            [
                _eigvals(block, None, None, symmetric=symmetric)
                for block in operator.operators
            ]
        )
    if isinstance(operator, Kronecker) and _square_factors(operator):
        return _eigvals_kronecker(operator, symmetric=symmetric)
    if isinstance(operator, KroneckerSum):
        return _eigvals_kronecker_sum(operator, symmetric=symmetric)
    return None


def _eig_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> tuple[Array, Array]:
    """Eigendecomposition of a diagonal operator diag(a).

        γ = a,   Q = I

    Args:
        operator: Diagonal operator diag(a).

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding a and I.
    """
    diagonal = lx.diagonal(operator)
    size = diagonal.shape[0]
    return diagonal, jnp.eye(size, dtype=diagonal.dtype)


def _square_factors(operator: BlockDiag | Kronecker) -> bool:
    """Whether every block or factor of a block-diagonal or Kronecker A is square.

    Args:
        operator: Block-diagonal or Kronecker operator.

    Returns:
        ``True`` when every block or factor has as many rows as columns.
    """
    return all(block.in_size() == block.out_size() for block in operator.operators)


def _factor_eig(
    operator: lx.AbstractLinearOperator, symmetric: bool
) -> tuple[Array, Array]:
    """Full eigendecomposition of one block or factor A_d.

    Args:
        operator: The square block or factor A_d.
        symmetric: Whether A_d is known to be real symmetric.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ_d and Q_d.
    """
    return _eig(operator, None, "lanczos", None, symmetric=symmetric)


def _eig_block_diag(
    operator: BlockDiag,
    *,
    symmetric: bool = False,
) -> tuple[Array, Array]:
    """Eigendecomposition of a block-diagonal operator, block by block.

        γ = (γ_1, …, γ_D),   Q = blockdiag(Q_1, …, Q_D)

    Args:
        operator: Block-diagonal operator with square blocks A_d.
        symmetric: Whether A is known to be real symmetric.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ and Q.
    """
    block_eigenvalues = []
    block_eigenvectors = []
    for block in operator.operators:
        # A_d = Q_d diag(γ_d) Q_d⁻¹
        eigenvalues_d, eigenvectors_d = _factor_eig(block, symmetric)
        block_eigenvalues.append(eigenvalues_d)
        block_eigenvectors.append(eigenvectors_d)
    # γ = (γ_1, …, γ_D)
    eigenvalues = jnp.concatenate(block_eigenvalues)
    # Q = blockdiag(Q_1, …, Q_D)
    eigenvectors = _block_diag(*block_eigenvectors)
    return eigenvalues, eigenvectors


def _eig_kronecker(
    operator: Kronecker,
    *,
    symmetric: bool = False,
) -> tuple[Array, Array]:
    """Eigendecomposition of a Kronecker product from its factors.

        γ = γ_1 ⊗ … ⊗ γ_D,   Q = Q_1 ⊗ … ⊗ Q_D

    Args:
        operator: Kronecker product A_1 ⊗ … ⊗ A_D with square factors.
        symmetric: Whether A is known to be real symmetric.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ and Q.
    """
    # A_d = Q_d diag(γ_d) Q_d⁻¹, per factor
    factor_eigs = [_factor_eig(factor, symmetric) for factor in operator.operators]
    # γ = γ_1 ⊗ … ⊗ γ_D
    eigenvalues = ft.reduce(jnp.kron, (values for values, _ in factor_eigs))
    # Q = Q_1 ⊗ … ⊗ Q_D
    eigenvectors = ft.reduce(jnp.kron, (vectors for _, vectors in factor_eigs))
    return eigenvalues, eigenvectors


def _eigvals_kronecker(operator: Kronecker, *, symmetric: bool = False) -> Array:
    """Eigenvalues of a Kronecker product from its factors.

        γ = γ_1 ⊗ … ⊗ γ_D

    Args:
        operator: Kronecker product A_1 ⊗ … ⊗ A_D with square factors.
        symmetric: Whether A is known to be real symmetric.

    Returns:
        The eigenvalues γ.
    """
    # γ = γ_1 ⊗ … ⊗ γ_D
    return ft.reduce(
        jnp.kron,
        (
            _eigvals(factor, None, None, symmetric=symmetric)
            for factor in operator.operators
        ),
    )


def _eig_kronecker_sum(
    operator: KroneckerSum,
    *,
    symmetric: bool = False,
) -> tuple[Array, Array]:
    """Eigendecomposition of a Kronecker sum from its factors.

        A_1 ⊕ A_2 = (Q_1 ⊗ Q_2) diag(γ_1,i + γ_2,j) (Q_1 ⊗ Q_2)ᵀ

    Args:
        operator: Kronecker sum A_1 ⊕ A_2.
        symmetric: Whether A is known to be real symmetric.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding the pairwise sums
        γ_1,i + γ_2,j and Q_1 ⊗ Q_2.
    """
    # A_1 = Q_1 diag(γ_1) Q_1ᵀ
    eigenvalues_1, eigenvectors_1 = _factor_eig(operator.A, symmetric)
    # A_2 = Q_2 diag(γ_2) Q_2ᵀ
    eigenvalues_2, eigenvectors_2 = _factor_eig(operator.B, symmetric)
    # γ_ij = γ_1,i + γ_2,j, flattened row-major
    eigenvalues = einx.add("a, b -> (a b)", eigenvalues_1, eigenvalues_2)
    # Q = Q_1 ⊗ Q_2
    eigenvectors = jnp.kron(eigenvectors_1, eigenvectors_2)
    return eigenvalues, eigenvectors


def _eigvals_kronecker_sum(operator: KroneckerSum, *, symmetric: bool = False) -> Array:
    """Eigenvalues of a Kronecker sum, without materialising its factors.

        γ_ij = γ_1,i + γ_2,j

    Args:
        operator: Kronecker sum A_1 ⊕ A_2.
        symmetric: Whether A is known to be real symmetric.

    Returns:
        The pairwise sums γ_1,i + γ_2,j, flattened row-major.
    """
    eigenvalues_1 = _eigvals(operator.A, None, None, symmetric=symmetric)
    eigenvalues_2 = _eigvals(operator.B, None, None, symmetric=symmetric)
    # γ_ij = γ_1,i + γ_2,j, flattened row-major
    return einx.add("a, b -> (a b)", eigenvalues_1, eigenvalues_2)


def _eig_partial(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
) -> tuple[Array, Array]:
    """Partial eigendecomposition of a symmetric A by matfree Lanczos.

        A Q ≈ Q diag(γ),   Q of size N × K

    Args:
        operator: The square symmetric operator A.
        rank: Number of eigenpairs K, clamped to N.
        key: PRNG key for the Lanczos start vector; ``None`` uses
            ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ, of length K, and
        Q, N × K.
    """
    if key is None:
        key = jr.PRNGKey(0)

    size = operator.in_size()
    rank = min(rank, size)
    start_vector = jr.normal(key, (size,), dtype=operator.in_structure().dtype)

    tridiag = matfree.decomp.tridiag_sym(rank, reortho="full")
    eigh_fn = matfree.eig.eigh_partial(tridiag)

    # matfree returns the eigenvectors as rows, Qᵀ of size K × N
    eigenvalues, eigenvectors_rows = eigh_fn(operator.mv, start_vector)
    # jnp, not einx: einx cannot resolve an empty basis (``rank=0``, K = 0).
    # Q = (Qᵀ)ᵀ
    return eigenvalues, eigenvectors_rows.T


def _is_real_symmetric(operator: lx.AbstractLinearOperator) -> bool:
    """Symmetric, or PSD/NSD-tagged with a real dtype (lineax's own rule).

    lineax reports ``is_symmetric(Tagged(X, positive_semidefinite_tag))`` as
    ``False``, yet a real PSD/NSD operator is symmetric (gh-314).

    Args:
        operator: The square operator A.

    Returns:
        ``True`` when A is symmetric, or real and PSD- or NSD-tagged.
    """
    if lx.is_symmetric(operator):
        return True
    definite = lx.is_positive_semidefinite(operator) or lx.is_negative_semidefinite(
        operator
    )
    structures = (operator.in_structure(), operator.out_structure())
    return definite and not any(
        jnp.issubdtype(leaf.dtype, jnp.complexfloating)
        for leaf in jax.tree_util.tree_leaves(structures)
    )


def _eig_dense(
    operator: lx.AbstractLinearOperator,
    *,
    symmetric: bool = False,
) -> tuple[Array, Array]:
    """Dense eigendecomposition fallback: ``eigh`` if symmetric, else ``eig``.

        A Q = Q diag(γ)

    Args:
        operator: The square operator A, densified here.
        symmetric: Whether A is known to be real symmetric.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ and Q.
    """
    matrix = operator.as_matrix()
    if symmetric or _is_real_symmetric(operator):
        return jnp.linalg.eigh(matrix)
    return jnp.linalg.eig(matrix)


def _eigvals_dense(
    operator: lx.AbstractLinearOperator,
    *,
    symmetric: bool = False,
) -> Array:
    """Dense eigenvalue fallback: ``eigvalsh`` if symmetric, else ``eigvals``.

    Args:
        operator: The square operator A, densified here.
        symmetric: Whether A is known to be real symmetric.

    Returns:
        The eigenvalues γ.
    """
    matrix = operator.as_matrix()
    if symmetric or _is_real_symmetric(operator):
        return jnp.linalg.eigvalsh(matrix)
    return jnp.linalg.eigvals(matrix)


# ---------------------------------------------------------------------------
# Generalised symmetric-definite eigenproblem A v = γ B v
# ---------------------------------------------------------------------------


def eigh_generalized(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    which: Literal["smallest", "largest"] = "smallest",
    rcond: float | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[Float[Array, " K"], Float[Array, "N K"]]:
    """Generalised eigenproblem A v = γ B v, A symmetric and B symmetric PSD.

        A Q = B Q diag(γ),   Qᵀ B Q = I

    The eigenvectors are B-orthonormal, so the smallest K of them minimise
    tr(Yᵀ A Y) subject to Yᵀ B Y = I (graph embeddings, LPP, manifold
    alignment). Dispatch is on the structure of B:

    - `lineax.DiagonalLinearOperator` B = diag(b) with positive entries:
      scale, S = B^−½ A B^−½ and v = B^−½ u for each eigenvector u of S.
      This is matrix-free in A: with ``rank=`` it runs Lanczos on
      x ↦ B^−½ A B^−½ x and never materialises A. A diagonal with zero
      entries (only checked outside `jit`) is sent to the singular path
      below.
    - Positive definite, i.e. tagged `lineax.positive_semidefinite_tag`
      (lineax's tag for Cholesky-factorisable operators): Cholesky
      whitening L⁻¹ A L⁻ᵀ with B = L Lᵀ.
    - Otherwise B is treated as PSD and possibly singular. B is
      eigendecomposed, B = Q_B diag(γ_B) Q_Bᵀ, and the directions with
      γ_B,i ≤ ``rcond`` · max_i γ_B,i span ker B; Q_B,+ holds the others
      and Q_B,0 the kernel. Writing A_++, A_+0, A_0+ and A_00 for the
      blocks of Q_Bᵀ A Q_B and v = Q_B,+ v_+ + Q_B,0 v_0, the ker B rows
      give v_0 = −A_00⁻¹ A_0+ v_+, and the finite eigenpairs solve the
      Schur-complement pencil (A_++ − A_+0 A_00⁻¹ A_0+) v_+ = γ diag(γ_B,+) v_+.
      A_00 must be positive definite, otherwise the trace minimisation is
      unbounded along ker B and a `ValueError` is raised. At most rank(B)
      finite eigenpairs exist; asking for more warns and returns fewer. The
      numerical rank decides output shapes, so this path cannot run under
      `jax.jit` (tag B positive definite for a traceable path).

    Krylov subspaces are shift-invariant, so the Lanczos path needs no
    spectral shift for ``which="smallest"``: it runs an oversampled Krylov
    space of dimension ``min(N, max(2 * rank + 1, rank + 20))`` and keeps
    the ``rank`` extreme Ritz pairs at the requested end.

    Algorithm:
        1. Validate ``which``, ``rank`` and the shapes of A and B.
        2. A positive diagonal B scales A to B^−½ A B^−½ and solves the
           standard problem, densely or by Lanczos.
        3. A PSD-tagged B whitens A by its Cholesky factor L.
        4. Any other B is eigendecomposed and its kernel eliminated by a
           Schur complement.
        5. Keep the K eigenpairs at the requested end and map them back
           to B-orthonormal eigenvectors.

    Args:
        A: Symmetric N × N operator A.
        B: Symmetric positive semidefinite N × N operator B.
        rank: Number of eigenpairs K. ``None`` returns all finite ones.
        which: ``"smallest"`` or ``"largest"`` eigenvalues.
        rcond: Relative threshold on B's eigenvalues γ_B below which a
            direction counts as ker B (singular path only). ``None`` means
            N ε for the machine epsilon ε of B's dtype.
        key: PRNG key for the Lanczos start vector (diagonal path with
            ``rank=`` only). ``None`` means ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ, of length K and in
        ascending order, and the B-orthonormal Q, N × K.

    Raises:
        ValueError: If ``which`` is invalid, ``rank < 1``, A and B are not
            square and of one size, B is zero, or A_00 is not positive
            definite.

    Examples:
        Laplacian eigenmaps on a path graph, L y = γ D y, whose eigenvalues
        are 1 − cos(πk/4); ``rank=`` runs Lanczos:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> W = jnp.diag(jnp.ones(4), 1) + jnp.diag(jnp.ones(4), -1)
        >>> degree = W @ jnp.ones(5)
        >>> L = lx.MatrixLinearOperator(jnp.diag(degree) - W, lx.symmetric_tag)
        >>> D = lx.DiagonalLinearOperator(degree)
        >>> eigenvalues, Q = gaussx.eigh_generalized(L, D, rank=2)
        >>> expected = 1 - jnp.cos(jnp.pi * jnp.arange(2) / 4)
        >>> bool(jnp.allclose(eigenvalues, expected, atol=1e-5))
        True

        A singular B has one finite eigenpair fewer than N:

        >>> B = lx.MatrixLinearOperator(jnp.diag(jnp.array([1.0, 2.0, 0.0])))
        >>> A = lx.MatrixLinearOperator(jnp.eye(3) + 0.5 * jnp.ones((3, 3)))
        >>> eigenvalues, Q = gaussx.eigh_generalized(A, B)
        >>> eigenvalues.shape
        (2,)

        An invalid ``which`` is rejected:

        >>> gaussx.eigh_generalized(A, B, which="middle")
        Traceback (most recent call last):
        ...
        ValueError: which must be 'smallest' or 'largest', got 'middle'.
    """
    if which not in ("smallest", "largest"):
        msg = f"which must be 'smallest' or 'largest', got {which!r}."
        raise ValueError(msg)
    if rank is not None and rank < 1:
        msg = f"rank must be a positive integer, got {rank}."
        raise ValueError(msg)
    if A.in_size() != B.in_size() or A.in_size() != A.out_size():
        msg = (
            f"A and B must be square and of the same size, got A "
            f"{A.out_size()}x{A.in_size()} and B {B.out_size()}x{B.in_size()}."
        )
        raise ValueError(msg)

    if isinstance(B, lx.DiagonalLinearOperator):
        # b, with B = diag(b)
        diagonal = lx.diagonal(B)
        # min_i b_i > τ, checked only outside jit
        if isinstance(diagonal, Tracer) or bool(
            jnp.min(diagonal) > _null_tol(diagonal, rcond, diagonal.shape[0])
        ):
            return _eigh_generalized_diagonal(A, diagonal, rank, which, key)
    elif lx.is_positive_semidefinite(B):
        return _eigh_generalized_cholesky(A, B, rank, which)
    return _eigh_generalized_singular(A, B, rank, which, rcond)


def _null_tol(eigenvalues: Array, rcond: float | None, size: int) -> Array:
    """Absolute threshold τ below which an eigenvalue of B counts as zero.

        τ = rcond · max_i ∣γ_B,i∣,   rcond = N ε by default

    Args:
        eigenvalues: Eigenvalues γ_B of B (or the diagonal b).
        rcond: Relative threshold, or ``None`` for N ε.
        size: The size N that scales the default threshold.

    Returns:
        The scalar threshold τ.
    """
    if rcond is None:
        # rcond = N ε
        rcond = size * float(jnp.finfo(eigenvalues.dtype).eps)
    # max_i ∣γ_B,i∣
    largest = jnp.max(jnp.abs(eigenvalues))
    # τ = rcond · max_i ∣γ_B,i∣
    return rcond * largest


def _select(
    eigenvalues: Array, eigenvectors: Array, count: int, which: str
) -> tuple[Array, Array]:
    """Keep the K eigenpairs at the requested end of ascending eigenvalues.

    Args:
        eigenvalues: Eigenvalues γ in ascending order.
        eigenvectors: Eigenvectors, one per column.
        count: Number of eigenpairs K to keep.
        which: ``"smallest"`` keeps the first K, ``"largest"`` the last K.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` with the K kept pairs.
    """
    if which == "smallest":
        return eigenvalues[:count], eigenvectors[:, :count]
    size = eigenvalues.shape[0]
    return eigenvalues[size - count :], eigenvectors[:, size - count :]


def _eigh_generalized_diagonal(
    A: lx.AbstractLinearOperator,
    diagonal: Array,
    rank: int | None,
    which: str,
    key: PRNGKeyArray | None,
) -> tuple[Array, Array]:
    """Positive diagonal B = diag(b): eigenpairs of B^−½ A B^−½, then rescale.

        B^−½ A B^−½ u = γ u,   v = B^−½ u

    Args:
        A: Symmetric operator A.
        diagonal: Positive diagonal b of B.
        rank: Number of eigenpairs K, or ``None`` for all of them (dense).
        which: ``"smallest"`` or ``"largest"``.
        key: PRNG key for the Lanczos start vector, or ``None``.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ and the
        B-orthonormal Q.
    """
    size = diagonal.shape[0]
    # √b_i
    sqrt_diagonal = jnp.sqrt(diagonal)
    # B^−½ = diag(1 / √b_i)
    inverse_sqrt_diagonal = 1.0 / sqrt_diagonal
    if rank is None:
        dense_A = A.as_matrix()
        # A B^−½
        right_scaled = einx.multiply(
            "row col, col -> row col", dense_A, inverse_sqrt_diagonal
        )
        # S = B^−½ A B^−½
        whitened = einx.multiply(
            "row, row col -> row col", inverse_sqrt_diagonal, right_scaled
        )
        # S U = U diag(γ)
        eigenvalues, eigenvectors = jnp.linalg.eigh(whitened)
        count = size
    else:
        count = min(rank, size)
        if key is None:
            key = jr.PRNGKey(0)
        # Oversampled Krylov depth, so the Ritz pairs at either end converge
        depth = min(size, max(2 * count + 1, count + 20))
        start_vector = jr.normal(key, (size,), dtype=inverse_sqrt_diagonal.dtype)
        tridiag = matfree.decomp.tridiag_sym(depth, reortho="full")
        eigh_fn = matfree.eig.eigh_partial(tridiag)

        def whitened_mv(vector: Array) -> Array:
            # B^−½ x
            scaled = inverse_sqrt_diagonal * vector
            # A B^−½ x
            applied = A.mv(scaled)
            # S x = B^−½ A B^−½ x
            return inverse_sqrt_diagonal * applied

        eigenvalues, eigenvectors = eigh_fn(whitened_mv, start_vector)
        # matfree returns the eigenvectors as rows; U = (Uᵀ)ᵀ
        eigenvectors = rearrange(eigenvectors, "k n -> n k")
    eigenvalues, eigenvectors = _select(eigenvalues, eigenvectors, count, which)
    # Q = B^−½ U
    return eigenvalues, einx.multiply(
        "i, i k -> i k", inverse_sqrt_diagonal, eigenvectors
    )


def _eigh_generalized_cholesky(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
    rank: int | None,
    which: str,
) -> tuple[Array, Array]:
    """Positive definite B = L Lᵀ: eigenpairs of L⁻¹ A L⁻ᵀ, then v = L⁻ᵀ u.

        L⁻¹ A L⁻ᵀ u = γ u,   v = L⁻ᵀ u

    Args:
        A: Symmetric operator A.
        B: Positive definite operator B, densified here.
        rank: Number of eigenpairs K, or ``None`` for all of them.
        which: ``"smallest"`` or ``"largest"``.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ and the
        B-orthonormal Q.
    """
    size = B.in_size()
    # B = L Lᵀ
    cholesky_factor = jnp.linalg.cholesky(B.as_matrix())
    # L⁻¹ A
    left_whitened = solve_triangular(cholesky_factor, A.as_matrix(), lower=True)
    # (L⁻¹ A)ᵀ = A L⁻ᵀ, since A is symmetric
    left_whitened_t = rearrange(left_whitened, "i j -> j i")
    # S = L⁻¹ A L⁻ᵀ
    whitened = solve_triangular(cholesky_factor, left_whitened_t, lower=True)
    # S U = U diag(γ)
    eigenvalues, eigenvectors = jnp.linalg.eigh(whitened)
    count = size if rank is None else min(rank, size)
    eigenvalues, eigenvectors = _select(eigenvalues, eigenvectors, count, which)
    # Q = L⁻ᵀ U
    return eigenvalues, solve_triangular(
        cholesky_factor, eigenvectors, lower=True, trans="T"
    )


def _eigh_generalized_singular(
    A: lx.AbstractLinearOperator,
    B: lx.AbstractLinearOperator,
    rank: int | None,
    which: str,
    rcond: float | None,
) -> tuple[Array, Array]:
    """PSD, possibly singular B: Schur-complement elimination of ker B.

        (A_++ − A_+0 A_00⁻¹ A_0+) v_+ = γ diag(γ_B,+) v_+
        v = Q_B,+ v_+ + Q_B,0 v_0,   v_0 = −A_00⁻¹ A_0+ v_+

    Args:
        A: Symmetric operator A, densified here.
        B: Symmetric PSD operator B, densified here.
        rank: Number of eigenpairs K, or ``None`` for all finite ones.
        which: ``"smallest"`` or ``"largest"``.
        rcond: Relative threshold for ker B, or ``None`` for N ε.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ and the
        B-orthonormal Q.

    Raises:
        ValueError: If B is numerically zero, or A_00 is not positive
            definite.
    """
    size = B.in_size()
    dense_A = A.as_matrix()
    # B = Q_B diag(γ_B) Q_Bᵀ
    metric_eigenvalues, metric_eigenvectors = jnp.linalg.eigh(B.as_matrix())
    # γ_B,i > τ marks the range of B
    is_range = np.asarray(
        metric_eigenvalues > _null_tol(metric_eigenvalues, rcond, size)
    )
    # rank(B), the number of finite eigenpairs
    range_size = int(is_range.sum())
    if range_size == 0:
        msg = "B is numerically zero: the pencil has no finite eigenvalues."
        raise ValueError(msg)
    count = range_size if rank is None else min(rank, range_size)
    if rank is not None and rank > range_size:
        warnings.warn(
            f"eigh_generalized: B has numerical rank {range_size} < rank={rank}, "
            f"so only {range_size} finite eigenpairs exist; returning "
            f"{range_size}.",
            stacklevel=2,
        )

    # Q_B,+ and γ_B,+
    range_basis = metric_eigenvectors[:, is_range]
    range_eigenvalues = metric_eigenvalues[is_range]
    # A_++ = Q_B,+ᵀ A Q_B,+
    schur = einsum(range_basis, dense_A, range_basis, "i a, i j, j b -> a b")
    if range_size < size:
        # Q_B,0
        kernel_basis = metric_eigenvectors[:, ~is_range]
        # A_00 = Q_B,0ᵀ A Q_B,0
        kernel_block = einsum(
            kernel_basis, dense_A, kernel_basis, "i a, i j, j b -> a b"
        )
        # A_0+ = Q_B,0ᵀ A Q_B,+
        coupling = einsum(kernel_basis, dense_A, range_basis, "i a, i j, j b -> a b")
        # γ(A_00), ascending
        kernel_eigenvalues = jnp.linalg.eigvalsh(kernel_block)
        if not bool(
            kernel_eigenvalues[0]
            > _null_tol(kernel_eigenvalues, None, size - range_size)
        ):
            msg = (
                "eigh_generalized: A restricted to ker B (A₀₀) is not positive "
                f"definite (smallest eigenvalue {float(kernel_eigenvalues[0]):.3e}), "
                "so tr(Yᵀ A Y) s.t. Yᵀ B Y = I is unbounded below along ker B."
            )
            raise ValueError(msg)
        # A_00 = L_00 L_00ᵀ
        kernel_cholesky = jnp.linalg.cholesky(kernel_block)
        # Z = A_00⁻¹ A_0+
        eliminated = cho_solve((kernel_cholesky, True), coupling)
        # A_+0 A_00⁻¹ A_0+
        correction = einsum(coupling, eliminated, "a p, a q -> p q")
        # A_++ − A_+0 A_00⁻¹ A_0+
        schur = schur - correction

    # √γ_B,+
    sqrt_range_eigenvalues = jnp.sqrt(range_eigenvalues)
    # diag(γ_B,+)^−½
    inverse_sqrt = 1.0 / sqrt_range_eigenvalues
    # S diag(γ_B,+)^−½
    right_scaled = einx.multiply("p q, q -> p q", schur, inverse_sqrt)
    # M = diag(γ_B,+)^−½ S diag(γ_B,+)^−½
    whitened = einx.multiply("p, p q -> p q", inverse_sqrt, right_scaled)
    # M U = U diag(γ)
    eigenvalues, whitened_eigenvectors = jnp.linalg.eigh(whitened)
    eigenvalues, whitened_eigenvectors = _select(
        eigenvalues, whitened_eigenvectors, count, which
    )
    # v_+ = diag(γ_B,+)^−½ u
    range_coefficients = einx.multiply(
        "p, p k -> p k", inverse_sqrt, whitened_eigenvectors
    )
    # Q_B,+ v_+
    eigenvectors = einsum(range_basis, range_coefficients, "i p, p k -> i k")
    if range_size < size:
        # A_00⁻¹ A_0+ v_+
        kernel_coefficients = einsum(eliminated, range_coefficients, "z p, p k -> z k")
        # Q_B,0 A_00⁻¹ A_0+ v_+
        kernel_part = einsum(kernel_basis, kernel_coefficients, "i z, z k -> i k")
        # v = Q_B,+ v_+ + Q_B,0 v_0,   v_0 = −A_00⁻¹ A_0+ v_+
        eigenvectors = eigenvectors - kernel_part
    return eigenvalues, eigenvectors
