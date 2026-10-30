"""Structured singular value decomposition, dispatched on the operator type of A.

A is an M × N operator with the singular value decomposition
A = U diag(s) Vᵀ: K ≤ min(M, N) singular values s ≥ 0, left singular vectors
in the columns of the M × K matrix U and right singular vectors in the
columns of the N × K matrix V. Here U and V are singular vectors, not the
factors of a low-rank update. A Kronecker product is A = A_1 ⊗ … ⊗ A_D with
factors A_d, d = 1, …, D, and a block-diagonal operator is
blockdiag(A_1, …, A_D).
"""

from __future__ import annotations

import functools as ft
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg
import lineax as lx
import matfree.decomp
import matfree.eig
from jaxtyping import Array, Float

from gaussx._operators._block_diag import BlockDiag
from gaussx._operators._kronecker import Kronecker
from gaussx._randomized._svd import randomized_svd


def svd(
    operator: lx.AbstractLinearOperator,
    *,
    rank: int | None = None,
    method: Literal["lanczos", "randomized"] = "lanczos",
    key: jax.Array | None = None,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """Singular value decomposition of A, with structural dispatch.

        A = U diag(s) Vᵀ

    ``rank`` follows one rule shared with `gaussx.eig` and `gaussx.eigvals`:
    an operator with a structured SVD (diagonal, `gaussx.Kronecker`,
    `gaussx.BlockDiag`) is decomposed exactly and truncated to its K =
    ``rank`` largest singular values. Any other operator gets a partial
    (truncated) SVD without materialising A, by ``method``:

    - ``"lanczos"`` (default): matfree's Golub-Kahan bidiagonalisation.
    - ``"randomized"``: `randomized_svd` with its defaults
      (``oversample=10``, ``n_power_iter=2``); call it directly to tune
      them. Use ``n_power_iter >= 2`` for slowly decaying spectra.

    Both target the **top** of the spectrum (the largest singular values).

    Algorithm:
        1. Check ``method`` and peel tags.
        2. Decompose a diagonal, Kronecker or block-diagonal A exactly from
           its entries, factors or blocks, sort the singular values in
           descending order and keep the top K when ``rank`` is given.
        3. Otherwise run a partial SVD of rank K by ``method`` when
           ``rank`` is given, else materialise A and run a dense SVD.

    Args:
        operator: The operator A.
        rank: Number K of singular values to compute. ``None`` computes
            all of them, which materialises an unstructured A.
        method: Partial-SVD algorithm, ``"lanczos"`` or ``"randomized"``.
            Only used when ``rank`` is given.
        key: PRNG key for the initial random vector (Lanczos) or the
            Gaussian test matrix (randomized) of a partial SVD. ``None``
            means ``PRNGKey(0)``.

    Returns:
        Tuple ``(U, s, Vt)`` holding the left singular vectors U (M × K),
        the singular values s (K) and the transposed right singular
        vectors Vᵀ (K × N).

    Raises:
        ValueError: If ``method`` is invalid, or ``"randomized"`` without
            ``rank``.

    Examples:
        A diagonal operator: singular values are |a_i|, sorted only when
        truncated:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([3.0, -1.0, 2.0]))
        >>> U, s, Vt = gaussx.svd(A)
        >>> [float(v) for v in s]
        [3.0, 1.0, 2.0]
        >>> U, s, Vt = gaussx.svd(A, rank=2)
        >>> U.shape, [float(v) for v in s], Vt.shape
        ((3, 2), [3.0, 2.0], (2, 3))

        A Kronecker product multiplies its factors' singular values:

        >>> A_1 = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
        >>> A_2 = lx.DiagonalLinearOperator(jnp.array([4.0, 5.0]))
        >>> _, s, _ = gaussx.svd(gaussx.Kronecker(A_1, A_2))
        >>> [float(v) for v in s]
        [10.0, 8.0, 5.0, 4.0]

        The factors reconstruct a dense matrix:

        >>> matrix = jnp.array([[2.0, 1.0], [0.0, 3.0], [1.0, 1.0]])
        >>> U, s, Vt = gaussx.svd(lx.MatrixLinearOperator(matrix))
        >>> bool(jnp.allclose((U * s) @ Vt, matrix, atol=1e-5))
        True
    """
    if method not in ("lanczos", "randomized"):
        raise ValueError(f"method must be 'lanczos' or 'randomized', got {method!r}.")
    if method == "randomized" and rank is None:
        raise ValueError("method='randomized' needs a rank.")
    if isinstance(operator, lx.TaggedLinearOperator):
        return svd(operator.operator, rank=rank, method=method, key=key)
    structured = _svd_structured(operator)
    if structured is not None:
        # Exact structured SVD first, then the top ``rank`` (gh-383).
        if rank is None:
            return structured
        left_vectors, singular_values, right_vectors_t = _sort_svd_descending(
            *structured
        )
        # U_K, s_K and V_Kᵀ, the top K = rank triplets
        return (
            left_vectors[:, :rank],
            singular_values[:rank],
            right_vectors_t[:rank, :],
        )
    if rank is not None:
        if method == "randomized":
            return randomized_svd(operator, rank, key=key)
        return _svd_partial(operator, rank, key)
    return _svd_dense(operator)


def _svd_structured(
    operator: lx.AbstractLinearOperator,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]] | None:
    """Exact SVD of a diagonal, Kronecker or block-diagonal operator.

    Args:
        operator: The operator A.

    Returns:
        Tuple ``(U, s, Vt)``, or ``None`` when A has no structured SVD.
    """
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _svd_diagonal(operator)
    if isinstance(operator, Kronecker):
        return _svd_kronecker(operator)
    if isinstance(operator, BlockDiag):
        return _svd_block_diag(operator)
    return None


def _svd_diagonal(
    operator: lx.DiagonalLinearOperator,
) -> tuple[Float[Array, "n n"], Float[Array, " n"], Float[Array, "n n"]]:
    """SVD of a diagonal operator diag(a), in its original order.

        diag(a) = diag(sign(a)) diag(|a|) I

    Args:
        operator: Diagonal operator diag(a).

    Returns:
        Tuple ``(U, s, Vt)`` with U = diag(sign(a)), s = |a| and Vᵀ = I.
    """
    diagonal = lx.diagonal(operator)
    size = diagonal.shape[0]
    # s = |a|
    singular_values = jnp.abs(diagonal)
    # sign(a), with +1 for a = 0, in a's dtype (Python-float branches would
    # be default-float).
    signs = jnp.where(diagonal >= 0, 1, -1).astype(diagonal.dtype)
    # U = diag(sign(a))
    left_vectors = jnp.diag(signs)
    # Vᵀ = I
    right_vectors_t = jnp.eye(size, dtype=diagonal.dtype)
    return left_vectors, singular_values, right_vectors_t


def _svd_partial(
    operator: lx.AbstractLinearOperator,
    rank: int,
    key: jax.Array | None,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """Partial SVD of rank K by matfree's Golub-Kahan bidiagonalisation.

        A ≈ U_K diag(s_K) V_Kᵀ

    Args:
        operator: The operator A, used through matvecs only.
        rank: Number K of singular values, clamped to min(M, N).
        key: PRNG key for the start vector. ``None`` means ``PRNGKey(0)``.

    Returns:
        Tuple ``(U, s, Vt)`` of the top K singular triplets.
    """
    if key is None:
        key = jr.PRNGKey(0)

    rank = min(rank, operator.in_size(), operator.out_size())
    size = operator.in_size()
    # Default float on purpose: matfree's bidiag builds its state in JAX's
    # default float, so a float32 start vector crashes its VJP under x64.
    start_vector = jr.normal(key, (size,))

    bidiag = matfree.decomp.bidiag(rank)
    svd_fn = matfree.eig.svd_partial(bidiag)

    # matfree returns Uᵀ (K × M), s (K) and Vᵀ (K × N).
    left_vectors_t, singular_values, right_vectors_t = svd_fn(operator.mv, start_vector)
    # jnp, not einx: einx cannot resolve an empty basis (K = 0 or M = 0).
    # U = (Uᵀ)ᵀ
    left_vectors = left_vectors_t.T
    return left_vectors, singular_values, right_vectors_t


def _sort_svd_descending(
    U: Float[Array, "m k"],
    s: Float[Array, " k"],
    Vt: Float[Array, "k n"],
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """Reorder an assembled SVD so the singular values descend.

    Args:
        U: Left singular vectors U, one per column.
        s: Singular values s.
        Vt: Right singular vectors Vᵀ, one per row.

    Returns:
        Tuple ``(U, s, Vt)`` with s in descending order and the vectors
        permuted to match.
    """
    # descending permutation of s
    order = jnp.argsort(-s)
    return U[:, order], s[order], Vt[order, :]


def _svd_kronecker(
    operator: Kronecker,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """SVD of a Kronecker product from its factors' SVDs.

        A_1 ⊗ … ⊗ A_D = (U_1 ⊗ … ⊗ U_D) diag(s_1 ⊗ … ⊗ s_D) (V_1 ⊗ … ⊗ V_D)ᵀ

    Only the (small) factors are decomposed; the full operator is never
    materialised as an input to a decomposition. The returned dense U and
    Vᵀ are inherent to the SVD output format.

    Args:
        operator: Kronecker product of factors A_d.

    Returns:
        Tuple ``(U, s, Vt)``, sorted by descending singular value.
    """
    # A_d = U_d diag(s_d) V_dᵀ, per factor
    factor_svds = [svd(factor) for factor in operator.operators]
    # U_1 ⊗ … ⊗ U_D
    left_vectors = ft.reduce(jnp.kron, (u for u, _, _ in factor_svds))
    # s_1 ⊗ … ⊗ s_D
    singular_values = ft.reduce(jnp.kron, (sv for _, sv, _ in factor_svds))
    # V_1ᵀ ⊗ … ⊗ V_Dᵀ
    right_vectors_t = ft.reduce(jnp.kron, (vt for _, _, vt in factor_svds))
    return _sort_svd_descending(left_vectors, singular_values, right_vectors_t)


def _svd_block_diag(
    operator: BlockDiag,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """SVD of a block-diagonal operator from its blocks' SVDs.

        blockdiag(A_1, …, A_D) = blockdiag(U_d) diag([s_1; …; s_D]) blockdiag(V_d)ᵀ

    Args:
        operator: Block-diagonal operator with blocks A_d.

    Returns:
        Tuple ``(U, s, Vt)``, sorted by descending singular value.
    """
    # A_d = U_d diag(s_d) V_dᵀ, per block
    block_svds = [svd(block) for block in operator.operators]
    # blockdiag(U_1, …, U_D)
    left_vectors = jax.scipy.linalg.block_diag(*(u for u, _, _ in block_svds))
    # [s_1; …; s_D]
    singular_values = jnp.concatenate([sv for _, sv, _ in block_svds])
    # blockdiag(V_1ᵀ, …, V_Dᵀ)
    right_vectors_t = jax.scipy.linalg.block_diag(*(vt for _, _, vt in block_svds))
    return _sort_svd_descending(left_vectors, singular_values, right_vectors_t)


def _svd_dense(
    operator: lx.AbstractLinearOperator,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """Thin dense SVD of a materialised operator.

    Args:
        operator: The operator A, materialised here.

    Returns:
        Tuple ``(U, s, Vt)`` with K = min(M, N).
    """
    matrix = operator.as_matrix()
    # A = U diag(s) Vᵀ, thin
    left_vectors, singular_values, right_vectors_t = jnp.linalg.svd(
        matrix,
        full_matrices=False,  # type: ignore[arg-type]
    )
    return left_vectors, singular_values, right_vectors_t
