"""Randomized SVD and symmetric eigendecomposition of an operator A.

A is M × N for the SVD, A ≈ U diag(s) Vᵀ, with k singular values s ≥ 0 and left and
right singular vectors in the columns of U and V. A is N × N and symmetric for the
eigendecomposition, A ≈ V diag(γ) Vᵀ, with eigenvalues γ and orthonormal V. Q is an
orthonormal basis from `range_finder` with ℓ = k + p columns, for oversampling p and
q power iterations.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._linalg._symmetrize import symmetrize
from gaussx._randomized._range_finder import _matmat, qb, range_finder


def randomized_svd(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    key: jax.Array | None = None,
) -> tuple[Float[Array, "m k"], Float[Array, " k"], Float[Array, "k n"]]:
    """Truncated SVD A ≈ U diag(s) Vᵀ by randomized QB.

        A ≈ Q B,   B = U_B diag(s) Vᵀ,   U = Q U_B

    The cost is (2q + 2) ℓ matvecs with A or Aᵀ, plus O((M + N) ℓ²) flops; A is
    never formed.

    Randomized methods target the **top** of the spectrum: the leading singular
    triplets are accurate, the trailing ones are not. Use ``n_power_iter >= 2`` for
    slowly decaying spectra (Matérn-½ Gram matrices, most geophysical fields); see
    `range_finder` for the error bound.

    Algorithm:
        1. Factorise A ≈ Q B with `qb`.
        2. Take the small SVD B = U_B diag(s) Vᵀ.
        3. Keep the top k triplets and lift the left vectors, U = Q U_B.

    Args:
        op: Operator A of size M × N; may be matrix-free.
        rank: Number of singular triplets k to return (at most min(M, N)).
        oversample: Extra columns p in the range finder.
        n_power_iter: Number of power iterations q.
        key: PRNG key for the Gaussian test matrix. ``None`` means
            ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(U, s, Vt)`` holding U (M × k), s (length k, descending) and
        Vᵀ (k × N).

    Raises:
        ValueError: On a non-positive ``rank``, or a negative ``oversample`` or
            ``n_power_iter``.

    Examples:
        Five EOFs of an anomaly matrix available only as a matvec, here a small dense
        stand-in:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> left = jr.normal(jr.key(0), (300, 8))
        >>> anomalies = left @ jr.normal(jr.key(1), (8, 120))
        >>> op = lx.MatrixLinearOperator(anomalies)
        >>> U, s, Vt = gaussx.randomized_svd(op, 5, key=jr.key(2))
        >>> U.shape, s.shape, Vt.shape
        ((300, 5), (5,), (5, 120))
        >>> pcs = einx.multiply("k, k time -> k time", s, Vt)
        >>> pcs.shape
        (5, 120)

        A ``rank`` above min(M, N) returns min(M, N) triplets:

        >>> small = lx.MatrixLinearOperator(jnp.eye(3))
        >>> gaussx.randomized_svd(small, 5)[1].shape
        (3,)

        A non-positive ``rank`` is rejected:

        >>> gaussx.randomized_svd(small, 0)
        Traceback (most recent call last):
        ...
        ValueError: rank must be a positive integer, got 0.

    References:
        Halko, N., Martinsson, P.-G. & Tropp, J. A. (2011), Algorithm 5.1.
    """
    # A ≈ Q B
    basis, projection = qb(
        op, rank, oversample=oversample, n_power_iter=n_power_iter, key=key
    )
    # B = U_B diag(s) Vᵀ
    small_left, singular_values, right_transpose = jnp.linalg.svd(
        projection, full_matrices=False
    )
    # min(k, ℓ)
    kept = min(rank, singular_values.shape[0])
    # U_B restricted to the top triplets
    top_small_left = small_left[:, :kept]
    # U = Q U_B
    left = einsum(basis, top_small_left, "m l, l k -> m k")
    return left, singular_values[:kept], right_transpose[:kept]


def randomized_eigh(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    which: Literal["largest", "magnitude"] = "largest",
    key: jax.Array | None = None,
) -> tuple[Float[Array, " k"], Float[Array, "n k"]]:
    """Partial eigendecomposition of a symmetric operator by Rayleigh-Ritz.

        T = Qᵀ A Q = W diag(γ) Wᵀ,   A ≈ (Q W) diag(γ) (Q W)ᵀ

    A may be indefinite. The range finder captures the eigenvalues of largest
    **magnitude** (randomized methods target the **top** of the spectrum), and
    ``rank`` Ritz pairs are kept by ``which``:

    - ``"largest"``: the algebraically largest Ritz values;
    - ``"magnitude"``: the Ritz values of largest absolute value.

    For an indefinite A whose large negative eigenvalues dominate, ``"largest"`` is
    only as good as the subspace, so prefer ``"magnitude"`` there. The small end of a
    spectrum (e.g. the smallest eigenvalues of a graph Laplacian) is Lanczos / LOBPCG
    territory.

    Use ``n_power_iter >= 2`` for slowly decaying spectra (Matérn-½ Gram matrices,
    most geophysical fields). With ``n_power_iter=0`` this is the one-pass randomized
    Rayleigh-Ritz projection, the successor of the algorithm behind
    `NystromPreconditioner` up to gaussx 0.4; it projects onto orth(A Ω) rather than
    orth(Ω), which is more accurate, so it is not numerically identical. For PSD
    operators, `randomized_nystrom` is strictly more accurate for the same number of
    matvecs (Tropp et al., 2017).

    Algorithm:
        1. Find an orthonormal Q for the dominant range of A with `range_finder`.
        2. Form the Rayleigh-Ritz matrix T = Qᵀ A Q (ℓ more matvecs) and symmetrise.
        3. Take T = W diag(γ) Wᵀ and keep k Ritz pairs by ``which``.
        4. Lift the kept eigenvectors, V = Q W.

    Args:
        op: Symmetric operator A of size N × N; may be matrix-free.
        rank: Number of eigenpairs k to return.
        oversample: Extra columns p in the range finder.
        n_power_iter: Number of power iterations q.
        which: ``"largest"`` or ``"magnitude"``.
        key: PRNG key for the Gaussian test matrix. ``None`` means
            ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(eigenvalues, eigenvectors)`` holding γ (length k, ascending, as
        `jax.numpy.linalg.eigh`) and the orthonormal V (N × k).

    Raises:
        ValueError: If ``which`` is invalid, the operator is not square, ``rank`` is
            non-positive, or ``oversample`` or ``n_power_iter`` is negative.

    Examples:
        An indefinite operator, where ``which`` decides between its pairs:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> spectrum = jnp.array([-9.0, 5.0, 1.0, 0.1, 0.01, 0.0])
        >>> vectors, _ = jnp.linalg.qr(jr.normal(jr.key(0), (6, 6)))
        >>> scaled = einx.multiply("row k, k -> row k", vectors, spectrum)
        >>> matrix = einx.dot("row k, col k -> row col", scaled, vectors)
        >>> op = lx.MatrixLinearOperator(matrix, lx.symmetric_tag)
        >>> values, _ = gaussx.randomized_eigh(op, 2, oversample=2, which="magnitude")
        >>> bool(jnp.allclose(values, jnp.array([-9.0, 5.0]), atol=1e-4))
        True
        >>> values, _ = gaussx.randomized_eigh(op, 2, oversample=2, which="largest")
        >>> bool(jnp.allclose(values, jnp.array([1.0, 5.0]), atol=1e-4))
        True

        A ``rank`` above N returns N pairs:

        >>> small = lx.MatrixLinearOperator(jnp.eye(3), lx.symmetric_tag)
        >>> gaussx.randomized_eigh(small, 5)[1].shape
        (3, 3)

        An unknown ``which`` is rejected:

        >>> gaussx.randomized_eigh(small, 2, which="smallest")
        Traceback (most recent call last):
        ...
        ValueError: which must be 'largest' or 'magnitude', got 'smallest'.
    """
    if which not in ("largest", "magnitude"):
        raise ValueError(f"which must be 'largest' or 'magnitude', got {which!r}.")
    if op.in_size() != op.out_size():
        raise ValueError(
            f"randomized_eigh needs a square operator, got "
            f"{op.out_size()}x{op.in_size()}."
        )
    # Q = orth((A Aᵀ)^q A Ω)
    basis = range_finder(
        op, rank, oversample=oversample, n_power_iter=n_power_iter, key=key
    )
    # A Q
    image = _matmat(op, basis)
    # Qᵀ A Q
    ritz_matrix = einsum(basis, image, "n a, n b -> a b")
    # T = sym(Qᵀ A Q)
    ritz_matrix = symmetrize(ritz_matrix)
    # T = W diag(γ) Wᵀ
    ritz_values, ritz_vectors = jnp.linalg.eigh(ritz_matrix)
    # min(k, ℓ)
    kept = min(rank, ritz_values.shape[0])
    if which == "largest":
        # eigh sorts γ ascending, so the last k are the largest
        keep = jnp.arange(ritz_values.shape[0] - kept, ritz_values.shape[0])
    else:
        # |γ|
        magnitudes = jnp.abs(ritz_values)
        # −|γ|, so an ascending sort puts the largest first
        order = jnp.argsort(-magnitudes)
        # the k largest |γ|, back in ascending order of γ
        keep = jnp.sort(order[:kept])
    # W restricted to the kept pairs
    kept_vectors = ritz_vectors[:, keep]
    # V = Q W
    eigenvectors = einsum(basis, kept_vectors, "n l, l k -> n k")
    return ritz_values[keep], eigenvectors
