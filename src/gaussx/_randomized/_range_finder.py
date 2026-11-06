"""Randomized range finder and QB factorisation of an M × N operator A.

A test matrix Ω (N × ℓ; the letter of the paper, not the spatial domain) with
ℓ = k + p columns, for rank k and oversampling p, sketches the range of A. Q is an
orthonormal basis of the sketch, B = Qᵀ A its projection and q the number of power
iterations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._sketching._base import AbstractSketch


def _matmat(op: lx.AbstractLinearOperator, columns: Float[Array, "n l"]) -> Array:
    """Apply A to each column of X, without materialising A.

        A X = [A x_1, …, A x_ℓ]

    Args:
        op: The operator A.
        columns: Matrix X whose columns A acts on.

    Returns:
        The product A X.
    """
    if isinstance(op, lx.MatrixLinearOperator):
        # A X, on the stored matrix
        return op.matrix @ columns
    # A x_j, for each column j
    return jax.vmap(op.mv, in_axes=1, out_axes=1)(columns)


def _orth(matrix: Float[Array, "m l"]) -> Float[Array, "m l"]:
    """Orthonormal basis Q of range(Y) from a thin QR.

        Y = Q R

    Args:
        matrix: Matrix Y to orthonormalise.

    Returns:
        The factor Q.
    """
    # Y = Q R
    basis, _ = jnp.linalg.qr(matrix)
    return basis


def _check_args(rank: int, oversample: int, n_power_iter: int) -> None:
    """Reject a non-positive rank k and a negative p or q, in that order.

    Args:
        rank: Target rank k.
        oversample: Extra columns p.
        n_power_iter: Number of power iterations q.

    Raises:
        ValueError: If k < 1, p < 0 or q < 0.
    """
    if rank < 1:
        raise ValueError(f"rank must be a positive integer, got {rank}.")
    if oversample < 0:
        raise ValueError(f"oversample must be non-negative, got {oversample}.")
    if n_power_iter < 0:
        raise ValueError(f"n_power_iter must be non-negative, got {n_power_iter}.")


def range_finder(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    sketch: AbstractSketch | None = None,
    key: jax.Array | None = None,
) -> Float[Array, "m l"]:
    """Orthonormal basis Q for the dominant range of A.

        Q = orth((A Aᵀ)^q A Ω),   Ω ∈ ℝ^(N × ℓ),   ℓ = k + p

    For a Gaussian Ω and q = 0 (Halko, Martinsson & Tropp, 2011, Thm 10.6), with
    singular values s_j of A,

        𝔼 ‖A − Q Qᵀ A‖₂ ≤ (1 + √(k / (p − 1))) s_k+1
                          + (e √(k + p) / p) (∑_j>k s_j²)^(½)

    The tail term dominates when the spectrum decays slowly (Matérn-½ Gram
    matrices, most geophysical fields). q power iterations apply the bound to
    (A Aᵀ)^q A, whose singular values are s_j^(2q+1), at the cost of 2q more passes
    over A, so use ``n_power_iter >= 2`` for slowly decaying spectra.

    Randomized methods target the **top** of the spectrum (the largest singular
    values). For the small end, use Lanczos or LOBPCG. A is touched only through
    ``mv`` (and the transpose's ``mv`` for power steps), vmapped over the ℓ columns.

    Algorithm:
        1. Draw the Gaussian test matrix Ω, or take Ω = Sᵀ from ``sketch``.
        2. Orthonormalise the sketch: Q = orth(A Ω).
        3. Repeat q times: Q̂ = orth(Aᵀ Q), then Q = orth(A Q̂).

    Args:
        op: Operator A of size M × N; may be matrix-free.
        rank: Target rank k.
        oversample: Extra columns p; ℓ = k + p, capped at min(M, N). Ignored when
            ``sketch`` is given.
        n_power_iter: Number of power iterations q.
        sketch: Optional test matrix as a sketch S with ``in_size == N`` (e.g.
            `SparseSignSketch`, `SRHTSketch`), applied as its transpose, Ω = Sᵀ,
            with ℓ = ``sketch.out_size``. ``None`` draws a Gaussian Ω.
        key: PRNG key for the Gaussian test matrix. ``None`` means
            ``jax.random.PRNGKey(0)``. Ignored when ``sketch`` is given.

    Returns:
        The basis Q, M × ℓ, with orthonormal columns.

    Raises:
        ValueError: On a non-positive ``rank``, negative ``oversample`` or
            ``n_power_iter``, a sketch whose ``in_size`` is not N, or a sketch with
            fewer than ``rank`` rows.

    Examples:
        A rank-five matrix is recovered by its basis:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> left = jr.normal(jr.key(0), (100, 5))
        >>> matrix = left @ jr.normal(jr.key(1), (5, 80))
        >>> op = lx.MatrixLinearOperator(matrix)
        >>> basis = gaussx.range_finder(op, 5, key=jr.key(2))
        >>> basis.shape
        (100, 15)
        >>> projection = einx.dot("row sketch, row col -> sketch col", basis, matrix)
        >>> bool(jnp.allclose(basis @ projection, matrix, atol=1e-4))
        True

        A sketch sets ℓ itself, so ``oversample`` no longer applies:

        >>> sketch = gaussx.GaussianSketch.sample(jr.key(3), 7, 80)
        >>> gaussx.range_finder(op, 5, sketch=sketch).shape
        (100, 7)

        A sketch of the wrong input size is rejected:

        >>> wrong = gaussx.GaussianSketch.sample(jr.key(3), 7, 60)
        >>> gaussx.range_finder(op, 5, sketch=wrong)
        Traceback (most recent call last):
        ...
        ValueError: sketch.in_size=60 must equal the operator's in_size=80.

    References:
        Halko, N., Martinsson, P.-G. & Tropp, J. A. (2011), Algorithm 4.4.
    """
    _check_args(rank, oversample, n_power_iter)
    out_size, in_size = op.out_size(), op.in_size()
    dtype = op.in_structure().dtype

    # --- test matrix ---
    if sketch is None:
        if key is None:
            key = jax.random.PRNGKey(0)
        # ℓ = min(k + p, M, N)
        sketch_size = min(rank + oversample, out_size, in_size)
        # Ω ~ 𝒩(0, I), N × ℓ
        test_matrix = jax.random.normal(key, (in_size, sketch_size), dtype=dtype)
    else:
        if sketch.in_size != in_size:
            raise ValueError(
                f"sketch.in_size={sketch.in_size} must equal the operator's "
                f"in_size={in_size}."
            )
        if sketch.out_size < rank:
            raise ValueError(
                f"sketch.out_size={sketch.out_size} must be at least rank={rank}."
            )
        # I_ℓ
        identity = jnp.eye(sketch.out_size, dtype=dtype)
        # Ω = Sᵀ
        test_matrix = sketch.apply_transpose(identity)

    # --- subspace iteration ---
    # A Ω
    image = _matmat(op, test_matrix)
    # Q = orth(A Ω)
    basis = _orth(image)
    if n_power_iter > 0:
        op_transpose = op.transpose()
        for _ in range(n_power_iter):
            # Aᵀ Q
            transpose_image = _matmat(op_transpose, basis)
            # Q̂ = orth(Aᵀ Q), so round-off does not lose the small directions
            transpose_basis = _orth(transpose_image)
            # A Q̂
            image = _matmat(op, transpose_basis)
            # Q = orth(A Q̂)
            basis = _orth(image)
    return basis


def qb(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    key: jax.Array | None = None,
) -> tuple[Float[Array, "m l"], Float[Array, "l n"]]:
    """Randomized QB factorisation A ≈ Q B with B = Qᵀ A.

        A ≈ Q B,   B = Qᵀ A = (Aᵀ Q)ᵀ

    ‖A − Q B‖ is the range-finder error (see `range_finder` for its bound and the
    advice on ``n_power_iter >= 2`` for slowly decaying spectra). Randomized methods
    target the **top** of the spectrum.

    Algorithm:
        1. Find the basis Q with `range_finder`.
        2. Project with ℓ more transpose matvecs, B = (Aᵀ Q)ᵀ, so A is never formed.

    Args:
        op: Operator A of size M × N; may be matrix-free.
        rank: Target rank k.
        oversample: Extra columns p; ℓ = k + p, capped at min(M, N).
        n_power_iter: Number of power iterations q.
        key: PRNG key for the Gaussian test matrix. ``None`` means
            ``jax.random.PRNGKey(0)``.

    Returns:
        Tuple ``(Q, B)`` of the basis Q (M × ℓ) and the projection B (ℓ × N).

    Raises:
        ValueError: On a non-positive ``rank``, or a negative ``oversample`` or
            ``n_power_iter``.

    Examples:
        A rank-four matrix factors exactly at ℓ = 8:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> left = jr.normal(jr.key(0), (60, 4))
        >>> matrix = left @ jr.normal(jr.key(1), (4, 40))
        >>> basis, projection = gaussx.qb(
        ...     lx.MatrixLinearOperator(matrix), 4, oversample=4
        ... )
        >>> basis.shape, projection.shape
        ((60, 8), (8, 40))
        >>> bool(jnp.allclose(basis @ projection, matrix, atol=1e-4))
        True

        ℓ is capped at min(M, N):

        >>> small = lx.MatrixLinearOperator(jnp.eye(3))
        >>> gaussx.qb(small, 2, oversample=10)[1].shape
        (3, 3)

        A negative ``oversample`` is rejected:

        >>> gaussx.qb(small, 2, oversample=-1)
        Traceback (most recent call last):
        ...
        ValueError: oversample must be non-negative, got -1.
    """
    # Q = orth((A Aᵀ)^q A Ω)
    basis = range_finder(
        op, rank, oversample=oversample, n_power_iter=n_power_iter, key=key
    )
    op_transpose = op.transpose()
    # Aᵀ Q
    transpose_image = _matmat(op_transpose, basis)
    # B = (Aᵀ Q)ᵀ
    projection = rearrange(transpose_image, "n l -> l n")
    return basis, projection
