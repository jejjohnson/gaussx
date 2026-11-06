"""Randomized Nyström approximation of a PSD operator A.

A is N × N and PSD. Its rank-k approximation Â = U diag(γ̂) Uᵀ, with orthonormal U and
eigenvalues γ̂, comes from a test matrix Ω (N × ℓ; the letter of the paper, not the
spatial domain) with ℓ = k + p columns, for oversampling p.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx

from gaussx._einx import einsum, rearrange
from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._low_rank_update import LowRankUpdate, svd_low_rank_plus_diag
from gaussx._randomized._range_finder import _matmat


def randomized_nystrom(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 0,
    key: jax.Array | None = None,
) -> LowRankUpdate:
    """Randomized Nyström approximation Â = U diag(γ̂) Uᵀ of a PSD A.

        Â = (A Ω) (Ωᵀ A Ω)⁺ (A Ω)ᵀ,   0 ⪯ Â ⪯ A

    It costs one pass (ℓ = k + p matvecs) and, for the same ℓ, is more accurate than
    the Rayleigh-Ritz approximation Q Qᵀ A Q Qᵀ of `randomized_eigh`.

    The shift ν only stabilises the small Cholesky (it keeps float32 finite) and is
    subtracted again in step 4. With ``oversample > 0`` the top k eigenpairs of the
    rank-ℓ approximation are kept.

    The result is an orthonormal `LowRankUpdate` with a zero diagonal base, so the
    same factors on a σ² I base (`gaussx.svd_low_rank_plus_diag`) give Â + σ² I, whose
    `gaussx.solve`, `gaussx.logdet` and the rest dispatch through the Woodbury rules.
    Randomized methods target the **top** of the spectrum; A must be PSD (for
    symmetric indefinite A use `randomized_eigh`).

    Algorithm:
        1. Orthonormalise a Gaussian draw into Ω.
        2. Sketch Y = A Ω and set the shift ν = √N ε ‖Y‖₂.
        3. Shift Y_ν = Y + ν Ω, factor C Cᵀ = Ωᵀ Y_ν and solve B = Y_ν C⁻ᵀ.
        4. Take B = U diag(s) (…)ᵀ and keep γ̂ = max(s² − ν, 0) for the top k.

    Args:
        op: PSD operator A of size N × N; may be matrix-free (touched only through
            ``mv``, vmapped over the ℓ columns).
        rank: Number of eigenpairs k to return.
        oversample: Extra columns p; ℓ = k + p, capped at N.
        key: PRNG key for the Gaussian test matrix. ``None`` means
            ``jax.random.PRNGKey(0)``.

    Returns:
        ``LowRankUpdate(base=0, U=U, d=γ̂, V=U, orthonormal=True)``, tagged symmetric
        and PSD, with U of size N × k and γ̂ descending.

    Raises:
        ValueError: On a non-positive ``rank``, a negative ``oversample``, or a
            non-square operator.

    Examples:
        A rank-four PSD matrix is recovered:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> factor = jr.normal(jr.key(0), (50, 4))
        >>> matrix = einx.dot("row r, col r -> row col", factor, factor)
        >>> op = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
        >>> approximation = gaussx.randomized_nystrom(
        ...     op, 4, oversample=2, key=jr.key(1)
        ... )
        >>> approximation.U.shape, approximation.d.shape
        ((50, 4), (4,))
        >>> bool(jnp.allclose(approximation.as_matrix(), matrix, atol=1e-3))
        True

        Add the noise and solve with the Woodbury identity:

        >>> noisy = gaussx.svd_low_rank_plus_diag(
        ...     jnp.full(50, 0.1),
        ...     approximation.U,
        ...     approximation.d,
        ...     approximation.U,
        ...     psd=True,
        ... )
        >>> gaussx.solve(noisy, jnp.ones(50)).shape
        (50,)

        A negative ``oversample`` is rejected:

        >>> gaussx.randomized_nystrom(op, 4, oversample=-1)
        Traceback (most recent call last):
        ...
        ValueError: oversample must be non-negative, got -1.

    References:
        Tropp, J. A., Yurtsever, A., Udell, M. & Cevher, V. (2017), Algorithm 3.
    """
    if rank < 1:
        raise ValueError(f"rank must be a positive integer, got {rank}.")
    if oversample < 0:
        raise ValueError(f"oversample must be non-negative, got {oversample}.")
    size = op.in_size()
    if op.out_size() != size:
        raise ValueError(
            f"randomized_nystrom needs a square operator, got {op.out_size()}x{size}."
        )
    if key is None:
        key = jax.random.PRNGKey(0)
    dtype = op.in_structure().dtype
    # ℓ = min(k + p, N)
    sketch_size = min(rank + oversample, size)

    # --- test matrix and sketch ---
    # randn(N, ℓ)
    gaussian_draw = jax.random.normal(key, (size, sketch_size), dtype=dtype)
    # Ω = orth(randn(N, ℓ))
    test_matrix, _ = jnp.linalg.qr(gaussian_draw)
    # Y = A Ω
    image = _matmat(op, test_matrix)

    # --- shifted Cholesky ---
    # √N
    root_size = jnp.sqrt(jnp.asarray(size, dtype))
    # √N ε
    relative_shift = root_size * jnp.finfo(dtype).eps
    # ‖Y‖₂
    image_norm = jnp.linalg.norm(image, 2)
    # ν = √N ε ‖Y‖₂, which only stabilises the small Cholesky and is removed below
    shift = relative_shift * image_norm
    # Y_ν = Y + ν Ω
    shifted_image = image + shift * test_matrix
    # Ωᵀ Y_ν
    core = einsum(test_matrix, shifted_image, "n a, n b -> a b")
    # sym(Ωᵀ Y_ν)
    core = symmetrize(core)
    # C Cᵀ = Ωᵀ Y_ν
    core_factor = jnp.linalg.cholesky(core)

    # --- singular values to eigenvalues ---
    # Y_νᵀ
    shifted_image_transpose = rearrange(shifted_image, "n l -> l n")
    # Bᵀ = C⁻¹ Y_νᵀ, so B Bᵀ = Y_ν (Ωᵀ Y_ν)⁻¹ Y_νᵀ
    nystrom_factor_transpose = jax.scipy.linalg.solve_triangular(
        core_factor, shifted_image_transpose, lower=True
    )
    # B = Y_ν C⁻ᵀ
    nystrom_factor = rearrange(nystrom_factor_transpose, "l n -> n l")
    # B = U diag(s) (…)ᵀ
    left, singular_values, _ = jnp.linalg.svd(nystrom_factor, full_matrices=False)
    # min(k, ℓ)
    kept = min(rank, sketch_size)
    # U restricted to the top k
    eigenvectors = left[:, :kept]
    # s_1:k
    top_singular_values = singular_values[:kept]
    # s²
    squared = top_singular_values**2
    # γ̂ = max(s² − ν, 0)
    eigenvalues = jnp.maximum(squared - shift, 0.0)
    # 0, the diagonal base
    zeros = jnp.zeros(size, dtype=dtype)
    # V is U (the same object), so the update is symmetric and PSD-tagged.
    return svd_low_rank_plus_diag(
        zeros, eigenvectors, eigenvalues, eigenvectors, psd=True
    )
