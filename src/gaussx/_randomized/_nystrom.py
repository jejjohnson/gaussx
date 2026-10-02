"""Randomized Nyström approximation of a PSD operator."""

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
    r"""Randomized Nyström approximation $\hat A = U\hat\Lambda U^\top$ of a PSD $A$.

    For a test matrix $\Omega$ the Nyström approximation is

    $$
    \hat A = (A\Omega)\,(\Omega^\top A\Omega)^{+}\,(A\Omega)^\top,
    \qquad 0 \preceq \hat A \preceq A.
    $$

    It costs one pass ($\ell$ = ``rank + oversample`` matvecs) and, for the
    same $\ell$, is more accurate than the Rayleigh-Ritz approximation
    $QQ^\top AQQ^\top$ of `randomized_eigh` (Tropp, Yurtsever, Udell &
    Cevher, 2017). The algorithm is their Algorithm 3:

    1. $\Omega = \operatorname{qr}(\text{randn}(n, \ell))$;
    2. $Y = A\Omega$ and the shift $\nu = \sqrt n\,\varepsilon\,\|Y\|_2$;
    3. $Y_\nu = Y + \nu\Omega$, $C = \operatorname{chol}(\Omega^\top Y_\nu)$,
       $B = Y_\nu C^{-\top}$;
    4. $U, \Sigma, \_ = \operatorname{svd}(B)$,
       $\hat\Lambda = \max(\Sigma^2 - \nu, 0)$.

    The shift $\nu$ only stabilises the small Cholesky (it keeps float32
    finite) and is subtracted again in step 4. With ``oversample > 0`` the
    top ``rank`` eigenpairs of the rank-$\ell$ approximation are kept.

    The result is an orthonormal `LowRankUpdate` with a zero diagonal base,
    so the same factors on a $\sigma^2 I$ base
    (`gaussx.svd_low_rank_plus_diag`) give $\hat A + \sigma^2 I$, whose
    `gaussx.solve`, `gaussx.logdet` and the rest dispatch through the
    Woodbury rules. Randomized methods target the **top** of the spectrum;
    $A$ must be PSD (for symmetric indefinite $A$ use `randomized_eigh`).

    Args:
        op: PSD operator $A$ of shape ``(n, n)``; may be matrix-free (touched
            only through ``mv``, vmapped over the $\ell$ columns).
        rank: Number of eigenpairs $k$ to return.
        oversample: Extra columns $p$; $\ell = k + p$, capped at $n$.
        key: PRNG key for the Gaussian test matrix. ``None`` means
            ``jax.random.PRNGKey(0)``.

    Returns:
        ``LowRankUpdate(base=0, U=U, d=Λ̂, V=U, orthonormal=True)``, tagged
        symmetric and PSD, with ``U`` of shape ``(n, k)`` and ``Λ̂``
        descending.

    Raises:
        ValueError: On a non-positive ``rank``, a negative ``oversample``, or
            a non-square operator.

    Examples:
        >>> import einx, jax.numpy as jnp, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> W = jr.normal(jr.key(0), (50, 4))
        >>> A = lx.MatrixLinearOperator(
        ...     einx.dot("i r, j r -> i j", W, W), lx.positive_semidefinite_tag
        ... )
        >>> A_hat = gx.randomized_nystrom(A, 4, oversample=2, key=jr.key(1))
        >>> A_hat.U.shape, A_hat.d.shape
        ((50, 4), (4,))
        >>> bool(jnp.allclose(A_hat.as_matrix(), A.as_matrix(), atol=1e-3))
        True

        Add the noise and solve with the Woodbury identity:

        >>> noisy = gx.svd_low_rank_plus_diag(
        ...     jnp.full(50, 0.1), A_hat.U, A_hat.d, A_hat.U, psd=True
        ... )
        >>> x = gx.solve(noisy, jnp.ones(50))
    """
    if rank < 1:
        raise ValueError(f"rank must be a positive integer, got {rank}.")
    if oversample < 0:
        raise ValueError(f"oversample must be non-negative, got {oversample}.")
    n = op.in_size()
    if op.out_size() != n:
        raise ValueError(
            f"randomized_nystrom needs a square operator, got {op.out_size()}x{n}."
        )
    if key is None:
        key = jax.random.PRNGKey(0)
    dtype = op.in_structure().dtype
    ell = min(rank + oversample, n)

    omega, _ = jnp.linalg.qr(jax.random.normal(key, (n, ell), dtype=dtype))
    Y = _matmat(op, omega)
    # The shift only stabilises the small Cholesky; it is removed again below.
    nu = jnp.sqrt(jnp.asarray(n, dtype)) * jnp.finfo(dtype).eps * jnp.linalg.norm(Y, 2)
    Y_nu = Y + nu * omega
    C = jnp.linalg.cholesky(symmetrize(einsum(omega, Y_nu, "n a, n b -> a b")))
    # B = Y_nu C⁻ᵀ, so B Bᵀ = Y_nu (Ωᵀ Y_nu)⁻¹ Y_nuᵀ.
    Bt = jax.scipy.linalg.solve_triangular(C, rearrange(Y_nu, "n l -> l n"), lower=True)
    U, s, _ = jnp.linalg.svd(rearrange(Bt, "l n -> n l"), full_matrices=False)
    k = min(rank, ell)
    U, eigenvalues = U[:, :k], jnp.maximum(s[:k] ** 2 - nu, 0.0)
    # V is U (the same object), so the update is symmetric and PSD-tagged.
    zeros = jnp.zeros(n, dtype=dtype)
    return svd_low_rank_plus_diag(zeros, U, eigenvalues, U, psd=True)
