"""Randomized SVD and symmetric eigendecomposition."""

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
    r"""Truncated SVD $A \approx U \operatorname{diag}(s) V^\top$ by randomized QB.

    Computes $A \approx QB$ with `qb`, the small SVD $B = U_B \Sigma V^\top$,
    and lifts $U = Q U_B$, keeping the top ``rank`` triplets (Halko,
    Martinsson & Tropp, 2011, Algorithm 5.1). The cost is
    $(2q + 2)\ell$ matvecs with $A$ or $A^\top$, $\ell$ = ``rank +
    oversample``, plus $O((m+n)\ell^2)$ flops; $A$ is never formed.

    Randomized methods target the **top** of the spectrum: the leading
    singular triplets are accurate, the trailing ones are not. Use
    ``n_power_iter >= 2`` for slowly decaying spectra (Matérn-½ Gram
    matrices, most geophysical fields); see `range_finder` for the error
    bound.

    Args:
        op: Operator $A$ of shape ``(m, n)``; may be matrix-free.
        rank: Number of singular triplets $k$ to return (at most
            ``min(m, n)``).
        oversample: Extra columns $p$ in the range finder.
        n_power_iter: Number of power iterations $q$.
        key: PRNG key for the Gaussian test matrix. ``None`` means
            ``jax.random.PRNGKey(0)``.

    Returns:
        ``(U, s, Vt)`` of shapes ``(m, k)``, ``(k,)`` and ``(k, n)``, with
        ``s`` descending.

    Examples:
        50 EOFs of an anomaly matrix available only as a matvec, here a
        small dense stand-in:

        >>> import einx, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> X = jr.normal(jr.key(0), (300, 8)) @ jr.normal(jr.key(1), (8, 120))
        >>> U, s, Vt = gx.randomized_svd(lx.MatrixLinearOperator(X), 5, key=jr.key(2))
        >>> U.shape, s.shape, Vt.shape
        ((300, 5), (5,), (5, 120))
        >>> eofs, pcs = U, einx.multiply("k, k t -> k t", s, Vt)
    """
    Q, B = qb(op, rank, oversample=oversample, n_power_iter=n_power_iter, key=key)
    U_b, s, Vt = jnp.linalg.svd(B, full_matrices=False)
    k = min(rank, s.shape[0])
    U = einsum(Q, U_b[:, :k], "m l, l k -> m k")
    return U, s[:k], Vt[:k]


def randomized_eigh(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    which: Literal["largest", "magnitude"] = "largest",
    key: jax.Array | None = None,
) -> tuple[Float[Array, " k"], Float[Array, "n k"]]:
    r"""Partial eigendecomposition of a symmetric operator by Rayleigh-Ritz.

    Finds an orthonormal $Q$ for the dominant range of $A$ with
    `range_finder`, forms the Rayleigh-Ritz matrix $T = Q^\top A Q$
    ($\ell$ more matvecs), and lifts the eigenpairs of $T$:
    $A \approx (QW)\Lambda(QW)^\top$ with $T = W\Lambda W^\top$. $A$ may be
    indefinite. The range finder captures the eigenvalues of largest
    **magnitude** (randomized methods target the **top** of the spectrum),
    and ``rank`` Ritz pairs are kept by ``which``:

    - ``"largest"``: the algebraically largest Ritz values;
    - ``"magnitude"``: the Ritz values of largest absolute value.

    For an indefinite $A$ whose large negative eigenvalues dominate,
    ``"largest"`` is only as good as the subspace, so prefer
    ``"magnitude"`` there. The small end of a spectrum (e.g. the smallest
    eigenvalues of a graph Laplacian) is Lanczos / LOBPCG territory.

    Use ``n_power_iter >= 2`` for slowly decaying spectra (Matérn-½ Gram
    matrices, most geophysical fields). With ``n_power_iter=0`` this is the
    one-pass randomized Rayleigh-Ritz projection. For PSD operators,
    `randomized_nystrom` is strictly more accurate for the same number of
    matvecs (Tropp et al., 2017).

    Args:
        op: Symmetric operator $A$ of shape ``(n, n)``; may be matrix-free.
        rank: Number of eigenpairs $k$ to return.
        oversample: Extra columns $p$ in the range finder.
        n_power_iter: Number of power iterations $q$.
        which: ``"largest"`` or ``"magnitude"``.
        key: PRNG key for the Gaussian test matrix. ``None`` means
            ``jax.random.PRNGKey(0)``.

    Returns:
        ``(eigenvalues, eigenvectors)`` of shapes ``(k,)`` and ``(n, k)``,
        eigenvalues in ascending order (as `jax.numpy.linalg.eigh`),
        eigenvectors orthonormal.

    Raises:
        ValueError: If ``which`` is invalid or the operator is not square.

    Examples:
        >>> import einx, jax.numpy as jnp, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> lam = jnp.array([-9.0, 5.0, 1.0, 0.1, 0.01, 0.0])
        >>> Q, _ = jnp.linalg.qr(jr.normal(jr.key(0), (6, 6)))
        >>> A = einx.dot("i k, j k -> i j", einx.multiply("i k, k -> i k", Q, lam), Q)
        >>> A = lx.MatrixLinearOperator(A, lx.symmetric_tag)
        >>> vals, vecs = gx.randomized_eigh(A, 2, oversample=2, which="magnitude")
        >>> bool(jnp.allclose(vals, jnp.array([-9.0, 5.0]), atol=1e-4))
        True
        >>> vals, _ = gx.randomized_eigh(A, 2, oversample=2, which="largest")
        >>> bool(jnp.allclose(vals, jnp.array([1.0, 5.0]), atol=1e-4))
        True
    """
    if which not in ("largest", "magnitude"):
        raise ValueError(f"which must be 'largest' or 'magnitude', got {which!r}.")
    if op.in_size() != op.out_size():
        raise ValueError(
            f"randomized_eigh needs a square operator, got "
            f"{op.out_size()}x{op.in_size()}."
        )
    Q = range_finder(
        op, rank, oversample=oversample, n_power_iter=n_power_iter, key=key
    )
    T = symmetrize(einsum(Q, _matmat(op, Q), "n a, n b -> a b"))
    vals, W = jnp.linalg.eigh(T)
    k = min(rank, vals.shape[0])
    if which == "largest":
        keep = jnp.arange(vals.shape[0] - k, vals.shape[0])
    else:
        keep = jnp.sort(jnp.argsort(-jnp.abs(vals))[:k])
    vecs = einsum(Q, W[:, keep], "n l, l k -> n k")
    return vals[keep], vecs
