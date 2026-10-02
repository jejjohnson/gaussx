"""Randomized range finder and QB factorisation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._sketching._base import AbstractSketch


def _matmat(op: lx.AbstractLinearOperator, X: Float[Array, "n l"]) -> Array:
    """``A X`` column by column, without materialising ``A``."""
    if isinstance(op, lx.MatrixLinearOperator):
        return op.matrix @ X
    return jax.vmap(op.mv, in_axes=1, out_axes=1)(X)


def _orth(Y: Float[Array, "m l"]) -> Float[Array, "m l"]:
    """Orthonormal basis of ``range(Y)`` from a thin QR."""
    Q, _ = jnp.linalg.qr(Y)
    return Q


def _check_args(rank: int, oversample: int, n_power_iter: int) -> None:
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
    r"""Orthonormal basis $Q$ for the dominant range of $A$.

    Randomized subspace iteration (Halko, Martinsson & Tropp, 2011,
    Algorithm 4.4): draw a test matrix $\Omega \in \mathbb R^{n\times\ell}$
    with $\ell$ = ``rank + oversample``, set $Q = \operatorname{orth}(A\Omega)$,
    then repeat ``n_power_iter`` times

    $$
    \hat Q = \operatorname{orth}(A^\top Q),\qquad Q = \operatorname{orth}(A\hat Q),
    $$

    re-orthonormalising after every half-step so that small directions are
    not lost to round-off. For a Gaussian $\Omega$ (HMT 2011, Thm 10.6),

    $$
    \mathbb E\,\|A - QQ^\top A\|_2 \le
    \Big(1+\sqrt{\tfrac{k}{p-1}}\Big)\sigma_{k+1}
    + \frac{e\sqrt{k+p}}{p}\Big(\sum_{j>k}\sigma_j^2\Big)^{1/2},
    $$

    with $k$ = ``rank`` and $p$ = ``oversample``. The tail term dominates
    when the spectrum decays slowly (Matérn-½ Gram matrices, most
    geophysical fields); $q$ power iterations apply the bound to
    $(AA^\top)^q A$, whose singular values are $\sigma_j^{2q+1}$, at the cost
    of $2q$ more passes over $A$. Use ``n_power_iter >= 2`` for slowly
    decaying spectra.

    Randomized methods target the **top** of the spectrum (the largest
    singular values). For the small end, use Lanczos or LOBPCG.

    $A$ is touched only through ``mv`` (and the transpose's ``mv`` for power
    steps), vmapped over the $\ell$ columns.

    Args:
        op: Operator $A$ of shape ``(m, n)``; may be matrix-free.
        rank: Target rank $k$.
        oversample: Extra columns $p$; $\ell = k + p$, capped at
            ``min(m, n)``. Ignored when ``sketch`` is given.
        n_power_iter: Number of power iterations $q$.
        sketch: Optional test matrix as a sketch $S$ with ``in_size == n``
            (e.g. `SparseSignSketch`, `SRHTSketch`), applied as its transpose,
            $\Omega = S^\top \in \mathbb R^{n\times\ell}$ with $\ell$ =
            ``sketch.out_size``. ``None`` draws a Gaussian $\Omega$.
        key: PRNG key for the Gaussian test matrix. ``None`` means
            ``jax.random.PRNGKey(0)``. Ignored when ``sketch`` is given.

    Returns:
        $Q$ with orthonormal columns, shape ``(m, l)``.

    Raises:
        ValueError: On a non-positive ``rank``, negative ``oversample`` or
            ``n_power_iter``, a sketch whose ``in_size`` is not ``n``, or a
            sketch with fewer than ``rank`` rows.

    Examples:
        >>> import einx, jax.numpy as jnp, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> A = jr.normal(jr.key(0), (100, 5)) @ jr.normal(jr.key(1), (5, 80))
        >>> Q = gx.range_finder(lx.MatrixLinearOperator(A), 5, key=jr.key(2))
        >>> Q.shape
        (100, 15)
        >>> QtA = einx.dot("m l, m n -> l n", Q, A)
        >>> bool(jnp.allclose(Q @ QtA, A, atol=1e-4))
        True
    """
    _check_args(rank, oversample, n_power_iter)
    m, n = op.out_size(), op.in_size()
    dtype = op.in_structure().dtype
    if sketch is None:
        if key is None:
            key = jax.random.PRNGKey(0)
        ell = min(rank + oversample, m, n)
        omega = jax.random.normal(key, (n, ell), dtype=dtype)
    else:
        if sketch.in_size != n:
            raise ValueError(
                f"sketch.in_size={sketch.in_size} must equal the operator's "
                f"in_size={n}."
            )
        if sketch.out_size < rank:
            raise ValueError(
                f"sketch.out_size={sketch.out_size} must be at least rank={rank}."
            )
        omega = sketch.apply_transpose(jnp.eye(sketch.out_size, dtype=dtype))

    Q = _orth(_matmat(op, omega))
    if n_power_iter > 0:
        op_t = op.transpose()
        for _ in range(n_power_iter):
            Q = _orth(_matmat(op, _orth(_matmat(op_t, Q))))
    return Q


def qb(
    op: lx.AbstractLinearOperator,
    rank: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    key: jax.Array | None = None,
) -> tuple[Float[Array, "m l"], Float[Array, "l n"]]:
    r"""Randomized QB factorisation $A \approx QB$ with $B = Q^\top A$.

    $Q$ comes from `range_finder`; $B$ costs $\ell$ more transpose-matvecs,
    $B = (A^\top Q)^\top$, so $A$ is never formed. $\|A - QB\|$ is the
    range-finder error (see `range_finder` for its bound and the advice on
    ``n_power_iter >= 2`` for slowly decaying spectra). Randomized methods
    target the **top** of the spectrum.

    Args:
        op: Operator $A$ of shape ``(m, n)``; may be matrix-free.
        rank: Target rank $k$.
        oversample: Extra columns $p$; $\ell = k + p$, capped at
            ``min(m, n)``.
        n_power_iter: Number of power iterations $q$.
        key: PRNG key for the Gaussian test matrix. ``None`` means
            ``jax.random.PRNGKey(0)``.

    Returns:
        ``(Q, B)`` of shapes ``(m, l)`` and ``(l, n)``.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> A = jr.normal(jr.key(0), (60, 4)) @ jr.normal(jr.key(1), (4, 40))
        >>> Q, B = gx.qb(lx.MatrixLinearOperator(A), 4, oversample=4)
        >>> Q.shape, B.shape
        ((60, 8), (8, 40))
        >>> bool(jnp.allclose(Q @ B, A, atol=1e-4))
        True
    """
    Q = range_finder(
        op, rank, oversample=oversample, n_power_iter=n_power_iter, key=key
    )
    B = rearrange(_matmat(op.transpose(), Q), "n l -> l n")
    return Q, B
