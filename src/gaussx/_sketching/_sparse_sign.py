"""Sparse sign (SJLT) sketch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from gaussx._einx import einsum, rearrange
from gaussx._sketching._base import AbstractSketch


class SparseSignSketch(AbstractSketch):
    r"""Sparse sign sketch (sparse Johnson-Lindenstrauss transform).

    Each column of $S \in \mathbb{R}^{d \times m}$ has exactly ``nnz``
    non-zeros, $\pm 1/\sqrt{\text{nnz}}$ with independent random signs, at
    ``nnz`` distinct uniformly random rows. With $O(\log n)$ non-zeros per
    column, $d = O(n \log n / \varepsilon^2)$ suffices for an
    $\varepsilon$-subspace embedding of an $n$-dimensional subspace (Cohen,
    2016), and applying $S$ costs $O(\text{nnz} \cdot m \cdot n)$: the default
    sketch for tall problems.

    `apply` is a single ``segment_sum`` of ``signs * A[column]`` into
    ``rows``; no sparse-matrix library is involved.

    Attributes:
        rows: Row index of each non-zero, shape ``(nnz, m)``; the ``nnz``
            entries of each column are distinct.
        signs: Value of each non-zero, $\pm 1/\sqrt{\text{nnz}}$, shape
            ``(nnz, m)``.
        in_size: $m$.
        out_size: $d$.

    Examples:
        Sketch a tall Jacobian (10⁶ residuals × 200 parameters), available
        only as a matrix-free ``J_op``, down to 800 rows:

        ```python
        S = gx.SparseSignSketch.sample(key, d=800, m=1_000_000, nnz=8)
        SJ = S.sketch_operator(J_op)  # (800, 200), matrix-free
        sv = jnp.linalg.svd(SJ, compute_uv=False)  # J's singular values, within (1 ± ε)
        ```

        A small runnable version:

        >>> import jax.numpy as jnp, jax.random as jr
        >>> import gaussx as gx
        >>> S = gx.SparseSignSketch.sample(jr.key(0), d=40, m=1000, nnz=4)
        >>> S.apply(jnp.ones((1000, 5))).shape
        (40, 5)
    """

    rows: Int[Array, "nnz m"]
    signs: Float[Array, "nnz m"]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    @classmethod
    def sample(
        cls, key: jax.Array | None, d: int, m: int, *, nnz: int = 8
    ) -> SparseSignSketch:
        r"""Draw a ``(d, m)`` sparse sign sketch.

        The ``nnz`` distinct rows of each column are drawn with Floyd's
        algorithm, vectorised over columns: $O(\text{nnz}^2 m)$ work, never
        $O(dm)$.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            d: Sketch size (rows of $S$).
            m: Input size (columns of $S$).
            nnz: Non-zeros per column; clipped to ``d``.

        Returns:
            The sampled `SparseSignSketch`.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        nnz = min(nnz, d)
        row_key, sign_key = jax.random.split(key)
        # Floyd: for j = d - nnz, ..., d - 1 draw t ~ U{0..j}; keep t unless it
        # is already taken, in which case keep j (never taken yet).
        rows: list[Array] = []
        for k, step_key in enumerate(jax.random.split(row_key, nnz)):
            j = d - nnz + k
            t = jax.random.randint(step_key, (m,), 0, j + 1)
            taken = jnp.zeros((m,), dtype=bool)
            for r in rows:
                taken = taken | (r == t)
            rows.append(jnp.where(taken, j, t))
        signs = jax.random.rademacher(sign_key, (nnz, m), dtype=float) / jnp.sqrt(nnz)
        return cls(rows=jnp.stack(rows), signs=signs, in_size=m, out_size=d)

    def apply(self, A: Float[Array, "m ..."]) -> Float[Array, "d ..."]:
        contributions = einsum(self.signs.astype(A.dtype), A, "k m, m ... -> k m ...")
        return jax.ops.segment_sum(
            rearrange(contributions, "k m ... -> (k m) ..."),
            rearrange(self.rows, "k m -> (k m)"),
            num_segments=self.out_size,
        )

    def apply_transpose(self, Y: Float[Array, "d ..."]) -> Float[Array, "m ..."]:
        # Accumulate one gather per non-zero: O(m · ncols) memory, not O(nnz · m).
        signs = self.signs.astype(Y.dtype)
        out = einsum(signs[0], Y[self.rows[0]], "m, m ... -> m ...")
        for k in range(1, signs.shape[0]):
            out = out + einsum(signs[k], Y[self.rows[k]], "m, m ... -> m ...")
        return out
