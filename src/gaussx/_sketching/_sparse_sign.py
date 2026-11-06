"""Sparse sign sketch S ∈ ℝ^(d × M), the sparse Johnson-Lindenstrauss transform (SJLT).

Each column of S has ``nnz`` non-zeros; k = 0, …, nnz − 1 indexes them. r_k,j is the row
of non-zero k in column j and s_k,j its value.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from gaussx._einx import einsum, rearrange
from gaussx._sketching._base import AbstractSketch


class SparseSignSketch(AbstractSketch):
    """Sparse sign sketch (sparse Johnson-Lindenstrauss transform).

        S_(r_k,j),j = s_k,j = ±1 / √nnz,   k = 0, …, nnz − 1

    Each column of S ∈ ℝ^(d × M) has exactly ``nnz`` non-zeros, with independent random
    signs, at ``nnz`` distinct uniformly random rows. With O(log N) non-zeros per
    column, d = O(N log N / ε²) suffices for an ε-subspace embedding of an N-dimensional
    subspace (Cohen, 2016), and applying S to an M × N matrix costs O(nnz M N): the
    default sketch for tall problems.

    `apply` is a single ``segment_sum`` of ``signs * A[column]`` into ``rows``; no
    sparse-matrix library is involved.

    Attributes:
        rows: Row r_k,j of each non-zero, nnz × M; the ``nnz`` entries of each column
            are distinct.
        signs: Value s_k,j = ±1 / √nnz of each non-zero, nnz × M.
        in_size: The input size M.
        out_size: The sketch size d.

    Examples:
        Compress 1000 rows to 40:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> sketch = gaussx.SparseSignSketch.sample(jr.key(0), d=40, m=1000, nnz=4)
        >>> sketch.apply(jnp.ones((1000, 5))).shape
        (40, 5)

        Sketch a tall Jacobian available only as a matrix-free operator; a real one with
        10⁶ residuals and 200 parameters works the same with ``d=800, nnz=8``, and the
        singular values of the sketch are those of J within (1 ± ε):

        >>> jacobian = jr.normal(jr.key(1), (1000, 5))
        >>> structure = jax.ShapeDtypeStruct((5,), jacobian.dtype)
        >>> jacobian_op = lx.FunctionLinearOperator(lambda x: jacobian @ x, structure)
        >>> sketched = sketch.sketch_operator(jacobian_op)
        >>> sketched.shape
        (40, 5)
        >>> jnp.linalg.svd(sketched, compute_uv=False).shape
        (5,)

        ``nnz`` is clipped to d:

        >>> clipped = gaussx.SparseSignSketch.sample(jr.key(0), d=2, m=6, nnz=8)
        >>> clipped.rows.shape
        (2, 6)
    """

    rows: Int[Array, "nnz m"]
    signs: Float[Array, "nnz m"]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    @classmethod
    def sample(
        cls, key: jax.Array | None, d: int, m: int, *, nnz: int = 8
    ) -> SparseSignSketch:
        """Draw a d × M sparse sign sketch.

            r_k,j ∈ {0, …, d − 1} distinct in k,   s_k,j = ±1 / √nnz

        The ``nnz`` distinct rows of each column are drawn with Floyd's algorithm,
        vectorised over columns: O(nnz² M) work, never O(d M).

        Algorithm:
            1. For each l = d − nnz, …, d − 1, draw t ~ U{0, …, l} per column.
            2. Keep t unless an earlier draw in that column took it, else keep l.
            3. Draw the signs, ±1 / √nnz.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            d: Sketch size d (rows of S).
            m: Input size M (columns of S).
            nnz: Non-zeros per column; clipped to d.

        Returns:
            The sampled `SparseSignSketch`.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        nnz = min(nnz, d)
        row_key, sign_key = jax.random.split(key)
        step_keys = jax.random.split(row_key, nnz)
        rows: list[Array] = []
        # Floyd: l is never taken yet when its step begins, so the rows stay distinct.
        for step, step_key in enumerate(step_keys):
            # l = d − nnz + k
            largest_row = d - nnz + step
            # t ~ U{0, …, l}, per column
            candidate = jax.random.randint(step_key, (m,), 0, largest_row + 1)
            taken = jnp.zeros((m,), dtype=bool)
            for earlier_rows in rows:
                # t = r_k′,j, for an earlier k′
                clash = earlier_rows == candidate
                taken = taken | clash
            # r_k,j = l where t is taken, else t
            row = jnp.where(taken, largest_row, candidate)
            rows.append(row)
        # ±1
        rademacher_draw = jax.random.rademacher(sign_key, (nnz, m), dtype=float)
        # √nnz
        scale = jnp.sqrt(nnz)
        # s_k,j = ±1 / √nnz
        signs = rademacher_draw / scale
        stacked_rows = jnp.stack(rows)
        return cls(rows=stacked_rows, signs=signs, in_size=m, out_size=d)

    def apply(self, A: Float[Array, "m ..."]) -> Float[Array, "d ..."]:
        """Compute S A along the leading axis of ``A``.

            [S A]_r,: = ∑_(k, j : r_k,j = r) s_k,j A_j,:

        Args:
            A: Array A whose leading axis has length M.

        Returns:
            The product S A, whose leading axis has length d.
        """
        # in the input's dtype, so a float32 input never meets a float64 sketch
        signs = self.signs.astype(A.dtype)
        # s_k,j A_j,:
        contributions = einsum(signs, A, "k m, m ... -> k m ...")
        flat_contributions = rearrange(contributions, "k m ... -> (k m) ...")
        flat_rows = rearrange(self.rows, "k m -> (k m)")
        # [S A]_r,: = ∑_(k, j : r_k,j = r) s_k,j A_j,:
        return jax.ops.segment_sum(
            flat_contributions, flat_rows, num_segments=self.out_size
        )

    def apply_transpose(self, Y: Float[Array, "d ..."]) -> Float[Array, "m ..."]:
        """Compute Sᵀ Y along the leading axis of ``Y``.

            [Sᵀ Y]_j,: = ∑_k s_k,j Y_(r_k,j),:

        Args:
            Y: Array Y whose leading axis has length d.

        Returns:
            The product Sᵀ Y, whose leading axis has length M.
        """
        # in the input's dtype, so a float32 input never meets a float64 sketch
        signs = self.signs.astype(Y.dtype)
        # Accumulate one gather per non-zero: O(M N) memory, not O(nnz M N).
        # s_0,j Y_(r_0,j),:
        first_signs = signs[0]
        first_gather = Y[self.rows[0]]
        out = einsum(first_signs, first_gather, "m, m ... -> m ...")
        for k in range(1, signs.shape[0]):
            # s_k,j
            signs_k = signs[k]
            # Y_(r_k,j),:
            gather_k = Y[self.rows[k]]
            # s_k,j Y_(r_k,j),:
            term = einsum(signs_k, gather_k, "m, m ... -> m ...")
            # ∑_k s_k,j Y_(r_k,j),:
            out = out + term
        return out
