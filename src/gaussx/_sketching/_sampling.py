"""Row-sampling sketch S ∈ ℝ^(d × M), with row k of S a scaled unit vector.

i_k ∈ {0, …, M − 1} is the k-th sampled row and p the probabilities it is drawn from.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int

from gaussx._einx import einsum
from gaussx._sketching._base import AbstractSketch


class RowSamplingSketch(AbstractSketch):
    """Weighted row-sampling sketch.

        S_k,: = e_(i_k)ᵀ / √(d p_(i_k)),   i_k ~ p,   𝔼[Sᵀ S] = I_M

    The i_k are drawn i.i.d. (with replacement). With leverage-score probabilities this
    is a subspace embedding; with uniform probabilities it is only safe once the
    leverage has been flattened (as inside `SRHTSketch`). Applying S to an M × N matrix
    is a gather: O(d N).

    Attributes:
        rows: Sampled row indices i_k, of length d.
        weights: Row weights 1 / √(d p_(i_k)), of length d.
        in_size: The input size M.
        out_size: The sketch size d.

    Examples:
        Uniform sampling, 10 rows of 100:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import gaussx
        >>> sketch = gaussx.RowSamplingSketch.sample(jr.key(0), d=10, m=100)
        >>> sketch.apply(jnp.ones((100, 2))).shape
        (10, 2)

        A row with zero probability is never drawn:

        >>> weights = jnp.array([1.0, 0.0, 1.0, 2.0])
        >>> weighted = gaussx.RowSamplingSketch.sample(
        ...     jr.key(1), d=20, m=4, probabilities=weights
        ... )
        >>> bool(jnp.all(weighted.rows != 1))
        True

        The transpose accumulates repeated rows onto the M input rows:

        >>> weighted.apply_transpose(jnp.ones(20)).shape
        (4,)
    """

    rows: Int[Array, " d"]
    weights: Float[Array, " d"]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    @classmethod
    def sample(
        cls,
        key: jax.Array | None,
        d: int,
        m: int,
        *,
        probabilities: Float[ArrayLike, " m"] | None = None,
    ) -> RowSamplingSketch:
        """Draw a d × M row-sampling sketch.

            i_k ~ p,   w_k = 1 / √(d p_(i_k))

        Algorithm:
            1. Normalise the weights to the probabilities p, or take p_i = 1 / M.
            2. Draw d rows i_k ~ p with replacement.
            3. Scale each sampled row by w_k.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            d: Number d of sampled rows.
            m: Input size M (columns of S).
            probabilities: Non-negative sampling weights, of length M, normalised
                internally to p. ``None`` means uniform.

        Returns:
            The sampled `RowSamplingSketch`.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        if probabilities is None:
            # p_i = 1 / M
            row_probabilities = jnp.full((m,), 1.0 / m)
        else:
            row_weights = jnp.asarray(probabilities)
            # the total, to normalise by
            total_weight = jnp.sum(row_weights)
            # p_i = weight_i / total
            row_probabilities = row_weights / total_weight
        # i_k ~ p
        rows = jax.random.choice(key, m, (d,), replace=True, p=row_probabilities)
        # p_(i_k)
        sampled_probabilities = row_probabilities[rows]
        # d p_(i_k)
        scaled_probabilities = d * sampled_probabilities
        # √(d p_(i_k))
        root_probabilities = jnp.sqrt(scaled_probabilities)
        # w_k = 1 / √(d p_(i_k))
        weights = 1.0 / root_probabilities
        return cls(rows=rows, weights=weights, in_size=m, out_size=d)

    def apply(self, A: Float[Array, "m ..."]) -> Float[Array, "d ..."]:
        """Compute S A along the leading axis of ``A``.

            [S A]_k,: = w_k A_(i_k),:

        Args:
            A: Array A whose leading axis has length M.

        Returns:
            The product S A, whose leading axis has length d.
        """
        # in the input's dtype, so a float32 input never meets a float64 sketch
        weights = self.weights.astype(A.dtype)
        # A_(i_k),:
        sampled_rows = A[self.rows]
        # [S A]_k,: = w_k A_(i_k),:
        return einsum(weights, sampled_rows, "d, d ... -> d ...")

    def apply_transpose(self, Y: Float[Array, "d ..."]) -> Float[Array, "m ..."]:
        """Compute Sᵀ Y along the leading axis of ``Y``.

            [Sᵀ Y]_i,: = ∑_(k : i_k = i) w_k Y_k,:

        Args:
            Y: Array Y whose leading axis has length d.

        Returns:
            The product Sᵀ Y, whose leading axis has length M.
        """
        # in the input's dtype, so a float32 input never meets a float64 sketch
        weights = self.weights.astype(Y.dtype)
        # w_k Y_k,:
        weighted = einsum(weights, Y, "d, d ... -> d ...")
        zeros = jnp.zeros((self.in_size, *Y.shape[1:]), dtype=Y.dtype)
        # [Sᵀ Y]_i,: = ∑_(k : i_k = i) w_k Y_k,:
        return zeros.at[self.rows].add(weighted)
