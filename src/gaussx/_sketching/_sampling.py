"""Row-sampling sketch."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int

from gaussx._einx import einsum
from gaussx._sketching._base import AbstractSketch


class RowSamplingSketch(AbstractSketch):
    r"""Weighted row-sampling sketch.

    Row $k$ of $S$ is $e_{i_k}^\top / \sqrt{d\, p_{i_k}}$ with
    $i_k \sim p$ drawn i.i.d. (with replacement), so
    $\mathbb{E}[S^\top S] = I_m$. With leverage-score probabilities this is
    a subspace embedding; with uniform probabilities it is only safe once the
    leverage has been flattened (as inside `SRHTSketch`). Applying $S$ is a
    gather: $O(dn)$.

    Attributes:
        rows: Sampled row indices $i_k$, shape ``(d,)``.
        weights: Row weights $1/\sqrt{d\, p_{i_k}}$, shape ``(d,)``.
        in_size: $m$.
        out_size: $d$.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> import gaussx as gx
        >>> S = gx.RowSamplingSketch.sample(jr.key(0), d=10, m=100)
        >>> S.apply(jnp.ones((100, 2))).shape
        (10, 2)
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
        """Draw a ``(d, m)`` row-sampling sketch.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            d: Number of sampled rows.
            m: Input size (columns of $S$).
            probabilities: Non-negative sampling weights $p$, shape ``(m,)``,
                normalised internally. ``None`` means uniform.

        Returns:
            The sampled `RowSamplingSketch`.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        if probabilities is None:
            p = jnp.full((m,), 1.0 / m)
        else:
            p = jnp.asarray(probabilities)
            p = p / jnp.sum(p)
        rows = jax.random.choice(key, m, (d,), replace=True, p=p)
        return cls(
            rows=rows, weights=1.0 / jnp.sqrt(d * p[rows]), in_size=m, out_size=d
        )

    def apply(self, A: Float[Array, "m ..."]) -> Float[Array, "d ..."]:
        return einsum(self.weights.astype(A.dtype), A[self.rows], "d, d ... -> d ...")

    def apply_transpose(self, Y: Float[Array, "d ..."]) -> Float[Array, "m ..."]:
        weighted = einsum(self.weights.astype(Y.dtype), Y, "d, d ... -> d ...")
        return (
            jnp.zeros((self.in_size, *Y.shape[1:]), dtype=Y.dtype)
            .at[self.rows]
            .add(weighted)
        )
