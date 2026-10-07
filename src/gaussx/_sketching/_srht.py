"""Subsampled randomized Hadamard transform (SRHT) sketch."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from gaussx._einx import einsum, rearrange
from gaussx._sketching._base import AbstractSketch
from gaussx._sketching._hadamard import hadamard_transform


class SRHTSketch(AbstractSketch):
    r"""Subsampled randomized Hadamard transform sketch.

    $$
    S = \sqrt{m_2 / d}\; R\, \tfrac{1}{\sqrt{m_2}} H\, D\, P ,
    $$

    where $P$ is a random permutation of the $m$ input rows, $D$ a diagonal
    of random signs, the result zero-padded to $m_2 = 2^{\lceil \log_2 m
    \rceil}$ rows, $H/\sqrt{m_2}$ the orthonormal Walsh-Hadamard transform
    (`hadamard_transform`) and $R$ selects $d$ distinct rows uniformly at
    random. $HD$ spreads each vector's mass evenly over the coordinates
    (flattens the leverage), so uniform row sampling afterwards is safe;
    $\mathbb{E}[S^\top S] = I_m$. A size $d = O((n + \log m)\log n /
    \varepsilon^2)$ suffices for an $\varepsilon$-subspace embedding (Tropp,
    2011), and applying $S$ costs $O(m n \log m)$.

    The padding to a power of two costs up to 2× the memory of the input
    while the transform runs.

    Attributes:
        permutation: Input row read by each position, $(Px)_i =
            x_{\text{permutation}_i}$, shape ``(m,)``.
        signs: Diagonal of $D$, $\pm 1$, shape ``(m,)``.
        rows: The $d$ distinct rows of the padded transform kept by $R$,
            shape ``(d,)``.
        in_size: $m$.
        out_size: $d$.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> import gaussx as gx
        >>> S = gx.SRHTSketch.sample(jr.key(0), d=16, m=100)
        >>> S.apply(jnp.ones((100, 3))).shape
        (16, 3)
    """

    permutation: Int[Array, " m"]
    signs: Float[Array, " m"]
    rows: Int[Array, " d"]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    @classmethod
    def sample(cls, key: jax.Array | None, d: int, m: int) -> SRHTSketch:
        """Draw a ``(d, m)`` SRHT sketch.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            d: Sketch size (rows of $S$); at most the padded size $m_2$.
            m: Input size (columns of $S$).

        Returns:
            The sampled `SRHTSketch`.

        Raises:
            ValueError: If ``d`` exceeds the padded size $m_2$.
        """
        m2 = _padded_size(m)
        if d > m2:
            raise ValueError(f"SRHTSketch needs d <= {m2} (m={m} padded), got {d}.")
        if key is None:
            key = jax.random.PRNGKey(0)
        perm_key, sign_key, row_key = jax.random.split(key, 3)
        return cls(
            permutation=jax.random.permutation(perm_key, m),
            signs=jax.random.rademacher(sign_key, (m,), dtype=float),
            rows=jax.random.choice(row_key, m2, (d,), replace=False),
            in_size=m,
            out_size=d,
        )

    # Jitted: eagerly, the gather, padding, transform and row selection would
    # each dispatch (and compile) as separate programs.
    @eqx.filter_jit
    def apply(self, A: Float[Array, "m ..."]) -> Float[Array, "d ..."]:
        x = einsum(self.signs.astype(A.dtype), A[self.permutation], "m, m ... -> m ...")
        pad = jnp.zeros(
            (_padded_size(self.in_size) - self.in_size, *A.shape[1:]), dtype=A.dtype
        )
        x = jnp.concatenate([x, pad])
        y = hadamard_transform(rearrange(x, "m ... -> ... m"))[..., self.rows]
        # √(m₂/d) · H/√m₂ = H/√d with the unnormalised transform.
        return rearrange(y, "... d -> d ...") / math.sqrt(self.out_size)

    @eqx.filter_jit
    def apply_transpose(self, Y: Float[Array, "d ..."]) -> Float[Array, "m ..."]:
        m2 = _padded_size(self.in_size)
        z = jnp.zeros((m2, *Y.shape[1:]), dtype=Y.dtype).at[self.rows].set(Y)
        x = rearrange(
            hadamard_transform(rearrange(z, "m ... -> ... m")), "... m -> m ..."
        )
        x = einsum(self.signs.astype(Y.dtype), x[: self.in_size], "m, m ... -> m ...")
        out = jnp.zeros_like(x).at[self.permutation].set(x)
        return out / math.sqrt(self.out_size)


def _padded_size(m: int) -> int:
    """Smallest power of two ``>= m``."""
    return 1 << max(m - 1, 0).bit_length()
