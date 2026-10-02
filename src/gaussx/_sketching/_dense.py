"""Dense sketches: Gaussian and orthonormal-row Gaussian."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._sketching._base import AbstractSketch


class _DenseSketch(AbstractSketch):
    """A sketch that stores its ``(d, m)`` matrix explicitly."""

    matrix: Float[Array, "d m"]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def apply(self, A: Float[Array, "m ..."]) -> Float[Array, "d ..."]:
        return einsum(self.matrix.astype(A.dtype), A, "d m, m ... -> d ...")

    def apply_transpose(self, Y: Float[Array, "d ..."]) -> Float[Array, "m ..."]:
        return einsum(self.matrix.astype(Y.dtype), Y, "d m, d ... -> m ...")

    def as_operator(self) -> lx.AbstractLinearOperator:
        """Return $S$ as a dense ``(d, m)`` `lineax.MatrixLinearOperator`."""
        return lx.MatrixLinearOperator(self.matrix)


class GaussianSketch(_DenseSketch):
    r"""Gaussian sketch: i.i.d. entries $S_{ij} \sim \mathcal{N}(0, 1/d)$.

    The scaling gives $\mathbb{E}[S^\top S] = I_m$. A Gaussian sketch of size
    $d = O(n/\varepsilon^2)$ is an $\varepsilon$-subspace embedding for any
    $n$-dimensional subspace; applying it costs $O(dmn)$.

    Attributes:
        matrix: The sketching matrix, shape ``(d, m)``.
        in_size: $m$.
        out_size: $d$.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> import gaussx as gx
        >>> S = gx.GaussianSketch.sample(jr.key(0), d=20, m=500)
        >>> S.apply(jnp.ones((500, 3))).shape
        (20, 3)
    """

    @classmethod
    def sample(cls, key: jax.Array | None, d: int, m: int) -> GaussianSketch:
        """Draw a ``(d, m)`` Gaussian sketch.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            d: Sketch size (rows of $S$).
            m: Input size (columns of $S$).

        Returns:
            The sampled `GaussianSketch`.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        matrix = jax.random.normal(key, (d, m)) / jnp.sqrt(d)
        return cls(matrix=matrix, in_size=m, out_size=d)


class OrthonormalSketch(_DenseSketch):
    r"""Gaussian sketch with orthonormal rows, $S S^\top = I_d$.

    Built from the thin QR of an $(m, d)$ Gaussian block, so the row space of
    $S$ is a uniformly random $d$-dimensional subspace of $\mathbb{R}^m$.
    Rows are orthonormal, so $\mathbb{E}[S^\top S] = (d/m)\, I_m$: rescale by
    $\sqrt{m/d}$ when an isotropic embedding is needed. Requires $d \le m$.

    Attributes:
        matrix: The sketching matrix with orthonormal rows, shape ``(d, m)``.
        in_size: $m$.
        out_size: $d$.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> import gaussx as gx
        >>> S = gx.OrthonormalSketch.sample(jr.key(0), d=4, m=50)
        >>> SSt = S.apply(S.apply_transpose(jnp.eye(4)))
        >>> bool(jnp.allclose(SSt, jnp.eye(4), atol=1e-5))
        True
    """

    @classmethod
    def sample(cls, key: jax.Array | None, d: int, m: int) -> OrthonormalSketch:
        """Draw a ``(d, m)`` sketch with orthonormal rows.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            d: Sketch size (rows of $S$); must satisfy ``d <= m``.
            m: Input size (columns of $S$).

        Returns:
            The sampled `OrthonormalSketch`.

        Raises:
            ValueError: If ``d > m``.
        """
        if d > m:
            raise ValueError(f"OrthonormalSketch needs d <= m, got d={d}, m={m}.")
        if key is None:
            key = jax.random.PRNGKey(0)
        q, _ = jnp.linalg.qr(jax.random.normal(key, (m, d)))
        return cls(matrix=rearrange(q, "m d -> d m"), in_size=m, out_size=d)
