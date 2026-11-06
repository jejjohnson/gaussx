"""Dense sketches S ∈ ℝ^(d × M): the Gaussian and the orthonormal-row Gaussian one."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._sketching._base import AbstractSketch


class _DenseSketch(AbstractSketch):
    """A sketch that stores its d × M matrix S explicitly.

    Attributes:
        matrix: The sketching matrix S, d × M.
        in_size: The input size M.
        out_size: The sketch size d.
    """

    matrix: Float[Array, "d m"]
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def apply(self, A: Float[Array, "m ..."]) -> Float[Array, "d ..."]:
        """Compute S A along the leading axis of ``A``.

        Args:
            A: Array A whose leading axis has length M.

        Returns:
            The product S A, whose leading axis has length d.
        """
        # in the input's dtype, so a float32 input never meets a float64 sketch
        matrix = self.matrix.astype(A.dtype)
        # S A
        return einsum(matrix, A, "d m, m ... -> d ...")

    def apply_transpose(self, Y: Float[Array, "d ..."]) -> Float[Array, "m ..."]:
        """Compute Sᵀ Y along the leading axis of ``Y``.

        Args:
            Y: Array Y whose leading axis has length d.

        Returns:
            The product Sᵀ Y, whose leading axis has length M.
        """
        # in the input's dtype, so a float32 input never meets a float64 sketch
        matrix = self.matrix.astype(Y.dtype)
        # Sᵀ Y
        return einsum(matrix, Y, "d m, d ... -> m ...")

    def as_operator(self) -> lx.AbstractLinearOperator:
        """Return S as a dense d × M `lineax.MatrixLinearOperator`.

        Returns:
            The operator S, on the stored matrix.
        """
        return lx.MatrixLinearOperator(self.matrix)


class GaussianSketch(_DenseSketch):
    """Gaussian sketch: i.i.d. entries S_ij ~ 𝒩(0, 1/d).

        𝔼[Sᵀ S] = I_M

    A Gaussian sketch of size d = O(N / ε²) is an ε-subspace embedding for any
    N-dimensional subspace; applying it to an M × N matrix costs O(d M N).

    Attributes:
        matrix: The sketching matrix S, d × M.
        in_size: The input size M.
        out_size: The sketch size d.

    Examples:
        Compress 500 rows to 20:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import gaussx
        >>> sketch = gaussx.GaussianSketch.sample(jr.key(0), d=20, m=500)
        >>> sketch.apply(jnp.ones((500, 3))).shape
        (20, 3)

        A vector is sketched in the same way, and comes back through the transpose:

        >>> sketched = sketch.apply(jnp.ones(500))
        >>> sketched.shape
        (20,)
        >>> sketch.apply_transpose(sketched).shape
        (500,)

        The dtype of the input wins over the dtype of the draws:

        >>> sketch.apply(jnp.ones(500, dtype=jnp.float32)).dtype.name
        'float32'
    """

    @classmethod
    def sample(cls, key: jax.Array | None, d: int, m: int) -> GaussianSketch:
        """Draw a d × M Gaussian sketch.

            S = G / √d,   G_ij ~ 𝒩(0, 1)

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            d: Sketch size d (rows of S).
            m: Input size M (columns of S).

        Returns:
            The sampled `GaussianSketch`.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        # G_ij ~ 𝒩(0, 1)
        gaussian_draw = jax.random.normal(key, (d, m))
        # √d
        scale = jnp.sqrt(d)
        # S = G / √d
        matrix = gaussian_draw / scale
        return cls(matrix=matrix, in_size=m, out_size=d)


class OrthonormalSketch(_DenseSketch):
    """Gaussian sketch with orthonormal rows, S Sᵀ = I_d.

        S = Qᵀ,   G = Q R,   G_ij ~ 𝒩(0, 1),   G ∈ ℝ^(M × d)

    Built from the thin QR of an M × d Gaussian block, so the row space of S is a
    uniformly random d-dimensional subspace of ℝ^M. Rows are orthonormal, so
    𝔼[Sᵀ S] = (d / M) I_M: rescale by √(M / d) when an isotropic embedding is needed.
    Requires d ≤ M.

    Attributes:
        matrix: The sketching matrix S with orthonormal rows, d × M.
        in_size: The input size M.
        out_size: The sketch size d.

    Examples:
        The rows are orthonormal:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import gaussx
        >>> sketch = gaussx.OrthonormalSketch.sample(jr.key(0), d=4, m=50)
        >>> gram = sketch.apply(sketch.apply_transpose(jnp.eye(4)))
        >>> bool(jnp.allclose(gram, jnp.eye(4), atol=1e-5))
        True

        With d = M the sketch is an orthogonal matrix, which preserves norms:

        >>> square = gaussx.OrthonormalSketch.sample(jr.key(0), d=5, m=5)
        >>> x = jnp.arange(5.0)
        >>> norm_ratio = jnp.linalg.norm(square.apply(x)) / jnp.linalg.norm(x)
        >>> bool(jnp.allclose(norm_ratio, 1.0, atol=1e-5))
        True
    """

    @classmethod
    def sample(cls, key: jax.Array | None, d: int, m: int) -> OrthonormalSketch:
        """Draw a d × M sketch with orthonormal rows.

            S = Qᵀ,   G = Q R

        Algorithm:
            1. Draw the M × d Gaussian block G.
            2. Factor G = Q R (thin QR).
            3. Take S = Qᵀ.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            d: Sketch size d (rows of S); must satisfy d ≤ M.
            m: Input size M (columns of S).

        Returns:
            The sampled `OrthonormalSketch`.

        Raises:
            ValueError: If d > M.

        Examples:
            >>> import jax.random as jr
            >>> import gaussx
            >>> gaussx.OrthonormalSketch.sample(jr.key(0), d=2, m=6).matrix.shape
            (2, 6)

            More rows than columns cannot be orthonormal:

            >>> gaussx.OrthonormalSketch.sample(jr.key(0), d=5, m=4)
            Traceback (most recent call last):
            ...
            ValueError: OrthonormalSketch needs d <= m, got d=5, m=4.
        """
        if d > m:
            raise ValueError(f"OrthonormalSketch needs d <= m, got d={d}, m={m}.")
        if key is None:
            key = jax.random.PRNGKey(0)
        # G_ij ~ 𝒩(0, 1)
        gaussian_draw = jax.random.normal(key, (m, d))
        # G = Q R
        orthogonal, _ = jnp.linalg.qr(gaussian_draw)
        # S = Qᵀ
        matrix = rearrange(orthogonal, "m d -> d m")
        return cls(matrix=matrix, in_size=m, out_size=d)
