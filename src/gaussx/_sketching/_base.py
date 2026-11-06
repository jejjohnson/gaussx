"""Abstract base class of the random sketches of a tall M × N operator A.

A sketch S ∈ ℝ^(d × M) with d ≪ M rows compresses A to S A ∈ ℝ^(d × N). S is an
ε-subspace embedding for range(A) when
(1 − ε) ‖A x‖₂ ≤ ‖S A x‖₂ ≤ (1 + ε) ‖A x‖₂ for every x (ε is the distortion, the letter
of the sketching literature).
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


class AbstractSketch(eqx.Module):
    """A random sketching matrix S ∈ ℝ^(d × M), sampled once.

        (1 − ε) ‖A x‖₂ ≤ ‖S A x‖₂ ≤ (1 + ε) ‖A x‖₂   for every x

    A sketch compresses a tall matrix A ∈ ℝ^(M × N) to S A ∈ ℝ^(d × N) with d ≪ M while
    approximately preserving the geometry of range(A). S is an ε-subspace embedding for
    range(A) when the bound above holds.

    The random draws live in the module, so `apply` and `apply_transpose` always refer
    to the same S. Concrete sketches are built with their `sample` classmethod, which
    takes a PRNG key (``key=None`` means ``jax.random.PRNGKey(0)``).

    Sketches cast their stored values to the dtype of the array they are applied to, so
    a float32 input never meets a float64 sketch.

    Attributes:
        in_size: Number of columns M of S (rows of the sketched input).
        out_size: Number of rows d of S (the sketch size).

    Examples:
        Every concrete sketch has the same interface:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import gaussx
        >>> sketch = gaussx.GaussianSketch.sample(jr.key(0), d=4, m=8)
        >>> isinstance(sketch, gaussx.AbstractSketch)
        True
        >>> sketch.out_size, sketch.in_size
        (4, 8)

        `apply_transpose` is the adjoint of `apply`, ⟨S x, y⟩ = ⟨x, Sᵀ y⟩:

        >>> x, y = jnp.arange(8.0), jnp.arange(4.0)
        >>> left = jnp.dot(sketch.apply(x), y)
        >>> right = jnp.dot(x, sketch.apply_transpose(y))
        >>> bool(jnp.allclose(left, right, rtol=1e-4))
        True

        `sketch_operator` rejects an operator with the wrong number of rows:

        >>> import lineax as lx
        >>> wrong = lx.MatrixLinearOperator(jnp.ones((6, 3)))
        >>> sketch.sketch_operator(wrong)
        Traceback (most recent call last):
        ...
        ValueError: Cannot sketch an operator with 6 rows using a sketch with in_size=8.
    """

    in_size: eqx.AbstractVar[int]
    out_size: eqx.AbstractVar[int]

    @abc.abstractmethod
    def apply(self, A: Float[Array, "m ..."]) -> Float[Array, "d ..."]:
        """Compute S A along the leading axis of ``A``.

        Args:
            A: Array A whose leading axis has length M.

        Returns:
            The product S A, whose leading axis has length d.
        """

    @abc.abstractmethod
    def apply_transpose(self, Y: Float[Array, "d ..."]) -> Float[Array, "m ..."]:
        """Compute Sᵀ Y along the leading axis of ``Y``.

        Args:
            Y: Array Y whose leading axis has length d.

        Returns:
            The product Sᵀ Y, whose leading axis has length M.
        """

    def sketch_operator(self, op: lx.AbstractLinearOperator) -> Float[Array, "d n"]:
        """Sketch a (possibly matrix-free) operator A.

            S A = (Aᵀ Sᵀ)ᵀ

        A `lineax.MatrixLinearOperator` is sketched directly with `apply`. Any other
        operator is sketched with d transpose-matvecs, vmapped over the rows of S; this
        materialises Sᵀ as an M × d block but never forms A.

        Args:
            op: Operator A, M × N.

        Returns:
            The dense sketch S A, d × N.

        Raises:
            ValueError: If ``op.out_size()`` is not the sketch's ``in_size``.

        Examples:
            A stored operator and a matrix-free one give the same sketch:

            >>> import jax
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> import lineax as lx
            >>> import gaussx
            >>> matrix = jr.normal(jr.key(1), (8, 3))
            >>> sketch = gaussx.GaussianSketch.sample(jr.key(0), d=4, m=8)
            >>> stored = lx.MatrixLinearOperator(matrix)
            >>> structure = jax.ShapeDtypeStruct((3,), matrix.dtype)
            >>> free = lx.FunctionLinearOperator(lambda x: matrix @ x, structure)
            >>> sketched = sketch.sketch_operator(stored)
            >>> sketched.shape
            (4, 3)
            >>> bool(jnp.allclose(sketched, sketch.sketch_operator(free), rtol=1e-4))
            True
        """
        if op.out_size() != self.in_size:
            raise ValueError(
                f"Cannot sketch an operator with {op.out_size()} rows using a "
                f"sketch with in_size={self.in_size}."
            )
        if isinstance(op, lx.MatrixLinearOperator):
            # S A, on the stored matrix
            return self.apply(op.matrix)
        dtype = op.out_structure().dtype
        identity = jnp.eye(self.out_size, dtype=dtype)
        # Sᵀ = Sᵀ I_d, whose columns are the rows of S
        rows_of_sketch = self.apply_transpose(identity)
        transpose = op.transpose()
        # S A = (Aᵀ Sᵀ)ᵀ, one row of S A per column of Sᵀ
        return jax.vmap(transpose.mv, in_axes=1)(rows_of_sketch)

    def as_operator(self) -> lx.AbstractLinearOperator:
        """Return S as a matrix-free d × M lineax operator.

            x ↦ S x

        The operator's dtype is that of the first inexact array stored on the sketch.

        Returns:
            The operator S, whose matvec is `apply`.

        Examples:
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> import gaussx
            >>> sketch = gaussx.SRHTSketch.sample(jr.key(0), d=4, m=10)
            >>> operator = sketch.as_operator()
            >>> operator.out_size(), operator.in_size()
            (4, 10)
            >>> x = jnp.arange(10.0)
            >>> bool(jnp.allclose(operator.mv(x), sketch.apply(x)))
            True
        """
        leaves = jax.tree.leaves(eqx.filter(self, eqx.is_inexact_array))
        dtype = leaves[0].dtype
        structure = jax.ShapeDtypeStruct((self.in_size,), dtype)
        return lx.FunctionLinearOperator(self.apply, structure)
