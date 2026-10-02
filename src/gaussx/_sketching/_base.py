"""Abstract base class for sketching operators."""

from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


class AbstractSketch(eqx.Module):
    r"""A random sketching matrix $S \in \mathbb{R}^{d \times m}$, sampled once.

    A sketch compresses a tall matrix $A \in \mathbb{R}^{m \times n}$ to
    $SA \in \mathbb{R}^{d \times n}$ with $d \ll m$ while approximately
    preserving the geometry of $\operatorname{range}(A)$. $S$ is an
    $\varepsilon$-subspace embedding for $\operatorname{range}(A)$ if

    $$
    (1-\varepsilon)\|Ax\| \le \|SAx\| \le (1+\varepsilon)\|Ax\|
    \qquad \forall x .
    $$

    The random draws live in the module, so `apply` and `apply_transpose`
    always refer to the same $S$. Concrete sketches are built with their
    `sample` classmethod, which takes a PRNG key (``key=None`` means
    ``jax.random.PRNGKey(0)``).

    Sketches cast their stored values to the dtype of the array they are
    applied to, so a float32 input never meets a float64 sketch.

    Attributes:
        in_size: Number of columns $m$ of $S$ (rows of the sketched input).
        out_size: Number of rows $d$ of $S$ (the sketch size).
    """

    in_size: eqx.AbstractVar[int]
    out_size: eqx.AbstractVar[int]

    @abc.abstractmethod
    def apply(self, A: Float[Array, "m ..."]) -> Float[Array, "d ..."]:
        """Compute $S A$ along the leading axis of ``A``."""

    @abc.abstractmethod
    def apply_transpose(self, Y: Float[Array, "d ..."]) -> Float[Array, "m ..."]:
        r"""Compute $S^\top Y$ along the leading axis of ``Y``."""

    def sketch_operator(self, op: lx.AbstractLinearOperator) -> Float[Array, "d n"]:
        r"""Sketch a (possibly matrix-free) operator: $S A$.

        A `lineax.MatrixLinearOperator` is sketched directly with `apply`.
        Any other operator is sketched with $d$ transpose-matvecs,
        $SA = (A^\top S^\top)^\top$, vmapped over the rows of $S$; this
        materialises $S^\top$ as an $(m, d)$ block but never forms $A$.

        Args:
            op: Operator $A$ of shape ``(m, n)``.

        Returns:
            The dense sketch $SA$, shape ``(d, n)``.

        Raises:
            ValueError: If ``op.out_size()`` is not the sketch's ``in_size``.
        """
        if op.out_size() != self.in_size:
            raise ValueError(
                f"Cannot sketch an operator with {op.out_size()} rows using a "
                f"sketch with in_size={self.in_size}."
            )
        if isinstance(op, lx.MatrixLinearOperator):
            return self.apply(op.matrix)
        dtype = op.out_structure().dtype
        rows_of_s = self.apply_transpose(jnp.eye(self.out_size, dtype=dtype))
        return jax.vmap(op.transpose().mv, in_axes=1)(rows_of_s)

    def as_operator(self) -> lx.AbstractLinearOperator:
        """Return $S$ as a matrix-free ``(d, m)`` lineax operator."""
        dtype = jax.tree.leaves(eqx.filter(self, eqx.is_inexact_array))[0].dtype
        return lx.FunctionLinearOperator(
            self.apply, jax.ShapeDtypeStruct((self.in_size,), dtype)
        )
