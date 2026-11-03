"""Kronecker product A = A_1 ⊗ … ⊗ A_D as a lineax operator.

A_d is the d-th factor, M_d × N_d, d = 1, …, D, so A is M × N with
M = ∏_d M_d and N = ∏_d N_d. vec(X) flattens a tensor X row by row (the
row-major layout of every gaussx operator), so for two factors
(A_1 ⊗ A_2) vec(X) = vec(A_1 X A_2ᵀ) with X ∈ ℝ^{N_1 × N_2}.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._operators._block_diag import _resolve_dtype, _to_frozenset
from gaussx._tags import kronecker_tag


class Kronecker(lx.AbstractLinearOperator):
    """Kronecker product A = A_1 ⊗ … ⊗ A_D, applied one factor at a time.

        (A_1 ⊗ A_2) vec(X) = vec(A_1 X A_2ᵀ)

    The matvec never forms A: it applies each factor A_d along its own axis
    of the unflattened input (Roth's column lemma), so it costs
    O(N ∑_d M_d) for square factors instead of the O(N²) of a dense A.

    Args:
        *operators: The factors A_1, …, A_D, two or more lineax operators in
            Kronecker order.
        tags: Extra lineax tags for A; ``kronecker_tag`` is always added.

    Raises:
        ValueError: If fewer than two factors are given.

    Examples:
        A 4 × 4 product stored as two 2 × 2 factors:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> B = lx.MatrixLinearOperator(jnp.array([[3.0, 0.5], [0.5, 1.0]]), psd)
        >>> K = gaussx.Kronecker(A, B)
        >>> K.in_size()
        4
        >>> dense = jnp.kron(A.as_matrix(), B.as_matrix())
        >>> bool(jnp.allclose(K.as_matrix(), dense))
        True

        The matvec matches the dense product:

        >>> v = jnp.array([1.0, -1.0, 0.5, 2.0])
        >>> bool(jnp.allclose(K.mv(v), dense @ v, atol=1e-5))
        True

        Primitives keep the structure:

        >>> type(gaussx.cholesky(K)).__name__
        'Kronecker'
    """

    operators: tuple[lx.AbstractLinearOperator, ...]
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        *operators: lx.AbstractLinearOperator,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if len(operators) < 2:
            raise ValueError("Kronecker requires at least two operators.")
        self.operators = operators
        # N = ∏_d N_d
        self._in_size = _prod(op.in_size() for op in operators)
        # M = ∏_d M_d
        self._out_size = _prod(op.out_size() for op in operators)
        self._dtype = _resolve_dtype(*operators)

        self.tags = _to_frozenset(tags) | {kronecker_tag}

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        """Apply A to a vector, one factor at a time.

            (A_1 ⊗ A_2) vec(X) = vec(A_1 X A_2ᵀ)

        Args:
            vector: Input vec(X), of length N.

        Returns:
            A vec(X), of length M.
        """
        return _kronecker_mv(self.operators, vector)

    def as_matrix(self) -> Float[Array, "m n"]:
        """Dense A = A_1 ⊗ … ⊗ A_D, M × N.

        Returns:
            The materialised product.
        """
        matrix = self.operators[0].as_matrix()
        for factor in self.operators[1:]:
            # A_1 ⊗ … ⊗ A_d = (A_1 ⊗ … ⊗ A_d−1) ⊗ A_d
            matrix = jnp.kron(matrix, factor.as_matrix())
        return matrix

    def transpose(self) -> Kronecker:
        """Aᵀ = A_1ᵀ ⊗ … ⊗ A_Dᵀ, still a `Kronecker`.

        Returns:
            The transposed product, with transposed tags.
        """
        return Kronecker(
            *(op.T for op in self.operators),
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (M,) and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))


def _kronecker_mv(
    operators: tuple[lx.AbstractLinearOperator, ...],
    vector: Float[Array, " n"],
) -> Float[Array, " m"]:
    """Kronecker matvec by Roth's column lemma, one factor at a time.

        (A_1 ⊗ … ⊗ A_D) vec(X) = vec(X ×_1 A_1 ×_2 … ×_D A_D)

    Algorithm:
        1. For d = D, …, 1, unfold the vector so its trailing axis is the
           N_d axis of the current factor.
        2. Apply A_d to every fibre along that axis.
        3. Flatten with the new M_d axis moved to the front, so that after
           all D factors the axes are back in Kronecker order.

    Args:
        operators: The factors A_1, …, A_D.
        vector: Input vec(X), of length N = ∏_d N_d.

    Returns:
        A vec(X), of length M = ∏_d M_d.
    """
    values = vector
    for index in range(len(operators) - 1, -1, -1):
        factor = operators[index]
        # X_(d) ∈ ℝ^{rest × N_d}, the trailing axis is the one A_d acts on
        unfolded = rearrange(
            values, "(rest fibre) -> rest fibre", fibre=factor.in_size()
        )
        # X_(d) A_dᵀ: A_d on each fibre, lazily, so nested operators keep
        # their structure
        applied = jax.vmap(factor.mv)(unfolded)
        # move the M_d axis to the front for the next factor
        values = rearrange(applied, "rest fibre -> (fibre rest)")
    return values


def _prod(iterable) -> int:
    """Product ∏_d n_d of an iterable of integer sizes."""
    result = 1
    for size in iterable:
        result *= size
    return result
