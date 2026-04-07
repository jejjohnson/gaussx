"""Sum of two Kronecker products: A1 kron B1 + A2 kron B2."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from einops import rearrange
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _resolve_dtype, _to_frozenset
from gaussx._operators._kronecker import Kronecker


class SumKronecker(lx.AbstractLinearOperator):
    r"""Sum of two Kronecker products ``A_1 \otimes B_1 + A_2 \otimes B_2``.

    Appears in multi-output GPs with correlated outputs, e.g.
    ``K_task \otimes K_spatial + \sigma^2 I_task \otimes I_spatial``.

    Matvec is computed as the sum of the two Kronecker matvecs.

    For solve and logdet, call :meth:`eigendecompose` which uses a
    joint eigendecomposition of the second Kronecker pair (requires
    ``A_2, B_2`` to be symmetric).  The eigendecomposition forms a
    dense ``(n_c n_d) x (n_c n_d)`` matrix internally, so it is
    intended for moderate factor sizes (typical for multi-output GPs
    where the task dimension is small).

    Args:
        kron1: First Kronecker product ``A_1 \otimes B_1``.
        kron2: Second Kronecker product ``A_2 \otimes B_2``.
    """

    kron1: Kronecker
    kron2: Kronecker
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        kron1: Kronecker,
        kron2: Kronecker,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        if len(kron1.operators) != 2 or len(kron2.operators) != 2:
            raise ValueError("SumKronecker requires two-factor Kronecker products.")
        if kron1.in_size() != kron2.in_size():
            raise ValueError(
                f"Kronecker products must have the same size: "
                f"{kron1.in_size()} != {kron2.in_size()}."
            )
        self.kron1 = kron1
        self.kron2 = kron2
        self._in_size = kron1.in_size()
        self._out_size = kron1.out_size()
        self._dtype = _resolve_dtype(kron1, kron2)
        self.tags = _to_frozenset(tags)

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        return self.kron1.mv(vector) + self.kron2.mv(vector)

    def as_matrix(self) -> Float[Array, "n n"]:
        return self.kron1.as_matrix() + self.kron2.as_matrix()

    def transpose(self) -> SumKronecker:
        return SumKronecker(
            Kronecker(
                self.kron1.operators[0].T,
                self.kron1.operators[1].T,
                tags=lx.transpose_tags(self.kron1.tags),
            ),
            Kronecker(
                self.kron2.operators[0].T,
                self.kron2.operators[1].T,
                tags=lx.transpose_tags(self.kron2.tags),
            ),
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))

    def eigendecompose(
        self,
    ) -> tuple[Float[Array, " n"], Float[Array, "n n"]]:
        r"""Eigendecompose via joint eigendecomposition of the second pair.

        Decomposes ``A_2 = Q_C \Lambda_C Q_C^T`` and
        ``B_2 = Q_D \Lambda_D Q_D^T``, then transforms the first pair
        into the eigenbasis and diagonalizes the result.

        .. note::

            This forms a dense ``(n_c n_d) x (n_c n_d)`` matrix
            internally and is O((n_c n_d)^3).  Intended for moderate
            factor sizes (e.g. multi-output GPs where task dimension
            is small).

        Raises:
            ValueError: If the factors of ``kron2`` are not symmetric.

        Returns:
            Tuple ``(eigenvalues, Q)`` where
            ``self == Q @ diag(eigenvalues) @ Q^T``.
        """
        A2_op, B2_op = self.kron2.operators
        if not lx.is_symmetric(A2_op) or not lx.is_symmetric(B2_op):
            raise ValueError("eigendecompose requires kron2 factors to be symmetric.")

        A1, B1 = (op.as_matrix() for op in self.kron1.operators)
        A2, B2 = A2_op.as_matrix(), B2_op.as_matrix()

        evals_c, Q_C = jnp.linalg.eigh(A2)
        evals_d, Q_D = jnp.linalg.eigh(B2)

        # Transform first pair into eigenbasis of second pair
        A1_tilde = Q_C.T @ A1 @ Q_C
        B1_tilde = Q_D.T @ B1 @ Q_D

        # kron(A1_tilde, B1_tilde) + diag(evals_c kron evals_d)
        diag_vals = rearrange(
            evals_c[:, None] * evals_d[None, :],
            "a b -> (a b)",
        )
        transformed = jnp.kron(A1_tilde, B1_tilde) + jnp.diag(diag_vals)
        evals, V = jnp.linalg.eigh(transformed)

        Q = jnp.kron(Q_C, Q_D) @ V
        return evals, Q
