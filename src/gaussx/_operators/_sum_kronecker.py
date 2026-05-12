"""Sum of Kronecker products: A1 kron B1 + A2 kron B2 + ..."""

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
    r"""Sum of Kronecker products ``Σ_k A_k \otimes B_k``.

    Appears in multi-output GPs with correlated outputs, e.g.
    ``K_task \otimes K_spatial + \sigma^2 I_task \otimes I_spatial``.

    Matvec is computed as the sum of the Kronecker matvecs.

    For solve and logdet, call :meth:`eigendecompose` which uses a
    joint eigendecomposition of the second Kronecker pair (requires
    ``A_2, B_2`` to be symmetric).  The eigendecomposition forms a
    dense ``(n_c n_d) x (n_c n_d)`` matrix internally, so it is
    intended for moderate factor sizes (typical for multi-output GPs
    where the task dimension is small).

    Args:
        kron1: First Kronecker product ``A_1 \otimes B_1``.
        kron2: Second Kronecker product ``A_2 \otimes B_2``.
        *krons: Additional two-factor Kronecker products.
    """

    operators: tuple[Kronecker, ...]
    _in_size: int = eqx.field(static=True)
    _out_size: int = eqx.field(static=True)
    _dtype: str = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        kron1: Kronecker,
        kron2: Kronecker,
        *krons: Kronecker,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        operators = (kron1, kron2, *krons)
        if any(len(kron.operators) != 2 for kron in operators):
            raise ValueError("SumKronecker requires two-factor Kronecker products.")
        if any(kron.in_size() != kron1.in_size() for kron in operators[1:]):
            raise ValueError("Kronecker products must have the same size (input size).")
        if any(kron.out_size() != kron1.out_size() for kron in operators[1:]):
            raise ValueError(
                "Kronecker products must have the same size (output size)."
            )
        self.operators = operators
        self._in_size = kron1.in_size()
        self._out_size = kron1.out_size()
        self._dtype = _resolve_dtype(*operators)
        self.tags = _to_frozenset(tags)

    @property
    def kron1(self) -> Kronecker:
        return self.operators[0]

    @property
    def kron2(self) -> Kronecker:
        return self.operators[1]

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        result = self.operators[0].mv(vector)
        for kron in self.operators[1:]:
            result = result + kron.mv(vector)
        return result

    def as_matrix(self) -> Float[Array, "n n"]:
        result = self.operators[0].as_matrix()
        for kron in self.operators[1:]:
            result = result + kron.as_matrix()
        return result

    def transpose(self) -> SumKronecker:
        return SumKronecker(
            *(
                Kronecker(
                    kron.operators[0].T,
                    kron.operators[1].T,
                    tags=lx.transpose_tags(kron.tags),
                )
                for kron in self.operators
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
        if len(self.operators) != 2:
            count = len(self.operators)
            raise ValueError(
                f"eigendecompose requires exactly two Kronecker products, got {count}."
            )
        A2_op, B2_op = self.kron2.operators
        A1_op, B1_op = self.kron1.operators
        # The final ``eigh(transformed)`` call requires ``transformed`` to
        # be symmetric, which in turn requires *both* kron pairs to have
        # symmetric factors (so that ``A1_tilde`` and ``B1_tilde`` stay
        # symmetric under the ``Q^T A Q`` rotation).
        if not lx.is_symmetric(A2_op) or not lx.is_symmetric(B2_op):
            raise ValueError("eigendecompose requires kron2 factors to be symmetric.")
        if not lx.is_symmetric(A1_op) or not lx.is_symmetric(B1_op):
            raise ValueError("eigendecompose requires kron1 factors to be symmetric.")

        from gaussx._primitives._eig import eig

        A1, B1 = (op.as_matrix() for op in self.kron1.operators)

        # Per-factor eigendecomposition routed through the structural
        # primitive: Diagonal / BlockDiag / nested Kronecker factors of
        # ``A_2`` or ``B_2`` skip materialization here.
        evals_c, Q_C = eig(A2_op)
        evals_d, Q_D = eig(B2_op)

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


def sumkronecker_sample(
    op: SumKronecker,
    *,
    key: jax.Array,
    num_samples: int = 1,
    lanczos_order: int = 50,
) -> Float[Array, "num_samples n"]:
    r"""Sample from ``𝒩(0, op)`` with matrix-free Lanczos square roots.

    The square-root action is evaluated by ``matfree`` Lanczos against
    ``op.mv``. This avoids materialising the dense ``(n_A n_B) ×
    (n_A n_B)`` covariance and costs ``lanczos_order`` SumKronecker
    matvecs per sample.

    Args:
        op: Positive-semidefinite SumKronecker covariance operator.
        key: JAX PRNG key.
        num_samples: Number of independent samples to draw.
        lanczos_order: Lanczos truncation order.

    Returns:
        Samples with shape ``(num_samples, op.in_size())``.
    """
    from gaussx._primitives._sqrt import sqrt

    if op.in_size() != op.out_size():
        raise ValueError(
            "sumkronecker_sample requires a square SumKronecker, got "
            f"in_size={op.in_size()} and out_size={op.out_size()}."
        )
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    sqrt_op = sqrt(op, lanczos_order=lanczos_order)
    eps = jax.random.normal(
        key, (num_samples, op.in_size()), dtype=op.in_structure().dtype
    )
    return jax.vmap(sqrt_op.mv)(eps)
