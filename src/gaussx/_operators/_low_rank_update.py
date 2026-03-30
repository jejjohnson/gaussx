"""Low-rank update linear operator: L + U diag(D) Vᵀ."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _to_frozenset


class LowRankUpdate(lx.AbstractLinearOperator):
    """Low-rank update operator ``L + U diag(d) Vᵀ``.

    Represents a base operator *L* plus a rank-k update. When *L*
    is cheap to solve (e.g. diagonal), the Woodbury identity gives
    efficient solves for the full operator.

    Args:
        base: The base operator *L*.
        U: Left factor, shape ``(n, k)``.
        d: Diagonal scaling, shape ``(k,)``. Defaults to ones.
        V: Right factor, shape ``(n, k)``. Defaults to *U*
            (symmetric update ``L + U diag(d) Uᵀ``).
    """

    base: lx.AbstractLinearOperator
    U: Float[Array, "n k"]
    d: Float[Array, " k"]
    V: Float[Array, "n k"]
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        base: lx.AbstractLinearOperator,
        U: Float[Array, "n k"],
        d: Float[Array, " k"] | None = None,
        V: Float[Array, "n k"] | None = None,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        n = base.in_size()
        k = U.shape[1] if U.ndim == 2 else 1
        if U.ndim == 1:
            U = U[:, None]
        if d is None:
            d = jnp.ones(k, dtype=U.dtype)
        if V is None:
            V = U
        if V.ndim == 1:
            V = V[:, None]
        if U.shape[0] != n or V.shape[0] != n:
            raise ValueError(
                f"U and V must have {n} rows to match base operator, "
                f"got U.shape={U.shape}, V.shape={V.shape}."
            )
        if U.shape[1] != d.shape[0] or V.shape[1] != d.shape[0]:
            raise ValueError(
                f"Rank dimensions must match: U has {U.shape[1]} cols, "
                f"V has {V.shape[1]} cols, d has {d.shape[0]} entries."
            )
        self.base = base
        self.U = U
        self.d = d
        self.V = V
        from gaussx._tags import low_rank_tag

        self.tags = _to_frozenset(tags) | {low_rank_tag}

    @property
    def rank(self) -> int:
        """Rank of the low-rank update."""
        return self.d.shape[0]

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        # (L + U diag(d) V^T) x = L x + U (d * (V^T x))
        base_part = self.base.mv(vector)
        vtx = self.V.T @ vector  # (k,)
        scaled = self.d * vtx  # (k,)
        update_part = self.U @ scaled  # (n,)
        return base_part + update_part

    def as_matrix(self) -> Float[Array, "n n"]:
        L = self.base.as_matrix()
        return L + self.U @ jnp.diag(self.d) @ self.V.T

    def transpose(self) -> LowRankUpdate:
        return LowRankUpdate(
            self.base.T,
            self.V,
            self.d,
            self.U,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self.base.in_structure()

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.base.out_structure()


def low_rank_plus_diag(
    diag: Float[Array, " n"],
    U: Float[Array, "n k"],
    d: Float[Array, " k"] | None = None,
    V: Float[Array, "n k"] | None = None,
) -> LowRankUpdate:
    """Construct ``diag(diag) + U diag(d) Vᵀ``.

    Common pattern for inducing-point / Nystrom approximations
    where the base is a diagonal matrix.

    Args:
        diag: Diagonal entries, shape ``(n,)``.
        U: Left factor, shape ``(n, k)``.
        d: Diagonal scaling, shape ``(k,)``. Defaults to ones.
        V: Right factor, shape ``(n, k)``. Defaults to *U*.

    Returns:
        A ``LowRankUpdate`` with a ``DiagonalLinearOperator`` base.
    """
    base = lx.DiagonalLinearOperator(diag)
    return LowRankUpdate(base, U, d, V)


def svd_low_rank_plus_diag(
    diag: Float[Array, " n"],
    U: Float[Array, "n k"],
    S: Float[Array, " k"],
    V: Float[Array, "n k"],
) -> LowRankUpdate:
    """Construct ``diag(diag) + U diag(S) Vᵀ`` from a truncated SVD.

    Args:
        diag: Diagonal entries, shape ``(n,)``.
        U: Left singular vectors, shape ``(n, k)``.
        S: Singular values, shape ``(k,)``.
        V: Right singular vectors, shape ``(n, k)``.

    Returns:
        A ``LowRankUpdate`` with a ``DiagonalLinearOperator`` base.
    """
    base = lx.DiagonalLinearOperator(diag)
    return LowRankUpdate(base, U, S, V)
