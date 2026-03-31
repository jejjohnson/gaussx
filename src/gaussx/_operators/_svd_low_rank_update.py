"""SVD-parameterized low-rank update: L + U diag(S) Vᵀ with orthonormal U, V."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _to_frozenset
from gaussx._operators._low_rank_update import _safe_query


class SVDLowRankUpdate(lx.AbstractLinearOperator):
    """SVD-parameterized low-rank update ``L + U diag(S) Vᵀ``.

    Like :class:`LowRankUpdate` but exploits the orthonormality of *U* and
    *V* for cheaper solves and log-determinants. Typical sources are
    truncated SVD (Nystrom approximation) or ensemble methods.

    Args:
        base: The base operator *L*, with shape ``(n, n)``.
        U: Left singular vectors, shape ``(n, k)`` with orthonormal columns.
        S: Singular values, shape ``(k,)``.
        V: Right singular vectors, shape ``(n, k)`` with orthonormal columns.
            Defaults to *U* for symmetric updates.
    """

    base: lx.AbstractLinearOperator
    U: Float[Array, "n k"]
    S: Float[Array, " k"]
    V: Float[Array, "n k"]
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        base: lx.AbstractLinearOperator,
        U: Float[Array, "n k"],
        S: Float[Array, " k"],
        V: Float[Array, "n k"] | None = None,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        n_out = base.out_size()
        n_in = base.in_size()
        if U.ndim == 1:
            U = U[:, None]
        if V is None:
            V = U
        if V.ndim == 1:
            V = V[:, None]
        k = S.shape[0]
        if U.shape != (n_out, k):
            raise ValueError(f"U must have shape ({n_out}, {k}), got {U.shape}.")
        if V.shape != (n_in, k):
            raise ValueError(f"V must have shape ({n_in}, {k}), got {V.shape}.")
        self.base = base
        self.U = U
        self.S = S
        self.V = V
        from gaussx._tags import low_rank_tag

        inferred: set[object] = set()
        if n_in == n_out and _safe_query(lx.is_symmetric, base) and U is V:
            inferred.add(lx.symmetric_tag)
            if _safe_query(lx.is_positive_semidefinite, base):
                inferred.add(lx.positive_semidefinite_tag)
        self.tags = _to_frozenset(tags) | frozenset(inferred) | {low_rank_tag}

    @property
    def rank(self) -> int:
        """Rank of the low-rank update."""
        return self.S.shape[0]

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        # (L + U diag(S) V^T) x = L x + U (S * (V^T x))
        base_part = self.base.mv(vector)
        vtx = self.V.T @ vector  # (k,)
        scaled = self.S * vtx  # (k,)
        update_part = self.U @ scaled  # (n,)
        return base_part + update_part

    def as_matrix(self) -> Float[Array, "n n"]:
        L = self.base.as_matrix()
        return L + self.U @ jnp.diag(self.S) @ self.V.T

    def transpose(self) -> SVDLowRankUpdate:
        return SVDLowRankUpdate(
            self.base.T,
            self.V,
            self.S,
            self.U,
            tags=lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self.base.in_structure()

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.base.out_structure()
