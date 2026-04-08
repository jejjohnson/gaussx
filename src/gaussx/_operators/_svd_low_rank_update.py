"""SVD-parameterized low-rank update: L + U diag(S) Vᵀ with orthonormal U, V."""

from __future__ import annotations

import lineax as lx
from jaxtyping import Array, Float

from gaussx._operators._block_diag import _to_frozenset
from gaussx._operators._low_rank_update import (
    LowRankUpdate,
    _is_nonnegative,
    _safe_query,
)


class SVDLowRankUpdate(LowRankUpdate):
    """SVD-parameterized low-rank update ``L + U diag(S) Vᵀ``.

    Like :class:`LowRankUpdate` but assumes orthonormality of *U* and
    *V* for cheaper solves and log-determinants. Typical sources are
    truncated SVD (Nyström approximation) or ensemble methods.

    Inherits ``mv``, ``as_matrix``, ``rank``, ``in_structure``, and
    ``out_structure`` from :class:`LowRankUpdate`. The singular values
    are stored in the inherited ``d`` field.

    Args:
        base: The base operator *L*, with shape ``(n, n)``.
        U: Left singular vectors, shape ``(n, k)`` with orthonormal columns.
        S: Singular values, shape ``(k,)``.
        V: Right singular vectors, shape ``(n, k)`` with orthonormal columns.
            Defaults to *U* for symmetric updates.
    """

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
        self.d = S
        self.V = V
        from gaussx._tags import low_rank_tag

        # Stricter tag inference: identity check for orthonormal U, V
        inferred: set[object] = set()
        if n_in == n_out and _safe_query(lx.is_symmetric, base) and U is V:
            inferred.add(lx.symmetric_tag)
            if _safe_query(lx.is_positive_semidefinite, base) and _is_nonnegative(S):
                inferred.add(lx.positive_semidefinite_tag)
        self.tags = _to_frozenset(tags) | frozenset(inferred) | {low_rank_tag}

    def transpose(self) -> SVDLowRankUpdate:
        return SVDLowRankUpdate(
            self.base.T,
            self.V,
            self.d,
            self.U,
            tags=lx.transpose_tags(self.tags),
        )
