"""Low-rank update linear operator: L + U diag(D) Vᵀ."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._operators._block_diag import _to_frozenset


class LowRankUpdate(lx.AbstractLinearOperator):
    """Low-rank update operator ``L + U diag(d) Vᵀ``.

    Represents a base operator *L* plus a rank-k update. When *L*
    is cheap to solve (e.g. diagonal), the Woodbury identity gives
    efficient solves for the full operator.

    Args:
        base: The base operator *L*, with shape ``(m, n)``.
        U: Left factor, shape ``(m, k)``.
        d: Diagonal scaling, shape ``(k,)``. Defaults to ones.
        V: Right factor, shape ``(n, k)``. Defaults to *U* for
            square operators, yielding the symmetric update
            ``L + U diag(d) Uᵀ``.
        tags: Extra lineax tags -- the caller's structural claims.
        orthonormal: The caller's claim that *U* (and *V*) have orthonormal
            columns, ``UᵀU = I_k``. Like a tag, it is never checked. It
            takes effect only for a symmetric update (*V* omitted or *U*
            itself) on a scaled-identity base ``c·I`` (a lineax
            ``IdentityLinearOperator``, possibly scaled, negated or
            tagged -- ``low_rank_plus_identity(..., orthonormal=True)``
            builds one): `gaussx.solve` and `gaussx.logdet` then take
            ``O(nk)`` / ``O(k)`` closed forms with no ``k x k``
            factorisation, and `gaussx.inv` / `gaussx.sqrt` return another
            orthonormal ``LowRankUpdate`` on a scaled-identity base. On any
            other base (e.g. a general diagonal, where orthonormality does
            not simplify Woodbury) it has no effect on the result.

    Tags are inferred from structure only, never from array values, so the
    same call gives the same tags (and pytree structure) eagerly and under
    ``jax.jit``:

    - ``symmetric_tag`` when the base is symmetric and the factors are
      shared -- *V* omitted or passed as the same object as *U*. This is
      recorded in the static ``symmetric_factors`` field. Value-equal but
      distinct factors (``V = U.copy()``) are not inferred symmetric; pass
      ``tags=lx.symmetric_tag`` to claim it.
    - ``positive_semidefinite_tag`` additionally when the base is PSD and
      *d* is omitted (all ones). A caller-supplied *d* can have any sign,
      so pass ``tags=lx.positive_semidefinite_tag`` to claim PSD.
    """

    base: lx.AbstractLinearOperator
    U: Float[Array, "m k"]
    d: Float[Array, " k"]
    V: Float[Array, "n k"]
    orthonormal: bool = eqx.field(static=True)
    symmetric_factors: bool = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        base: lx.AbstractLinearOperator,
        U: Float[Array, "n k"],
        d: Float[Array, " k"] | None = None,
        V: Float[Array, "n k"] | None = None,
        *,
        tags: object | frozenset[object] = frozenset(),
        orthonormal: bool = False,
    ) -> None:
        m = base.out_size()
        n = base.in_size()
        k = U.shape[1] if U.ndim == 2 else 1
        if U.ndim == 1:
            U = U[:, None]
        unit_weights = d is None
        if d is None:
            d = jnp.ones(k, dtype=U.dtype)
        symmetric_factors = V is None or V is U
        if V is None:
            V = U
        if V.ndim == 1:
            V = V[:, None]
        if U.shape[0] != m or V.shape[0] != n:
            raise ValueError(
                f"U must have {m} rows and V must have {n} rows to match "
                f"base operator, "
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
        self.orthonormal = orthonormal
        self.symmetric_factors = symmetric_factors and m == n
        from gaussx._tags import low_rank_tag

        inferred_tags = _infer_tags(
            base,
            symmetric_factors=self.symmetric_factors,
            unit_weights=unit_weights,
        )
        self.tags = _to_frozenset(tags) | inferred_tags | {low_rank_tag}

    @property
    def rank(self) -> int:
        """Rank of the low-rank update."""
        return self.d.shape[0]

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        # (L + U diag(d) V^T) x = L x + U (d * (V^T x))
        base_part = self.base.mv(vector)
        vtx = self.V.T @ vector  # (k,)
        scaled = self.d * vtx  # (k,)
        update_part = self.U @ scaled  # (m,)
        return base_part + update_part

    def as_matrix(self) -> Float[Array, "m n"]:
        L = self.base.as_matrix()
        return L + self.U @ jnp.diag(self.d) @ self.V.T

    def transpose(self) -> LowRankUpdate:
        # Shared factors are passed once so the transpose keeps them shared
        # even under tracing, where ``U`` and ``V`` are distinct tracers.
        return LowRankUpdate(
            self.base.T,
            self.V,
            self.d,
            None if self.symmetric_factors else self.U,
            tags=lx.transpose_tags(self.tags),
            orthonormal=self.orthonormal,
        )

    def _dtype(self):
        # ``mv`` returns the promoted dtype of base, U, d and V; the declared
        # structure must agree, since lineax compares the two in solves.
        return jnp.result_type(self.base.in_structure().dtype, self.U, self.d, self.V)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.base.in_structure().shape, self._dtype())

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.base.out_structure().shape, self._dtype())


def low_rank_plus_diag(
    diag: Float[Array, " n"],
    U: Float[Array, "n k"],
    d: Float[Array, " k"] | None = None,
    V: Float[Array, "n k"] | None = None,
    *,
    psd: bool = False,
) -> LowRankUpdate:
    """Construct ``diag(diag) + U diag(d) Vᵀ``.

    Common pattern for inducing-point / Nystrom approximations
    where the base is a diagonal matrix.

    Args:
        diag: Diagonal entries, shape ``(n,)``.
        U: Left factor, shape ``(n, k)``.
        d: Diagonal scaling, shape ``(k,)``. Defaults to ones.
        V: Right factor, shape ``(n, k)``. Defaults to *U*.
        psd: Claim that ``diag`` and ``d`` are non-negative (and the
            factors shared), so the base and the update are tagged
            positive semidefinite. The sign of ``diag`` is never inspected,
            so without this the result is only symmetric-tagged.

    Returns:
        A ``LowRankUpdate`` with a ``DiagonalLinearOperator`` base.
    """
    return _low_rank_update_with_diag_base(diag, U, d, V, psd=psd)


def svd_low_rank_plus_diag(
    diag: Float[Array, " n"],
    U: Float[Array, "n k"],
    S: Float[Array, " k"],
    V: Float[Array, "n k"],
    *,
    psd: bool = False,
) -> LowRankUpdate:
    """Construct ``diag(diag) + U diag(S) Vᵀ`` from a truncated SVD.

    Args:
        diag: Diagonal entries, shape ``(n,)``.
        U: Left singular vectors, shape ``(n, k)``.
        S: Singular values, shape ``(k,)``.
        V: Right singular vectors, shape ``(n, k)``. Pass the same array
            object as *U* for a symmetric update; value-equal copies are
            not inferred symmetric.
        psd: Claim that ``diag`` is non-negative and the update is
            symmetric, so the result is tagged positive semidefinite.

    Returns:
        A ``LowRankUpdate`` with a ``DiagonalLinearOperator`` base.
    """
    return _low_rank_update_with_diag_base(diag, U, S, V, orthonormal=True, psd=psd)


def low_rank_plus_identity(
    U: Float[Array, "n k"],
    d: Float[Array, " k"] | None = None,
    V: Float[Array, "n k"] | None = None,
    *,
    scale: float = 1.0,
    psd: bool = False,
    orthonormal: bool = False,
) -> LowRankUpdate:
    """Construct ``scale * I + U diag(d) Vᵀ``.

    Common pattern for regularised low-rank models (e.g. noise + signal).

    Args:
        U: Left factor, shape ``(n, k)``.
        d: Diagonal scaling, shape ``(k,)``. Defaults to ones.
        V: Right factor, shape ``(n, k)``. Defaults to *U*.
        scale: Scalar multiplier on the identity. Default 1.0.
        psd: Claim that ``scale`` and ``d`` are non-negative (and the
            factors shared), so the result is tagged positive semidefinite.
            A Python-number ``scale >= 0`` already makes the base PSD
            (it is static, not an array value), so the default
            ``low_rank_plus_identity(U)`` is PSD-tagged without it.
        orthonormal: Claim that *U* has orthonormal columns (unchecked).
            The base is then a scaled lineax ``IdentityLinearOperator``
            rather than a ``DiagonalLinearOperator``, so a symmetric update
            (*V* omitted) takes the closed-form orthonormal ``solve`` /
            ``logdet`` / ``inv`` / ``sqrt`` (see `LowRankUpdate`).

    Returns:
        A ``LowRankUpdate`` with a scaled identity base.
    """
    n = U.shape[0]
    static_nonnegative = isinstance(scale, (int, float)) and scale >= 0
    if orthonormal:
        identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), U.dtype))
        base: lx.AbstractLinearOperator = jnp.asarray(scale, dtype=U.dtype) * identity
        if psd or static_nonnegative:
            base = lx.TaggedLinearOperator(base, lx.positive_semidefinite_tag)
        tags = (
            frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
            if psd
            else frozenset()
        )
        return LowRankUpdate(base, U, d, V, tags=tags, orthonormal=True)
    diag = jnp.full(n, scale, dtype=U.dtype)
    return _low_rank_update_with_diag_base(
        diag, U, d, V, psd=psd, psd_base=static_nonnegative
    )


def _low_rank_update_with_diag_base(
    diag: Float[Array, " n"],
    U: Float[Array, "n k"],
    d: Float[Array, " k"] | None = None,
    V: Float[Array, "n k"] | None = None,
    *,
    orthonormal: bool = False,
    psd: bool = False,
    psd_base: bool = False,
) -> LowRankUpdate:
    """Construct a low-rank update with a diagonal base operator.

    ``psd`` is the caller's claim that the whole update is PSD;
    ``psd_base`` that only the diagonal is non-negative.
    """
    base = lx.DiagonalLinearOperator(diag)
    if psd or psd_base:
        base = lx.TaggedLinearOperator(base, lx.positive_semidefinite_tag)
    tags = (
        frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
        if psd
        else frozenset()
    )
    return LowRankUpdate(base, U, d, V, tags=tags, orthonormal=orthonormal)


def _infer_tags(
    base: lx.AbstractLinearOperator,
    *,
    symmetric_factors: bool,
    unit_weights: bool,
) -> frozenset[object]:
    """Infer tags from static structure only (never from array values).

    Symmetric when the base is symmetric and the factors are shared; PSD
    when, in addition, the base is PSD and the weights are the default
    ones. Anything else is the caller's claim to make through ``tags``.
    """
    if not symmetric_factors or not _safe_query(lx.is_symmetric, base):
        return frozenset()
    inferred: set[object] = {lx.symmetric_tag}
    if unit_weights and _safe_query(lx.is_positive_semidefinite, base):
        inferred.add(lx.positive_semidefinite_tag)
    return frozenset(inferred)


def orthonormal_scaled_identity(operator: LowRankUpdate) -> Array | None:
    """The scalar ``c`` when ``operator`` is ``c·I + U diag(d) Uᵀ``, orthonormal.

    Decided from static structure only -- the ``orthonormal`` claim, shared
    factors and the base's operator types -- so the same operator takes the
    same branch eagerly and under ``jax.jit`` (gh-328, gh-333). Returns
    ``None`` when the orthonormal closed forms do not apply.
    """
    if not (operator.orthonormal and operator.symmetric_factors):
        return None
    return _scaled_identity_scalar(operator.base)


def scaled_identity_like(
    base: lx.AbstractLinearOperator,
    c: Array,
    tags: frozenset[object] = frozenset(),
) -> lx.AbstractLinearOperator:
    """``c·I`` on ``base``'s structure, wrapped in ``tags`` when given."""
    identity = lx.IdentityLinearOperator(base.in_structure())
    scaled = c * identity
    return lx.TaggedLinearOperator(scaled, tuple(tags)) if tags else scaled


def _scaled_identity_scalar(base: lx.AbstractLinearOperator) -> Array | None:
    """``c`` for a structurally scaled identity ``c·I``, else ``None``."""
    if isinstance(base, lx.TaggedLinearOperator):
        return _scaled_identity_scalar(base.operator)
    if isinstance(base, lx.IdentityLinearOperator):
        if base.in_size() != base.out_size():
            return None
        return jnp.ones((), dtype=base.in_structure().dtype)
    if isinstance(base, lx.NegLinearOperator):
        inner = _scaled_identity_scalar(base.operator)
        return None if inner is None else -inner
    if isinstance(base, lx.MulLinearOperator | lx.DivLinearOperator):
        inner = _scaled_identity_scalar(base.operator)
        if inner is None:
            return None
        scalar = jnp.asarray(base.scalar)
        if scalar.ndim != 0:
            return None
        if isinstance(base, lx.MulLinearOperator):
            return scalar * inner
        return inner / scalar
    return None


def _safe_query(query, operator: lx.AbstractLinearOperator) -> bool:
    """Evaluate a lineax tag query without propagating unsupported cases."""
    try:
        return bool(query(operator))
    except NotImplementedError:
        return False


# ---------------------------------------------------------------------------
# Deprecated compatibility class
# ---------------------------------------------------------------------------


class SVDLowRankUpdate(LowRankUpdate):
    """Deprecated subclass of `LowRankUpdate` with ``orthonormal=True``.

    Preserves the pre-consolidation public API until gaussx 0.7.0:

    - Same constructor signature as the old class — ``S`` defaults to
      ones (via the parent ``LowRankUpdate``) if omitted, and ``V``
      defaults to ``U`` so calls like ``SVDLowRankUpdate(base, U, S)``
      and ``SVDLowRankUpdate(base, U)`` continue to work.
    - Inherits from `LowRankUpdate` so ``isinstance`` /
      ``issubclass`` checks and ``singledispatch`` registrations keyed
      on this class keep working.
    - Forces ``orthonormal=True`` and emits a
      `DeprecationWarning` on construction.

    New code should construct ``LowRankUpdate(base, U, S, V,
    orthonormal=True)`` (or use `svd_low_rank_plus_diag`)
    directly. Will be removed in gaussx 0.7.0.
    """

    def __init__(
        self,
        base: lx.AbstractLinearOperator,
        U: Float[Array, "n k"],
        S: Float[Array, " k"] | None = None,
        V: Float[Array, "n k"] | None = None,
        *,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:

        warn_deprecated(
            "SVDLowRankUpdate is deprecated and will be removed in gaussx 0.7.0; "
            "use LowRankUpdate(..., orthonormal=True) or svd_low_rank_plus_diag()."
        )
        super().__init__(base, U, S, V, tags=tags, orthonormal=True)
