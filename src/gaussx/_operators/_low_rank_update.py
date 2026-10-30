"""Low-rank update A = A₀ + U D Vᵀ of a base operator, and its builders.

The base A₀ is M × N, the factors U and V are M × k and N × k, and
D = diag(d) holds the k weights d ∈ ℝ^k; k is the rank of the update and may
be zero. A solve goes through the k × k capacitance C = I + D Vᵀ A₀⁻¹ U
(Woodbury) instead of an N × N factorisation. A scaled identity base is
c I, with c a scalar.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._operators._block_diag import _to_frozenset
from gaussx._tags import low_rank_tag


class LowRankUpdate(lx.AbstractLinearOperator):
    """Low-rank update A = A₀ + U D Vᵀ of a base operator, with D = diag(d).

        A x = A₀ x + U (d ⊙ Vᵀ x)

    A₀ is M × N, U is M × k, V is N × k and d ∈ ℝ^k. The matvec never forms
    U D Vᵀ, and when A₀ is cheap to solve (a diagonal, say) `gaussx.solve`
    and `gaussx.logdet` go through the k × k capacitance
    C = I + D Vᵀ A₀⁻¹ U (Woodbury and the matrix determinant lemma).

    Tags are inferred from structure only, never from array values, so the
    same call gives the same tags (and pytree structure) eagerly and under
    ``jax.jit``:

    - ``symmetric_tag`` when A₀ is symmetric and the factors are shared:
      ``V`` omitted or passed as the same object as ``U``. This is recorded
      in the static ``symmetric_factors`` field. Value-equal but distinct
      factors (``V = U.copy()``) are not inferred symmetric; pass
      ``tags=lx.symmetric_tag`` to claim it.
    - ``positive_semidefinite_tag`` additionally when A₀ is PSD and ``d``
      is omitted (D = I_k). A caller-supplied d can have any sign, so pass
      ``tags=lx.positive_semidefinite_tag`` to claim PSD.

    Args:
        base: The base operator A₀, M × N.
        U: Left factor U, M × k; a vector is one column (k = 1).
        d: Weights d of D = diag(d), length k; defaults to ones.
        V: Right factor V, N × k; defaults to U, giving the symmetric
            update A₀ + U D Uᵀ for a square A₀.
        tags: Extra lineax tags, the caller's structural claims;
            ``low_rank_tag`` is always added.
        orthonormal: The caller's claim that U (and V) have orthonormal
            columns, UᵀU = I_k. Like a tag, it is never checked. It takes
            effect only for a symmetric update (``V`` omitted or ``U``
            itself) on a scaled-identity base c I (a lineax
            ``IdentityLinearOperator``, possibly scaled, negated or tagged;
            ``low_rank_plus_identity(..., orthonormal=True)`` builds one):
            `gaussx.solve` and `gaussx.logdet` then take O(N k) and O(k)
            closed forms with no k × k factorisation, and `gaussx.inv` /
            `gaussx.sqrt` return another orthonormal ``LowRankUpdate`` on a
            scaled-identity base. On any other base (a general diagonal,
            where orthonormality does not simplify Woodbury) it has no
            effect on the result.

    Raises:
        ValueError: If the rows of U or V do not match A₀, or U, V and d
            disagree on the rank k.

    Examples:
        A diagonal base plus a rank-1 update, solved by Woodbury with one
        1 × 1 solve and no 3 × 3 factorisation:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> base = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 3.0]))
        >>> U = jnp.array([[1.0], [0.0], [1.0]])
        >>> A = gaussx.LowRankUpdate(base, U)
        >>> A.rank
        1
        >>> b = jnp.ones(3)
        >>> x = gaussx.solve(A, b)
        >>> bool(jnp.allclose(A.as_matrix() @ x, b, atol=1e-5))
        True

        Shared factors and the default weights infer the symmetric and PSD
        tags from a PSD base:

        >>> psd = lx.positive_semidefinite_tag
        >>> A = gaussx.LowRankUpdate(lx.TaggedLinearOperator(base, psd), U)
        >>> lx.is_positive_semidefinite(A)
        True

        A rank-0 update is the base itself:

        >>> A0 = gaussx.LowRankUpdate(base, jnp.zeros((3, 0)))
        >>> bool(jnp.allclose(A0.as_matrix(), base.as_matrix()))
        True
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
            # a vector U is one column, k = 1
            # jnp, not einx: einx rejects the empty row axis of a 0 × 0 base
            U = U[:, None]
        unit_weights = d is None
        if d is None:
            # D = I_k
            d = jnp.ones(k, dtype=U.dtype)
        symmetric_factors = V is None or V is U
        if V is None:
            # V = U, the symmetric update A₀ + U D Uᵀ
            V = U
        if V.ndim == 1:
            # jnp, not einx: einx rejects the empty row axis of a 0 × 0 base
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

        inferred_tags = _infer_tags(
            base,
            symmetric_factors=self.symmetric_factors,
            unit_weights=unit_weights,
        )
        self.tags = _to_frozenset(tags) | inferred_tags | {low_rank_tag}

    @property
    def rank(self) -> int:
        """Rank k of the update, the number of columns of U and V."""
        return self.d.shape[0]

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        """Apply A to a vector without forming U D Vᵀ.

            A x = A₀ x + U (d ⊙ Vᵀ x)

        Args:
            vector: Input x, of length N.

        Returns:
            A x, of length M.
        """
        # A₀ x
        base_output = self.base.mv(vector)
        # Vᵀ x
        # jnp, not einx: einx rejects the empty rank axis of a k = 0 update
        projection = self.V.T @ vector
        # D Vᵀ x = d ⊙ Vᵀ x
        weighted_projection = self.d * projection
        # U D Vᵀ x
        update_output = self.U @ weighted_projection
        # A x = A₀ x + U D Vᵀ x
        return base_output + update_output

    def as_matrix(self) -> Float[Array, "m n"]:
        """Dense A = A₀ + U D Vᵀ, M × N.

        Returns:
            The materialised matrix.
        """
        base_matrix = self.base.as_matrix()
        # D = diag(d)
        weights = jnp.diag(self.d)
        # U D
        # jnp, not einx: einx rejects the empty rank axis of a k = 0 update
        weighted_left = self.U @ weights
        # U D Vᵀ
        update = weighted_left @ self.V.T
        # A₀ + U D Vᵀ
        return base_matrix + update

    def transpose(self) -> LowRankUpdate:
        """Aᵀ = A₀ᵀ + V D Uᵀ, still a `LowRankUpdate`.

        Returns:
            The transposed update, with transposed tags and the same
            ``orthonormal`` claim.
        """
        # Shared factors are passed once so the transpose keeps them shared
        # even under tracing, where ``U`` and ``V`` are distinct tracers.
        # Aᵀ = A₀ᵀ + V D Uᵀ
        return LowRankUpdate(
            self.base.T,
            self.V,
            self.d,
            None if self.symmetric_factors else self.U,
            tags=lx.transpose_tags(self.tags),
            orthonormal=self.orthonormal,
        )

    def _dtype(self):
        """Promoted dtype of A₀, U, d and V, the dtype ``mv`` returns."""
        # The declared structure must agree with ``mv``, since lineax
        # compares the two in solves.
        return jnp.result_type(self.base.in_structure().dtype, self.U, self.d, self.V)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (N,) of A₀'s input, with the promoted dtype."""
        return jax.ShapeDtypeStruct(self.base.in_structure().shape, self._dtype())

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape (M,) of A₀'s output, with the promoted dtype."""
        return jax.ShapeDtypeStruct(self.base.out_structure().shape, self._dtype())


def low_rank_plus_diag(
    diag: Float[Array, " n"],
    U: Float[Array, "n k"],
    d: Float[Array, " k"] | None = None,
    V: Float[Array, "n k"] | None = None,
    *,
    psd: bool = False,
) -> LowRankUpdate:
    """Build A = diag(a) + U D Vᵀ, a low-rank update of a diagonal base.

        A₀ = diag(a),   A = A₀ + U D Vᵀ,   D = diag(d)

    The common pattern for inducing-point and Nyström approximations.

    Args:
        diag: Diagonal entries a of the base A₀, length N.
        U: Left factor U, N × k.
        d: Weights d of D = diag(d), length k; defaults to ones.
        V: Right factor V, N × k; defaults to U.
        psd: Claim that a and d are non-negative (and the factors shared),
            so the base and the update are tagged positive semidefinite.
            The sign of a is never inspected, so without this the result is
            only symmetric-tagged.

    Returns:
        A `LowRankUpdate` with a ``DiagonalLinearOperator`` base.

    Examples:
        A 3 × 3 diagonal plus a rank-1 update:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> a = jnp.array([1.0, 2.0, 3.0])
        >>> U = jnp.array([[1.0], [0.0], [1.0]])
        >>> A = gaussx.low_rank_plus_diag(a, U)
        >>> bool(jnp.allclose(A.as_matrix(), jnp.diag(a) + U @ U.T))
        True

        Without ``psd`` only symmetry is inferred:

        >>> lx.is_symmetric(A), lx.is_positive_semidefinite(A)
        (True, False)
        >>> P = gaussx.low_rank_plus_diag(jnp.ones(3), U, psd=True)
        >>> lx.is_positive_semidefinite(P)
        True
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
    """Build A = diag(a) + U diag(s) Vᵀ from a truncated SVD.

        A₀ = diag(a),   A = A₀ + U diag(s) Vᵀ,   UᵀU = VᵀV = I_k

    The factors are claimed orthonormal (``orthonormal=True``), which only
    changes the result on a scaled-identity base (see `LowRankUpdate`).

    Args:
        diag: Diagonal entries a of the base A₀, length N.
        U: Left singular vectors U, N × k.
        S: Singular values s, length k.
        V: Right singular vectors V, N × k. Pass the same array object as
            ``U`` for a symmetric update; value-equal copies are not
            inferred symmetric.
        psd: Claim that a is non-negative and the update is symmetric, so
            the result is tagged positive semidefinite.

    Returns:
        An orthonormal `LowRankUpdate` with a ``DiagonalLinearOperator``
        base.

    Examples:
        One orthonormal column with singular value 2:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> U = jnp.array([[1.0], [0.0], [0.0]])
        >>> A = gaussx.svd_low_rank_plus_diag(jnp.ones(3), U, jnp.array([2.0]), U)
        >>> A.orthonormal, A.rank
        (True, 1)
        >>> bool(jnp.allclose(gaussx.diag(A), jnp.array([3.0, 1.0, 1.0])))
        True
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
    """Build A = c I + U D Vᵀ, a low-rank update of a scaled identity.

        A₀ = c I_N,   A = A₀ + U D Vᵀ,   D = diag(d)

    The common pattern for regularised low-rank models (noise plus signal).

    Args:
        U: Left factor U, N × k.
        d: Weights d of D = diag(d), length k; defaults to ones.
        V: Right factor V, N × k; defaults to U.
        scale: The scalar c on the identity.
        psd: Claim that c and d are non-negative (and the factors shared),
            so the result is tagged positive semidefinite. A Python-number
            c ≥ 0 already makes the base PSD (it is static, not an array
            value), so the default ``low_rank_plus_identity(U)`` is
            PSD-tagged without it.
        orthonormal: Claim that U has orthonormal columns, UᵀU = I_k
            (unchecked). The base is then a scaled lineax
            ``IdentityLinearOperator`` rather than a
            ``DiagonalLinearOperator``, so a symmetric update (``V``
            omitted) takes the closed-form orthonormal ``solve`` /
            ``logdet`` / ``inv`` / ``sqrt`` (see `LowRankUpdate`).

    Returns:
        A `LowRankUpdate` with a scaled identity base.

    Examples:
        Noise 0.5 I plus a rank-1 signal:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> U = jnp.array([[1.0], [1.0], [0.0]])
        >>> A = gaussx.low_rank_plus_identity(U, scale=0.5)
        >>> bool(jnp.allclose(A.as_matrix(), 0.5 * jnp.eye(3) + U @ U.T))
        True
        >>> lx.is_positive_semidefinite(A)
        True

        An orthonormal factor keeps an identity base, so the inverse is
        another low-rank update:

        >>> Q = jnp.array([[1.0], [0.0], [0.0]])
        >>> B = gaussx.low_rank_plus_identity(Q, scale=2.0, orthonormal=True)
        >>> type(gaussx.inv(B)).__name__
        'LowRankUpdate'
    """
    n = U.shape[0]
    static_nonnegative = isinstance(scale, (int, float)) and scale >= 0
    if orthonormal:
        identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), U.dtype))
        # A₀ = c I, in U's dtype
        base: lx.AbstractLinearOperator = jnp.asarray(scale, dtype=U.dtype) * identity
        if psd or static_nonnegative:
            base = lx.TaggedLinearOperator(base, lx.positive_semidefinite_tag)
        tags = (
            frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
            if psd
            else frozenset()
        )
        return LowRankUpdate(base, U, d, V, tags=tags, orthonormal=True)
    # A₀ = diag(c, …, c), in U's dtype
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
    """Build A = diag(a) + U D Vᵀ with the caller's PSD claims as tags.

    Args:
        diag: Diagonal entries a of the base A₀ = diag(a).
        U: Left factor U, N × k.
        d: Weights d of D = diag(d); defaults to ones.
        V: Right factor V; defaults to U.
        orthonormal: The orthonormal-columns claim passed to
            `LowRankUpdate`.
        psd: The caller's claim that the whole of A is PSD.
        psd_base: The caller's claim that only a is non-negative.

    Returns:
        The `LowRankUpdate`, with A₀ tagged PSD when either claim is made
        and A tagged symmetric and PSD under ``psd``.
    """
    # A₀ = diag(a)
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
    """Infer the tags of A₀ + U D Vᵀ from static structure, never values.

    Symmetric when A₀ is symmetric and the factors are shared (V = U); PSD
    when, in addition, A₀ is PSD and the weights are the default D = I_k.
    Anything else is the caller's claim to make through ``tags``.

    Args:
        base: The base operator A₀.
        symmetric_factors: Whether V is U, by object identity.
        unit_weights: Whether d was omitted, so D = I_k.

    Returns:
        The inferred lineax tags, possibly empty.
    """
    if not symmetric_factors or not _safe_query(lx.is_symmetric, base):
        return frozenset()
    inferred: set[object] = {lx.symmetric_tag}
    if unit_weights and _safe_query(lx.is_positive_semidefinite, base):
        inferred.add(lx.positive_semidefinite_tag)
    return frozenset(inferred)


def orthonormal_scaled_identity(operator: LowRankUpdate) -> Array | None:
    """The scalar c when the operator is c I + U D Uᵀ with UᵀU = I_k.

    Decided from static structure only (the ``orthonormal`` claim, shared
    factors and the base's operator types), so the same operator takes the
    same branch eagerly and under ``jax.jit`` (gh-328, gh-333).

    Args:
        operator: The low-rank update A₀ + U D Vᵀ.

    Returns:
        The rank-0 array c, or ``None`` when the orthonormal closed forms
        do not apply.
    """
    if not (operator.orthonormal and operator.symmetric_factors):
        return None
    return _scaled_identity_scalar(operator.base)


def scaled_identity_like(
    base: lx.AbstractLinearOperator,
    c: Array,
    tags: frozenset[object] = frozenset(),
) -> lx.AbstractLinearOperator:
    """The scaled identity c I on ``base``'s input structure.

    Args:
        base: The operator whose input structure sets the size and dtype.
        c: The scalar c.
        tags: Tags for c I; empty leaves it untagged.

    Returns:
        c I as a lineax operator, wrapped in ``tags`` when given.
    """
    identity = lx.IdentityLinearOperator(base.in_structure())
    # c I
    scaled = c * identity
    return lx.TaggedLinearOperator(scaled, tuple(tags)) if tags else scaled


def _scaled_identity_scalar(base: lx.AbstractLinearOperator) -> Array | None:
    """The scalar c of a structurally scaled identity c I, else ``None``.

    Unwraps tags, negation and scalar multiplication or division around a
    lineax ``IdentityLinearOperator``.

    Args:
        base: The candidate base operator A₀.

    Returns:
        The rank-0 array c, or ``None`` when A₀ is not structurally c I.
    """
    if isinstance(base, lx.TaggedLinearOperator):
        return _scaled_identity_scalar(base.operator)
    if isinstance(base, lx.IdentityLinearOperator):
        if base.in_size() != base.out_size():
            return None
        return jnp.ones((), dtype=base.in_structure().dtype)
    if isinstance(base, lx.NegLinearOperator):
        inner = _scaled_identity_scalar(base.operator)
        # −(c I) = (−c) I
        return None if inner is None else -inner
    if isinstance(base, lx.MulLinearOperator | lx.DivLinearOperator):
        inner = _scaled_identity_scalar(base.operator)
        if inner is None:
            return None
        scalar = jnp.asarray(base.scalar)
        if scalar.ndim != 0:
            return None
        if isinstance(base, lx.MulLinearOperator):
            # s (c I) = (s c) I
            return scalar * inner
        # (c I) / s = (c / s) I
        return inner / scalar
    return None


def _safe_query(query, operator: lx.AbstractLinearOperator) -> bool:
    """Evaluate a lineax tag query, reading an unsupported one as ``False``.

    Args:
        query: A lineax predicate such as ``lx.is_symmetric``.
        operator: The operator to query.

    Returns:
        The predicate's answer, or ``False`` when lineax raises
        ``NotImplementedError`` for this operator type.
    """
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

    Args:
        base: The base operator A₀.
        U: Left singular vectors U, N × k.
        S: Singular values s, length k; defaults to ones.
        V: Right singular vectors V, N × k; defaults to U.
        tags: Extra lineax tags.
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
