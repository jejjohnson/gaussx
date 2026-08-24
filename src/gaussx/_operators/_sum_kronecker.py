"""Sum of Kronecker products: A1 kron B1 + A2 kron B2 + ..."""

from __future__ import annotations

import abc
import functools as ft
from operator import add as operator_add

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange
from gaussx._operators._block_diag import _resolve_dtype, _to_frozenset
from gaussx._operators._kronecker import Kronecker


class SumOfKroneckers(lx.AbstractLinearOperator):
    r"""Sum of Kronecker products ``Σ_k A_k \otimes B_k``.

    Appears in multi-output GPs with correlated outputs, e.g.
    ``K_task \otimes K_spatial + \sigma^2 I_task \otimes I_spatial``.

    Matvec is computed as the sum of the Kronecker matvecs.

    `gaussx.solve` and `gaussx.logdet` dispatch structurally on **two**
    terms with one of them positive definite — ``B ⊗ C + σ² I`` and the
    general ``B₁ ⊗ C₁ + B₂ ⊗ C₂`` — by whitening with the positive-definite
    term and eigendecomposing each factor of the other. That costs
    ``O(n_c³ + n_d³)`` rather than the ``O((n_c n_d)³)`` of a dense solve,
    and the same reduction is reachable through the ``AddLinearOperator``
    chains `SumOperator` builds. The factors of the non-anchor term must
    advertise symmetry, and the anchor's must be diagonal, a multiple of the
    identity, or tagged positive-semidefinite; anything else keeps the dense
    fallback. Diagonal anchor factors are taken on trust to be positive, as
    everywhere else a Cholesky is implied.

    Three or more terms have no closed form and keep the dense fallback.
    Drive the structured `mv` instead with
    ``solve(op, b, solver=lineax.CG(...))`` and `gaussx.SLQLogdet`.

    `eigendecompose` is the separate, more general route: a joint
    eigendecomposition of the second Kronecker pair (requires
    ``A_2, B_2`` to be symmetric).  It forms a dense
    ``(n_c n_d) x (n_c n_d)`` matrix internally, so it is
    intended for moderate factor sizes (typical for multi-output GPs
    where the task dimension is small).

    Not to be confused with `KroneckerSum`, which is the standard
    matrix-theory Kronecker *sum* ``A \otimes I + I \otimes B`` — a
    different object with a closed-form eigendecomposition. This class is a
    sum of arbitrary Kronecker *products*, which has none in general.

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
            raise ValueError("SumOfKroneckers requires two-factor Kronecker products.")
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

    def transpose(self) -> SumOfKroneckers:
        # ``type(self)`` rather than the literal class so the deprecated
        # `SumKronecker` subclass survives a transpose — external
        # ``isinstance`` checks and ``singledispatch`` registrations keyed on
        # it would otherwise silently stop applying on ``op.T``.
        return type(self)(
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

        !!! note

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


# ---------------------------------------------------------------------------
# Structural solve / logdet: two-term simultaneous diagonalization
# ---------------------------------------------------------------------------

_KroneckerTerm = tuple[lx.AbstractLinearOperator, lx.AbstractLinearOperator]


class _Whitener(eqx.Module):
    r"""One axis of the anchor term's whitener ``L``, with ``S = L Lᵀ``.

    Solving and taking a logdet both need ``L^{-1}``, ``L^{-T}`` and
    ``log|det(S)|`` for each of the two Kronecker axes. Which subclass is
    used is fixed at trace time by `_whitener_kind`, so the identity and
    diagonal shortcuts cost nothing at runtime.
    """

    @abc.abstractmethod
    def inv(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        r"""Apply ``L^{-1}`` on the left."""

    @abc.abstractmethod
    def inv_t(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        r"""Apply ``L^{-T}`` on the left."""

    @abc.abstractmethod
    def logdet(self) -> Float[Array, ""]:
        r"""``log|det(S)| = log|det(L Lᵀ)|``."""

    def conjugate(self, matrix: Float[Array, "n n"]) -> Float[Array, "n n"]:
        r"""Whiten a symmetric factor: ``L^{-1} M L^{-T}``."""
        return self.inv(self.inv(matrix).T).T


class _IdentityWhitener(_Whitener):
    """``L = I`` — every application is a no-op."""

    def inv(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        return matrix

    def inv_t(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        return matrix

    def logdet(self) -> Float[Array, ""]:
        return jnp.zeros(())


class _DiagonalWhitener(_Whitener):
    """``L = diag(scale)`` — applications are elementwise.

    Attributes:
        scale: Square roots of the anchor factor's diagonal.
    """

    scale: Float[Array, " n"]

    def inv(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        return matrix / self.scale[:, None]

    def inv_t(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        return matrix / self.scale[:, None]

    def logdet(self) -> Float[Array, ""]:
        return 2.0 * jnp.sum(jnp.log(jnp.abs(self.scale)))


class _CholeskyWhitener(_Whitener):
    """Dense lower-triangular ``L`` — applications are triangular solves.

    Attributes:
        chol: Lower-triangular Cholesky factor of the anchor factor.
    """

    chol: Float[Array, "n n"]

    def inv(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        return jax.scipy.linalg.solve_triangular(self.chol, matrix, lower=True)

    def inv_t(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        return jax.scipy.linalg.solve_triangular(self.chol, matrix, lower=True, trans=1)

    def logdet(self) -> Float[Array, ""]:
        return 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(self.chol))))


class _SumOfKroneckersEigen(eqx.Module):
    r"""Simultaneous diagonalization of ``A₁ ⊗ B₁ + A₂ ⊗ B₂``.

    Built by `_sum_of_kroneckers_eigen`, which picks the positive-definite
    term as the *anchor* ``A₂ ⊗ B₂`` and whitens the other one by it:

    $$
    K = (L_a \otimes L_b)\,(U \otimes V)\,\Lambda\,(U \otimes V)^\top\,
        (L_a \otimes L_b)^\top ,
    \qquad A_2 = L_a L_a^\top,\; B_2 = L_b L_b^\top
    $$

    with ``Λ = diag(λ_a ⊗ λ_b + 1)`` from the per-factor eigendecompositions
    of ``L_a^{-1} A₁ L_a^{-T}`` and ``L_b^{-1} B₁ L_b^{-T}``. When the anchor
    is a multiple of the identity the whiteners drop out and the shift folds
    straight into the eigenvalues instead.

    Both operations cost two eigendecompositions of the *factors* — total
    ``O(n_a³ + n_b³ + n_a n_b (n_a + n_b))`` — instead of the ``O((n_a n_b)³)``
    of the dense fallback.

    Attributes:
        wa: Whitener for the first (``n_a``) axis.
        wb: Whitener for the second (``n_b``) axis.
        U: Eigenvectors of the whitened first factor, shape ``(n_a, n_a)``.
        V: Eigenvectors of the whitened second factor, shape ``(n_b, n_b)``.
        evals: Eigenvalues of the whitened operator, shape ``(n_a, n_b)``.
    """

    wa: _Whitener
    wb: _Whitener
    U: Float[Array, "n_a n_a"]
    V: Float[Array, "n_b n_b"]
    evals: Float[Array, "n_a n_b"]

    def solve(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        r"""Solve ``K x = b`` in the factorized basis.

        Args:
            vector: Right-hand side ``b`` of size ``n_a * n_b``.

        Returns:
            The solution ``x``.
        """
        n_a, n_b = self.evals.shape
        Y = rearrange(vector, "(a b) -> a b", a=n_a, b=n_b)
        # Z = L_a^{-1} Y L_b^{-T}
        Z = self.wb.inv(self.wa.inv(Y).T).T
        # Rotate into the eigenbasis, divide, rotate back.
        C = (self.U.T @ Z @ self.V) / self.evals
        Z = self.U @ C @ self.V.T
        # X = L_a^{-T} Z L_b^{-1}
        X = self.wb.inv_t(self.wa.inv_t(Z).T).T
        return rearrange(X, "a b -> (a b)")

    def logdet(self) -> Float[Array, ""]:
        r"""``log|det(K)|`` from the factor eigenvalues and the anchor."""
        n_a, n_b = self.evals.shape
        return (
            jnp.sum(jnp.log(jnp.abs(self.evals)))
            + n_b * self.wa.logdet()
            + n_a * self.wb.logdet()
        )


def _unwrap_tagged(
    operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    return operator


def _scaled_identity(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, ""] | None:
    """Return ``c`` when ``operator`` is *statically* ``c · I``, else ``None``.

    The recognition is by operator type, not by inspecting entries, so the
    answer is available at trace time; only the returned scalar is traced.
    """
    operator = _unwrap_tagged(operator)
    if isinstance(operator, lx.IdentityLinearOperator):
        return jnp.ones(())
    if isinstance(operator, lx.MulLinearOperator):
        inner = _scaled_identity(operator.operator)
        return None if inner is None else inner * operator.scalar
    if isinstance(operator, lx.DivLinearOperator):
        inner = _scaled_identity(operator.operator)
        return None if inner is None else inner / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        inner = _scaled_identity(operator.operator)
        return None if inner is None else -inner
    return None


def _whitener_kind(
    operator: lx.AbstractLinearOperator,
) -> type[_Whitener] | None:
    """Static classification of an anchor factor, or ``None`` if unusable.

    Diagonal factors are accepted on the documented precondition that their
    entries are positive — the same contract `cholesky` carries. Anything
    else has to advertise symmetry *and* positive-semidefiniteness through
    its lineax tags; untagged operators fall through to the dense path
    rather than risk a ``NaN`` whitening.
    """
    operator = _unwrap_tagged(operator)
    if isinstance(operator, lx.IdentityLinearOperator):
        return _IdentityWhitener
    if isinstance(operator, lx.DiagonalLinearOperator):
        return _DiagonalWhitener
    if lx.is_symmetric(operator) and lx.is_positive_semidefinite(operator):
        return _CholeskyWhitener
    return None


def _build_whitener(
    kind: type[_Whitener], operator: lx.AbstractLinearOperator
) -> _Whitener:
    operator = _unwrap_tagged(operator)
    if kind is _IdentityWhitener:
        return _IdentityWhitener()
    if kind is _DiagonalWhitener:
        return _DiagonalWhitener(jnp.sqrt(lx.diagonal(operator)))
    from gaussx._primitives._cholesky import cholesky

    return _CholeskyWhitener(cholesky(operator).as_matrix())


def _add_leaves(
    operator: lx.AbstractLinearOperator,
) -> tuple[lx.AbstractLinearOperator, ...]:
    """Flatten a chain of lineax ``AddLinearOperator`` compositions."""
    operator = _unwrap_tagged(operator)
    if isinstance(operator, lx.AddLinearOperator):
        return (*_add_leaves(operator.operator1), *_add_leaves(operator.operator2))
    return (operator,)


def _kronecker_terms(
    operator: lx.AbstractLinearOperator,
) -> tuple[_KroneckerTerm, ...] | None:
    r"""Normalize ``operator`` to a tuple of two-factor Kronecker terms.

    Accepts a `SumOfKroneckers`, a `Kronecker`, or an ``AddLinearOperator``
    chain — the shape `SumOperator` produces — whose leaves are `Kronecker` /
    `SumOfKroneckers` operators plus any number of scalar-identity shifts.
    The identity shifts are summed and returned as a single ``I ⊗ cI`` term.

    Returns ``None`` when the operator is not of that form, or when the
    Kronecker factors are not square with sizes shared across terms — the
    caller then keeps its existing fallback.

    Args:
        operator: Any lineax operator.

    Returns:
        The ``((A₁, B₁), (A₂, B₂), …)`` factor pairs, or ``None``.
    """
    krons: list[Kronecker] = []
    shifts: list[Float[Array, ""]] = []
    for leaf in _add_leaves(operator):
        if isinstance(leaf, SumOfKroneckers):
            krons.extend(leaf.operators)
            continue
        if isinstance(leaf, Kronecker):
            krons.append(leaf)
            continue
        scalar = _scaled_identity(leaf)
        if scalar is None:
            return None
        shifts.append(scalar)
    if not krons or any(len(kron.operators) != 2 for kron in krons):
        return None

    terms: list[_KroneckerTerm] = [
        (kron.operators[0], kron.operators[1]) for kron in krons
    ]
    n_a = terms[0][0].in_size()
    n_b = terms[0][1].in_size()
    for factor_a, factor_b in terms:
        square = (
            factor_a.in_size() == factor_a.out_size()
            and factor_b.in_size() == factor_b.out_size()
        )
        if not square or factor_a.in_size() != n_a or factor_b.in_size() != n_b:
            return None

    if shifts:
        dtype = terms[0][0].in_structure().dtype
        total = ft.reduce(operator_add, shifts)
        identity_a = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n_a,), dtype))
        identity_b = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n_b,), dtype))
        terms.append((identity_a, total * identity_b))
    return tuple(terms)


def _select_anchor(
    terms: tuple[_KroneckerTerm, ...],
) -> tuple[int, type[_Whitener], type[_Whitener]] | None:
    """Pick the positive-definite term to whiten by — a static decision.

    Returns ``(anchor_index, kind_a, kind_b)``, or ``None`` when no exact
    two-term reduction applies. The second term is tried first: a noise or
    jitter shift is conventionally written last.
    """
    if len(terms) != 2:
        return None
    for anchor_index in (1, 0):
        main = terms[1 - anchor_index]
        anchor = terms[anchor_index]
        # ``eigh`` on the whitened main factors needs them symmetric.
        if not (lx.is_symmetric(main[0]) and lx.is_symmetric(main[1])):
            continue
        kind_a = _whitener_kind(anchor[0])
        kind_b = _whitener_kind(anchor[1])
        if kind_a is not None and kind_b is not None:
            return anchor_index, kind_a, kind_b
    return None


def _is_eigen_reducible(operator: lx.AbstractLinearOperator) -> bool:
    """Whether `_sum_of_kroneckers_eigen` has an exact path for ``operator``.

    A purely structural query — operator types and lineax tags only, no
    array work — so it is safe to call from solver-strategy selection.

    Args:
        operator: Any lineax operator.

    Returns:
        ``True`` if the exact two-term reduction applies.
    """
    terms = _kronecker_terms(operator)
    return terms is not None and _select_anchor(terms) is not None


def _sum_of_kroneckers_eigen(
    operator: lx.AbstractLinearOperator,
) -> _SumOfKroneckersEigen | None:
    r"""Factorize a two-term sum of Kronecker products, if one applies.

    Recognizes ``A₁ ⊗ B₁ + A₂ ⊗ B₂`` — written as a `SumOfKroneckers`, or as
    the ``AddLinearOperator`` chain `SumOperator` builds — with one term
    positive definite, and returns the simultaneous diagonalization that
    `gaussx.solve` and `gaussx.logdet` dispatch on. Returns ``None`` for
    anything else, including three or more Kronecker terms, which have no
    closed form; the iterative alternatives for that case are
    ``solve(..., solver=lineax.CG(...))`` and `gaussx.SLQLogdet`, both of
    which drive the structured `SumOfKroneckers.mv`.

    Args:
        operator: Any lineax operator.

    Returns:
        The factorization, or ``None`` when no exact path applies.
    """
    terms = _kronecker_terms(operator)
    if terms is None:
        return None
    selection = _select_anchor(terms)
    if selection is None:
        return None
    anchor_index, kind_a, kind_b = selection
    main = terms[1 - anchor_index]
    anchor = terms[anchor_index]

    scalar_a = _scaled_identity(anchor[0])
    scalar_b = _scaled_identity(anchor[1])
    A1 = main[0].as_matrix()
    B1 = main[1].as_matrix()
    if scalar_a is not None and scalar_b is not None:
        # Anchor is c·I: no whitening, and the shift folds into the
        # eigenvalues — which keeps the path exact for a negative shift too.
        wa = _IdentityWhitener()
        wb = wa
        shift = scalar_a * scalar_b
    else:
        wa = _build_whitener(kind_a, anchor[0])
        wb = _build_whitener(kind_b, anchor[1])
        A1 = wa.conjugate(A1)
        B1 = wb.conjugate(B1)
        shift = jnp.ones(())

    evals_a, U = jnp.linalg.eigh(A1)
    evals_b, V = jnp.linalg.eigh(B1)
    evals = evals_a[:, None] * evals_b[None, :] + shift
    return _SumOfKroneckersEigen(wa=wa, wb=wb, U=U, V=V, evals=evals)


def sumkronecker_sample(
    op: SumOfKroneckers,
    *,
    key: jax.Array,
    num_samples: int = 1,
    lanczos_order: int = 50,
) -> Float[Array, "num_samples n"]:
    r"""Sample from ``𝒩(0, op)`` with matrix-free Lanczos square roots.

    The square-root action is evaluated by ``matfree`` Lanczos against
    ``op.mv``. This avoids materialising the dense ``(n_A n_B) ×
    (n_A n_B)`` covariance and costs ``lanczos_order`` operator
    matvecs per sample.

    Args:
        op: Positive-semidefinite `SumOfKroneckers` covariance operator.
        key: JAX PRNG key.
        num_samples: Number of independent samples to draw.
        lanczos_order: Lanczos truncation order.

    Returns:
        Samples with shape ``(num_samples, op.in_size())``.
    """
    from gaussx._primitives._sqrt import sqrt

    if op.in_size() != op.out_size():
        raise ValueError(
            "sumkronecker_sample requires a square SumOfKroneckers, got "
            f"in_size={op.in_size()} and out_size={op.out_size()}."
        )
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    sqrt_op = sqrt(op, lanczos_order=lanczos_order)
    eps = jax.random.normal(
        key, (num_samples, op.in_size()), dtype=op.in_structure().dtype
    )
    return jax.vmap(sqrt_op.mv)(eps)


# ---------------------------------------------------------------------------
# Deprecated compatibility class
# ---------------------------------------------------------------------------


class SumKronecker(SumOfKroneckers):
    """Deprecated alias for `SumOfKroneckers`.

    The old name was one word-order away from `KroneckerSum`, a
    different operator with a different eigendecomposition — picking the wrong
    one produced silently wrong solves. Renamed in gh-136.

    Subclasses `SumOfKroneckers` so ``isinstance`` checks and
    ``singledispatch`` registrations keyed on this class keep working, and
    emits a `DeprecationWarning` on construction. Structural operations that
    rebuild the operator — ``transpose`` / ``.T`` — preserve this type, so
    they warn again. Note that instances built via `SumOfKroneckers` are
    *not* instances of this subclass; internal dispatch keys on the parent.
    Will be removed in a future release.
    """

    def __init__(
        self,
        kron1: Kronecker,
        kron2: Kronecker,
        *krons: Kronecker,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        import warnings

        warnings.warn(
            "SumKronecker is deprecated; use SumOfKroneckers "
            "(KroneckerSum remains a different operator).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(kron1, kron2, *krons, tags=tags)
