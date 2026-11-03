"""Sum of Kronecker products ∑_j A_j ⊗ B_j and its two-term structured solve.

Each term is a two-factor Kronecker product A_j ⊗ B_j, j = 1, …, J, with A_j
of size N_1 × N_1 and B_j of size N_2 × N_2 for the solve, so the sum is
N × N with N = N_1 N_2, and vec(Z) flattens Z ∈ ℝ^{N_1 × N_2} row by row.
Two terms K = A_1 ⊗ B_1 + A_2 ⊗ B_2 have an exact reduction when the
anchor term A_2 ⊗ B_2 is positive definite: whiten by its factors
A_2 = L_A L_Aᵀ and B_2 = L_B L_Bᵀ, eigendecompose the whitened main
factors, L_A⁻¹ A_1 L_A⁻ᵀ = Q_A diag(γ_A) Q_Aᵀ and
L_B⁻¹ B_1 L_B⁻ᵀ = Q_B diag(γ_B) Q_Bᵀ, and

    K = (L_A ⊗ L_B) (Q_A ⊗ Q_B) diag(γ_A ⊗ γ_B + 1) (Q_A ⊗ Q_B)ᵀ (L_A ⊗ L_B)ᵀ.

An anchor that is a scalar multiple c I folds in as a shift of the main
term's factored eigenvalues instead. The reduction has a known float32 issue
(gh-647): a noise factor written as ``c * operator`` loses its PSD tag, so an
ill-conditioned main term can be chosen as the anchor; the numerics here are
unchanged.
"""

from __future__ import annotations

import abc
import functools as ft
from operator import add as operator_add
from typing import NamedTuple

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.core import Tracer
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import einsum, rearrange
from gaussx._operators._block_diag import _resolve_dtype, _to_frozenset
from gaussx._operators._factored_eigen import FactoredEigen, kronecker_eigen
from gaussx._operators._kronecker import Kronecker
from gaussx._primitives._eig import eig


class SumOfKroneckers(lx.AbstractLinearOperator):
    """Sum of Kronecker products ∑_j A_j ⊗ B_j, applied term by term.

        (∑_j A_j ⊗ B_j) vec(Z) = vec(∑_j A_j Z B_jᵀ)

    Appears in multi-output GPs with correlated outputs, e.g.
    K_task ⊗ K_space + σ² I ⊗ I. The matvec is the sum of the Kronecker
    matvecs.

    `gaussx.solve` and `gaussx.logdet` dispatch structurally on **two**
    terms with one of them positive definite (A ⊗ B + σ² I and the general
    A_1 ⊗ B_1 + A_2 ⊗ B_2) by whitening with the positive-definite term and
    eigendecomposing each factor of the other. That costs O(N_1³ + N_2³)
    rather than the O((N_1 N_2)³) of a dense solve,
    and the same reduction is reachable through the ``AddLinearOperator``
    chains `sum_operator` builds. The factors of the non-anchor term must
    advertise symmetry, and the anchor's must be diagonal, a multiple of the
    identity, or tagged positive-semidefinite; anything else keeps the dense
    fallback.

    !!! warning "The anchor must be positive *definite*"

        Whitening inverts the anchor term. ``positive_semidefinite_tag``,
        the strongest statement lineax can make, permits a singular
        operator, and whitening by one of those returns ``NaN`` where the
        dense fallback would have answered. When both terms qualify, the one
        whose inversion needs no factorization is preferred, so a scalar or
        diagonal shift always wins over a tagged dense factor; a
        multiple-of-the-identity anchor is folded in as a shift and is never
        inverted at all. Beyond that the definiteness is the caller's to
        guarantee, exactly as it is for `gaussx.cholesky`.

        A diagonal or scaled-identity anchor *factor* is whitened by its
        square root, so each entry must be strictly positive. With concrete
        values a non-positive one is skipped in favour of the other term, or
        of the dense fallback. A traced one cannot be inspected, so it keeps
        the diagonal whitening and raises an ``EquinoxRuntimeError`` at run
        time rather than returning ``NaN``.

    Three or more terms have no closed form and keep the dense fallback.
    Drive the structured `mv` instead with
    ``solve(op, b, solver=lineax.CG(...))`` and `gaussx.SLQLogdet`.

    `eigendecompose` is the separate, more general route: a joint
    eigendecomposition in the eigenbasis of the second Kronecker pair
    (all four factors symmetric). It forms a dense N × N matrix internally,
    so it is intended for moderate factor sizes (typical for multi-output
    GPs where the task dimension is small).

    Not to be confused with `KroneckerSum`, which is the standard
    matrix-theory Kronecker *sum* A_1 ⊕ A_2 = A_1 ⊗ I + I ⊗ A_2, a different
    object with a closed-form eigendecomposition. This class is a sum of
    arbitrary Kronecker *products*, which has none in general.

    Args:
        kron1: First Kronecker product A_1 ⊗ B_1.
        kron2: Second Kronecker product A_2 ⊗ B_2.
        *krons: Further two-factor Kronecker products A_j ⊗ B_j, j ≥ 3.
        tags: Extra lineax tags for the sum.

    Raises:
        ValueError: If a term is not a two-factor `Kronecker`, or the terms
            differ in input or output size.

    Examples:
        A Kronecker product plus the identity, the multi-output GP shape:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> B = lx.MatrixLinearOperator(jnp.array([[3.0, 0.5], [0.5, 1.0]]), psd)
        >>> I2 = lx.IdentityLinearOperator(A.in_structure())
        >>> K = gaussx.SumOfKroneckers(gaussx.Kronecker(A, B), gaussx.Kronecker(I2, I2))
        >>> dense = jnp.kron(A.as_matrix(), B.as_matrix()) + jnp.eye(4)
        >>> bool(jnp.allclose(K.as_matrix(), dense))
        True

        The logdet takes the two-term reduction, one eigh per factor:

        >>> sign, expected = jnp.linalg.slogdet(dense)
        >>> bool(jnp.allclose(gaussx.logdet(K), expected, atol=1e-5))
        True
        >>> gaussx.is_eigen_reducible(K)
        True
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
        """The first term A_1 ⊗ B_1."""
        return self.operators[0]

    @property
    def kron2(self) -> Kronecker:
        """The second term A_2 ⊗ B_2."""
        return self.operators[1]

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " m"]:
        """Apply the sum to a vector, one Kronecker matvec per term.

            (∑_j A_j ⊗ B_j) vec(Z) = ∑_j vec(A_j Z B_jᵀ)

        Args:
            vector: Input vec(Z).

        Returns:
            The sum of the term matvecs.
        """
        # (A_1 ⊗ B_1) vec(Z)
        result = self.operators[0].mv(vector)
        for kron in self.operators[1:]:
            # + (A_j ⊗ B_j) vec(Z)
            result = result + kron.mv(vector)
        return result

    def as_matrix(self) -> Float[Array, "n n"]:
        """Dense ∑_j A_j ⊗ B_j.

        Returns:
            The materialised sum.
        """
        # A_1 ⊗ B_1
        result = self.operators[0].as_matrix()
        for kron in self.operators[1:]:
            # + A_j ⊗ B_j
            result = result + kron.as_matrix()
        return result

    def transpose(self) -> SumOfKroneckers:
        """(∑_j A_j ⊗ B_j)ᵀ = ∑_j A_jᵀ ⊗ B_jᵀ, of the same class.

        Returns:
            The transposed sum, with transposed tags.
        """
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
        """Shape and dtype of the input vector."""
        return jax.ShapeDtypeStruct((self._in_size,), jnp.dtype(self._dtype))

    def out_structure(self) -> jax.ShapeDtypeStruct:
        """Shape and dtype of the output vector."""
        return jax.ShapeDtypeStruct((self._out_size,), jnp.dtype(self._dtype))

    def eigendecompose(
        self,
    ) -> tuple[Float[Array, " n"], Float[Array, "n n"]]:
        """Eigendecomposition through the eigenbasis of the second pair.

            K = (Q_A ⊗ Q_B) V diag(γ) Vᵀ (Q_A ⊗ Q_B)ᵀ,
            Ã_1 ⊗ B̃_1 + diag(γ_A ⊗ γ_B) = V diag(γ) Vᵀ

        Here A_2 = Q_A diag(γ_A) Q_Aᵀ, B_2 = Q_B diag(γ_B) Q_Bᵀ, and
        Ã_1 = Q_Aᵀ A_1 Q_A, B̃_1 = Q_Bᵀ B_1 Q_B are the first pair rotated
        into that basis.

        !!! note

            This forms a dense N × N matrix internally and is O(N³), with
            N = N_1 N_2. Intended for moderate factor sizes (e.g.
            multi-output GPs where the task dimension is small).

        Algorithm:
            1. Eigendecompose A_2 and B_2 through `gaussx.eig`, so diagonal,
               block-diagonal or Kronecker factors skip materialisation.
            2. Rotate A_1 and B_1 into those eigenbases.
            3. Densely eigendecompose Ã_1 ⊗ B̃_1 + diag(γ_A ⊗ γ_B).
            4. Rotate the eigenvectors back, Q = (Q_A ⊗ Q_B) V.

        Raises:
            ValueError: If there are not exactly two terms, or a factor of
                either term is not symmetric.

        Returns:
            Tuple ``(eigenvalues, Q)`` with K = Q diag(eigenvalues) Qᵀ.
        """
        if len(self.operators) != 2:
            count = len(self.operators)
            raise ValueError(
                f"eigendecompose requires exactly two Kronecker products, got {count}."
            )
        A2_op, B2_op = self.kron2.operators
        A1_op, B1_op = self.kron1.operators
        # The final eigh needs a symmetric matrix, which needs *both* pairs
        # symmetric (so Ã_1 and B̃_1 stay symmetric under the rotation).
        if not lx.is_symmetric(A2_op) or not lx.is_symmetric(B2_op):
            raise ValueError("eigendecompose requires kron2 factors to be symmetric.")
        if not lx.is_symmetric(A1_op) or not lx.is_symmetric(B1_op):
            raise ValueError("eigendecompose requires kron1 factors to be symmetric.")

        A1, B1 = (op.as_matrix() for op in self.kron1.operators)

        # --- eigenbasis of the second pair ---
        # Through the structural primitive: diagonal, block-diagonal or
        # nested Kronecker factors of A_2 or B_2 skip materialisation here.
        # A_2 = Q_A diag(γ_A) Q_Aᵀ
        evals_c, Q_C = eig(A2_op)
        # B_2 = Q_B diag(γ_B) Q_Bᵀ
        evals_d, Q_D = eig(B2_op)

        # --- rotate the first pair ---
        # Q_Aᵀ A_1
        rotated_left_a = einsum(Q_C, A1, "row eig, row col -> eig col")
        # Ã_1 = Q_Aᵀ A_1 Q_A
        A1_tilde = einsum(rotated_left_a, Q_C, "eig col, col other -> eig other")
        # Q_Bᵀ B_1
        rotated_left_b = einsum(Q_D, B1, "row eig, row col -> eig col")
        # B̃_1 = Q_Bᵀ B_1 Q_B
        B1_tilde = einsum(rotated_left_b, Q_D, "eig col, col other -> eig other")

        # --- joint eigendecomposition ---
        # γ_A ⊗ γ_B, as an N_1 × N_2 grid
        pairwise = einx.multiply("a, b -> a b", evals_c, evals_d)
        # vec(γ_A ⊗ γ_B)
        diag_vals = rearrange(pairwise, "a b -> (a b)")
        # Ã_1 ⊗ B̃_1 + diag(γ_A ⊗ γ_B)
        transformed = jnp.kron(A1_tilde, B1_tilde) + jnp.diag(diag_vals)
        # V diag(γ) Vᵀ
        evals, V = jnp.linalg.eigh(transformed)

        # Q = (Q_A ⊗ Q_B) V
        Q = einsum(jnp.kron(Q_C, Q_D), V, "row mid, mid col -> row col")
        return evals, Q


# ---------------------------------------------------------------------------
# Structural solve / logdet: two-term simultaneous diagonalisation
# ---------------------------------------------------------------------------

_KroneckerTerm = tuple[lx.AbstractLinearOperator, lx.AbstractLinearOperator]


class _Whitener(eqx.Module):
    """One axis of the anchor term's whitener L, with anchor factor M = L Lᵀ.

    Solving and taking a logdet both need L⁻¹, L⁻ᵀ and log|det M| for each
    of the two Kronecker axes. Which subclass is used is fixed at trace time
    by `_whitener_kind`, so the identity and diagonal shortcuts cost nothing
    at run time.
    """

    @abc.abstractmethod
    def inv(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        """Apply L⁻¹ on the left, L⁻¹ X."""

    @abc.abstractmethod
    def inv_t(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        """Apply L⁻ᵀ on the left, L⁻ᵀ X."""

    @abc.abstractmethod
    def logdet(self) -> Float[Array, ""]:
        """log|det M| = log|det(L Lᵀ)|."""

    def conjugate(self, matrix: Float[Array, "n n"]) -> Float[Array, "n n"]:
        """Whiten a symmetric factor, L⁻¹ X L⁻ᵀ = (L⁻¹ (L⁻¹ X)ᵀ)ᵀ."""
        # L⁻¹ X
        left = self.inv(matrix)
        # L⁻¹ (L⁻¹ X)ᵀ = L⁻¹ Xᵀ L⁻ᵀ
        both = self.inv(rearrange(left, "row col -> col row"))
        # L⁻¹ X L⁻ᵀ, for a symmetric X
        return rearrange(both, "row col -> col row")


class _IdentityWhitener(_Whitener):
    """L = I: every application is a no-op."""

    def inv(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        # I⁻¹ X = X
        return matrix

    def inv_t(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        # I⁻ᵀ X = X
        return matrix

    def logdet(self) -> Float[Array, ""]:
        # log|det I| = 0
        return jnp.zeros(())


class _DiagonalWhitener(_Whitener):
    """L = diag(ℓ): applications are elementwise.

    Attributes:
        scale: The diagonal ℓ, square roots of the anchor factor's diagonal.
    """

    scale: Float[Array, " n"]

    def inv(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        # jnp, not einx: einx cannot resolve an empty axis (no columns).
        # L⁻¹ X = X_ik / ℓ_i
        return matrix / self.scale[:, None]

    def inv_t(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        # jnp, not einx: einx cannot resolve an empty axis (no columns).
        # L⁻ᵀ X = X_ik / ℓ_i
        return matrix / self.scale[:, None]

    def logdet(self) -> Float[Array, ""]:
        # log|det(L Lᵀ)| = 2 ∑ log|ℓ|
        return 2.0 * jnp.sum(jnp.log(jnp.abs(self.scale)))


class _CholeskyWhitener(_Whitener):
    """Dense lower-triangular L: applications are triangular solves.

    Attributes:
        chol: Lower-triangular Cholesky factor L of the anchor factor.
    """

    chol: Float[Array, "n n"]

    def inv(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        # L⁻¹ X
        return jax.scipy.linalg.solve_triangular(self.chol, matrix, lower=True)

    def inv_t(self, matrix: Float[Array, "n k"]) -> Float[Array, "n k"]:
        # L⁻ᵀ X
        return jax.scipy.linalg.solve_triangular(self.chol, matrix, lower=True, trans=1)

    def logdet(self) -> Float[Array, ""]:
        # log|det(L Lᵀ)| = 2 ∑ log|L_ii|
        return 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(self.chol))))


class _SumOfKroneckersEigen(eqx.Module):
    """Simultaneous diagonalisation of K = A_1 ⊗ B_1 + A_2 ⊗ B_2.

        K = (L_A ⊗ L_B) (Q_A ⊗ Q_B) diag(γ) (Q_A ⊗ Q_B)ᵀ (L_A ⊗ L_B)ᵀ,
        γ = γ_A ⊗ γ_B + 1

    Built by `_sum_of_kroneckers_eigen`, which picks the positive-definite
    term as the *anchor* A_2 ⊗ B_2 = (L_A L_Aᵀ) ⊗ (L_B L_Bᵀ) and whitens the
    other by it; Q_A diag(γ_A) Q_Aᵀ and Q_B diag(γ_B) Q_Bᵀ are the
    eigendecompositions of L_A⁻¹ A_1 L_A⁻ᵀ and L_B⁻¹ B_1 L_B⁻ᵀ. When the
    anchor is a multiple of the identity the whiteners drop out and the shift
    folds straight into the eigenvalues instead (`_shifted_factorization`).

    Both operations cost two eigendecompositions of the *factors*, in total
    O(N_1³ + N_2³ + N (N_1 + N_2)), instead of the O(N³) of the dense
    fallback.

    Attributes:
        wa: Whitener L_A for the first (N_1) axis.
        wb: Whitener L_B for the second (N_2) axis.
        U: Eigenvectors Q_A of the whitened first factor, N_1 × N_1.
        V: Eigenvectors Q_B of the whitened second factor, N_2 × N_2.
        evals: Eigenvalues γ = γ_A ⊗ γ_B + 1, N_1 × N_2.
    """

    wa: _Whitener
    wb: _Whitener
    U: Float[Array, "n_a n_a"]
    V: Float[Array, "n_b n_b"]
    evals: Float[Array, "n_a n_b"]

    def solve(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        """Solve K x = b in the factorised basis.

            X = L_A⁻ᵀ Q_A ((Q_Aᵀ L_A⁻¹ Y L_B⁻ᵀ Q_B) ⊘ γ) Q_Bᵀ L_B⁻¹

        with vec(Y) = b and vec(X) = x.

        Algorithm:
            1. Whiten the right-hand side, W = L_A⁻¹ Y L_B⁻ᵀ.
            2. Rotate into the eigenbasis, divide by γ, rotate back.
            3. Unwhiten, X = L_A⁻ᵀ W′ L_B⁻¹.

        Args:
            vector: Right-hand side b, of length N = N_1 N_2.

        Returns:
            The solution x.
        """
        n_a, n_b = self.evals.shape
        # Y, vec(Y) = b
        Y = rearrange(vector, "(a b) -> a b", a=n_a, b=n_b)

        # --- whiten ---
        # L_A⁻¹ Y
        whitened_rows = self.wa.inv(Y)
        # L_B⁻¹ (L_A⁻¹ Y)ᵀ
        whitened_both = self.wb.inv(rearrange(whitened_rows, "a b -> b a"))
        # W = L_A⁻¹ Y L_B⁻ᵀ
        Z = rearrange(whitened_both, "b a -> a b")

        # --- divide in the eigenbasis ---
        # Q_Aᵀ W
        rotated_left = einsum(self.U, Z, "a eig_a, a b -> eig_a b")
        # Q_Aᵀ W Q_B
        rotated = einsum(rotated_left, self.V, "eig_a b, b eig_b -> eig_a eig_b")
        # C = (Q_Aᵀ W Q_B) ⊘ γ
        C = rotated / self.evals
        # Q_A C
        back_left = einsum(self.U, C, "a eig_a, eig_a eig_b -> a eig_b")
        # W′ = Q_A C Q_Bᵀ
        Z = einsum(back_left, self.V, "a eig_b, b eig_b -> a b")

        # --- unwhiten ---
        # L_A⁻ᵀ W′
        unwhitened_rows = self.wa.inv_t(Z)
        # L_B⁻ᵀ (L_A⁻ᵀ W′)ᵀ
        unwhitened_both = self.wb.inv_t(rearrange(unwhitened_rows, "a b -> b a"))
        # X = L_A⁻ᵀ W′ L_B⁻¹
        X = rearrange(unwhitened_both, "b a -> a b")
        # x = vec(X)
        return rearrange(X, "a b -> (a b)")

    def logdet(self) -> Float[Array, ""]:
        """log|det K| from the factor eigenvalues and the anchor.

            log|det K| = ∑ log|γ| + N_2 log|det A_2| + N_1 log|det B_2|

        Returns:
            The log absolute determinant.
        """
        n_a, n_b = self.evals.shape
        # ∑ log|γ| + N_2 log|det A_2| + N_1 log|det B_2|
        return (
            jnp.sum(jnp.log(jnp.abs(self.evals)))
            + n_b * self.wa.logdet()
            + n_a * self.wb.logdet()
        )


def _unwrap_tagged(
    operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Strip any `lineax.TaggedLinearOperator` wrappers."""
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    return operator


def _scaled_identity(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, ""] | None:
    """The scalar c when an operator is *statically* c I, else ``None``.

    The recognition is by operator type, not by inspecting entries, so the
    answer is available at trace time; only the returned scalar is traced.

    Args:
        operator: Any lineax operator.

    Returns:
        The traced scalar c, or ``None``.
    """
    operator = _unwrap_tagged(operator)
    if isinstance(operator, lx.IdentityLinearOperator):
        # I = 1 · I
        return jnp.ones(())
    if isinstance(operator, lx.MulLinearOperator):
        inner = _scaled_identity(operator.operator)
        # (c I) · a = (c a) I
        return None if inner is None else inner * operator.scalar
    if isinstance(operator, lx.DivLinearOperator):
        inner = _scaled_identity(operator.operator)
        # (c I) / a = (c / a) I
        return None if inner is None else inner / operator.scalar
    if isinstance(operator, lx.NegLinearOperator):
        inner = _scaled_identity(operator.operator)
        # −(c I) = (−c) I
        return None if inner is None else -inner
    return None


def _whitener_kind(
    operator: lx.AbstractLinearOperator,
) -> type[_Whitener] | None:
    """Static classification of an anchor factor, or ``None`` if unusable.

    The tag queries run against the operator *as given*: a factor may carry
    its symmetry and positive-semidefiniteness on a native
    ``TaggedLinearOperator`` wrapper, and unwrapping first would throw away
    the only structural evidence there is. Only the ``isinstance`` checks
    look through the wrapper. Untagged operators fall through to the dense
    path rather than risk a ``NaN`` whitening.

    Whitening needs the anchor factor to be positive *definite*, and lineax
    has no tag for that; `_select_anchor` therefore prefers the kinds
    returned earlier here, which need no factorisation to invert.

    Args:
        operator: An anchor factor A_2 or B_2.

    Returns:
        The `_Whitener` subclass to build, or ``None``.
    """
    inner = _unwrap_tagged(operator)
    if isinstance(inner, lx.IdentityLinearOperator):
        return _IdentityWhitener
    if (
        isinstance(inner, lx.DiagonalLinearOperator)
        or _scaled_identity(operator) is not None
    ):
        return _DiagonalWhitener
    if lx.is_symmetric(operator) and lx.is_positive_semidefinite(operator):
        return _CholeskyWhitener
    return None


def _anchor_diagonal(
    operator: lx.AbstractLinearOperator,
) -> Float[Array, " n"]:
    """The diagonal of a diagonal-or-scaled-identity anchor factor.

    Args:
        operator: A diagonal or c I anchor factor.

    Returns:
        Its diagonal, c 1 for c I.
    """
    scalar = _scaled_identity(operator)
    if scalar is not None:
        dtype = operator.in_structure().dtype
        # diag(c I) = c 1
        return jnp.full((operator.in_size(),), scalar, dtype=dtype)
    return lx.diagonal(_unwrap_tagged(operator))


def _build_whitener(
    kind: type[_Whitener], operator: lx.AbstractLinearOperator
) -> _Whitener:
    """Build the whitener L, M = L Lᵀ, of one anchor factor M.

    Args:
        kind: The `_Whitener` subclass `_whitener_kind` chose.
        operator: The anchor factor M.

    Returns:
        L as identity, diagonal ℓ = √diag(M), or a dense Cholesky factor.
    """
    if kind is _IdentityWhitener:
        return _IdentityWhitener()
    if kind is _DiagonalWhitener:
        diagonal = _anchor_diagonal(operator)
        # Only a traced diagonal can reach here non-positive: a concrete one
        # is filtered out by `_select_anchor` (gh-317).
        diagonal = eqx.error_if(
            diagonal,
            ~jnp.all(diagonal > 0),
            "SumOfKroneckers: a diagonal or scaled-identity anchor factor must "
            "be strictly positive to whiten by its square root.",
        )
        # ℓ = √diag(M)
        return _DiagonalWhitener(jnp.sqrt(diagonal))
    # lazy import, cycle: _primitives._cholesky -> _operators._sum_kronecker
    from gaussx._primitives._cholesky import cholesky

    # ``operator`` unwrapped, so `cholesky` sees the tags and can dispatch.
    # M = L Lᵀ
    return _CholeskyWhitener(cholesky(operator).as_matrix())


def _add_leaves(
    operator: lx.AbstractLinearOperator,
) -> tuple[lx.AbstractLinearOperator, ...]:
    """Flatten a chain of lineax ``AddLinearOperator`` compositions.

    Args:
        operator: Any lineax operator.

    Returns:
        The summands, in order (the operator itself if it is not a sum).
    """
    operator = _unwrap_tagged(operator)
    if isinstance(operator, lx.AddLinearOperator):
        return (*_add_leaves(operator.operator1), *_add_leaves(operator.operator2))
    return (operator,)


def _kronecker_terms(
    operator: lx.AbstractLinearOperator,
) -> tuple[_KroneckerTerm, ...] | None:
    """Normalise an operator to a tuple of two-factor Kronecker terms.

        ∑_j A_j ⊗ B_j + ∑_m c_m I  ↦  ((A_1, B_1), …, (I, (∑_m c_m) I))

    Accepts a `SumOfKroneckers`, a `Kronecker`, or an ``AddLinearOperator``
    chain (the shape `sum_operator` produces) whose leaves are `Kronecker` /
    `SumOfKroneckers` operators plus any number of scalar-identity shifts.
    The identity shifts are summed and returned as a single I ⊗ c I term.

    Returns ``None`` when the operator is not of that form, or when the
    Kronecker factors are not square with sizes shared across terms; the
    caller then keeps its existing fallback.

    Args:
        operator: Any lineax operator.

    Returns:
        The ``((A_1, B_1), (A_2, B_2), …)`` factor pairs, or ``None``.
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
        # c = ∑_m c_m
        total = ft.reduce(operator_add, shifts)
        identity_a = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n_a,), dtype))
        identity_b = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n_b,), dtype))
        # I ⊗ c I
        terms.append((identity_a, total * identity_b))
    return tuple(terms)


# Whitener kinds in order of preference: the whitening has to *invert* the
# anchor, and only the first two are invertible on evidence available at
# trace time: an identity needs no inversion, a diagonal only a reciprocal.
# ``_CholeskyWhitener`` rests on ``positive_semidefinite_tag``, which permits
# a singular operator; whitening by one of those yields NaN where the dense
# fallback would have answered.
_ANCHOR_PREFERENCE = (_IdentityWhitener, _DiagonalWhitener, _CholeskyWhitener)


class _AnchorPlan(NamedTuple):
    """How to reduce by one of the two terms, decided at trace time.

    Attributes:
        index: Which term is the anchor.
        kinds: Whitener kind per axis, or ``None`` to fold the anchor in as
            a scalar shift instead of whitening by it.
    """

    index: int
    kinds: tuple[type[_Whitener], type[_Whitener]] | None

    def rank(self) -> int:
        """Lower is safer; see `_ANCHOR_PREFERENCE`."""
        if self.kinds is None:
            return 0
        return 1 + max(_ANCHOR_PREFERENCE.index(kind) for kind in self.kinds)


def _is_concretely_nonpositive(operator: lx.AbstractLinearOperator) -> bool:
    """Whether a diagonal anchor factor has a known non-positive entry.

    ``False`` for a traced diagonal: its values are unknown until run time.

    Args:
        operator: A diagonal or c I anchor factor.

    Returns:
        Whether a concrete diagonal entry is ≤ 0.
    """
    diagonal = _anchor_diagonal(operator)
    if isinstance(diagonal, Tracer):
        return False
    return bool(np.any(np.asarray(diagonal) <= 0))


def _select_anchor(
    terms: tuple[_KroneckerTerm, ...], *, check_values: bool = False
) -> _AnchorPlan | None:
    """Pick the term to reduce by, and how.

    Returns ``None`` when no exact two-term reduction applies. Where both
    terms qualify the safer plan wins, so a scalar or diagonal shift is
    never passed over for a tagged dense factor; ties go to the second
    term, since a noise or jitter shift is conventionally written last.

    With ``check_values=False`` this is a static decision. With
    ``check_values=True`` a diagonal whitening whose concrete diagonal has
    a non-positive entry is dropped too (gh-317), since its square root is
    ``NaN``; a traced diagonal is kept and checked at run time instead.

    Args:
        terms: The factor pairs from `_kronecker_terms`.
        check_values: Also drop a diagonal whitening with a concrete
            non-positive entry.

    Returns:
        The safest `_AnchorPlan`, or ``None``.
    """
    if len(terms) != 2:
        return None
    plans: list[_AnchorPlan] = []
    for anchor_index in (1, 0):
        main = terms[1 - anchor_index]
        anchor = terms[anchor_index]
        # eigh on the whitened main factors needs them symmetric.
        if not (lx.is_symmetric(main[0]) and lx.is_symmetric(main[1])):
            continue
        if all(_scaled_identity(factor) is not None for factor in anchor):
            # The whole anchor is c I: fold it into the eigenvalues rather
            # than inverting it, which keeps a negative c exact.
            plans.append(_AnchorPlan(anchor_index, None))
            continue
        kind_a = _whitener_kind(anchor[0])
        kind_b = _whitener_kind(anchor[1])
        if kind_a is None or kind_b is None:
            continue
        if check_values and any(
            kind is _DiagonalWhitener and _is_concretely_nonpositive(factor)
            for kind, factor in zip((kind_a, kind_b), anchor, strict=True)
        ):
            continue
        plans.append(_AnchorPlan(anchor_index, (kind_a, kind_b)))
    if not plans:
        return None
    # ``min`` is stable, so equally safe plans keep the term-1-first order.
    return min(plans, key=_AnchorPlan.rank)


def is_eigen_reducible(operator: lx.AbstractLinearOperator) -> bool:
    """Whether a sum of Kronecker products has the exact two-term reduction.

        K = A_1 ⊗ B_1 + A_2 ⊗ B_2,   A_1, B_1 symmetric,   A_2 ⊗ B_2 ≻ 0

    "Eigen-reducible" means the operator is A_1 ⊗ B_1 + A_2 ⊗ B_2 (a
    two-term `SumOfKroneckers`, or the same two terms built with
    `sum_operator`) whose main factors are symmetric and whose other
    ("anchor") term is positive definite. Then whitening by the anchor
    leaves one eigendecomposition per factor, so `gaussx.solve`,
    `gaussx.logdet` and `SumOfKroneckers.eigendecompose` cost
    O(N_1³ + N_2³) instead of the O((N_1 N_2)³) dense fallback. Three or
    more terms are never eigen-reducible.

    A purely structural query (operator types and lineax tags only, no
    array work), so it is safe to call from solver-strategy selection and
    under ``jax.jit``. The exact path can still decline an operator this
    accepts, when a diagonal anchor factor is concretely non-positive.

    Args:
        operator: Any lineax operator.

    Returns:
        ``True`` if the exact two-term reduction applies.

    Examples:
        A tagged Kronecker product plus a scalar shift reduces:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> I2 = lx.IdentityLinearOperator(A.in_structure())
        >>> K = gaussx.SumOfKroneckers(gaussx.Kronecker(A, A), gaussx.Kronecker(I2, I2))
        >>> gaussx.is_eigen_reducible(K)
        True

        Untagged factors do not:

        >>> plain = lx.MatrixLinearOperator(A.as_matrix())
        >>> K = gaussx.SumOfKroneckers(
        ...     gaussx.Kronecker(plain, plain), gaussx.Kronecker(plain, plain)
        ... )
        >>> gaussx.is_eigen_reducible(K)
        False
    """
    terms = _kronecker_terms(operator)
    return terms is not None and _select_anchor(terms) is not None


# The private name predates the export (gh-322); it was never public, so it
# stays a plain alias with no warning.
_is_eigen_reducible = is_eigen_reducible


def _sum_of_kroneckers_eigen(
    operator: lx.AbstractLinearOperator,
) -> _SumOfKroneckersEigen | FactoredEigen | None:
    """Factorise a two-term sum of Kronecker products, if one applies.

        A_1 ⊗ B_1 + A_2 ⊗ B_2 = (L_A ⊗ L_B) (Q_A ⊗ Q_B) diag(γ_A ⊗ γ_B + 1)
                                 (Q_A ⊗ Q_B)ᵀ (L_A ⊗ L_B)ᵀ

    Recognises A_1 ⊗ B_1 + A_2 ⊗ B_2 (written as a `SumOfKroneckers`, or as
    the ``AddLinearOperator`` chain `sum_operator` builds) with one term
    positive definite, and returns the simultaneous diagonalisation that
    `gaussx.solve` and `gaussx.logdet` dispatch on. Returns ``None`` for
    anything else, including three or more Kronecker terms, which have no
    closed form; the iterative alternatives for that case are
    ``solve(..., solver=lineax.CG(...))`` and `gaussx.SLQLogdet`, both of
    which drive the structured `SumOfKroneckers.mv`.

    Algorithm:
        1. Normalise to factor pairs and pick the anchor term.
        2. A c I anchor folds into the main term's eigenvalues as a shift.
        3. Otherwise whiten each main factor by the anchor's factor and
           eigendecompose the two whitened factors with ``eigh``.

    Args:
        operator: Any lineax operator.

    Returns:
        The factorisation, or ``None`` when no exact path applies.
    """
    terms = _kronecker_terms(operator)
    if terms is None:
        return None
    plan = _select_anchor(terms, check_values=True)
    if plan is None:
        return None
    main = terms[1 - plan.index]
    anchor = terms[plan.index]

    if plan.kinds is None:
        return _shifted_factorization(main, anchor)
    kind_a, kind_b = plan.kinds
    # A_2 = L_A L_Aᵀ
    wa = _build_whitener(kind_a, anchor[0])
    # B_2 = L_B L_Bᵀ
    wb = _build_whitener(kind_b, anchor[1])
    # Ã_1 = L_A⁻¹ A_1 L_A⁻ᵀ
    A1 = wa.conjugate(main[0].as_matrix())
    # B̃_1 = L_B⁻¹ B_1 L_B⁻ᵀ
    B1 = wb.conjugate(main[1].as_matrix())

    # Ã_1 = Q_A diag(γ_A) Q_Aᵀ
    evals_a, U = jnp.linalg.eigh(A1)
    # B̃_1 = Q_B diag(γ_B) Q_Bᵀ
    evals_b, V = jnp.linalg.eigh(B1)
    # γ = γ_A ⊗ γ_B + 1
    evals = einx.multiply("a, b -> a b", evals_a, evals_b) + jnp.ones(
        (), dtype=evals_a.dtype
    )
    return _SumOfKroneckersEigen(wa=wa, wb=wb, U=U, V=V, evals=evals)


def _shifted_factorization(
    main: _KroneckerTerm, anchor: _KroneckerTerm
) -> FactoredEigen:
    """A ⊗ B + c I in the joint eigenbasis of its factors (G3).

        (A ⊗ B + c I)⁻¹ = (Q_A ⊗ Q_B) (γ_A ⊗ γ_B + c)⁻¹ (Q_A ⊗ Q_B)ᵀ

    Each factor keeps an eigenbasis it already carries
    (`DiagonalizedOperator`, `KroneckerSum`, a diagonal), so a space-time
    AR(1) ⊗ grid + c I never forms the spatial factor; any other (symmetric)
    factor is decomposed densely with ``eigh``.

    Args:
        main: The main factor pair (A, B).
        anchor: The anchor pair (c_A I, c_B I).

    Returns:
        The `FactoredEigen` of A ⊗ B shifted by c = c_A c_B.
    """
    scalar_a = _scaled_identity(anchor[0])
    scalar_b = _scaled_identity(anchor[1])
    assert scalar_a is not None and scalar_b is not None
    # A ⊗ B = W⁻¹ diag(γ_A ⊗ γ_B) W
    factorization = kronecker_eigen(*main)
    # `_select_anchor` admits only symmetric main factors, which always have one.
    assert factorization is not None
    # γ_A ⊗ γ_B + c_A c_B
    return factorization.shifted(scalar_a * scalar_b)


def _shifted_kronecker_eigen(
    operator: lx.AbstractLinearOperator,
) -> FactoredEigen | None:
    """`_shifted_factorization` of an operator if it is A ⊗ B + c I.

    Args:
        operator: Any lineax operator.

    Returns:
        The shifted `FactoredEigen`, or ``None`` for any other operator.
    """
    terms = _kronecker_terms(operator)
    if terms is None:
        return None
    plan = _select_anchor(terms)
    if plan is None or plan.kinds is not None:
        return None
    return _shifted_factorization(terms[1 - plan.index], terms[plan.index])


def _sum_of_kroneckers_solve(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " n"],
) -> Float[Array, " n"] | None:
    """Solve K x = b by simultaneous diagonalisation, if one applies.

        x = K⁻¹ b,   K = A_1 ⊗ B_1 + A_2 ⊗ B_2

    The `gaussx.solve` entry point for the exact two-term reduction. Returns
    ``None`` when `_sum_of_kroneckers_eigen` has no exact path, so the
    caller keeps its fallback.

    Differentiation goes through `_eigen_solve`'s implicit rule rather than
    through ``eigh``; see there for why.

    Args:
        operator: Any lineax operator.
        vector: Right-hand side b.

    Returns:
        The solution x, or ``None`` when no exact path applies.
    """
    if not is_eigen_reducible(operator):
        return None
    return _eigen_solve(operator, vector)


@eqx.filter_custom_jvp
def _eigen_solve(
    operator: lx.AbstractLinearOperator,
    vector: Float[Array, " n"],
) -> Float[Array, " n"]:
    """x = K⁻¹ b through `_sum_of_kroneckers_eigen`, with an implicit JVP.

    Args:
        operator: An eigen-reducible operator K.
        vector: Right-hand side b.

    Returns:
        The solution x.
    """
    factorization = _sum_of_kroneckers_eigen(operator)
    if factorization is None:
        # Structurally reducible, but the only plan whitens by a concretely
        # non-positive diagonal (gh-317). The implicit JVP below does not
        # depend on how the primal was computed.
        # lazy import, cycle: _primitives._solve -> _operators._sum_kronecker
        from gaussx._primitives._solve import _solve_fallback

        return _solve_fallback(operator, vector, None)
    # x = K⁻¹ b
    return factorization.solve(vector)


@_eigen_solve.def_jvp
def _eigen_solve_jvp(primals, tangents):
    """Implicit JVP of x = K⁻¹ b, dx = K⁻¹ (db − dK x).

    Autodiff through the factor ``eigh`` calls is wrong in practice: the
    eigenvector derivative carries 1 / (γ_i − γ_j), which blows up on the
    repeated or clustered eigenvalues that multi-output GP factors have as a
    rule (a rank-deficient coregionalisation B = W Wᵀ, or a smooth kernel's
    Gram matrix with its tail of near-zero eigenvalues). The solution itself
    is invariant to how a degenerate eigenbasis is chosen, so the derivative
    written in terms of the operator alone is exact, and costs one more
    structured solve plus the tangent operator's matvec.

    Args:
        primals: The operator K and right-hand side b.
        tangents: Their tangents dK (possibly empty) and db (or ``None``).

    Returns:
        Tuple ``(x, dx)``.
    """
    operator, vector = primals
    t_operator, t_vector = tangents
    # x = K⁻¹ b
    x = _eigen_solve(operator, vector)
    # db, or 0
    rhs = jnp.zeros_like(vector) if t_vector is None else t_vector
    if jax.tree_util.tree_leaves(t_operator):
        # dK x
        _, dKx = eqx.filter_jvp(lambda op: op.mv(x), (operator,), (t_operator,))
        # db − dK x
        rhs = rhs - dKx
    # dx = K⁻¹ (db − dK x)
    return x, _eigen_solve(operator, rhs)


def sum_of_kroneckers_sample(
    op: SumOfKroneckers,
    *,
    key: jax.Array,
    num_samples: int = 1,
    lanczos_order: int = 50,
) -> Float[Array, "num_samples n"]:
    """Sample from 𝒩(0, K) with a matrix-free Lanczos square root.

        z = K^½ ε,   ε ~ 𝒩(0, I),   K = ∑_j A_j ⊗ B_j

    The square-root action is evaluated by ``matfree`` Lanczos against
    ``op.mv``. This avoids materialising the dense N × N covariance and
    costs ``lanczos_order`` operator matvecs per sample.

    Args:
        op: Positive-semidefinite `SumOfKroneckers` covariance K.
        key: PRNG key for the standard normal draws ε.
        num_samples: Number of independent samples to draw.
        lanczos_order: Lanczos truncation order.

    Returns:
        Samples of shape (num_samples, N).

    Raises:
        ValueError: If K is not square or ``num_samples`` is less than 1.

    Examples:
        Two draws from a 4-dimensional sum of Kronecker products:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 2.0]]), psd)
        >>> I2 = lx.MatrixLinearOperator(jnp.eye(2), psd)
        >>> K = gaussx.SumOfKroneckers(gaussx.Kronecker(A, A), gaussx.Kronecker(I2, I2))
        >>> key = jax.random.PRNGKey(0)
        >>> gaussx.sum_of_kroneckers_sample(K, key=key, num_samples=2).shape
        (2, 4)
    """
    # lazy import, cycle: _primitives._sqrt -> _operators._sum_kronecker
    from gaussx._primitives._sqrt import sqrt

    if op.in_size() != op.out_size():
        raise ValueError(
            "sum_of_kroneckers_sample requires a square SumOfKroneckers, got "
            f"in_size={op.in_size()} and out_size={op.out_size()}."
        )
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}.")

    # K^½, matrix-free
    sqrt_op = sqrt(op, lanczos_order=lanczos_order)
    # ε ~ 𝒩(0, I)
    eps = jax.random.normal(
        key, (num_samples, op.in_size()), dtype=op.in_structure().dtype
    )
    # z = K^½ ε
    return jax.vmap(sqrt_op.mv)(eps)


def sumkronecker_sample(
    op: SumOfKroneckers,
    *,
    key: jax.Array,
    num_samples: int = 1,
    lanczos_order: int = 50,
) -> Float[Array, "num_samples n"]:
    """Deprecated alias for `sum_of_kroneckers_sample` (gh-297).

    Will be removed in gaussx 0.7.0.

    Args:
        op: Positive-semidefinite `SumOfKroneckers` covariance K.
        key: PRNG key for the standard normal draws.
        num_samples: Number of independent samples to draw.
        lanczos_order: Lanczos truncation order.

    Returns:
        Samples of shape (num_samples, N).
    """
    warn_deprecated(
        "sumkronecker_sample is deprecated and will be removed in gaussx 0.7.0; "
        "use sum_of_kroneckers_sample."
    )
    return sum_of_kroneckers_sample(
        op, key=key, num_samples=num_samples, lanczos_order=lanczos_order
    )


# ---------------------------------------------------------------------------
# Deprecated compatibility class
# ---------------------------------------------------------------------------


class SumKronecker(SumOfKroneckers):
    """Deprecated alias for `SumOfKroneckers`.

    The old name was one word-order away from `KroneckerSum`, a different
    operator with a different eigendecomposition, and picking the wrong one
    produced silently wrong solves. Renamed in gh-136.

    Subclasses `SumOfKroneckers` so ``isinstance`` checks and
    ``singledispatch`` registrations keyed on this class keep working, and
    emits a `DeprecationWarning` on construction. Structural operations that
    rebuild the operator — ``transpose`` / ``.T`` — preserve this type, so
    they warn again. Note that instances built via `SumOfKroneckers` are
    *not* instances of this subclass; internal dispatch keys on the parent.
    Will be removed in gaussx 0.7.0.
    """

    def __init__(
        self,
        kron1: Kronecker,
        kron2: Kronecker,
        *krons: Kronecker,
        tags: object | frozenset[object] = frozenset(),
    ) -> None:
        warn_deprecated(
            "SumKronecker is deprecated and will be removed in gaussx 0.7.0; use "
            "SumOfKroneckers "
            "(KroneckerSum remains a different operator)."
        )
        super().__init__(kron1, kron2, *krons, tags=tags)
