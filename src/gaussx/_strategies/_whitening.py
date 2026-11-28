"""Whiten a system by a preconditioner: R⁻¹ A R⁻ᵀ with R Rᵀ = P.

A preconditioned stochastic log-determinant splits

    log det A = log det P + log det(R⁻¹ A R⁻ᵀ),   R Rᵀ = P,

and estimates only the second term, whose operator is near the identity
when P ≈ A. Any factor R gives the same log-determinant, so R need not be
the symmetric root P^½. Every preconditioner in gaussx gives P as a diagonal
or as a low-rank update of a diagonal, and for those R has a closed form
with an O(N k) inverse (gh-825); any other P falls back to the symmetric
root P^(−½) by contour quadrature.
"""

from __future__ import annotations

import functools as ft
from collections.abc import Callable

import einx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._operators import LowRankUpdate
from gaussx._operators._split import _summand_diagonal
from gaussx._primitives._sqrt_matmul import sqrt_inv_matmul


# The whitened operator is symmetric and PSD; Lanczos checks the first tag.
_PSD_TAGS = (lx.symmetric_tag, lx.positive_semidefinite_tag)


def whitened_operator(
    operator: lx.AbstractLinearOperator,
    approximation: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """R⁻¹ A R⁻ᵀ for a factor R Rᵀ = P, symmetric and PSD-tagged.

        log det(R⁻¹ A R⁻ᵀ) = log det A − log det P

    Args:
        operator: The symmetric PSD system A.
        approximation: The preconditioner's P ≈ A, from its
            ``approximation`` method.

    Returns:
        The whitened operator, matrix-free: one matvec of A and two O(N k)
        triangular-factor products per matvec for a diagonal or
        low-rank-plus-diagonal P, two contour-quadrature roots otherwise.
    """
    whitened, _ = whitening(operator, approximation)
    return whitened


def whitening(
    operator: lx.AbstractLinearOperator,
    approximation: lx.AbstractLinearOperator,
) -> tuple[lx.AbstractLinearOperator, Callable[[Array], Array]]:
    """R⁻¹ A R⁻ᵀ and the map u ↦ R⁻ᵀ u, for a factor R Rᵀ = P.

        z = R u,   P⁻¹ z = R⁻ᵀ u

    A probe u in the whitened space is the probe z = R u of A, with
    𝔼[z zᵀ] = P for an isotropic u, and R⁻ᵀ u = P⁻¹ z is what the
    log-determinant's control variate pairs with it (gh-826).

    Args:
        operator: The symmetric PSD system A.
        approximation: The preconditioner's P ≈ A.

    Returns:
        Tuple ``(whitened, inverse_transpose)``: R⁻¹ A R⁻ᵀ as a PSD-tagged
        operator, and the function u ↦ R⁻ᵀ u.
    """
    inner = approximation
    while isinstance(inner, lx.TaggedLinearOperator):
        inner = inner.operator
    diagonal = _summand_diagonal(inner)
    if diagonal is not None:
        # R = diag(√a): a diagonal P
        return _closed_form_whitening(operator, diagonal, None, None)
    if isinstance(inner, LowRankUpdate) and inner.symmetric_factors:
        base_diagonal = _summand_diagonal(inner.base)
        if base_diagonal is not None:
            weights = inner.d
            if weights is None:
                weights = jnp.ones(inner.U.shape[1], dtype=inner.U.dtype)
            return _closed_form_whitening(operator, base_diagonal, inner.U, weights)
    return contour_whitened_operator(operator, approximation), ft.partial(
        _contour_inverse_root, approximation
    )


def _contour_inverse_root(
    approximation: lx.AbstractLinearOperator, vector: Float[Array, " n"]
) -> Float[Array, " n"]:
    """P^(−½) v by contour quadrature, R⁻ᵀ v for the symmetric root R = P^½.

    Args:
        approximation: P.
        vector: v.

    Returns:
        P^(−½) v.
    """
    column = rearrange(vector, "row -> row 1")
    # P^(−½) v
    return rearrange(sqrt_inv_matmul(approximation, column), "row 1 -> row")


def _closed_form_whitening(
    operator: lx.AbstractLinearOperator,
    base_diagonal: Float[Array, " n"],
    factor: Float[Array, "n k"] | None,
    weights: Float[Array, " k"] | None,
) -> tuple[lx.AbstractLinearOperator, Callable[[Array], Array]]:
    """R⁻¹ A R⁻ᵀ for P = diag(a) + U S Uᵀ, with R = diag(√a) R_C.

        P = diag(√a) (I + V S Vᵀ) diag(√a),   V = diag(a)^(−½) U = Q R_V,
        I + V S Vᵀ = R_C R_Cᵀ,   R_C = I + Q (L − I) Qᵀ,
        L Lᵀ = I_k + R_V S R_Vᵀ,   R_C⁻¹ = I + Q (L⁻¹ − I) Qᵀ

    The cross terms of R_C R_Cᵀ vanish because Qᵀ (I − Q Qᵀ) = 0. Only a
    thin QR and a k × k Cholesky factor are formed, with no eigenvalue
    gaps to differentiate through.

    Args:
        operator: The system A.
        base_diagonal: The diagonal a of P's base.
        factor: U, or ``None`` for a diagonal P.
        weights: S's diagonal, or ``None`` for a diagonal P.

    Returns:
        Tuple ``(whitened, inverse_transpose)``: the PSD-tagged R⁻¹ A R⁻ᵀ
        and u ↦ R⁻ᵀ u.
    """
    # diag(a)^(−½)
    inverse_root = 1.0 / jnp.sqrt(base_diagonal)
    structure = operator.in_structure()
    if factor is None:

        def diagonal_inverse_transpose(
            vector: Float[Array, " n"],
        ) -> Float[Array, " n"]:
            # R⁻ᵀ v = diag(a)^(−½) v
            return inverse_root * vector

        def diagonal_matvec(vector: Float[Array, " n"]) -> Float[Array, " n"]:
            # diag(a)^(−½) A diag(a)^(−½) v
            return inverse_root * operator.mv(inverse_root * vector)

        whitened = lx.FunctionLinearOperator(diagonal_matvec, structure, _PSD_TAGS)
        return whitened, diagonal_inverse_transpose

    # V = diag(a)^(−½) U
    scaled_factor = einx.multiply("row rank, row -> row rank", factor, inverse_root)
    # V = Q R_V
    basis, triangle = jnp.linalg.qr(scaled_factor)
    # R_V S
    weighted = einx.multiply("row rank, rank -> row rank", triangle, weights)
    # I_k + R_V S R_Vᵀ
    capacitance = jnp.eye(triangle.shape[0], dtype=triangle.dtype) + einsum(
        weighted, triangle, "row rank, col rank -> row col"
    )
    # L Lᵀ = I_k + R_V S R_Vᵀ, symmetrised against round-off
    lower = jnp.linalg.cholesky(
        0.5 * (capacitance + rearrange(capacitance, "row col -> col row"))
    )

    def apply_factor_inverse(
        vector: Float[Array, " n"], transpose: bool
    ) -> Float[Array, " n"]:
        # Qᵀ v
        coefficients = einsum(basis, vector, "row rank, row -> rank")
        # L⁻¹ Qᵀ v, or L⁻ᵀ Qᵀ v
        solved = jax.scipy.linalg.solve_triangular(
            lower, coefficients, lower=True, trans=1 if transpose else 0
        )
        # R_C⁻¹ v = v + Q (L⁻¹ − I) Qᵀ v
        return vector + einsum(basis, solved - coefficients, "row rank, rank -> row")

    def inverse_transpose(vector: Float[Array, " n"]) -> Float[Array, " n"]:
        # R⁻ᵀ v = diag(a)^(−½) R_C⁻ᵀ v
        return inverse_root * apply_factor_inverse(vector, transpose=True)

    def matvec(vector: Float[Array, " n"]) -> Float[Array, " n"]:
        # R⁻¹ A R⁻ᵀ v = R_C⁻¹ diag(a)^(−½) A R⁻ᵀ v
        applied = operator.mv(inverse_transpose(vector))
        return apply_factor_inverse(inverse_root * applied, transpose=False)

    whitened = lx.FunctionLinearOperator(matvec, structure, _PSD_TAGS)
    return whitened, inverse_transpose


def contour_whitened_operator(
    operator: lx.AbstractLinearOperator,
    preconditioner: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    """Build the symmetric whitened operator P^(−1/2) A P^(−1/2).

        P^(−1/2) A P^(−1/2) ~ P⁻¹ A

    It is similar to P⁻¹ A, so it has the same log-determinant while staying
    symmetric enough for Lanczos. Each matvec costs two contour-quadrature
    square roots (`gaussx.sqrt_inv_matmul`), so this is worthwhile only when
    P is cheap to solve against; it is the fallback for a P with no
    closed-form factor.

    Args:
        operator: The operator A.
        preconditioner: The preconditioner P.

    Returns:
        A matrix-free PSD-tagged operator P^(−1/2) A P^(−1/2).
    """

    def matvec(vector: Float[Array, " N"]) -> Float[Array, " N"]:
        column = rearrange(vector, "row -> row 1")
        # P^(−1/2) v
        whitened = sqrt_inv_matmul(preconditioner, column)
        # A P^(−1/2) v
        applied = operator.mv(rearrange(whitened, "row 1 -> row"))
        # P^(−1/2) A P^(−1/2) v
        result = sqrt_inv_matmul(preconditioner, rearrange(applied, "row -> row 1"))
        return rearrange(result, "row 1 -> row")

    return lx.FunctionLinearOperator(matvec, operator.in_structure(), _PSD_TAGS)
