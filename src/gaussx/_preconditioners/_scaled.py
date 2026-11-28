"""Per-point shifts by symmetric scaling: a preconditioner of A = K + diag(d).

Scaling the system by D^(−½), D = diag(d), gives a system with a unit shift,

    B = D^(−½) A D^(−½) = K_B + I,   K_B = D^(−½) K D^(−½),

so any preconditioner P_B of a scalar shift of one serves a per-point shift:

    P = D^½ P_B D^½,   P⁻¹ = D^(−½) P_B⁻¹ D^(−½),
    log det P = log det P_B + ∑_n log d_n.

`_ScaledPreconditioner` holds P_B and d and gives P to the partial Cholesky
and Nyström preconditioners and to `gaussx.NystromLogdet` (gh-878).
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._operators import LowRankUpdate, low_rank_plus_diag
from gaussx._operators._split import DiagonalSplit, _summand_diagonal, split_diagonal
from gaussx._preconditioners._base import AbstractPreconditioner
from gaussx._primitives._diag import matrix_free_diag


def checked_shift(
    shift: Float[Array, " n"],
    owner: str,
) -> Float[Array, " n"]:
    """The per-point shift d, with an error under ``jit`` if any d_n ≤ 0.

    Args:
        shift: The per-point shift d.
        owner: The class that takes it, named in the error.

    Returns:
        ``shift`` unchanged.
    """
    return eqx.error_if(
        shift,
        jnp.any(shift <= 0.0),
        f"{owner}: every entry of the per-point `shift` d must be positive; "
        "a noise-free point (d_n = 0) has no scaled form.",
    )


def system_split(
    operator: lx.AbstractLinearOperator,
    owner: str,
) -> DiagonalSplit:
    """A = K + diag(d), read back by `gaussx.split_diagonal`, for ``shift=None``.

    Args:
        operator: The system A.
        owner: The class that reads it, named in the error.

    Returns:
        The split, with d checked positive.

    Raises:
        ValueError: If the system is not a kernel part plus a diagonal.
    """
    split = split_diagonal(operator)
    if split is None:
        raise ValueError(
            f"{owner}(shift=None) reads the per-point shift d from the system "
            "with gaussx.split_diagonal, which finds no kernel part plus a "
            f"diagonal in {type(operator).__name__}. Pass shift=d explicitly."
        )
    return DiagonalSplit(
        base=split.base, diagonal=checked_shift(split.diagonal, owner)
    )


def resolve_shift(
    shift: Float[Array, " n"] | None,
    operator: lx.AbstractLinearOperator,
    owner: str,
) -> Float[Array, " n"]:
    """The per-point shift d: ``shift`` itself, or read from the system.

    Args:
        shift: A vector d, or ``None`` to read it with `gaussx.split_diagonal`.
        operator: The system A.
        owner: The class that takes it, named in the errors.

    Returns:
        d in the system's dtype, checked positive.

    Raises:
        ValueError: If ``shift`` is ``None`` and the system does not split.
    """
    if shift is None:
        return system_split(operator, owner).diagonal
    dtype = operator.in_structure().dtype
    return checked_shift(jnp.asarray(shift, dtype=dtype), owner)


def scaled_operator(
    operator: lx.AbstractLinearOperator,
    shift: Float[Array, " n"],
) -> lx.AbstractLinearOperator:
    """D^(−½) A D^(−½), matrix-free and PSD-tagged.

    Args:
        operator: A symmetric PSD operator A (the system, or its kernel part).
        shift: The per-point shift d > 0.

    Returns:
        The scaled operator, one matvec of A and two products per matvec.
    """
    # D^(−½)
    inverse_root = 1.0 / jnp.sqrt(shift)

    def matvec(vector: Float[Array, " n"]) -> Float[Array, " n"]:
        # D^(−½) A D^(−½) v
        return inverse_root * operator.mv(inverse_root * vector)

    return lx.FunctionLinearOperator(
        matvec,
        operator.in_structure(),
        (lx.symmetric_tag, lx.positive_semidefinite_tag),
    )


def scaled_diagonal(
    diagonal: Float[Array, " n"] | None,
    operator: lx.AbstractLinearOperator,
    shift: Float[Array, " n"],
    exact: bool,
) -> Float[Array, " n"] | None:
    """diag(D^(−½) A D^(−½)) = diag(A) / d, when diag(A) is known exactly.

    Args:
        diagonal: diag(A) if the caller gave it, else ``None``.
        operator: The operator A.
        shift: The per-point shift d.
        exact: Whether A's diagonal is cheap and exact.

    Returns:
        diag(A) / d, or ``None`` when diag(A) must be estimated (the scaled
        operator is then probed instead).
    """
    if diagonal is None and exact:
        diagonal = matrix_free_diag(operator, estimate=False)
    if diagonal is None:
        return None
    # diag(A) / d
    return diagonal / shift


def _scaled_approximation(
    approximation: lx.AbstractLinearOperator,
    root: Float[Array, " n"],
) -> lx.AbstractLinearOperator:
    """P = D^½ P_B D^½ for a diagonal or low-rank-plus-diagonal P_B.

        D^½ (A₀ + U S Uᵀ) D^½ = D^½ A₀ D^½ + (D^½ U) S (D^½ U)ᵀ

    Args:
        approximation: P_B, a (tagged) diagonal operator or a symmetric
            `gaussx.LowRankUpdate` on a diagonal base.
        root: D^½.

    Returns:
        P, a PSD-tagged diagonal or `gaussx.LowRankUpdate`.

    Raises:
        NotImplementedError: For any other P_B.
    """
    inner = approximation
    while isinstance(inner, lx.TaggedLinearOperator):
        inner = inner.operator
    variance = root * root
    if isinstance(inner, LowRankUpdate) and inner.symmetric_factors:
        base_diagonal = _summand_diagonal(inner.base)
        if base_diagonal is not None:
            # D^½ U
            factor = einx.multiply("row rank, row -> row rank", inner.U, root)
            # D A₀ + (D^½ U) S (D^½ U)ᵀ
            return low_rank_plus_diag(
                variance * base_diagonal, factor, inner.d, psd=True
            )
    diagonal = _summand_diagonal(inner)
    if diagonal is not None:
        # D^½ diag(p) D^½ = diag(d ⊙ p)
        return lx.TaggedLinearOperator(
            lx.DiagonalLinearOperator(variance * diagonal),
            lx.positive_semidefinite_tag,
        )
    raise NotImplementedError(
        f"Cannot scale a preconditioner approximation {type(inner).__name__}."
    )


class _ScaledPreconditioner(AbstractPreconditioner):
    """P = D^½ P_B D^½, from a preconditioner P_B of B = D^(−½) A D^(−½).

        P⁻¹ = D^(−½) P_B⁻¹ D^(−½),   log det P = log det P_B + ∑_n log d_n

    ``inner`` is built (or lazily builds itself) for the unit-shift system B,
    and receives B wherever the protocol passes the system.

    Attributes:
        inner: The preconditioner P_B of B = K_B + I.
        shift: The per-point shift d > 0.
    """

    inner: AbstractPreconditioner
    shift: Float[Array, " n"]

    def _scaled_system(
        self, operator: lx.AbstractLinearOperator | None
    ) -> lx.AbstractLinearOperator | None:
        """B = D^(−½) A D^(−½), or ``None`` without a system.

        Args:
            operator: The system A, or ``None``.

        Returns:
            B, or ``None``.
        """
        if operator is None:
            return None
        return scaled_operator(operator, self.shift)

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """P⁻¹ = D^(−½) P_B⁻¹ D^(−½) as a PSD operator.

        Args:
            operator: The system A, passed to ``inner`` as B.

        Returns:
            The matrix-free P⁻¹, tagged positive semidefinite.
        """
        inner_inverse = self.inner.as_operator(self._scaled_system(operator))
        # D^(−½)
        inverse_root = 1.0 / jnp.sqrt(self.shift)
        structure = jax.ShapeDtypeStruct(self.shift.shape, self.shift.dtype)

        def matvec(vector: Float[Array, " n"]) -> Float[Array, " n"]:
            # D^(−½) v
            scaled = inverse_root * vector
            if inner_inverse is not None:
                # P_B⁻¹ D^(−½) v
                scaled = inner_inverse.mv(scaled)
            # D^(−½) P_B⁻¹ D^(−½) v
            return inverse_root * scaled

        return lx.FunctionLinearOperator(
            matvec, structure, lx.positive_semidefinite_tag
        )

    def supports_slq(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> bool:
        """Whether ``inner`` knows P_B, and so P.

        Args:
            operator: The system A; unused.

        Returns:
            ``inner.supports_slq()``.
        """
        del operator
        return self.inner.supports_slq()

    def logdet(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> Float[Array, ""]:
        """log det P = log det P_B + ∑_n log d_n.

        Args:
            operator: The system A, passed to ``inner`` as B.

        Returns:
            The scalar log det P.
        """
        inner_logdet = self.inner.logdet(self._scaled_system(operator))
        # log det P_B + ∑_n log d_n
        return inner_logdet + jnp.sum(jnp.log(self.shift))

    def sample(
        self,
        key: PRNGKeyArray,
        operator: lx.AbstractLinearOperator | None = None,
        *,
        shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch n"]:
        """Draw z = D^½ z_B ~ 𝒩(0, P), with z_B ~ 𝒩(0, P_B).

        Args:
            key: PRNG key for z_B.
            operator: The system A, passed to ``inner`` as B.
            shape: Batch shape of the draws.

        Returns:
            The draws z, of shape ``(*shape, n)``.
        """
        inner_draws = self.inner.sample(
            key, self._scaled_system(operator), shape=shape
        )
        # D^½ z_B
        return einx.multiply(
            "... row, row -> ... row", inner_draws, jnp.sqrt(self.shift)
        )

    def approximation(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """P = D^½ P_B D^½, a PSD diagonal or low-rank-plus-diagonal operator.

        Args:
            operator: The system A, passed to ``inner`` as B.

        Returns:
            P, with a structured solve and log-determinant.
        """
        inner_approximation = self.inner.approximation(self._scaled_system(operator))
        return _scaled_approximation(inner_approximation, jnp.sqrt(self.shift))
