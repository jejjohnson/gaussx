"""Pivoted partial-Cholesky preconditioner."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._preconditioners._base import AbstractPreconditioner
from gaussx._randomized._rpcholesky import rp_cholesky


class PartialCholeskyPreconditioner(AbstractPreconditioner):
    r"""Preconditioner from a pivoted partial Cholesky factor.

    For a system ``A = K + σ² I`` it builds a rank-``k`` partial Cholesky
    factor ``F`` of the PSD part, ``F Fᵀ ≈ K``, and applies
    ``(σ² I + F Fᵀ)⁻¹`` through the Woodbury identity. At full rank this is
    exactly ``A⁻¹``. Pivots are greedy (``argmax`` of the residual diagonal)
    or random (`gaussx.rp_cholesky`, proportional to it).

    Two ways to build it:

    - **Once**, with `from_operator` on the PSD part ``K`` and an explicit
      ``shift=σ²``. The factor and the Woodbury capacitance are stored as
      arrays, so `as_operator` ignores its argument and repeated solves cost
      no further matvecs of ``K`` (#371).
    - **Lazily**, as ``PartialCholeskyPreconditioner(rank, shift)``. Then
      `as_operator` receives the *system* operator ``A`` at every solve and
      factors ``A − shift · I`` implicitly, so ``shift`` is the part of
      ``A``'s diagonal treated as noise and the noise is never counted twice
      (#345). ``shift`` must not exceed the ``σ²`` actually in ``A``.

    The factor is guarded like LAPACK ``?pstrf``: once ``rank`` exceeds the
    numerical rank (small datasets, duplicated inputs, noiseless kernels),
    the surplus columns are exactly zero instead of NaN or inf, and the
    preconditioner degrades gracefully to the lower-rank one (gh-237).

    Covariance form only: the factor captures the *top* of ``K``'s spectrum,
    which is right for ``K + σ² I``. It is not a good preconditioner for a
    precision-form system ``Q + Aᵀ W A``, whose hard directions are its
    smallest eigenvalues.

    Attributes:
        rank: Rank of the partial Cholesky. ``<= 0`` disables the lazy
            preconditioner (`as_operator` returns ``None``).
        shift: Noise variance ``σ²`` for the lazy path: the part of the
            system's diagonal treated as noise.
        pivoting: ``"greedy"`` (default) or ``"random"``.
        key: PRNG key for ``"random"`` pivoting. ``None`` means
            ``jax.random.PRNGKey(0)``.
        factor: Stored factor ``F``, shape ``(n, k)``; set by `from_operator`.
        capacitance: Lower Cholesky factor of ``σ² I + Fᵀ F``, shape
            ``(k, k)``; set by `from_operator`.
        factor_shift: The ``σ²`` the stored factor was built with; set by
            `from_operator`.

    Examples:
        Build once on the PSD part, then reuse across solves.

        >>> import einx, jax.numpy as jnp, lineax as lx, gaussx
        >>> x = jnp.linspace(0.0, 10.0, 50)
        >>> K = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2)
        >>> psd = lx.positive_semidefinite_tag
        >>> P = gaussx.PartialCholeskyPreconditioner.from_operator(
        ...     lx.MatrixLinearOperator(K, psd), rank=20, shift=0.1
        ... )
        >>> A = lx.MatrixLinearOperator(K + 0.1 * jnp.eye(50), psd)
        >>> solver = gaussx.CGSolver(preconditioner=P)
        >>> x1 = solver.solve(A, jnp.ones(50))
        >>> x2 = solver.solve(A, jnp.arange(50.0))  # no rebuild
    """

    rank: int = eqx.field(static=True, default=50)
    shift: float = eqx.field(static=True, default=1.0)
    pivoting: Literal["greedy", "random"] = eqx.field(static=True, default="greedy")
    key: jax.Array | None = None
    factor: Float[Array, "n k"] | None = None
    capacitance: Float[Array, "k k"] | None = None
    factor_shift: Float[Array, ""] | None = None

    @classmethod
    def from_operator(
        cls,
        operator: lx.AbstractLinearOperator,
        rank: int = 50,
        *,
        shift: float | Float[Array, ""],
        pivoting: Literal["greedy", "random"] = "greedy",
        key: jax.Array | None = None,
    ) -> PartialCholeskyPreconditioner:
        """Factor the PSD part once and store the preconditioner.

        Args:
            operator: The PSD part ``K`` (e.g. a kernel matrix), **not**
                ``K + σ² I``.
            rank: Number of pivots, clamped to the operator size.
            shift: The noise variance ``σ²``; the preconditioner approximates
                ``(K + σ² I)⁻¹``. Must be positive.
            pivoting: ``"greedy"`` or ``"random"`` (see `gaussx.rp_cholesky`).
            key: PRNG key for ``"random"`` pivoting. ``None`` means
                ``jax.random.PRNGKey(0)``.

        Returns:
            A built `PartialCholeskyPreconditioner` whose `as_operator`
            ignores its argument.

        Raises:
            ValueError: If ``rank < 1``.
        """
        if rank < 1:
            raise ValueError("rank must be at least 1")
        rank = min(rank, operator.in_size())
        factor = _factor(lx.diagonal(operator), operator, rank, 0.0, pivoting, key)
        shift = jnp.asarray(shift, dtype=factor.dtype)
        return cls(
            rank=rank,
            pivoting=pivoting,
            key=key,
            factor=factor,
            capacitance=_capacitance(factor, shift),
            factor_shift=shift,
        )

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator | None:
        """Return ``(σ² I + F Fᵀ)⁻¹`` as a PSD operator.

        A preconditioner built by `from_operator` ignores *operator*. The lazy
        one factors ``operator − shift · I``, where *operator* is the system.
        """
        if self.factor is not None:
            assert self.capacitance is not None and self.factor_shift is not None
            structure = jax.ShapeDtypeStruct((self.factor.shape[0],), self.factor.dtype)
            return _woodbury(
                self.factor, self.capacitance, self.factor_shift, structure
            )
        if self.rank <= 0:
            return None
        if operator is None:
            raise ValueError(
                "PartialCholeskyPreconditioner.as_operator requires the system "
                "operator to build its factor (or build it once with from_operator)."
            )

        rank = min(self.rank, operator.in_size())
        dtype = operator.in_structure().dtype
        shift = jnp.asarray(self.shift, dtype=dtype)
        # Factor K = A − σ²I, never A itself: the Woodbury step adds σ² back
        # (#345).
        diagonal = lx.diagonal(operator) - shift
        factor = _factor(diagonal, operator, rank, shift, self.pivoting, self.key)
        return _woodbury(
            factor, _capacitance(factor, shift), shift, operator.out_structure()
        )


def _factor(
    diagonal: Float[Array, " n"],
    operator: lx.AbstractLinearOperator,
    rank: int,
    shift: float | Float[Array, ""],
    pivoting: Literal["greedy", "random"],
    key: jax.Array | None,
) -> Float[Array, "n k"]:
    """Partial Cholesky of ``operator − shift · I``, one matvec per pivot."""
    n = operator.in_size()

    def column(k):
        e_k = jnp.zeros(n, dtype=diagonal.dtype).at[k].set(1.0)
        return operator.mv(e_k) - shift * e_k

    factor, _ = rp_cholesky(diagonal, column, rank, pivoting=pivoting, key=key)
    return factor


def _capacitance(
    factor: Float[Array, "n k"], shift: Float[Array, ""]
) -> Float[Array, "k k"]:
    """Lower Cholesky factor of ``σ² I + Fᵀ F``.

    Zero surplus columns only add ``σ²`` to the diagonal, so it stays
    positive definite.
    """
    eye = jnp.eye(factor.shape[1], dtype=factor.dtype)
    gram = einsum(factor, factor, "n i, n j -> i j")
    return jnp.linalg.cholesky(shift * eye + gram)


def _woodbury(
    factor: Float[Array, "n k"],
    capacitance: Float[Array, "k k"],
    shift: Float[Array, ""],
    structure: jax.ShapeDtypeStruct,
) -> lx.AbstractLinearOperator:
    """``(σ² I + F Fᵀ)⁻¹ v = (v − F (σ² I + Fᵀ F)⁻¹ Fᵀ v) / σ²``."""

    def matvec(v: Float[Array, " n"]) -> Float[Array, " n"]:
        coeffs = jax.scipy.linalg.cho_solve(
            (capacitance, True), einsum(factor, v, "n k, n -> k")
        )
        return (v - factor @ coeffs) / shift

    return lx.FunctionLinearOperator(matvec, structure, lx.positive_semidefinite_tag)
