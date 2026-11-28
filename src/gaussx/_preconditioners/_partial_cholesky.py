"""Pivoted partial-Cholesky preconditioner.

A = K + σ² I is an N × N symmetric positive-definite system with a PSD part K
and a noise variance σ² (the ``shift``). A pivoted rank-k partial Cholesky
factor L_k (N × k) with K ≈ L_k L_kᵀ gives the preconditioner
M = σ² I + L_k L_kᵀ ≈ A, applied by the Woodbury identity

    M⁻¹ v = σ⁻² (v − L_k C⁻¹ L_kᵀ v),   C = σ² I_k + L_kᵀ L_k,

with the k × k capacitance C held as its lower Cholesky factor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float, Int, PRNGKeyArray

from gaussx._einx import einsum
from gaussx._operators import BlockDiag, LowRankUpdate, low_rank_plus_diag
from gaussx._preconditioners._base import AbstractPreconditioner, batched_draws
from gaussx._preconditioners._scaled import (
    _ScaledPreconditioner,
    checked_shift,
    resolve_shift,
    scaled_diagonal,
    scaled_operator,
)
from gaussx._primitives._diag import _CHEAP_DIAGONAL, matrix_free_diag
from gaussx._primitives._gaussian_draws import diagonal_draws, low_rank_update_draws
from gaussx._primitives._logdet import cholesky_logdet
from gaussx._randomized._rpcholesky import _pivoted_cholesky


# Hutchinson probes for the diagonal of a matrix-free operator (gh-361). The
# diagonal only selects pivots; the factor columns are exact either way.
_DIAGONAL_PROBES = 20


class PartialCholeskyPreconditioner(AbstractPreconditioner):
    """Preconditioner from a pivoted partial Cholesky factor.

        M = σ² I + L_k L_kᵀ ≈ A = K + σ² I,
        M⁻¹ v = σ⁻² (v − L_k (σ² I_k + L_kᵀ L_k)⁻¹ L_kᵀ v)

    For a system A = K + σ² I it builds a rank-k pivoted partial Cholesky
    factor L_k of the PSD part, K ≈ L_k L_kᵀ, and applies M⁻¹ through the
    Woodbury identity. At full rank this is exactly A⁻¹. Pivots are greedy
    (``argmax`` of the residual diagonal) or random (`gaussx.rp_cholesky`,
    proportional to it).

    Two ways to build it:

    - **Once**, with `from_operator` on the PSD part K and an explicit
      ``shift=σ²``. The factor and the Woodbury capacitance are stored as
      arrays, so `as_operator` ignores its argument and repeated solves cost
      no further matvecs of K (#371).
    - **Lazily**, as ``PartialCholeskyPreconditioner(rank, shift)``. Then
      `as_operator` receives the *system* operator A at every solve and
      factors K = A − σ² I implicitly, so ``shift`` is the part of A's
      diagonal treated as noise and the noise is never counted twice
      (#345). ``shift`` must not exceed the σ² actually in A.

    The factor is guarded like LAPACK ``?pstrf``: once ``rank`` exceeds the
    numerical rank (small datasets, duplicated inputs, noiseless kernels),
    the surplus columns are exactly zero instead of NaN or inf, and the
    preconditioner degrades gracefully to the lower-rank one (gh-237).

    **Cost.** A build applies the operator to ``rank`` pivot columns (or
    calls ``column``, see `from_operator`) and needs its diagonal. A given
    ``diagonal``, a stored matrix and the structured operators
    (`gaussx.diag`) cost nothing more. For a matrix-free part, such as a bare
    `lineax.FunctionLinearOperator`, the diagonal is a Hutchinson estimate
    from 20 matvecs: it only selects the pivots, and each factor column is
    read exactly from its pivot column, so the operator is never
    materialised (O(N k) memory, gh-361).

    Covariance form only: the factor captures the *top* of K's spectrum,
    which is right for K + σ² I. It is not a good preconditioner for a
    precision-form system Q + Aᵀ W A, whose hard directions are its
    smallest eigenvalues.

    **As P itself.** `approximation`, `logdet` and `sample` use
    M = σ² I + L_k L_kᵀ, built once per call from the same factor as
    `as_operator`:

        log det M = N log σ² + log det(I_k + L_kᵀ L_k / σ²),
        z = σ ε_N + L_k ε_k ~ 𝒩(0, M),

    the first by the matrix determinant lemma from the stored capacitance
    factor. A disabled lazy preconditioner (``rank <= 0``) is M = I.

    **Per-point shifts.** A system A = K + diag(d) whose diagonal varies by
    point (the site variances d_n = 1 / λ_n of a non-Gaussian GP, or
    heteroscedastic noise) takes a vector ``shift=d``, or ``shift=None`` on
    the lazy path to read d from the system with `gaussx.split_diagonal`.
    The system is scaled to B = D^(−½) A D^(−½) = K_B + I, D = diag(d), and
    the scalar-shift preconditioner of B with σ² = 1 is pulled back:

        M = D^½ (I + L_B L_Bᵀ) D^½,   M⁻¹ = D^(−½) M_B⁻¹ D^(−½),
        log det M = log det M_B + ∑_n log d_n,

    with K_B ≈ L_B L_Bᵀ. The factor captures the top of K_B, the kernel
    weighted by how strongly each point is observed, instead of the top of
    K, so it is a much stronger preconditioner than any scalar σ² ≤ min d.
    Every d_n must be positive (``eqx.error_if``); a masked site
    (λ_n = 0, d_n = ∞) is out of scope. A scalar ``shift`` keeps the code
    path above unchanged.

    Attributes:
        rank: Rank k of the partial Cholesky. ``<= 0`` disables the lazy
            preconditioner (`as_operator` returns ``None``).
        shift: Noise variance σ² for the lazy path: the part of the
            system's diagonal treated as noise. A vector d is a per-point
            shift, and ``None`` reads d from the system with
            `gaussx.split_diagonal` (see **Per-point shifts**). A PyTree
            leaf, so it can be traced.
        pivoting: ``"greedy"`` (default) or ``"random"``.
        key: PRNG key for ``"random"`` pivoting and for the diagonal
            estimate of a matrix-free operator. ``None`` means
            ``jax.random.PRNGKey(0)``.
        diagonal: The diagonal of the *system* operator A for the lazy path.
            ``None`` takes it from the operator (exact unless it is
            matrix-free; see **Cost**).
        factor: Stored factor L_k, N × k; set by `from_operator`.
        capacitance: Lower Cholesky factor of C = σ² I_k + L_kᵀ L_k,
            k × k; set by `from_operator`.
        factor_shift: The σ² the stored factor was built with, or the
            per-point d of a scaled factor (of K_B, with a unit shift); set
            by `from_operator`.

    Examples:
        Build once on the PSD part, then reuse across solves:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> inputs = jnp.linspace(0.0, 10.0, 50)
        >>> gaps = einx.subtract("row, col -> row col", inputs, inputs)
        >>> kernel = jnp.exp(-0.5 * gaps**2)
        >>> psd = lx.positive_semidefinite_tag
        >>> preconditioner = gaussx.PartialCholeskyPreconditioner.from_operator(
        ...     lx.MatrixLinearOperator(kernel, psd), rank=20, shift=0.1
        ... )
        >>> preconditioner.factor.shape
        (50, 20)
        >>> A = lx.MatrixLinearOperator(kernel + 0.1 * jnp.eye(50), psd)
        >>> solver = gaussx.CGSolver(preconditioner=preconditioner)
        >>> x1 = solver.solve(A, jnp.ones(50))
        >>> x2 = solver.solve(A, jnp.arange(50.0))  # no rebuild
        >>> bool(jnp.allclose(A.mv(x1), jnp.ones(50), atol=1e-2))
        True

        Built lazily from the system operator at every solve:

        >>> lazy = gaussx.PartialCholeskyPreconditioner(rank=20, shift=0.1)
        >>> M_inv = lazy.as_operator(A)
        >>> M_inv.in_structure().shape
        (50,)

        M itself, with a structured log-determinant:

        >>> M = preconditioner.approximation()
        >>> type(M).__name__
        'LowRankUpdate'
        >>> bool(jnp.allclose(gaussx.logdet(M), preconditioner.logdet()))
        True

        A per-point shift, read from a site system A = K + diag(d):

        >>> variances = jnp.linspace(0.1, 10.0, 50)
        >>> site_system = lx.TaggedLinearOperator(
        ...     lx.MatrixLinearOperator(kernel)
        ...     + lx.DiagonalLinearOperator(variances),
        ...     psd,
        ... )
        >>> scaled = gaussx.PartialCholeskyPreconditioner(rank=20, shift=None)
        >>> x = gaussx.CGSolver(preconditioner=scaled).solve(site_system, jnp.ones(50))
        >>> bool(jnp.allclose(site_system.mv(x), jnp.ones(50), atol=1e-2))
        True
    """

    rank: int = eqx.field(static=True, default=50)
    shift: float | Float[Array, ""] | Float[Array, " n"] | None = 1.0
    pivoting: Literal["greedy", "random"] = eqx.field(static=True, default="greedy")
    key: jax.Array | None = None
    diagonal: Float[Array, " n"] | None = None
    factor: Float[Array, "n k"] | None = None
    capacitance: Float[Array, "k k"] | None = None
    factor_shift: Float[Array, ""] | Float[Array, " n"] | None = None

    @classmethod
    def from_operator(
        cls,
        operator: lx.AbstractLinearOperator,
        rank: int = 50,
        *,
        shift: float | Float[Array, ""] | Float[Array, " n"],
        pivoting: Literal["greedy", "random"] = "greedy",
        key: jax.Array | None = None,
        diagonal: Float[Array, " n"] | None = None,
        column: Callable[[Int[Array, ""]], Float[Array, " n"]] | None = None,
    ) -> PartialCholeskyPreconditioner:
        """Factor the PSD part K once and store the preconditioner.

            K ≈ L_k L_kᵀ,   C = σ² I_k + L_kᵀ L_k

        Algorithm:
            1. Clamp k to N and run the pivoted partial Cholesky of K.
            2. Factor the capacitance C = σ² I_k + L_kᵀ L_k by Cholesky.
            3. Store L_k, the factor of C and σ².

        Args:
            operator: The PSD part K (e.g. a kernel matrix), **not**
                K + σ² I.
            rank: Number of pivots k, clamped to the operator size N.
            shift: Noise variance σ²; the preconditioner approximates
                (K + σ² I)⁻¹. Must be positive. A vector d is a per-point
                shift: the factor is of K_B = D^(−½) K D^(−½) and the
                preconditioner approximates (K + diag(d))⁻¹.
            pivoting: ``"greedy"`` or ``"random"`` (see `gaussx.rp_cholesky`).
            key: PRNG key for ``"random"`` pivoting and the diagonal
                estimate. ``None`` means ``jax.random.PRNGKey(0)``.
            diagonal: The diagonal of K. ``None`` takes it from ``operator``
                (see **Cost** in the class docstring).
            column: ``n -> K[:, n]``, used instead of ``operator.mv(e_n)``
                (gh-544). For a matrix-free kernel operator a column is N
                kernel evaluations, where ``mv(e_n)`` is a full matvec. With
                ``column`` and ``diagonal`` both given, the build never
                applies ``operator``.

        Returns:
            A built `PartialCholeskyPreconditioner` whose `as_operator`
            ignores its argument.

        Raises:
            ValueError: If ``rank < 1``.
        """
        if rank < 1:
            raise ValueError("rank must be at least 1")
        rank = min(rank, operator.in_size())
        if jnp.ndim(shift) == 1:
            return _scaled_from_operator(
                cls, operator, rank, shift, pivoting, key, diagonal, column
            )
        # K ≈ L_k L_kᵀ
        factor = _factor(operator, rank, 0.0, pivoting, key, diagonal, column)
        noise_variance = jnp.asarray(shift, dtype=factor.dtype)
        return cls(
            rank=rank,
            pivoting=pivoting,
            key=key,
            factor=factor,
            capacitance=_capacitance(factor, noise_variance),
            factor_shift=noise_variance,
        )

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator | None:
        """Return M⁻¹ = (σ² I + L_k L_kᵀ)⁻¹ as a PSD operator.

            M⁻¹ v = σ⁻² (v − L_k (σ² I_k + L_kᵀ L_k)⁻¹ L_kᵀ v)

        A preconditioner built by `from_operator` ignores ``operator``. The
        lazy one factors K = A − σ² I, where ``operator`` is the system A.

        Args:
            operator: The system operator A = K + σ² I, needed by the lazy
                path only.

        Returns:
            The matrix-free operator M⁻¹, tagged positive semidefinite, or
            ``None`` when the lazy ``rank`` is ``<= 0``.

        Raises:
            ValueError: If the lazy path gets no ``operator``.
        """
        scaled = self._scaled(operator)
        if scaled is not None:
            return scaled.as_operator(operator)
        parts = self._parts(operator)
        if parts is None:
            return None
        factor, capacitance, noise_variance, structure = parts
        return _woodbury(factor, capacitance, noise_variance, structure)

    def supports_slq(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> bool:
        """``True``: M = σ² I + L_k L_kᵀ is known from its factor.

        Args:
            operator: The system operator A; unused.

        Returns:
            ``True``.
        """
        del operator
        return True

    def logdet(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> Float[Array, ""]:
        """log det M by the matrix determinant lemma.

            log det(σ² I + L_k L_kᵀ) = (N − k) log σ² + log det C,
            C = σ² I_k + L_kᵀ L_k

        Args:
            operator: The system operator A = K + σ² I, needed by the lazy
                path only.

        Returns:
            The scalar log det M (0 when the lazy ``rank`` is ``<= 0``).

        Raises:
            ValueError: If the lazy path gets no ``operator``.
        """
        scaled = self._scaled(operator)
        if scaled is not None:
            return scaled.logdet(operator)
        parts = self._parts(operator)
        if parts is None:
            # M = I
            return jnp.zeros((), dtype=_dtype(operator))
        factor, capacitance, noise_variance, _ = parts
        size, rank = factor.shape
        # log det C = 2 ∑ log diag(L_C)
        capacitance_logdet = cholesky_logdet(capacitance)
        # log det M = (N − k) log σ² + log det C
        return (size - rank) * jnp.log(noise_variance) + capacitance_logdet

    def sample(
        self,
        key: PRNGKeyArray,
        operator: lx.AbstractLinearOperator | None = None,
        *,
        shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch n"]:
        """Draw z = σ ε_N + L_k ε_k ~ 𝒩(0, σ² I + L_k L_kᵀ).

        Args:
            key: PRNG key for ε_N and ε_k.
            operator: The system operator A = K + σ² I, needed by the lazy
                path only.
            shape: Batch shape of the draws.

        Returns:
            The draws z, of shape ``(*shape, n)``.

        Raises:
            ValueError: If the lazy path gets no ``operator``.
        """
        # M = σ² I + L_k L_kᵀ, or I when disabled
        approximation = self.approximation(operator)

        def draws(key: PRNGKeyArray, num_draws: int) -> Float[Array, "draw n"]:
            if isinstance(approximation, LowRankUpdate):
                # z = σ ε_N + L_k ε_k, the low-rank path of `gaussx.sample_mvn`
                return low_rank_update_draws(
                    approximation, key, num_draws, diagonal_draws
                )
            # M = I: z = ε_N
            return diagonal_draws(approximation, key, num_draws)

        return batched_draws(draws, key, shape)

    def approximation(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """M = σ² I + L_k L_kᵀ as a PSD `LowRankUpdate` on a diagonal base.

        Its solve (Woodbury) and log-determinant (matrix determinant lemma)
        cost O(N k²), and neither densifies it.

        Args:
            operator: The system operator A = K + σ² I, needed by the lazy
                path only.

        Returns:
            M as a PSD-tagged `gaussx.LowRankUpdate`, or the identity when
            the lazy ``rank`` is ``<= 0``.

        Raises:
            ValueError: If the lazy path gets no ``operator``.
        """
        scaled = self._scaled(operator)
        if scaled is not None:
            # D^½ (I + L_B L_Bᵀ) D^½ = D + (D^½ L_B)(D^½ L_B)ᵀ
            return scaled.approximation(operator)
        parts = self._parts(operator)
        if parts is None:
            # M = I
            return lx.IdentityLinearOperator(_disabled_structure(operator))
        factor, _, noise_variance, _ = parts
        # σ² on every diagonal entry
        base_diagonal = jnp.full(factor.shape[0], noise_variance, dtype=factor.dtype)
        # M = σ² I + L_k L_kᵀ
        return low_rank_plus_diag(base_diagonal, factor, psd=True)

    def _scaled(
        self, operator: lx.AbstractLinearOperator | None
    ) -> _ScaledPreconditioner | None:
        """The scaled form for a per-point shift, else ``None``.

        A stored factor with a vector ``factor_shift`` is the factor of K_B;
        a lazy preconditioner with a vector or ``None`` ``shift`` builds a
        unit-shift lazy preconditioner of B = D^(−½) A D^(−½).

        Args:
            operator: The system operator A, needed by the lazy path only.

        Returns:
            P = D^½ P_B D^½ as a `_ScaledPreconditioner`, or ``None`` for a
            scalar shift and for a disabled lazy preconditioner.

        Raises:
            ValueError: If the lazy path gets no ``operator``, or
                ``shift=None`` and the system does not split.
        """
        if self.factor is not None:
            shift = self.factor_shift
            assert shift is not None
            if jnp.ndim(shift) == 0:
                return None
            # the stored factor and capacitance are of K_B, shift one
            inner = PartialCholeskyPreconditioner(
                rank=self.rank,
                pivoting=self.pivoting,
                key=self.key,
                factor=self.factor,
                capacitance=self.capacitance,
                factor_shift=jnp.ones((), dtype=self.factor.dtype),
            )
            return _ScaledPreconditioner(inner=inner, shift=shift)
        if self.rank <= 0:
            return None
        if self.shift is not None and jnp.ndim(self.shift) == 0:
            return None
        if operator is None:
            raise ValueError(
                "PartialCholeskyPreconditioner with a per-point shift requires "
                "the system operator (or build it once with from_operator)."
            )
        # d, given or read from A = K + diag(d)
        shift = resolve_shift(self.shift, operator, "PartialCholeskyPreconditioner")
        # diag(B) = diag(A) / d, exact when diag(A) is
        diagonal = scaled_diagonal(
            self.diagonal, operator, shift, _diagonal_is_exact(operator)
        )
        # factors B − I = K_B at solve time
        inner = PartialCholeskyPreconditioner(
            rank=self.rank,
            shift=1.0,
            pivoting=self.pivoting,
            key=self.key,
            diagonal=diagonal,
        )
        return _ScaledPreconditioner(inner=inner, shift=shift)

    def _parts(
        self, operator: lx.AbstractLinearOperator | None
    ) -> (
        tuple[
            Float[Array, "n k"],
            Float[Array, "k k"],
            Float[Array, ""],
            jax.ShapeDtypeStruct,
        ]
        | None
    ):
        """The factor L_k, the capacitance factor L_C, σ² and the structure.

        Read from the stored fields after `from_operator`; otherwise built
        once from the system operator A, factoring K = A − σ² I.

        Args:
            operator: The system operator A, needed by the lazy path only.

        Returns:
            ``(L_k, L_C, σ², structure)``, or ``None`` when the lazy
            ``rank`` is ``<= 0``.

        Raises:
            ValueError: If the lazy path gets no ``operator``.
        """
        if self.factor is not None:
            assert self.capacitance is not None and self.factor_shift is not None
            structure = jax.ShapeDtypeStruct((self.factor.shape[0],), self.factor.dtype)
            return self.factor, self.capacitance, self.factor_shift, structure
        if self.rank <= 0:
            return None
        if operator is None:
            raise ValueError(
                "PartialCholeskyPreconditioner.as_operator requires the system "
                "operator to build its factor (or build it once with from_operator)."
            )

        rank = min(self.rank, operator.in_size())
        dtype = operator.in_structure().dtype
        noise_variance = jnp.asarray(self.shift, dtype=dtype)
        # Factor K = A − σ² I, never A itself: the Woodbury step adds σ² back
        # (#345).
        # A − σ² I ≈ L_k L_kᵀ
        factor = _factor(
            operator, rank, noise_variance, self.pivoting, self.key, self.diagonal, None
        )
        capacitance = _capacitance(factor, noise_variance)
        return factor, capacitance, noise_variance, operator.out_structure()


def _scaled_from_operator(
    cls: type[PartialCholeskyPreconditioner],
    operator: lx.AbstractLinearOperator,
    rank: int,
    shift: float | Float[Array, ""] | Float[Array, " n"],
    pivoting: Literal["greedy", "random"],
    key: jax.Array | None,
    diagonal: Float[Array, " n"] | None,
    column: Callable[[Int[Array, ""]], Float[Array, " n"]] | None,
) -> PartialCholeskyPreconditioner:
    """`from_operator` for a per-point shift d: factor K_B = D^(−½) K D^(−½).

        K_B ≈ L_B L_Bᵀ,   C = I_k + L_Bᵀ L_B

    Args:
        cls: The class to build.
        operator: The PSD part K.
        rank: Number of pivots k, already clamped.
        shift: The per-point shift d > 0.
        pivoting: ``"greedy"`` or ``"random"``.
        key: PRNG key for random pivots and the diagonal estimate.
        diagonal: diag(K), or ``None``.
        column: ``n -> K[:, n]``, or ``None``.

    Returns:
        The preconditioner, with ``factor_shift`` = d.
    """
    dtype = operator.in_structure().dtype
    shift = checked_shift(
        jnp.asarray(shift, dtype=dtype), "PartialCholeskyPreconditioner"
    )
    # D^(−½)
    inverse_root = 1.0 / jnp.sqrt(shift)
    # K_B = D^(−½) K D^(−½)
    scaled_kernel = scaled_operator(operator, shift)
    # diag(K_B) = diag(K) / d
    kernel_diagonal = scaled_diagonal(
        diagonal, operator, shift, _diagonal_is_exact(operator)
    )
    scaled_column = None
    if column is not None:
        given_column = column

        def scaled_column(index: Int[Array, ""]) -> Float[Array, " n"]:
            # K_B[:, n] = D^(−½) K[:, n] d_n^(−½)
            return inverse_root * given_column(index) * inverse_root[index]

    # K_B ≈ L_B L_Bᵀ
    factor = _factor(
        scaled_kernel, rank, 0.0, pivoting, key, kernel_diagonal, scaled_column
    )
    unit = jnp.ones((), dtype=factor.dtype)
    return cls(
        rank=rank,
        pivoting=pivoting,
        key=key,
        factor=factor,
        capacitance=_capacitance(factor, unit),
        factor_shift=shift,
    )


def _disabled_structure(
    operator: lx.AbstractLinearOperator | None,
) -> jax.ShapeDtypeStruct:
    """The structure of M = I for a disabled preconditioner.

    Args:
        operator: The system operator A, which sets the size.

    Returns:
        A's input structure.

    Raises:
        ValueError: If ``operator`` is ``None``, so the size is unknown.
    """
    if operator is None:
        raise ValueError(
            "A disabled PartialCholeskyPreconditioner (rank <= 0) is M = I and "
            "needs the system operator for its size."
        )
    return operator.in_structure()


def _dtype(operator: lx.AbstractLinearOperator | None) -> jnp.dtype:
    """The dtype of ``operator``, or the default float without one.

    Args:
        operator: An operator, or ``None``.

    Returns:
        Its input dtype, or ``jnp.result_type(float)``.
    """
    if operator is None:
        return jnp.result_type(float)
    return operator.in_structure().dtype


def _factor(
    operator: lx.AbstractLinearOperator,
    rank: int,
    shift: float | Float[Array, ""],
    pivoting: Literal["greedy", "random"],
    key: jax.Array | None,
    diagonal: Float[Array, " n"] | None,
    column: Callable[[Int[Array, ""]], Float[Array, " n"]] | None,
) -> Float[Array, "n k"]:
    """Pivoted partial Cholesky of B − σ² I, one column per pivot.

        B − σ² I ≈ L_k L_kᵀ

    ``diagonal`` and ``column`` describe ``operator`` B itself; the shift
    σ² is subtracted here.

    Args:
        operator: Operator B (K on the built path, A on the lazy path).
        rank: Number of pivots k.
        shift: σ² to subtract from B (0 on the built path).
        pivoting: ``"greedy"`` or ``"random"``.
        key: PRNG key for random pivots and the Hutchinson diagonal.
        diagonal: diag(B), or ``None`` to read or estimate it.
        column: ``n -> B[:, n]``, or ``None`` to use ``operator.mv(e_n)``.

    Returns:
        The factor L_k, N × k, with exactly zero surplus columns past the
        numerical rank.
    """
    size = operator.in_size()
    dtype = operator.in_structure().dtype
    # Only a Hutchinson estimate is approximate; the pivots then take their
    # values from the exact columns.
    approximate = diagonal is None and not _diagonal_is_exact(operator)
    if diagonal is None:
        diagonal = matrix_free_diag(
            operator, estimate=True, num_probes=_DIAGONAL_PROBES, key=key
        )
    # diag(B) − σ²
    diagonal = jnp.asarray(diagonal, dtype=dtype) - shift
    if column is None:
        column = _mv_column(operator, size, dtype)

    def shifted_column(index):
        # B[:, n] − σ² e_n
        return column(index) - shift * jnp.zeros(size, dtype=dtype).at[index].set(1.0)

    factor, _ = _pivoted_cholesky(
        diagonal,
        shifted_column,
        rank,
        pivoting,
        key,
        approximate_diagonal=approximate,
    )
    return factor


def _mv_column(operator, n, dtype):
    """Column reader n ↦ B e_n through one matvec.

    Args:
        operator: Operator B.
        n: Operator size N.
        dtype: Dtype of the unit vectors e_n.

    Returns:
        A function ``index -> B e_index``.
    """

    def column(index):
        # B e_n
        return operator.mv(jnp.zeros(n, dtype=dtype).at[index].set(1.0))

    return column


def _diagonal_is_exact(operator: lx.AbstractLinearOperator) -> bool:
    """Whether `matrix_free_diag` gives the operator's exact diagonal.

    Args:
        operator: Operator B, walked through tags, negations, scalings,
            sums, `BlockDiag` and the base of a `LowRankUpdate`.

    Returns:
        ``True`` when every leaf has a cheap exact diagonal.
    """
    if isinstance(operator, lx.TaggedLinearOperator | lx.NegLinearOperator):
        return _diagonal_is_exact(operator.operator)
    if isinstance(operator, lx.MulLinearOperator | lx.DivLinearOperator):
        return _diagonal_is_exact(operator.operator)
    if isinstance(operator, lx.AddLinearOperator):
        return _diagonal_is_exact(operator.operator1) and _diagonal_is_exact(
            operator.operator2
        )
    if isinstance(operator, BlockDiag):
        return all(_diagonal_is_exact(op) for op in operator.operators)
    if isinstance(operator, LowRankUpdate):
        return _diagonal_is_exact(operator.base)
    return isinstance(operator, _CHEAP_DIAGONAL)


def _capacitance(
    factor: Float[Array, "n k"], shift: Float[Array, ""]
) -> Float[Array, "k k"]:
    """Lower Cholesky factor of the capacitance C = σ² I_k + L_kᵀ L_k.

        C = σ² I_k + L_kᵀ L_k = L_C L_Cᵀ

    Zero surplus columns only add σ² to the diagonal, so C stays positive
    definite.

    Args:
        factor: Partial Cholesky factor L_k, N × k.
        shift: Noise variance σ².

    Returns:
        The lower-triangular L_C, k × k.
    """
    identity = jnp.eye(factor.shape[1], dtype=factor.dtype)
    # L_kᵀ L_k
    gram = einsum(factor, factor, "row rank, row other -> rank other")
    # C = σ² I_k + L_kᵀ L_k = L_C L_Cᵀ
    return jnp.linalg.cholesky(shift * identity + gram)


def _woodbury(
    factor: Float[Array, "n k"],
    capacitance: Float[Array, "k k"],
    shift: Float[Array, ""],
    structure: jax.ShapeDtypeStruct,
) -> lx.AbstractLinearOperator:
    """M⁻¹ = (σ² I + L_k L_kᵀ)⁻¹ by the Woodbury identity, as an operator.

        M⁻¹ v = σ⁻² (v − L_k C⁻¹ L_kᵀ v),   C = σ² I_k + L_kᵀ L_k

    Args:
        factor: Partial Cholesky factor L_k, N × k.
        capacitance: Lower Cholesky factor L_C of C, k × k.
        shift: Noise variance σ².
        structure: Input structure of M⁻¹.

    Returns:
        The matrix-free operator M⁻¹, tagged positive semidefinite.
    """

    def matvec(v: Float[Array, " n"]) -> Float[Array, " n"]:
        # L_kᵀ v
        projection = einsum(factor, v, "row rank, row -> rank")
        # C⁻¹ L_kᵀ v
        coefficients = jax.scipy.linalg.cho_solve((capacitance, True), projection)
        # M⁻¹ v = σ⁻² (v − L_k C⁻¹ L_kᵀ v)
        return (v - factor @ coefficients) / shift

    return lx.FunctionLinearOperator(matvec, structure, lx.positive_semidefinite_tag)
