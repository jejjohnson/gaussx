"""Randomized Nyström preconditioner (Frangella, Tropp & Udell, 2023).

A = K + σ² I is an N × N symmetric positive-definite system with a PSD part K
and a noise variance σ² > 0 (the ``shift``). The rank-k randomized Nyström
approximation K̂ = K Ω (Ωᵀ K Ω)⁺ Ωᵀ K = U diag(γ̂) Uᵀ of K, with test matrix Ω
(the letters of the paper, not the spatial domain), orthonormal U (N × k)
and eigenvalues γ̂₁ ≥ … ≥ γ̂_k, gives the preconditioner M ≈ A whose inverse
M⁻¹ ≈ A⁻¹ conjugate gradients applies to each residual r_j.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._einx import einsum
from gaussx._operators import LowRankUpdate, low_rank_plus_identity
from gaussx._preconditioners._base import AbstractPreconditioner, batched_draws
from gaussx._preconditioners._scaled import (
    _ScaledPreconditioner,
    checked_shift,
    scaled_operator,
    system_split,
)
from gaussx._primitives._gaussian_draws import diagonal_draws, low_rank_update_draws
from gaussx._randomized._nystrom import randomized_nystrom


class NystromPreconditioner(AbstractPreconditioner):
    """Randomized Nyström preconditioner for K + σ² I.

        M⁻¹ r = (γ̂_k + σ²) U diag(γ̂ + σ²)⁻¹ Uᵀ r + (r − U Uᵀ r)

    For a system A = K + σ² I with K PSD (e.g. a kernel matrix plus noise),
    it builds a rank-k randomized Nyström approximation K̂ = U diag(γ̂) Uᵀ of
    the PSD part K alone (`gaussx.randomized_nystrom`) and applies the
    inverse above (Frangella, Tropp & Udell, 2023). On range(U) the
    preconditioned operator is ≈ γ̂_k + σ²; on the complement it is A
    itself, with eigenvalues in [σ², γ_k+1 + σ²]. So

        κ(M^(−½) A M^(−½)) ≤ (γ̂_k + σ² + ‖K − K̂‖₂) / σ²,

    which is O(1) once k ≳ d_eff(σ²) = tr(K (K + σ² I)⁻¹), the effective
    dimension. With k = 2⌈1.5 d_eff(σ²)⌉ + 1 the expected condition number
    is below 28, so CG needs a number of steps that does not grow with N.

    Build it once with `from_operator` on the PSD part K and an explicit
    ``shift=σ²``, like `gaussx.PartialCholeskyPreconditioner`: K must not
    already contain σ², or the noise is counted twice (#345). The factors
    are stored as arrays, so `as_operator` ignores its argument and
    repeated solves cost no further matvecs.

    **Covariance form only.** K̂ captures the *top* of K's spectrum, which
    is right for K + σ² I. It is not a good preconditioner for a
    precision-form GMRF system Q + Aᵀ W A, whose hard directions are its
    smallest eigenvalues; use an exact sparse factorisation or
    Jacobi-preconditioned CG there.

    Up to gaussx 0.4 this class was a randomized Rayleigh-Ritz projection
    onto a random subspace, which made CG *slower* below full rank (#354).
    For a Rayleigh-Ritz eigendecomposition use
    ``gaussx.randomized_eigh(op, rank, n_power_iter=0)``; it projects onto
    orth(K Ω) rather than orth(Ω), which is more accurate, so it is not
    numerically identical to the old preconditioner.

    **As P itself.** The inverse of M⁻¹ is

        P = I + U diag(s − 1) Uᵀ,   s_j = (γ̂_j + σ²) / (γ̂_k + σ²) ≥ 1,

    which is A / (γ̂_k + σ²) on range(U) and I on its complement, so
    P ≈ A / (γ̂_k + σ²) up to the scale. `approximation`, `logdet` and
    `sample` use this P, the one CG applies, so that
    log det A = log det P + log det(P⁻¹ A) holds exactly and the stochastic
    part sees the near-identity P⁻¹ A:

        log det P = ∑_j log s_j,
        z = ε_n + U (√(s − 1) ⊙ ε_k) ~ 𝒩(0, P).

    **Per-point shifts.** For A = K + diag(d) with a diagonal that varies by
    point, `from_operator` takes a vector ``shift=d`` and sketches
    K_B = D^(−½) K D^(−½), D = diag(d), so the preconditioner above with
    σ² = 1 targets B = D^(−½) A D^(−½) = K_B + I and is pulled back:

        M⁻¹ = D^(−½) M_B⁻¹ D^(−½),   P = D^½ P_B D^½,
        log det P = log det P_B + ∑_n log d_n.

    The sketch then captures the top of K_B, the kernel weighted by how
    strongly each point is observed. `lazy` stores only the rank and key and
    sketches, at every solve, the kernel part K of whatever system
    A = K + diag(d) it is given (read with `gaussx.split_diagonal`), which
    suits a site system that changes at every Newton step. Every d_n must
    be positive (``eqx.error_if``); a masked site (d_n = ∞) is out of
    scope.

    Attributes:
        basis: Orthonormal basis U, N × k; ``None`` for a lazy one.
        eigenvalues: Nyström eigenvalues γ̂, descending, of length k;
            ``None`` for a lazy one.
        shift: Noise variance σ², or the per-point shift d of a sketch of
            K_B; ``None`` for a lazy one.
        rank: Sketch rank k of a lazy preconditioner.
        oversample: Extra sketch columns of a lazy preconditioner.
        key: PRNG key for the test matrix Ω of a lazy preconditioner.

    Examples:
        Build once on the kernel matrix K, then solve K + σ² I by CG:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> inputs = jnp.linspace(0.0, 10.0, 100)
        >>> gaps = einx.subtract("row, col -> row col", inputs, inputs)
        >>> kernel = jnp.exp(-0.5 * gaps**2)
        >>> psd = lx.positive_semidefinite_tag
        >>> preconditioner = gaussx.NystromPreconditioner.from_operator(
        ...     lx.MatrixLinearOperator(kernel, psd), rank=30, shift=0.1, key=jr.key(0)
        ... )
        >>> preconditioner.basis.shape, preconditioner.eigenvalues.shape
        ((100, 30), (30,))
        >>> A = lx.MatrixLinearOperator(kernel + 0.1 * jnp.eye(100), psd)
        >>> solver = gaussx.PreconditionedCGSolver(preconditioner=preconditioner)
        >>> x = solver.solve(A, jnp.ones(100))
        >>> bool(jnp.allclose(A.mv(x), jnp.ones(100), atol=1e-2))
        True

        P itself, an orthonormal low-rank update with an exact inverse:

        >>> P = preconditioner.approximation()
        >>> type(P).__name__
        'LowRankUpdate'
        >>> identity = preconditioner.as_operator().mv(P.mv(jnp.ones(100)))
        >>> bool(jnp.allclose(identity, jnp.ones(100), atol=1e-3))
        True

        A lazy one, built per solve from a site system A = K + diag(d):

        >>> variances = jnp.linspace(0.1, 10.0, 100)
        >>> site_system = lx.TaggedLinearOperator(
        ...     lx.MatrixLinearOperator(kernel)
        ...     + lx.DiagonalLinearOperator(variances),
        ...     psd,
        ... )
        >>> lazy = gaussx.NystromPreconditioner.lazy(30, key=jr.key(0))
        >>> x = gaussx.CGSolver(preconditioner=lazy).solve(site_system, jnp.ones(100))
        >>> bool(jnp.allclose(site_system.mv(x), jnp.ones(100), atol=1e-2))
        True

        A non-positive shift is rejected:

        >>> gaussx.NystromPreconditioner.from_operator(
        ...     lx.MatrixLinearOperator(kernel, psd), rank=5, shift=0.0
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: shift must be positive, got 0.0.
    """

    basis: Float[Array, "n l"] | None = None
    eigenvalues: Float[Array, " l"] | None = None
    shift: Float[Array, ""] | Float[Array, " n"] | None = None
    rank: int = eqx.field(static=True, default=50)
    oversample: int = eqx.field(static=True, default=0)
    key: jax.Array | None = None

    @classmethod
    def lazy(
        cls,
        rank: int = 50,
        *,
        oversample: int = 0,
        key: jax.Array | None = None,
    ) -> NystromPreconditioner:
        """A preconditioner that sketches the kernel part of each system.

            A = K + diag(d),   K_B = D^(−½) K D^(−½) ≈ U diag(γ̂) Uᵀ

        At every solve, `gaussx.split_diagonal` reads the system back into K
        and d, and K_B is sketched at rank k (see **Per-point shifts**).

        Args:
            rank: Rank k (``rank`` matvecs of K per build).
            oversample: Extra sketch columns.
            key: PRNG key for the test matrix Ω. ``None`` means
                ``jax.random.PRNGKey(0)``.

        Returns:
            A lazy `NystromPreconditioner`.

        Raises:
            ValueError: If ``rank < 1``.
        """
        if rank < 1:
            raise ValueError(f"rank must be at least 1, got {rank}.")
        return cls(rank=rank, oversample=oversample, key=key)

    @classmethod
    def from_operator(
        cls,
        operator: lx.AbstractLinearOperator,
        rank: int = 50,
        *,
        shift: float | Float[Array, ""] | Float[Array, " n"],
        oversample: int = 0,
        key: jax.Array | None = None,
    ) -> NystromPreconditioner:
        """Sketch the PSD part K once and store the preconditioner.

            K̂ = K Ω (Ωᵀ K Ω)⁺ Ωᵀ K = U diag(γ̂) Uᵀ

        Args:
            operator: The PSD part K (e.g. a kernel matrix), **not**
                K + σ² I. May be matrix-free.
            rank: Rank k (``rank`` matvecs of K), clamped to the operator
                size. Aim for k ≳ 2⌈1.5 d_eff(σ²)⌉ + 1.
            shift: Noise variance σ²; the preconditioner targets K + σ² I.
                Must be positive. A vector d is a per-point shift: the
                sketch is of K_B = D^(−½) K D^(−½), and the preconditioner
                targets K + diag(d).
            oversample: Extra sketch columns, discarded after the
                approximation (see `gaussx.randomized_nystrom`).
            key: PRNG key for the test matrix Ω. ``None`` means
                ``jax.random.PRNGKey(0)``.

        Returns:
            A built `NystromPreconditioner`.

        Raises:
            ValueError: If ``rank < 1`` or a concrete ``shift`` is not
                positive.
        """
        if isinstance(shift, (int, float)) and shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}.")
        if jnp.ndim(shift) == 1:
            dtype = operator.in_structure().dtype
            shift = checked_shift(
                jnp.asarray(shift, dtype=dtype), "NystromPreconditioner"
            )
            # K_B = D^(−½) K D^(−½) ≈ U diag(γ̂) Uᵀ
            approx = randomized_nystrom(
                scaled_operator(operator, shift),
                rank,
                oversample=oversample,
                key=key,
            )
            return cls(basis=approx.U, eigenvalues=approx.d, shift=shift)
        # K̂ = U diag(γ̂) Uᵀ
        approx = randomized_nystrom(operator, rank, oversample=oversample, key=key)
        return cls(
            basis=approx.U,
            eigenvalues=approx.d,
            shift=jnp.asarray(shift, dtype=approx.d.dtype),
        )

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """Return M⁻¹ as a PSD operator; ``operator`` is ignored.

            M⁻¹ = I + U diag((γ̂_k + σ²) / (γ̂ + σ²) − 1) Uᵀ

        Args:
            operator: The system operator A; unused, since the factors are
                stored.

        Returns:
            The matrix-free operator M⁻¹, tagged positive semidefinite.

        Raises:
            ValueError: If a lazy preconditioner gets no ``operator``, or a
                system that does not split into K + diag(d).
        """
        scaled = self._scaled(operator)
        if scaled is not None:
            return scaled.as_operator(operator)
        basis, noise_variance = self.basis, self.shift
        assert basis is not None and self.eigenvalues is not None
        assert noise_variance is not None
        # (γ̂_k + σ²) / (γ̂ + σ²) − 1
        scale = (self.eigenvalues[-1] + noise_variance) / (
            self.eigenvalues + noise_variance
        ) - 1.0
        structure = jax.ShapeDtypeStruct((basis.shape[0],), basis.dtype)

        def matvec(x: Float[Array, " n"]) -> Float[Array, " n"]:
            # Uᵀ r
            coefficients = einsum(basis, x, "row rank, row -> rank")
            # M⁻¹ r = r + U (((γ̂_k + σ²) / (γ̂ + σ²) − 1) ⊙ Uᵀ r)
            return x + einsum(basis, scale * coefficients, "row rank, rank -> row")

        return lx.FunctionLinearOperator(
            matvec, structure, lx.positive_semidefinite_tag
        )

    def supports_slq(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> bool:
        """``True``: P = (M⁻¹)⁻¹ is known from the stored factors.

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
        """log det P = ∑_j log s_j, s_j = (γ̂_j + σ²) / (γ̂_k + σ²).

        Args:
            operator: The system operator A; unused, since the factors are
                stored.

        Returns:
            The scalar log det P (≥ 0, since every s_j ≥ 1, for a scalar
            shift).
        """
        scaled = self._scaled(operator)
        if scaled is not None:
            return scaled.logdet(operator)
        # s_j = (γ̂_j + σ²) / (γ̂_k + σ²)
        ratio = self._ratio()
        # log det P = ∑_j log s_j
        return jnp.sum(jnp.log(ratio))

    def sample(
        self,
        key: PRNGKeyArray,
        operator: lx.AbstractLinearOperator | None = None,
        *,
        shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch n"]:
        """Draw z = ε_n + U (√(s − 1) ⊙ ε_k) ~ 𝒩(0, P).

        The two parts are independent, so Cov(z) = I + U diag(s − 1) Uᵀ = P;
        this is the low-rank path of `gaussx.sample_mvn` on `approximation`.

        Args:
            key: PRNG key for ε_k and ε_n.
            operator: The system operator A; unused, since the factors are
                stored.
            shape: Batch shape of the draws.

        Returns:
            The draws z, of shape ``(*shape, n)``.
        """
        scaled = self._scaled(operator)
        if scaled is not None:
            return scaled.sample(key, operator, shape=shape)
        # P = I + U diag(s − 1) Uᵀ
        approximation = self._low_rank_update()

        def draws(key: PRNGKeyArray, num_draws: int) -> Float[Array, "draw n"]:
            # z = ε_n + U (√(s − 1) ⊙ ε_k)
            return low_rank_update_draws(approximation, key, num_draws, diagonal_draws)

        return batched_draws(draws, key, shape)

    def approximation(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """P = I + U diag(s − 1) Uᵀ, an orthonormal `LowRankUpdate`.

        Its solve and log-determinant take the O(N k) orthonormal closed
        forms, with no k × k factorisation.

        Args:
            operator: The system operator A; unused, since the factors are
                stored.

        Returns:
            P as a PSD-tagged `gaussx.LowRankUpdate` on an identity base
            (on a diagonal base D, and not orthonormal, for a per-point
            shift).
        """
        scaled = self._scaled(operator)
        if scaled is not None:
            return scaled.approximation(operator)
        return self._low_rank_update()

    def _scaled(
        self, operator: lx.AbstractLinearOperator | None
    ) -> _ScaledPreconditioner | None:
        """The scaled form for a per-point shift or a lazy build, else ``None``.

        Args:
            operator: The system A, needed by a lazy preconditioner only.

        Returns:
            P = D^½ P_B D^½ as a `_ScaledPreconditioner`, or ``None`` for a
            built preconditioner with a scalar shift.

        Raises:
            ValueError: If a lazy preconditioner gets no ``operator``, or a
                system that does not split into K + diag(d).
        """
        if self.basis is None:
            if operator is None:
                raise ValueError(
                    "A lazy NystromPreconditioner requires the system operator "
                    "A = K + diag(d) to sketch its kernel part."
                )
            # A = K + diag(d)
            split = system_split(operator, "NystromPreconditioner.lazy")
            built = NystromPreconditioner.from_operator(
                split.base,
                self.rank,
                shift=split.diagonal,
                oversample=self.oversample,
                key=self.key,
            )
            return built._scaled(None)
        assert self.shift is not None and self.eigenvalues is not None
        if jnp.ndim(self.shift) == 0:
            return None
        # the stored sketch is of K_B, shift one
        inner = NystromPreconditioner(
            basis=self.basis,
            eigenvalues=self.eigenvalues,
            shift=jnp.ones((), dtype=self.basis.dtype),
        )
        return _ScaledPreconditioner(inner=inner, shift=self.shift)

    def _low_rank_update(self) -> LowRankUpdate:
        """P = I + U diag(s − 1) Uᵀ, built from the stored factors.

        Returns:
            P as a PSD-tagged orthonormal `gaussx.LowRankUpdate`.
        """
        assert self.basis is not None
        # s − 1 = (γ̂ − γ̂_k) / (γ̂_k + σ²) ≥ 0
        update = self._ratio() - 1.0
        # P = I + U diag(s − 1) Uᵀ
        return low_rank_plus_identity(self.basis, update, psd=True, orthonormal=True)

    def _ratio(self) -> Float[Array, " l"]:
        """The eigenvalues s_j = (γ̂_j + σ²) / (γ̂_k + σ²) of P on range(U).

        Returns:
            s, of length k.
        """
        noise_variance = self.shift
        assert self.eigenvalues is not None and noise_variance is not None
        # s_j = (γ̂_j + σ²) / (γ̂_k + σ²)
        return (self.eigenvalues + noise_variance) / (
            self.eigenvalues[-1] + noise_variance
        )
