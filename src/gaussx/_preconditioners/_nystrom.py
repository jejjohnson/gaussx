"""Randomized Nyström preconditioner (Frangella, Tropp & Udell, 2023).

A = K + σ² I is an N × N symmetric positive-definite system with a PSD part K
and a noise variance σ² > 0 (the ``shift``). The rank-k randomized Nyström
approximation K̂ = K Ω (Ωᵀ K Ω)⁺ Ωᵀ K = U diag(γ̂) Uᵀ of K, with test matrix Ω
(the letters of the paper, not the spatial domain), orthonormal U (N × k)
and eigenvalues γ̂₁ ≥ … ≥ γ̂_k, gives the preconditioner M ≈ A whose inverse
M⁻¹ ≈ A⁻¹ conjugate gradients applies to each residual r_j.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._preconditioners._base import AbstractPreconditioner
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

    Attributes:
        basis: Orthonormal basis U, N × k.
        eigenvalues: Nyström eigenvalues γ̂, descending, of length k.
        shift: Noise variance σ².

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

        A non-positive shift is rejected:

        >>> gaussx.NystromPreconditioner.from_operator(
        ...     lx.MatrixLinearOperator(kernel, psd), rank=5, shift=0.0
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: shift must be positive, got 0.0.
    """

    basis: Float[Array, "n l"]
    eigenvalues: Float[Array, " l"]
    shift: Float[Array, ""]

    @classmethod
    def from_operator(
        cls,
        operator: lx.AbstractLinearOperator,
        rank: int = 50,
        *,
        shift: float | Float[Array, ""],
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
                Must be positive.
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
        """
        basis, noise_variance = self.basis, self.shift
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
