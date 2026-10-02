"""Randomized Nyström preconditioner (Frangella, Tropp & Udell, 2023)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._preconditioners._base import AbstractPreconditioner
from gaussx._randomized._nystrom import randomized_nystrom


class NystromPreconditioner(AbstractPreconditioner):
    r"""Randomized Nyström preconditioner for ``A + μ I``.

    For a system ``A + μ I`` with ``A`` PSD (e.g. ``K + σ² I``), it builds a
    rank-``l`` randomized Nyström approximation ``Â = U Λ̂ Uᵀ`` of the PSD
    part ``A`` alone (`gaussx.randomized_nystrom`) and applies (Frangella,
    Tropp & Udell, 2023)

    $$
    P^{-1}x = (\hat\lambda_\ell + \mu)\,U(\hat\Lambda + \mu I)^{-1}U^\top x
    + (x - UU^\top x).
    $$

    On ``range(U)`` the preconditioned operator is ``≈ λ̂_l + μ``; on the
    complement it is ``A + μ I`` itself, with eigenvalues in
    ``[μ, λ_{l+1} + μ]``. So

    $$
    \kappa\big(P^{-1/2}(A+\mu I)P^{-1/2}\big)
    \le \frac{\hat\lambda_\ell + \mu + \|A - \hat A\|}{\mu},
    $$

    which is ``O(1)`` once ``l ≳ d_eff(μ) = tr(A (A + μ I)⁻¹)``, the
    effective dimension. With ``l = 2⌈1.5 d_eff(μ)⌉ + 1`` the expected
    condition number is below 28, so CG needs a number of iterations that
    does not grow with ``n``.

    Build it once with `from_operator` on the PSD part ``A`` and an explicit
    ``shift=μ``, like `gaussx.PartialCholeskyPreconditioner`: ``A`` must not
    already contain ``μ``, or the noise is counted twice (#345). The
    factors are stored as arrays, so `as_operator` ignores its argument and
    repeated solves cost no further matvecs.

    **Covariance form only.** ``Â`` captures the *top* of ``A``'s spectrum,
    which is right for ``K + σ² I``. It is not a good preconditioner for a
    precision-form GMRF system ``Q + Aᵀ W A``, whose hard directions are its
    smallest eigenvalues; use an exact sparse factorisation or
    Jacobi-preconditioned CG there.

    Up to gaussx 0.4 this class was a randomized Rayleigh-Ritz projection
    onto a random subspace, which made CG *slower* below full rank (#354).
    For a Rayleigh-Ritz eigendecomposition use
    ``gaussx.randomized_eigh(op, rank, n_power_iter=0)``; it projects onto
    ``orth(AΩ)`` rather than ``orth(Ω)``, which is more accurate, so it is not
    numerically identical to the old preconditioner.

    Attributes:
        basis: Orthonormal ``U``, shape ``(n, l)``.
        eigenvalues: ``Λ̂``, descending, shape ``(l,)``.
        shift: ``μ`` (e.g. the noise variance ``σ²``).

    Examples:
        A GP with n = 200k and a Matérn-3/2 kernel is solved by preconditioned
        CG in tens of iterations, not thousands::

            K_op = kl.to_operator(kernel, X, implicit=True)
            P = gx.NystromPreconditioner.from_operator(
                K_op, rank=500, shift=noise_var, key=key
            )
            A_op = K_op + lx.DiagonalLinearOperator(jnp.full(n, noise_var))
            alpha = gx.PreconditionedCGSolver(preconditioner=P).solve(A_op, y)

        A small runnable version:

        >>> import einx, jax.numpy as jnp, jax.random as jr, lineax as lx
        >>> import gaussx as gx
        >>> x = jnp.linspace(0.0, 10.0, 100)
        >>> K = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2)
        >>> psd = lx.positive_semidefinite_tag
        >>> P = gx.NystromPreconditioner.from_operator(
        ...     lx.MatrixLinearOperator(K, psd), rank=30, shift=0.1, key=jr.key(0)
        ... )
        >>> A = lx.MatrixLinearOperator(K + 0.1 * jnp.eye(100), psd)
        >>> alpha = gx.PreconditionedCGSolver(preconditioner=P).solve(A, jnp.ones(100))
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
        """Sketch the PSD part once and store the preconditioner.

        Args:
            operator: The PSD part ``A`` (e.g. a kernel matrix ``K``),
                **not** ``A + μ I``. May be matrix-free.
            rank: Sketch size ``l`` (``rank`` matvecs of ``A``), clamped to
                the operator size. Aim for ``l ≳ 2⌈1.5 d_eff(μ)⌉ + 1``.
            shift: ``μ``, e.g. the noise variance ``σ²``; the preconditioner
                targets ``A + μ I``. Must be positive.
            oversample: Extra sketch columns, discarded after the
                approximation (see `gaussx.randomized_nystrom`).
            key: PRNG key for the test matrix. ``None`` means
                ``jax.random.PRNGKey(0)``.

        Returns:
            A built `NystromPreconditioner`.

        Raises:
            ValueError: If ``rank < 1`` or a concrete ``shift`` is not
                positive.
        """
        if isinstance(shift, (int, float)) and shift <= 0:
            raise ValueError(f"shift must be positive, got {shift}.")
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
        """Return ``P⁻¹`` as a PSD operator; *operator* is ignored."""
        U, mu = self.basis, self.shift
        # P⁻¹ = I + U diag((λ̂_l + μ)/(λ̂ + μ) − 1) Uᵀ.
        scale = (self.eigenvalues[-1] + mu) / (self.eigenvalues + mu) - 1.0
        structure = jax.ShapeDtypeStruct((U.shape[0],), U.dtype)

        def matvec(x: Float[Array, " n"]) -> Float[Array, " n"]:
            coeffs = einsum(U, x, "n l, n -> l")
            return x + einsum(U, scale * coeffs, "n l, l -> n")

        return lx.FunctionLinearOperator(
            matvec, structure, lx.positive_semidefinite_tag
        )
