"""FITC preconditioner: a Nyström approximation plus its diagonal correction.

A = K_XX + Λ is an N × N system with a kernel part K_XX and a diagonal Λ (the
noise σ² I, or W⁻¹ for a Laplace fit). With R inducing inputs Z, the Nyström
approximation of K_XX is Q = U Uᵀ, U = K_XZ L_ZZ⁻ᵀ, L_ZZ L_ZZᵀ = K_ZZ, and the
"fully independent training conditional" (FITC) preconditioner adds back the
diagonal it misses (Gyger, Furrer & Sigrist, 2026, §4.3.2):

    P = U Uᵀ + diag(r + λ),   r = max(diag(K_XX − Q), 0),

a rank-R update of a diagonal, so its solve is Woodbury and its
log-determinant the matrix determinant lemma.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._einx import rearrange, reduce
from gaussx._operators import LowRankUpdate, low_rank_plus_diag
from gaussx._preconditioners._base import AbstractPreconditioner, batched_draws
from gaussx._preconditioners._scaled import checked_shift, resolve_shift
from gaussx._primitives._gaussian_draws import diagonal_draws, low_rank_update_draws
from gaussx._primitives._inv import inv
from gaussx._primitives._logdet import logdet as _logdet


def _kernel_diagonal(
    kernel: Callable[[Array, Array], Array],
    inputs: Float[Array, "n d"],
) -> Float[Array, " n"]:
    """diag(K_XX) = [k(x_n, x_n)]_n, without forming K_XX.

    Args:
        kernel: Gram function ``(X₁, X₂) -> K(X₁, X₂)``.
        inputs: Inputs X, one row per point.

    Returns:
        The prior variances k(x_n, x_n).
    """

    def variance(point: Float[Array, " d"]) -> Float[Array, ""]:
        row = rearrange(point, "d -> 1 d")
        # k(x_n, x_n)
        return kernel(row, row)[0, 0]

    return jax.vmap(variance)(inputs)


class FITCPreconditioner(AbstractPreconditioner):
    """Nyström-plus-diagonal ("FITC") preconditioner of A = K_XX + Λ.

        P = U Uᵀ + diag(r + λ),   r = max(diag(K_XX) − rowsum(U ⊙ U), 0)

    with U = K_XZ L_ZZ⁻ᵀ the whitened cross-covariance of R inducing inputs
    Z and λ the diagonal of Λ. Q = U Uᵀ is the Nyström approximation of
    K_XX, and r the prior variance it misses, so P matches A exactly on the
    diagonal. P is a rank-R update of a diagonal, so

        P⁻¹ = D⁻¹ − D⁻¹ U (I_R + Uᵀ D⁻¹ U)⁻¹ Uᵀ D⁻¹,   D = diag(r + λ),
        log det P = ∑_n log D_n + log det(I_R + Uᵀ D⁻¹ U),
        z = √D ⊙ ε_N + U ε_R ~ 𝒩(0, P),

    each O(N R²) to build and O(N R) to apply (Woodbury, the matrix
    determinant lemma, and the sum of two independent draws). The
    inducing inputs are the preconditioner's own: a larger R than the
    model's gives a stronger preconditioner. Unlike a pivoted partial
    Cholesky without diagonal correction, the CG rate does not depend on
    the largest eigenvalue of K_XX (Gyger et al., 2026, Theorem 6).

    **The diagonal.** With ``diagonal_shift=None`` (the default) λ is read
    from the system at every call with `gaussx.split_diagonal`, so one
    object serves every Newton step of a Laplace fit, where Λ = W⁻¹ moves.
    A scalar or a vector fixes it, as for Gaussian noise σ². A kernel whose
    hyperparameters have moved since `from_kernel` gives a stale U: CG then
    takes more steps to the same answer.

    **Measured (gh-824).** At the Laplace mode of a Bernoulli GP (Matérn-3/2,
    N = 400, R = 60) reading the current Λ took CG from 20 to 6 steps on a
    smooth kernel and from 46 to 21 on a rough one, while a Λ fixed at the
    start of the fit took 11 and 38. The diagonally scaled partial Cholesky
    and Nyström preconditioners took 5 and 12 steps on the same systems
    with random inducing inputs here, so this one is an alternative, not a
    replacement.

    Attributes:
        U: Whitened cross-covariance factor K_XZ L_ZZ⁻ᵀ, N × R.
        prior_residual: r = diag(K_XX − Q), clamped at zero, length N.
        diagonal_shift: λ as a scalar or a length-N vector, or ``None`` to
            read it from the system.

    Examples:
        One object, built once, for two systems with different diagonals:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> def kernel(left, right):
        ...     gaps = einx.subtract("n d, m d -> n m d", left, right)
        ...     return jnp.exp(-0.5 * einx.sum("n m [d]", gaps**2) / 0.3**2)
        >>> X = einx.id("n -> n 1", jnp.linspace(0.0, 1.0, 60))
        >>> pre = gaussx.FITCPreconditioner.from_kernel(kernel, X, X[::6])
        >>> pre.U.shape, pre.prior_residual.shape
        ((60, 10), (60,))
        >>> psd = lx.positive_semidefinite_tag
        >>> K = lx.MatrixLinearOperator(kernel(X, X))
        >>> for noise in (0.1, 2.0):
        ...     system = lx.TaggedLinearOperator(
        ...         K + lx.DiagonalLinearOperator(jnp.full(60, noise)), psd
        ...     )
        ...     x = gaussx.CGSolver(preconditioner=pre).solve(system, jnp.ones(60))
        ...     print(bool(jnp.allclose(system.mv(x), 1.0, atol=1e-3)))
        True
        True

        A fixed shift, and P itself as a low-rank update of a diagonal:

        >>> fixed = gaussx.FITCPreconditioner.from_kernel(
        ...     kernel, X, X[::6], diagonal_shift=0.1
        ... )
        >>> type(fixed.approximation()).__name__
        'LowRankUpdate'
        >>> round(float(fixed.logdet() - gaussx.logdet(fixed.approximation())), 4)
        0.0

        Without a fixed shift, P needs the system:

        >>> pre.as_operator()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: FITCPreconditioner(diagonal_shift=None) reads ...

    References:
        Gyger, Furrer & Sigrist (2026), JMLR 27(115), §4.3.2 and Theorem 6.
        Quiñonero-Candela, J. & Rasmussen, C. E. (2005). A unifying view of
        sparse approximate Gaussian process regression. JMLR 6, 1939-1959.
    """

    U: Float[Array, "n r"]
    prior_residual: Float[Array, " n"]
    diagonal_shift: Float[Array, ""] | Float[Array, " n"] | None = None

    @classmethod
    def from_kernel(
        cls,
        kernel: Callable[[Array, Array], Array],
        X: Float[Array, "n d"],
        Z: Float[Array, "r d"],
        *,
        diagonal_shift: float | Float[Array, ""] | Float[Array, " n"] | None = None,
        jitter: float = 1e-6,
    ) -> FITCPreconditioner:
        """Build U and r from a kernel, the inputs X and inducing inputs Z.

            L_ZZ L_ZZᵀ = K_ZZ + β I,   U = K_XZ L_ZZ⁻ᵀ

        Only K_ZZ (R × R), K_XZ (N × R) and diag(K_XX) are evaluated.

        Args:
            kernel: Gram function ``(X₁, X₂) -> K(X₁, X₂)``, differentiable
                in its hyperparameters.
            X: Inputs X, N × d.
            Z: Inducing inputs Z of the preconditioner, R × d.
            diagonal_shift: λ, or ``None`` to read it from the system.
            jitter: Jitter β added to K_ZZ before its Cholesky factor
                (factored by `gaussx.safe_cholesky`).

        Returns:
            The `FITCPreconditioner`.
        """
        # lazy import, cycle: _linalg -> _primitives -> _preconditioners
        from gaussx._linalg._safe_cholesky import safe_cholesky

        gram = kernel(Z, Z)
        # β I, in the kernel's dtype
        jitter_operator = lx.DiagonalLinearOperator(
            jnp.full(gram.shape[0], jitter, dtype=gram.dtype)
        )
        # K_ZZ + β I, the jitter folded into the diagonal
        inducing_gram = lx.TaggedLinearOperator(
            lx.MatrixLinearOperator(gram) + jitter_operator,
            lx.positive_semidefinite_tag,
        )
        # L_ZZ L_ZZᵀ = K_ZZ + β I
        inducing_factor = safe_cholesky(inducing_gram)
        # K_ZX
        cross_covariance = rearrange(kernel(X, Z), "point inducing -> inducing point")
        # L_ZZ⁻¹ K_ZX
        whitened = jax.scipy.linalg.solve_triangular(
            inducing_factor, cross_covariance, lower=True
        )
        # U = K_XZ L_ZZ⁻ᵀ
        factor = rearrange(whitened, "inducing point -> point inducing")
        return cls.from_factors(factor, _kernel_diagonal(kernel, X), diagonal_shift)

    @classmethod
    def from_factors(
        cls,
        U: Float[Array, "n r"],
        prior_diagonal: Float[Array, " n"],
        diagonal_shift: float | Float[Array, ""] | Float[Array, " n"] | None = None,
    ) -> FITCPreconditioner:
        """Build from a factor U of Q = U Uᵀ and the prior variances diag(K_XX).

            r = max(diag(K_XX) − rowsum(U ⊙ U), 0)

        r is clamped at zero before the shift is added: in float32 it goes
        slightly negative, and a negative entry of D would make `sample`
        return ``NaN``.

        Args:
            U: Factor of the Nyström approximation Q = U Uᵀ, N × R.
            prior_diagonal: The prior variances diag(K_XX), length N.
            diagonal_shift: λ, or ``None`` to read it from the system.

        Returns:
            The `FITCPreconditioner`.
        """
        # diag(Q) = rowsum(U ⊙ U)
        nystrom_diagonal = reduce(U * U, "point inducing -> point", "sum")
        # r = max(diag(K_XX) − diag(Q), 0)
        residual = jnp.maximum(prior_diagonal - nystrom_diagonal, 0.0)
        if diagonal_shift is not None:
            diagonal_shift = jnp.asarray(diagonal_shift, dtype=U.dtype)
        return cls(U=U, prior_residual=residual, diagonal_shift=diagonal_shift)

    def _low_rank_update(
        self, operator: lx.AbstractLinearOperator | None
    ) -> LowRankUpdate:
        """P = diag(r + λ) + U Uᵀ, with λ fixed or read from ``operator``.

        Args:
            operator: The system A = K_XX + Λ, needed when λ is not fixed.

        Returns:
            P as a PSD-tagged `gaussx.LowRankUpdate` on a diagonal base.

        Raises:
            ValueError: Without a fixed shift, if ``operator`` is ``None`` or
                does not split into a kernel part plus a diagonal.
        """
        shift = self.diagonal_shift
        if shift is not None:
            # λ > 0, so that D = r + λ is positive
            shift = checked_shift(jnp.asarray(shift), "FITCPreconditioner")
        else:
            if operator is None:
                raise ValueError(
                    "FITCPreconditioner(diagonal_shift=None) reads λ from the "
                    "system A = K + diag(λ), so it needs the system operator."
                )
            # λ from A = K + diag(λ), checked positive
            shift = resolve_shift(None, operator, "FITCPreconditioner")
        # D = r + λ
        diagonal = self.prior_residual + shift
        # P = diag(D) + U Uᵀ
        return low_rank_plus_diag(diagonal, self.U, psd=True)

    def as_operator(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """P⁻¹ by the Woodbury identity, kept low rank and PSD-tagged.

            P⁻¹ = D⁻¹ − D⁻¹ U (I_R + Uᵀ D⁻¹ U)⁻¹ Uᵀ D⁻¹

        Args:
            operator: The system A, read for λ when it is not fixed.

        Returns:
            P⁻¹ as a `gaussx.LowRankUpdate`, O(N R) per matvec.

        Raises:
            ValueError: Without a fixed shift, if ``operator`` is ``None`` or
                does not split into a kernel part plus a diagonal.
        """
        # P⁻¹, by gaussx.inv's symmetric Woodbury path
        return inv(self._low_rank_update(operator))

    def supports_slq(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> bool:
        """``True``: P is known in closed form.

        Args:
            operator: The system A; unused.

        Returns:
            ``True``.
        """
        del operator
        return True

    def logdet(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> Float[Array, ""]:
        """log det P by the matrix determinant lemma.

            log det P = ∑_n log D_n + log det(I_R + Uᵀ D⁻¹ U)

        Args:
            operator: The system A, read for λ when it is not fixed.

        Returns:
            The scalar log det P.
        """
        return _logdet(self._low_rank_update(operator))

    def sample(
        self,
        key: PRNGKeyArray,
        operator: lx.AbstractLinearOperator | None = None,
        *,
        shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch n"]:
        """Draw z = √D ⊙ ε_N + U ε_R ~ 𝒩(0, P).

        Args:
            key: PRNG key for ε_N and ε_R.
            operator: The system A, read for λ when it is not fixed.
            shape: Batch shape of the draws.

        Returns:
            The draws z, of shape ``(*shape, n)``.
        """
        approximation = self._low_rank_update(operator)

        def draws(key: PRNGKeyArray, num_draws: int) -> Float[Array, "draw n"]:
            # z = √D ⊙ ε_N + U ε_R
            return low_rank_update_draws(approximation, key, num_draws, diagonal_draws)

        return batched_draws(draws, key, shape)

    def approximation(
        self,
        operator: lx.AbstractLinearOperator | None = None,
    ) -> lx.AbstractLinearOperator:
        """P = diag(r + λ) + U Uᵀ as a `gaussx.LowRankUpdate`.

        Args:
            operator: The system A, read for λ when it is not fixed.

        Returns:
            P, PSD-tagged, with a Woodbury solve and a determinant-lemma
            log-determinant.
        """
        return self._low_rank_update(operator)
