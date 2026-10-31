"""Nyström-preconditioned stochastic log-determinant (G17).

A = K + σ² I is an N × N symmetric positive-definite system with a PSD part K
and a noise variance σ² > 0 (the ``shift``). A rank-k randomized Nyström
approximation K̂ = K Ω (Ωᵀ K Ω)⁺ Ωᵀ K = U diag(γ̂) Uᵀ of K, with test matrix Ω
(the letters of the paper, not the spatial domain), orthonormal U (N × k)
and eigenvalues γ̂_j, gives the log-determinant preconditioner
P = K̂ + σ² I ≈ A. The remainder log det(P^(−½) A P^(−½)) is the stochastic
Lanczos quadrature (SLQ) estimate over probes z^(i), i = 1, …, N_probe.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._primitives._samplers import SamplerName
from gaussx._randomized._nystrom import randomized_nystrom
from gaussx._strategies._base import AbstractLogdetStrategy
from gaussx._strategies._slq_logdet import SLQLogdet, _check_symmetric


class NystromLogdet(AbstractLogdetStrategy):
    """Nyström-preconditioned SLQ log-determinant of K + σ² I (Wenger et al., 2022).

        log det A = log det P + log det(P^(−½) A P^(−½)),
        log det P = ∑_j log(γ̂_j + σ²) + (N − k) log σ²

    for the system A = K + σ² I with K PSD, and the rank-k randomized
    Nyström approximation K̂ = U diag(γ̂) Uᵀ of K (`gaussx.randomized_nystrom`)
    in the preconditioner P = K̂ + σ² I.

    The first term is exact. The second is SLQ (`gaussx.SLQLogdet`) on the
    preconditioned operator A_prec = P^(−½) A P^(−½), whose eigenvalues
    cluster at one once k exceeds the effective dimension
    d_eff(σ²) = tr(K (K + σ² I)⁻¹). SLQ's variance scales with
    ‖log A_prec‖_F², so it collapses as A_prec → I, and its Lanczos quadrature
    converges in few steps. With

        P^(−½) = U (diag(γ̂ + σ²)^(−½) − σ⁻¹ I) Uᵀ + σ⁻¹ I,

    every product with A_prec costs one product with A and O(N k).

    **Covariance form only**: K̂ captures the *top* of K's spectrum, as for
    `gaussx.NystromPreconditioner`. For a precision-form GMRF system
    Q + Aᵀ W A the hard directions are the smallest eigenvalues; use
    `SLQLogdet` (or an exact sparse factorisation) there.

    ``operator`` is the full system A = K + σ² I, as for every other logdet
    strategy; the strategy sketches its PSD part as the matrix-free
    A − σ² I, plus a guard c = 10 N eps σ² that is removed again from γ̂,
    so the rounding of the subtraction cannot make it indefinite. So
    ``shift`` must be the σ² that ``operator`` contains (#345). ``shift``
    is a PyTree leaf, so it can be a traced, learned noise variance.

    Attributes:
        shift: Noise variance σ² > 0 contained in A.
        rank: Nyström rank k (``rank`` matvecs), clamped to N. Aim for
            k ≳ d_eff(σ²).
        oversample: Extra sketch columns (see `gaussx.randomized_nystrom`).
        num_probes: Number of SLQ probe vectors N_probe for the remainder.
        lanczos_order: Number of Lanczos steps per probe.
        seed: `logdet` without a ``key`` uses ``PRNGKey(seed)`` for both the
            sketch Ω and the probes z^(i) (common random numbers, as
            `SLQLogdet`).
        sampler: SLQ probe distribution: ``"signs"``, ``"normal"`` or
            ``"sphere"``.

    Raises:
        ValueError: If a concrete ``shift`` is not positive or ``rank < 1``.

    Examples:
        A noisy RBF Gram matrix, within 1% of the exact log-determinant:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> inputs = jnp.linspace(0.0, 10.0, 200)
        >>> gaps = einx.subtract("row, col -> row col", inputs, inputs)
        >>> kernel = jnp.exp(-0.5 * gaps**2)
        >>> A = lx.MatrixLinearOperator(
        ...     kernel + 0.1 * jnp.eye(200), lx.positive_semidefinite_tag
        ... )
        >>> estimate = gaussx.NystromLogdet(shift=0.1, rank=40).logdet(A)
        >>> exact = jnp.linalg.slogdet(A.as_matrix())[1]
        >>> bool(jnp.abs(estimate - exact) < 1e-2 * jnp.abs(exact))
        True

        A non-positive shift is rejected at construction:

        >>> gaussx.NystromLogdet(shift=0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: NystromLogdet needs shift > 0, got 0.0.

    References:
        Wenger, J., Pleiss, G., Hennig, P., Cunningham, J. P. & Gardner,
        J. R. (2022). Preconditioning for scalable Gaussian process
        hyperparameter optimization. *ICML*, PMLR 162, 23751-23780.

        Frangella, Z., Tropp, J. A. & Udell, M. (2023). Randomized Nyström
        preconditioning. *SIAM J. Matrix Anal. Appl.*, 44(2), 718-752.
    """

    shift: float | Float[Array, ""]
    rank: int = eqx.field(static=True, default=50)
    oversample: int = eqx.field(static=True, default=0)
    num_probes: int = eqx.field(static=True, default=10)
    lanczos_order: int = eqx.field(static=True, default=30)
    seed: int = eqx.field(static=True, default=0)
    sampler: SamplerName = eqx.field(static=True, default="signs")

    def __check_init__(self) -> None:
        if isinstance(self.shift, (int, float)) and self.shift <= 0:
            raise ValueError(f"NystromLogdet needs shift > 0, got {self.shift}.")
        if self.rank < 1:
            raise ValueError(f"NystromLogdet needs rank >= 1, got {self.rank}.")

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Estimate log det(K + σ² I) with a Nyström log-determinant preconditioner.

            log det A ≈ ∑_j log(γ̂_j + σ²) + (N − k) log σ²
                + (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁

        with T^(i) the Lanczos tridiagonal of P^(−½) A P^(−½).

        Algorithm:
            1. Split the key into a sketch key and a probe key.
            2. Sketch K + c I = A − (σ² − c) I at rank k and remove the
               guard, γ̂ ← max(γ̂ − c, 0).
            3. Sum the exact log det P = ∑_j log(γ̂_j + σ²) + (N − k) log σ².
            4. Run SLQ on A_prec = P^(−½) A P^(−½) and add the remainder.

        Args:
            operator: The symmetric PSD system A = K + σ² I, ``shift`` = σ²
                included; may be matrix-free.
            key: PRNG key for the sketch Ω and the probes z^(i). ``None``
                means ``jax.random.PRNGKey(seed)``.

        Returns:
            Scalar estimate of log det(K + σ² I).
        """
        _check_symmetric(operator, "NystromLogdet")
        if key is None:
            key = jax.random.PRNGKey(self.seed)
        sketch_key, probe_key = jax.random.split(key)
        size = operator.in_size()
        structure = operator.in_structure()
        noise_variance = jnp.asarray(self.shift, dtype=structure.dtype)

        # A − σ² I carries rounding noise of ~eps σ², indefinite once K is
        # small next to σ² (e.g. a zero kernel amplitude), which breaks the
        # Nyström Cholesky (gh-486 review). Sketch K + c I with c well above
        # that noise, then shift the eigenvalues back.
        # c = 10 N eps σ²
        guard = 10 * size * jnp.finfo(structure.dtype).eps * noise_variance
        # K + c I = A − (σ² − c) I
        psd_part = lx.FunctionLinearOperator(
            lambda v: operator.mv(v) - (noise_variance - guard) * v,
            structure,
            lx.positive_semidefinite_tag,
        )
        # K̂ = K Ω (Ωᵀ K Ω)⁺ Ωᵀ K = U diag(γ̂) Uᵀ
        approx = randomized_nystrom(
            psd_part, min(self.rank, size), oversample=self.oversample, key=sketch_key
        )
        basis = approx.U
        # γ̂ ← max(γ̂ − c, 0)
        eigenvalues = jnp.maximum(approx.d - guard, 0.0)
        rank = eigenvalues.shape[0]
        # ∑_j log(γ̂_j + σ²)
        logdet_low_rank = jnp.sum(jnp.log(eigenvalues + noise_variance))
        # log det P = ∑_j log(γ̂_j + σ²) + (N − k) log σ²
        logdet_preconditioner = logdet_low_rank + (size - rank) * jnp.log(
            noise_variance
        )
        # σ⁻¹
        inv_noise_std = 1.0 / jnp.sqrt(noise_variance)
        # (γ̂ + σ²)^(−½) − σ⁻¹
        correction = 1.0 / jnp.sqrt(eigenvalues + noise_variance) - inv_noise_std

        def inv_sqrt_preconditioner(v: Float[Array, " n"]) -> Float[Array, " n"]:
            # Uᵀ v
            projection = einsum(basis, v, "row rank, row -> rank")
            # ((γ̂ + σ²)^(−½) − σ⁻¹) ⊙ Uᵀ v
            coefficients = correction * projection
            # P^(−½) v = σ⁻¹ v + U (((γ̂ + σ²)^(−½) − σ⁻¹) ⊙ Uᵀ v)
            return inv_noise_std * v + einsum(
                basis, coefficients, "row rank, rank -> row"
            )

        # A_prec = P^(−½) A P^(−½)
        preconditioned = lx.FunctionLinearOperator(
            lambda v: inv_sqrt_preconditioner(operator.mv(inv_sqrt_preconditioner(v))),
            structure,
            (lx.symmetric_tag, lx.positive_semidefinite_tag),
        )
        # log det A_prec ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁
        remainder = SLQLogdet(
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
            seed=self.seed,
            sampler=self.sampler,
        ).logdet(preconditioned, key=probe_key)
        # log det A = log det P + log det A_prec
        return logdet_preconditioner + remainder
