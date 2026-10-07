"""Nyström-preconditioned stochastic log-determinant (G17)."""

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
    r"""Nyström-preconditioned log-determinant of $A + \mu I$ (Wenger et al., 2022).

    For a covariance-form system $A + \mu I$ ($A$ PSD, e.g. $K + \sigma^2 I$)
    and the rank-$k$ randomized Nyström approximation $\hat A = U\hat\Lambda
    U^\top$ of $A$ (`gaussx.randomized_nystrom`), the preconditioner
    $P = \hat A + \mu I$ splits the log-determinant as

    $$
    \log|A + \mu I| = \log|P| + \log\big|P^{-1/2}(A + \mu I)P^{-1/2}\big|,
    \qquad
    \log|P| = \sum_{i=1}^{k}\log(\hat\lambda_i + \mu) + (n - k)\log\mu .
    $$

    The first term is exact. The second is SLQ (`gaussx.SLQLogdet`) on the
    preconditioned operator, whose eigenvalues cluster at one once $k$
    exceeds the effective dimension $d_\text{eff}(\mu) = \operatorname{tr}(A
    (A + \mu I)^{-1})$; SLQ's variance scales with $\|\log M\|_F^2$, so it
    collapses as $M \to I$, and its Lanczos quadrature converges in few
    steps. With

    $$
    P^{-1/2} = U\big((\hat\Lambda + \mu I)^{-1/2} - \mu^{-1/2} I\big)U^\top
    + \mu^{-1/2} I,
    $$

    every matvec with $M$ costs one matvec with $A + \mu I$ and $O(nk)$.

    ```text
    key₁, key₂ = split(key)
    U, Λ̂ = randomized_nystrom(A + μI − μI, rank; key₁)     # rank matvecs
    log|P| = Σ log(λ̂ᵢ + μ) + (n − k) log μ
    M = P^{-1/2} (A + μI) P^{-1/2}
    return log|P| + SLQ(M; num_probes, lanczos_order; key₂)
    ```

    **Covariance form only**: $\hat A$ captures the *top* of $A$'s
    spectrum, as for `gaussx.NystromPreconditioner`. For a precision-form
    GMRF system $Q + A^\top W A$ the hard directions are the smallest
    eigenvalues; use `SLQLogdet` (or an exact sparse factorisation) there.

    ``operator`` is the full system $A + \mu I$, as for every other logdet
    strategy; the strategy sketches its PSD part as the matrix-free
    ``operator − shift·I``, so ``shift`` must be the $\mu$ that
    ``operator`` contains (#345). ``shift`` is a PyTree leaf, so it can be
    a traced, learned noise variance.

    Attributes:
        shift: $\mu > 0$, e.g. the noise variance $\sigma^2$.
        rank: Nyström rank $k$ (``rank`` matvecs), clamped to $n$. Aim for
            $k \gtrsim d_\text{eff}(\mu)$.
        oversample: Extra sketch columns (see `gaussx.randomized_nystrom`).
        num_probes: SLQ probe vectors for the remainder.
        lanczos_order: Lanczos steps per probe.
        seed: ``logdet`` without a ``key`` uses ``PRNGKey(seed)`` for both
            the sketch and the probes (common random numbers, as
            `SLQLogdet`).
        sampler: SLQ probe distribution (``"signs"``, ``"normal"``,
            ``"sphere"``).

    References:
        Wenger, J., Pleiss, G., Hennig, P., Cunningham, J. P. & Gardner,
        J. R. (2022). Preconditioning for scalable Gaussian process
        hyperparameter optimization. *ICML*, PMLR 162, 23751-23780.

        Frangella, Z., Tropp, J. A. & Udell, M. (2023). Randomized Nyström
        preconditioning. *SIAM J. Matrix Anal. Appl.*, 44(2), 718-752.

    Examples:
        >>> import einx, jax.numpy as jnp, lineax as lx
        >>> import gaussx as gx
        >>> x = jnp.linspace(0.0, 10.0, 200)
        >>> K = jnp.exp(-0.5 * einx.subtract("i, j -> i j", x, x) ** 2)
        >>> A = lx.MatrixLinearOperator(
        ...     K + 0.1 * jnp.eye(200), lx.positive_semidefinite_tag
        ... )
        >>> est = gx.NystromLogdet(shift=0.1, rank=40).logdet(A)
        >>> exact = jnp.linalg.slogdet(A.as_matrix())[1]
        >>> bool(jnp.abs(est - exact) < 1e-2 * jnp.abs(exact))
        True
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
        r"""Estimate $\log|A + \mu I|$.

        Args:
            operator: The symmetric PSD system $A + \mu I$, ``shift`` $= \mu$
                included; may be matrix-free.
            key: PRNG key for the sketch and the probes. ``None`` means
                ``jax.random.PRNGKey(seed)``.

        Returns:
            Scalar estimate of $\log|A + \mu I|$.
        """
        _check_symmetric(operator, "NystromLogdet")
        if key is None:
            key = jax.random.PRNGKey(self.seed)
        sketch_key, probe_key = jax.random.split(key)
        n = operator.in_size()
        structure = operator.in_structure()
        mu = jnp.asarray(self.shift, dtype=structure.dtype)

        psd_part = lx.FunctionLinearOperator(
            lambda v: operator.mv(v) - mu * v,
            structure,
            lx.positive_semidefinite_tag,
        )
        approx = randomized_nystrom(
            psd_part, min(self.rank, n), oversample=self.oversample, key=sketch_key
        )
        U, eigenvalues = approx.U, approx.d
        k = eigenvalues.shape[0]
        logdet_preconditioner = jnp.sum(jnp.log(eigenvalues + mu)) + (n - k) * jnp.log(
            mu
        )
        inv_sqrt_mu = 1.0 / jnp.sqrt(mu)
        correction = 1.0 / jnp.sqrt(eigenvalues + mu) - inv_sqrt_mu

        def inv_sqrt_preconditioner(v: Float[Array, " n"]) -> Float[Array, " n"]:
            coefficients = correction * einsum(U, v, "n k, n -> k")
            return inv_sqrt_mu * v + einsum(U, coefficients, "n k, k -> n")

        preconditioned = lx.FunctionLinearOperator(
            lambda v: inv_sqrt_preconditioner(operator.mv(inv_sqrt_preconditioner(v))),
            structure,
            (lx.symmetric_tag, lx.positive_semidefinite_tag),
        )
        remainder = SLQLogdet(
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
            seed=self.seed,
            sampler=self.sampler,
        ).logdet(preconditioned, key=probe_key)
        return logdet_preconditioner + remainder
