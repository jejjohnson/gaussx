"""Standalone log-determinant strategies: SLQ, indefinite SLQ and dense.

A is a square N × N symmetric operator and log det A = tr(log A). Stochastic
Lanczos quadrature (SLQ) estimates the trace with probes z^(i) ∈ ℝᴺ,
i = 1, …, N_probe, 𝔼[z zᵀ] = I,

    log det A ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁,

with T^(i) the k × k Lanczos tridiagonal of A started at v₁ = z^(i) / ‖z^(i)‖.
The Lanczos recurrence

    α_j = v_jᵀ A v_j,   β_j+1 v_j+1 = A v_j − α_j v_j − β_j v_j−1,

gives T's diagonal α_j and off-diagonal β_j+1, j = 1, …, k, and the Gauss
quadrature reads e₁ᵀ f(T) e₁ = ∑_j [S]₁ⱼ² f(θ_j) from the Ritz values θ_j and
vectors S of T = S diag(θ) Sᵀ.
"""

from __future__ import annotations

import functools as ft
from collections.abc import Callable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import matfree.stochtrace
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import einsum
from gaussx._primitives._logdet import logdet as _logdet
from gaussx._primitives._samplers import SamplerName, resolve_sampler
from gaussx._strategies._base import AbstractLogdetStrategy


def _lanczos(
    matvec: Callable[..., Array],
    v0: Float[Array, " n"],
    order: int,
    *parameters,
) -> tuple[Float[Array, " k"], Float[Array, " k"]]:
    """Fully reorthogonalised Lanczos that stops cleanly at breakdown (gh-520).

        α_j = v_jᵀ A v_j,   β_j+1 v_j+1 = A v_j − α_j v_j − β_j v_j−1

    Once the Krylov space is exhausted (β_j+1 ≤ √eps ‖T‖) the remaining
    steps are padded with β = 0 and α = 1, a decoupled identity block that
    carries no quadrature weight. Every division and square root is
    double-``where`` guarded, so reverse mode through the breakdown is
    finite instead of 0 · ∞ = NaN.

    Algorithm:
        1. Store v_j as row j of the basis V and form
           w = A v_j − β_j v_j−1.
        2. Take α_j = v_jᵀ w and subtract α_j v_j.
        3. Reorthogonalise w against every stored row of V, twice.
        4. Track the scale ‖T‖ ≈ maxⱼ (|α_j| + β_j) and flag a breakdown
           once ‖w‖₂² ≤ (√eps ‖T‖)².
        5. Set β_j+1 = ‖w‖₂ and v_j+1 = w / β_j+1, or pad after a
           breakdown.

    Args:
        matvec: ``(v, *parameters) -> A v`` for a symmetric A.
        v0: Unit-norm starting vector v₁.
        order: Number of Lanczos steps k.
        *parameters: Passed to ``matvec``.

    Returns:
        ``(alphas, betas)``: the diagonal α_j of the k × k tridiagonal T
        and, in ``betas[:-1]``, its off-diagonal β_j+1.
    """
    size = v0.shape[0]
    dtype = v0.dtype
    # √eps
    sqrt_eps = jnp.sqrt(jnp.finfo(dtype).eps)
    zero = jnp.zeros((), dtype=dtype)

    def step(carry, j):
        basis, lanczos_vector, previous_vector, beta, broken, scale = carry
        # row j of V is v_j
        basis = basis.at[j].set(lanczos_vector)
        # A v_j
        operator_vector = matvec(lanczos_vector, *parameters)
        # w = A v_j − β_j v_j−1
        krylov_residual = operator_vector - beta * previous_vector
        # α_j = v_jᵀ w
        alpha = jnp.dot(lanczos_vector, krylov_residual)
        # w ← w − α_j v_j
        krylov_residual = krylov_residual - alpha * lanczos_vector
        # Full reorthogonalisation against every stored vector, twice.
        for _ in range(2):
            # V w
            overlaps = einsum(basis, krylov_residual, "step row, row -> step")
            # w ← w − Vᵀ V w
            krylov_residual = krylov_residual - einsum(
                basis, overlaps, "step row, step -> row"
            )
        # ‖w‖₂²
        squared = jnp.dot(krylov_residual, krylov_residual)
        # ‖T‖ ≈ maxⱼ (|α_j| + β_j)
        scale = jnp.maximum(scale, jnp.abs(alpha) + beta)
        # breakdown once ‖w‖₂² ≤ (√eps ‖T‖)²
        broken_now = broken | (squared <= (sqrt_eps * scale) ** 2)
        # Double where: the untaken √0 branch would make the gradient NaN.
        safe_squared = jnp.where(broken_now, jnp.ones((), dtype=dtype), squared)
        # β_j+1 = ‖w‖₂
        norm = jnp.sqrt(safe_squared)
        beta_next = jnp.where(broken_now, zero, norm)
        # v_j+1 = w / β_j+1
        next_vector = jnp.where(
            broken_now, jnp.zeros_like(krylov_residual), krylov_residual / norm
        )
        # α_j = 1 in the padded identity block
        alpha = jnp.where(broken, jnp.ones((), dtype=dtype), alpha)
        carry = (basis, next_vector, lanczos_vector, beta_next, broken_now, scale)
        return carry, (alpha, beta_next)

    init = (
        jnp.zeros((order, size), dtype=dtype),
        v0,
        jnp.zeros_like(v0),
        zero,
        jnp.zeros((), dtype=bool),
        zero,
    )
    _, (alphas, betas) = jax.lax.scan(step, init, jnp.arange(order))
    return alphas, betas


@ft.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _gauss_quadrature(
    fun: Callable[[Array], Array],
    dfun: Callable[[Array], Array],
    tridiagonal: Float[Array, "k k"],
) -> Float[Array, ""]:
    """Gauss quadrature e₁ᵀ f(T) e₁ for a symmetric tridiagonal T.

        e₁ᵀ f(T) e₁ = ∑_j [S]₁ⱼ² f(θ_j),   T = S diag(θ) Sᵀ

    Args:
        fun: Matrix function f, applied to the Ritz values θ_j.
        dfun: Its derivative f′, used only by the JVP.
        tridiagonal: Symmetric tridiagonal T, k × k.

    Returns:
        The scalar e₁ᵀ f(T) e₁.
    """
    del dfun
    # T = S diag(θ) Sᵀ
    ritz_values, ritz_vectors = jnp.linalg.eigh(tridiagonal)
    # ∑_j [S]₁ⱼ² f(θ_j)
    return jnp.sum(ritz_vectors[0] ** 2 * fun(ritz_values))


@_gauss_quadrature.defjvp
def _gauss_quadrature_jvp(fun, dfun, primals, tangents):
    """Daleckii-Krein derivative of e₁ᵀ f(T) e₁.

        d e₁ᵀ f(T) e₁ = ∑_jl c_j [Sᵀ dT S]_jl f[θ_j, θ_l] c_l,   c = Sᵀ e₁

    with the divided difference f[θ_j, θ_l] = (f(θ_j) − f(θ_l)) / (θ_j − θ_l),
    and f′ at the midpoint on ties. Unlike differentiating ``eigh``, which
    divides by eigenvalue gaps, this stays finite when Ritz values coincide,
    as they do in the padded block after a Lanczos breakdown (gh-520).

    Args:
        fun: Matrix function f.
        dfun: Its derivative f′.
        primals: Tuple ``(T,)``.
        tangents: Tuple ``(dT,)``.

    Returns:
        Tuple ``(e₁ᵀ f(T) e₁, d e₁ᵀ f(T) e₁)``.
    """
    (tridiagonal,) = primals
    (tangent,) = tangents
    # T = S diag(θ) Sᵀ
    ritz_values, ritz_vectors = jnp.linalg.eigh(tridiagonal)
    # c = Sᵀ e₁
    weights = ritz_vectors[0]
    # θ_j − θ_l
    gaps = einx.subtract("ritz, other -> ritz other", ritz_values, ritz_values)
    # f(θ_j) − f(θ_l)
    rises = einx.subtract(
        "ritz, other -> ritz other", fun(ritz_values), fun(ritz_values)
    )
    # (θ_j + θ_l) / 2
    mids = 0.5 * einx.add("ritz, other -> ritz other", ritz_values, ritz_values)
    # |θ_j| + |θ_l|
    sizes = einx.add(
        "ritz, other -> ritz other", jnp.abs(ritz_values), jnp.abs(ritz_values)
    )
    # tie: |θ_j − θ_l| ≤ √eps (|θ_j| + |θ_l|)
    tie = jnp.abs(gaps) <= jnp.sqrt(jnp.finfo(gaps.dtype).eps) * sizes
    # Double where: the untaken 0 / 0 branch would make the gradient NaN.
    safe_gaps = jnp.where(tie, jnp.ones_like(gaps), gaps)
    # f[θ_j, θ_l], f′((θ_j + θ_l) / 2) on ties
    divided = jnp.where(tie, dfun(mids), rises / safe_gaps)
    # Sᵀ dT S
    rotated = einsum(
        ritz_vectors,
        tangent,
        ritz_vectors,
        "row ritz, row col, col other -> ritz other",
    )
    # ∑_j [S]₁ⱼ² f(θ_j)
    primal = jnp.sum(weights**2 * fun(ritz_values))
    # ∑_jl c_j [Sᵀ dT S]_jl f[θ_j, θ_l] c_l
    tangent_out = einsum(
        weights, rotated * divided, weights, "ritz, ritz other, other ->"
    )
    return primal, tangent_out


def _quadform_integrand(
    order: int, fun: Callable[[Array], Array], dfun: Callable[[Array], Array]
):
    """SLQ integrand ‖z‖² e₁ᵀ f(T) e₁ in matfree's integrand signature.

        z ↦ ‖z‖² e₁ᵀ f(T) e₁,   T the Lanczos tridiagonal from v₁ = z / ‖z‖

    Args:
        order: Number of Lanczos steps k.
        fun: Matrix function f.
        dfun: Its derivative f′.

    Returns:
        A function ``(matvec, z, *parameters) -> ‖z‖² e₁ᵀ f(T) e₁``.
    """

    def quadform(matvec, v0, *parameters):
        # ‖z‖₂
        length = jnp.linalg.norm(v0)
        # α_j, β_j+1 from v₁ = z / ‖z‖₂
        alphas, betas = _lanczos(matvec, v0 / length, order, *parameters)
        off_diagonal = betas[:-1]
        # T = tridiag(β, α, β)
        tridiagonal = (
            jnp.diag(alphas) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
        )
        # ‖z‖² e₁ᵀ f(T) e₁
        return length**2 * _gauss_quadrature(fun, dfun, tridiagonal)

    return quadform


def _reciprocal(x: Array) -> Array:
    """Derivative of log: x ↦ 1 / x."""
    return 1.0 / x


def _log_abs(x: Array) -> Array:
    """Matrix function of the indefinite log-determinant: x ↦ log |x|."""
    return jnp.log(jnp.abs(x))


def _logdet_integrand(order: int):
    """SLQ integrand for log det A of a PSD operator, with f = log.

    Args:
        order: Number of Lanczos steps k.

    Returns:
        The integrand ``(matvec, z, *parameters) -> ‖z‖² e₁ᵀ log(T) e₁``.
    """
    return _quadform_integrand(order, jnp.log, _reciprocal)


def _logabsdet_integrand(order: int):
    """SLQ integrand for log |det A| of a symmetric operator, f = log |·|.

    Args:
        order: Number of Lanczos steps k.

    Returns:
        The integrand ``(matvec, z, *parameters) -> ‖z‖² e₁ᵀ log|T| e₁``.
    """
    return _quadform_integrand(order, _log_abs, _reciprocal)


def _slq_estimators(
    integrand,
    n: int,
    num_probes: int,
    sampler: SamplerName,
    dtype: DTypeLike,
) -> tuple[Callable, Callable]:
    """Build the point and the mean-and-SEM SLQ estimators on one sampler.

        (1 / N_probe) ∑_i g(z^(i)),   z^(i) ~ p(z)

    Args:
        integrand: The SLQ integrand g, e.g. ‖z‖² e₁ᵀ log(T) e₁.
        n: Probe length N.
        num_probes: Number of probes N_probe.
        sampler: Probe distribution name.
        dtype: The operator's dtype, for the probes and the result.

    Returns:
        Tuple ``(point, with_sem)`` of functions ``(matvec, key) -> …``,
        the estimate and the estimate with its standard error.
    """
    probe_fn = resolve_sampler(sampler, n, num_probes, dtype)
    point_raw = matfree.stochtrace.estimator_monte_carlo(integrand, probe_fn)
    with_sem_raw = matfree.stochtrace.estimator_monte_carlo_mean_and_sem(
        integrand, probe_fn
    )

    # The probes and matvecs run in the operator's dtype, but matfree's funm
    # integrand projects onto ``np.eye(k)[0]``, built in JAX's default float,
    # so the scalar comes back float64 under x64. Cast it back.
    def point(matvec, key):
        return jnp.asarray(point_raw(matvec, key), dtype=dtype)

    def with_sem(matvec, key):
        return jax.tree.map(
            lambda x: jnp.asarray(x, dtype=dtype), with_sem_raw(matvec, key)
        )

    return point, with_sem


def _check_symmetric(operator: lx.AbstractLinearOperator, name: str) -> None:
    """Warn (for now) when SLQ gets an operator not tagged symmetric (gh-402).

    Symmetric Lanczos on a non-symmetric operator returns a silently wrong
    log-determinant. The check is on the tags, so it is free under ``jit``.

    Args:
        operator: Operator A handed to the strategy.
        name: Strategy name for the warning.
    """
    if not lx.is_symmetric(operator):
        warn_deprecated(
            f"{name} got an operator that is not tagged symmetric; symmetric "
            "Lanczos then gives a wrong log-determinant. Tag it with "
            "lineax.symmetric_tag or lineax.positive_semidefinite_tag. "
            "This will raise a ValueError in gaussx 0.7.0."
        )


class SLQLogdet(AbstractLogdetStrategy):
    """Stochastic log-determinant by stochastic Lanczos quadrature (SLQ).

        log det A = tr(log A) ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁

    for a PSD operator A, with T^(i) the k × k Lanczos tridiagonal of A
    started at z^(i) / ‖z^(i)‖, k = min(``lanczos_order``, N). The Lanczos
    run is fully reorthogonalised and stops cleanly at breakdown; the
    probes are sign-flip (Rademacher) vectors by default.

    With no ``key``, `logdet` uses ``PRNGKey(seed)``, so every call sees the
    same probes: common random numbers, which suit stochastic-gradient
    training but give no variance reduction when estimates are averaged and
    a fixed pseudo-likelihood under MCMC. Pass a fresh ``key`` (or wrap a
    strategy in `gaussx.KeyedSolver`) to decorrelate calls.

    Attributes:
        num_probes: Number of probe vectors N_probe.
        lanczos_order: Number of Lanczos steps k, clamped to N.
        seed: Seed of the probes z^(i) when `logdet` gets no ``key``.
        sampler: Probe distribution: ``"signs"``, ``"normal"`` or
            ``"sphere"``.

    Examples:
        On a diagonal operator one Lanczos step per probe is exact:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 4.0]))
        >>> estimate = gaussx.SLQLogdet(num_probes=4).logdet(A)
        >>> round(float(estimate), 3), round(float(jnp.log(8.0)), 3)
        (2.079, 2.079)

        The standard error across probes comes with ``logdet_and_error``:

        >>> estimate, error = gaussx.SLQLogdet().logdet_and_error(A)
        >>> error.shape
        ()
    """

    num_probes: int = eqx.field(static=True, default=20)
    lanczos_order: int = eqx.field(static=True, default=30)
    seed: int = eqx.field(static=True, default=0)
    sampler: SamplerName = eqx.field(static=True, default="signs")

    def _integrand(self, n: int):
        """SLQ integrand with k = min(``lanczos_order``, N) Lanczos steps.

        Args:
            n: Operator size N.

        Returns:
            The integrand ``(matvec, z) -> ‖z‖² e₁ᵀ log(T) e₁``.
        """
        return _logdet_integrand(min(self.lanczos_order, n))

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Estimate log det A by stochastic Lanczos quadrature.

            log det A ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁

        Algorithm:
            1. Warn when A is not tagged symmetric.
            2. Draw N_probe probes z^(i) from ``key`` (or ``PRNGKey(seed)``).
            3. Run k Lanczos steps from each z^(i) / ‖z^(i)‖ and average
               the Gauss quadratures ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁.

        Args:
            operator: Symmetric positive-definite operator A, N × N.
            key: PRNG key for the probes z^(i). ``None`` uses
                ``jax.random.PRNGKey(seed)``: the same probes on every call.

        Returns:
            Scalar estimate of log det A.
        """
        _check_symmetric(operator, "SLQLogdet")
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        size = operator.in_size()
        point, _ = _slq_estimators(
            self._integrand(size),
            size,
            self.num_probes,
            self.sampler,
            operator.in_structure().dtype,
        )
        return point(operator.mv, key)

    def logdet_and_error(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Estimate log det A and the standard error of the probe mean.

            log det A ≈ ḡ = (1 / N_probe) ∑_i g^(i),
            SEM = std(g^(i)) / √N_probe,   g^(i) = ‖z^(i)‖² e₁ᵀ log(T^(i)) e₁

        Args:
            operator: Symmetric positive-definite operator A, N × N.
            key: PRNG key for the probes z^(i). ``None`` uses
                ``jax.random.PRNGKey(seed)``.

        Returns:
            Tuple ``(estimate, standard_error)``: the SLQ estimate and the
            standard error of the mean across probes.
        """
        _check_symmetric(operator, "SLQLogdet")
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        size = operator.in_size()
        _, with_sem = _slq_estimators(
            self._integrand(size),
            size,
            self.num_probes,
            self.sampler,
            operator.in_structure().dtype,
        )
        return with_sem(operator.mv, key)


class IndefiniteSLQLogdet(AbstractLogdetStrategy):
    """Stochastic log |det(A + σ I)| for symmetric, possibly indefinite, A.

        log |det(A + σ I)| ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log|T^(i)| e₁

    with T^(i) the Lanczos tridiagonal of A + σ I and σ the diagonal
    ``shift``. Like `SLQLogdet`, but the matrix function is log |θ|, so it
    works on indefinite and negative-definite operators.

    Attributes:
        num_probes: Number of probe vectors N_probe.
        lanczos_order: Number of Lanczos steps k, clamped to N.
        shift: Diagonal shift σ, applied before the log-determinant.
        seed: Seed of the probes z^(i) when `logdet` gets no ``key``.
        sampler: Probe distribution: ``"signs"``, ``"normal"`` or
            ``"sphere"``.

    Examples:
        A diagonal with a negative entry:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([-2.0, 1.0, 4.0]))
        >>> estimate = gaussx.IndefiniteSLQLogdet(num_probes=4).logdet(A)
        >>> round(float(estimate), 3), round(float(jnp.log(8.0)), 3)
        (2.079, 2.079)
    """

    num_probes: int = eqx.field(static=True, default=20)
    lanczos_order: int = eqx.field(static=True, default=30)
    shift: float = eqx.field(static=True, default=0.0)
    seed: int = eqx.field(static=True, default=0)
    sampler: SamplerName = eqx.field(static=True, default="signs")

    def _shifted_matvec(self, operator: lx.AbstractLinearOperator):
        """The shifted product v ↦ (A + σ I) v.

        Args:
            operator: Symmetric operator A.

        Returns:
            A function ``v -> A v + σ v``.
        """
        shift = self.shift

        def matvec(v):
            # (A + σ I) v
            return operator.mv(v) + shift * v

        return matvec

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Estimate log |det(A + σ I)| by stochastic Lanczos quadrature.

            log |det(A + σ I)| ≈ (1 / N_probe) ∑_i ‖z^(i)‖² e₁ᵀ log|T^(i)| e₁

        Algorithm:
            1. Warn when A is not tagged symmetric.
            2. Draw N_probe probes z^(i) from ``key`` (or ``PRNGKey(seed)``).
            3. Run k Lanczos steps of A + σ I from each z^(i) / ‖z^(i)‖ and
               average ‖z^(i)‖² e₁ᵀ log|T^(i)| e₁.

        Args:
            operator: Symmetric operator A, N × N.
            key: PRNG key for the probes z^(i). ``None`` uses
                ``jax.random.PRNGKey(seed)``.

        Returns:
            Scalar estimate of log |det(A + σ I)|.
        """
        _check_symmetric(operator, "IndefiniteSLQLogdet")
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        size = operator.in_size()
        order = min(self.lanczos_order, size)
        point, _ = _slq_estimators(
            _logabsdet_integrand(order),
            size,
            self.num_probes,
            self.sampler,
            operator.in_structure().dtype,
        )
        return point(self._shifted_matvec(operator), key)

    def logdet_and_error(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Estimate log |det(A + σ I)| and the standard error of the mean.

            log |det(A + σ I)| ≈ ḡ = (1 / N_probe) ∑_i g^(i),
            SEM = std(g^(i)) / √N_probe,   g^(i) = ‖z^(i)‖² e₁ᵀ log|T^(i)| e₁

        Args:
            operator: Symmetric operator A, N × N.
            key: PRNG key for the probes z^(i). ``None`` uses
                ``jax.random.PRNGKey(seed)``.

        Returns:
            Tuple ``(estimate, standard_error)``.
        """
        _check_symmetric(operator, "IndefiniteSLQLogdet")
        if key is None:
            key = jax.random.PRNGKey(self.seed)

        size = operator.in_size()
        order = min(self.lanczos_order, size)
        _, with_sem = _slq_estimators(
            _logabsdet_integrand(order),
            size,
            self.num_probes,
            self.sampler,
            operator.in_structure().dtype,
        )
        return with_sem(self._shifted_matvec(operator), key)


class DenseLogdet(AbstractLogdetStrategy):
    """Exact log-determinant by gaussx's structural dispatch.

        log |det A|

    Delegates to `gaussx.logdet`, which picks the algorithm from the
    operator's structure (diagonal, `BlockDiag`, `Kronecker`,
    `LowRankUpdate`, or the dense fallback).

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> A = lx.DiagonalLinearOperator(jnp.array([2.0, 4.0]))
        >>> round(float(gaussx.DenseLogdet().logdet(A)), 4)
        2.0794
    """

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        """Exact log |det A| through structural dispatch.

            log |det A|

        Args:
            operator: Square operator A.
            key: Ignored; the result is deterministic.

        Returns:
            Scalar log |det A|.
        """
        return _logdet(operator)
