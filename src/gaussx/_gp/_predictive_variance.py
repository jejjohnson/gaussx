"""Simulation-based predictive variances: SBPV and SPV (Gyger et al., 2026, §4.2).

A predictive covariance at N_* test points splits into a part known in
closed form and a part that needs the inverse of an N × N precision,

    Ω = Ω_det − Cᵀ M C,   M = (W + Σ⁻¹)⁻¹,

with Ω_det, C ∈ ℝ^{N × N_*} and the precision W + Σ⁻¹ given by the model
(for exact GP regression, Ω_det = diag(K_**), C = K_X*, M = (K_XX + σ² I)⁻¹).
Only diag(Ω) is wanted, and the two estimators here get diag(Cᵀ M C) from
draws or probes, with an error that falls as 1 / √ℓ in the number of draws ℓ
(their Propositions 3 and 4: unbiased and consistent).

Three routes to diag(Ω) in gaussx, and when to use each:

- `gaussx.love_cache` (LOVE): a rank-k Lanczos approximation of M, cheap
  per test point, with the certified over-estimate of
  `gaussx.love_variance_error_bound`. Inverting reverses the spectrum, so M
  needs a high rank when the kernel is smooth and the noise small; when the
  bound is too loose, use the estimators here.
- `predictive_variance_sbpv` and `predictive_variance_spv`: unbiased
  estimates for the dense and low-rank-plus-dense cases, at a cost of ℓ
  solves; SBPV is the more accurate at an equal count.
- `gaussx.selected_inverse` (or `gaussx.takahashi`) on a sparse precision
  (Vecchia, GMRF): the entries of M on the factor's pattern exactly, with no
  sampling, which beats both estimators where it applies.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import matfree.stochtrace
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._primitives._samplers import resolve_sampler


def predictive_variance_sbpv(
    key: PRNGKeyArray,
    *,
    deterministic_part: Float[Array, " n_star"],
    sample_fn: Callable[[PRNGKeyArray], Float[Array, " n"]],
    solve_fn: Callable[[Float[Array, " n"]], Float[Array, " n_star"]],
    num_samples: int = 100,
) -> Float[Array, " n_star"]:
    """Simulation-based predictive variance (SBPV), unbiased and consistent.

        r^(i) ~ 𝒩(0, W + Σ⁻¹),   s^(i) = Cᵀ (W + Σ⁻¹)⁻¹ r^(i) ~ 𝒩(0, Cᵀ M C),
        diag(Ω) ≈ max(Ω_det − (1 / ℓ) ∑_i s^(i) ⊙ s^(i), 0)

    ``sample_fn`` draws from the *precision* W + Σ⁻¹, which is cheap when
    both terms are (a diagonal W and a prior sample), and ``solve_fn`` applies
    Cᵀ (W + Σ⁻¹)⁻¹, one solve per draw. The covariance of s^(i) is then
    Cᵀ M (W + Σ⁻¹) M C = Cᵀ M C, so the sample mean of s ⊙ s is unbiased for
    diag(Cᵀ M C) (Gyger, Furrer & Sigrist, 2026, §4.2, Algorithm 1). The
    estimate is clamped at zero: the stochastic part can push a small
    variance below it.

    The signature is operator-agnostic: the dense, sparse-variational and
    other models pass their own draws and solves. See the module docstring
    for the choice between LOVE (`gaussx.love_cache`), these estimators
    and the exact sparse route (`gaussx.selected_inverse`).

    Args:
        key: PRNG key, split into one key per draw.
        deterministic_part: diag(Ω_det), the closed-form part.
        sample_fn: ``key -> r``, one draw r ~ 𝒩(0, W + Σ⁻¹).
        solve_fn: ``r -> Cᵀ (W + Σ⁻¹)⁻¹ r``.
        num_samples: Number of draws ℓ.

    Returns:
        The estimated predictive variances diag(Ω), non-negative.

    Examples:
        GP regression on a dense kernel, where M = (K_XX + σ² I)⁻¹, so the
        draws come from K_XX + σ² I (σ² = 0.1):

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> X = jnp.linspace(0.0, 1.0, 20)
        >>> Xs = jnp.linspace(0.0, 1.0, 5)
        >>> def k(a, b):
        ...     return jnp.exp(-0.5 * einx.subtract("i, j -> i j", a, b) ** 2 / 0.04)
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(k(X, X) + 0.1 * jnp.eye(20), psd)
        >>> def sample_fn(key):
        ...     return gaussx.sample_mvn(jnp.zeros(20), A, key=key)[0]
        >>> def solve_fn(r):
        ...     return k(Xs, X) @ gaussx.solve(A, r)
        >>> estimate = gaussx.predictive_variance_sbpv(
        ...     jr.key(0),
        ...     deterministic_part=jnp.ones(5),
        ...     sample_fn=sample_fn,
        ...     solve_fn=solve_fn,
        ...     num_samples=400,
        ... )
        >>> cross = k(Xs, X)
        >>> exact = 1.0 - jnp.diag(cross @ jnp.linalg.solve(A.as_matrix(), cross.T))
        >>> estimate.shape, bool(jnp.all(jnp.abs(estimate - exact) < 0.1))
        ((5,), True)

    References:
        Gyger, Furrer & Sigrist (2026), JMLR 27(115), §4.2, Algorithm 1,
        Proposition 3.
    """
    estimate, _ = sbpv_mean_and_error(
        key,
        deterministic_part=deterministic_part,
        sample_fn=sample_fn,
        solve_fn=solve_fn,
        num_samples=num_samples,
    )
    # clamped at zero, as base_conditional is (gh-363)
    return jnp.maximum(estimate, 0.0)


def predictive_variance_spv(
    key: PRNGKeyArray,
    *,
    deterministic_part: Float[Array, " n_star"],
    quadratic_fn: Callable[[Float[Array, " n_star"]], Float[Array, " n_star"]],
    num_probes: int = 100,
) -> Float[Array, " n_star"]:
    """Stochastic predictive variance (SPV) by Rademacher diagonal probes.

        z^(i) ∈ {−1, 1}^{N_*},   diag(Cᵀ M C) ≈ (1 / ℓ) ∑_i z^(i) ⊙ Cᵀ M C z^(i),
        diag(Ω) ≈ max(Ω_det − (1 / ℓ) ∑_i z^(i) ⊙ Cᵀ M C z^(i), 0)

    The Bekas, Kokiopoulou & Saad (2007) diagonal estimator, through
    matfree's `monte_carlo_diagonal`, on the product v ↦ Cᵀ M C v (two
    products with C and one solve per probe). It needs no draw from the
    precision, unlike `predictive_variance_sbpv`, which is the more accurate
    at an equal count (Gyger, Furrer & Sigrist, 2026, §7.3). The estimate is
    clamped at zero. See the module docstring for the choice between LOVE,
    these estimators and the exact sparse route (`gaussx.selected_inverse`).

    Args:
        key: PRNG key for the probes.
        deterministic_part: diag(Ω_det), the closed-form part.
        quadratic_fn: ``v -> Cᵀ (W + Σ⁻¹)⁻¹ C v`` on the N_* test points.
        num_probes: Number of probes ℓ.

    Returns:
        The estimated predictive variances diag(Ω), non-negative.

    Examples:
        The same GP as in `predictive_variance_sbpv`:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> X = jnp.linspace(0.0, 1.0, 20)
        >>> Xs = jnp.linspace(0.0, 1.0, 5)
        >>> def k(a, b):
        ...     return jnp.exp(-0.5 * einx.subtract("i, j -> i j", a, b) ** 2 / 0.04)
        >>> psd = lx.positive_semidefinite_tag
        >>> A = lx.MatrixLinearOperator(k(X, X) + 0.1 * jnp.eye(20), psd)
        >>> def quadratic_fn(v):
        ...     return k(Xs, X) @ gaussx.solve(A, k(X, Xs) @ v)
        >>> estimate = gaussx.predictive_variance_spv(
        ...     jr.key(0), deterministic_part=jnp.ones(5), quadratic_fn=quadratic_fn
        ... )
        >>> estimate.shape, bool(jnp.all(estimate >= 0.0))
        ((5,), True)

    References:
        Bekas, C., Kokiopoulou, E. & Saad, Y. (2007). An estimator for the
        diagonal of a matrix. Applied Numerical Mathematics 57, 1214-1229.
        Gyger, Furrer & Sigrist (2026), JMLR 27(115), §4.2, Algorithm 2,
        Proposition 4.
    """
    estimate, _ = spv_mean_and_error(
        key,
        deterministic_part=deterministic_part,
        quadratic_fn=quadratic_fn,
        num_probes=num_probes,
    )
    # clamped at zero, as base_conditional is (gh-363)
    return jnp.maximum(estimate, 0.0)


def sbpv_mean_and_error(
    key: PRNGKeyArray,
    *,
    deterministic_part: Float[Array, " n_star"],
    sample_fn: Callable[[PRNGKeyArray], Float[Array, " n"]],
    solve_fn: Callable[[Float[Array, " n"]], Float[Array, " n_star"]],
    num_samples: int,
) -> tuple[Float[Array, " n_star"], Float[Array, " n_star"]]:
    """The unclamped SBPV estimate and its standard error, per test point.

        Ω_det − (1 / ℓ) ∑_i s^(i) ⊙ s^(i),   SEM = std(s ⊙ s) / √ℓ

    Args:
        key: PRNG key, split into one key per draw.
        deterministic_part: diag(Ω_det).
        sample_fn: ``key -> r``, one draw r ~ 𝒩(0, W + Σ⁻¹).
        solve_fn: ``r -> Cᵀ (W + Σ⁻¹)⁻¹ r``.
        num_samples: Number of draws ℓ.

    Returns:
        Tuple ``(estimate, standard_error)``; the estimate is not clamped,
        so a caller can test it against its own bound.
    """

    def sampler(sample_key: PRNGKeyArray) -> Float[Array, "draw n"]:
        # r^(i) ~ 𝒩(0, W + Σ⁻¹), one row per draw
        return jax.vmap(sample_fn)(jr.split(sample_key, num_samples))

    def integrand(solve, draw: Float[Array, " n"]) -> Float[Array, " n_star"]:
        # s^(i) = Cᵀ (W + Σ⁻¹)⁻¹ r^(i)
        projected = solve(draw)
        # s^(i) ⊙ s^(i)
        return projected * projected

    estimator = matfree.stochtrace.estimator_monte_carlo_mean_and_sem(
        integrand, sampler
    )
    # (1 / ℓ) ∑_i s^(i) ⊙ s^(i) and its standard error
    stochastic, error = estimator(solve_fn, key)
    # Ω_det − diag(Cᵀ M C)
    return deterministic_part - stochastic, error


def spv_mean_and_error(
    key: PRNGKeyArray,
    *,
    deterministic_part: Float[Array, " n_star"],
    quadratic_fn: Callable[[Float[Array, " n_star"]], Float[Array, " n_star"]],
    num_probes: int,
) -> tuple[Float[Array, " n_star"], Float[Array, " n_star"]]:
    """The unclamped SPV estimate and its standard error, per test point.

        Ω_det − (1 / ℓ) ∑_i z^(i) ⊙ Cᵀ M C z^(i),   SEM = std(z ⊙ Cᵀ M C z) / √ℓ

    Args:
        key: PRNG key for the sign probes.
        deterministic_part: diag(Ω_det).
        quadratic_fn: ``v -> Cᵀ (W + Σ⁻¹)⁻¹ C v``.
        num_probes: Number of probes ℓ.

    Returns:
        Tuple ``(estimate, standard_error)``, the estimate not clamped.
    """
    sampler = resolve_sampler(
        "signs", deterministic_part.shape[0], num_probes, deterministic_part.dtype
    )
    estimator = matfree.stochtrace.estimator_monte_carlo_mean_and_sem(
        matfree.stochtrace.monte_carlo_diagonal(), sampler
    )
    # (1 / ℓ) ∑_i z^(i) ⊙ Cᵀ M C z^(i) and its standard error
    stochastic, error = estimator(quadratic_fn, key)
    # Ω_det − diag(Cᵀ M C)
    return deterministic_part - stochastic, error
