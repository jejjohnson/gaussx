"""Ensemble transform Kalman filter (ETKF): deterministic square-root analysis."""

from __future__ import annotations

import functools

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange, reduce
from gaussx._inference._enkf import _check_observation_shapes
from gaussx._linalg._linalg import solve_rows
from gaussx._linalg._symmetrize import symmetrize
from gaussx._strategies._base import AbstractSolverStrategy


def etkf_transform(
    obs_particles: Float[Array, "J M"],
    y: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    inflation: float | Float[Array, ""] = 1.0,
    solver: AbstractSolverStrategy | None = None,
    low_rank: bool | None = None,
) -> tuple[Float[Array, " J"], Float[Array, "J J"]]:
    """Ensemble Transform Kalman Filter (ETKF) analysis weights.

        Λ_w = ((|ℰ| − 1) / ζ) I + H̃ Σ_ε⁻¹ H̃ᵀ,   Σ_w = Λ_w⁻¹
        w̄ = Σ_w H̃ Σ_ε⁻¹ r̄,   W = ((|ℰ| − 1) Σ_w)^½

    Deterministic, perturbation-free ensemble square-root analysis in the
    |ℰ|-dimensional ensemble space, where H̃ stacks the observation anomalies
    h̃⁽ᵉ⁾ = h⁽ᵉ⁾ − h̄ as rows, r̄ = y − h̄ is the mean residual, ζ is the
    multiplicative ``inflation`` and W is the **symmetric** square root. With
    the state anomalies X̃ stacked as rows, the analysis ensemble is

        x̄_t|t = x̄_t|t−1 + X̃ᵀ w̄,   X̃_t|t = W X̃

    The symmetric (eigendecomposition) square root is required, not a Cholesky
    factor: the observation anomalies have zero mean, so the all-ones vector is
    an eigenvector of W with eigenvalue √ζ, which makes the transform exactly
    mean-preserving (∑ₑ x̃⁽ᵉ⁾_t|t = 0).

    Σ_ε⁻¹ is applied to the |ℰ| + 1 right-hand sides [H̃, r̄] in one `solve_rows`
    call, so a structured Σ_ε (diagonal, `gaussx.BlockDiag`, `gaussx.Kronecker`,
    ...) keeps its own solve and is never materialised, and a dense Σ_ε is
    factored once. The ensemble-space algebra then takes a single
    eigendecomposition Λ_w = Φ diag(γ) Φᵀ, from which Σ_w = Φ diag(1 / γ) Φᵀ and
    W = Φ diag(√((|ℰ| − 1) / γ)) Φᵀ are read off, with no explicit inverse. Its
    derivative is taken in closed form in the eigenbasis (Daleckii-Krein), which
    stays finite at the repeated prior eigenvalue (|ℰ| − 1) / ζ that has
    multiplicity |ℰ| − D_y whenever D_y < |ℰ|.

    Λ_w exceeds its prior term by rank at most D_y, so with D_y < |ℰ| the
    default replaces that (|ℰ|, |ℰ|) eigendecomposition with a thin QR of H̃ and
    a (D_y, D_y) one. The two routes agree to round-off, and in float32 the
    low-rank route stays accurate where the other turns non-finite. Use
    `etkf_analysis` to avoid forming W at all.

    Algorithm:
        1. Centre the observation ensemble: H̃ and r̄ = y − h̄.
        2. Solve Σ_ε⁻¹ against [H̃, r̄] in one call.
        3. Read Σ_w and Λ_w^−½ off the spectrum of Λ_w, by either route.
        4. Return w̄ = Σ_w H̃ Σ_ε⁻¹ r̄ and W = √(|ℰ| − 1) Λ_w^−½.

    Args:
        obs_particles: Forecast ensemble h⁽ᵉ⁾ in observation space.
        y: Observation y.
        obs_noise: Observation-error covariance Σ_ε.
        inflation: Multiplicative covariance inflation ζ ≥ 1, applied to the
            prior term (|ℰ| − 1) / ζ.
        solver: Solver strategy for the Σ_ε⁻¹ solves; ``None`` uses structural
            dispatch.
        low_rank: Use the rank-D_y route. ``None`` chooses by shape, low rank
            when D_y < |ℰ|; ``True`` / ``False`` force a route.

    Returns:
        Tuple ``(w_mean, transform)`` holding w̄ and W. With state anomalies
        ``Xp`` stacked as rows, apply them as ``x_mean + w_mean @ Xp`` and
        ``transform @ Xp``.

    Raises:
        ValueError: If |ℰ| < 2 (the prior term (|ℰ| − 1) / ζ is then zero and
            Λ_w singular), if ``y`` or ``obs_noise`` do not match D_y, or if a
            concrete ``inflation`` is not positive.

    Examples:
        The transform has the all-ones eigenvector, so it preserves the mean:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> obs_particles = jr.normal(jr.key(0), (5, 2))
        >>> obs_noise = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> w_mean, transform = gaussx.etkf_transform(
        ...     obs_particles, jnp.zeros(2), obs_noise
        ... )
        >>> w_mean.shape, transform.shape
        ((5,), (5, 5))
        >>> bool(jnp.allclose(transform @ jnp.ones(5), jnp.ones(5), atol=1e-4))
        True

        Inflation scales that eigenvalue to √ζ:

        >>> _, inflated = gaussx.etkf_transform(
        ...     obs_particles, jnp.zeros(2), obs_noise, inflation=4.0
        ... )
        >>> bool(jnp.allclose(inflated @ jnp.ones(5), 2.0 * jnp.ones(5), atol=1e-3))
        True

    References:
        Bishop et al. (2001); Hunt et al. (2007).
    """
    _check_etkf_inputs(obs_particles, y, obs_noise, inflation)
    ensemble_size, obs_size = obs_particles.shape
    obs_anomalies, mean_residual, whitened_anomalies, whitened_residual = _whiten(
        obs_particles, y, obs_noise, solver
    )
    # √(|ℰ| − 1)
    root_size = jnp.sqrt(jnp.asarray(ensemble_size - 1, dtype=obs_anomalies.dtype))
    identity = jnp.eye(ensemble_size, dtype=obs_anomalies.dtype)
    # (|ℰ| − 1) / ζ
    prior_precision = jnp.asarray(
        (ensemble_size - 1) / inflation, dtype=obs_anomalies.dtype
    )

    if _use_low_rank(low_rank, ensemble_size, obs_size):
        # w̄ = Σ_w H̃ Σ_ε⁻¹ r̄ and Λ_w^−½ I
        w_mean, inverse_sqrt = _low_rank_route(
            obs_noise,
            solver,
            obs_anomalies,
            mean_residual,
            whitened_anomalies,
            whitened_residual,
            prior_precision,
            identity,
        )
        # W = √(|ℰ| − 1) Λ_w^−½
        return w_mean, root_size * inverse_sqrt

    # --- ensemble space ---
    # H̃ Σ_ε⁻¹ H̃ᵀ
    obs_precision = einsum(
        obs_anomalies, whitened_anomalies, "member obs, other obs -> member other"
    )
    # Λ_w = ((|ℰ| − 1) / ζ) I + H̃ Σ_ε⁻¹ H̃ᵀ
    weight_precision = prior_precision * identity + obs_precision
    weight_precision = symmetrize(weight_precision)
    # Σ_w = Λ_w⁻¹ and Λ_w^−½
    weight_covariance, inverse_sqrt = _inverse_and_inverse_sqrt(weight_precision)

    # H̃ Σ_ε⁻¹ r̄
    projected_residual = einsum(
        obs_anomalies, whitened_residual, "member obs, obs -> member"
    )
    # w̄ = Σ_w H̃ Σ_ε⁻¹ r̄
    w_mean = einsum(
        weight_covariance, projected_residual, "member other, other -> member"
    )
    # W = √(|ℰ| − 1) Λ_w^−½
    transform = root_size * inverse_sqrt
    return w_mean, transform


def etkf_analysis(
    particles: Float[Array, "J N"],
    obs_particles: Float[Array, "J M"],
    y: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    *,
    inflation: float | Float[Array, ""] = 1.0,
    solver: AbstractSolverStrategy | None = None,
    low_rank: bool | None = None,
) -> Float[Array, "J N"]:
    """ETKF analysis ensemble, without forming the (|ℰ|, |ℰ|) transform.

        x̄_t|t = x̄_t|t−1 + X̃ᵀ w̄,   X̃_t|t = W X̃
        x⁽ᵉ⁾_t|t = x̄_t|t + x̃⁽ᵉ⁾_t|t

    The deterministic counterpart of `enkf_analysis`, and the same update as
    `etkf_transform` followed by those two products, with w̄ and W as defined
    there. On the rank-D_y route W is applied by matrix-vector products, so with
    D_y < |ℰ| one analysis costs O(|ℰ| D_y² + |ℰ| D_y D_x) and no (|ℰ|, |ℰ|) array
    exists, which is what makes a large ensemble with few observations cheap.

    Algorithm:
        1. Centre both ensembles: X̃, H̃ and r̄ = y − h̄.
        2. Solve Σ_ε⁻¹ against [H̃, r̄] in one call.
        3. Apply Σ_w to H̃ Σ_ε⁻¹ r̄ and Λ_w^−½ to X̃, by either route.
        4. Assemble the mean and the anomalies.

    Args:
        particles: Forecast ensemble x⁽ᵉ⁾ in state space.
        obs_particles: Forecast ensemble h⁽ᵉ⁾ in observation space.
        y: Observation y.
        obs_noise: Observation-error covariance Σ_ε.
        inflation: Multiplicative covariance inflation ζ ≥ 1, applied to the
            prior term (|ℰ| − 1) / ζ.
        solver: Solver strategy for the Σ_ε⁻¹ solves; ``None`` uses structural
            dispatch.
        low_rank: Use the rank-D_y route. ``None`` chooses by shape, low rank
            when D_y < |ℰ|; ``True`` / ``False`` force a route.

    Returns:
        The analysis ensemble.

    Raises:
        ValueError: If |ℰ| < 2, if ``particles`` and ``obs_particles`` disagree
            on |ℰ|, if ``y`` or ``obs_noise`` do not match D_y, or if a concrete
            ``inflation`` is not positive.

    Examples:
        A linear observation of the first two of three state components:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> particles = jr.normal(jr.key(0), (20, 3))
        >>> obs_particles = particles[:, :2]
        >>> obs_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> analysis = gaussx.etkf_analysis(
        ...     particles, obs_particles, jnp.zeros(2), obs_noise
        ... )
        >>> analysis.shape
        (20, 3)

        The observed components move towards the observation:

        >>> before = jnp.abs(particles[:, :2].mean(0))
        >>> bool(jnp.all(jnp.abs(analysis[:, :2].mean(0)) <= before))
        True

        Both routes give the same ensemble:

        >>> full = gaussx.etkf_analysis(
        ...     particles, obs_particles, jnp.zeros(2), obs_noise, low_rank=False
        ... )
        >>> bool(jnp.allclose(analysis, full, atol=1e-4))
        True

    References:
        Bishop et al. (2001); Hunt et al. (2007).
    """
    mean_increment, analysis_anomalies = _etkf_increments(
        particles, obs_particles, y, obs_noise, inflation, solver, low_rank
    )
    # x̄ = (1 / |ℰ|) ∑ₑ x⁽ᵉ⁾
    particle_mean = reduce(particles, "member state -> state", "mean")
    # x⁽ᵉ⁾_t|t = x̄ + X̃ᵀ w̄ + x̃⁽ᵉ⁾_t|t
    return einx.add(
        "state, member state -> member state",
        particle_mean + mean_increment,
        analysis_anomalies,
    )


def _etkf_increments(
    particles: Float[Array, "J N"],
    obs_particles: Float[Array, "J M"],
    y: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    inflation: float | Float[Array, ""],
    solver: AbstractSolverStrategy | None,
    low_rank: bool | None,
) -> tuple[Float[Array, " N"], Float[Array, "J N"]]:
    """Mean increment X̃ᵀ w̄ and analysis anomalies W X̃, by either route.

    Shared by `etkf_analysis` and `gaussx.eki_step`'s deterministic variant.

    Args:
        particles: Forecast ensemble x⁽ᵉ⁾ in state space.
        obs_particles: Forecast ensemble h⁽ᵉ⁾ in observation space.
        y: Observation y.
        obs_noise: Observation-error covariance Σ_ε.
        inflation: Multiplicative covariance inflation ζ.
        solver: Solver strategy for the Σ_ε⁻¹ solves.
        low_rank: Route; ``None`` chooses by shape.

    Returns:
        Tuple ``(mean_increment, analysis_anomalies)``.
    """
    _check_etkf_inputs(obs_particles, y, obs_noise, inflation)
    ensemble_size, obs_size = obs_particles.shape
    if particles.shape[0] != ensemble_size:
        raise ValueError(
            "particles and obs_particles must share the same ensemble size, "
            f"got J={particles.shape[0]} and J={ensemble_size}."
        )
    # x̄ = (1 / |ℰ|) ∑ₑ x⁽ᵉ⁾
    particle_mean = reduce(particles, "member state -> state", "mean")
    # x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
    anomalies = einx.subtract(
        "member state, state -> member state", particles, particle_mean
    )

    if _use_low_rank(low_rank, ensemble_size, obs_size):
        obs_anomalies, mean_residual, whitened_anomalies, whitened_residual = _whiten(
            obs_particles, y, obs_noise, solver
        )
        # (|ℰ| − 1) / ζ
        prior_precision = jnp.asarray(
            (ensemble_size - 1) / inflation, dtype=obs_anomalies.dtype
        )
        # w̄ = Σ_w H̃ Σ_ε⁻¹ r̄ and Λ_w^−½ X̃
        w_mean, inverse_sqrt_anomalies = _low_rank_route(
            obs_noise,
            solver,
            obs_anomalies,
            mean_residual,
            whitened_anomalies,
            whitened_residual,
            prior_precision,
            anomalies,
        )
        # W X̃ = √(|ℰ| − 1) Λ_w^−½ X̃
        root_size = jnp.sqrt(jnp.asarray(ensemble_size - 1, dtype=anomalies.dtype))
        analysis_anomalies = root_size * inverse_sqrt_anomalies
    else:
        w_mean, transform = etkf_transform(
            obs_particles,
            y,
            obs_noise,
            inflation=inflation,
            solver=solver,
            low_rank=False,
        )
        # W X̃
        analysis_anomalies = einsum(
            transform, anomalies, "member other, other state -> member state"
        )
    # X̃ᵀ w̄
    mean_increment = einsum(w_mean, anomalies, "member, member state -> state")
    return mean_increment, analysis_anomalies


def _check_etkf_inputs(
    obs_particles: Float[Array, "J M"],
    y: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    inflation: float | Float[Array, ""],
) -> None:
    """Static checks shared by `etkf_transform` and `etkf_analysis`.

    Args:
        obs_particles: Forecast ensemble h⁽ᵉ⁾ in observation space.
        y: Observation y.
        obs_noise: Observation-error covariance Σ_ε.
        inflation: Multiplicative covariance inflation ζ.

    Raises:
        ValueError: If |ℰ| < 2, if ``y`` or ``obs_noise`` do not match D_y, or
            if a concrete ``inflation`` is not positive.
    """
    _check_observation_shapes(obs_particles, y, obs_noise, observation_name="y")
    ensemble_size = obs_particles.shape[0]
    if ensemble_size < 2:
        raise ValueError(
            "The ETKF requires J >= 2 ensemble members (the prior term "
            f"is (J - 1) / inflation); got J={ensemble_size}."
        )
    if isinstance(inflation, (int, float)) and inflation <= 0:
        raise ValueError(f"inflation must be positive, got {inflation}.")


def _use_low_rank(low_rank: bool | None, ensemble_size: int, obs_size: int) -> bool:
    """Whether to take the rank-D_y route; by default when D_y < |ℰ|.

    Args:
        low_rank: The caller's choice, or ``None`` to choose by shape.
        ensemble_size: Ensemble size |ℰ|.
        obs_size: Observation size D_y.

    Returns:
        ``True`` for the rank-D_y route.
    """
    return obs_size < ensemble_size if low_rank is None else low_rank


def _whiten(
    obs_particles: Float[Array, "J M"],
    y: Float[Array, " M"],
    obs_noise: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy | None,
) -> tuple[
    Float[Array, "J M"], Float[Array, " M"], Float[Array, "J M"], Float[Array, " M"]
]:
    """H̃, r̄, Σ_ε⁻¹ H̃ᵀ as rows and Σ_ε⁻¹ r̄, from one structured solve.

    Args:
        obs_particles: Forecast ensemble h⁽ᵉ⁾ in observation space.
        y: Observation y.
        obs_noise: Observation-error covariance Σ_ε.
        solver: Solver strategy for the Σ_ε⁻¹ solves.

    Returns:
        Tuple ``(obs_anomalies, mean_residual, whitened_anomalies,
        whitened_residual)``.
    """
    # h̄ = (1 / |ℰ|) ∑ₑ h⁽ᵉ⁾
    obs_mean = reduce(obs_particles, "member obs -> obs", "mean")
    # h̃⁽ᵉ⁾ = h⁽ᵉ⁾ − h̄
    obs_anomalies = einx.subtract(
        "member obs, obs -> member obs", obs_particles, obs_mean
    )
    # r̄ = y − h̄
    mean_residual = y - obs_mean
    # [H̃, r̄]: all |ℰ| + 1 right-hand sides at once, so a dense Σ_ε is factored
    # once and a structured Σ_ε is never densified.
    right_hand_sides = jnp.vstack([obs_anomalies, mean_residual])
    # Σ_ε⁻¹ [H̃, r̄]
    whitened = solve_rows(obs_noise, right_hand_sides, solver=solver)
    return obs_anomalies, mean_residual, whitened[:-1], whitened[-1]


@jax.custom_jvp
def _inverse_and_inverse_sqrt(
    matrix: Float[Array, "J J"],
) -> tuple[Float[Array, "J J"], Float[Array, "J J"]]:
    """A⁻¹ and A^−½ of a symmetric positive-definite matrix from a single `eigh`.

    The custom JVP exists for the same reason as `dense_symmetric_sqrt`'s:
    differentiating through `jax.numpy.linalg.eigh` divides by eigenvalue gaps
    and is non-finite at a repeated eigenvalue, which `etkf_transform`'s
    ensemble-space precision always has when D_y < |ℰ| − 1. See
    `_inverse_and_inverse_sqrt_jvp` for the gap-free derivative.

    Args:
        matrix: Symmetric positive-definite matrix A.

    Returns:
        Tuple ``(inverse, inverse_sqrt)`` holding A⁻¹ and A^−½.
    """
    # A = Φ diag(γ) Φᵀ
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return _spectral_inverse_and_inverse_sqrt(eigenvalues, eigenvectors)


def _spectral_inverse_and_inverse_sqrt(
    eigenvalues: Float[Array, " J"],
    eigenvectors: Float[Array, "J J"],
) -> tuple[Float[Array, "J J"], Float[Array, "J J"]]:
    """A⁻¹ and A^−½ from the eigendecomposition A = Φ diag(γ) Φᵀ.

    Args:
        eigenvalues: Eigenvalues γ of A.
        eigenvectors: Eigenvectors Φ of A, one column per eigenvalue.

    Returns:
        Tuple ``(inverse, inverse_sqrt)`` holding A⁻¹ and A^−½.
    """

    def _apply(values: Float[Array, " J"]) -> Float[Array, "J J"]:
        # Φ diag(f(γ))
        scaled = einx.multiply("member mode, mode -> member mode", eigenvectors, values)
        # Φ diag(f(γ)) Φᵀ
        return einsum(scaled, eigenvectors, "member mode, other mode -> member other")

    # A⁻¹ = Φ diag(1 / γ) Φᵀ
    inverse = _apply(1.0 / eigenvalues)
    # A^−½ = Φ diag(1 / √γ) Φᵀ
    inverse_sqrt = _apply(1.0 / jnp.sqrt(eigenvalues))
    return inverse, inverse_sqrt


@_inverse_and_inverse_sqrt.defjvp
def _inverse_and_inverse_sqrt_jvp(primals, tangents):
    """Daleckii-Krein derivative of A⁻¹ and A^−½.

    For a spectral function f(A) = Φ f(diag(γ)) Φᵀ the derivative is

        df(A) = Φ (D ⊙ Φᵀ dA Φ) Φᵀ

    with the divided differences Dᵢⱼ = (f(γᵢ) − f(γⱼ)) / (γᵢ − γⱼ), or f′(γᵢ) on
    ties. For these two functions the divided differences have gap-free closed
    forms,

        (γᵢ⁻¹ − γⱼ⁻¹) / (γᵢ − γⱼ) = −1 / (γᵢ γⱼ)
        (γᵢ^−½ − γⱼ^−½) / (γᵢ − γⱼ) = −1 / (√(γᵢ γⱼ) (√γᵢ + √γⱼ))

    which reduce to f′(γᵢ) on the diagonal, so the derivative stays finite
    however many eigenvalues coincide.

    Args:
        primals: One-element tuple holding the matrix A.
        tangents: One-element tuple holding the tangent dA.

    Returns:
        Tuple ``(primal_out, tangent_out)`` with the values and the tangents of
        ``(A⁻¹, A^−½)``.
    """
    (matrix,) = primals
    (tangent,) = tangents
    # A = Φ diag(γ) Φᵀ
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    primal_out = _spectral_inverse_and_inverse_sqrt(eigenvalues, eigenvectors)

    # eigh reads one triangle, so project the tangent onto the symmetric part.
    # dA ← ½ (dA + dAᵀ)
    tangent = 0.5 * (tangent + rearrange(tangent, "row col -> col row"))
    # Φᵀ dA Φ
    rotated = einsum(
        eigenvectors,
        tangent,
        eigenvectors,
        "row mode, row col, col other -> mode other",
    )
    # √γ
    root = jnp.sqrt(eigenvalues)
    # −1 / (γᵢ γⱼ)
    inverse_difference = -1.0 / einx.multiply(
        "mode, other -> mode other", eigenvalues, eigenvalues
    )
    # −1 / (√(γᵢ γⱼ) (√γᵢ + √γⱼ))
    inverse_sqrt_difference = -1.0 / einx.multiply(
        "mode, other, mode other -> mode other",
        root,
        root,
        einx.add("mode, other -> mode other", root, root),
    )

    def _back(coefficients: Float[Array, "J J"]) -> Float[Array, "J J"]:
        # D ⊙ Φᵀ dA Φ
        weighted = coefficients * rotated
        # Φ (D ⊙ Φᵀ dA Φ) Φᵀ
        return einsum(
            eigenvectors,
            weighted,
            eigenvectors,
            "row mode, mode other, col other -> row col",
        )

    return primal_out, (_back(inverse_difference), _back(inverse_sqrt_difference))


def _low_rank_spectrum(
    obs_anomalies: Float[Array, "J M"],
    whitened_anomalies: Float[Array, "J M"],
) -> tuple[Float[Array, "J R"], Float[Array, " R"]]:
    """U and λ with H̃ Σ_ε⁻¹ H̃ᵀ = U diag(λ) Uᵀ, from a thin QR and a small eigh.

    Args:
        obs_anomalies: Observation anomalies H̃, stacked as rows.
        whitened_anomalies: Σ_ε⁻¹ H̃ᵀ, stacked as rows.

    Returns:
        Tuple ``(basis, excess)`` holding orthonormal U and λ ≥ 0.
    """
    # H̃ = Q T
    orthogonal, _ = jnp.linalg.qr(obs_anomalies)
    # Qᵀ H̃
    projected = einsum(orthogonal, obs_anomalies, "member mode, member obs -> mode obs")
    # Qᵀ Σ_ε⁻¹ H̃ᵀ, as rows
    projected_whitened = einsum(
        orthogonal, whitened_anomalies, "member mode, member obs -> mode obs"
    )
    # Qᵀ H̃ Σ_ε⁻¹ H̃ᵀ Q
    small = einsum(projected, projected_whitened, "mode obs, other obs -> mode other")
    # Qᵀ H̃ Σ_ε⁻¹ H̃ᵀ Q = V diag(λ) Vᵀ
    excess, rotation = jnp.linalg.eigh(symmetrize(small))
    # U = Q V
    basis = einsum(orthogonal, rotation, "member mode, mode other -> member other")
    # H̃ Σ_ε⁻¹ H̃ᵀ is PSD, so a negative λ is round-off
    return basis, jnp.maximum(excess, 0.0)


def _span_apply(
    basis: Float[Array, "J R"],
    mode_values: Float[Array, " R"],
    rhs: Float[Array, " J"],
) -> Float[Array, " J"]:
    """f(Λ_w) b = U diag(f(c + λ)) Uᵀ b, for b in the span of U.

    Without the complement of `_low_rank_apply`, nothing of the prior term can
    leak into a mean update, which matters most for a strongly informative
    observation in float32.

    Args:
        basis: Orthonormal U.
        mode_values: f(c + λ) on each mode.
        rhs: Right-hand side b, in the span of U.

    Returns:
        f(Λ_w) b.
    """
    # Uᵀ b
    projected = einsum(basis, rhs, "member mode, member -> mode")
    # U diag(f(c + λ)) Uᵀ b
    return einsum(basis, mode_values * projected, "member mode, mode -> member")


def _complement(
    basis: Float[Array, "J R"],
    rhs: Float[Array, "J C"],
) -> Float[Array, "J C"]:
    """P B = (I − U Uᵀ)² B, projected twice.

    One pass leaves a component of about ε ‖B‖ in the span of U when B lies
    mostly in that span. f(c) then amplifies it past the true f(c + λ) on that
    span, so a second pass removes it.

    Args:
        basis: Orthonormal U.
        rhs: Right-hand sides B, one per column.

    Returns:
        P B.
    """

    def _project_out(values: Float[Array, "J C"]) -> Float[Array, "J C"]:
        # Uᵀ B
        projected = einsum(basis, values, "member mode, member col -> mode col")
        # B − U Uᵀ B
        return values - einsum(basis, projected, "member mode, mode col -> member col")

    return _project_out(_project_out(rhs))


def _low_rank_apply(
    basis: Float[Array, "J R"],
    prior_value: Float[Array, ""],
    mode_values: Float[Array, " R"],
    rhs: Float[Array, "J C"],
) -> Float[Array, "J C"]:
    """f(Λ_w) B = U diag(f(c + λ)) Uᵀ B + f(c) P B, no (|ℰ|, |ℰ|) array.

    The two components are applied apart: f(c) B + U diag(f(c + λ) − f(c)) Uᵀ B
    is the same in exact arithmetic, but it cancels once λ / c reaches about
    1 / ε.

    Args:
        basis: Orthonormal U.
        prior_value: f(c), for the prior term c = (|ℰ| − 1) / ζ.
        mode_values: f(c + λ) on each mode.
        rhs: Right-hand sides B, one per column.

    Returns:
        f(Λ_w) B.
    """
    # Uᵀ B
    projected = einsum(basis, rhs, "member mode, member col -> mode col")
    # diag(f(c + λ)) Uᵀ B
    scaled = einx.multiply("mode, mode col -> mode col", mode_values, projected)
    # U diag(f(c + λ)) Uᵀ B + f(c) P B
    return einsum(
        basis, scaled, "member mode, mode col -> member col"
    ) + prior_value * _complement(basis, rhs)


def _low_rank_route(
    obs_noise: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy | None,
    obs_anomalies: Float[Array, "J M"],
    mean_residual: Float[Array, " M"],
    whitened_anomalies: Float[Array, "J M"],
    whitened_residual: Float[Array, " M"],
    prior_precision: Float[Array, ""],
    anomaly_rhs: Float[Array, "J C"],
) -> tuple[Float[Array, " J"], Float[Array, "J C"]]:
    """`_low_rank_matmul`, with Σ_ε split into its arrays and the rest.

    The custom JVP takes dΣ_ε itself, and reads only the primal values of the
    whitened arrays, because their tangents have already lost it to rounding.

    Args:
        obs_noise: Observation-error covariance Σ_ε.
        solver: Solver strategy for the Σ_ε⁻¹ solves.
        obs_anomalies: Observation anomalies H̃, stacked as rows.
        mean_residual: r̄.
        whitened_anomalies: Σ_ε⁻¹ H̃ᵀ, stacked as rows.
        whitened_residual: Σ_ε⁻¹ r̄.
        prior_precision: The prior term c = (|ℰ| − 1) / ζ.
        anomaly_rhs: Right-hand sides B of Λ_w^−½, one per column.

    Returns:
        Tuple ``(weights, inverse_sqrt_rhs)`` holding w̄ and Λ_w^−½ B.
    """
    noise_arrays, noise_static = eqx.partition(obs_noise, eqx.is_inexact_array)
    return _low_rank_matmul(
        noise_static,
        solver,
        noise_arrays,
        obs_anomalies,
        mean_residual,
        whitened_anomalies,
        whitened_residual,
        prior_precision,
        anomaly_rhs,
    )


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _low_rank_matmul(
    noise_static: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy | None,
    noise_arrays: lx.AbstractLinearOperator,
    obs_anomalies: Float[Array, "J M"],
    mean_residual: Float[Array, " M"],
    whitened_anomalies: Float[Array, "J M"],
    whitened_residual: Float[Array, " M"],
    prior_precision: Float[Array, ""],
    anomaly_rhs: Float[Array, "J C"],
) -> tuple[Float[Array, " J"], Float[Array, "J C"]]:
    """w̄ = Σ_w H̃ Σ_ε⁻¹ r̄ and Λ_w^−½ B for Λ_w = c I + H̃ Σ_ε⁻¹ H̃ᵀ, from one spectrum.

    The custom JVP exists because JAX's QR rule needs H̃ to have full column
    rank, which a constant or duplicated observation channel breaks. See
    `_low_rank_matmul_jvp` for the derivative, which uses neither QR's nor eigh's.

    Args:
        noise_static: The non-array part of Σ_ε.
        solver: Solver strategy for the Σ_ε⁻¹ solves in the derivative.
        noise_arrays: The array part of Σ_ε.
        obs_anomalies: Observation anomalies H̃, stacked as rows.
        mean_residual: r̄, read only by the derivative.
        whitened_anomalies: Σ_ε⁻¹ H̃ᵀ, stacked as rows.
        whitened_residual: Σ_ε⁻¹ r̄.
        prior_precision: The prior term c = (|ℰ| − 1) / ζ.
        anomaly_rhs: Right-hand sides B of Λ_w^−½, one per column.

    Returns:
        Tuple ``(weights, inverse_sqrt_rhs)`` holding w̄ and Λ_w^−½ B.
    """
    basis, excess = _low_rank_spectrum(obs_anomalies, whitened_anomalies)
    # γ = c + λ
    precision = prior_precision + excess
    # H̃ Σ_ε⁻¹ r̄, in the span of U by construction
    projected_residual = einsum(
        obs_anomalies, whitened_residual, "member obs, obs -> member"
    )
    weights = _span_apply(basis, 1.0 / precision, projected_residual)
    inverse_sqrt_rhs = _low_rank_apply(
        basis, 1.0 / jnp.sqrt(prior_precision), 1.0 / jnp.sqrt(precision), anomaly_rhs
    )
    return weights, inverse_sqrt_rhs


def _analysis_residual(
    obs_anomalies: Float[Array, "J M"],
    whitened_anomalies: Float[Array, "J M"],
    prior_precision: Float[Array, ""],
    whitened_residual: Float[Array, " M"],
    basis: Float[Array, "J R"],
    excess: Float[Array, " R"],
) -> Float[Array, " M"]:
    """ρₐ = Σ_ε⁻¹ (r̄ − H̃ᵀ w̄), without subtracting nearly equal vectors.

    With ρ = Σ_ε⁻¹ r̄, a = Uᵀ H̃ (rows aᵢ), q = Uᵀ Σ_ε⁻¹ H̃ᵀ (rows qᵢ) and
    α = a ρ, the plain form is ρₐ = ρ − ∑ᵢ qᵢ αᵢ / γᵢ, which cancels on a mode with
    λᵢ ≫ c. Since aᵢᵀ qⱼ = λᵢ δᵢⱼ, the part of ρ along those modes is removed by the
    oblique projection T = ∑ᵢ qᵢ aᵢᵀ / λᵢ, which leaves the small remainder in
    closed form:

        1 / λᵢ − 1 / γᵢ = c / (λᵢ γᵢ)

        ρₐ = (I − T)² ρ + ∑ᵢ qᵢ αᵢ c / (λᵢ γᵢ) − ∑ⱼ qⱼ αⱼ / γⱼ

    with i over λᵢ > c and j over the rest, where there is nothing to cancel. T is
    idempotent, and applying it twice removes the rounding that one pass leaves.

    Args:
        obs_anomalies: Observation anomalies H̃, stacked as rows.
        whitened_anomalies: Σ_ε⁻¹ H̃ᵀ, stacked as rows.
        prior_precision: The prior term c = (|ℰ| − 1) / ζ.
        whitened_residual: ρ = Σ_ε⁻¹ r̄.
        basis: Orthonormal U.
        excess: λ, with H̃ Σ_ε⁻¹ H̃ᵀ = U diag(λ) Uᵀ.

    Returns:
        ρₐ.
    """
    # a = Uᵀ H̃
    mode_obs = einsum(basis, obs_anomalies, "member mode, member obs -> mode obs")
    # q = Uᵀ Σ_ε⁻¹ H̃ᵀ
    mode_whitened = einsum(
        basis, whitened_anomalies, "member mode, member obs -> mode obs"
    )
    # γ = c + λ
    precision = prior_precision + excess
    # a mode that cancels; λ > c > 0 there, so dividing by λ is safe
    strong = excess > prior_precision
    safe_excess = jnp.where(strong, excess, 1.0)

    def _remove_strong_modes(values: Float[Array, " M"]) -> Float[Array, " M"]:
        # a v
        projected = einsum(mode_obs, values, "mode obs, obs -> mode")
        # v − ∑ᵢ qᵢ (a v)ᵢ / λᵢ
        coefficients = jnp.where(strong, projected / safe_excess, 0.0)
        return values - einsum(mode_whitened, coefficients, "mode obs, mode -> obs")

    # (I − T)² ρ
    remainder = _remove_strong_modes(_remove_strong_modes(whitened_residual))
    # α = a ρ
    mode_residual = einsum(mode_obs, whitened_residual, "mode obs, obs -> mode")
    # c / (λᵢ γᵢ) on strong modes, −1 / γⱼ on the rest
    coefficients = jnp.where(
        strong, prior_precision / (safe_excess * precision), -1.0 / precision
    )
    # (I − T)² ρ + ∑ᵢ qᵢ αᵢ c / (λᵢ γᵢ) − ∑ⱼ qⱼ αⱼ / γⱼ
    return remainder + einsum(
        mode_whitened, coefficients * mode_residual, "mode obs, mode -> obs"
    )


def _inverse_difference(
    first: Float[Array, "*shape"], second: Float[Array, "*shape"]
) -> Float[Array, "*shape"]:
    """(a⁻¹ − b⁻¹) / (a − b) = −1 / (a b), finite at a = b.

    Args:
        first: Argument a.
        second: Argument b.

    Returns:
        The divided difference of s ↦ s⁻¹.
    """
    return -1.0 / (first * second)


def _inverse_sqrt_difference(
    first: Float[Array, "*shape"], second: Float[Array, "*shape"]
) -> Float[Array, "*shape"]:
    """(a^−½ − b^−½) / (a − b) = −1 / (√(a b) (√a + √b)), finite at a = b.

    Args:
        first: Argument a.
        second: Argument b.

    Returns:
        The divided difference of s ↦ s^−½.
    """
    root_first, root_second = jnp.sqrt(first), jnp.sqrt(second)
    return -1.0 / (root_first * root_second * (root_first + root_second))


@_low_rank_matmul.defjvp
def _low_rank_matmul_jvp(noise_static, solver, primals, tangents):
    """Daleckii-Krein derivative of w̄ and Λ_w^−½ B in the basis [U, U⊥].

    The same rule as `_inverse_and_inverse_sqrt_jvp`, written in blocks so that
    U⊥ is never formed: I − U Uᵀ applies it, and every eigenvalue on U⊥ equals c.

        df(Λ_w) B = U (D ⊙ Uᵀ dΛ_w U) Uᵀ B
                    + U diag(d) Uᵀ dΛ_w P B
                    + P dΛ_w U diag(d) Uᵀ B
                    + f′(c) P dΛ_w P B

    with P = I − U Uᵀ, Dᵢⱼ the divided difference at (γᵢ, γⱼ) and dᵢ the one at
    (γᵢ, c). With dΛ_w = dc I + dH̃ Wᵀ + H̃ dWᵀ and W = Σ_ε⁻¹ H̃ᵀ, the identities
    P H̃ = 0 and Wᵀ P = 0 reduce those blocks, before any is evaluated, to

        Uᵀ dΛ_w P = Uᵀ H̃ dWᵀ P,   P dΛ_w U = P dH̃ Wᵀ U,   P dΛ_w P = dc P

    so no term of about ε ‖W‖ meets the large f′(c), which grew as ε (λ / c).

    The mean weights take their three terms combined before any is evaluated,
    because evaluated apart they cancel for a strongly informative observation.
    With ρ = Σ_ε⁻¹ r̄, W = Σ_ε⁻¹ H̃ᵀ and dΛ_w = dc I + dH̃ Wᵀ + H̃ dWᵀ,

        dw̄ = dΣ_w H̃ ρ + Σ_w dH̃ ρ + Σ_w H̃ dρ
           = Σ_w dH̃ ρₐ + Σ_w (H̃ (dρ − dWᵀ w̄) − dc w̄)
        dρ − dWᵀ w̄ = Σ_ε⁻¹ (dr̄ − dH̃ᵀ w̄ − dΣ_ε ρₐ)

    where ρₐ = ρ − Wᵀ w̄ comes from `_analysis_residual`, and the second term of
    dw̄ acts on a vector known to lie in the span of U. dW comes from dΣ_ε in the
    same way, because the tangents of Σ_ε⁻¹ r̄ and Σ_ε⁻¹ H̃ᵀ would already have lost
    this to rounding.

    Args:
        primals: The arguments of `_low_rank_matmul`.
        tangents: Their tangents.

    Returns:
        Tuple ``(primal_out, tangent_out)``.
    """
    (
        noise_arrays,
        obs_anomalies,
        _,
        whitened_anomalies,
        whitened_residual,
        prior_precision,
        anomaly_rhs,
    ) = primals
    (
        d_noise_arrays,
        d_obs,
        d_mean_residual,
        _,
        _,
        d_prior,
        d_anomaly_rhs,
    ) = tangents
    primal_out = _low_rank_matmul(noise_static, solver, *primals)
    basis, excess = _low_rank_spectrum(obs_anomalies, whitened_anomalies)
    # γ = c + λ
    precision = prior_precision + excess

    # w̄
    weights = primal_out[0]
    # ρₐ = Σ_ε⁻¹ (r̄ − H̃ᵀ w̄)
    analysis_residual = _analysis_residual(
        obs_anomalies,
        whitened_anomalies,
        prior_precision,
        whitened_residual,
        basis,
        excess,
    )

    # --- what moves through Σ_ε ---
    # A factorisation of a symmetric-tagged matrix sees only the symmetric part of
    # a raw tangent, and the tagged operator's own transpose is a no-op.
    noise = eqx.combine(noise_arrays, noise_static)
    if isinstance(noise, lx.MatrixLinearOperator) and lx.is_symmetric(noise):
        d_noise_arrays = eqx.tree_at(
            lambda operator: operator.matrix,
            d_noise_arrays,
            symmetrize(d_noise_arrays.matrix),
        )

    def d_noise_matvec(vector: Float[Array, " M"]) -> Float[Array, " M"]:
        # dΣ_ε v
        def noise_matvec(arrays):
            return eqx.combine(arrays, noise_static).mv(vector)

        return jax.jvp(noise_matvec, (noise_arrays,), (d_noise_arrays,))[1]

    # dH̃ᵀ w̄
    moved_obs = einsum(d_obs, weights, "member obs, member -> obs")
    # dr̄ − dH̃ᵀ w̄ − dΣ_ε ρₐ
    residual_rhs = d_mean_residual - moved_obs - d_noise_matvec(analysis_residual)
    # dH̃ᵀ − dΣ_ε W, as rows
    anomaly_rhs_rows = d_obs - jax.vmap(d_noise_matvec)(whitened_anomalies)
    # Σ_ε⁻¹ [dH̃ᵀ − dΣ_ε W, dr̄ − dH̃ᵀ w̄ − dΣ_ε ρₐ], in one solve
    solved = solve_rows(
        noise,
        jnp.vstack([anomaly_rhs_rows, residual_rhs]),
        solver=solver,
    )
    # dW = Σ_ε⁻¹ (dH̃ᵀ − dΣ_ε W) and dρ − dWᵀ w̄
    d_whitened, moved_residual = solved[:-1], solved[-1]

    # a = Uᵀ H̃
    mode_obs = einsum(basis, obs_anomalies, "member mode, member obs -> mode obs")
    # Wᵀ U
    whitened_modes = einsum(
        whitened_anomalies, basis, "member obs, member mode -> obs mode"
    )
    # dH̃ Wᵀ U
    d_obs_modes = einsum(d_obs, whitened_modes, "member obs, obs mode -> member mode")
    # dWᵀ U
    d_whitened_modes = einsum(d_whitened, basis, "member obs, member mode -> obs mode")
    # Uᵀ (dH̃ Wᵀ + H̃ dWᵀ) U
    rotated = einsum(
        basis, d_obs_modes, "member mode, member other -> mode other"
    ) + einsum(mode_obs, d_whitened_modes, "mode obs, obs other -> mode other")
    # Uᵀ dΛ_w U = dc I + ½ (Uᵀ (dH̃ Wᵀ + H̃ dWᵀ) U + transpose)
    rotated = symmetrize(rotated) + d_prior * jnp.eye(basis.shape[1], dtype=basis.dtype)

    def tangent(difference, rhs):
        # D, d and f′(c)
        block_modes = difference(
            rearrange(precision, "mode -> mode 1"),
            rearrange(precision, "other -> 1 other"),
        )
        block_mixed = difference(precision, prior_precision)
        block_prior = difference(prior_precision, prior_precision)
        # Uᵀ B and P B
        projected = einsum(basis, rhs, "member mode, member col -> mode col")
        perpendicular = _complement(basis, rhs)
        # U (D ⊙ Uᵀ dΛ_w U) Uᵀ B
        mode_block = einsum(
            basis,
            block_modes * rotated,
            projected,
            "member mode, mode other, other col -> member col",
        )
        # dWᵀ P B
        d_whitened_perp = einsum(
            d_whitened, perpendicular, "member obs, member col -> obs col"
        )
        # Uᵀ dΛ_w P B = Uᵀ H̃ dWᵀ P B
        mode_perp = einsum(mode_obs, d_whitened_perp, "mode obs, obs col -> mode col")
        # U diag(d) Uᵀ H̃ dWᵀ P B
        mixed_left = einsum(
            basis,
            einx.multiply("mode, mode col -> mode col", block_mixed, mode_perp),
            "member mode, mode col -> member col",
        )
        # P dH̃ Wᵀ U diag(d) Uᵀ B
        mixed_right = _complement(
            basis,
            einsum(
                d_obs_modes,
                einx.multiply("mode, mode col -> mode col", block_mixed, projected),
                "member mode, mode col -> member col",
            ),
        )
        # f′(c) dc P B
        prior_block = block_prior * d_prior * perpendicular
        return mode_block + mixed_left + mixed_right + prior_block

    # dH̃ ρₐ
    moved_anomalies = einsum(d_obs, analysis_residual, "member obs, obs -> member")
    # Σ_w dH̃ ρₐ
    moved_term = _low_rank_apply(
        basis,
        1.0 / prior_precision,
        1.0 / precision,
        rearrange(moved_anomalies, "member -> member 1"),
    )
    # H̃ (dρ − dWᵀ w̄) − dc w̄, in the span of U by construction
    in_span = (
        einsum(obs_anomalies, moved_residual, "member obs, obs -> member")
        - d_prior * weights
    )
    # dw̄ = Σ_w dH̃ ρₐ + Σ_w (H̃ (dρ − dWᵀ w̄) − dc w̄)
    d_weights = rearrange(moved_term, "member 1 -> member") + _span_apply(
        basis, 1.0 / precision, in_span
    )
    # dΛ_w^−½ B + Λ_w^−½ dB
    d_inverse_sqrt = tangent(_inverse_sqrt_difference, anomaly_rhs) + _low_rank_apply(
        basis, 1.0 / jnp.sqrt(prior_precision), 1.0 / jnp.sqrt(precision), d_anomaly_rhs
    )
    return primal_out, (d_weights, d_inverse_sqrt)
