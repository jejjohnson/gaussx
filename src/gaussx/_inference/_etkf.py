"""Ensemble transform Kalman filter (ETKF): deterministic square-root analysis."""

from __future__ import annotations

import einx
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

    Algorithm:
        1. Centre the observation ensemble: H̃ and r̄ = y − h̄.
        2. Solve Σ_ε⁻¹ against [H̃, r̄] in one call.
        3. Assemble Λ_w and symmetrise it.
        4. Eigendecompose Λ_w to read off Σ_w and Λ_w^−½.
        5. Return w̄ = Σ_w H̃ Σ_ε⁻¹ r̄ and W = √(|ℰ| − 1) Λ_w^−½.

    Args:
        obs_particles: Forecast ensemble h⁽ᵉ⁾ in observation space.
        y: Observation y.
        obs_noise: Observation-error covariance Σ_ε.
        inflation: Multiplicative covariance inflation ζ ≥ 1, applied to the
            prior term (|ℰ| − 1) / ζ.
        solver: Solver strategy for the Σ_ε⁻¹ solves; ``None`` uses structural
            dispatch.

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
    _check_observation_shapes(obs_particles, y, obs_noise, observation_name="y")
    ensemble_size = obs_particles.shape[0]
    if ensemble_size < 2:
        raise ValueError(
            "etkf_transform requires J >= 2 ensemble members (the prior term "
            f"is (J - 1) / inflation); got J={ensemble_size}."
        )
    if isinstance(inflation, (int, float)) and inflation <= 0:
        raise ValueError(f"inflation must be positive, got {inflation}.")
    # h̄ = (1 / |ℰ|) ∑ₑ h⁽ᵉ⁾
    obs_mean = reduce(obs_particles, "member obs -> obs", "mean")
    # h̃⁽ᵉ⁾ = h⁽ᵉ⁾ − h̄
    obs_anomalies = einx.subtract(
        "member obs, obs -> member obs", obs_particles, obs_mean
    )

    # --- whiten ---
    # r̄ = y − h̄
    mean_residual = y - obs_mean
    # [H̃, r̄]: all |ℰ| + 1 right-hand sides at once, so a dense Σ_ε is factored
    # once and a structured Σ_ε is never densified.
    right_hand_sides = jnp.vstack([obs_anomalies, mean_residual])
    # Σ_ε⁻¹ [H̃, r̄]
    whitened = solve_rows(obs_noise, right_hand_sides, solver=solver)
    whitened_anomalies, whitened_residual = whitened[:-1], whitened[-1]

    # --- ensemble space ---
    identity = jnp.eye(ensemble_size, dtype=whitened_anomalies.dtype)
    # ((|ℰ| − 1) / ζ) I
    prior_precision = (ensemble_size - 1) / inflation * identity
    # H̃ Σ_ε⁻¹ H̃ᵀ
    obs_precision = einsum(
        obs_anomalies, whitened_anomalies, "member obs, other obs -> member other"
    )
    # Λ_w = ((|ℰ| − 1) / ζ) I + H̃ Σ_ε⁻¹ H̃ᵀ
    weight_precision = prior_precision + obs_precision
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
    transform = (
        jnp.sqrt(jnp.asarray(ensemble_size - 1, dtype=inverse_sqrt.dtype))
        * inverse_sqrt
    )
    return w_mean, transform


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
