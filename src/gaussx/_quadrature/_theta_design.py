"""θ-designs: integration points over INLA hyperparameters.

Rue, Martino & Chopin (2009), *Approximate Bayesian inference for latent
Gaussian models by using integrated nested Laplace approximations*, JRSS-B
71(2), §6.5.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Literal

import einx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange


# Hard cap on the axis-wise grid walk, so a flat ``log_post`` cannot loop
# forever. In z a well-posed posterior drops by ‖z‖²/2, so this is never reached
# for any sensible ``grid_step`` / ``grid_threshold``.
_MAX_GRID_STEPS = 100


def theta_design(
    log_post: Callable[[Array], Array],
    mode: Float[Array, " m"],
    *,
    method: Literal["eb", "grid", "ccd"] | None = None,
    hessian: Float[Array, "m m"] | None = None,
    grid_step: float = 1.0,
    grid_threshold: float = 2.5,
    ccd_f0: float = 1.1,
) -> tuple[Float[Array, "K m"], Float[Array, " K"]]:
    """Integration points and log-weights over hyperparameters θ.

        θ(z) = θ⋆ + S z,    S = Q diag(γ)^(−1/2)
        −∇² log π̃(θ⋆ ∣ y) = Q diag(γ) Qᵀ
        log w_k = log Δ_k + log π̃(θ_k ∣ y) − log ∑ₗ Δₗ π̃(θₗ ∣ y)

    Builds the design R-INLA uses to integrate over θ. Points are placed in the
    standardised coordinates z, in which the posterior is approximately
    𝒩(0, I). Each point gets a design weight Δ_k, which is corrected by the
    evaluated posterior and normalised so that ∑_k w_k = 1. Then

        π̃(xᵢ ∣ y) ≈ ∑_k w_k π̃(xᵢ ∣ θ_k, y)

    ``"eb"`` is empirical Bayes, the mode alone.

    ``"grid"`` walks each z-axis in both directions in steps of ``grid_step``
    while log π̃ stays within ``grid_threshold`` log-units of the mode, takes the
    product grid of those axis ranges, and keeps its points that are within
    ``grid_threshold`` of the mode. Δ_k is constant. The point count depends on
    ``log_post``, so this method cannot be traced under `jax.jit`.

    ``"ccd"`` is a central composite design of O(D_θ²) points: the centre, 2 D_θ
    axial points and a resolution-V fractional factorial 2^(D_θ − p), all
    non-centre points on the sphere of radius f₀ √D_θ (15 points at D_θ = 3, 27
    at D_θ = 5). The centre and shell weights are fixed, as in Rue et al. (2009)
    §6.5, so that the design integrates a Gaussian posterior exactly up to its
    second moments:

        Δ₀ ∝ 1 − f₀⁻²
        Δ_shell ∝ exp(D_θ f₀² / 2) / ((K − 1) f₀²)

    In INLA, run one `argminax.solve.laplace_mode` per design point under `jax.vmap` and
    mix the fits with exp(log w_k).

    Algorithm:
        1. For ``"eb"``, return the mode with log-weight zero.
        2. Take the Hessian at the mode and factor it: S = Q diag(γ)^(−1/2).
        3. For ``"grid"``, walk the z-axes, map the product grid to θ, and drop
           points below the threshold.
        4. For ``"ccd"``, lay out the design in z, map it to θ, and add log Δ_k
           to the evaluated posterior.
        5. Normalise the log-weights.

    Args:
        log_post: Unnormalised log-posterior log π̃(θ ∣ y), mapping θ to a scalar.
        mode: Its mode θ⋆. Finding it is the caller's job.
        method: ``"eb"``, ``"grid"`` or ``"ccd"``. ``None`` (default) picks
            ``"grid"`` for D_θ ≤ 2 and ``"ccd"`` for D_θ > 2.
        hessian: Hessian of ``log_post`` at ``mode`` (negative definite). If
            ``None``, computed reverse-over-reverse
            (``jax.jacrev(jax.jacrev(log_post))``), which also differentiates
            through custom-VJP-only code such as the sparse Cholesky
            log-determinant (`jax.hessian` is forward-over-reverse).
        grid_step: Step size in z for ``"grid"``.
        grid_threshold: Log-density drop from the mode beyond which ``"grid"``
            stops and discards points.
        ccd_f0: Shell radius factor f₀ > 1 for ``"ccd"``.

    Returns:
        Tuple ``(points, log_weights)`` holding the K points θ_k and the
        normalised log-weights log w_k (``logsumexp(log_weights) == 0``). The
        first point is the mode.

    Raises:
        ValueError: If ``mode`` is not 1-D, ``method`` is unknown, ``grid_step``
            or ``grid_threshold`` is not positive, or ``ccd_f0`` is not above 1.

    Examples:
        A central composite design recovers the mean of a Gaussian posterior,
        and composes with `jax.vmap` to mix a function of θ:

        >>> import einx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> def log_post(theta):
        ...     return -0.5 * jnp.sum((theta - 1.0) ** 2 / jnp.array([1.0, 4.0, 9.0]))
        >>> points, log_weights = theta_design(log_post, jnp.ones(3), method="ccd")
        >>> points.shape, log_weights.shape
        ((15, 3), (15,))
        >>> weights = jnp.exp(log_weights)
        >>> posterior_mean = einx.dot("k, k m -> m", weights, points)
        >>> bool(jnp.allclose(posterior_mean, jnp.ones(3), atol=1e-4))
        True
        >>> doubled = jax.vmap(lambda theta: 2.0 * theta)(points)
        >>> mixture = einx.dot("k, k m -> m", weights, doubled)
        >>> bool(jnp.allclose(mixture, 2.0 * jnp.ones(3), atol=1e-4))
        True

        Empirical Bayes keeps the mode alone:

        >>> points, log_weights = theta_design(log_post, jnp.ones(3), method="eb")
        >>> points.shape, bool(log_weights[0] == 0.0)
        ((1, 3), True)

        The mode must be a vector:

        >>> theta_design(log_post, jnp.ones((1, 2)))
        Traceback (most recent call last):
        ...
        ValueError: mode must be 1-D, got shape (1, 2)
    """
    mode = jnp.asarray(mode)
    if mode.ndim != 1:
        raise ValueError(f"mode must be 1-D, got shape {mode.shape}")
    theta_size = mode.shape[0]
    if method is None:
        method = "grid" if theta_size <= 2 else "ccd"
    dtype = mode.dtype

    # The patterns below keep their axis names: einx prints them when an empty
    # mode or a Hessian of the wrong size raises.
    if method == "eb":
        return rearrange(mode, "m -> 1 m"), jnp.zeros((1,), dtype=dtype)

    if hessian is None:
        # ∇ log π̃
        gradient_fn = jax.jacrev(log_post)
        # ∇² log π̃(θ⋆ ∣ y)
        hessian = jax.jacrev(gradient_fn)(mode)
    # −∇² log π̃(θ⋆ ∣ y)
    negative_hessian = -jnp.asarray(hessian, dtype=dtype)
    # Q diag(γ) Qᵀ = −∇² log π̃(θ⋆ ∣ y)
    eigenvalues, eigenvectors = jnp.linalg.eigh(negative_hessian)
    # diag(γ)^(−1/2)
    inverse_root_eigenvalues = jax.lax.rsqrt(eigenvalues)
    # S = Q diag(γ)^(−1/2)
    scale = einx.multiply("i j, j -> i j", eigenvectors, inverse_root_eigenvalues)

    def to_theta(z: Float[Array, "K m"]) -> Float[Array, "K m"]:
        # S z
        shifts = einsum(scale, z, "i j, k j -> k i")
        # θ(z) = θ⋆ + S z
        return einx.add("k i, i -> k i", shifts, mode)

    if method == "grid":
        if grid_step <= 0 or grid_threshold <= 0:
            raise ValueError("grid_step and grid_threshold must be positive")
        z = _grid_z(log_post, scale, mode, grid_step, grid_threshold)
        z_array = jnp.asarray(z, dtype=dtype)
        # θ_k = θ(z_k)
        points = to_theta(z_array)
        # log π̃(θ_k ∣ y)
        log_posteriors = jax.vmap(log_post)(points)
        # The grid's floor: log π̃(θ⋆ ∣ y) − threshold
        floor = log_post(mode) - grid_threshold
        keep = np.asarray(log_posteriors >= floor)
        points, log_posteriors = points[keep], log_posteriors[keep]
        # log ∑ₗ π̃(θₗ ∣ y)
        log_normaliser = jsp.logsumexp(log_posteriors)
        # log w_k = log π̃(θ_k ∣ y) − log ∑ₗ π̃(θₗ ∣ y), with Δ_k constant
        return points, log_posteriors - log_normaliser

    if method == "ccd":
        if ccd_f0 <= 1:
            raise ValueError(f"ccd_f0 must be > 1, got {ccd_f0}")
        # Centre first; shell points on radius √D_θ
        z_unit = _ccd_design(theta_size)
        # K
        n_points = z_unit.shape[0]
        # log Δ_shell = D_θ f₀² / 2 − log((K − 1) f₀²)
        log_delta = np.full(
            n_points,
            theta_size * ccd_f0**2 / 2 - math.log((n_points - 1) * ccd_f0**2),
        )
        # log Δ₀ = log(1 − f₀⁻²)
        log_delta[0] = math.log1p(-(ccd_f0**-2))
        # z_k on the shell of radius f₀ √D_θ
        z_array = jnp.asarray(ccd_f0 * z_unit, dtype=dtype)
        # θ_k = θ(z_k)
        points = to_theta(z_array)
        log_delta_array = jnp.asarray(log_delta, dtype=dtype)
        # log Δ_k + log π̃(θ_k ∣ y)
        log_unnormalised = log_delta_array + jax.vmap(log_post)(points)
        # log ∑ₗ Δₗ π̃(θₗ ∣ y)
        log_normaliser = jsp.logsumexp(log_unnormalised)
        # log w_k = log Δ_k + log π̃(θ_k ∣ y) − log ∑ₗ Δₗ π̃(θₗ ∣ y)
        return points, log_unnormalised - log_normaliser

    raise ValueError(f"method must be 'eb', 'grid' or 'ccd', got {method!r}")


def _grid_z(
    log_post: Callable[[Array], Array],
    scale: Float[Array, "m m"],
    mode: Float[Array, " m"],
    step: float,
    threshold: float,
) -> np.ndarray:
    """Product grid in z of the axis-wise explored ranges, centre first.

        θ = θ⋆ ± k h S_j,    while log π̃(θ ∣ y) ≥ log π̃(θ⋆ ∣ y) − threshold

    Args:
        log_post: Unnormalised log-posterior log π̃(θ ∣ y).
        scale: Scale S, whose column S_j is the j-th z-axis in θ.
        mode: Mode θ⋆.
        step: Step size h in z.
        threshold: Log-density drop from the mode that ends a walk.

    Returns:
        The grid points z, one row per point.
    """
    # One compile for the whole walk
    log_post = jax.jit(log_post)
    # log π̃(θ⋆ ∣ y) − threshold
    floor = float(log_post(mode)) - threshold
    axes = []
    for axis in range(mode.shape[0]):
        step_indices = [0]
        for sign in (1, -1):
            step_index = 1
            while step_index <= _MAX_GRID_STEPS:
                # θ = θ⋆ ± k h S_j
                theta = mode + (sign * step_index * step) * scale[:, axis]
                if float(log_post(theta)) < floor:
                    break
                step_indices.append(sign * step_index)
                step_index += 1
        axes.append(step * np.asarray(step_indices, dtype=float))
    return np.array(list(itertools.product(*axes)))


def _resolution_v_columns(m: int) -> tuple[int, list[int]]:
    """Walsh column indices of a minimal resolution-V 2^(m − p) design.

    Greedy construction of Sanchez & Sanchez (2005): a new column index may not
    equal the XOR of at most three chosen ones, so no defining word has length
    below five (main effects and two-factor interactions unaliased).

    Args:
        m: Number of factors D_θ.

    Returns:
        Tuple ``(n_runs, columns)``.
    """
    cols: list[int] = []
    forbidden = {0}
    c = 0
    while len(cols) < m:
        c += 1
        if c in forbidden:
            continue
        sums = {c}
        sums |= {c ^ a for a in cols}
        sums |= {c ^ a ^ b for i, a in enumerate(cols) for b in cols[i + 1 :]}
        forbidden |= sums
        cols.append(c)
    return 1 << max(cols).bit_length(), cols


def _ccd_design(m: int) -> np.ndarray:
    """CCD in z with shell radius √D_θ: centre, axial, factorial.

    Args:
        m: Number of factors D_θ.

    Returns:
        The design points z, one row per point, centre first.
    """
    # ±√D_θ eᵢ
    axial = math.sqrt(m) * np.concatenate([np.eye(m), -np.eye(m)])
    # At D_θ = 1 the factorial points would duplicate the axial ones
    if m == 1:
        return np.concatenate([np.zeros((1, 1)), axial])
    n_runs, cols = _resolution_v_columns(m)
    runs = np.arange(n_runs)
    parity = np.array([[bin(r & c).count("1") % 2 for c in cols] for r in runs])
    # Factorial points ±1, from the parity of each run against each column
    factorial = 1.0 - 2.0 * parity
    return np.concatenate([np.zeros((1, m)), axial, factorial])
