"""Kronecker GP recipes: exact inference on grid-structured data.

The N training inputs form a grid with D axes of sizes N_1, …, N_D, so
N = ∏_d N_d and the kernel matrix is K_XX = K_1 ⊗ … ⊗ K_D, flattened row
by row as `gaussx.Kronecker` does. Each factor has the eigendecomposition
K_d = Q_d diag(γ_d) Q_dᵀ, so K_XX = Q diag(γ) Qᵀ with Q = Q_1 ⊗ … ⊗ Q_D and
γ = γ_1 ⊗ … ⊗ γ_D. The training covariance is Σ_y = K_XX + σ_ε² I, and the
M test inputs form a grid of sizes M_1, …, M_D with per-axis
cross-covariances K_*X,d, M_d × N_d.
"""

from __future__ import annotations

import functools as ft
import math

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._distributions._utils import _axis_names
from gaussx._einx import rearrange
from gaussx._primitives._eig import eig


def _normalize_axis(axis: int, ndim: int) -> int:
    """Map a possibly negative axis index into ``range(ndim)``.

    Args:
        axis: Axis index, negative counting from the end.
        ndim: Number of axes.

    Returns:
        The non-negative axis index.

    Raises:
        ValueError: If ``axis`` is outside ``[-ndim, ndim)``.
    """
    # Check before the modulo, which would wrap any out-of-range axis.
    if not -ndim <= axis < ndim:
        msg = f"axis {axis} is out of bounds for array with ndim {ndim}"
        raise ValueError(msg)
    return axis % ndim


def _validate_grid(
    K_factors: list[lx.AbstractLinearOperator],
    grid_shape: tuple[int, ...],
    n_obs: int,
) -> None:
    """Check ``grid_shape`` against the factors and the number of observations.

    Args:
        K_factors: Kernel factors K_1, …, K_D.
        grid_shape: Grid sizes (N_1, …, N_D).
        n_obs: Number of observations N in y.

    Raises:
        ValueError: If ``grid_shape`` does not have one entry per factor,
            an entry does not match its factor's size, or ∏_d N_d ≠ N.
    """
    if len(grid_shape) != len(K_factors):
        msg = (
            f"grid_shape must have one entry per Kronecker factor: got "
            f"{len(grid_shape)} entries for {len(K_factors)} factors."
        )
        raise ValueError(msg)
    for i, (n_i, K_i) in enumerate(zip(grid_shape, K_factors, strict=True)):
        if n_i != K_i.in_size():
            msg = (
                f"grid_shape[{i}] = {n_i} does not match Kronecker factor {i} "
                f"of size {K_i.in_size()}."
            )
            raise ValueError(msg)
    if math.prod(grid_shape) != n_obs:
        msg = (
            f"prod(grid_shape) = {math.prod(grid_shape)} does not match the "
            f"{n_obs} observations in y."
        )
        raise ValueError(msg)


def _reshape_flat_to_grid(
    values: Float[Array, " N"],
    grid_shape: tuple[int, ...],
) -> Array:
    """Unflatten a row-major vector onto the grid, the inverse of vec.

    Args:
        values: Flattened grid values, vec(X).
        grid_shape: Grid sizes (N_1, …, N_D).

    Returns:
        The grid values X, one axis per grid dimension.
    """
    axes = _axis_names(len(grid_shape))
    axis_lengths = dict(zip(axes, grid_shape, strict=True))
    grid_pattern = " ".join(axes)
    return rearrange(values, f"({grid_pattern}) -> {grid_pattern}", **axis_lengths)


def _move_axis_to_last(values: Array, axis: int) -> Array:
    """Move one axis of a grid to the end, keeping the others in order.

    Args:
        values: Grid values.
        axis: Axis to move.

    Returns:
        The grid values with ``axis`` last.
    """
    axis = _normalize_axis(axis, values.ndim)
    axes = list(_axis_names(values.ndim))
    source_pattern = " ".join(axes)
    target_axes = [name for i, name in enumerate(axes) if i != axis] + [axes[axis]]
    target_pattern = " ".join(target_axes)
    return rearrange(values, f"{source_pattern} -> {target_pattern}")


def _move_last_axis_to(values: Array, axis: int) -> Array:
    """Move the last axis of a grid back to position ``axis``.

    The inverse of `_move_axis_to_last`.

    Args:
        values: Grid values, with the moved axis last.
        axis: Position to move the last axis to.

    Returns:
        The grid values with the last axis at ``axis``.
    """
    axis = _normalize_axis(axis, values.ndim)
    axes = list(_axis_names(values.ndim))
    source_axes = [name for i, name in enumerate(axes) if i != axis] + [axes[axis]]
    source_pattern = " ".join(source_axes)
    target_pattern = " ".join(axes)
    return rearrange(values, f"{source_pattern} -> {target_pattern}")


def kronecker_mll(
    K_factors: list[lx.AbstractLinearOperator],
    y: Float[Array, " N"],
    noise_var: float,
    grid_shape: tuple[int, ...],
) -> Float[Array, ""]:
    """Exact marginal log-likelihood log p(y) of a Kronecker-structured GP.

        log p(y) = −½ (yᵀ Σ_y⁻¹ y + log det Σ_y + N log 2π),
        Σ_y = K_1 ⊗ … ⊗ K_D + σ_ε² I

    In the eigenbasis Q of K_XX every term is a sum over N entries,
    yᵀ Σ_y⁻¹ y = ∑_n [Qᵀ y]_n² / (γ_n + σ_ε²) and
    log det Σ_y = ∑_n log(γ_n + σ_ε²), and Qᵀ y is applied one factor at a
    time, so the N × N matrix is never formed. Complexity: O(∑_d N_d³ + N)
    instead of O(N³).

    Algorithm:
        1. Eigendecompose each factor, K_d = Q_d diag(γ_d) Q_dᵀ.
        2. Form γ = γ_1 ⊗ … ⊗ γ_D and add σ_ε².
        3. Rotate the observations, Qᵀ y, one factor at a time.
        4. Sum the data fit, the log-determinant and the constant.

    Args:
        K_factors: Kernel factors K_1, …, K_D, each square and symmetric.
        y: Observations y, flattened row by row over the grid.
        noise_var: Observation-noise variance σ_ε².
        grid_shape: Grid sizes (N_1, …, N_D).

    Returns:
        The log marginal likelihood log p(y).

    Raises:
        ValueError: If ``grid_shape`` does not have one entry per factor
            matching that factor's size, or ``prod(grid_shape) != N``.

    Examples:
        A 2 × 2 grid matches the dense Gaussian log-density:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> K_1 = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        >>> K_2 = jnp.array([[1.0, 0.3], [0.3, 1.0]])
        >>> factors = [
        ...     lx.MatrixLinearOperator(K_1, psd), lx.MatrixLinearOperator(K_2, psd)
        ... ]
        >>> y = jnp.array([0.1, -0.2, 0.3, 0.0])
        >>> log_likelihood = gaussx.kronecker_mll(factors, y, 0.1, (2, 2))
        >>> covariance = jnp.kron(K_1, K_2) + 0.1 * jnp.eye(4)
        >>> dense = jax.scipy.stats.multivariate_normal.logpdf(
        ...     y, jnp.zeros(4), covariance
        ... )
        >>> bool(jnp.allclose(log_likelihood, dense, atol=1e-4))
        True

        A grid that does not match the factors is rejected:

        >>> gaussx.kronecker_mll(
        ...     factors, y, 0.1, (4, 1)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: grid_shape[0] = 4 does not match Kronecker factor 0 of size 2.
    """
    num_obs = y.shape[0]
    _validate_grid(K_factors, grid_shape, num_obs)

    # --- eigenbasis ---
    # K_d = Q_d diag(γ_d) Q_dᵀ
    factor_eigendecompositions = [eig(K_d) for K_d in K_factors]
    factor_eigenvalues = [values for values, _ in factor_eigendecompositions]
    factor_eigenvectors = [vectors for _, vectors in factor_eigendecompositions]
    # γ = γ_1 ⊗ … ⊗ γ_D
    eigenvalues = ft.reduce(jnp.kron, factor_eigenvalues)
    # γ + σ_ε²
    noisy_eigenvalues = eigenvalues + noise_var
    # Qᵀ y with Q = Q_1 ⊗ … ⊗ Q_D, one factor at a time
    rotated_obs = _kron_rotate(factor_eigenvectors, y, grid_shape)

    # --- log-likelihood, O(N) in the eigenbasis ---
    # [Qᵀ y]_n²
    rotated_obs_squared = rotated_obs**2
    # [Qᵀ y]_n² / (γ_n + σ_ε²)
    weighted_obs_squared = rotated_obs_squared / noisy_eigenvalues
    # yᵀ Σ_y⁻¹ y = ∑_n [Qᵀ y]_n² / (γ_n + σ_ε²)
    data_fit = jnp.sum(weighted_obs_squared)
    # log(γ_n + σ_ε²)
    log_noisy_eigenvalues = jnp.log(noisy_eigenvalues)
    # log det Σ_y = ∑_n log(γ_n + σ_ε²)
    logdet = jnp.sum(log_noisy_eigenvalues)
    # N log 2π
    const = num_obs * jnp.log(2.0 * jnp.pi)

    # log p(y) = −½ (yᵀ Σ_y⁻¹ y + log det Σ_y + N log 2π)
    return -0.5 * (data_fit + logdet + const)


def kronecker_posterior_predictive(
    K_factors: list[lx.AbstractLinearOperator],
    y: Float[Array, " N"],
    noise_var: float,
    grid_shape: tuple[int, ...],
    K_cross_factors: list[Float[Array, "Ni_test Ni_train"]],
    *,
    K_test_diag_factors: list[Float[Array, " Ni_test"]],
) -> tuple[Float[Array, " N_test"], Float[Array, " N_test"]]:
    """Posterior mean and variance of a Kronecker GP at grid test inputs.

        μ_* = K_*X Σ_y⁻¹ y
        σ²_m = [K_**]_mm − [K_*X Σ_y⁻¹ K_X*]_mm

    The test cross-covariance and prior variances factorise over the grid
    axes, and so does the cross-covariance B projected onto the eigenbasis
    Q of K_XX, in which Σ_y⁻¹ = Q diag(γ + σ_ε²)⁻¹ Qᵀ:

        K_*X = K_*X,1 ⊗ … ⊗ K_*X,D,   diag(K_**) = diag(K_**,1) ⊗ … ⊗ diag(K_**,D)
        B = K_*X Q = (K_*X,1 Q_1) ⊗ … ⊗ (K_*X,D Q_D)
        μ_* = B diag(γ + σ_ε²)⁻¹ Qᵀ y
        σ²_m = [K_**]_mm − ∑_n [B]_mn² / (γ_n + σ_ε²)

    [B]_mn² is itself a Kronecker product of the per-axis squares B_d². Both
    are computed via the per-factor eigendecompositions in
    O(∑_d N_d³ + N + M) time.

    Algorithm:
        1. Eigendecompose each factor, K_d = Q_d diag(γ_d) Q_dᵀ, and form
           (γ + σ_ε²)⁻¹ with γ = γ_1 ⊗ … ⊗ γ_D.
        2. Rotate the observations, Qᵀ y, and weight them by
           (γ + σ_ε²)⁻¹.
        3. Project the per-axis cross-covariances onto the eigenvectors,
           ``B_d = K_*X,d Q_d``, and apply B_1 ⊗ … ⊗ B_D to the weights for
           the mean.
        4. Apply the element-wise squares B_d² to (γ + σ_ε²)⁻¹ and subtract
           the result from ``diag(K_**)``, clipped at zero.

    Args:
        K_factors: Training kernel factors K_1, …, K_D.
        y: Observations y, flattened row by row over the grid.
        noise_var: Observation-noise variance σ_ε².
        grid_shape: Training grid sizes (N_1, …, N_D).
        K_cross_factors: Per-axis cross-covariances ``K_*X,d`` between the test
            and the training inputs, M_d × N_d.
        K_test_diag_factors: Per-axis prior variances at the test inputs,
            the diagonals of ``K_**,d``, of length M_d.

    Returns:
        Tuple ``(mean, variance)`` holding ``μ_*`` and σ²_m at the M test
        inputs, flattened row by row over the test grid.

    Raises:
        ValueError: If ``grid_shape`` does not match the factors or ``y``
            (as in `kronecker_mll`), or a per-factor list has the wrong length.

    Examples:
        Two axes, matching the dense GP posterior:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> psd = lx.positive_semidefinite_tag
        >>> K_1 = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        >>> K_2 = jnp.array([[1.0, 0.3], [0.3, 1.0]])
        >>> factors = [
        ...     lx.MatrixLinearOperator(K_1, psd), lx.MatrixLinearOperator(K_2, psd)
        ... ]
        >>> y = jnp.array([0.1, -0.2, 0.3, 0.0])
        >>> cross = [jnp.array([[1.0, 0.2]]), jnp.array([[0.4, 0.9], [0.1, 0.1]])]
        >>> prior = [jnp.array([2.0]), jnp.array([1.0, 1.0])]
        >>> mean, variance = gaussx.kronecker_posterior_predictive(
        ...     factors, y, 0.1, (2, 2), cross, K_test_diag_factors=prior
        ... )
        >>> mean.shape, variance.shape
        ((2,), (2,))
        >>> covariance = jnp.kron(K_1, K_2) + 0.1 * jnp.eye(4)
        >>> K_cross = jnp.kron(*cross)
        >>> dense_mean = K_cross @ jnp.linalg.solve(covariance, y)
        >>> bool(jnp.allclose(mean, dense_mean, atol=1e-4))
        True
        >>> explained = jnp.diag(K_cross @ jnp.linalg.solve(covariance, K_cross.T))
        >>> bool(jnp.allclose(variance, 2.0 - explained, atol=1e-4))
        True
    """
    _validate_grid(K_factors, grid_shape, y.shape[0])
    if len(K_cross_factors) != len(K_factors):
        msg = "K_cross_factors must have one matrix per Kronecker factor"
        raise ValueError(msg)
    if len(K_test_diag_factors) != len(K_factors):
        msg = "K_test_diag_factors must have one vector per Kronecker factor"
        raise ValueError(msg)

    # --- eigenbasis ---
    # K_d = Q_d diag(γ_d) Q_dᵀ
    factor_eigendecompositions = [eig(K_d) for K_d in K_factors]
    factor_eigenvalues = [values for values, _ in factor_eigendecompositions]
    factor_eigenvectors = [vectors for _, vectors in factor_eigendecompositions]
    # γ = γ_1 ⊗ … ⊗ γ_D
    eigenvalues = ft.reduce(jnp.kron, factor_eigenvalues)
    # γ + σ_ε²
    noisy_eigenvalues = eigenvalues + noise_var
    # (γ + σ_ε²)⁻¹
    inv_noisy_eigenvalues = 1.0 / noisy_eigenvalues

    # --- mean ---
    # Qᵀ y
    rotated_obs = _kron_rotate(factor_eigenvectors, y, grid_shape)
    # diag(γ + σ_ε²)⁻¹ Qᵀ y
    weights = inv_noisy_eigenvalues * rotated_obs
    # B_d = K_*X,d Q_d (jnp, not einx: `@` keeps the original lowering)
    projected_factors = [
        cross_d @ vectors_d
        for cross_d, vectors_d in zip(K_cross_factors, factor_eigenvectors, strict=True)
    ]
    # μ_* = (B_1 ⊗ … ⊗ B_D) diag(γ + σ_ε²)⁻¹ Qᵀ y
    mean = _kron_matvec(projected_factors, weights, grid_shape)

    # --- variance ---
    # B_d², element-wise
    squared_factors = [factor**2 for factor in projected_factors]
    # [K_*X Σ_y⁻¹ K_X*]_mm = ∑_n [B]_mn² / (γ_n + σ_ε²)
    explained = _kron_matvec(squared_factors, inv_noisy_eigenvalues, grid_shape)
    # diag(K_**) = diag(K_**,1) ⊗ … ⊗ diag(K_**,D)
    test_prior_variance = ft.reduce(jnp.kron, K_test_diag_factors)
    # [K_**]_mm − [K_*X Σ_y⁻¹ K_X*]_mm
    variance = test_prior_variance - explained
    # σ²_m = max([K_**]_mm − [K_*X Σ_y⁻¹ K_X*]_mm, 0)
    variance = jnp.clip(variance, 0.0)

    return mean, variance


def _kron_rotate(
    vecs_list: list[Float[Array, "Ni Ni"]],
    y: Float[Array, " N"],
    grid_shape: tuple[int, ...],
) -> Float[Array, " N"]:
    """Rotate a grid vector into the Kronecker eigenbasis, one factor at a time.

        (Q_1 ⊗ … ⊗ Q_D)ᵀ y

    Algorithm:
        1. Unflatten y onto the grid.
        2. For each axis d, move it last, apply Q_dᵀ along it and move it
           back.
        3. Flatten the result.

    Args:
        vecs_list: Factor eigenvectors Q_1, …, Q_D, one per column.
        y: Grid values, flattened row by row.
        grid_shape: Grid sizes (N_1, …, N_D).

    Returns:
        The rotated vector Qᵀ y, flattened row by row.
    """
    grid_values = _reshape_flat_to_grid(y, grid_shape)

    for axis, eigenvectors in enumerate(vecs_list):
        grid_values = _move_axis_to_last(grid_values, axis)
        # Q_dᵀ along axis d
        # jnp, not einx: `@` keeps the original matmul lowering bit for bit
        grid_values = grid_values @ eigenvectors
        grid_values = _move_last_axis_to(grid_values, axis)

    return rearrange(grid_values, "... -> (...)")


def _kron_matvec(
    A_factors: list[Float[Array, "Mi Ni"]],
    v: Float[Array, " N"],
    grid_shape: tuple[int, ...],
) -> Float[Array, " M"]:
    """Apply a Kronecker product to a grid vector, one factor at a time.

        (A_1 ⊗ … ⊗ A_D) v

    This is Roth's column lemma, as in `gaussx.Kronecker`.

    Algorithm:
        1. Unflatten v onto the grid.
        2. For each axis d, move it last, apply A_d along it and move it
           back.
        3. Flatten the result.

    Args:
        A_factors: Factors A_d, M_d × N_d.
        v: Grid values, flattened row by row.
        grid_shape: Input grid sizes (N_1, …, N_D).

    Returns:
        The product (A_1 ⊗ … ⊗ A_D) v, flattened row by row.
    """
    grid_values = _reshape_flat_to_grid(v, grid_shape)

    for axis, factor in enumerate(A_factors):
        grid_values = _move_axis_to_last(grid_values, axis)
        # A_d along axis d
        # jnp, not einx: an einsum here differs from `@ A_dᵀ` by an ULP in
        # float32 once three or more factors are chained
        grid_values = grid_values @ factor.T
        grid_values = _move_last_axis_to(grid_values, axis)

    return rearrange(grid_values, "... -> (...)")
