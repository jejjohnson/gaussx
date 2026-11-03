"""Conditional Gaussian p(x_A ∣ x_B = b) of a joint Gaussian from observed entries.

The joint x ~ 𝒩(μ, Σ) splits into the free entries x_A, indexed by A, and the
observed entries x_B, indexed by B, with observed values b. The blocks of μ
and Σ follow the index sets: μ_A, μ_B, Σ_AA, Σ_AB, Σ_BB and Σ_BA = Σ_ABᵀ.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.core import Tracer
from jaxtyping import Array, Float, Int

from gaussx._linalg._linalg import solve_matrix
from gaussx._linalg._symmetrize import symmetrize
from gaussx._primitives._submatrix import submatrix
from gaussx._strategies._base import AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_solve


def conditional(
    loc: Float[Array, " N"],
    cov: lx.AbstractLinearOperator,
    obs_idx: Int[Array, " M"],
    obs_values: Float[Array, " M"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " R"], lx.AbstractLinearOperator]:
    """Conditional p(x_A ∣ x_B = b) of a joint Gaussian 𝒩(μ, Σ) from observed entries.

        μ_A|B = μ_A + Σ_AB Σ_BB⁻¹ (b − μ_B)
        Σ_A|B = Σ_AA − Σ_AB Σ_BB⁻¹ Σ_BA

    B holds the observed indices and A the remaining, free ones, in
    increasing order.

    Algorithm:
        1. Validate the observed indices B (on the host when concrete, with
           `equinox.error_if` when traced).
        2. Extract Σ_AA, Σ_AB and Σ_BB by structural dispatch, so structured
           operators (diagonal, block-diagonal) are not materialised whole.
        3. Solve Σ_BB⁻¹ (b − μ_B) and update the mean.
        4. Solve Σ_BB⁻¹ Σ_BA with one factorisation, update the covariance
           and symmetrise it.

    Args:
        loc: Mean μ of the joint Gaussian.
        cov: Covariance Σ of the joint Gaussian.
        obs_idx: Observed indices B.
        obs_values: Observed values b of x_B.
        solver: Solver strategy for Σ_BB; ``None`` uses structural dispatch.

    Returns:
        Tuple ``(cond_mean, cond_cov)`` holding μ_A|B and Σ_A|B, the latter
        as a PSD-tagged `lineax.MatrixLinearOperator`.

    Raises:
        ValueError: If ``obs_idx`` is not 1D, does not match ``obs_values``
            in shape, or, when it is concrete, is out of bounds or contains
            duplicates.
        EquinoxRuntimeError: At run time, if a traced ``obs_idx`` is out of
            bounds or contains duplicates.

    Note:
        Usable under ``jax.jit``. Concrete indices (NumPy or JAX constants,
        including ones closed over by the jitted function) are validated
        on the host; traced indices are validated at run time with
        `equinox.error_if`, since an out-of-bounds gather would otherwise
        be silently clamped.

    Examples:
        Observe the second of two correlated entries:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> joint = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        >>> covariance = lx.MatrixLinearOperator(joint, lx.positive_semidefinite_tag)
        >>> mean, cov = gaussx.conditional(
        ...     jnp.zeros(2), covariance, jnp.array([1]), jnp.array([1.0])
        ... )
        >>> bool(jnp.allclose(mean, jnp.array([0.5])))
        True
        >>> bool(jnp.allclose(cov.as_matrix(), jnp.array([[1.5]])))
        True

        Independent entries keep their prior block:

        >>> diagonal = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 3.0]))
        >>> mean, cov = gaussx.conditional(
        ...     jnp.ones(3), diagonal, jnp.array([0]), jnp.array([5.0])
        ... )
        >>> bool(jnp.allclose(cov.as_matrix(), jnp.diag(jnp.array([2.0, 3.0]))))
        True

        Duplicate indices are rejected:

        >>> gaussx.conditional(
        ...     jnp.zeros(2), covariance, jnp.array([1, 1]), jnp.ones(2)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: obs_idx must not contain duplicates.
    """
    size = loc.shape[0]
    obs_values = jnp.asarray(obs_values, dtype=loc.dtype)

    if isinstance(obs_idx, Tracer):
        obs_idx = obs_idx.astype(jnp.int32)
        _check_shapes(obs_idx, obs_values)
        bad = jnp.any((obs_idx < 0) | (obs_idx >= size)) | jnp.any(
            jnp.diff(jnp.sort(obs_idx)) == 0
        )
        # Attached to obs_idx, which feeds both outputs, so the check cannot
        # be dead-code-eliminated when a caller uses only one of them.
        obs_idx = eqx.error_if(
            obs_idx,
            bad,
            f"obs_idx must be within bounds [0, {size}) and must not contain "
            "duplicates.",
        )
    else:
        # Cast before validating, so e.g. [1.1, 1.2] is caught as the
        # duplicate [1, 1] it becomes.
        idx_np = np.asarray(obs_idx).astype(np.int32)
        _check_shapes(idx_np, obs_values)
        if np.any((idx_np < 0) | (idx_np >= size)):
            raise ValueError(f"obs_idx must be within bounds [0, {size}).")
        if np.any(np.diff(np.sort(idx_np)) == 0):
            raise ValueError("obs_idx must not contain duplicates.")
        obs_idx = jnp.asarray(idx_np)

    # --- index sets ---
    # A, the complement of B, in increasing order
    free_mask = jnp.ones(size, dtype=bool).at[obs_idx].set(False)
    free_idx = jnp.where(free_mask, size=size - obs_idx.shape[0])[0]

    # --- blocks ---
    # Structural dispatch avoids materialising the whole joint Σ for
    # structured operators (Diagonal, BlockDiag); unstructured ones fall
    # back to full materialisation, where there is no cheaper alternative.
    # Σ_AA
    free_covariance = submatrix(cov, free_idx, free_idx)
    # Σ_AB
    cross_covariance = submatrix(cov, free_idx, obs_idx)
    # Σ_BB
    observed_covariance = submatrix(cov, obs_idx, obs_idx)
    # μ_A
    free_mean = loc[free_idx]
    # μ_B
    observed_mean = loc[obs_idx]

    # --- mean ---
    # b − μ_B
    residual = obs_values - observed_mean
    observed_covariance_op = lx.MatrixLinearOperator(
        observed_covariance, lx.positive_semidefinite_tag
    )
    # Σ_BB⁻¹ (b − μ_B)
    solved_residual = dispatch_solve(observed_covariance_op, residual, solver)
    # μ_A|B = μ_A + Σ_AB Σ_BB⁻¹ (b − μ_B)
    cond_mean = free_mean + cross_covariance @ solved_residual

    # --- covariance ---
    # jnp, not einx: A or B may be empty here, and einx rejects a zero axis.
    # Σ_BA = Σ_ABᵀ
    cross_covariance_transpose = cross_covariance.T
    # Σ_BB⁻¹ Σ_BA, one factorisation for every column in the PSD path
    solved_cross = solve_matrix(
        observed_covariance_op, cross_covariance_transpose, solver=solver
    )
    # jnp, not einx: A or B may be empty here, and einx rejects a zero axis.
    # Σ_AB Σ_BB⁻¹ Σ_BA
    correction = cross_covariance @ solved_cross
    # Σ_A|B = Σ_AA − Σ_AB Σ_BB⁻¹ Σ_BA
    cond_cov_matrix = free_covariance - correction
    # Round-off leaves Σ_A|B slightly asymmetric.
    cond_cov_matrix = symmetrize(cond_cov_matrix)
    cond_cov = lx.MatrixLinearOperator(cond_cov_matrix, lx.positive_semidefinite_tag)

    return cond_mean, cond_cov


def _check_shapes(obs_idx, obs_values: Float[Array, " M"]) -> None:
    """Check that the observed indices B and values b are matching 1D arrays.

    Args:
        obs_idx: Observed indices B, as a NumPy or JAX array.
        obs_values: Observed values b.

    Raises:
        ValueError: If ``obs_idx`` is not 1D or its shape differs from
            ``obs_values``.
    """
    if obs_idx.ndim != 1:
        raise ValueError("obs_idx must be a 1D array.")
    if obs_values.shape != obs_idx.shape:
        raise ValueError("obs_values must have the same shape as obs_idx.")
