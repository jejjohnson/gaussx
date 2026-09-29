"""Conditional Gaussian distribution from partial observations."""

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
    r"""Compute ``p(x_A | x_B = b)`` from a joint Gaussian ``p(x_A, x_B)``.

    Given a joint distribution $\mathcal{N}(\mu, \Sigma)$ and
    observed indices *B* with values *b*, returns the conditional
    distribution over the remaining indices *A*:

    $$
    \begin{aligned}
    \mu_{A|B} &= \mu_A + \Sigma_{AB} \Sigma_{BB}^{-1} (b - \mu_B) \\
    \Sigma_{A|B} &= \Sigma_{AA} - \Sigma_{AB} \Sigma_{BB}^{-1} \Sigma_{BA}
    \end{aligned}
    $$

    Args:
        loc: Mean vector of the joint distribution, shape ``(N,)``.
        cov: Covariance operator of the joint distribution, shape ``(N, N)``.
        obs_idx: Indices of the observed variables, shape ``(M,)``.
        obs_values: Observed values, shape ``(M,)``.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        Tuple ``(cond_mean, cond_cov)`` — mean and covariance of the
        conditional distribution over unobserved variables.

    Raises:
        ValueError: If ``obs_idx`` is not 1D, does not match
            ``obs_values`` in shape, or -- when it is concrete -- is out
            of bounds or contains duplicates.
        EquinoxRuntimeError: At run time, if a *traced* ``obs_idx`` is
            out of bounds or contains duplicates.

    Note:
        Usable under ``jax.jit``. Concrete indices (NumPy or JAX constants,
        including ones closed over by the jitted function) are validated
        on the host; traced indices are validated at run time with
        `equinox.error_if`, since an out-of-bounds gather would otherwise
        be silently clamped.
    """
    N = loc.shape[0]
    obs_values = jnp.asarray(obs_values, dtype=loc.dtype)

    if isinstance(obs_idx, Tracer):
        obs_idx = obs_idx.astype(jnp.int32)
        _check_shapes(obs_idx, obs_values)
        bad = jnp.any((obs_idx < 0) | (obs_idx >= N)) | jnp.any(
            jnp.diff(jnp.sort(obs_idx)) == 0
        )
        obs_values = eqx.error_if(
            obs_values,
            bad,
            f"obs_idx must be within bounds [0, {N}) and must not contain duplicates.",
        )
    else:
        idx_np = np.asarray(obs_idx)
        _check_shapes(idx_np, obs_values)
        if np.any((idx_np < 0) | (idx_np >= N)):
            raise ValueError(f"obs_idx must be within bounds [0, {N}).")
        if np.any(np.diff(np.sort(idx_np)) == 0):
            raise ValueError("obs_idx must not contain duplicates.")
        obs_idx = jnp.asarray(idx_np, dtype=jnp.int32)

    # Build mask for unobserved indices
    mask = jnp.ones(N, dtype=bool).at[obs_idx].set(False)
    free_idx = jnp.where(mask, size=N - obs_idx.shape[0])[0]

    # Extract sub-blocks via structural dispatch — avoids materializing
    # the full ``(N, N)`` joint covariance for structured operators
    # (Diagonal, BlockDiag). Falls back to full materialization for
    # unstructured operators where there is no efficient alternative.
    Sigma_AA = submatrix(cov, free_idx, free_idx)
    Sigma_AB = submatrix(cov, free_idx, obs_idx)
    Sigma_BB = submatrix(cov, obs_idx, obs_idx)

    mu_A = loc[free_idx]
    mu_B = loc[obs_idx]

    # Sigma_BB^{-1} (b - mu_B)
    residual = obs_values - mu_B
    Sigma_BB_op = lx.MatrixLinearOperator(Sigma_BB, lx.positive_semidefinite_tag)
    alpha = dispatch_solve(Sigma_BB_op, residual, solver)

    # Conditional mean: mu_A + Sigma_AB @ alpha
    cond_mean = mu_A + Sigma_AB @ alpha

    # Sigma_BB^{-1} Sigma_BA — single matrix solve (one factorization for
    # the whole RHS in the default PSD path).
    Sigma_BA = Sigma_AB.T
    X = solve_matrix(Sigma_BB_op, Sigma_BA, solver=solver)

    # Conditional covariance: Sigma_AA - Sigma_AB @ X
    cond_cov_mat = Sigma_AA - Sigma_AB @ X

    # Symmetrize for numerical stability
    cond_cov_mat = symmetrize(cond_cov_mat)
    cond_cov = lx.MatrixLinearOperator(cond_cov_mat, lx.positive_semidefinite_tag)

    return cond_mean, cond_cov


def _check_shapes(obs_idx, obs_values: Float[Array, " M"]) -> None:
    if obs_idx.ndim != 1:
        raise ValueError("obs_idx must be a 1D array.")
    if obs_values.shape != obs_idx.shape:
        raise ValueError("obs_values must have the same shape as obs_idx.")
