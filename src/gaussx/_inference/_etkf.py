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
    r"""Ensemble Transform Kalman Filter (ETKF) analysis weights.

    Deterministic (perturbed-obs-free) ensemble square-root analysis in the
    ``J``-dimensional ensemble space (Bishop et al. 2001; Hunt et al. 2007).
    With raw observation perturbations ``Y = H X'^f`` (columns are members) and
    ``d = y - H x_bar^f``,

    $$
    \tilde{A}^{-1} = \tfrac{J-1}{\lambda} I + Y^T R^{-1} Y, \qquad
    \bar{w} = \tilde{A}\, Y^T R^{-1} d, \qquad
    W = \big((J-1)\,\tilde{A}\big)^{1/2},
    $$

    where ``lambda`` is the (multiplicative) ``inflation`` and ``W`` is the
    **symmetric** square root. The analysis ensemble is reconstructed as

    $$
    \bar{x}^a = \bar{x}^f + X'^f \bar{w}, \qquad X'^a = X'^f\, W.
    $$

    The symmetric (eigendecomposition) square root -- not a Cholesky factor --
    is required: because the observation perturbations are zero-mean, ``1`` is
    an eigenvector of ``W`` (with eigenvalue ``sqrt(lambda)``), which makes the
    transform exactly mean-preserving (``sum_j X'^a_j = 0``).

    Cost. ``R^{-1}`` is applied to the ``J + 1`` right-hand sides
    ``[Y^T, d]`` in one `solve_rows` call, so a structured ``R`` (diagonal,
    `gaussx.BlockDiag`, `gaussx.Kronecker`, ...) keeps its own solve and is
    never materialised, and a dense ``R`` is factored once. The ensemble-space
    algebra then takes a single ``(J, J)`` eigendecomposition
    ``\tilde{A}^{-1} = V \operatorname{diag}(s) V^T``, from which both
    ``\tilde{A} = V \operatorname{diag}(1/s) V^T`` and
    ``W = V \operatorname{diag}(\sqrt{(J-1)/s}) V^T`` are read off (Hunt et
    al. 2007), with no explicit inverse. Its derivative is taken in closed
    form in the eigenbasis (Daleckii-Krein), which stays finite at the
    repeated prior eigenvalue ``(J - 1)/\lambda`` that has multiplicity
    ``J - M`` whenever ``M < J``.

    Args:
        obs_particles: Forecast ensemble in observation space, shape ``(J, M)``.
        y: Observation vector, shape ``(M,)``.
        obs_noise: Observation error covariance operator ``R``, shape ``(M, M)``.
        inflation: Multiplicative covariance inflation ``lambda >= 1``, applied
            to the prior term ``(J - 1) / lambda``.
        solver: Optional solver strategy for the ``R^{-1}`` solves. ``None``
            uses structural dispatch.

    Returns:
        ``(w_mean, transform)`` where ``w_mean`` has shape ``(J,)`` and
        ``transform`` has shape ``(J, J)``. Apply to forecast state
        perturbations ``Xp`` (shape ``(J, N)``) as
        ``x_bar^a = x_bar^f + w_mean @ Xp`` and ``X'^a = transform @ Xp``.

    Raises:
        ValueError: If ``J < 2`` (the prior term ``(J - 1) / lambda`` is then
            zero and the ensemble-space precision singular), if ``y`` or
            ``obs_noise`` do not match ``M``, or if a concrete ``inflation``
            is not positive.
    """
    _check_observation_shapes(obs_particles, y, obs_noise, observation_name="y")
    n_ens = obs_particles.shape[0]
    if n_ens < 2:
        raise ValueError(
            "etkf_transform requires J >= 2 ensemble members (the prior term "
            f"is (J - 1) / inflation); got J={n_ens}."
        )
    if isinstance(inflation, (int, float)) and inflation <= 0:
        raise ValueError(f"inflation must be positive, got {inflation}.")
    obs_mean = reduce(obs_particles, "J M -> M", "mean")
    obs_pert = einx.subtract("J M, M -> J M", obs_particles, obs_mean)  # zero-mean

    # R^{-1} applied to all J + 1 right-hand sides at once: one structured
    # solve, so a dense R is factored once and a structured R never densified.
    rhs = jnp.vstack([obs_pert, y - obs_mean])  # (J + 1, M)
    weighted = solve_rows(obs_noise, rhs, solver=solver)  # (J + 1, M)
    rinv_pert, rinv_d = weighted[:-1], weighted[-1]

    eye = jnp.eye(n_ens, dtype=rinv_pert.dtype)
    precision = (n_ens - 1) / inflation * eye + einsum(
        obs_pert, rinv_pert, "J M, K M -> J K"
    )
    analysis_cov, inv_sqrt = _inverse_and_inverse_sqrt(symmetrize(precision))

    w_mean = einsum(
        analysis_cov, einsum(obs_pert, rinv_d, "J M, M -> J"), "J K, K -> J"
    )
    transform = jnp.sqrt(jnp.asarray(n_ens - 1, dtype=inv_sqrt.dtype)) * inv_sqrt
    return w_mean, transform


@jax.custom_jvp
def _inverse_and_inverse_sqrt(
    matrix: Float[Array, "J J"],
) -> tuple[Float[Array, "J J"], Float[Array, "J J"]]:
    """``(A^{-1}, A^{-1/2})`` of an SPD matrix from a single `eigh`.

    The custom JVP exists for the same reason as `dense_symmetric_sqrt`'s:
    differentiating through `jax.numpy.linalg.eigh` divides by eigenvalue
    gaps and is non-finite at a repeated eigenvalue, which `etkf_transform`'s
    ensemble-space precision always has when ``M < J - 1``. See
    `_inverse_and_inverse_sqrt_jvp` for the gap-free derivative.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return _spectral_inverse_and_inverse_sqrt(eigenvalues, eigenvectors)


def _spectral_inverse_and_inverse_sqrt(
    eigenvalues: Float[Array, " J"],
    eigenvectors: Float[Array, "J J"],
) -> tuple[Float[Array, "J J"], Float[Array, "J J"]]:
    def _apply(values: Float[Array, " J"]) -> Float[Array, "J J"]:
        scaled = einx.multiply("J i, i -> J i", eigenvectors, values)
        return einsum(scaled, eigenvectors, "J i, K i -> J K")  # V diag V^T

    return _apply(1.0 / eigenvalues), _apply(1.0 / jnp.sqrt(eigenvalues))


@_inverse_and_inverse_sqrt.defjvp
def _inverse_and_inverse_sqrt_jvp(primals, tangents):
    r"""Daleckii-Krein derivative of ``A^{-1}`` and ``A^{-1/2}``.

    For a spectral function ``f(A) = V f(\Lambda) V^T`` the derivative is
    ``V (F \circ V^T dA V) V^T`` with the divided differences
    ``F_ij = (f(s_i) - f(s_j)) / (s_i - s_j)`` (``f'(s_i)`` on ties). For
    these two functions the divided differences have gap-free closed forms,

    $$
    \frac{s_i^{-1} - s_j^{-1}}{s_i - s_j} = -\frac{1}{s_i s_j},
    \qquad
    \frac{s_i^{-1/2} - s_j^{-1/2}}{s_i - s_j}
        = -\frac{1}{\sqrt{s_i s_j}\,(\sqrt{s_i} + \sqrt{s_j})},
    $$

    which reduce to ``f'(s_i)`` on the diagonal, so the derivative stays
    finite however many eigenvalues coincide.
    """
    (matrix,) = primals
    (tangent,) = tangents
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    primal_out = _spectral_inverse_and_inverse_sqrt(eigenvalues, eigenvectors)

    # eigh reads one triangle, so project the tangent onto the symmetric part.
    tangent = 0.5 * (tangent + rearrange(tangent, "J K -> K J"))
    rotated = einsum(eigenvectors, tangent, eigenvectors, "a i, a b, b j -> i j")
    root = jnp.sqrt(eigenvalues)
    inv_diff = -1.0 / einx.multiply("i, j -> i j", eigenvalues, eigenvalues)
    inv_sqrt_diff = -1.0 / einx.multiply(
        "i, j, i j -> i j",
        root,
        root,
        einx.add("i, j -> i j", root, root),
    )

    def _back(coefficients: Float[Array, "J J"]) -> Float[Array, "J J"]:
        return einsum(
            eigenvectors,
            coefficients * rotated,
            eigenvectors,
            "J i, i j, K j -> J K",
        )

    return primal_out, (_back(inv_diff), _back(inv_sqrt_diff))
