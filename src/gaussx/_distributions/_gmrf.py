"""Gaussian Markov random fields x ~ 𝒩(μ, Λ⁻¹) with a structured precision Λ.

`GaussianMRF` is a proper field with precision Λ, `ConstrainedGMRF` the same
field conditioned on hard linear constraints A_c x = e, `IntrinsicGMRF` the
improper field x ~ 𝒩(μ, (τ R)⁺) with structure matrix R, precision scale τ
and null-space basis V, and `BYM2GMRF` the BYM2 pair with mixing φ on a
scaled structure R*. All four are NumPyro distributions whose precision is a
lineax operator, and every operation dispatches on its structure:

- sample, in order: (1) a factorisable precision (dense, `BlockTriDiag`, or a
  `SparseOperator` through its sparse Cholesky factor) gives μ + L_Λ⁻ᵀ z;
  (2) a Kronecker-structured one (`Kronecker`, `KroneckerSum`,
  `SpectralFunction`, `DiagonalizedOperator`) the symmetric root μ + Λ^(−½) z
  in its factored eigenbasis; (3) otherwise perturbation-optimisation: with
  Λ = ∑_k F_kᵀ F_k, the right-hand side ∑_k F_kᵀ z_k ~ 𝒩(0, Λ), so
  Λ⁻¹ ∑_k F_kᵀ z_k ~ 𝒩(0, Λ⁻¹) costs one CG solve per draw and no
  factorisation;
- log_prob: log det Λ from a known value, a strategy, or the structural
  `logdet`;
- marginal variances: `diag_inv` (block or sparse Takahashi, factor
  eigenvectors);
- constraints A_c x = e: conditioning by kriging (Rue & Held, 2005, §2.3.3),
  x ↦ x − W S⁻¹ (A_c x − e) with W = Λ⁻¹ A_cᵀ and S = A_c Λ⁻¹ A_cᵀ, at D_c
  extra solves.

D_x is the event size, z ~ 𝒩(0, I) the standard normal noise of a draw, L_Λ
the lower Cholesky factor of Λ = L_Λ L_Λᵀ, and D_c the number of constraints
(rows of A_c) or of null-space directions (columns of V).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
import numpy as np
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike, Float
from numpyro.distributions.util import lazy_property, validate_sample

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._distributions._utils import _reshape_batch, _reshape_samples
from gaussx._einx import einsum, rearrange, reduce
from gaussx._gmrf._areal import _add_ridge, _as_sparse, bym2_precision
from gaussx._linalg._diag_inv import diag_inv
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._factored_eigen import FactoredEigen, _reciprocal, factored_eigen
from gaussx._operators._kronecker import Kronecker
from gaussx._operators._kronecker_sum import KroneckerSum, KroneckerSumSqrt
from gaussx._operators._sparse import SparseOperator
from gaussx._operators._spectral_function import SpectralFunction
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._logdet import pseudo_logdet
from gaussx._primitives._solve import solve
from gaussx._strategies._base import AbstractLogdetStrategy, AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_logdet, dispatch_solve
from gaussx._strategies._sparse_cholesky import SparseCholeskySolver


_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
# Operators sampled in their factored eigenbasis (dispatch branch 2).
_EIGEN = (Kronecker, KroneckerSum, SpectralFunction, DiagonalizedOperator)
# Operators factored by Cholesky even when factors are given (branch 1).
_FACTORABLE = (
    lx.MatrixLinearOperator,
    lx.DiagonalLinearOperator,
    lx.IdentityLinearOperator,
)

# f(z): a zero-mean draw from standard normal noise z.
_Draw = Callable[[Array], Array]


# ---------------------------------------------------------------------------
# Sampling dispatch
# ---------------------------------------------------------------------------


def _unwrap(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """Strip every `lineax.TaggedLinearOperator` wrapper from an operator.

    Args:
        operator: The operator, possibly tagged.

    Returns:
        The innermost untagged operator.
    """
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    return operator


def _is_factorable(operator: lx.AbstractLinearOperator) -> bool:
    """Whether Cholesky is the cheapest exact path for a precision Λ.

    True for a dense, diagonal or identity Λ and a symmetric `BlockTriDiag`.

    Args:
        operator: The untagged precision Λ.

    Returns:
        ``True`` if Λ should be sampled through its Cholesky factor.
    """
    return isinstance(operator, _FACTORABLE) or (
        isinstance(operator, BlockTriDiag) and operator.symmetric
    )


def _cholesky_draw(operator: lx.AbstractLinearOperator) -> _Draw:
    """Draw map z ↦ L_Λ⁻ᵀ z from the structured Cholesky factor of Λ.

        Λ = L_Λ L_Λᵀ,   L_Λ⁻ᵀ z ~ 𝒩(0, Λ⁻¹)

    Args:
        operator: The precision Λ, factored here once.

    Returns:
        The map z ↦ L_Λ⁻ᵀ z.
    """
    # Λ = L_Λ L_Λᵀ
    cholesky_factor = cholesky(operator)
    if isinstance(cholesky_factor, lx.MatrixLinearOperator):
        factor_matrix = cholesky_factor.matrix
        # L_Λ⁻ᵀ z, by a dense triangular solve
        return lambda noise: jsl.solve_triangular(
            factor_matrix, noise, lower=True, trans="T"
        )
    # L_Λᵀ
    upper_factor = cholesky_factor.T
    # L_Λ⁻ᵀ z, by the structured solve
    return lambda noise: solve(upper_factor, noise)


def _eigen_draw(eigen: FactoredEigen, z: Array, *, pinv: bool = False) -> Array:
    """Symmetric-root draw Λ^(−½) z in the factored eigenbasis of Λ.

        Λ = W⁻¹ diag(γ) W,   Λ^(−½) z = W⁻¹ diag(γ^(−½)) W z

    W is the joint forward transform of `FactoredEigen` and γ its eigenvalue
    tensor. With ``pinv``, numerically zero eigenvalues get a zero root, so
    the draw is from 𝒩(0, Λ⁺).

    Args:
        eigen: Factored eigendecomposition of Λ, holding W and γ.
        z: Standard normal noise z.
        pinv: Use the pseudo-inverse γ⁺ in place of γ⁻¹.

    Returns:
        The draw Λ^(−½) z, or (Λ⁺)^½ z with ``pinv``.
    """
    # γ⁻¹, or γ⁺ with pinv
    inverse_eigenvalues = _reciprocal(eigen.eigenvalues, pinv=pinv)
    # γ^(−½)
    inverse_root_eigenvalues = jnp.sqrt(inverse_eigenvalues)
    # W z
    coeffs = eigen.to_eigenbasis(z)
    # diag(γ^(−½)) W z
    scaled_coeffs = coeffs * inverse_root_eigenvalues
    # Λ^(−½) z = W⁻¹ diag(γ^(−½)) W z
    draw = eigen.from_eigenbasis(scaled_coeffs)
    # Drop the imaginary part a real-by-construction basis leaves behind.
    return eigen._output(draw, z)


def _cg_solve(operator: lx.AbstractLinearOperator, vector: Array) -> Array:
    """CG solve of Λ x = b at tolerance √eps of the dtype.

        x = Λ⁻¹ b

    CG converges on a singular Λ too, when b lies in range(Λ).

    Args:
        operator: The symmetric positive (semi)definite Λ.
        vector: Right-hand side b.

    Returns:
        The solution x.
    """
    # √eps of b's dtype
    tolerance = float(jnp.finfo(vector.dtype).eps) ** 0.5
    # x = Λ⁻¹ b
    return lx.linear_solve(
        operator, vector, lx.CG(rtol=tolerance, atol=tolerance)
    ).value


def _perturbation_draw(
    precision: lx.AbstractLinearOperator,
    factors: tuple[lx.AbstractLinearOperator, ...],
    solver: AbstractSolverStrategy | None,
) -> tuple[int, _Draw]:
    """Perturbation-optimisation draw map: solve Λ x = ∑_k F_kᵀ z_k per draw.

        Λ = ∑_k F_kᵀ F_k,   ∑_k F_kᵀ z_k ~ 𝒩(0, Λ),   x ~ 𝒩(0, Λ⁻¹)

    Algorithm:
        1. Split the noise z into one block z_k per factor F_k.
        2. Form ∑_k F_kᵀ z_k, whose covariance is Λ.
        3. Solve Λ x = ∑_k F_kᵀ z_k with ``solver``, or by CG without one.

    Args:
        precision: The precision Λ.
        factors: The factors F_k with Λ = ∑_k F_kᵀ F_k.
        solver: Strategy for the solve with Λ; ``None`` runs CG.

    Returns:
        Tuple ``(noise_size, draw)``: the length of z (the total number of
        rows of the F_k) and the map z ↦ Λ⁻¹ ∑_k F_kᵀ z_k.
    """
    # rows of each F_k, the length of its noise block z_k
    factor_sizes = [factor.out_size() for factor in factors]
    split_points = np.cumsum(factor_sizes)[:-1].tolist()
    psd_precision = lx.TaggedLinearOperator(precision, _PSD)

    def draw(noise: Array) -> Array:
        # z = (z_1, z_2, …), one block per factor
        noise_blocks = jnp.split(noise, split_points)
        # ∑_k F_kᵀ z_k
        perturbed_rhs = sum(
            factor.transpose().mv(block)
            for factor, block in zip(factors, noise_blocks, strict=True)
        )
        if solver is not None:
            # x = Λ⁻¹ ∑_k F_kᵀ z_k, by the given strategy
            return solver.solve(psd_precision, perturbed_rhs)
        # x = Λ⁻¹ ∑_k F_kᵀ z_k, by CG
        return _cg_solve(psd_precision, perturbed_rhs)

    return sum(factor_sizes), draw


def _precision_sampler(
    precision: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy | None,
    factors: tuple[lx.AbstractLinearOperator, ...] | None,
) -> tuple[int, _Draw]:
    """Draw map f with f(z) ~ 𝒩(0, Λ⁻¹), chosen from the structure of Λ.

        f(z) ~ 𝒩(0, Λ⁻¹)   for   z ~ 𝒩(0, I)

    The factorisation is done here, once, outside any ``vmap`` over draws.

    Algorithm:
        1. A `SparseOperator` with no factors, or with a sparse Cholesky
           solver: f(z) = Pᵀ L⁻ᵀ z from the sparse factor P Λ Pᵀ = L Lᵀ.
        2. Dense, diagonal or symmetric `BlockTriDiag`: f(z) = L_Λ⁻ᵀ z.
        3. Kronecker-structured with a factored eigenbasis: f(z) = Λ^(−½) z.
        4. Factors F_k given: perturbation-optimisation.
        5. Anything else: the structured Cholesky, f(z) = L_Λ⁻ᵀ z.

    Args:
        precision: The precision Λ.
        solver: Solve strategy; a `SparseCholeskySolver` keeps a sparse Λ
            on branch 1 even when factors are given; on branch 4 it does
            the solve with Λ (CG when ``None``).
        factors: The factors F_k with Λ = ∑_k F_kᵀ F_k, or ``None``.

    Returns:
        Tuple ``(noise_size, draw)``: the length of z and the map f.
    """
    inner = _unwrap(precision)
    event_size = precision.in_size()
    if isinstance(inner, SparseOperator) and (
        factors is None or isinstance(solver, SparseCholeskySolver)
    ):
        if not isinstance(solver, SparseCholeskySolver):
            solver = SparseCholeskySolver()
        # Pᵀ L⁻ᵀ z with P Λ Pᵀ = L Lᵀ
        return event_size, solver.factor(inner).solve_lower_transpose
    if _is_factorable(inner):
        # L_Λ⁻ᵀ z
        return event_size, _cholesky_draw(inner)
    if isinstance(inner, _EIGEN):
        # Λ = W⁻¹ diag(γ) W
        eigen = factored_eigen(inner)
        if eigen is not None:
            # Λ^(−½) z
            return event_size, lambda noise: _eigen_draw(eigen, noise)
    if factors is not None:
        # Λ⁻¹ ∑_k F_kᵀ z_k
        return _perturbation_draw(precision, factors, solver)
    # L_Λ⁻ᵀ z
    return event_size, _cholesky_draw(inner)


def _draw_samples(
    key: jax.Array | None,
    sample_shape: tuple[int, ...],
    noise_size: int,
    draw: _Draw,
    dtype: jnp.dtype,
) -> Float[Array, "*sample N"]:
    """Zero-mean draws f(z), one ``normal`` call for every sample.

        z ~ 𝒩(0, I),   f(z) ~ 𝒩(0, Λ⁻¹)

    Args:
        key: PRNG key; ``None`` means ``jax.random.PRNGKey(0)``.
        sample_shape: Leading sample shape.
        noise_size: Length of the noise z of one draw.
        draw: The draw map f.
        dtype: Dtype of the noise z.

    Returns:
        Draws f(z), shape ``sample_shape + (N,)``.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    num_draws = math.prod(sample_shape)
    # z ~ 𝒩(0, I), one row per draw
    noise = jax.random.normal(key, (num_draws, noise_size), dtype=dtype)
    # f(z), one row per draw
    draws = jax.vmap(draw)(noise)
    return _reshape_samples(draws, sample_shape)


def _flat(value: Float[Array, "*sample N"]) -> Float[Array, "S N"]:
    """Flatten every leading sample axis into one.

    Args:
        value: Points, shape ``(..., N)``.

    Returns:
        The points, one row per sample.
    """
    return rearrange(value, "... state -> (...) state")


def _quadratic(operator: lx.AbstractLinearOperator, residual: Array) -> Array:
    """Quadratic form rᵀ Λ r for each row r of ``residual``.

        rᵀ Λ r,   r = x − μ

    Args:
        operator: The symmetric operator Λ (a precision, or a structure R).
        residual: Residuals r, shape ``(..., N)``.

    Returns:
        The quadratic forms, shape ``(...)``.
    """
    flat_residual = _flat(residual)

    def quadratic_form(row: Array) -> Array:
        # Λ r
        weighted_row = operator.mv(row)
        # rᵀ Λ r
        return einsum(row, weighted_row, "state, state ->")

    values = jax.vmap(quadratic_form)(flat_residual)
    return _reshape_batch(values, residual.shape[:-1])


def _solve_strategy(
    precision: lx.AbstractLinearOperator, solver: AbstractSolverStrategy | None
) -> AbstractSolverStrategy | None:
    """The solve strategy for Λ: ``solver``, or sparse Cholesky for a sparse Λ.

    Args:
        precision: The precision Λ.
        solver: The caller's strategy, or ``None``.

    Returns:
        ``solver``, or a `SparseCholeskySolver` when it is ``None`` and Λ is
        a `SparseOperator`.
    """
    if solver is None and isinstance(_unwrap(precision), SparseOperator):
        return SparseCholeskySolver()
    return solver


def _solve_fn(
    precision: lx.AbstractLinearOperator, solver: AbstractSolverStrategy | None
) -> Callable[[Array], Array]:
    """Solve map b ↦ Λ⁻¹ b, factoring a sparse precision once for many solves.

    Args:
        precision: The precision Λ.
        solver: The caller's strategy, or ``None`` for structural dispatch.

    Returns:
        The map b ↦ Λ⁻¹ b.
    """
    strategy = _solve_strategy(precision, solver)
    if isinstance(strategy, SparseCholeskySolver):
        # P Λ Pᵀ = L Lᵀ, factored once
        return strategy.factor(precision).solve
    # Λ⁻¹ b
    return lambda rhs: dispatch_solve(precision, rhs, strategy)


class _Sqrt(eqx.Module):
    """Square root γ ↦ √f(γ) of a spectral function f.

    A pytree, so the parameters of f stay leaves.

    Attributes:
        fn: The spectral function f.
    """

    fn: Callable[[Array], Array]

    def __call__(self, eigenvalues: Array) -> Array:
        """Evaluate √f(γ).

        Args:
            eigenvalues: Eigenvalue tensor γ.

        Returns:
            The tensor √f(γ).
        """
        # f(γ)
        values = self.fn(eigenvalues)
        # √f(γ)
        return jnp.sqrt(values)


def _symmetric_root(
    operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator | None:
    """Symmetric square root Λ^½ of a grid precision.

        (Λ^½)ᵀ Λ^½ = Λ

    Args:
        operator: The untagged precision Λ.

    Returns:
        Λ^½ for a `SpectralFunction` or a `KroneckerSum`, else ``None``.
    """
    if isinstance(operator, SpectralFunction):
        # f(A)^½ = W⁻¹ diag(√f(γ)) W
        return SpectralFunction(
            operator.base, _Sqrt(operator.fn), _eigen=operator.eigen
        )
    if isinstance(operator, KroneckerSum):
        # (A ⊕ B)^½
        return KroneckerSumSqrt(operator.A, operator.B)
    return None


def _as_operator(A: lx.AbstractLinearOperator | ArrayLike) -> lx.AbstractLinearOperator:
    """Wrap a dense matrix as a `lineax.MatrixLinearOperator`.

    Args:
        A: An operator, returned unchanged, or a dense matrix.

    Returns:
        The operator.
    """
    if isinstance(A, lx.AbstractLinearOperator):
        return A
    return lx.MatrixLinearOperator(jnp.asarray(A))


def _scaled(
    operator: lx.AbstractLinearOperator, scale: Float[Array, ""]
) -> lx.AbstractLinearOperator:
    """Scaled structure τ R, keeping a sparse, banded or dense structure.

    Args:
        operator: The structure R.
        scale: The precision scale τ.

    Returns:
        τ R, tagged symmetric positive semidefinite.
    """
    inner = _unwrap(operator)
    if isinstance(inner, SparseOperator):
        # τ R on R's pattern
        return SparseOperator(scale * inner.values, inner.pattern, tags=_PSD)
    if isinstance(inner, BlockTriDiag):
        # τ R, block by block
        return BlockTriDiag(
            scale * inner.diagonal,
            scale * inner.sub_diagonal,
            symmetric=inner.symmetric,
            tags=_PSD,
        )
    if isinstance(inner, lx.MatrixLinearOperator):
        # τ R, dense
        return lx.MatrixLinearOperator(scale * inner.matrix, _PSD)
    # τ R, lazily
    return lx.TaggedLinearOperator(scale * operator, _PSD)


# ---------------------------------------------------------------------------
# GaussianMRF
# ---------------------------------------------------------------------------


class GaussianMRF(dist.Distribution):
    """Gaussian Markov random field x ~ 𝒩(μ, Λ⁻¹) with a structured precision Λ.

        log p(x) = ½ log det Λ − ½ (x − μ)ᵀ Λ (x − μ) − (D_x / 2) log 2π

    ``log_prob``, ``sample`` and ``marginal_variances`` take the cheapest
    exact path the structure of Λ allows:

    - `SparseOperator`: sparse Cholesky P Λ Pᵀ = L Lᵀ and draws
      μ + Pᵀ L⁻ᵀ z; Takahashi for the marginal variances.
    - `BlockTriDiag`: block Cholesky and the block selected inverse.
    - `Kronecker`, `KroneckerSum`, `SpectralFunction`: the factored
      eigenbasis Λ = W⁻¹ diag(γ) W, draws μ + W⁻¹ diag(γ^(−½)) W z, and the
      factor eigenvalues for the variances.
    - dense: Cholesky Λ = L_Λ L_Λᵀ.
    - anything else, with ``precision_factors``: perturbation-optimisation,
      one solve of Λ x = ∑_k F_kᵀ z_k per draw; ``solver`` and
      ``logdet_strategy`` for log det Λ.

    A `SparseOperator` defaults to `SparseCholeskySolver` for every solve and
    log-determinant; pass ``precision_factors`` (and no sparse Cholesky
    solver) to sample matrix-free by CG instead (``solver``, if given, does
    that solve). Graph precisions come factored (Bᵀ diag(w) B from an
    incidence matrix B, proper CAR, Leroux), so a 10⁶-node field samples
    with no factorisation at all.

    Requires the ``numpyro`` optional extra. ``loc`` is a single field
    (``batch_shape == ()``); ``log_prob`` takes any leading sample axes.

    Args:
        loc: Mean μ of the D_x nodes.
        precision: Precision Λ, a symmetric positive-definite operator of
            size D_x.
        precision_factors: Operators F_k with Λ = ∑_k F_kᵀ F_k, for the
            perturbation-optimisation sampler.
        solver: Solve (and log-determinant) strategy, e.g.
            `SparseCholeskySolver` or `CGSolver`.
        logdet_strategy: Log-determinant strategy; takes precedence over
            ``solver`` for log det Λ (e.g. `SLQLogdet` on a huge field).
        log_det_precision: A known log det Λ, which overrides both (callers
            with a closed form, such as a proper CAR, skip a stochastic
            estimate inside an MCMC log-density).
        validate_args: Whether to validate input arguments.

    Raises:
        ValueError: If ``loc`` does not have shape ``(D_x,)``.

    Examples:
        A proper field on the path graph 0 − 1 − 2 − 3 − 4, with Λ = R + I
        for the graph Laplacian R, takes sparse Cholesky throughout:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx as gx
        >>> n = 5
        >>> precision = gx.SparseOperator.from_coo(
        ...     np.r_[np.arange(n), np.arange(1, n)],
        ...     np.r_[np.arange(n), np.arange(n - 1)],
        ...     jnp.r_[jnp.array([2.0, 3.0, 3.0, 3.0, 2.0]), -jnp.ones(n - 1)],
        ...     (n, n),
        ...     symmetric=True,
        ... )
        >>> prior = gx.GaussianMRF(jnp.zeros(n), precision)
        >>> x = prior.sample(jax.random.PRNGKey(0), (3,))
        >>> x.shape, prior.log_prob(x).shape
        ((3, 5), (3,))
        >>> dense_variances = jnp.diag(jnp.linalg.inv(precision.as_matrix()))
        >>> bool(jnp.allclose(prior.marginal_variances(), dense_variances, atol=1e-5))
        True

        Observing nodes 0 and 3 with noise standard deviation 0.1 keeps the
        posterior precision sparse:

        >>> obs_operator = gx.SparseOperator.from_coo(
        ...     [0, 1], [0, 3], jnp.ones(2), (2, n)
        ... )
        >>> obs = jnp.array([1.0, -1.0])
        >>> posterior = prior.condition_on_observations(obs_operator, 100.0, obs)
        >>> type(posterior.precision).__name__
        'SparseOperator'

        A sum-to-zero field, by kriging:

        >>> zero_sum = prior.condition_on_constraints(jnp.ones((1, n)), jnp.zeros(1))
        >>> draw = zero_sum.sample(jax.random.PRNGKey(1))
        >>> bool(jnp.abs(draw.sum()) < 1e-4)
        True

    References:
        Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory
        and Applications. Chapman & Hall/CRC.
    """

    arg_constraints = {"loc": dist.constraints.real_vector}  # noqa: RUF012
    support = dist.constraints.real_vector
    reparametrized_params = ["loc"]  # noqa: RUF012
    pytree_data_fields = (
        "loc",
        "precision",
        "precision_factors",
        "solver",
        "logdet_strategy",
        "log_det_precision",
    )

    def __init__(
        self,
        loc: Float[ArrayLike, " N"],
        precision: lx.AbstractLinearOperator,
        precision_factors: tuple[lx.AbstractLinearOperator, ...] | None = None,
        solver: AbstractSolverStrategy | None = None,
        logdet_strategy: AbstractLogdetStrategy | None = None,
        log_det_precision: Float[ArrayLike, ""] | None = None,
        *,
        validate_args: bool | None = None,
    ) -> None:
        loc = jnp.asarray(loc)
        n = precision.in_size()
        if loc.shape != (n,):
            raise ValueError(
                f"loc must have shape ({n},) to match the precision, got {loc.shape}."
            )
        self.loc = loc
        self.precision = precision
        self.precision_factors = (
            None if precision_factors is None else tuple(precision_factors)
        )
        self.solver = solver
        self.logdet_strategy = logdet_strategy
        self.log_det_precision = (
            None if log_det_precision is None else jnp.asarray(log_det_precision)
        )
        super().__init__(batch_shape=(), event_shape=(n,), validate_args=validate_args)

    # -- solves ---------------------------------------------------------------

    def _log_det(self) -> Float[Array, ""]:
        """Log-determinant log det Λ of the precision.

        Taken from ``log_det_precision`` if given, else from
        ``logdet_strategy``, else from the structural `logdet`.

        Returns:
            The scalar log det Λ.
        """
        if self.log_det_precision is not None:
            return self.log_det_precision
        if self.logdet_strategy is not None:
            # log det Λ, by the given strategy
            return self.logdet_strategy.logdet(self.precision)
        strategy = _solve_strategy(self.precision, self.solver)
        # log det Λ, by structural dispatch
        return dispatch_logdet(self.precision, strategy)

    # -- density --------------------------------------------------------------

    @validate_sample
    def log_prob(self, value: Float[Array, "*sample N"]) -> Float[Array, "*sample"]:
        """Log-density log 𝒩(x; μ, Λ⁻¹) of the field.

            log p(x) = ½ (log det Λ − (x − μ)ᵀ Λ (x − μ) − D_x log 2π)

        Args:
            value: Points x, shape ``(..., N)``.

        Returns:
            Log-densities, shape ``(...)``.
        """
        event_size = self.event_shape[0]
        # r = x − μ
        residual = value - self.loc
        # rᵀ Λ r
        mahalanobis = _quadratic(self.precision, residual)
        # log p(x) = ½ (log det Λ − rᵀ Λ r − D_x log 2π)
        return 0.5 * (self._log_det() - mahalanobis - event_size * _LOG_2PI)

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample N"]:
        """Exact draws x = μ + f(z) through the structural sampling dispatch.

            x = μ + f(z),   z ~ 𝒩(0, I),   f(z) ~ 𝒩(0, Λ⁻¹)

        Algorithm:
            1. Pick the draw map f from the structure of Λ (sparse or dense
               Cholesky, factored eigenbasis, or perturbation-optimisation),
               factoring Λ once.
            2. Draw all the noise z in one call and map every row through f.
            3. Add the mean μ.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            sample_shape: Leading sample shape.

        Returns:
            Draws, shape ``sample_shape + (N,)``.
        """
        noise_size, draw = _precision_sampler(
            self.precision, self.solver, self.precision_factors
        )
        # f(z) ~ 𝒩(0, Λ⁻¹)
        draws = _draw_samples(key, sample_shape, noise_size, draw, self.loc.dtype)
        # x = μ + f(z)
        return self.loc + draws

    def marginal_variances(self) -> Float[Array, " N"]:
        """Marginal variances diag(Λ⁻¹) through `diag_inv`'s structural dispatch.

            𝕍[x_n] = [Λ⁻¹]_nn

        Returns:
            Marginal variances, shape ``(N,)``.
        """
        strategy = _solve_strategy(self.precision, self.solver)
        # diag(Λ⁻¹)
        return diag_inv(self.precision, solver=strategy)

    @lazy_property
    def mean(self) -> Float[Array, " N"]:
        """The mean μ."""
        return self.loc

    @lazy_property
    def variance(self) -> Float[Array, " N"]:
        """The marginal variances diag(Λ⁻¹)."""
        return self.marginal_variances()

    # -- conditioning ---------------------------------------------------------

    def condition_on_observations(
        self,
        A: lx.AbstractLinearOperator | Float[ArrayLike, "M N"],
        noise_precision: Float[ArrayLike, ""] | Float[ArrayLike, " M"],
        y: Float[ArrayLike, " M"],
    ) -> GaussianMRF:
        """Posterior field given linear observations y = H x + ε, ε ~ 𝒩(0, Λ_ε⁻¹).

            Λ_x|y = Λ + Hᵀ Λ_ε H
            μ_x|y = μ + Λ_x|y⁻¹ Hᵀ Λ_ε (y − H μ)

        The posterior precision keeps what structure it can:

        - a `SparseOperator` prior observed through a `SparseOperator` H
          stays sparse, on the union of the two patterns
          (`SparseOperator.congruence`, planned once per pair of patterns),
          so it keeps the sparse Cholesky path;
        - a dense prior stays dense;
        - anything else (a grid `SpectralFunction`, a `KroneckerSum`, a
          `BlockTriDiag` observed at scattered points) becomes the
          matrix-free sum Λ + (Λ_ε^½ H)ᵀ (Λ_ε^½ H): solves by `solve`
          (CG when large), and draws by perturbation-optimisation with the
          factors (Λ^½, Λ_ε^½ H), Λ^½ the symmetric root of a
          `SpectralFunction` or `KroneckerSum` prior.

        ``precision_factors`` gain Λ_ε^½ H; a known ``log_det_precision`` is
        dropped.

        Algorithm:
            1. Broadcast the diagonal Λ_ε to one entry per observation and
               whiten the observation operator, Λ_ε^½ H.
            2. Build Λ_x|y = Λ + Hᵀ Λ_ε H: sparse on the union pattern, dense,
               or as a matrix-free sum with its perturbation factors.
            3. Drop a sparse Cholesky solver if Λ_x|y is not sparse.
            4. Solve Λ_x|y⁻¹ Hᵀ Λ_ε (y − H μ) once and shift the mean.

        Args:
            A: Observation operator H, shape ``(M, N)`` (array or operator).
            noise_precision: Diagonal of the observation-noise precision Λ_ε,
                a scalar or shape ``(M,)``.
            y: Observations y, shape ``(M,)``.

        Returns:
            The posterior `GaussianMRF` 𝒩(μ_x|y, Λ_x|y⁻¹), with the same
            solver strategies (a `SparseCholeskySolver` is dropped if the
            posterior is not sparse).
        """
        obs_operator = _as_operator(A)
        obs_size = obs_operator.out_size()
        dtype = self.loc.dtype
        noise_precision_value = jnp.asarray(noise_precision, dtype=dtype)
        # diag(Λ_ε), one entry per observation
        obs_noise_precision = jnp.broadcast_to(noise_precision_value, (obs_size,))
        # Λ_ε^½
        noise_root = jnp.sqrt(obs_noise_precision)
        # Λ_ε^½ H
        whitened_obs_operator = lx.DiagonalLinearOperator(noise_root) @ obs_operator
        prior_precision = _unwrap(self.precision)
        factors = self.precision_factors
        if isinstance(prior_precision, SparseOperator) and isinstance(
            obs_operator, SparseOperator
        ):
            # Hᵀ Λ_ε H, sparse
            gauss_newton_term = prior_precision.congruence(
                obs_operator, obs_noise_precision
            )
            # Λ_x|y = Λ + Hᵀ Λ_ε H, on the union of the two patterns
            posterior_precision: lx.AbstractLinearOperator = prior_precision.union(
                gauss_newton_term, tags=_PSD
            )
        elif isinstance(prior_precision, lx.MatrixLinearOperator):
            # H, densified to meet the dense prior
            obs_matrix = obs_operator.as_matrix()
            # Λ_ε H
            weighted_obs_matrix = einx.multiply(
                "obs state, obs -> obs state", obs_matrix, obs_noise_precision
            )
            # Hᵀ Λ_ε H
            gauss_newton_term = einsum(
                weighted_obs_matrix, obs_matrix, "obs row, obs col -> row col"
            )
            # Λ_x|y = Λ + Hᵀ Λ_ε H
            posterior_precision = lx.MatrixLinearOperator(
                prior_precision.matrix + gauss_newton_term, _PSD
            )
        else:
            # Hᵀ Λ_ε H = (Λ_ε^½ H)ᵀ (Λ_ε^½ H), matrix-free
            gauss_newton_operator = whitened_obs_operator.T @ whitened_obs_operator
            # Λ_x|y = Λ + Hᵀ Λ_ε H, matrix-free
            posterior_precision = lx.TaggedLinearOperator(
                lx.AddLinearOperator(self.precision, gauss_newton_operator), _PSD
            )
            if factors is None:
                # Λ^½, so that Λ = (Λ^½)ᵀ Λ^½
                root = _symmetric_root(prior_precision)
                factors = None if root is None else (root,)
        if factors is not None:
            # Λ_x|y = ∑_k F_kᵀ F_k + (Λ_ε^½ H)ᵀ (Λ_ε^½ H)
            factors = (*factors, whitened_obs_operator)
        solver = self.solver
        if isinstance(solver, SparseCholeskySolver) and not isinstance(
            posterior_precision, SparseOperator
        ):
            solver = None
        observations = jnp.asarray(y, dtype=dtype)
        # H μ
        predicted_obs = obs_operator.mv(self.loc)
        # r = y − H μ
        residual = observations - predicted_obs
        # Λ_ε r
        weighted_residual = obs_noise_precision * residual
        # Hᵀ Λ_ε r
        rhs = obs_operator.transpose().mv(weighted_residual)
        posterior_solve = _solve_fn(posterior_precision, solver)
        # Λ_x|y⁻¹ Hᵀ Λ_ε r
        mean_shift = posterior_solve(rhs)
        # μ_x|y = μ + Λ_x|y⁻¹ Hᵀ Λ_ε r
        posterior_mean = self.loc + mean_shift
        return GaussianMRF(
            posterior_mean,
            posterior_precision,
            factors,
            solver=solver,
            logdet_strategy=self.logdet_strategy,
        )

    def condition_on_constraints(
        self,
        A_c: Float[ArrayLike, "c N"] | Float[ArrayLike, " N"],
        e: Float[ArrayLike, " c"] | Float[ArrayLike, ""],
    ) -> ConstrainedGMRF:
        """Condition on hard linear constraints A_c x = e, by kriging.

            x ∣ A_c x = e,   x ~ 𝒩(μ, Λ⁻¹)

        Args:
            A_c: Constraint matrix A_c, shape ``(c, N)`` (or ``(N,)`` for one
                constraint).
            e: Constraint values e, shape ``(c,)``.

        Returns:
            The `ConstrainedGMRF` of x ∣ A_c x = e.
        """
        return ConstrainedGMRF(self, A_c, e)


# ---------------------------------------------------------------------------
# ConstrainedGMRF
# ---------------------------------------------------------------------------


class ConstrainedGMRF(dist.Distribution):
    """A `GaussianMRF` conditioned on hard linear constraints A_c x = e.

        x ↦ x − W S⁻¹ (A_c x − e),   W = Λ⁻¹ A_cᵀ,   S = A_c W

    Conditioning by kriging (Rue & Held, 2005, §2.3.3): W takes D_c solves
    with one factorisation, and an unconstrained draw x maps to the exact
    constrained draw above, so A_c x = e to machine precision. The marginal
    variances are diag(Λ⁻¹) − diag(W S⁻¹ Wᵀ), and the density (Rue & Held,
    eq. 2.30) is

        log p(x ∣ A_c x = e) = log p(x) − log 𝒩(e; A_c μ, S)
                               − ½ log det(A_c A_cᵀ),

    a density on the affine subspace {x : A_c x = e} (with respect to its
    Lebesgue measure), evaluated at points assumed to satisfy the
    constraint.

    Built by `GaussianMRF.condition_on_constraints`. Requires ``numpyro``.

    Args:
        base: The unconstrained field 𝒩(μ, Λ⁻¹).
        constraint_matrix: Constraint matrix A_c, shape ``(c, N)`` (or
            ``(N,)``), of full row rank.
        constraint_value: Constraint values e, shape ``(c,)``.
        validate_args: Whether to validate input arguments.

    Raises:
        ValueError: If A_c and e do not have shapes ``(c, N)`` and ``(c,)``.

    Examples:
        A sum-to-one constraint on four iid nodes of precision 2:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx as gx
        >>> precision = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(4), lx.positive_semidefinite_tag
        ... )
        >>> prior = gx.GaussianMRF(jnp.zeros(4), precision)
        >>> field = prior.condition_on_constraints(jnp.ones(4), 1.0)
        >>> x = field.sample(jax.random.PRNGKey(0), (10,))
        >>> bool(jnp.allclose(x.sum(-1), 1.0, atol=1e-5))
        True

        The variances lose the share of the constraint, 0.5 − 0.5 / 4:

        >>> bool(jnp.allclose(field.marginal_variances(), 0.375))
        True

    References:
        Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory
        and Applications. Chapman & Hall/CRC.
    """

    arg_constraints = {}  # noqa: RUF012
    support = dist.constraints.real_vector
    pytree_data_fields = ("base", "constraint_matrix", "constraint_value")

    def __init__(
        self,
        base: GaussianMRF,
        constraint_matrix: Float[ArrayLike, "c N"] | Float[ArrayLike, " N"],
        constraint_value: Float[ArrayLike, " c"] | Float[ArrayLike, ""],
        *,
        validate_args: bool | None = None,
    ) -> None:
        dtype = base.loc.dtype
        A = jnp.atleast_2d(jnp.asarray(constraint_matrix, dtype=dtype))
        e = jnp.atleast_1d(jnp.asarray(constraint_value, dtype=dtype))
        n = base.event_shape[0]
        if A.shape[1] != n or e.shape != (A.shape[0],):
            raise ValueError(
                f"constraint_matrix must have shape (c, {n}) and constraint_value "
                f"shape (c,), got {A.shape} and {e.shape}."
            )
        self.base = base
        self.constraint_matrix = A
        self.constraint_value = e
        super().__init__(batch_shape=(), event_shape=(n,), validate_args=validate_args)

    def _kriging(self) -> tuple[Float[Array, "c N"], Float[Array, "c N"], Array]:
        """Kriging terms Wᵀ, S⁻¹ Wᵀ and the Cholesky factor L_S of S.

            W = Λ⁻¹ A_cᵀ,   S = A_c W = L_S L_Sᵀ

        Returns:
            Tuple ``(solved_constraints, kriging_rows, constraint_cholesky)``
            holding Wᵀ and S⁻¹ Wᵀ (one row per constraint) and L_S.
        """
        constraint_matrix = self.constraint_matrix
        precision_solve = _solve_fn(self.base.precision, self.base.solver)
        # Wᵀ = (Λ⁻¹ A_cᵀ)ᵀ, one solve per constraint
        solved_constraints = jax.vmap(precision_solve)(constraint_matrix)
        # S = A_c Λ⁻¹ A_cᵀ
        constraint_covariance = einsum(
            constraint_matrix, solved_constraints, "row state, col state -> row col"
        )
        # S = L_S L_Sᵀ
        constraint_cholesky = jnp.linalg.cholesky(constraint_covariance)
        # S⁻¹ Wᵀ
        kriging_rows = jsl.cho_solve((constraint_cholesky, True), solved_constraints)
        return solved_constraints, kriging_rows, constraint_cholesky

    def _correct(self, points: Float[Array, "*sample N"], kriging_rows: Array) -> Array:
        """Kriging correction x ↦ x − W S⁻¹ (A_c x − e) onto A_c x = e.

        Args:
            points: Points x, shape ``(..., N)``.
            kriging_rows: S⁻¹ Wᵀ, from ``_kriging``.

        Returns:
            The corrected points, shape ``(..., N)``.
        """
        flat_points = _flat(points)
        # A_c x
        constraint_values = einsum(
            flat_points,
            self.constraint_matrix,
            "draw state, constraint state -> draw constraint",
        )
        # A_c x − e
        excess = constraint_values - self.constraint_value
        # W S⁻¹ (A_c x − e)
        correction = einsum(
            excess, kriging_rows, "draw constraint, constraint state -> draw state"
        )
        # x − W S⁻¹ (A_c x − e)
        corrected = flat_points - correction
        return _reshape_samples(corrected, points.shape[:-1])

    @validate_sample
    def log_prob(self, value: Float[Array, "*sample N"]) -> Float[Array, "*sample"]:
        """Constrained log-density on A_c x = e (Rue & Held, eq. 2.30).

            log p(x ∣ A_c x = e) = log p(x) − log 𝒩(e; A_c μ, S)
                                   − ½ log det(A_c A_cᵀ)

        Algorithm:
            1. Factor S = A_c Λ⁻¹ A_cᵀ = L_S L_Sᵀ.
            2. Evaluate log 𝒩(e; A_c μ, S) from the whitened residual
               L_S⁻¹ (e − A_c μ) and log det S = 2 ∑ log diag(L_S).
            3. Subtract it and ½ log det(A_c A_cᵀ) from the base log-density.

        Args:
            value: Points x satisfying the constraint, shape ``(..., N)``.

        Returns:
            Log-densities, shape ``(...)``.
        """
        constraint_matrix = self.constraint_matrix
        constraint_value = self.constraint_value
        _, _, constraint_cholesky = self._kriging()
        num_constraints = constraint_matrix.shape[0]
        # A_c μ
        constraint_mean = einsum(
            constraint_matrix, self.base.loc, "constraint state, state -> constraint"
        )
        # e − A_c μ
        residual = constraint_value - constraint_mean
        # L_S⁻¹ (e − A_c μ)
        whitened_residual = jsl.solve_triangular(
            constraint_cholesky, residual, lower=True
        )
        # diag(L_S)
        cholesky_diagonal = jnp.diagonal(constraint_cholesky)
        # log det S = 2 ∑ log diag(L_S), a full reduction
        log_det_constraint = 2.0 * jnp.sum(jnp.log(cholesky_diagonal))
        # (e − A_c μ)ᵀ S⁻¹ (e − A_c μ)
        mahalanobis = einsum(
            whitened_residual, whitened_residual, "constraint, constraint ->"
        )
        # log 𝒩(e; A_c μ, S) = −½ (D_c log 2π + log det S + Mahalanobis term)
        log_constraint = -0.5 * (
            num_constraints * _LOG_2PI + log_det_constraint + mahalanobis
        )
        # A_c A_cᵀ
        constraint_gram = einsum(
            constraint_matrix, constraint_matrix, "row state, col state -> row col"
        )
        # log det(A_c A_cᵀ)
        log_det_gram = jnp.linalg.slogdet(constraint_gram)[1]
        # log p(x)
        base_log_prob = self.base.log_prob(value)
        # log p(x ∣ A_c x = e)
        return base_log_prob - log_constraint - 0.5 * log_det_gram

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample N"]:
        """Base draws, kriged onto A_c x = e.

            x ~ 𝒩(μ, Λ⁻¹),   x ↦ x − W S⁻¹ (A_c x − e)

        Algorithm:
            1. Draw x from the unconstrained base field.
            2. Apply the kriging correction, which needs W and S⁻¹ Wᵀ only.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            sample_shape: Leading sample shape.

        Returns:
            Draws, shape ``sample_shape + (N,)``.
        """
        _, kriging_rows, _ = self._kriging()
        # x ~ 𝒩(μ, Λ⁻¹)
        base_draws = self.base.sample(key, sample_shape)
        # x − W S⁻¹ (A_c x − e)
        return self._correct(base_draws, kriging_rows)

    def marginal_variances(self) -> Float[Array, " N"]:
        """Constrained marginal variances diag(Λ⁻¹) − diag(W S⁻¹ Wᵀ).

        Returns:
            Constrained marginal variances, shape ``(N,)``.
        """
        solved_constraints, kriging_rows, _ = self._kriging()
        # W ⊙ (S⁻¹ Wᵀ)ᵀ, entry by entry
        products = solved_constraints * kriging_rows
        # diag(W S⁻¹ Wᵀ)
        correction = reduce(products, "constraint state -> state", "sum")
        # diag(Λ⁻¹) − diag(W S⁻¹ Wᵀ)
        return self.base.marginal_variances() - correction

    @lazy_property
    def mean(self) -> Float[Array, " N"]:
        """The kriged mean μ − W S⁻¹ (A_c μ − e)."""
        _, kriging_rows, _ = self._kriging()
        # μ − W S⁻¹ (A_c μ − e)
        return self._correct(self.base.loc, kriging_rows)

    @lazy_property
    def variance(self) -> Float[Array, " N"]:
        """The constrained marginal variances."""
        return self.marginal_variances()


# ---------------------------------------------------------------------------
# IntrinsicGMRF
# ---------------------------------------------------------------------------


class IntrinsicGMRF(dist.Distribution):
    """Improper Gaussian field x ~ 𝒩(μ, (τ R)⁺) on the complement of ker R.

        log p(x) = ((D_x − D_c) / 2) log τ − (τ / 2) (x − μ)ᵀ R (x − μ)
                   [+ ½ log det₊ R − ((D_x − D_c) / 2) log 2π]

    R is a positive-semidefinite structure matrix (RW1, RW2, Besag / ICAR,
    a grid Laplacian) whose null space ker R is spanned by the D_c columns
    of ``null_space``, V, and τ is the precision scale. The bracket is added
    only with ``include_normalizer``: it is τ-free, so an MCMC target can
    skip it, and `pseudo_logdet` computes it per call. The τ term is always
    included, which is what makes τ identifiable under a hyperprior.
    D_x − D_c must be rank(R). The bracket is constant only when R itself
    does not depend on the hyperparameters: for BYM2, whose structure holds
    (τ, φ), use `BYM2GMRF`.

    ``constraint`` chooses what happens on ker R:

    - ``"hard"`` (default, what INLA needs): the field lives on
      {x : Vᵀ x = 0}. The log-density above (with the normaliser) is then
      exactly the constrained density on that subspace. Samples are kriged
      onto it: R-INLA's recipe of factoring τ R + β I (with the ridge
      β = √eps times the mean diagonal) and kriging against Vᵀ x = 0
      reduces, since (τ R + β I)⁻¹ V = V / β, to the orthogonal projection
      P⊥ x = x − V (Vᵀ V)⁻¹ Vᵀ x, which is what is applied (exactly, with no
      extra solves). Marginal variances are kriging-corrected
      (`ConstrainedGMRF`) on the shifted factor, or exact from the
      pseudo-inverse on a Kronecker-structured R.
    - ``"soft"`` (for NUTS, which cannot sample a hard constraint): adds a
      tight Gaussian Vᵀ x ~ 𝒩(0, s² I), s = ``soft_constraint_scale``, to
      the density and to the samples.
    - ``"none"``: only μ plus a draw from 𝒩(0, (τ R)⁺); the density is
      improper and invariant along ker R.

    The draw from 𝒩(0, (τ R)⁺) dispatches like `GaussianMRF`: Cholesky of
    τ R + β I (sparse, banded or dense), the pseudo-inverse root in the
    factored eigenbasis (exact, no shift), or perturbation-optimisation
    with ``precision_factors`` (R = ∑_k F_kᵀ F_k, so the right-hand side
    lies in range(R) and CG converges on the singular system); each is then
    projected orthogonally to ``null_space``.

    **Odd-n RW2.** `rw2_structure` with odd n returns n + 1 rows, the last a
    decoupled unit-precision node. Build the field on all D_x = n + 1 nodes
    with ``null_space`` zero on the padding row (D_c = 2): then
    D_x − D_c = n − 1 = rank(R), and the padding node is an independent
    𝒩(0, 1/τ) coordinate whose own ½ log τ is the one extra in
    ((D_x − D_c) / 2) log τ. Sample it with the rest and drop it
    (``x[:n]``).

    Requires the ``numpyro`` optional extra.

    Args:
        loc: Mean μ of the D_x nodes.
        precision_scale: Precision scale τ > 0.
        structure: Structure matrix R, symmetric positive semidefinite, of
            size D_x.
        null_space: Basis V of ker R, shape ``(N, c)`` (or ``(N,)``); for a
            graph, one column per connected component (kernellib's
            ``graph_null_space``).
        precision_factors: Factors F_k with R = ∑_k F_kᵀ F_k (e.g. the
            weighted incidence matrix), for the CG sampler.
        constraint: ``"hard"``, ``"soft"`` or ``"none"``.
        soft_constraint_scale: Scale s of the soft constraint, for
            ``"soft"``.
        include_normalizer: Add the τ-free constant (exact for an
            orthonormal ``null_space`` under ``"soft"``).
        validate_args: Whether to validate input arguments.

    Raises:
        ValueError: For an unknown ``constraint`` or mismatched shapes.

    Examples:
        An RW1 trend with a hard sum-to-zero constraint:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import gaussx as gx
        >>> n = 50
        >>> structure = gx.rw1_structure(n)
        >>> null_space = jnp.ones((n, 1)) / jnp.sqrt(n)
        >>> trend = gx.IntrinsicGMRF(jnp.zeros(n), 4.0, structure, null_space)
        >>> u = trend.sample(jax.random.PRNGKey(0), (5,))
        >>> bool(jnp.allclose(u.sum(-1), 0.0, atol=1e-4))
        True
        >>> trend.log_prob(u).shape
        (5,)

        RW2 with odd n: one padding node, and the null space {1, t} is zero
        on it:

        >>> structure2 = gx.rw2_structure(7)
        >>> t = jnp.arange(7.0)
        >>> null_space2 = (
        ...     jnp.zeros((8, 2)).at[:7, 0].set(1.0).at[:7, 1].set(t - t.mean())
        ... )
        >>> smooth = gx.IntrinsicGMRF(jnp.zeros(8), 10.0, structure2, null_space2)
        >>> smooth.sample(jax.random.PRNGKey(1))[:7].shape
        (7,)

        An unknown constraint is rejected:

        >>> gx.IntrinsicGMRF(
        ...     jnp.zeros(n), 4.0, structure, null_space, constraint="firm"
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: constraint must be 'none', 'soft' or 'hard', got 'firm'.

    References:
        Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory
        and Applications. Chapman & Hall/CRC.
    """

    arg_constraints = {"loc": dist.constraints.real_vector}  # noqa: RUF012
    support = dist.constraints.real_vector
    reparametrized_params = ["loc"]  # noqa: RUF012
    pytree_data_fields = (
        "loc",
        "precision_scale",
        "structure",
        "null_space",
        "precision_factors",
    )
    pytree_aux_fields = ("constraint", "soft_constraint_scale", "include_normalizer")

    def __init__(
        self,
        loc: Float[ArrayLike, " N"],
        precision_scale: Float[ArrayLike, ""],
        structure: lx.AbstractLinearOperator,
        null_space: Float[ArrayLike, "N c"] | Float[ArrayLike, " N"],
        precision_factors: tuple[lx.AbstractLinearOperator, ...] | None = None,
        constraint: Literal["none", "soft", "hard"] = "hard",
        soft_constraint_scale: float = 1e-3,
        include_normalizer: bool = False,
        *,
        validate_args: bool | None = None,
    ) -> None:
        if constraint not in ("none", "soft", "hard"):
            raise ValueError(
                f"constraint must be 'none', 'soft' or 'hard', got {constraint!r}."
            )
        loc = jnp.asarray(loc)
        n = structure.in_size()
        if loc.shape != (n,):
            raise ValueError(
                f"loc must have shape ({n},) to match the structure, got {loc.shape}."
            )
        null_space = jnp.asarray(null_space, dtype=loc.dtype)
        if null_space.ndim == 1:
            null_space = rearrange(null_space, "state -> state 1")
        if null_space.shape[0] != n:
            raise ValueError(
                f"null_space must have {n} rows, got shape {null_space.shape}."
            )
        self.loc = loc
        self.precision_scale = jnp.asarray(precision_scale, dtype=loc.dtype)
        self.structure = structure
        self.null_space = null_space
        self.precision_factors = (
            None if precision_factors is None else tuple(precision_factors)
        )
        self.constraint = constraint
        self.soft_constraint_scale = soft_constraint_scale
        self.include_normalizer = include_normalizer
        super().__init__(batch_shape=(), event_shape=(n,), validate_args=validate_args)

    # -- null space -----------------------------------------------------------

    def _null_dual(self) -> Float[Array, "N c"]:
        """Dual basis V (Vᵀ V)⁻¹ of the null space.

            P⊥ x = x − V (Vᵀ V)⁻¹ Vᵀ x

        Returns:
            V (Vᵀ V)⁻¹, shape ``(N, c)``.
        """
        null_basis = self.null_space
        # Vᵀ V
        null_gram = einsum(null_basis, null_basis, "state row, state col -> row col")
        # Vᵀ
        null_basis_transpose = rearrange(null_basis, "state null -> null state")
        # (Vᵀ V)⁻¹ Vᵀ, a small dense solve
        solved = jnp.linalg.solve(null_gram, null_basis_transpose)
        # V (Vᵀ V)⁻¹
        return rearrange(solved, "null state -> state null")

    def _project(self, x: Float[Array, "*sample N"]) -> Float[Array, "*sample N"]:
        """Orthogonal projection P⊥ x onto the complement of the null space.

            P⊥ x = x − V (Vᵀ V)⁻¹ Vᵀ x

        Args:
            x: Points x, shape ``(..., N)``.

        Returns:
            P⊥ x, shape ``(..., N)``.
        """
        # Vᵀ x
        null_coeffs = einsum(x, self.null_space, "... state, state null -> ... null")
        # V (Vᵀ V)⁻¹ Vᵀ x
        null_component = einsum(
            null_coeffs, self._null_dual(), "... null, state null -> ... state"
        )
        # P⊥ x = x − V (Vᵀ V)⁻¹ Vᵀ x
        return x - null_component

    def _shifted(self) -> lx.AbstractLinearOperator:
        """Ridge-shifted precision τ R + β I, as R-INLA builds it.

            β = √eps · mean(diag(τ R))

        Returns:
            τ R + β I, keeping a sparse or block-tridiagonal structure.
        """
        # τ R
        scaled_structure = _scaled(self.structure, self.precision_scale)
        # diag(τ R)
        diagonal = diag(scaled_structure)
        # β = √eps · mean(diag(τ R))
        ridge = float(jnp.finfo(diagonal.dtype).eps) ** 0.5 * jnp.mean(diagonal)
        # τ R + β I
        return _add_ridge(scaled_structure, ridge)

    def _eigen(self) -> FactoredEigen | None:
        """Factored eigendecomposition of τ R, for a Kronecker structure.

            R = W⁻¹ diag(γ) W,   τ R = W⁻¹ diag(τ γ) W

        Returns:
            The `FactoredEigen` of τ R, or ``None`` if R has no factored
            eigenbasis.
        """
        operator = _unwrap(self.structure)
        if not isinstance(operator, _EIGEN):
            return None
        # R = W⁻¹ diag(γ) W
        eigen = factored_eigen(operator)
        if eigen is None:
            return None
        # τ γ
        scaled_eigenvalues = self.precision_scale * eigen.eigenvalues
        return FactoredEigen(eigen.bases, scaled_eigenvalues)

    def _sampler(self) -> tuple[int, _Draw]:
        """Draw map f with P⊥ f(z) ~ 𝒩(0, (τ R)⁺), from the structure of R.

            P⊥ f(z) ~ 𝒩(0, (τ R)⁺)   for   z ~ 𝒩(0, I)

        Algorithm:
            1. A factored eigenbasis: the pseudo-inverse root (τ R)^(+½) z.
            2. Factors F_k on a structure Cholesky does not suit:
               perturbation-optimisation on τ R = ∑_k (√τ F_k)ᵀ (√τ F_k).
            3. Otherwise: the `GaussianMRF` dispatch on τ R + β I.

        Returns:
            Tuple ``(noise_size, draw)``: the length of z and the map f.
        """
        event_size = self.event_shape[0]
        eigen = self._eigen()
        if eigen is not None:
            # (τ R)^(+½) z
            return event_size, lambda noise: _eigen_draw(eigen, noise, pinv=True)
        factors = self.precision_factors
        if factors is not None and not _is_factorable(_unwrap(self.structure)):
            # √τ
            root_scale = jnp.sqrt(self.precision_scale)
            # √τ F_k, so that τ R = ∑_k (√τ F_k)ᵀ (√τ F_k)
            scaled_factors = tuple(root_scale * factor for factor in factors)
            # τ R
            scaled_structure = _scaled(self.structure, self.precision_scale)
            # (τ R)⁻¹ ∑_k √τ F_kᵀ z_k, by CG on the singular system
            return _perturbation_draw(scaled_structure, scaled_factors, None)
        # (τ R + β I)^(−½) z, by its structural factor
        return _precision_sampler(self._shifted(), None, None)

    # -- density --------------------------------------------------------------

    def _rank(self) -> int:
        """Rank D_x − D_c of the structure R.

        Returns:
            The event size minus the number of null-space columns.
        """
        return self.event_shape[0] - self.null_space.shape[1]

    def _half_log_pdet(self) -> Float[Array, ""]:
        """Half log pseudo-determinant ½ log det₊ R of the structure.

        Taken from the eigenvalues of a Kronecker sum or a diagonal R, else
        through the null space.

        Returns:
            The scalar ½ log det₊ R.
        """
        operator = _unwrap(self.structure)
        if isinstance(
            operator, KroneckerSum | DiagonalizedOperator | lx.DiagonalLinearOperator
        ):
            # ½ log det₊ R = ½ ∑_{γ_i > 0} log γ_i
            return 0.5 * pseudo_logdet(operator)
        # ½ log det₊ R = ½ log det(R + V Vᵀ), V orthonormal
        return 0.5 * pseudo_logdet(operator, null_space=self.null_space)

    @validate_sample
    def log_prob(self, value: Float[Array, "*sample N"]) -> Float[Array, "*sample"]:
        """Log-density of the intrinsic field, with the optional extras.

            log p(x) = ((D_x − D_c) / 2) log τ − (τ / 2) (x − μ)ᵀ R (x − μ)

        ``include_normalizer`` adds ½ log det₊ R − ((D_x − D_c) / 2) log 2π;
        ``"soft"`` adds −‖Vᵀ x‖² / (2 s²), and with the normaliser also
        −D_c (log s + ½ log 2π).

        Args:
            value: Points x, shape ``(..., N)``.

        Returns:
            Log-densities, shape ``(...)``.
        """
        precision_scale = self.precision_scale
        rank = self._rank()
        # r = x − μ
        residual = value - self.loc
        # rᵀ R r
        mahalanobis = _quadratic(self.structure, residual)
        # ((D_x − D_c) / 2) log τ − (τ / 2) rᵀ R r
        log_density = (
            0.5 * rank * jnp.log(precision_scale) - 0.5 * precision_scale * mahalanobis
        )
        if self.include_normalizer:
            # + ½ log det₊ R − ((D_x − D_c) / 2) log 2π
            log_density = log_density + self._half_log_pdet() - 0.5 * rank * _LOG_2PI
        if self.constraint == "soft":
            soft_scale = self.soft_constraint_scale
            # Vᵀ x
            null_coeffs = einsum(
                value, self.null_space, "... state, state null -> ... null"
            )
            # (Vᵀ x)²
            squared_coeffs = null_coeffs**2
            # ‖Vᵀ x‖²
            null_norm_sq = reduce(squared_coeffs, "... null -> ...", "sum")
            # − ‖Vᵀ x‖² / (2 s²)
            log_density = log_density - 0.5 * null_norm_sq / soft_scale**2
            if self.include_normalizer:
                num_null = self.null_space.shape[1]
                # − D_c (log s + ½ log 2π)
                log_density = log_density - num_null * (
                    math.log(soft_scale) + 0.5 * _LOG_2PI
                )
        return log_density

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample N"]:
        """Draws orthogonal to ``null_space``, plus the soft or ``"none"`` part.

            x = P⊥ μ + P⊥ f(z) [+ V (Vᵀ V)⁻¹ s z_null for ``"soft"``]

        Algorithm:
            1. Split the key into one for the field and one for the null
               space.
            2. Draw f(z) by the structural dispatch of ``_sampler``.
            3. Project twice with P⊥: a shifted-Cholesky draw carries a
               null component of order β^(−½).
            4. ``"none"``: return μ + P⊥ f(z). Otherwise centre on P⊥ μ, and
               for ``"soft"`` add V (Vᵀ V)⁻¹ s z_null, so Vᵀ x ~ 𝒩(0, s² I).

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            sample_shape: Leading sample shape.

        Returns:
            Draws, shape ``sample_shape + (N,)``. With ``"hard"``,
            Vᵀ x = 0 to machine precision.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        key_field, key_null = jax.random.split(key)
        noise_size, draw = self._sampler()
        dtype = self.loc.dtype
        # f(z)
        field_draws = _draw_samples(key_field, sample_shape, noise_size, draw, dtype)
        # Twice is enough: a shifted-Cholesky draw carries a ~β^(−½) null
        # component, and one projection leaves eps times that.
        # P⊥ f(z)
        projected_once = self._project(field_draws)
        # P⊥ P⊥ f(z)
        field_draws = self._project(projected_once)
        if self.constraint == "none":
            # x = μ + P⊥ f(z)
            return self.loc + field_draws
        # x = P⊥ μ + P⊥ f(z)
        draws = self._project(self.loc) + field_draws
        if self.constraint == "soft":
            num_null = self.null_space.shape[1]
            # z_null ~ 𝒩(0, I)
            null_noise = jax.random.normal(
                key_null, (*sample_shape, num_null), dtype=dtype
            )
            # s z_null
            null_noise = self.soft_constraint_scale * null_noise
            # V (Vᵀ V)⁻¹ s z_null
            null_component = einsum(
                null_noise, self._null_dual(), "... null, state null -> ... state"
            )
            # x + V (Vᵀ V)⁻¹ s z_null, so that Vᵀ x ~ 𝒩(0, s² I)
            draws = draws + null_component
        return draws

    def marginal_variances(self) -> Float[Array, " N"]:
        """Marginal variances diag((τ R)⁺), plus s² diag(V (Vᵀ V)⁻² Vᵀ) for soft.

        The pseudo-inverse diagonal is exact from the factor eigenvalues on a
        Kronecker structure, and otherwise the kriging-corrected
        diag((τ R + β I)⁻¹) (R-INLA's constrained marginal variances).

        Algorithm:
            1. A factored eigenbasis: diag((τ R)⁺) from the eigenvalues.
            2. Otherwise: krige 𝒩(0, (τ R + β I)⁻¹) onto Vᵀ x = 0 and take
               its `ConstrainedGMRF` variances.
            3. ``"soft"``: add s² diag(V (Vᵀ V)⁻² Vᵀ).

        Returns:
            Marginal variances of the samples, shape ``(N,)``.
        """
        eigen = self._eigen()
        if eigen is not None:
            # diag((τ R)⁺), exact
            variances = eigen.diag_inv(pinv=True)
        else:
            shifted = GaussianMRF(jnp.zeros_like(self.loc), self._shifted())
            # Vᵀ, the constraint matrix
            null_constraints = rearrange(self.null_space, "state null -> null state")
            null_values = jnp.zeros(self.null_space.shape[1], dtype=self.loc.dtype)
            # x ∣ Vᵀ x = 0, x ~ 𝒩(0, (τ R + β I)⁻¹)
            kriged = shifted.condition_on_constraints(null_constraints, null_values)
            # diag((τ R + β I)⁻¹) − diag(W S⁻¹ Wᵀ)
            variances = kriged.marginal_variances()
        if self.constraint == "soft":
            null_dual = self._null_dual()
            soft_scale = self.soft_constraint_scale
            # (V (Vᵀ V)⁻¹)², entry by entry
            squared_dual = null_dual**2
            # diag(V (Vᵀ V)⁻² Vᵀ)
            dual_row_norms = reduce(squared_dual, "state null -> state", "sum")
            # + s² diag(V (Vᵀ V)⁻² Vᵀ)
            variances = variances + soft_scale**2 * dual_row_norms
        return variances

    @lazy_property
    def mean(self) -> Float[Array, " N"]:
        """The mean of the samples: μ for ``"none"``, else P⊥ μ."""
        if self.constraint == "none":
            return self.loc
        # P⊥ μ
        return self._project(self.loc)

    @lazy_property
    def variance(self) -> Float[Array, " N"]:
        """The marginal variances of the samples."""
        return self.marginal_variances()


# ---------------------------------------------------------------------------
# BYM2
# ---------------------------------------------------------------------------


class BYM2GMRF(IntrinsicGMRF):
    """The BYM2 pair (b, u*) (Riebler et al., 2016), with its exact density.

        b = (√(1 − φ) v + √φ u*) / √τ,   v ~ 𝒩(0, I)

    u* is the scaled ICAR field on n areas (structure R*,
    `generalized_variance_scale`) under one sum-to-zero constraint per
    connected component. The joint precision Λ(τ, φ) of (b, u*) is
    `bym2_precision`, and the field lives on {Vᵀ u* = 0}, where its
    log-density (Rue & Held, eq. 2.30) is

        log p(b, u*) = −½ rᵀ Λ r + (n / 2) log(τ / (1 − φ))
                       [+ ½ log det₊ R* − ((2n − D_c) / 2) log 2π],

    with r = (b, u*) − μ, because the Schur complement of Λ's b-block is
    exactly R*.

    Why not `IntrinsicGMRF` on Λ: Λ's null vector is (√(φ/τ) 1, 1), which
    depends on θ = (τ, φ). So the bracket `IntrinsicGMRF` drops by default
    is not θ-free (it holds the (n / 2) log(τ / (1 − φ)) above), its
    normaliser takes a dense determinant of size 2n at every θ, and its
    sampler's projection along ``null_space`` leaves the b-part of a
    shifted-Cholesky draw unbounded. Here the θ-dependent term is always
    included, the bracket is truly constant (from R* alone: the matrix-tree
    theorem, one sparse Cholesky, or ``log_pdet``), and draws are exact: u*
    from the constrained ICAR on R*, then b from its definition.

    It is an `IntrinsicGMRF` whose ``structure`` is Λ, ``precision_scale``
    is 1 and ``null_space`` is V padded with zeros on the b rows, so
    `argminax.solve.laplace_mode` and `vb_mean_correction` take it as a prior unchanged
    (their log-marginal then has the correct θ-dependence).

    Args:
        structure_scaled: Scaled structure R* = s R, a `SparseOperator`
            graph Laplacian of size n (or a scalar multiple of one).
        tau: Precision τ > 0 of b (may be traced).
        phi: Mixing 0 ≤ φ < 1 (may be traced).
        null_space: Basis V of ker R*, shape ``(n, c)`` (or ``(n,)``): one
            column per connected component (kernellib's
            ``graph_null_space``).
        loc: Mean μ of the stacked (b, u*), shape ``(2n,)``; zeros by
            default.
        constraint: ``"hard"`` (default), ``"soft"`` or ``"none"``, on u* as
            in `IntrinsicGMRF`.
        soft_constraint_scale: Scale s of the soft constraint, for
            ``"soft"``.
        include_normalizer: Add the θ-free constant in brackets.
        log_pdet: log det₊ R*, if already known: it is θ-free, so compute it
            once outside a θ loop. ``None`` computes it per call by the
            matrix-tree theorem.
        validate_args: Whether to validate input arguments.

    Raises:
        ValueError: If ``null_space`` does not have n rows, or for an
            unknown ``constraint``.

    Examples:
        A path of 3 areas, R = D − W for the degree matrix D and adjacency
        W:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx as gx
        >>> structure = gx.SparseOperator.from_coo(
        ...     np.array([0, 1, 2, 1, 2]),
        ...     np.array([0, 1, 2, 0, 1]),
        ...     jnp.array([1.0, 2.0, 1.0, -1.0, -1.0]),
        ...     (3, 3),
        ...     symmetric=True,
        ... )
        >>> null_space = jnp.ones(3) / jnp.sqrt(3.0)
        >>> scale = gx.generalized_variance_scale(structure, null_space)
        >>> prior = gx.BYM2GMRF(
        ...     scale * structure, tau=2.0, phi=0.6, null_space=null_space
        ... )
        >>> x = prior.sample(jax.random.PRNGKey(0), (100,))
        >>> x.shape, prior.log_prob(x).shape
        ((100, 6), (100,))

        Under the hard constraint u* sums to zero in every draw:

        >>> bool(jnp.allclose(x[:, 3:].sum(-1), 0.0, atol=1e-4))
        True

    References:
        Riebler, A., Sørbye, S. H., Simpson, D. and Rue, H. (2016). An
        intuitive Bayesian spatial model for disease mapping that accounts
        for scaling. Statistical Methods in Medical Research 25(4),
        1145-1165.
    """

    pytree_data_fields = (
        "tau",
        "phi",
        "structure_scaled",
        "icar_null_space",
        "log_pdet",
    )
    pytree_aux_fields = ()

    def __init__(
        self,
        structure_scaled: lx.AbstractLinearOperator,
        tau: Float[ArrayLike, ""],
        phi: Float[ArrayLike, ""],
        null_space: Float[ArrayLike, "n c"] | Float[ArrayLike, " n"],
        loc: Float[ArrayLike, " N"] | None = None,
        *,
        constraint: Literal["none", "soft", "hard"] = "hard",
        soft_constraint_scale: float = 1e-3,
        include_normalizer: bool = False,
        log_pdet: Float[ArrayLike, ""] | None = None,
        validate_args: bool | None = None,
    ) -> None:
        n = structure_scaled.in_size()
        # Λ(τ, φ) of (b, u*)
        precision = bym2_precision(structure_scaled, tau, phi)
        dtype = precision.values.dtype
        if loc is None:
            loc = jnp.zeros(2 * n, dtype=dtype)
        V = jnp.asarray(null_space, dtype=dtype)
        if V.ndim == 1:
            V = rearrange(V, "area -> area 1")
        if V.shape[0] != n:
            raise ValueError(f"null_space must have {n} rows, got shape {V.shape}.")
        # jnp, not einx: an empty null space (D_c = 0) must still construct,
        # and einx rejects a zero axis.
        # (0, V), zero on the b rows
        padded = jnp.concatenate([jnp.zeros_like(V), V])
        super().__init__(
            loc,
            1.0,
            precision,
            padded,
            constraint=constraint,
            soft_constraint_scale=soft_constraint_scale,
            include_normalizer=include_normalizer,
            validate_args=validate_args,
        )
        self.tau = jnp.asarray(tau, dtype=dtype)
        self.phi = jnp.asarray(phi, dtype=dtype)
        self.structure_scaled = structure_scaled
        self.icar_null_space = V
        self.log_pdet = None if log_pdet is None else jnp.asarray(log_pdet, dtype)

    def _n(self) -> int:
        """Number of areas n.

        Returns:
            The size n of R*.
        """
        return self.icar_null_space.shape[0]

    def _icar(self, constraint: str) -> IntrinsicGMRF:
        """The unit-precision ICAR on R* that u* is drawn from.

            u* ~ 𝒩(0, (R*)⁺)

        Args:
            constraint: ``"hard"``, ``"soft"`` or ``"none"``.

        Returns:
            The `IntrinsicGMRF` of u*.
        """
        n = self._n()
        return IntrinsicGMRF(
            jnp.zeros(n, dtype=self.loc.dtype),
            1.0,
            self.structure_scaled,
            self.icar_null_space,
            constraint=constraint,
            soft_constraint_scale=self.soft_constraint_scale,
        )

    def _half_log_pdet(self) -> Float[Array, ""]:
        """Half log pseudo-determinant ½ log det₊ R* of the scaled structure.

        Taken from ``log_pdet`` if given, else by the matrix-tree theorem
        (one sparse Cholesky).

        Returns:
            The scalar ½ log det₊ R*.
        """
        if self.log_pdet is not None:
            return 0.5 * self.log_pdet
        values, pattern = _as_sparse(self.structure_scaled)
        laplacian = SparseOperator(values, pattern)
        # ½ log det₊ R*, by the matrix-tree theorem
        return 0.5 * pseudo_logdet(laplacian, structure="laplacian")

    @validate_sample
    def log_prob(self, value: Float[Array, "*sample N"]) -> Float[Array, "*sample"]:
        """Log-density of the BYM2 pair, with the optional extras.

            log p(b, u*) = −½ rᵀ Λ r + (n / 2) log(τ / (1 − φ))

        ``include_normalizer`` adds ½ log det₊ R* − ((2n − D_c) / 2) log 2π;
        ``"soft"`` adds the soft-constraint terms of `IntrinsicGMRF`.

        Args:
            value: Points (b, u*), shape ``(..., 2n)``.

        Returns:
            Log-densities, shape ``(...)``.
        """
        n = self._n()
        # r = (b, u*) − μ
        residual = value - self.loc
        # rᵀ Λ r
        mahalanobis = _quadratic(self.structure, residual)
        # −½ rᵀ Λ r + (n / 2) log(τ / (1 − φ))
        log_density = -0.5 * mahalanobis + 0.5 * n * jnp.log(
            self.tau / (1.0 - self.phi)
        )
        if self.include_normalizer:
            # rank(Λ) = 2n − D_c
            rank = 2 * n - self.icar_null_space.shape[1]
            # + ½ log det₊ R* − ((2n − D_c) / 2) log 2π
            log_density = log_density + self._half_log_pdet() - 0.5 * rank * _LOG_2PI
        if self.constraint == "soft":
            soft_scale = self.soft_constraint_scale
            # Vᵀ u*, through the zero-padded basis
            null_coeffs = einsum(
                value, self.null_space, "... state, state null -> ... null"
            )
            # (Vᵀ u*)²
            squared_coeffs = null_coeffs**2
            # ‖Vᵀ u*‖²
            null_norm_sq = reduce(squared_coeffs, "... null -> ...", "sum")
            # − ‖Vᵀ u*‖² / (2 s²)
            log_density = log_density - 0.5 * null_norm_sq / soft_scale**2
            if self.include_normalizer:
                num_null = self.null_space.shape[1]
                # − D_c (log s + ½ log 2π)
                log_density = log_density - num_null * (
                    math.log(soft_scale) + 0.5 * _LOG_2PI
                )
        return log_density

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample N"]:
        """Exact draws: u* from the ICAR on R*, then b from u*.

            u* ~ 𝒩(0, (R*)⁺),   v ~ 𝒩(0, I),   b = (√(1 − φ) v + √φ u*) / √τ

        Algorithm:
            1. Split the key into one for u* and one for v.
            2. Draw u* from the ICAR on R* under the same constraint.
            3. Draw v and form b from its definition.
            4. Stack (b, u*) and add the centre, μ for ``"none"`` and P⊥ μ
               otherwise.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            sample_shape: Leading sample shape.

        Returns:
            Draws (b, u*), shape ``sample_shape + (2n,)``.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        key_icar, key_iid = jax.random.split(key)
        n = self._n()
        # u* ~ 𝒩(0, (R*)⁺)
        icar_draws = self._icar(self.constraint).sample(key_icar, sample_shape)
        # v ~ 𝒩(0, I)
        iid_draws = jax.random.normal(key_iid, (*sample_shape, n), dtype=self.loc.dtype)
        # b = (√(1 − φ) v + √φ u*) / √τ
        total_draws = (
            jnp.sqrt(1.0 - self.phi) * iid_draws + jnp.sqrt(self.phi) * icar_draws
        ) / jnp.sqrt(self.tau)
        # μ for "none", else P⊥ μ
        centre = self.loc if self.constraint == "none" else self._project(self.loc)
        # (b, u*)
        stacked = einx.id(
            "... area, ... icar -> ... (area + icar)", total_draws, icar_draws
        )
        # x = centre + (b, u*)
        return centre + stacked

    def marginal_variances(self) -> Float[Array, " N"]:
        """Marginal variances 𝕍[u*] from the ICAR on R*, and 𝕍[b] from them.

            𝕍[b_i] = ((1 − φ) + φ 𝕍[u*_i]) / τ

        Returns:
            Marginal variances of (b, u*), shape ``(2n,)``.
        """
        # 𝕍[u*]
        icar_variances = self._icar(self.constraint).marginal_variances()
        # 𝕍[b] = ((1 − φ) + φ 𝕍[u*]) / τ
        total_variances = ((1.0 - self.phi) + self.phi * icar_variances) / self.tau
        # (𝕍[b], 𝕍[u*])
        return jnp.concatenate([total_variances, icar_variances])
