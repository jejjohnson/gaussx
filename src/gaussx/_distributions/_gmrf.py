"""Gaussian Markov random fields x ~ 𝒩(μ, Λ⁻¹) with a structured precision Λ.

`GaussianMRF` is a proper field with precision Λ, `ConstrainedGMRF` the same
field conditioned on hard linear constraints A_c x = e, `IntrinsicGMRF` the
improper field x ~ 𝒩(μ, (τ R)⁺) with structure matrix R, precision scale τ
and null-space basis V, and `BYM2GMRF` the BYM2 pair with mixing φ on a
scaled structure R*. All four are equinox modules whose precision is a lineax
operator, and every operation dispatches on its structure:

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
from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import lineax as lx
from jaxtyping import Array, ArrayLike, Float

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._distributions._gmrf_dispatch import (
    _EIGEN,
    _PSD,
    _as_operator,
    _Draw,
    _draw_samples,
    _eigen_draw,
    _flat,
    _is_factorable,
    _perturbation_draw,
    _precision_sampler,
    _quadratic,
    _scaled,
    _solve_fn,
    _solve_strategy,
    _symmetric_root,
    _unwrap,
)
from gaussx._distributions._utils import _reshape_samples
from gaussx._einx import einsum, rearrange, reduce
from gaussx._gmrf._areal import _as_sparse, bym2_precision
from gaussx._gmrf._intrinsic import grounded_structure, null_dual, pinv_diagonal
from gaussx._gmrf._regularized import (
    RegularizedFactor,
    is_groundable,
    regularized_factor,
)
from gaussx._linalg._diag_inv import diag_inv
from gaussx._operators._diagonalised import DiagonalizedOperator
from gaussx._operators._factored_eigen import FactoredEigen, factored_eigen
from gaussx._operators._grounding import ground_nodes
from gaussx._operators._kronecker_sum import KroneckerSum
from gaussx._operators._sparse import SparseOperator
from gaussx._primitives._logdet import pseudo_logdet
from gaussx._strategies._base import AbstractLogdetStrategy, AbstractSolverStrategy
from gaussx._strategies._dispatch import dispatch_logdet
from gaussx._strategies._sparse_cholesky import SparseCholeskySolver


# ---------------------------------------------------------------------------
# GaussianMRF
# ---------------------------------------------------------------------------


class GaussianMRF(eqx.Module):
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

    ``loc`` is a single field (``batch_shape == ()``); ``log_prob`` takes any
    leading sample axes. The field is an equinox module, not a NumPyro
    distribution: wrap it in ``pyrox_gauss.GMRFDistribution`` for ``numpyro.sample``.

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

    loc: Float[Array, " N"]
    precision: lx.AbstractLinearOperator
    precision_factors: tuple[lx.AbstractLinearOperator, ...] | None
    solver: AbstractSolverStrategy | None
    logdet_strategy: AbstractLogdetStrategy | None
    log_det_precision: Float[Array, ""] | None

    def __init__(
        self,
        loc: Float[ArrayLike, " N"],
        precision: lx.AbstractLinearOperator,
        precision_factors: tuple[lx.AbstractLinearOperator, ...] | None = None,
        solver: AbstractSolverStrategy | None = None,
        logdet_strategy: AbstractLogdetStrategy | None = None,
        log_det_precision: Float[ArrayLike, ""] | None = None,
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

    @property
    def event_shape(self) -> tuple[int]:
        """The event shape ``(D_x,)``."""
        return (self.loc.shape[-1],)

    @property
    def batch_shape(self) -> tuple[()]:
        """The batch shape, always ``()``: one field per object."""
        return ()

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

    @property
    def mean(self) -> Float[Array, " N"]:
        """The mean μ."""
        return self.loc

    @property
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
        *,
        null_space: Float[ArrayLike, "N p"] | Float[ArrayLike, " N"] | None = None,
    ) -> ConstrainedGMRF:
        """Condition on hard linear constraints A_c x = e, by kriging.

            x ∣ A_c x = e,   x ~ 𝒩(μ, Λ⁻¹)

        Args:
            A_c: Constraint matrix A_c, shape ``(c, N)`` (or ``(N,)`` for one
                constraint).
            e: Constraint values e, shape ``(c,)``.
            null_space: A basis whose span holds every null vector of Λ, for a Λ
                that is positive definite on {A_c x = 0} but may be singular
                outside it; ``None`` for a positive-definite Λ.

        Returns:
            The `ConstrainedGMRF` of x ∣ A_c x = e.
        """
        return ConstrainedGMRF(self, A_c, e, null_space)


# ---------------------------------------------------------------------------
# ConstrainedGMRF
# ---------------------------------------------------------------------------


class ConstrainedGMRF(eqx.Module):
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

    **Singular Λ.** The conditional is proper as soon as Λ is positive
    definite on {A_c x = 0}. With ``null_space``, a basis whose span holds
    ker Λ, every formula above takes Λ + A_cᵀ B A_c in place of Λ, for a
    positive diagonal B: the added term is constant on {A_c x = e}, so the
    conditional is the same, and the sum is positive definite. It is never
    formed: Λ plus a diagonal at D_p nodes, on Λ's sparsity pattern, is
    factored, and the rest is a correction of rank D_p + D_c.

    Built by `GaussianMRF.condition_on_constraints`.

    Args:
        base: The unconstrained field 𝒩(μ, Λ⁻¹).
        constraint_matrix: Constraint matrix A_c, shape ``(c, N)`` (or
            ``(N,)``), of full row rank.
        constraint_value: Constraint values e, shape ``(c,)``.
        null_space: A basis whose span holds every null vector of Λ, shape
            ``(N, p)`` (or ``(N,)``), for a Λ singular outside {A_c x = 0};
            ``None`` for a positive-definite Λ.

    Raises:
        ValueError: If A_c and e do not have shapes ``(c, N)`` and ``(c,)``, or
            ``null_space`` does not have N rows.
        TypeError: For a ``null_space`` with a precision that is neither sparse,
            symmetric block-tridiagonal nor dense.

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

        A random walk under a sum-to-zero constraint. Its precision is singular
        along the constants, which the constraint removes:

        >>> walk = gx.rw1_structure(6)
        >>> field = gx.GaussianMRF(jnp.zeros(6), walk).condition_on_constraints(
        ...     jnp.ones(6), 0.0, null_space=jnp.ones(6)
        ... )
        >>> exact = jnp.diag(jnp.linalg.pinv(walk.as_matrix()))
        >>> bool(jnp.allclose(field.marginal_variances(), exact, atol=1e-4))
        True

    References:
        Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory
        and Applications. Chapman & Hall/CRC.
    """

    base: GaussianMRF
    constraint_matrix: Float[Array, "c N"]
    constraint_value: Float[Array, " c"]
    null_space: Float[Array, "N p"] | None

    def __init__(
        self,
        base: GaussianMRF,
        constraint_matrix: Float[ArrayLike, "c N"] | Float[ArrayLike, " N"],
        constraint_value: Float[ArrayLike, " c"] | Float[ArrayLike, ""],
        null_space: Float[ArrayLike, "N p"] | Float[ArrayLike, " N"] | None = None,
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
        if null_space is not None:
            null_space = jnp.asarray(null_space, dtype=dtype)
            if null_space.ndim == 1:
                null_space = rearrange(null_space, "state -> state 1")
            if null_space.shape[0] != n:
                raise ValueError(
                    f"null_space must have {n} rows, got shape {null_space.shape}."
                )
            if not is_groundable(_unwrap(base.precision)):
                raise TypeError(
                    "a singular precision must be a SparseOperator, a symmetric "
                    "BlockTriDiag or a dense operator, got "
                    f"{type(_unwrap(base.precision)).__name__}."
                )
        self.base = base
        self.constraint_matrix = A
        self.constraint_value = e
        self.null_space = null_space

    @property
    def event_shape(self) -> tuple[int]:
        """The event shape ``(D_x,)``, the base field's."""
        return self.base.event_shape

    @property
    def batch_shape(self) -> tuple[()]:
        """The batch shape, always ``()``: one field per object."""
        return ()

    def _regularized(self) -> RegularizedFactor | None:
        """Factor of Λ + A_cᵀ B A_c for a singular Λ, else ``None``."""
        if self.null_space is None:
            return None
        solver = self.base.solver
        return regularized_factor(
            _unwrap(self.base.precision),
            rearrange(self.constraint_matrix, "con state -> state con"),
            ground_nodes(self.null_space),
            solver if isinstance(solver, SparseCholeskySolver) else None,
        )

    def _kriging(
        self, factor: RegularizedFactor | None
    ) -> tuple[Float[Array, "c N"], Float[Array, "c N"], Array]:
        """Kriging terms Wᵀ, S⁻¹ Wᵀ and the Cholesky factor L_S of S.

            W = Λ⁻¹ A_cᵀ,   S = A_c W = L_S L_Sᵀ

        Args:
            factor: The factor of Λ + A_cᵀ B A_c for a singular Λ, which then
                stands for Λ; ``None`` otherwise.

        Returns:
            Tuple ``(solved_constraints, kriging_rows, constraint_cholesky)``
            holding Wᵀ and S⁻¹ Wᵀ (one row per constraint) and L_S.
        """
        if factor is None:
            precision_solve = _solve_fn(self.base.precision, self.base.solver)
            # Wᵀ = (Λ⁻¹ A_cᵀ)ᵀ, one solve per constraint
            solved_constraints = jax.vmap(precision_solve)(self.constraint_matrix)
        else:
            solved_constraints = factor.solve_constraints()
        return self._krige(solved_constraints)

    def _krige(
        self, solved_constraints: Float[Array, "c N"]
    ) -> tuple[Float[Array, "c N"], Float[Array, "c N"], Array]:
        """The kriging terms of ``_kriging`` from Wᵀ.

        Args:
            solved_constraints: Wᵀ, one row per constraint.

        Returns:
            Wᵀ, S⁻¹ Wᵀ and L_S.
        """
        constraint_matrix = self.constraint_matrix
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
        factor = self._regularized()
        _, _, constraint_cholesky = self._kriging(factor)
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
        if factor is None:
            # log p(x)
            base_log_prob = self.base.log_prob(value)
        else:
            # log 𝒩(x; μ, (Λ + A_cᵀ B A_c)⁻¹), at A_c x = e
            base_log_prob = self._regularized_log_prob(factor, value, residual)
        # log p(x ∣ A_c x = e)
        return base_log_prob - log_constraint - 0.5 * log_det_gram

    def _regularized_log_prob(
        self,
        factor: RegularizedFactor,
        value: Float[Array, "*sample N"],
        constraint_residual: Float[Array, " c"],
    ) -> Float[Array, "*sample"]:
        """Log-density of 𝒩(μ, Λ'⁻¹) on A_c x = e, Λ' = Λ + A_cᵀ B A_c.

            rᵀ Λ' r = rᵀ Λ r + (e − A_c μ)ᵀ B (e − A_c μ),   r = x − μ

        Args:
            factor: The factor of Λ'.
            value: Points x satisfying the constraint, shape ``(..., N)``.
            constraint_residual: e − A_c μ.

        Returns:
            Log-densities, shape ``(...)``.
        """
        event_size = self.event_shape[0]
        # rᵀ Λ r
        mahalanobis = _quadratic(self.base.precision, value - self.base.loc)
        # B (e − A_c μ)
        weighted_residual = factor.constraint_precision * constraint_residual
        # (e − A_c μ)ᵀ B (e − A_c μ)
        constraint_term = einsum(constraint_residual, weighted_residual, "con, con ->")
        return 0.5 * (
            factor.logdet() - mahalanobis - constraint_term - event_size * _LOG_2PI
        )

    def _singular_sample(
        self,
        factor: RegularizedFactor,
        key: jax.Array | None,
        sample_shape: tuple[int, ...],
    ) -> Float[Array, "*sample N"]:
        """Exact constrained draws for a singular Λ.

            x = (kriged draw from 𝒩(μ, M⁻¹)) + F ξ,   ξ ~ 𝒩(0, I)

        M = Λ + E K Eᵀ is positive definite, and F Fᵀ is what the kriged
        covariance of M lacks (`RegularizedFactor.draw_correction`).

        Args:
            factor: The factor of Λ + A_cᵀ B A_c, which holds M.
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            sample_shape: Leading sample shape.

        Returns:
            Draws, shape ``sample_shape + (N,)``.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        key_base, key_correction = jax.random.split(key)
        # A_c M⁻¹, the constraint rows of the border's solves
        num_nodes = factor.node_precision.shape[0]
        solved_constraints, kriging_rows, _ = self._krige(
            factor.solved_border[num_nodes:]
        )
        correction_rows = factor.draw_correction(solved_constraints, kriging_rows)
        solver = self.base.solver
        grounded_solver = solver if isinstance(solver, SparseCholeskySolver) else None
        noise_size, draw = _precision_sampler(factor.grounded, grounded_solver, None)
        dtype = self.base.loc.dtype
        # f(z) ~ 𝒩(0, M⁻¹)
        base_draws = _draw_samples(key_base, sample_shape, noise_size, draw, dtype)
        # ξ ~ 𝒩(0, I)
        noise = jax.random.normal(
            key_correction, (*sample_shape, correction_rows.shape[0]), dtype=dtype
        )
        # F ξ
        correction = einsum(noise, correction_rows, "... node, node state -> ... state")
        # the conditional mean, plus the zero-mean draw off the constraint
        excess = einsum(
            base_draws, self.constraint_matrix, "... state, con state -> ... con"
        )
        # W S⁻¹ A_c f
        null_part = einsum(excess, kriging_rows, "... con, con state -> ... state")
        draws = base_draws - null_part + correction
        return self._mean(factor) + draws

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

            For a singular Λ, the base field is the one with a diagonal added at
            a few nodes, and an independent draw restores the variance it lacks.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            sample_shape: Leading sample shape.

        Returns:
            Draws, shape ``sample_shape + (N,)``.
        """
        factor = self._regularized()
        if factor is not None:
            return self._singular_sample(factor, key, sample_shape)
        _, kriging_rows, _ = self._kriging(None)
        # x ~ 𝒩(μ, Λ⁻¹)
        base_draws = self.base.sample(key, sample_shape)
        # x − W S⁻¹ (A_c x − e)
        return self._correct(base_draws, kriging_rows)

    def marginal_variances(self) -> Float[Array, " N"]:
        """Constrained marginal variances diag(Λ⁻¹) − diag(W S⁻¹ Wᵀ).

        Returns:
            Constrained marginal variances, shape ``(N,)``.
        """
        factor = self._regularized()
        solved_constraints, kriging_rows, _ = self._kriging(factor)
        # W ⊙ (S⁻¹ Wᵀ)ᵀ, entry by entry
        products = solved_constraints * kriging_rows
        # diag(W S⁻¹ Wᵀ)
        correction = reduce(products, "constraint state -> state", "sum")
        if factor is None:
            # diag(Λ⁻¹)
            variances = self.base.marginal_variances()
        else:
            # diag((Λ + A_cᵀ B A_c)⁻¹)
            variances = factor.diag_inv()
        # diag(Λ⁻¹) − diag(W S⁻¹ Wᵀ)
        return variances - correction

    def _mean(self, factor: RegularizedFactor | None) -> Float[Array, " N"]:
        """The kriged mean, given the factor of a singular Λ."""
        _, kriging_rows, _ = self._kriging(factor)
        # μ − W S⁻¹ (A_c μ − e)
        return self._correct(self.base.loc, kriging_rows)

    @property
    def mean(self) -> Float[Array, " N"]:
        """The kriged mean μ − W S⁻¹ (A_c μ − e)."""
        return self._mean(self._regularized())

    @property
    def variance(self) -> Float[Array, " N"]:
        """The constrained marginal variances."""
        return self.marginal_variances()


# ---------------------------------------------------------------------------
# IntrinsicGMRF
# ---------------------------------------------------------------------------


class IntrinsicGMRF(eqx.Module):
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
      exactly the constrained density on that subspace, and samples and
      marginal variances are those of 𝒩(μ, (τ R)⁺).
    - ``"soft"`` (for NUTS, which cannot sample a hard constraint): adds a
      tight Gaussian Vᵀ x ~ 𝒩(0, s² I), s = ``soft_constraint_scale``, to
      the density and to the samples.
    - ``"none"``: only μ plus a draw from 𝒩(0, (τ R)⁺); the density is
      improper and invariant along ker R.

    The draw from 𝒩(0, (τ R)⁺) dispatches like `GaussianMRF`: the Cholesky
    factor of the grounded structure (sparse, banded or dense), the
    pseudo-inverse root in the factored eigenbasis, or
    perturbation-optimisation with ``precision_factors``
    (R = ∑_k F_kᵀ F_k, so the right-hand side lies in range(R) and CG
    converges on the singular system); each is then projected orthogonally
    to ``null_space``.

    **Grounding.** R is singular, so it is factored with the rows and columns
    of D_c nodes replaced by a unit diagonal, nodes on which every vector of
    ker R is determined (one per connected component of a graph, the two ends
    of an RW2). With zeros on those nodes, that inverse G is a generalised
    inverse of R, and R⁺ = P⊥ G P⊥ with P⊥ = I − V (Vᵀ V)⁻¹ Vᵀ: exact, on
    R's sparsity pattern, with no ridge. The grounded matrix of a path is
    conditioned like N² for RW1 and N⁴ for RW2, so float64 carries an RW2 of a
    few thousand nodes and float32 one of about fifty.

    **Odd-n RW2.** `rw2_structure` with odd n returns n + 1 rows, the last a
    decoupled unit-precision node. Build the field on all D_x = n + 1 nodes
    with ``null_space`` zero on the padding row (D_c = 2): then
    D_x − D_c = n − 1 = rank(R), and the padding node is an independent
    𝒩(0, 1/τ) coordinate whose own ½ log τ is the one extra in
    ((D_x − D_c) / 2) log τ. Sample it with the rest and drop it
    (``x[:n]``).

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
        constraint_basis: Basis of the hard constraint a Laplace fit puts on
            the field, when it is only part of ker R, shape ``(N, d)`` (or
            ``(N,)``): R-INLA's RW2 constrains the sum and leaves the trend
            free. ``None`` constrains all of ``null_space``. The field's own
            samples and variances are always those of 𝒩(μ, (τ R)⁺).

    Raises:
        ValueError: For an unknown ``constraint`` or mismatched shapes, or a
            ``constraint_basis`` without ``constraint="hard"``.

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

    loc: Float[Array, " N"]
    precision_scale: Float[Array, ""]
    structure: lx.AbstractLinearOperator
    null_space: Float[Array, "N c"]
    precision_factors: tuple[lx.AbstractLinearOperator, ...] | None
    constraint: Literal["none", "soft", "hard"] = eqx.field(static=True)
    soft_constraint_scale: float = eqx.field(static=True)
    include_normalizer: bool = eqx.field(static=True)
    constraint_basis: Float[Array, "N d"] | None

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
        constraint_basis: Float[ArrayLike, "N d"]
        | Float[ArrayLike, " N"]
        | None = None,
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
        if constraint_basis is not None:
            if constraint != "hard":
                raise ValueError(
                    "constraint_basis is the basis of a hard constraint, got "
                    f"constraint={constraint!r}."
                )
            constraint_basis = jnp.asarray(constraint_basis, dtype=loc.dtype)
            if constraint_basis.ndim == 1:
                constraint_basis = rearrange(constraint_basis, "state -> state 1")
            if constraint_basis.shape[0] != n:
                raise ValueError(
                    f"constraint_basis must have {n} rows, got shape "
                    f"{constraint_basis.shape}."
                )
        self.constraint_basis = constraint_basis

    @property
    def event_shape(self) -> tuple[int]:
        """The event shape ``(D_x,)``."""
        return (self.loc.shape[-1],)

    @property
    def batch_shape(self) -> tuple[()]:
        """The batch shape, always ``()``: one field per object."""
        return ()

    @property
    def hard_constraint(self) -> Float[Array, "N d"]:
        """Basis of the hard constraint: ``constraint_basis``, or ``null_space``."""
        if self.constraint_basis is None:
            return self.null_space
        return self.constraint_basis

    # -- null space -----------------------------------------------------------

    def _null_dual(self) -> Float[Array, "N c"]:
        """Dual basis V (Vᵀ V)⁻¹ of the null space."""
        return null_dual(self.null_space)

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
            3. Otherwise: the `GaussianMRF` dispatch on the grounded R, with
               zeros on the fixed nodes, over √τ.

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
        grounded, free = grounded_structure(self.structure, self.null_space)
        noise_size, draw = _precision_sampler(grounded, None, None)
        # √τ
        root_scale = jnp.sqrt(self.precision_scale)

        def grounded_draw(noise: Array) -> Array:
            # g ~ 𝒩(0, G), zero on the fixed nodes
            free_draw = jnp.where(free, draw(noise), 0.0)
            # g / √τ
            return free_draw / root_scale

        return noise_size, grounded_draw

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
            3. Project with P⊥, twice for the rounding of the first.
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
        # A grounded draw's null component can be far larger than the draw; the
        # second projection removes what rounding left of it.
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

        The pseudo-inverse diagonal comes from the factor eigenvalues on a
        Kronecker structure, and from the grounded structure otherwise.

        Algorithm:
            1. A factored eigenbasis: diag((τ R)⁺) from the eigenvalues.
            2. Otherwise: diag(R⁺) / τ, with R⁺ = P⊥ G P⊥.
            3. ``"soft"``: add s² diag(V (Vᵀ V)⁻² Vᵀ).

        Returns:
            Marginal variances of the samples, shape ``(N,)``.
        """
        eigen = self._eigen()
        if eigen is not None:
            # diag((τ R)⁺), exact
            variances = eigen.diag_inv(pinv=True)
        else:
            # diag(R⁺) / τ
            variances = (
                pinv_diagonal(self.structure, self.null_space) / self.precision_scale
            )
        if self.constraint == "soft":
            soft_scale = self.soft_constraint_scale
            # (V (Vᵀ V)⁻¹)², entry by entry
            squared_dual = self._null_dual() ** 2
            # diag(V (Vᵀ V)⁻² Vᵀ)
            dual_row_norms = reduce(squared_dual, "state null -> state", "sum")
            # + s² diag(V (Vᵀ V)⁻² Vᵀ)
            variances = variances + soft_scale**2 * dual_row_norms
        return variances

    @property
    def mean(self) -> Float[Array, " N"]:
        """The mean of the samples: μ for ``"none"``, else P⊥ μ."""
        if self.constraint == "none":
            return self.loc
        # P⊥ μ
        return self._project(self.loc)

    @property
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
    sampler's projection along ``null_space`` leaves the b-part of a draw
    off the null vector. Here the θ-dependent term is always
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

    tau: Float[Array, ""]
    phi: Float[Array, ""]
    structure_scaled: lx.AbstractLinearOperator
    icar_null_space: Float[Array, "n c"]
    log_pdet: Float[Array, ""] | None

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

    def _icar(self, constraint: Literal["none", "soft", "hard"]) -> IntrinsicGMRF:
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
