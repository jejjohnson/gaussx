r"""Gaussian Markov random fields: `GaussianMRF`, `IntrinsicGMRF`, `ConstrainedGMRF`.

All three are NumPyro distributions whose precision is a structured lineax
operator, and every operation dispatches on that structure (roadmap §3):

- **sample**, in order: (1) a factorisable precision (dense, `BlockTriDiag`,
  or a `SparseOperator` through its sparse Cholesky factor) gives
  $\mu + L^{-\top}z$; (2) a Kronecker-structured one (`Kronecker`,
  `KroneckerSum`, `SpectralFunction`, `DiagonalisedOperator`) the symmetric
  root $V^{-1}\Lambda^{-1/2}Vz$ in its factored eigenbasis; (3) otherwise
  perturbation-optimisation: with $Q = \sum_k F_k^\top F_k$ and
  $r = \sum_k F_k^\top z_k$, $\operatorname{Cov}(Q^{-1}r) = Q^{-1}$, so one
  CG solve per draw and no factorisation;
- **log_prob**: $\log|Q|$ from a known value, a strategy, or the structural
  `logdet`;
- **marginal variances**: `diag_inv` (block / sparse Takahashi, factor
  eigenvectors);
- **constraints** $A_cx = e$: conditioning by kriging (Rue & Held, 2005,
  §2.3.3), $x^\ast = x - Q^{-1}A_c^\top S^{-1}(A_cx - e)$ with
  $S = A_cQ^{-1}A_c^\top$, at ``c`` extra solves.
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

from gaussx._distributions._utils import _reshape_batch, _reshape_samples
from gaussx._einx import einsum, rearrange, reduce
from gaussx._gmrf._areal import _add_ridge
from gaussx._linalg._diag_inv import diag_inv
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._diagonalised import DiagonalisedOperator
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


_LOG_2PI = math.log(2.0 * math.pi)
_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
# Operators sampled in their factored eigenbasis (dispatch branch 2).
_EIGEN = (Kronecker, KroneckerSum, SpectralFunction, DiagonalisedOperator)
# Operators factored by Cholesky even when factors are given (branch 1).
_FACTORABLE = (
    lx.MatrixLinearOperator,
    lx.DiagonalLinearOperator,
    lx.IdentityLinearOperator,
)

# ``f(z)``: a zero-mean draw from standard normal noise ``z``.
_Draw = Callable[[Array], Array]


# ---------------------------------------------------------------------------
# Sampling dispatch
# ---------------------------------------------------------------------------


def _unwrap(operator: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    return operator


def _is_factorable(operator: lx.AbstractLinearOperator) -> bool:
    """Dense, diagonal or a symmetric `BlockTriDiag`: Cholesky is the cheapest."""
    return isinstance(operator, _FACTORABLE) or (
        isinstance(operator, BlockTriDiag) and operator.symmetric
    )


def _cholesky_draw(operator: lx.AbstractLinearOperator) -> _Draw:
    """``z ↦ L⁻ᵀz`` with ``Q = LLᵀ`` (structured `cholesky`)."""
    L = cholesky(operator)
    if isinstance(L, lx.MatrixLinearOperator):
        matrix = L.matrix
        return lambda z: jsl.solve_triangular(matrix, z, lower=True, trans="T")
    upper = L.T
    return lambda z: solve(upper, z)


def _eigen_draw(eigen: FactoredEigen, z: Array, *, pinv: bool = False) -> Array:
    """``V⁻¹ Λ^{-1/2} V z``, the symmetric root (``Λ⁺`` with ``pinv``)."""
    inv_root = jnp.sqrt(_reciprocal(eigen.eigenvalues, pinv=pinv))
    x = eigen.from_eigenbasis(eigen.to_eigenbasis(z) * inv_root)
    return eigen._output(x, z)


def _cg_solve(operator: lx.AbstractLinearOperator, vector: Array) -> Array:
    """CG at ``√eps`` of the dtype (converges on a singular ``Q`` if ``b ∈ range``)."""
    tol = float(jnp.finfo(vector.dtype).eps) ** 0.5
    return lx.linear_solve(operator, vector, lx.CG(rtol=tol, atol=tol)).value


def _perturbation_draw(
    precision: lx.AbstractLinearOperator,
    factors: tuple[lx.AbstractLinearOperator, ...],
    solver: AbstractSolverStrategy | None,
) -> tuple[int, _Draw]:
    """Perturbation-optimisation: solve ``Q x = Σ_k F_kᵀ z_k``."""
    sizes = [factor.out_size() for factor in factors]
    splits = np.cumsum(sizes)[:-1].tolist()
    operator = lx.TaggedLinearOperator(precision, _PSD)

    def draw(z: Array) -> Array:
        parts = jnp.split(z, splits)
        rhs = sum(
            factor.transpose().mv(part)
            for factor, part in zip(factors, parts, strict=True)
        )
        if solver is not None:
            return solver.solve(operator, rhs)
        return _cg_solve(operator, rhs)

    return sum(sizes), draw


def _precision_sampler(
    precision: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy | None,
    factors: tuple[lx.AbstractLinearOperator, ...] | None,
) -> tuple[int, _Draw]:
    """``(m, f)`` with ``f(z) ~ N(0, Q⁻¹)`` for ``z ~ N(0, I_m)``.

    The factorisation is done here, once, outside any ``vmap`` over draws.
    """
    operator = _unwrap(precision)
    n = precision.in_size()
    if isinstance(operator, SparseOperator) and (
        factors is None or isinstance(solver, SparseCholeskySolver)
    ):
        if not isinstance(solver, SparseCholeskySolver):
            solver = SparseCholeskySolver()
        return n, solver.factor(operator).solve_lower_transpose
    if _is_factorable(operator):
        return n, _cholesky_draw(operator)
    if isinstance(operator, _EIGEN):
        eigen = factored_eigen(operator)
        if eigen is not None:
            return n, lambda z: _eigen_draw(eigen, z)
    if factors is not None:
        return _perturbation_draw(precision, factors, solver)
    return n, _cholesky_draw(operator)


def _draw_samples(
    key: jax.Array | None,
    sample_shape: tuple[int, ...],
    size: int,
    draw: _Draw,
    dtype: jnp.dtype,
) -> Float[Array, "*sample N"]:
    """Zero-mean draws of shape ``sample_shape + (N,)``, one ``normal`` call."""
    if key is None:
        key = jax.random.PRNGKey(0)
    count = math.prod(sample_shape)
    z = jax.random.normal(key, (count, size), dtype=dtype)
    return _reshape_samples(jax.vmap(draw)(z), sample_shape)


def _flat(value: Float[Array, "*sample N"]) -> Float[Array, "S N"]:
    return rearrange(value, "... n -> (...) n")


def _quadratic(operator: lx.AbstractLinearOperator, residual: Array) -> Array:
    """``rᵀ Q r`` for each row of ``residual`` (shape ``(..., N)``)."""
    flat = _flat(residual)
    values = jax.vmap(lambda r: einsum(r, operator.mv(r), "n, n ->"))(flat)
    return _reshape_batch(values, residual.shape[:-1])


def _solve_strategy(
    precision: lx.AbstractLinearOperator, solver: AbstractSolverStrategy | None
) -> AbstractSolverStrategy | None:
    """``solver``, or sparse Cholesky for a `SparseOperator`."""
    if solver is None and isinstance(_unwrap(precision), SparseOperator):
        return SparseCholeskySolver()
    return solver


def _solve_fn(
    precision: lx.AbstractLinearOperator, solver: AbstractSolverStrategy | None
) -> Callable[[Array], Array]:
    """``b ↦ Q⁻¹b``, factoring a sparse precision once for many solves."""
    strategy = _solve_strategy(precision, solver)
    if isinstance(strategy, SparseCholeskySolver):
        return strategy.factor(precision).solve
    return lambda b: dispatch_solve(precision, b, strategy)


class _Sqrt(eqx.Module):
    """``λ ↦ √f(λ)``, a pytree so ``f``'s parameters stay leaves."""

    fn: Callable[[Array], Array]

    def __call__(self, eigenvalues: Array) -> Array:
        return jnp.sqrt(self.fn(eigenvalues))


def _symmetric_root(
    operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator | None:
    """``Q^{1/2}`` (symmetric, so ``(Q^{1/2})ᵀQ^{1/2} = Q``) of a grid precision."""
    if isinstance(operator, SpectralFunction):
        return SpectralFunction(
            operator.base, _Sqrt(operator.fn), _eigen=operator.eigen
        )
    if isinstance(operator, KroneckerSum):
        return KroneckerSumSqrt(operator.A, operator.B)
    return None


def _as_operator(A: lx.AbstractLinearOperator | ArrayLike) -> lx.AbstractLinearOperator:
    if isinstance(A, lx.AbstractLinearOperator):
        return A
    return lx.MatrixLinearOperator(jnp.asarray(A))


def _scaled(
    operator: lx.AbstractLinearOperator, scale: Float[Array, ""]
) -> lx.AbstractLinearOperator:
    """``scale · R``, keeping a sparse, banded or dense structure."""
    inner = _unwrap(operator)
    if isinstance(inner, SparseOperator):
        return SparseOperator(scale * inner.values, inner.pattern, tags=_PSD)
    if isinstance(inner, BlockTriDiag):
        return BlockTriDiag(
            scale * inner.diagonal,
            scale * inner.sub_diagonal,
            symmetric=inner.symmetric,
            tags=_PSD,
        )
    if isinstance(inner, lx.MatrixLinearOperator):
        return lx.MatrixLinearOperator(scale * inner.matrix, _PSD)
    return lx.TaggedLinearOperator(scale * operator, _PSD)


# ---------------------------------------------------------------------------
# GaussianMRF
# ---------------------------------------------------------------------------


class GaussianMRF(dist.Distribution):
    r"""Gaussian Markov random field $x \sim \mathcal N(\mu, Q^{-1})$.

    The precision is a structured lineax operator, and ``log_prob``,
    ``sample`` and ``marginal_variances`` take its cheapest exact path:

    - `SparseOperator`: sparse Cholesky, ``μ + Pᵀ L⁻ᵀ z``; Takahashi for the
      marginal variances.
    - `BlockTriDiag`: block Cholesky and the block selected inverse.
    - `Kronecker`, `KroneckerSum`, `SpectralFunction`: the factor
      eigenvectors, ``V⁻¹ Λ^{-1/2} V z``, and the factor eigenvalues.
    - dense: Cholesky.
    - anything else, with ``precision_factors``: perturbation-optimisation,
      one solve of ``Q x = Σ_k F_kᵀ z_k`` per draw; ``solver`` /
      ``logdet_strategy`` for ``log|Q|``.

    A `SparseOperator` defaults to `SparseCholeskySolver` for every solve and
    log-determinant; pass ``precision_factors`` (and no sparse Cholesky
    solver) to sample matrix-free by CG instead (``solver``, if given, does
    that solve). Graph precisions come factored — ``Bᵀ diag(w) B`` from an
    incidence matrix, proper CAR, Leroux — so a 10⁶-node field samples with
    no factorisation at all.

    Requires the ``numpyro`` optional extra. ``loc`` is a single field
    (``batch_shape == ()``); ``log_prob`` takes any leading sample axes.

    Args:
        loc: Mean $\mu$, shape ``(N,)``.
        precision: $Q$, a symmetric positive-definite operator of size ``N``.
        precision_factors: Operators $F_k$ with $Q = \sum_k F_k^\top F_k$,
            for the perturbation-optimisation sampler.
        solver: Solve (and log-determinant) strategy, e.g.
            `SparseCholeskySolver` or `CGSolver`.
        logdet_strategy: Log-determinant strategy; takes precedence over
            ``solver`` for ``log|Q|`` (e.g. `SLQLogdet` on a huge field).
        log_det_precision: A known $\log|Q|$, which overrides both (callers
            with a closed form, such as a proper CAR, skip a stochastic
            estimate inside an MCMC log-density).
        validate_args: Whether to validate input arguments.

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np
        import gaussx as gx

        # A proper field on the path graph 0 - 1 - 2 - 3 - 4: Q = R + I
        n = 5
        Q = gx.SparseOperator.from_coo(
            np.r_[np.arange(n), np.arange(1, n)],
            np.r_[np.arange(n), np.arange(n - 1)],
            jnp.r_[jnp.array([2.0, 3.0, 3.0, 3.0, 2.0]), -jnp.ones(n - 1)],
            (n, n),
            symmetric=True,
        )
        prior = gx.GaussianMRF(jnp.zeros(n), Q)  # sparse Cholesky throughout
        x = prior.sample(jax.random.PRNGKey(0), (3,))  # (3, 5)
        lp = prior.log_prob(x)  # (3,)
        sd = jnp.sqrt(prior.marginal_variances())  # Takahashi

        # Observe nodes 0 and 3 with noise sd 0.1: the posterior is a GMRF
        A = gx.SparseOperator.from_coo([0, 1], [0, 3], jnp.ones(2), (2, n))
        post = prior.condition_on_observations(A, 1 / 0.1**2, jnp.array([1.0, -1.0]))

        # A sum-to-zero field, by kriging
        zero_sum = prior.condition_on_constraints(jnp.ones((1, n)), jnp.zeros(1))
        u = zero_sum.sample(jax.random.PRNGKey(1))  # u.sum() == 0
        ```
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
        """``log|Q|``: ``log_det_precision``, else the strategy, else `logdet`.

        Returns:
            Scalar log-determinant of the precision.
        """
        if self.log_det_precision is not None:
            return self.log_det_precision
        if self.logdet_strategy is not None:
            return self.logdet_strategy.logdet(self.precision)
        return dispatch_logdet(
            self.precision, _solve_strategy(self.precision, self.solver)
        )

    # -- density --------------------------------------------------------------

    @validate_sample
    def log_prob(self, value: Float[Array, "*sample N"]) -> Float[Array, "*sample"]:
        r"""$\tfrac12\log|Q| - \tfrac12(x-\mu)^\top Q(x-\mu) - \tfrac N2\log 2\pi$.

        Args:
            value: Points, shape ``(..., N)``.

        Returns:
            Log-densities, shape ``(...)``.
        """
        n = self.event_shape[0]
        quad = _quadratic(self.precision, value - self.loc)
        return 0.5 * (self._log_det() - quad - n * _LOG_2PI)

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample N"]:
        """Exact draws through the structural sampling dispatch.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            sample_shape: Leading sample shape.

        Returns:
            Draws, shape ``sample_shape + (N,)``.
        """
        size, draw = _precision_sampler(
            self.precision, self.solver, self.precision_factors
        )
        draws = _draw_samples(key, sample_shape, size, draw, self.loc.dtype)
        return self.loc + draws

    def marginal_variances(self) -> Float[Array, " N"]:
        """``diag(Q⁻¹)`` through `diag_inv`'s structural dispatch.

        Returns:
            Marginal variances, shape ``(N,)``.
        """
        strategy = _solve_strategy(self.precision, self.solver)
        return diag_inv(self.precision, solver=strategy)

    @lazy_property
    def mean(self) -> Float[Array, " N"]:
        return self.loc

    @lazy_property
    def variance(self) -> Float[Array, " N"]:
        return self.marginal_variances()

    # -- conditioning ---------------------------------------------------------

    def condition_on_observations(
        self,
        A: lx.AbstractLinearOperator | Float[ArrayLike, "M N"],
        noise_precision: Float[ArrayLike, ""] | Float[ArrayLike, " M"],
        y: Float[ArrayLike, " M"],
    ) -> GaussianMRF:
        r"""The posterior given $y = Ax + \varepsilon$ with noise precision $\Lambda$.

        $Q_\text{post} = Q + A^\top\Lambda A$ and
        $\mu_\text{post} = \mu + Q_\text{post}^{-1}A^\top\Lambda(y - A\mu)$,
        one solve. The posterior precision keeps what structure it can:

        - a `SparseOperator` prior observed through a `SparseOperator` ``A``
          stays sparse, on the union of the two patterns
          (`SparseOperator.congruence`, planned once per pair of patterns),
          so it keeps the sparse Cholesky path;
        - a dense prior stays dense;
        - anything else (a grid `SpectralFunction`, a `KroneckerSum`, a
          `BlockTriDiag` observed at scattered points) becomes the
          matrix-free sum ``Q + (Λ^{1/2}A)ᵀ(Λ^{1/2}A)``: solves by `solve`
          (CG when large), and draws by perturbation-optimisation with the
          factors ``(Q^{1/2}, Λ^{1/2}A)``, ``Q^{1/2}`` the symmetric root of
          a `SpectralFunction` or `KroneckerSum` prior.

        ``precision_factors`` gain ``Λ^{1/2}A``; a known ``log_det_precision``
        is dropped.

        Args:
            A: Observation operator, shape ``(M, N)`` (array or operator).
            noise_precision: $\Lambda$'s diagonal, a scalar or shape ``(M,)``.
            y: Observations, shape ``(M,)``.

        Returns:
            The posterior `GaussianMRF`, with the same solver strategies (a
            `SparseCholeskySolver` is dropped if the posterior is not sparse).
        """
        A = _as_operator(A)
        m = A.out_size()
        dtype = self.loc.dtype
        w = jnp.broadcast_to(jnp.asarray(noise_precision, dtype=dtype), (m,))
        root_A = lx.DiagonalLinearOperator(jnp.sqrt(w)) @ A
        prior = _unwrap(self.precision)
        factors = self.precision_factors
        if isinstance(prior, SparseOperator) and isinstance(A, SparseOperator):
            q_post: lx.AbstractLinearOperator = prior.union(
                prior.congruence(A, w), tags=_PSD
            )
        elif isinstance(prior, lx.MatrixLinearOperator):
            dense = A.as_matrix()
            weighted = einx.multiply("k i, k -> k i", dense, w)
            update = einsum(weighted, dense, "k i, k j -> i j")
            q_post = lx.MatrixLinearOperator(prior.matrix + update, _PSD)
        else:
            q_post = lx.TaggedLinearOperator(
                lx.AddLinearOperator(self.precision, root_A.T @ root_A), _PSD
            )
            if factors is None:
                root = _symmetric_root(prior)
                factors = None if root is None else (root,)
        if factors is not None:
            factors = (*factors, root_A)
        solver = self.solver
        if isinstance(solver, SparseCholeskySolver) and not isinstance(
            q_post, SparseOperator
        ):
            solver = None
        residual = jnp.asarray(y, dtype=dtype) - A.mv(self.loc)
        rhs = A.transpose().mv(w * residual)
        loc = self.loc + _solve_fn(q_post, solver)(rhs)
        return GaussianMRF(
            loc, q_post, factors, solver=solver, logdet_strategy=self.logdet_strategy
        )

    def condition_on_constraints(
        self,
        A_c: Float[ArrayLike, "c N"] | Float[ArrayLike, " N"],
        e: Float[ArrayLike, " c"] | Float[ArrayLike, ""],
    ) -> ConstrainedGMRF:
        """Hard linear constraints ``A_c x = e``, by conditioning by kriging.

        Args:
            A_c: Constraint matrix, shape ``(c, N)`` (or ``(N,)`` for one).
            e: Constraint values, shape ``(c,)``.

        Returns:
            The `ConstrainedGMRF` ``x | A_c x = e``.
        """
        return ConstrainedGMRF(self, A_c, e)


# ---------------------------------------------------------------------------
# ConstrainedGMRF
# ---------------------------------------------------------------------------


class ConstrainedGMRF(dist.Distribution):
    r"""A `GaussianMRF` conditioned on hard linear constraints $A_cx = e$.

    Conditioning by kriging (Rue & Held, 2005, §2.3.3): with
    $W = Q^{-1}A_c^\top$ (``c`` solves with one factorisation) and
    $S = A_cW$, an unconstrained draw $x$ maps to the exact constrained draw
    $x^\ast = x - WS^{-1}(A_cx - e)$, so $A_cx^\ast = e$ to machine
    precision. The marginal variances are
    $\operatorname{diag}(Q^{-1}) - \operatorname{diag}(WS^{-1}W^\top)$, and
    the density (Rue & Held, eq. 2.30) is

    $$
    \log\pi(x \mid A_cx = e) = \log\pi(x) - \log\mathcal N(e;\,A_c\mu,\,S)
        - \tfrac12\log|A_cA_c^\top|,
    $$

    a density on the affine subspace $\{A_cx = e\}$ (w.r.t. its Lebesgue
    measure), evaluated at points assumed to satisfy the constraint.

    Built by `GaussianMRF.condition_on_constraints`. Requires ``numpyro``.

    Args:
        base: The unconstrained field.
        constraint_matrix: $A_c$, shape ``(c, N)`` (or ``(N,)``), full row rank.
        constraint_value: $e$, shape ``(c,)``.
        validate_args: Whether to validate input arguments.

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import lineax as lx
        import gaussx as gx

        Q = lx.MatrixLinearOperator(2.0 * jnp.eye(4), lx.positive_semidefinite_tag)
        prior = gx.GaussianMRF(jnp.zeros(4), Q)
        field = prior.condition_on_constraints(jnp.ones(4), 1.0)  # Σ x = 1
        x = field.sample(jax.random.PRNGKey(0), (10,))  # each row sums to 1
        var = field.marginal_variances()  # 0.5 - 0.5 / 4
        ```
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
        """``W = Q⁻¹A_cᵀ`` (rows), ``K = S⁻¹Wᵀ`` (rows) and ``chol(S)``."""
        A = self.constraint_matrix
        W = jax.vmap(_solve_fn(self.base.precision, self.base.solver))(A)
        S = einsum(A, W, "c n, d n -> c d")
        chol_S = jnp.linalg.cholesky(S)
        K = jsl.cho_solve((chol_S, True), W)
        return W, K, chol_S

    def _correct(self, x: Float[Array, "*sample N"], K: Array) -> Array:
        """``x − W S⁻¹ (A_c x − e)``."""
        flat = _flat(x)
        excess = einsum(flat, self.constraint_matrix, "s n, c n -> s c")
        excess = excess - self.constraint_value
        corrected = flat - einsum(excess, K, "s c, c n -> s n")
        return _reshape_samples(corrected, x.shape[:-1])

    @validate_sample
    def log_prob(self, value: Float[Array, "*sample N"]) -> Float[Array, "*sample"]:
        """The constrained density (Rue & Held, eq. 2.30) on ``A_c x = e``.

        Args:
            value: Points satisfying the constraint, shape ``(..., N)``.

        Returns:
            Log-densities, shape ``(...)``.
        """
        A, e = self.constraint_matrix, self.constraint_value
        _, _, chol_S = self._kriging()
        c = A.shape[0]
        residual = e - einsum(A, self.base.loc, "c n, n -> c")
        white = jsl.solve_triangular(chol_S, residual, lower=True)
        log_det_S = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol_S)))
        log_constraint = -0.5 * (
            c * _LOG_2PI + log_det_S + einsum(white, white, "c, c ->")
        )
        gram = einsum(A, A, "c n, d n -> c d")
        log_det_gram = jnp.linalg.slogdet(gram)[1]
        return self.base.log_prob(value) - log_constraint - 0.5 * log_det_gram

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample N"]:
        """Base draws, kriged onto ``A_c x = e``.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            sample_shape: Leading sample shape.

        Returns:
            Draws, shape ``sample_shape + (N,)``.
        """
        _, K, _ = self._kriging()
        return self._correct(self.base.sample(key, sample_shape), K)

    def marginal_variances(self) -> Float[Array, " N"]:
        """``diag(Q⁻¹) − diag(W S⁻¹ Wᵀ)``.

        Returns:
            Constrained marginal variances, shape ``(N,)``.
        """
        W, K, _ = self._kriging()
        correction = reduce(W * K, "c n -> n", "sum")
        return self.base.marginal_variances() - correction

    @lazy_property
    def mean(self) -> Float[Array, " N"]:
        _, K, _ = self._kriging()
        return self._correct(self.base.loc, K)

    @lazy_property
    def variance(self) -> Float[Array, " N"]:
        return self.marginal_variances()


# ---------------------------------------------------------------------------
# IntrinsicGMRF
# ---------------------------------------------------------------------------


class IntrinsicGMRF(dist.Distribution):
    r"""Improper $\mathcal N(\mu, (\tau R)^+)$ on the complement of $\ker R$.

    $R$ is a positive-semidefinite structure matrix (RW1, RW2, Besag / ICAR,
    a grid Laplacian) with null space spanned by the columns of
    ``null_space`` (``c`` of them), and $\tau$ is the precision scale. The
    log-density is

    $$
    \log\pi(x) = \tfrac{N-c}{2}\log\tau - \tfrac\tau2(x-\mu)^\top R(x-\mu)
        \;\Big[+\ \tfrac12\log|R|_+ - \tfrac{N-c}{2}\log 2\pi\Big],
    $$

    the bracket only with ``include_normalizer`` (it is $\tau$-free, so an
    MCMC target can skip it; `pseudo_logdet` computes it per call). The
    $\tau$ term is always included, which is what makes $\tau$ identifiable
    under a hyperprior. ``N − c`` must be ``rank(R)``.

    ``constraint`` chooses what happens on $\ker R$:

    - ``"hard"`` (default, what INLA needs): the field lives on
      $\{V^\top x = 0\}$, $V$ = ``null_space``. The log-density above (with
      the normaliser) is then exactly the constrained density on that
      subspace. Samples are kriged onto it: R-INLA's recipe of factoring
      $\tau R + \varepsilon I$ (with $\varepsilon = \sqrt{\text{eps}}$ times
      the mean diagonal) and kriging against $V^\top x = 0$ reduces, since
      $(\tau R + \varepsilon I)^{-1}V = V/\varepsilon$, to the orthogonal
      projection $x - V(V^\top V)^{-1}V^\top x$, which is what is applied
      (exactly, with no extra solves). Marginal variances are kriging-
      corrected (`ConstrainedGMRF`) on the shifted factor, or exact from
      the pseudo-inverse on a Kronecker-structured $R$.
    - ``"soft"`` (for NUTS, which cannot sample a hard constraint): adds a
      tight Gaussian $V^\top x \sim \mathcal N(0, s^2 I)$,
      $s$ = ``soft_constraint_scale``, to the density and to the samples.
    - ``"none"``: only $\mu + $ a draw from $\mathcal N(0, (\tau R)^+)$ —
      the density is improper and invariant along $\ker R$.

    The draw from $\mathcal N(0, (\tau R)^+)$ dispatches like `GaussianMRF`:
    Cholesky of $\tau R + \varepsilon I$ (sparse, banded or dense), the
    pseudo-inverse root in the factored eigenbasis (exact, no shift), or
    perturbation-optimisation with ``precision_factors``
    ($R = \sum_k F_k^\top F_k$, so the right-hand side lies in
    $\operatorname{range}(R)$ and CG converges on the singular system); each
    is then projected orthogonally to ``null_space``.

    **Odd-``n`` RW2.** `rw2_structure` with odd ``n`` returns ``n + 1`` rows,
    the last a decoupled unit-precision node. Build the field on all
    ``N = n + 1`` nodes with ``null_space`` zero on the padding row
    (``c = 2``): then ``N − c = n − 1 = rank(R)``, and the padding node is an
    independent $\mathcal N(0, 1/\tau)$ coordinate whose own ½ log τ is the
    one extra in ``(N − c)/2 · log τ``. Sample it with the rest and drop it
    (``x[:n]``).

    Requires the ``numpyro`` optional extra.

    Args:
        loc: $\mu$, shape ``(N,)``.
        precision_scale: $\tau > 0$.
        structure: $R$, symmetric positive semidefinite, size ``N``.
        null_space: A basis of $\ker R$, shape ``(N, c)`` (or ``(N,)``); for a
            graph, one column per connected component (kernellib's
            ``graph_null_space``).
        precision_factors: $F_k$ with $R = \sum_k F_k^\top F_k$ (e.g. the
            weighted incidence matrix), for the CG sampler.
        constraint: ``"hard"``, ``"soft"`` or ``"none"``.
        soft_constraint_scale: $s$ for ``"soft"``.
        include_normalizer: Add the $\tau$-free constant (exact for an
            orthonormal ``null_space`` under ``"soft"``).
        validate_args: Whether to validate input arguments.

    Raises:
        ValueError: For an unknown ``constraint`` or mismatched shapes.

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import gaussx as gx

        # An RW1 trend with a hard sum-to-zero constraint
        n = 50
        R = gx.rw1_structure(n)
        trend = gx.IntrinsicGMRF(jnp.zeros(n), 4.0, R, jnp.ones((n, 1)) / jnp.sqrt(n))
        u = trend.sample(jax.random.PRNGKey(0), (5,))  # rows sum to 0
        lp = trend.log_prob(u)  # includes (n - 1)/2 · log 4

        # RW2 with odd n: one padding node, null space {1, t} zero on it
        R2 = gx.rw2_structure(7)  # 8 x 8
        t = jnp.arange(7.0)
        V = jnp.zeros((8, 2)).at[:7, 0].set(1.0).at[:7, 1].set(t - t.mean())
        smooth = gx.IntrinsicGMRF(jnp.zeros(8), 10.0, R2, V)
        x = smooth.sample(jax.random.PRNGKey(1))[:7]  # drop the padding node
        ```
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
            null_space = rearrange(null_space, "n -> n 1")
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
        """``V (VᵀV)⁻¹``, so that ``P⊥x = x − V (VᵀV)⁻¹ Vᵀx``."""
        V = self.null_space
        gram = einsum(V, V, "n c, n d -> c d")
        return rearrange(
            jnp.linalg.solve(gram, rearrange(V, "n c -> c n")), "c n -> n c"
        )

    def _project(self, x: Float[Array, "*sample N"]) -> Float[Array, "*sample N"]:
        """``x − V (VᵀV)⁻¹ Vᵀx``, orthogonal to ``null_space``."""
        coeffs = einsum(x, self.null_space, "... n, n c -> ... c")
        return x - einsum(coeffs, self._null_dual(), "... c, n c -> ... n")

    def _shifted(self) -> lx.AbstractLinearOperator:
        """``τR + εI`` with ``ε = √eps · mean(diag(τR))``, as R-INLA does."""
        scaled = _scaled(self.structure, self.precision_scale)
        diagonal = diag(scaled)
        eps = float(jnp.finfo(diagonal.dtype).eps) ** 0.5 * jnp.mean(diagonal)
        return _add_ridge(scaled, eps)

    def _eigen(self) -> FactoredEigen | None:
        """``τR``'s factored eigendecomposition, for a Kronecker structure."""
        operator = _unwrap(self.structure)
        if not isinstance(operator, _EIGEN):
            return None
        eigen = factored_eigen(operator)
        if eigen is None:
            return None
        return FactoredEigen(eigen.bases, self.precision_scale * eigen.eigenvalues)

    def _sampler(self) -> tuple[int, _Draw]:
        """``(m, f)`` with ``P⊥ f(z) ~ N(0, (τR)⁺)`` for ``z ~ N(0, I_m)``."""
        n = self.event_shape[0]
        eigen = self._eigen()
        if eigen is not None:
            return n, lambda z: _eigen_draw(eigen, z, pinv=True)
        factors = self.precision_factors
        if factors is not None and not _is_factorable(_unwrap(self.structure)):
            root = jnp.sqrt(self.precision_scale)
            scaled = tuple(root * factor for factor in factors)
            return _perturbation_draw(
                _scaled(self.structure, self.precision_scale), scaled, None
            )
        return _precision_sampler(self._shifted(), None, None)

    # -- density --------------------------------------------------------------

    def _rank(self) -> int:
        return self.event_shape[0] - self.null_space.shape[1]

    def _half_log_pdet(self) -> Float[Array, ""]:
        """``½ log|R|₊``: eigenvalues of a Kronecker sum, else via the null space."""
        operator = _unwrap(self.structure)
        if isinstance(
            operator, KroneckerSum | DiagonalisedOperator | lx.DiagonalLinearOperator
        ):
            return 0.5 * pseudo_logdet(operator)
        return 0.5 * pseudo_logdet(operator, null_space=self.null_space)

    @validate_sample
    def log_prob(self, value: Float[Array, "*sample N"]) -> Float[Array, "*sample"]:
        r"""$\tfrac{N-c}{2}\log\tau - \tfrac\tau2(x-\mu)^\top R(x-\mu)$, and extras.

        Args:
            value: Points, shape ``(..., N)``.

        Returns:
            Log-densities, shape ``(...)``.
        """
        tau = self.precision_scale
        rank = self._rank()
        quad = _quadratic(self.structure, value - self.loc)
        result = 0.5 * rank * jnp.log(tau) - 0.5 * tau * quad
        if self.include_normalizer:
            result = result + self._half_log_pdet() - 0.5 * rank * _LOG_2PI
        if self.constraint == "soft":
            scale = self.soft_constraint_scale
            coeffs = einsum(value, self.null_space, "... n, n c -> ... c")
            result = result - 0.5 * reduce(coeffs**2, "... c -> ...", "sum") / scale**2
            if self.include_normalizer:
                c = self.null_space.shape[1]
                result = result - c * (math.log(scale) + 0.5 * _LOG_2PI)
        return result

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample N"]:
        """Draws orthogonal to ``null_space`` (plus the soft or ``"none"`` part).

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            sample_shape: Leading sample shape.

        Returns:
            Draws, shape ``sample_shape + (N,)``. With ``"hard"``,
            ``null_spaceᵀ x = 0`` to machine precision.
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        key_field, key_null = jax.random.split(key)
        size, draw = self._sampler()
        dtype = self.loc.dtype
        u = _draw_samples(key_field, sample_shape, size, draw, dtype)
        # Twice is enough: a shifted-Cholesky draw carries a ~ε^{-1/2} null
        # component, and one projection leaves eps times that.
        u = self._project(self._project(u))
        if self.constraint == "none":
            return self.loc + u
        x = self._project(self.loc) + u
        if self.constraint == "soft":
            c = self.null_space.shape[1]
            z = jax.random.normal(key_null, (*sample_shape, c), dtype=dtype)
            z = self.soft_constraint_scale * z
            x = x + einsum(z, self._null_dual(), "... c, n c -> ... n")
        return x

    def marginal_variances(self) -> Float[Array, " N"]:
        r"""``diag((τR)⁺)``, plus ``s² diag(V (VᵀV)⁻² Vᵀ)`` for ``"soft"``.

        The pseudo-inverse diagonal is exact from the factor eigenvalues on a
        Kronecker structure, and otherwise the kriging-corrected
        ``diag((τR + εI)⁻¹)`` (R-INLA's constrained marginal variances).

        Returns:
            Marginal variances of the samples, shape ``(N,)``.
        """
        eigen = self._eigen()
        if eigen is not None:
            variances = eigen.diag_inv(pinv=True)
        else:
            shifted = GaussianMRF(jnp.zeros_like(self.loc), self._shifted())
            kriged = shifted.condition_on_constraints(
                rearrange(self.null_space, "n c -> c n"),
                jnp.zeros(self.null_space.shape[1], dtype=self.loc.dtype),
            )
            variances = kriged.marginal_variances()
        if self.constraint == "soft":
            dual = self._null_dual()
            scale = self.soft_constraint_scale
            variances = variances + scale**2 * reduce(dual**2, "n c -> n", "sum")
        return variances

    @lazy_property
    def mean(self) -> Float[Array, " N"]:
        if self.constraint == "none":
            return self.loc
        return self._project(self.loc)

    @lazy_property
    def variance(self) -> Float[Array, " N"]:
        return self.marginal_variances()
