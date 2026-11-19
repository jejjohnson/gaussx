"""Precision-form Laplace approximation, one Newton step of the INLA inner loop.

With the linear predictor η = A x + o, a likelihood that factorises over sites,
g = ∇_η log p(y ∣ η) and the site precision Λ_site = −diag(∇²_η log p(y ∣ η)), the
cost 𝒥(x) = −log p(y ∣ A x + o) + ½ (x − μ_x)ᵀ Λ_x (x − μ_x) has gradient
−Aᵀ g + Λ_x (x − μ_x) and Hessian Λ_x|y = Λ_x + Aᵀ Λ_site A, so a Newton step solves

    Λ_x|y x⁽ʲ⁺¹⁾ = Λ_x μ_x + Aᵀ (g + Λ_site A x⁽ʲ⁾)

with a matrix whose pattern never changes. gaussx holds the step (`laplace_model`,
`LaplaceModel.newton`) and the result at a mode (`laplace_result`); the loop that
iterates the step and differentiates the mode implicitly is
``argminax.solve.laplace_mode``. The Laplace log-marginal differentiates exactly
through the structured or sparse log-determinants.
"""

from __future__ import annotations

import math
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from gaussx._distributions._gmrf import GaussianMRF, IntrinsicGMRF, _scaled, _unwrap
from gaussx._einx import einsum, rearrange
from gaussx._inference._inference import newton_update
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._sparse import SparseOperator
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._diag import diag
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve
from gaussx._quadrature._likelihood import AbstractLikelihood
from gaussx._strategies._sparse_cholesky import SparseCholeskySolver


_LOG_2PI = math.log(2.0 * math.pi)
_PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
# ``newton_update``'s default: keeps Λ_site positive for non-log-concave sites.
_PRECISION_FLOOR = 1e-6


class LaplaceResult(eqx.Module):
    """The Gaussian approximation 𝒩(x̂, Λ_x|y⁻¹) at the mode.

    Attributes:
        mode: The mode x̂, implicitly differentiable in every array the prior and
            likelihood hold (θ).
        hessian: Λ_x|y = Λ_x + Aᵀ Λ_site A at the mode: a `BlockTriDiag`,
            `SparseOperator`, diagonal or dense operator like Λ_x where the
            projector allows (see `LaplaceModel`).
        factor: The factorisation of Λ_x|y, with ``solve(b)`` and ``logdet()``: a
            `SparseCholeskyFactor` (whose ``diag_inv`` gives the marginal variances
            by Takahashi) for a sparse Λ_x|y, else the structured Cholesky factor's
            solves.
        log_marginal: The Laplace log-marginal log π̃(y ∣ θ); see `laplace_result`
            for the constants it includes.
        n_iter: Newton steps taken.
        converged: Whether the last step moved x by less than the tolerance.
            Implicit gradients assume it; re-run or discard a θ where it is
            ``False``.
    """

    mode: Float[Array, " N"]
    hessian: lx.AbstractLinearOperator
    factor: Any
    log_marginal: Float[Array, ""]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]


# ---------------------------------------------------------------------------
# Factorisations of Λ_x|y
# ---------------------------------------------------------------------------


class _CholeskyFactor(eqx.Module):
    """Λ_x|y = L Lᵀ with a structured lower factor (block, diagonal or dense)."""

    lower: lx.AbstractLinearOperator

    def solve(self, b: Float[Array, " N"]) -> Float[Array, " N"]:
        # Lᵀ, formed first: the order of the trace fixes reverse-mode accumulation
        upper = self.lower.T
        # L⁻¹ b
        forward = solve(self.lower, b)
        # Λ_x|y⁻¹ b = L⁻ᵀ L⁻¹ b
        return solve(upper, forward)

    def logdet(self) -> Float[Array, ""]:
        # diag(L)
        diagonal = diag(self.lower)
        # log diag(L)
        log_diagonal = jnp.log(diagonal)
        # jnp, not einx: einx cannot resolve an empty factor (D_x = 0).
        # log det Λ_x|y = 2 ∑_d log [L]_dd
        return 2.0 * jnp.sum(log_diagonal)


class _OperatorFactor(eqx.Module):
    """No factor: `gaussx.solve` and `gaussx.logdet` dispatch on Λ_x|y itself."""

    operator: lx.AbstractLinearOperator

    def solve(self, b: Float[Array, " N"]) -> Float[Array, " N"]:
        # Λ_x|y⁻¹ b
        return solve(self.operator, b)

    def logdet(self) -> Float[Array, ""]:
        # log det Λ_x|y
        return logdet(self.operator)


_CHOLESKY = (BlockTriDiag, lx.MatrixLinearOperator, lx.DiagonalLinearOperator)


def _factor(hessian: lx.AbstractLinearOperator) -> Any:
    """Factorise Λ_x|y by the cheapest route its class allows.

    Args:
        hessian: The operator Λ_x|y.

    Returns:
        An object with ``solve(b)`` and ``logdet()``: a `SparseCholeskyFactor`, a
        `_CholeskyFactor`, or an `_OperatorFactor` that dispatches on Λ_x|y.
    """
    if isinstance(hessian, SparseOperator):
        # The symbolic analysis is cached per pattern: every Newton step and
        # every θ reuses it, and only the numeric factorisation runs.
        return SparseCholeskySolver().factor(hessian)
    if isinstance(hessian, _CHOLESKY):
        # Λ_x|y = L Lᵀ
        return _CholeskyFactor(cholesky(hessian))
    return _OperatorFactor(hessian)


# ---------------------------------------------------------------------------
# Λ_x|y = Λ_x + Aᵀ Λ_site A, values only on a fixed structure
# ---------------------------------------------------------------------------


def _is_selection(projector: lx.AbstractLinearOperator | None) -> bool:
    """Whether a sparse A has at most one entry per row, so Aᵀ Λ_site A is diagonal.

    Args:
        projector: The projector A, or ``None``.

    Returns:
        ``True`` for a `SparseOperator` with at most one stored entry per row.
    """
    if not isinstance(projector, SparseOperator):
        return False
    rows, _, _ = projector.pattern._full
    entries_per_row = np.bincount(rows, minlength=projector.pattern.shape[0])
    return bool(np.all(entries_per_row <= 1))


def _selection_diagonal(
    projector: SparseOperator, site_precision: Float[Array, " M"]
) -> Array:
    """Diagonal of Aᵀ Λ_site A for an A with one entry per row.

        [diag(Aᵀ Λ_site A)]_d = ∑ₘ [Λ_site]ₘₘ [A]²_m,d

    Args:
        projector: The selection-shaped projector A.
        site_precision: Diagonal of Λ_site.

    Returns:
        The diagonal of Aᵀ Λ_site A.
    """
    rows, cols, _ = projector.pattern._full
    values = projector._full_values()
    # [Λ_site]ₘₘ [A]²_m,d, one term per stored entry
    terms = site_precision[rows] * values**2
    # ∑ₘ [Λ_site]ₘₘ [A]²_m,d
    return jax.ops.segment_sum(terms, cols, num_segments=projector.pattern.shape[1])


def _add_diagonal(precision: lx.AbstractLinearOperator, diagonal: Float[Array, " N"]):
    """Add a diagonal to Λ_x, keeping a sparse, banded, diagonal or dense structure.

        Λ_x|y = Λ_x + diag(d)

    Args:
        precision: The prior precision Λ_x.
        diagonal: The diagonal d.

    Returns:
        Λ_x + diag(d) as an operator of Λ_x's class where it has one, else a
        PSD-tagged sum.
    """
    if isinstance(precision, SparseOperator):
        return precision.add_diagonal(diagonal, tags=_PSD)
    if isinstance(precision, BlockTriDiag):
        size = precision._block_size
        # d, one row per diagonal block
        blocks = rearrange(diagonal, "(block row) -> block row", row=size)
        eye = jnp.eye(size, dtype=diagonal.dtype)
        # diag(d) as a stack of diagonal blocks
        added = einx.multiply("block row, row col -> block row col", blocks, eye)
        # [Λ_x]_t,t + diag(d)
        new_diagonal = precision.diagonal + added
        return BlockTriDiag(
            new_diagonal,
            precision.sub_diagonal,
            symmetric=precision.symmetric,
            tags=_PSD,
        )
    if isinstance(precision, lx.DiagonalLinearOperator):
        return lx.DiagonalLinearOperator(precision.diagonal + diagonal)
    if isinstance(precision, lx.MatrixLinearOperator):
        return lx.MatrixLinearOperator(precision.matrix + jnp.diag(diagonal), _PSD)
    return lx.TaggedLinearOperator(
        lx.AddLinearOperator(precision, lx.DiagonalLinearOperator(diagonal)), _PSD
    )


def _as_sparse(precision: lx.AbstractLinearOperator) -> SparseOperator | None:
    """A banded or diagonal Λ_x as a symmetric `SparseOperator`, values traced.

    Args:
        precision: The prior precision Λ_x.

    Returns:
        The sparse form of Λ_x, or ``None`` for any other class.
    """
    if isinstance(precision, SparseOperator):
        return precision
    if isinstance(precision, lx.DiagonalLinearOperator):
        size = precision.in_size()
        return SparseOperator.from_coo(
            np.arange(size),
            np.arange(size),
            precision.diagonal,
            (size, size),
            symmetric=True,
        )
    if isinstance(precision, BlockTriDiag) and precision.symmetric:
        num_blocks, block_size = precision._num_blocks, precision._block_size
        # Host index arrays: the lower triangle of each diagonal block, then
        # every entry of each sub-diagonal block. They stay NumPy, so .ravel stays.
        tril_rows, tril_cols = np.tril_indices(block_size)
        start = np.arange(num_blocks) * block_size
        diag_rows = np.add.outer(start, tril_rows).ravel()
        diag_cols = np.add.outer(start, tril_cols).ravel()
        diag_values = rearrange(
            precision.diagonal[:, tril_rows, tril_cols],
            "block entry -> (block entry)",
        )
        sub_row_offsets, sub_col_offsets = np.divmod(
            np.arange(block_size * block_size), block_size
        )
        sub_rows = np.add.outer(start[1:], sub_row_offsets).ravel()
        sub_cols = np.add.outer(start[:-1], sub_col_offsets).ravel()
        sub_values = rearrange(
            precision.sub_diagonal[:, sub_row_offsets, sub_col_offsets],
            "block entry -> (block entry)",
        )
        size = num_blocks * block_size
        return SparseOperator.from_coo(
            np.concatenate([diag_rows, sub_rows]),
            np.concatenate([diag_cols, sub_cols]),
            jnp.concatenate([diag_values, sub_values]),
            (size, size),
            symmetric=True,
        )
    return None


def _hessian(
    precision: lx.AbstractLinearOperator,
    projector: lx.AbstractLinearOperator | None,
    site_precision: Float[Array, " M"],
) -> lx.AbstractLinearOperator:
    """Posterior precision on a structure fixed by (Λ_x, A) alone.

        Λ_x|y = Λ_x + Aᵀ Λ_site A

    Args:
        precision: The prior precision Λ_x.
        projector: The projector A; ``None`` is the identity.
        site_precision: Diagonal of Λ_site.

    Returns:
        The operator Λ_x|y; `LaplaceModel` lists the class each pair gives.
    """
    if projector is None:
        return _add_diagonal(precision, site_precision)
    if _is_selection(projector):
        assert isinstance(projector, SparseOperator)
        return _add_diagonal(precision, _selection_diagonal(projector, site_precision))
    sparse = _as_sparse(precision) if isinstance(projector, SparseOperator) else None
    if sparse is not None:
        assert isinstance(projector, SparseOperator)
        # Aᵀ Λ_site A
        congruence = sparse.congruence(projector, site_precision)
        # Λ_x + Aᵀ Λ_site A, on the union pattern
        return sparse.union(congruence, tags=_PSD)
    if isinstance(precision, lx.MatrixLinearOperator) or not isinstance(
        projector, SparseOperator
    ):
        dense = projector.as_matrix()
        # Λ_site A
        weighted = einx.multiply("obs row, obs -> obs row", dense, site_precision)
        # Aᵀ Λ_site A
        update = einsum(weighted, dense, "obs row, obs col -> row col")
        # Λ_x + Aᵀ Λ_site A
        return lx.MatrixLinearOperator(precision.as_matrix() + update, _PSD)
    # Λ_site^(½) A
    root = lx.DiagonalLinearOperator(jnp.sqrt(site_precision)) @ projector
    # Λ_x + (Λ_site^(½) A)ᵀ (Λ_site^(½) A), matrix-free
    return lx.TaggedLinearOperator(lx.AddLinearOperator(precision, root.T @ root), _PSD)


# ---------------------------------------------------------------------------
# The model pieces
# ---------------------------------------------------------------------------


class LaplaceModel(eqx.Module):
    """The latent Gaussian model of a Laplace fit, with the maps one Newton step needs.

    x ~ 𝒩(μ_x, Λ_x⁻¹), η = A x + o, y ∣ η ~ ``likelihood``; build one with
    `laplace_model`. Its methods are one step's algebra: `newton` is the Newton
    target x̃ (kriged onto Vᵀ x = 0 for a hard constraint), `grad` the ascent
    direction −∇_x 𝒥, and `hessian` the site information and Λ_x|y at x. The loop
    that iterates them to a tolerance lives in the caller
    (``argminax.solve.laplace_mode``).

    Λ_x|y is rebuilt values-only on a structure fixed by Λ_x and A:

    - A the identity (``None``) or one entry per row (a row selection, or a scaled
      one): Aᵀ Λ_site A is diagonal, so a `BlockTriDiag` Λ_x (RW1, RW2, AR(1)) stays
      `BlockTriDiag`, and a `SparseOperator`, diagonal or dense Λ_x keeps its class;
    - a general `SparseOperator` A (a FEM projector, an intercept and covariates):
      `SparseOperator.congruence` on the union pattern, with a banded or diagonal
      Λ_x converted to sparse;
    - a dense A or Λ_x: dense; any other Λ_x (a grid `SpectralFunction` or
      `KroneckerSum` with a non-diagonal update): the matrix-free sum, solved and
      log-determined by `gaussx.solve` and `gaussx.logdet` dispatch.

    Attributes:
        precision: Prior precision Λ_x.
        loc: Prior mean μ_x.
        projector: Projector A; ``None`` is the identity.
        offset: Offset o of the linear predictor.
        likelihood: Site-factorising likelihood, which holds y.
        null_space: Basis V of a hard constraint Vᵀ x = 0, or ``None``.
    """

    precision: lx.AbstractLinearOperator
    loc: Float[Array, " N"]
    projector: lx.AbstractLinearOperator | None
    offset: Float[Array, " M"] | Float[Array, ""]
    likelihood: AbstractLikelihood
    null_space: Float[Array, "N c"] | None

    def project(self, x: Array) -> Array:
        """The linear predictor without its offset, A x."""
        # A x
        return x if self.projector is None else self.projector.mv(x)

    def project_t(self, r: Array) -> Array:
        """The transpose of the projector applied to r, Aᵀ r."""
        # Aᵀ r
        return r if self.projector is None else self.projector.transpose().mv(r)

    def grad(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        """Ascent direction −∇_x 𝒥(x) = Aᵀ g − Λ_x (x − μ_x)."""
        # η = A x + o
        predictor = self.project(x) + self.offset
        # g = ∇_η log p(y ∣ η)
        site_gradient = jax.grad(self.likelihood.log_prob)(predictor)
        # Aᵀ g
        data_term = self.project_t(site_gradient)
        # Λ_x (x − μ_x)
        prior_term = self.precision.mv(x - self.loc)
        return data_term - prior_term

    def hessian(self, x: Float[Array, " N"]) -> tuple[Array, lx.AbstractLinearOperator]:
        """Site information Λμ_site = g + Λ_site A x and Λ_x|y at x."""
        # A x
        projected = self.project(x)
        # g and diag(∇²_η log p(y ∣ η)) at η = A x + o
        site_gradient, site_hessian = self.likelihood.site_derivatives(
            projected + self.offset
        )
        # Λμ_site = g + Λ_site A x, Λ_site = max(−diag(∇²_η log p), tol)
        site_information, site_precision = newton_update(
            projected, site_gradient, site_hessian, precision_floor=_PRECISION_FLOOR
        )
        return site_information, _hessian(
            self.precision, self.projector, site_precision
        )

    # -- the hard constraint Vᵀ x = 0 ---------------------------------------

    def null_component(self, x: Array) -> Array:
        """Component V (Vᵀ V)⁻¹ Vᵀ x in the null space; zero without a constraint."""
        if self.null_space is None:
            return jnp.zeros_like(x)
        null_space = self.null_space
        # Vᵀ V
        gram = einsum(null_space, null_space, "state con, state other -> con other")
        # Vᵀ x
        projection = einsum(null_space, x, "state con, state -> con")
        # (Vᵀ V)⁻¹ Vᵀ x
        coeffs = jnp.linalg.solve(gram, projection)
        # V (Vᵀ V)⁻¹ Vᵀ x
        return einsum(null_space, coeffs, "state con, con -> state")

    def krige(self, factor: Any, z: Array) -> tuple[Array, Array | None]:
        """Condition on Vᵀ x = 0: z − W S⁻¹ Vᵀ z, W = Λ_x|y⁻¹ V, S = Vᵀ W, and S."""
        if self.null_space is None:
            return z, None
        # Vᵀ
        null_rows = rearrange(self.null_space, "state con -> con state")
        # Wᵀ with W = Λ_x|y⁻¹ V, one row per constraint
        solved_rows = jax.vmap(factor.solve)(null_rows)
        # S = Vᵀ Λ_x|y⁻¹ V
        schur = einsum(null_rows, solved_rows, "con state, other state -> con other")
        # Vᵀ z
        projection = einsum(null_rows, z, "con state, state -> con")
        # S⁻¹ Vᵀ z
        coeffs = jnp.linalg.solve(schur, projection)
        # W S⁻¹ Vᵀ z
        correction = einsum(coeffs, solved_rows, "con, con state -> state")
        return z - correction, schur

    def newton(self, x: Float[Array, " N"]) -> Float[Array, " N"]:
        """Constrained Newton target: krige(Λ_x|y⁻¹ (Λ_x μ_x + Aᵀ Λμ_site))."""
        site_information, hessian = self.hessian(x)
        factor = _factor(hessian)
        # Λ_x μ_x
        prior_information = self.precision.mv(self.loc)
        # Λ_x μ_x + Aᵀ Λμ_site
        information = prior_information + self.project_t(site_information)
        # Λ_x|y⁻¹ (Λ_x μ_x + Aᵀ Λμ_site)
        target = factor.solve(information)
        return self.krige(factor, target)[0]


def laplace_model(
    prior: GaussianMRF | IntrinsicGMRF,
    likelihood: AbstractLikelihood,
    *,
    projector: lx.AbstractLinearOperator | None = None,
    offset: ArrayLike | None = None,
) -> LaplaceModel:
    """Collect and check the pieces of the latent Gaussian model of a Laplace fit.

    One Newton step is `LaplaceModel.newton` on this model; a caller that runs the
    loop (``argminax.solve.laplace_mode``) builds the model here and the result
    with `laplace_result`.

    Args:
        prior: The latent field.
        likelihood: The site-factorising likelihood.
        projector: The projector A; ``None`` is the identity.
        offset: The offset o; ``None`` is zero.

    Returns:
        The `LaplaceModel`.

    Examples:
        One Newton step of a Poisson model on a diagonal prior:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> prior = gaussx.GaussianMRF(
        ...     jnp.zeros(3), lx.DiagonalLinearOperator(jnp.ones(3))
        ... )
        >>> model = gaussx.laplace_model(
        ...     prior, gaussx.PoissonLikelihood(jnp.array([0.0, 1.0, 4.0]))
        ... )
        >>> model.newton(jnp.zeros(3)).shape
        (3,)

    Raises:
        NotImplementedError: For an `IntrinsicGMRF` with ``constraint="soft"``.
        TypeError: For another prior class.
        ValueError: If ``projector`` does not have D_x columns.
    """
    if isinstance(prior, GaussianMRF):
        precision = _unwrap(prior.precision)
        null_space = None
    elif isinstance(prior, IntrinsicGMRF):
        if prior.constraint == "soft":
            raise NotImplementedError(
                "laplace_model supports IntrinsicGMRF with constraint='hard' or "
                "'none'; the soft constraint's V Vᵀ / s² term is dense."
            )
        # Λ_x = τ R
        precision = _unwrap(_scaled(prior.structure, prior.precision_scale))
        null_space = prior.null_space if prior.constraint == "hard" else None
    else:
        raise TypeError(
            "laplace_model needs a GaussianMRF or IntrinsicGMRF prior, got "
            f"{type(prior).__name__}."
        )
    state_size = prior.event_shape[0]
    if projector is not None and projector.in_size() != state_size:
        raise ValueError(
            f"projector must have {state_size} columns, got shape "
            f"({projector.out_size()}, {projector.in_size()})."
        )
    dtype = prior.loc.dtype
    offset = jnp.zeros((), dtype=dtype) if offset is None else jnp.asarray(offset)
    return LaplaceModel(precision, prior.loc, projector, offset, likelihood, null_space)


def laplace_result(
    model: LaplaceModel,
    prior: GaussianMRF | IntrinsicGMRF,
    mode: Float[Array, " N"],
    n_iter: Int[ArrayLike, ""],
    converged: Bool[ArrayLike, ""],
) -> LaplaceResult:
    """The Gaussian approximation and the Laplace log-marginal at a mode.

        log π̃(y ∣ θ) = log p(y ∣ A x̂ + o) + log π(x̂) − log π̃_G(x̂)

    The last step of a Laplace fit: refactorise Λ_x|y at the mode x̂ and assemble
    the log-marginal, with the constraint terms of an `IntrinsicGMRF` with
    ``constraint="hard"``. A caller running its own Newton loop on a
    `LaplaceModel` (``argminax.solve.laplace_mode``) builds its result here.

    In the log-marginal, log π is the prior's ``log_prob`` and

        log π̃_G(x̂) = ½ log det Λ_x|y − (D_x / 2) log 2π
            [+ ½ log det(Vᵀ Λ_x|y⁻¹ V) − ½ log det(Vᵀ V) + (D_c / 2) log 2π]

    the bracket for the hard constraint. For a `GaussianMRF` this is
    log p(y ∣ A x̂ + o) − ½ (x̂ − μ_x)ᵀ Λ_x (x̂ − μ_x) + ½ log det Λ_x
    − ½ log det Λ_x|y, the exact log-evidence for a Gaussian likelihood. For an
    `IntrinsicGMRF` it is exact with ``include_normalizer=True`` and otherwise off
    by the θ-free constant ½ log det₊ R − ((D_x − D_c) / 2) log 2π. The likelihood's
    own normalising terms are included.

    Args:
        model: The model, from `laplace_model`.
        prior: The prior the model was built from; its ``log_prob`` enters the
            log-marginal.
        mode: The mode x̂, made implicitly differentiable by the caller if its
            gradients are wanted.
        n_iter: Newton steps taken, recorded in the result.
        converged: Whether the loop converged, recorded in the result.

    Returns:
        The `LaplaceResult` at x̂.

    Examples:
        Run a few Newton steps on a Poisson model, then build the result at the
        mode:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> prior = gaussx.GaussianMRF(
        ...     jnp.zeros(3), lx.DiagonalLinearOperator(jnp.ones(3))
        ... )
        >>> likelihood = gaussx.PoissonLikelihood(jnp.array([0.0, 1.0, 4.0]))
        >>> model = gaussx.laplace_model(prior, likelihood)
        >>> mode = jnp.zeros(3)
        >>> for _ in range(10):
        ...     mode = model.newton(mode)
        >>> fit = gaussx.laplace_result(model, prior, mode, 10, True)
        >>> bool(jnp.max(jnp.abs(model.grad(fit.mode))) < 1e-4)
        True
        >>> type(fit.hessian).__name__, bool(jnp.isfinite(fit.log_marginal))
        ('DiagonalLinearOperator', True)
    """
    _, hessian = model.hessian(mode)
    factor = _factor(hessian)
    # η̂ = A x̂ + o
    predictor = model.project(mode) + model.offset
    # log π̃_G(x̂) = ½ log det Λ_x|y − (D_x / 2) log 2π
    state_size = mode.shape[0]
    log_gaussian = 0.5 * factor.logdet() - 0.5 * state_size * _LOG_2PI
    _, schur = model.krige(factor, mode)
    if schur is not None:
        null_space = model.null_space
        assert null_space is not None
        num_constraints = null_space.shape[1]
        # Vᵀ V
        gram = einsum(null_space, null_space, "state con, state other -> con other")
        # S = L_S L_Sᵀ
        schur_factor = jnp.linalg.cholesky(schur)
        # jnp, not einx: einx cannot resolve an empty constraint (D_c = 0).
        # ½ log det S = ∑_c log [L_S]_cc
        half_log_det_schur = jnp.sum(jnp.log(jnp.diagonal(schur_factor)))
        # + ½ log det S
        log_gaussian = log_gaussian + half_log_det_schur
        # ½ log det(Vᵀ V)
        half_log_det_gram = 0.5 * jnp.linalg.slogdet(gram)[1]
        # − ½ log det(Vᵀ V) + (D_c / 2) log 2π
        log_gaussian = log_gaussian - half_log_det_gram
        log_gaussian = log_gaussian + 0.5 * num_constraints * _LOG_2PI
    # log π̃(y ∣ θ) = log p(y ∣ η̂) + log π(x̂) − log π̃_G(x̂)
    log_marginal = (
        model.likelihood.log_prob(predictor) + prior.log_prob(mode) - log_gaussian
    )
    return LaplaceResult(
        mode, hessian, factor, log_marginal, jnp.asarray(n_iter), jnp.asarray(converged)
    )


# The private names vb_mean_correction (and its tests) import.
_Model = LaplaceModel


def _model(
    prior: GaussianMRF | IntrinsicGMRF,
    likelihood: AbstractLikelihood,
    projector: lx.AbstractLinearOperator | None,
    offset: ArrayLike | None,
) -> LaplaceModel:
    """`laplace_model` with positional arguments, as `vb_mean_correction` calls it."""
    return laplace_model(prior, likelihood, projector=projector, offset=offset)
