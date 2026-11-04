"""Low-rank variational-Bayes mean correction of the Laplace approximation.

Van Niekerk & Rue (2024): keep the Laplace covariance Λ_x|y⁻¹ and move the mean
within a p-dimensional subspace, q_δ = 𝒩(x̂ + Λ_x|y⁻¹ S δ, Λ_x|y⁻¹), choosing δ to
maximise 𝔼_q[log p(y ∣ η)] − KL(q ‖ π). Only the linear predictor's marginal
variances diag(A Λ_x|y⁻¹ Aᵀ) and p solves with the Laplace factor are needed.
"""

from __future__ import annotations

import functools as ft
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int

from gaussx._distributions._gmrf import GaussianMRF, IntrinsicGMRF
from gaussx._einx import einsum, rearrange
from gaussx._inference._laplace import LaplaceResult, _is_selection, _Model, _model
from gaussx._linalg._diag_inv import diag_inv
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._operators._sparse import SparseOperator, SparsityPattern
from gaussx._quadrature._gauss_hermite import GaussHermiteIntegrator
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._likelihood import AbstractLikelihood
from gaussx._quadrature._types import GaussianState
from gaussx._sparse._factor import SparseCholeskyFactor


_GAUSS_HERMITE_20 = GaussHermiteIntegrator(order=20)


# ---------------------------------------------------------------------------
# Predictor variances diag(A Σ Aᵀ)
# ---------------------------------------------------------------------------


@ft.lru_cache(maxsize=64)
def _pair_plan(
    projector: SparsityPattern, selected: SparsityPattern
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Host plan for the predictor variances over each row's pairs of entries.

        v_m = ∑_a,b [A]_m,a [A]_m,b [Σ]_a,b

    Args:
        projector: Sparsity pattern of A.
        selected: Symmetric (lower) pattern of the selected inverse Σ.

    Returns:
        ``(row, entry_a, entry_b, position)``: the pair's row, its two entries in
        A's full (row-major) layout, and the position of [Σ]_a,b in ``selected``.
        ``None`` if some pair is missing from ``selected``; for the
        Λ_x|y = Λ_x + Aᵀ Λ_site A of `laplace_mode` none is, since each row of A
        is a clique of Aᵀ Λ_site A.
    """
    rows, cols, _ = projector._full
    num_rows = projector.shape[0]
    counts = np.bincount(rows, minlength=num_rows)
    starts = np.cumsum(counts) - counts
    slot = np.arange(rows.shape[0]) - starts[rows]
    table = np.full((num_rows, max(int(counts.max(initial=0)), 1)), -1)
    table[rows, slot] = np.arange(rows.shape[0])
    width = table.shape[1]
    first = np.repeat(np.arange(width), width)
    second = np.tile(np.arange(width), width)
    entry_a, entry_b = table[:, first], table[:, second]
    keep = (entry_a >= 0) & (entry_b >= 0)
    row = np.nonzero(keep)[0]
    entry_a, entry_b = entry_a[keep], entry_b[keep]
    upper_index = np.maximum(cols[entry_a], cols[entry_b]).astype(np.int64)
    lower_index = np.minimum(cols[entry_a], cols[entry_b]).astype(np.int64)
    num_cols = selected.shape[1]
    keys = selected.rows.astype(np.int64) * num_cols + selected.cols
    wanted = upper_index * num_cols + lower_index
    position = np.minimum(np.searchsorted(keys, wanted), keys.shape[0] - 1)
    if not np.array_equal(keys[position], wanted):
        return None
    as_int = ft.partial(np.asarray, dtype=np.int32)
    return as_int(row), as_int(entry_a), as_int(entry_b), as_int(position)


def _diag_inverse(hessian: lx.AbstractLinearOperator, factor: Any) -> Array | None:
    """Diagonal of Λ_x|y⁻¹ by Takahashi or a structured selected inverse, if any.

    Args:
        hessian: The operator Λ_x|y.
        factor: Its factorisation.

    Returns:
        diag(Λ_x|y⁻¹), or ``None`` when no selected inverse covers the class.
    """
    if isinstance(factor, SparseCholeskyFactor):
        return factor.diag_inv()
    if isinstance(hessian, BlockTriDiag):
        return diag_inv(hessian)
    if isinstance(hessian, lx.DiagonalLinearOperator):
        return 1.0 / hessian.diagonal
    return None


def _unconstrained_variances(model: _Model, result: LaplaceResult) -> Array:
    """Predictor variances diag(A Λ_x|y⁻¹ Aᵀ), from a selected inverse if it suffices.

    Args:
        model: The model, which holds A.
        result: The Laplace fit, which holds Λ_x|y and its factor.

    Returns:
        The marginal variance of each site of the linear predictor.
    """
    projector, hessian, factor = model.projector, result.hessian, result.factor
    if projector is None or _is_selection(projector):
        # diag(Λ_x|y⁻¹)
        marginals = _diag_inverse(hessian, factor)
        if marginals is not None:
            if projector is None:
                return marginals
            assert isinstance(projector, SparseOperator)
            rows, cols, _ = projector.pattern._full
            variances = jnp.zeros(projector.pattern.shape[0], dtype=marginals.dtype)
            # v_m = [A]²_m,d [Λ_x|y⁻¹]_d,d
            return variances.at[rows].set(
                projector._full_values() ** 2 * marginals[cols]
            )
    elif isinstance(projector, SparseOperator) and isinstance(
        factor, SparseCholeskyFactor
    ):
        plan = _pair_plan(projector.pattern, factor.symbolic.inverse_plan[0])
        if plan is not None:
            row, entry_a, entry_b, position = plan
            selected_inverse = factor.selected_inverse().values
            values = projector._full_values()
            # [A]_m,a [A]_m,b [Σ]_a,b, one term per pair
            terms = values[entry_a] * values[entry_b] * selected_inverse[position]
            # v_m = ∑_a,b [A]_m,a [A]_m,b [Σ]_a,b
            return jax.ops.segment_sum(
                terms, row, num_segments=projector.pattern.shape[0]
            )
    # No selected inverse covers A's rows: one solve per row of A.
    state_size = result.mode.shape[0]
    if projector is None:
        rows_dense = jnp.eye(state_size, dtype=result.mode.dtype)
    else:
        rows_dense = projector.as_matrix()
    # Λ_x|y⁻¹ Aᵀ, one row per site
    solved = jax.vmap(factor.solve)(rows_dense)
    # v_m = [A]_m,· [Λ_x|y⁻¹ Aᵀ]_·,m
    return einsum(rows_dense, solved, "obs state, obs state -> obs")


# ---------------------------------------------------------------------------
# vb_mean_correction
# ---------------------------------------------------------------------------


def _basis(
    subspace: Int[ArrayLike, " p"] | Float[ArrayLike, "N p"],
    state_size: int,
    dtype: Any,
) -> Float[Array, "p N"]:
    """Subspace basis Sᵀ: unit rows for an index array, else the basis transposed.

    Args:
        subspace: Integer indices of the columns of I, or a dense basis S.
        state_size: Length D_x of the latent field.
        dtype: Floating dtype of the result.

    Returns:
        Sᵀ, one row per direction.

    Raises:
        ValueError: If ``subspace`` has the wrong shape.
    """
    basis = jnp.asarray(subspace)
    if jnp.issubdtype(basis.dtype, jnp.integer):
        if basis.ndim != 1:
            raise ValueError(f"an index subspace must be 1-D, got shape {basis.shape}.")
        return jax.nn.one_hot(basis, state_size, dtype=dtype)
    if basis.ndim != 2 or basis.shape[0] != state_size:
        raise ValueError(
            f"a dense subspace must have shape ({state_size}, p), "
            f"got {tuple(basis.shape)}."
        )
    return rearrange(basis.astype(dtype), "state dir -> dir state")


def vb_mean_correction(
    result: LaplaceResult,
    prior: GaussianMRF | IntrinsicGMRF,
    likelihood: AbstractLikelihood,
    *,
    subspace: Int[ArrayLike, " p"] | Float[ArrayLike, "N p"],
    projector: lx.AbstractLinearOperator | None = None,
    offset: Float[ArrayLike, " M"] | Float[ArrayLike, ""] | None = None,
    n_iter: int = 5,
    integrator: AbstractIntegrator = _GAUSS_HERMITE_20,
) -> Float[Array, " N"]:
    """Low-rank VB correction of the Laplace mean (Van Niekerk & Rue, 2024).

        x̄(δ) = x̂ + Σ S δ,   δ⋆ = argmax_δ ELBO(δ)

    The Laplace approximation 𝒩(x̂, Λ_x|y⁻¹) is centred at the mode, which under a
    skewed likelihood (Poisson with small counts, Bernoulli) is a biased estimate of
    the posterior mean. Keeping the covariance Σ = Λ_x|y⁻¹, shift the mean within
    the span of S (``subspace``, D_x × p), so q_δ = 𝒩(x̄(δ), Σ), and choose δ to
    maximise the evidence lower bound, whose covariance terms do not depend on δ:

        ELBO(δ) = ∑ₘ 𝔼_{ηₘ ~ 𝒩(μ_η,m(δ), vₘ)}[log p(yₘ ∣ ηₘ)]
            − ½ (x̄(δ) − μ_x)ᵀ Λ_x (x̄(δ) − μ_x)

    with μ_η(δ) = A x̄(δ) + o and vₘ = [A Σ Aᵀ]ₘₘ. With U = Σ S, B = A U and the site
    expectations ḡₘ = 𝔼[∇_η log p(yₘ ∣ ηₘ)], h̄ₘ = 𝔼[∇²_η log p(yₘ ∣ ηₘ)] (1-D
    quadrature by ``integrator`` on each site), ``n_iter`` Newton steps on the
    p-dimensional δ, from δ = 0, use

        ∇_δ ELBO = Bᵀ ḡ − Uᵀ Λ_x (x̄ − μ_x)
        ∇²_δ ELBO = Bᵀ diag(h̄) B − Uᵀ Λ_x U

    For a Gaussian likelihood the mode is the exact posterior mean and the correction
    is zero.

    **Cost.** p solves with ``result.factor`` (plus D_c for a hard constraint), then
    p-dimensional Newton steps whose site expectations are ``order × M`` likelihood
    derivatives. The predictor variances come from the selected inverse, not from
    solves: for an identity or row-selection projector they are [A]²_m,d [Σ]_d,d (a
    `BlockTriDiag`, sparse or diagonal Λ_x|y), and for a general sparse projector (a
    FEM projector, fixed-effect columns) every pair of nodes in a row of A is coupled
    in Aᵀ Λ_site A, so each [Σ]_a,b needed lies in Takahashi's pattern(L + Lᵀ). A
    dense A or Λ_x|y costs one solve per row of A. The Newton step is the minimum-norm
    one, so directions of S that do not move x̄ (dependent columns, or ones in the
    constrained-out span of V) are harmless.

    **Intrinsic priors.** With ``constraint="hard"`` both the covariance and the shift
    are those of the constrained approximation, kriged onto Vᵀ x = 0:
    Σ = Λ_x|y⁻¹ − Λ_x|y⁻¹ V (Vᵀ Λ_x|y⁻¹ V)⁻¹ Vᵀ Λ_x|y⁻¹ and x̄(δ) = x̂ + Σ S δ.

    **Gradients.** The result is an ordinary JAX function of the arrays in ``result``,
    ``prior``, ``likelihood`` and ``projector``, so it can be differentiated in reverse
    mode (through ``laplace_mode``'s implicit mode and the factor's solves); ``n_iter``
    fixed Newton steps are differentiated as unrolled.

    Algorithm:
        1. Build the model and the basis Sᵀ.
        2. Take the predictor variances and the p solves U = Σ S.
        3. Run ``n_iter`` minimum-norm Newton steps on δ.
        4. Return x̂ + U δ.

    Args:
        result: `laplace_mode`'s output for the same model.
        prior: The latent field passed to `laplace_mode`.
        likelihood: The site-factorising likelihood passed to `laplace_mode`, which
            holds y.
        subspace: The span of the mean shift S: an integer index array (those columns
            of the identity, e.g. the fixed effects' nodes) or a dense basis of shape
            (D_x, p).
        projector: The projector A as passed to `laplace_mode`; ``None`` is the
            identity.
        offset: The offset o as passed to `laplace_mode`.
        n_iter: Newton steps on δ.
        integrator: A point-based 1-D rule for the site expectations; its points and
            weights for 𝒩(0, 1) are reused for every site.

    Returns:
        The corrected mean x̄(δ).

    Raises:
        ValueError: If ``subspace`` has the wrong shape.

    Examples:
        Rare detections along a transect: logit pₙ = β₀ + uₙ, with u an RW2 under
        ∑ uₙ = ∑ (tₙ − t̄) uₙ = 0, β₀ ~ 𝒩(0, 10³) and latent x = (u, β₀):

        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> import gaussx
        >>> n = 20
        >>> times = np.arange(n, dtype=float)
        >>> p_true = 1 / (1 + np.exp(2 - np.sin(times / 5)))
        >>> detected = np.random.default_rng(0).random(n) < p_true
        >>> detected = jnp.asarray(detected, dtype=jnp.zeros(()).dtype)
        >>> structure = np.asarray(gaussx.rw2_structure(n).as_matrix())
        >>> rows, cols = np.nonzero(np.tril(structure))
        >>> values = jnp.r_[2.0 * structure[rows, cols], 1e-3]
        >>> precision = gaussx.SparseOperator.from_coo(
        ...     np.r_[rows, n], np.r_[cols, n], values, (n + 1, n + 1), symmetric=True
        ... )
        >>> null_space = jnp.zeros((n + 1, 2)).at[:n, 0].set(1.0)
        >>> null_space = null_space.at[:n, 1].set(times - times.mean())
        >>> prior = gaussx.IntrinsicGMRF(jnp.zeros(n + 1), 1.0, precision, null_space)
        >>> projector = gaussx.SparseOperator.from_coo(
        ...     np.r_[np.arange(n), np.arange(n)],
        ...     np.r_[np.arange(n), np.full(n, n)],
        ...     jnp.ones(2 * n),
        ...     (n, n + 1),
        ... )
        >>> likelihood = gaussx.BernoulliLikelihood(detected)
        >>> fit = gaussx.laplace_mode(prior, likelihood, projector=projector)
        >>> mean = gaussx.vb_mean_correction(
        ...     fit, prior, likelihood, projector=projector, subspace=jnp.array([n])
        ... )
        >>> mean.shape, bool(jnp.all(jnp.isfinite(mean)))
        ((21,), True)

        A dense subspace must have D_x rows:

        >>> gaussx.vb_mean_correction(
        ...     fit, prior, likelihood, projector=projector, subspace=jnp.ones((2, 1))
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: a dense subspace must have shape (21, p), got (2, 1).
    """
    model = _model(prior, likelihood, projector, offset)
    subspace_rows = _basis(subspace, result.mode.shape[0], result.mode.dtype)
    return _correct(model, result, subspace_rows, n_iter, integrator)


@eqx.filter_jit
def _correct(
    model: _Model,
    result: LaplaceResult,
    subspace_rows: Float[Array, "p N"],
    n_iter: int,
    integrator: AbstractIntegrator,
) -> Float[Array, " N"]:
    """Run ``n_iter`` Newton steps on the VB objective and return x̂ + Σ S δ.

    Algorithm:
        1. Take the predictor variances and, under a hard constraint, subtract the
           kriging reduction and build the constrained solve.
        2. Form U = Σ S, B = A U and Uᵀ Λ_x U.
        3. Get the standard-normal quadrature rule, reused for every site.
        4. Take minimum-norm Newton steps on δ.

    Args:
        model: The model.
        result: The Laplace fit.
        subspace_rows: The basis Sᵀ, one row per direction.
        n_iter: Newton steps on δ.
        integrator: A point-based 1-D quadrature rule.

    Returns:
        The corrected mean x̄(δ).
    """
    mode, factor = result.mode, result.factor
    dtype = mode.dtype

    # --- constrained solves ---
    # Σ b = Λ_x|y⁻¹ b − W S_V⁻¹ Vᵀ Λ_x|y⁻¹ b, W = Λ_x|y⁻¹ V, S_V = Vᵀ W
    variances = _unconstrained_variances(model, result)
    if model.null_space is None:
        cov_solve = factor.solve
    else:
        # Vᵀ
        null_rows = rearrange(model.null_space, "state con -> con state")
        # Wᵀ, one row per constraint
        solved_rows = jax.vmap(factor.solve)(null_rows)
        # S_V = Vᵀ W
        schur = einsum(null_rows, solved_rows, "con state, other state -> con other")

        def cov_solve(rhs: Array) -> Array:
            # Λ_x|y⁻¹ b
            solved = factor.solve(rhs)
            # Vᵀ Λ_x|y⁻¹ b
            projection = einsum(null_rows, solved, "con state, state -> con")
            # S_V⁻¹ Vᵀ Λ_x|y⁻¹ b
            coeffs = jnp.linalg.solve(schur, projection)
            # W S_V⁻¹ Vᵀ Λ_x|y⁻¹ b
            correction = einsum(coeffs, solved_rows, "con, con state -> state")
            return solved - correction

        # (A W)ᵀ, one row per constraint
        projected_rows = jax.vmap(model.project)(solved_rows)
        # S_V⁻¹ (A W)ᵀ
        solved_projected = jnp.linalg.solve(schur, projected_rows)
        # diag(A W S_V⁻¹ Wᵀ Aᵀ)
        reduction = einsum(projected_rows, solved_projected, "con obs, con obs -> obs")
        # vₘ ← vₘ − [A W S_V⁻¹ Wᵀ Aᵀ]ₘₘ
        variances = variances - reduction

    # --- the subspace ---
    # U = Σ S, one row per direction
    shift_rows = jax.vmap(cov_solve)(subspace_rows)
    # B = A U
    predictor_rows = jax.vmap(model.project)(shift_rows)
    # Λ_x U
    precision_rows = jax.vmap(model.precision.mv)(shift_rows)
    # Uᵀ Λ_x U
    prior_curvature = einsum(
        shift_rows, precision_rows, "dir state, other state -> dir other"
    )
    # √vₘ
    scale = jnp.sqrt(jnp.maximum(variances, 0.0))
    # η̂ = A x̂ + o
    predictor_mode = model.project(mode) + model.offset

    # --- standard-normal quadrature ---
    standard = GaussianState(
        jnp.zeros(1, dtype=dtype),
        lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1,), dtype)),
    )
    points, weights, _ = integrator.points_and_weights(standard)
    nodes = rearrange(points, "point 1 -> point")

    def step(_: int, shift: Array) -> Array:
        # x̄ = x̂ + U δ
        mean = mode + einsum(shift, shift_rows, "dir, dir state -> state")
        # μ_η = η̂ + B δ
        predictor_mean = predictor_mode + einsum(
            shift, predictor_rows, "dir, dir obs -> obs"
        )
        # √vₘ z_q
        spread = einx.multiply("obs, point -> point obs", scale, nodes)
        # ηₘ = μ_η,m + √vₘ z_q
        predictor = einx.add("obs, point obs -> point obs", predictor_mean, spread)
        # ∇_η log p and ∇²_η log p at each node
        site_gradients, site_hessians = jax.vmap(model.likelihood.site_derivatives)(
            predictor
        )
        # ḡₘ = ∑_q w_q ∇_η log p(yₘ ∣ η_q,m)
        expected_gradient = einsum(weights, site_gradients, "point, point obs -> obs")
        # h̄ₘ = ∑_q w_q ∇²_η log p(yₘ ∣ η_q,m)
        expected_hessian = einsum(weights, site_hessians, "point, point obs -> obs")
        # Bᵀ ḡ
        data_gradient = einsum(predictor_rows, expected_gradient, "dir obs, obs -> dir")
        # Uᵀ Λ_x (x̄ − μ_x)
        prior_gradient = einsum(
            precision_rows, mean - model.loc, "dir state, state -> dir"
        )
        # ∇_δ ELBO = Bᵀ ḡ − Uᵀ Λ_x (x̄ − μ_x)
        gradient = data_gradient - prior_gradient
        # diag(h̄) B
        weighted_rows = einx.multiply(
            "other obs, obs -> other obs", predictor_rows, expected_hessian
        )
        # Bᵀ diag(h̄) B
        data_curvature = einsum(
            predictor_rows, weighted_rows, "dir obs, other obs -> dir other"
        )
        # Min-norm step: S δ in span(V) (a hard constraint) or dependent
        # columns of S leave x̄ unchanged and make the matrix singular.
        # (∇²_δ ELBO)⁺
        pseudo_inverse = jnp.linalg.pinv(
            data_curvature - prior_curvature, hermitian=True
        )
        # δ ← δ − (∇²_δ ELBO)⁺ ∇_δ ELBO
        return shift - pseudo_inverse @ gradient

    initial_shift = jnp.zeros(shift_rows.shape[0], dtype=dtype)
    shift = jax.lax.fori_loop(0, n_iter, step, initial_shift)
    # x̄ = x̂ + U δ
    return mode + einsum(shift, shift_rows, "dir, dir state -> state")
