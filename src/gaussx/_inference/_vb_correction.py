r"""Low-rank variational-Bayes mean correction of the Laplace approximation.

Van Niekerk & Rue (2024): keep the Laplace covariance $H^{-1}$ and move the
mean within a $p$-dimensional subspace,
$q_\delta = \mathcal N(\hat x + H^{-1}S\delta,\ H^{-1})$, choosing $\delta$ to
maximise $\mathbb E_q[\log p(y\mid\eta)] - \mathrm{KL}(q\,\|\,\pi)$. Only the
linear predictor's marginal variances $\operatorname{diag}(AH^{-1}A^\top)$
and $p$ solves with the Laplace factor are needed.
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
    """Host plan for ``v_k = Σ_{a,b} A_ka A_kb Σ_ab`` over each row's pairs.

    Returns ``(row, entry_a, entry_b, position)``: the pair's row, its two
    entries in ``A``'s full (row-major) layout, and the position of
    ``Σ_ab`` in the symmetric (lower) ``selected`` pattern. ``None`` if some
    pair is missing from ``selected``; for the ``H = Q + AᵀWA`` of
    `laplace_mode` none is, since each row of ``A`` is a clique of ``AᵀWA``.
    """
    rows, cols, _ = projector._full
    m = projector.shape[0]
    counts = np.bincount(rows, minlength=m)
    starts = np.cumsum(counts) - counts
    slot = np.arange(rows.shape[0]) - starts[rows]
    table = np.full((m, max(int(counts.max(initial=0)), 1)), -1)
    table[rows, slot] = np.arange(rows.shape[0])
    width = table.shape[1]
    first = np.repeat(np.arange(width), width)
    second = np.tile(np.arange(width), width)
    ea, eb = table[:, first], table[:, second]
    keep = (ea >= 0) & (eb >= 0)
    row = np.nonzero(keep)[0]
    ea, eb = ea[keep], eb[keep]
    i = np.maximum(cols[ea], cols[eb]).astype(np.int64)
    j = np.minimum(cols[ea], cols[eb]).astype(np.int64)
    n = selected.shape[1]
    keys = selected.rows.astype(np.int64) * n + selected.cols
    wanted = i * n + j
    position = np.minimum(np.searchsorted(keys, wanted), keys.shape[0] - 1)
    if not np.array_equal(keys[position], wanted):
        return None
    as_int = ft.partial(np.asarray, dtype=np.int32)
    return as_int(row), as_int(ea), as_int(eb), as_int(position)


def _diag_inverse(H: lx.AbstractLinearOperator, factor: Any) -> Array | None:
    """``diag(H⁻¹)`` by Takahashi or a structured selected inverse, if any."""
    if isinstance(factor, SparseCholeskyFactor):
        return factor.diag_inv()
    if isinstance(H, BlockTriDiag):
        return diag_inv(H)
    if isinstance(H, lx.DiagonalLinearOperator):
        return 1.0 / H.diagonal
    return None


def _unconstrained_variances(model: _Model, result: LaplaceResult) -> Array:
    """``diag(A H⁻¹ Aᵀ)``, from the selected inverse whenever it suffices."""
    A, H, factor = model.projector, result.hessian, result.factor
    if A is None or _is_selection(A):
        d = _diag_inverse(H, factor)
        if d is not None:
            if A is None:
                return d
            assert isinstance(A, SparseOperator)
            rows, cols, _ = A.pattern._full
            v = jnp.zeros(A.pattern.shape[0], dtype=d.dtype)
            return v.at[rows].set(A._full_values() ** 2 * d[cols])
    elif isinstance(A, SparseOperator) and isinstance(factor, SparseCholeskyFactor):
        plan = _pair_plan(A.pattern, factor.symbolic.inverse_plan[0])
        if plan is not None:
            row, ea, eb, position = plan
            Z = factor.selected_inverse().values
            a = A._full_values()
            terms = a[ea] * a[eb] * Z[position]
            return jax.ops.segment_sum(terms, row, num_segments=A.pattern.shape[0])
    # No selected inverse covers A's rows: one solve per row of A.
    n = result.mode.shape[0]
    rows = jnp.eye(n, dtype=result.mode.dtype) if A is None else A.as_matrix()
    solved = jax.vmap(factor.solve)(rows)
    return einsum(rows, solved, "m n, m n -> m")


# ---------------------------------------------------------------------------
# vb_mean_correction
# ---------------------------------------------------------------------------


def _basis(
    subspace: Int[ArrayLike, " p"] | Float[ArrayLike, "N p"], n: int, dtype: Any
) -> Float[Array, "p N"]:
    """``Sᵀ``: unit rows for an index array, else the dense basis transposed."""
    s = jnp.asarray(subspace)
    if jnp.issubdtype(s.dtype, jnp.integer):
        if s.ndim != 1:
            raise ValueError(f"an index subspace must be 1-D, got shape {s.shape}.")
        return jax.nn.one_hot(s, n, dtype=dtype)
    if s.ndim != 2 or s.shape[0] != n:
        raise ValueError(
            f"a dense subspace must have shape ({n}, p), got {tuple(s.shape)}."
        )
    return rearrange(s.astype(dtype), "n p -> p n")


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
    r"""Low-rank VB correction of the Laplace mean (Van Niekerk & Rue, 2024).

    The Laplace approximation $\mathcal N(\hat x, H^{-1})$ is centred at the
    mode, which under a skewed likelihood (Poisson with small counts,
    Bernoulli) is a biased estimate of the posterior mean. Keeping the
    covariance $H^{-1}$, shift the mean within the span of $S$ (``subspace``,
    $N\times p$),

    $$
    q_\delta = \mathcal N\big(\bar x(\delta),\ H^{-1}\big),\qquad
    \bar x(\delta) = \hat x + H^{-1}S\delta,
    $$

    and choose $\delta$ to maximise the evidence lower bound, whose
    covariance terms do not depend on $\delta$:

    $$
    \mathcal L(\delta) = \sum_i\mathbb E_{\eta_i\sim\mathcal N(m_i(\delta),\,v_i)}
        \big[\log p(y_i\mid\eta_i)\big]
        - \tfrac12\big(\bar x(\delta)-\mu\big)^\top Q\big(\bar x(\delta)-\mu\big),
    $$

    with $m(\delta) = A\bar x(\delta) + o$ and $v_i = [AH^{-1}A^\top]_{ii}$.
    With $U = H^{-1}S$, $B = AU$ and the site expectations
    $\bar g_i = \mathbb E[\partial_\eta\log p(y_i\mid\eta_i)]$,
    $\bar h_i = \mathbb E[\partial^2_\eta\log p(y_i\mid\eta_i)]$ (1-D
    quadrature by ``integrator`` on each site), ``n_iter`` Newton steps
    on the $p$-dimensional $\delta$ (from $\delta = 0$) use

    $$
    \nabla\mathcal L = B^\top\bar g - U^\top Q(\bar x-\mu),\qquad
    \nabla^2\mathcal L = B^\top\operatorname{diag}(\bar h)B - U^\top QU.
    $$

    For a Gaussian likelihood the mode is the exact posterior mean and the
    correction is zero.

    **Cost.** $p$ solves with ``result.factor`` (plus $c$ for a hard
    constraint), then $p$-dimensional Newton steps whose site expectations
    are ``order × M`` likelihood derivatives. The predictor variances come
    from the selected inverse, not from solves: for an identity or
    row-selection projector they are $A_{ij}^2\Sigma_{jj}$ (a
    `BlockTriDiag`, sparse or diagonal $H$), and for a general sparse
    projector (a FEM projector, fixed-effect columns) every pair of nodes
    in a row of $A$ is coupled in $A^\top WA$, so each $\Sigma_{ab}$ needed
    lies in Takahashi's ``pattern(L + Lᵀ)``. A dense $A$ or $H$ costs one
    solve per row of $A$. The Newton step is the minimum-norm one, so
    directions of $S$ that do not move $\bar x$ (dependent columns, or
    ones in the constrained-out span of $V$) are harmless.

    **Intrinsic priors.** With ``constraint="hard"`` both the covariance
    and the shift are those of the constrained approximation, kriged onto
    $V^\top x = 0$: $\Sigma = H^{-1} - H^{-1}V(V^\top H^{-1}V)^{-1}V^\top H^{-1}$
    and $\bar x(\delta) = \hat x + \Sigma S\delta$.

    **Gradients.** The result is an ordinary JAX function of the arrays in
    ``result``, ``prior``, ``likelihood`` and ``projector``, so it can be
    differentiated in reverse mode (through ``laplace_mode``'s implicit
    mode and the factor's solves); ``n_iter`` fixed Newton steps are
    differentiated as unrolled.

    Args:
        result: `laplace_mode`'s output for the same model.
        prior: The latent field passed to `laplace_mode`.
        likelihood: The site-factorising likelihood passed to `laplace_mode`
            (it holds $y$).
        subspace: The span of the mean shift: an integer index array
            (``S`` is those columns of the identity, e.g. the fixed effects'
            nodes) or a dense basis of shape ``(N, p)``.
        projector: $A$ as passed to `laplace_mode`; ``None`` is the identity.
        offset: $o$ as passed to `laplace_mode`.
        n_iter: Newton steps on $\delta$.
        integrator: A point-based 1-D rule for the site expectations; its
            points and weights for $\mathcal N(0, 1)$ are reused for every
            site.

    Returns:
        The corrected mean $\bar x(\delta)$, shape ``(N,)``.

    Raises:
        ValueError: If ``subspace`` has the wrong shape.

    Examples:
        ```python
        import jax.numpy as jnp
        import numpy as np
        import gaussx as gx

        # Rare detections along a transect: logit p_i = β₀ + u_i, u an RW2
        # under Σu = Σ(t − t̄)u = 0 and β₀ ~ N(0, 10³), latent x = (u, β₀).
        n = 40
        t = np.arange(n, dtype=float)
        p_true = 1 / (1 + np.exp(2 - np.sin(t / 5)))
        detected = jnp.asarray(np.random.default_rng(0).random(n) < p_true, float)
        R = np.asarray(gx.rw2_structure(n).as_matrix())
        rows, cols = np.nonzero(np.tril(R))
        Q = gx.SparseOperator.from_coo(
            np.r_[rows, n], np.r_[cols, n], jnp.r_[2.0 * R[rows, cols], 1e-3],
            (n + 1, n + 1), symmetric=True,
        )
        V = jnp.zeros((n + 1, 2)).at[:n, 0].set(1.0).at[:n, 1].set(t - t.mean())
        prior = gx.IntrinsicGMRF(jnp.zeros(n + 1), 1.0, Q, V)
        A = gx.SparseOperator.from_coo(  # η = u + β₀
            np.r_[np.arange(n), np.arange(n)], np.r_[np.arange(n), np.full(n, n)],
            jnp.ones(2 * n), (n, n + 1),
        )
        lik = gx.BernoulliLikelihood(detected)

        res = gx.laplace_mode(prior, lik, projector=A)
        fixed_effect_idx = jnp.array([n])
        mean_vb = gx.vb_mean_correction(
            res, prior, lik, projector=A, subspace=fixed_effect_idx
        )
        ```
    """
    model = _model(prior, likelihood, projector, offset)
    St = _basis(subspace, result.mode.shape[0], result.mode.dtype)
    return _correct(model, result, St, n_iter, integrator)


@eqx.filter_jit
def _correct(
    model: _Model,
    result: LaplaceResult,
    St: Float[Array, "p N"],
    n_iter: int,
    integrator: AbstractIntegrator,
) -> Float[Array, " N"]:
    """``x̂ + Σ S δ`` after ``n_iter`` Newton steps on the VB objective."""
    mode, factor = result.mode, result.factor
    dtype = mode.dtype

    # Constrained solves Σb = H⁻¹b − W C⁻¹ Vᵀ H⁻¹b, W = H⁻¹V, C = VᵀW.
    variances = _unconstrained_variances(model, result)
    if model.null_space is None:
        cov_solve = factor.solve
    else:
        Vt = rearrange(model.null_space, "n c -> c n")
        Wt = jax.vmap(factor.solve)(Vt)
        C = einsum(Vt, Wt, "c n, d n -> c d")

        def cov_solve(b: Array) -> Array:
            u = factor.solve(b)
            coeffs = jnp.linalg.solve(C, einsum(Vt, u, "c n, n -> c"))
            return u - einsum(coeffs, Wt, "c, c n -> n")

        AW = jax.vmap(model.project)(Wt)
        reduction = einsum(AW, jnp.linalg.solve(C, AW), "c m, c m -> m")
        variances = variances - reduction

    U = jax.vmap(cov_solve)(St)  # (p, N)
    B = jax.vmap(model.project)(U)  # (p, M)
    QU = jax.vmap(model.precision.mv)(U)
    UQU = einsum(U, QU, "k n, l n -> k l")
    scale = jnp.sqrt(jnp.maximum(variances, 0.0))
    eta_hat = model.project(mode) + model.offset

    standard = GaussianState(
        jnp.zeros(1, dtype=dtype),
        lx.IdentityLinearOperator(jax.ShapeDtypeStruct((1,), dtype)),
    )
    points, weights, _ = integrator.points_and_weights(standard)
    z = rearrange(points, "q 1 -> q")

    def step(_: int, delta: Array) -> Array:
        x = mode + einsum(delta, U, "k, k n -> n")
        m = eta_hat + einsum(delta, B, "k, k m -> m")
        eta = einx.add("m, q m -> q m", m, einx.multiply("m, q -> q m", scale, z))
        g, h = jax.vmap(model.likelihood.site_derivatives)(eta)
        g_bar = einsum(weights, g, "q, q m -> m")
        h_bar = einsum(weights, h, "q, q m -> m")
        grad = einsum(B, g_bar, "k m, m -> k") - einsum(
            QU, x - model.loc, "k n, n -> k"
        )
        curv = einsum(B, einx.multiply("l m, m -> l m", B, h_bar), "k m, l m -> k l")
        # Min-norm step: Sδ in span(V) (a hard constraint) or dependent
        # columns of S leave x̄ unchanged and make the matrix singular.
        return delta - jnp.linalg.pinv(curv - UQU, hermitian=True) @ grad

    delta = jax.lax.fori_loop(0, n_iter, step, jnp.zeros(U.shape[0], dtype=dtype))
    return mode + einsum(delta, U, "k, k n -> n")
