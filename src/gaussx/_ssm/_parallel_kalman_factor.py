r"""Square-root (factor-propagating) parallel Kalman filter.

The associative-scan filter of Särkkä & García-Fernández (2021) carries
elements $(A, b, C, \eta, J)$. The square-root form of Yaghoobi, Corenflos,
Hassan & Särkkä (2022, §III) carries $(A, b, U, \eta, Z)$ with $C = UU^\top$
and $J = ZZ^\top$ instead, and builds every element and every combination
from QR decompositions (``tria``) of stacked factors, so no covariance is
formed inside the scan and every returned covariance is a Gram matrix
$UU^\top$, PSD by construction. This is the filter behind
``parallel_kalman_filter(..., square_root=True)`` (gh-454).

``Q``, ``R`` and the initial covariance are factored once, outside the
scan, by a Cholesky of their correlation matrix after a ``4 n ε`` diagonal
shift (`_input_factor`), so each component is perturbed relative to its own
scale and a covariance that is singular or indefinite only by rounding
still factors. The factors are the differentiated path: there is no
projection and no ``stop_gradient``.

Pseudocode:

    L_Q, L_R, N0 = chol-factors of Q_t, R_t, P0        (once, outside the scan)
    element_t = (A, b, U, η, Z) from tria([[H L_Q, L_R], [L_Q, 0]])
    element_0 absorbs N(m0, N0 N0ᵀ)
    (_, m_filt, U_filt, _, _) = associative_scan(combine, elements)
    N_pred,t = tria([F U_filt,t−1, L_Q]);  chol(S_t) = tria([H N_pred, L_R])
    log p(y) = Σ_t log N(y_t | H m_pred, S_t) via chol(S_t)

References:
    Särkkä, S. & García-Fernández, Á. F. (2021). Temporal parallelization
    of Bayesian smoothers. *IEEE Transactions on Automatic Control* 66(1),
    299-306.

    Yaghoobi, F., Corenflos, A., Hassan, S. & Särkkä, S. (2022). Parallel
    square-root statistical linear regression for inference in nonlinear
    state space models. arXiv:2207.00426.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jaxtyping import Array, Bool, Float

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._einx import einsum, rearrange, reduce, repeat
from gaussx._ssm._kalman import FilterState
from gaussx._ssm._utils import _masked_obs_inputs, _normalise_tv_inputs


# ----------------------------------------------------------------
# Triangularisation
# ----------------------------------------------------------------


def _t(X: Float[Array, "a b"]) -> Float[Array, "b a"]:
    return rearrange(X, "a b -> b a")


def _pad_columns(X: Float[Array, "n k"]) -> Float[Array, "n kp"]:
    """Zero-pad ``X`` to at least as many columns as rows (static shapes)."""
    n, k = X.shape
    if k >= n:
        return X
    return jnp.block([[X, jnp.zeros((n, n - k), dtype=X.dtype)]])


def _qr_lower(
    X: Float[Array, "n k"],
) -> tuple[Float[Array, "n n"], Float[Array, "k n"]]:
    """``L`` lower-triangular with ``L Lᵀ = X Xᵀ``, and ``Q`` with ``X Q = L``.

    The reduced QR ``Xᵀ = Q R`` gives ``X = Rᵀ Qᵀ``; flipping signs so
    ``diag(R) >= 0`` makes ``L = Rᵀ`` unique when ``X`` has full row rank.
    Requires ``k >= n``.
    """
    q, r = jnp.linalg.qr(_t(X), mode="reduced")
    sign = jnp.where(jnp.diagonal(r) < 0, -1.0, 1.0).astype(r.dtype)
    r = einsum(sign, r, "i, i j -> i j")
    q = einsum(q, sign, "k i, i -> k i")
    return _t(r), q


def _tria(X: Float[Array, "n k"]) -> Float[Array, "n n"]:
    r"""Lower-triangular ``L`` with ``L Lᵀ = X Xᵀ`` (Yaghoobi et al.'s ``tria``).

    Differentiated through JAX's QR rule, which needs ``X`` to have full
    row rank. Used for the factors that feed triangular solves, which are
    always built from stacks containing an identity or a noise factor.
    """
    return _qr_lower(_pad_columns(X))[0]


@jax.custom_jvp
def _tria_gram(X: Float[Array, "n k"]) -> Float[Array, "n n"]:
    """`_tria` with a tangent that is valid for rank-deficient ``X``.

    See `_tria_gram_jvp`. ``X`` must already have ``k >= n`` columns.
    """
    return _qr_lower(X)[0]


@_tria_gram.defjvp
def _tria_gram_jvp(primals, tangents):
    r"""Tangent ``dL = dX Q`` for a factor used only through ``L Lᵀ``.

    With ``L = X Q``, the exact tangent is ``dX Q + X dQ = dX Q + L Ω``
    with ``Ω = Qᵀ dQ`` skew-symmetric. The term ``L Ω`` only rotates the
    factor, ``L e^{Ωs}``, which leaves ``L Lᵀ`` unchanged; every consumer
    of these factors in the filter depends on them only through
    ``L Lᵀ``, so dropping it leaves the chain rule exact. Unlike JAX's QR
    rule, this needs no ``R⁻¹``, so it stays finite when ``X`` is rank
    deficient, as the information factors ``Z`` are when ``M < N``.
    """
    (X,), (dX,) = primals, tangents
    L, q = _qr_lower(X)
    return L, dX @ q


def _factor(X: Float[Array, "n k"]) -> Float[Array, "n n"]:
    """`_tria_gram` after zero-padding to a square-or-wider stack."""
    return _tria_gram(_pad_columns(X))


def _input_factor(
    X: Float[Array, "n n"], shift_mask: Bool[Array, " n"] | None = None
) -> Float[Array, "n n"]:
    r"""Cholesky factor of a model covariance, robust to rounding.

    With $D = \mathrm{diag}(X)^{1/2}$ (floored at ``tiny``) and the
    correlation matrix $C = D^{-1} X D^{-1}$, returns
    $L = D\,\mathrm{chol}(C + 4 n \varepsilon I)$ ($\varepsilon$ the
    dtype's machine epsilon). Rounding perturbs each entry of $X$ by at
    most $\varepsilon |X_{ij}| \le \varepsilon \sqrt{X_{ii} X_{jj}}$, i.e.
    $C$ by $\varepsilon$ per entry, so the shift covers a covariance that
    is singular or indefinite only by rounding (a float64 model cast to
    float32) while perturbing every component relative to its own scale —
    which matters for SDE process noise, whose diagonal spans many orders
    of magnitude at small steps. An all-zero ``X`` (e.g. ``Q₀ = 0``)
    factors to a negligible ``L``. ``shift_mask`` limits the shift to the
    ``True`` components: the unit blocks `_masked_obs_inputs` substitutes
    for masked channels are exact, and shifting them would leave a
    ``-½ log(1 + 4 n ε)`` term per masked channel in the likelihood.
    Factor ``X`` in its own dtype, so ``ε`` matches its rounding.
    """
    n = X.shape[-1]
    finfo = jnp.finfo(X.dtype)
    X = 0.5 * (X + _t(X))
    d = jnp.sqrt(jnp.maximum(jnp.diagonal(X), finfo.tiny))
    C = einsum(X, 1.0 / d, 1.0 / d, "i j, i, j -> i j")
    shift = jnp.full((n,), 4 * n * finfo.eps, dtype=X.dtype)
    if shift_mask is not None:
        shift = jnp.where(shift_mask, shift, jnp.zeros_like(shift))
    L_C = jnp.linalg.cholesky(C + jnp.diag(shift))
    return einsum(d, L_C, "i, i j -> i j")


def _solve_lower(
    L: Float[Array, "n n"], B: Float[Array, "n ..."]
) -> Float[Array, "n ..."]:
    return jax.scipy.linalg.solve_triangular(L, B, lower=True)


def _gram(U: Float[Array, "n n"]) -> Float[Array, "n n"]:
    G = U @ _t(U)
    return 0.5 * (G + _t(G))


# ----------------------------------------------------------------
# Elements
# ----------------------------------------------------------------


def _update_factors(N_pred, H, L_R):
    r"""``Ψ = tria([[H N, L_R], [N, 0]])`` and its blocks.

    ``Ψ Ψᵀ = [[S, H P], [P Hᵀ, P]]`` with ``P = N Nᵀ`` and
    ``S = H P Hᵀ + R``, so ``Ψ₁₁ = chol(S)``, ``Ψ₂₁ Ψ₁₁⁻¹ = K`` (the gain)
    and ``Ψ₂₂ Ψ₂₂ᵀ = P − K S Kᵀ`` (the updated covariance).
    """
    M, N = H.shape
    stack = jnp.block(
        [[H @ N_pred, L_R], [N_pred, jnp.zeros((N, M), dtype=N_pred.dtype)]]
    )
    Psi = _tria(stack)
    Psi11, Psi21, Psi22 = Psi[:M, :M], Psi[M:, :M], Psi[M:, M:]
    # K = Ψ₂₁ Ψ₁₁⁻¹, i.e. Kᵀ = Ψ₁₁⁻ᵀ Ψ₂₁ᵀ.
    K = _t(jax.scipy.linalg.solve_triangular(Psi11, _t(Psi21), lower=True, trans=1))
    return Psi11, K, Psi22


def _generic_element(F, H, L_Q, L_R, y):
    """Element for ``t >= 1``: ``x_{t-1}`` given, predict then update with ``y``."""
    N = F.shape[0]
    M = H.shape[0]
    Psi11, K, U = _update_factors(L_Q, H, L_R)
    A = F - K @ (H @ F)
    b = K @ y
    # J = Fᵀ Hᵀ S⁻¹ H F = Z Zᵀ with Z = Fᵀ Hᵀ Ψ₁₁⁻ᵀ; η = Z Ψ₁₁⁻¹ y.
    Z_raw = _t(_solve_lower(Psi11, H @ F))  # (N, M)
    eta = Z_raw @ _solve_lower(Psi11, y)
    Z = _factor(Z_raw) if M > N else _pad_columns(Z_raw)
    return A, b, U, eta, Z


def _first_element(F, H, L_Q, L_R, y, m0, N0):
    """Element 0: absorbs the prior ``N(m0, N0 N0ᵀ)``, predicts, then updates.

    A fully masked step 0 arrives with ``H = 0``, ``R = I``, ``y = 0``
    (`_masked_obs_inputs`), which makes ``K = 0``: the update is then the
    identity and the element is predict-only, without branching.
    """
    N = F.shape[0]
    m_pred = F @ m0
    N_pred = _factor(jnp.block([[F @ N0, L_Q]]))
    _, K, U = _update_factors(N_pred, H, L_R)
    b = m_pred + K @ (y - H @ m_pred)
    zeros = jnp.zeros((N, N), dtype=F.dtype)
    return zeros, b, U, jnp.zeros(N, dtype=F.dtype), zeros


# ----------------------------------------------------------------
# Combination
# ----------------------------------------------------------------


def _combine_one(elem_i, elem_j):
    r"""Combine an earlier element ``i`` with a later element ``j``.

    With ``C_i = U_i U_iᵀ`` and ``J_j = Z_j Z_jᵀ`` (Yaghoobi et al., 2022):

    ``Ξ₁₁ = tria([U_iᵀ Z_j, I])`` (``Ξ₁₁ Ξ₁₁ᵀ = I + U_iᵀ J_j U_i``),
    ``W = U_i Ξ₁₁⁻ᵀ`` and ``Pₓ = Z_jᵀ W``, so that
    ``(I + C_i J_j)⁻¹ = I − W Pₓᵀ Z_jᵀ`` and
    ``(I + C_i J_j)⁻¹ C_i = W Wᵀ``; and
    ``Γ = tria([Z_jᵀ U_i, I])`` with ``(I + J_j C_i)⁻¹ J_j = Ξ₂₂ Ξ₂₂ᵀ``,
    ``Ξ₂₂ = Z_j Γ⁻ᵀ``. Then

        A_ij = A_j (I + C_i J_j)⁻¹ A_i
        b_ij = A_j (I + C_i J_j)⁻¹ (b_i + C_i η_j) + b_j
        U_ij = tria([A_j W, U_j])
        η_ij = A_iᵀ (I + J_j C_i)⁻¹ (η_j − J_j b_i) + η_i
        Z_ij = tria([A_iᵀ Ξ₂₂, Z_i])

    Every inverse is a triangular solve against ``Ξ₁₁`` or ``Γ``, whose
    stacks contain an identity block and so are never singular.
    """
    A_i, b_i, U_i, eta_i, Z_i = elem_i
    A_j, b_j, U_j, eta_j, Z_j = elem_j
    eye = jnp.eye(A_i.shape[0], dtype=A_i.dtype)

    Xi11 = _tria(jnp.block([[_t(U_i) @ Z_j, eye]]))
    W = _t(_solve_lower(Xi11, _t(U_i)))  # U_i Ξ₁₁⁻ᵀ
    Px = _t(Z_j) @ W

    def inv_cj(X):  # (I + C_i J_j)⁻¹ X
        return X - W @ (_t(Px) @ (_t(Z_j) @ X))

    def inv_jc(x):  # (I + J_j C_i)⁻¹ x
        return x - Z_j @ (Px @ (_t(W) @ x))

    A = A_j @ inv_cj(A_i)
    b = A_j @ inv_cj(b_i + U_i @ (_t(U_i) @ eta_j)) + b_j
    U = _factor(jnp.block([[A_j @ W, U_j]]))

    Gamma = _tria(jnp.block([[_t(Z_j) @ U_i, eye]]))
    Xi22 = _t(_solve_lower(Gamma, _t(Z_j)))  # Z_j Γ⁻ᵀ
    eta = _t(A_i) @ inv_jc(eta_j - Z_j @ (_t(Z_j) @ b_i)) + eta_i
    Z = _factor(jnp.block([[_t(A_i) @ Xi22, Z_i]]))
    return A, b, U, eta, Z


# ``lax.associative_scan`` hands the combinator chunks with a leading
# (possibly zero-length) scan axis; vmapping the single-element rule keeps
# the per-element einx patterns free of that axis.
_combine = jax.vmap(_combine_one)


# ----------------------------------------------------------------
# Filter
# ----------------------------------------------------------------


def parallel_kalman_filter_factor(
    transition,
    obs_model,
    process_noise,
    obs_noise,
    observations: Float[Array, "T M"],
    init_mean: Float[Array, " N"],
    init_cov: Float[Array, "N N"],
    *,
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
) -> FilterState:
    """Square-root parallel Kalman filter; see `parallel_kalman_filter`."""
    M_obs = observations.shape[-1]
    T = observations.shape[0]
    N = init_mean.shape[0]
    dtype = jnp.result_type(init_mean, init_cov, observations)

    if T == 0:
        return FilterState(
            filtered_means=jnp.zeros((0, N), dtype=init_mean.dtype),
            filtered_covs=jnp.zeros((0, N, N), dtype=init_cov.dtype),
            predicted_means=jnp.zeros((0, N), dtype=init_mean.dtype),
            predicted_covs=jnp.zeros((0, N, N), dtype=init_cov.dtype),
            log_likelihood=jnp.zeros((), dtype=init_mean.dtype),
        )

    A_seq, H_seq, Q_seq, R_seq, mask_seq, _ = _normalise_tv_inputs(
        transition, obs_model, process_noise, obs_noise, T=T, mask=mask, M=M_obs
    )
    # Per-channel throughout; a (T,) gate broadcasts to every channel.
    mask_ch = mask_seq if mask_seq.ndim == 2 else repeat(mask_seq, "t -> t m", m=M_obs)
    step_active = reduce(mask_ch, "t m -> t", "any")

    def _masked(H, R, y, m):
        H_eff, R_eff, y_eff, n_missing = _masked_obs_inputs(H, R, y, m)
        return H_eff, _input_factor(R_eff, shift_mask=m), y_eff, n_missing

    H_eff, L_R, y_eff, n_missing = jax.vmap(_masked)(
        H_seq, R_seq, observations, mask_ch
    )
    L_Q = jax.vmap(_input_factor)(Q_seq)
    # Factor in the prior's own dtype (its rounding sets the shift), then
    # promote.
    N0 = _input_factor(init_cov).astype(dtype)

    elems = jax.vmap(_generic_element)(A_seq, H_eff, L_Q, L_R, y_eff)
    first = _first_element(A_seq[0], H_eff[0], L_Q[0], L_R[0], y_eff[0], init_mean, N0)
    elems = tuple(arr.at[0].set(val) for arr, val in zip(elems, first, strict=True))

    _, filtered_means, U_out, _, _ = jax.lax.associative_scan(_combine, elems)

    prev_means = jnp.concatenate(
        [rearrange(init_mean, "n -> 1 n"), filtered_means[:-1]]
    )
    prev_factors = jnp.concatenate([rearrange(N0, "n k -> 1 n k"), U_out[:-1]])

    def _predict_and_score(F, m_prev, U_prev, L_Q_t, H, L_R_t, y, n_miss, active):
        m_pred = F @ m_prev
        N_pred = _factor(jnp.block([[F @ U_prev, L_Q_t]]))
        # chol(S) from tria([H N, L_R]): S = H P Hᵀ + R, never formed.
        L_S = _tria(jnp.block([[H @ N_pred, L_R_t]]))
        w = _solve_lower(L_S, y - H @ m_pred)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diagonal(L_S))))
        ll = -0.5 * (w @ w + logdet + M_obs * _LOG_2PI) + 0.5 * n_miss * _LOG_2PI
        return m_pred, _gram(N_pred), jnp.where(active, ll, jnp.zeros_like(ll))

    predicted_means, predicted_covs, ll = jax.vmap(_predict_and_score)(
        A_seq, prev_means, prev_factors, L_Q, H_eff, L_R, y_eff, n_missing, step_active
    )

    return FilterState(
        filtered_means=filtered_means,
        filtered_covs=jax.vmap(_gram)(U_out),
        predicted_means=predicted_means,
        predicted_covs=predicted_covs,
        log_likelihood=jnp.sum(ll),
    )
