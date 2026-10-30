"""Square-root (factor-propagating) parallel Kalman filter.

The associative-scan filter of Särkkä & García-Fernández (2021) carries the
elements (F_cond, μ_cond, Σ_cond, Λμ_lik, Λ_lik) of
`gaussx._ssm._parallel_kalman`. The square-root form of Yaghoobi,
Corenflos, Hassan & Särkkä (2022, §III) carries (F_cond, μ_cond, L_cond,
Λμ_lik, L_lik) instead, with Σ_cond = L_cond L_condᵀ and
Λ_lik = L_lik L_likᵀ; the papers' letters map as A → F_cond, b → μ_cond,
U → L_cond, η → Λμ_lik and Z → L_lik. Every element and every combination
is built from QR decompositions (``tria``) of stacked factors, so no
covariance is formed inside the scan and every returned covariance is a
Gram matrix L Lᵀ, PSD by construction. This is the filter behind
``parallel_kalman_filter(..., square_root=True)`` (gh-454).

A factor takes the subscript of the matrix it factors: L_η, L_ε and L_x,0
for Σ_η, Σ_ε and Σ_x,0, L_y for Σ_y, L_x,t|t−1 for Σ_x,t|t−1. Σ_η, Σ_ε
and Σ_x,0 are factored once, outside the scan, by a Cholesky of their
correlation matrix after a 4 n ε diagonal shift (`_input_factor`), so each
component is perturbed relative to its own scale and a covariance that is
singular or indefinite only by rounding still factors. The factors are the
differentiated path: there is no projection and no ``stop_gradient``.

Pseudocode:

    L_η, L_ε, L_x,0 = factors of Σ_η,t, Σ_ε,t, Σ_x,0       (once, outside)
    element_t = (F_cond, μ_cond, L_cond, Λμ_lik, L_lik)
                from tria([[H_t L_η, L_ε], [L_η, 0]])
    element_1 absorbs 𝒩(μ_x,0, L_x,0 L_x,0ᵀ)
    (_, μ_x,t|t, L_x,t|t, _, _) = associative_scan(combine, elements)
    L_x,t|t−1 = tria([F_t L_x,t−1|t−1, L_η]);  L_y,t = tria([H_t L_x,t|t−1, L_ε])
    log p(y_1:T) = ∑_t log 𝒩(y_t; H_t μ_x,t|t−1, L_y,t L_y,tᵀ)

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


FactorElement = tuple[
    Float[Array, "N N"],
    Float[Array, " N"],
    Float[Array, "N N"],
    Float[Array, " N"],
    Float[Array, "N N"],
]


# ----------------------------------------------------------------
# Triangularisation
# ----------------------------------------------------------------


def _t(X: Float[Array, "row col"]) -> Float[Array, "col row"]:
    """Transpose of a dense matrix, Xᵀ."""
    return rearrange(X, "row col -> col row")


def _pad_columns(X: Float[Array, "n k"]) -> Float[Array, "n kp"]:
    """Zero-pad ``X`` to at least as many columns as rows (static shapes).

    Args:
        X: Stacked factor.

    Returns:
        ``[X, 0]`` with at least as many columns as rows; X itself if it
        already has.
    """
    num_rows, num_cols = X.shape
    if num_cols >= num_rows:
        return X
    padding = jnp.zeros((num_rows, num_rows - num_cols), dtype=X.dtype)
    return jnp.block([[X, padding]])


def _qr_lower(
    X: Float[Array, "n k"],
) -> tuple[Float[Array, "n n"], Float[Array, "k n"]]:
    """Lower-triangular L with L Lᵀ = X Xᵀ, and the Q with X Q = L.

        Xᵀ = Q R,   X = Rᵀ Qᵀ,   L = Rᵀ,   diag(R) ≥ 0

    Flipping signs so diag(R) ≥ 0 makes L unique when X has full row rank.

    Args:
        X: Stacked factor with ``k >= n`` columns.

    Returns:
        Tuple ``(L, Q)``.
    """
    # Xᵀ = Q R
    orthogonal, upper = jnp.linalg.qr(_t(X), mode="reduced")
    sign = jnp.where(jnp.diagonal(upper) < 0, -1.0, 1.0).astype(upper.dtype)
    # flip the rows of R so diag(R) ≥ 0
    upper = einsum(sign, upper, "i, i j -> i j")
    # flip the columns of Q to match, keeping Xᵀ = Q R
    orthogonal = einsum(orthogonal, sign, "k i, i -> k i")
    # L = Rᵀ
    return _t(upper), orthogonal


def _tria(X: Float[Array, "n k"]) -> Float[Array, "n n"]:
    """Lower-triangular L with L Lᵀ = X Xᵀ (Yaghoobi et al.'s ``tria``).

    Differentiated through JAX's QR rule, which needs X to have full row
    rank. Used for the factors that feed triangular solves, which are
    always built from stacks containing an identity or a noise factor.

    Args:
        X: Stacked factor.

    Returns:
        The lower-triangular L.
    """
    return _qr_lower(_pad_columns(X))[0]


@jax.custom_jvp
def _tria_gram(X: Float[Array, "n k"]) -> Float[Array, "n n"]:
    """`_tria` with a tangent that is valid for rank-deficient X.

    See `_tria_gram_jvp`.

    Args:
        X: Stacked factor with ``k >= n`` columns.

    Returns:
        The lower-triangular L with L Lᵀ = X Xᵀ.
    """
    return _qr_lower(X)[0]


@_tria_gram.defjvp
def _tria_gram_jvp(primals, tangents):
    """Tangent dL = dX Q for a factor used only through L Lᵀ.

    With L = X Q, the exact tangent is dX Q + X dQ = dX Q + L Ω with
    Ω = Qᵀ dQ skew-symmetric. The term L Ω only rotates the factor,
    L e^{Ωs}, which leaves L Lᵀ unchanged; every consumer of these factors
    in the filter depends on them only through L Lᵀ, so dropping it leaves
    the chain rule exact. Unlike JAX's QR rule, this needs no R⁻¹, so it
    stays finite when X is rank deficient, as the likelihood factors L_lik
    are when D_y < D_x.
    """
    (X,), (dX,) = primals, tangents
    factor, orthogonal = _qr_lower(X)
    # dL = dX Q
    factor_tangent = dX @ orthogonal
    return factor, factor_tangent


def _factor(X: Float[Array, "n k"]) -> Float[Array, "n n"]:
    """`_tria_gram` after zero-padding to a square-or-wider stack.

    Args:
        X: Stacked factor.

    Returns:
        The lower-triangular L with L Lᵀ = X Xᵀ.
    """
    return _tria_gram(_pad_columns(X))


def _input_factor(
    X: Float[Array, "n n"], shift_mask: Bool[Array, " n"] | None = None
) -> Float[Array, "n n"]:
    """Cholesky factor of a model covariance, robust to rounding.

        D = diag(X)^½,   C = D⁻¹ X D⁻¹,   L = D chol(C + 4 n ε I)

    with D floored at the dtype's ``tiny`` and ε its machine epsilon.
    Rounding perturbs each entry of X by at most
    ε ∣X_ij∣ ≤ ε (X_ii X_jj)^½, i.e. the correlation matrix C by ε per
    entry, so the shift covers a covariance that is singular or indefinite
    only by rounding (a float64 model cast to float32), while perturbing
    every component relative to its own scale. That matters for SDE
    process noise, whose diagonal spans many orders of magnitude at small
    steps. An all-zero X (e.g. Σ_η,1 = 0) factors to a negligible L.

    Args:
        X: Covariance to factor, in its own dtype so that ε matches its
            rounding.
        shift_mask: Components that get the shift; ``None`` shifts all.
            The unit blocks `_masked_obs_inputs` substitutes for masked
            channels are exact, and shifting them would leave a
            −½ log(1 + 4 n ε) term per masked channel in the likelihood.

    Returns:
        The lower-triangular factor L with L Lᵀ ≈ X.
    """
    size = X.shape[-1]
    finfo = jnp.finfo(X.dtype)
    # ½ (X + Xᵀ)
    X = 0.5 * (X + _t(X))
    # D = diag(X)^½, floored at tiny
    scale = jnp.sqrt(jnp.maximum(jnp.diagonal(X), finfo.tiny))
    # C = D⁻¹ X D⁻¹
    correlation = einsum(X, 1.0 / scale, 1.0 / scale, "i j, i, j -> i j")
    # 4 n ε on the shifted components
    shift = jnp.full((size,), 4 * size * finfo.eps, dtype=X.dtype)
    if shift_mask is not None:
        shift = jnp.where(shift_mask, shift, jnp.zeros_like(shift))
    # chol(C + 4 n ε I)
    correlation_factor = jnp.linalg.cholesky(correlation + jnp.diag(shift))
    # L = D chol(C + 4 n ε I)
    return einsum(scale, correlation_factor, "i, i j -> i j")


def _solve_lower(
    L: Float[Array, "n n"], B: Float[Array, "n ..."]
) -> Float[Array, "n ..."]:
    """Triangular solve L⁻¹ B for a lower-triangular L.

    Args:
        L: Lower-triangular factor.
        B: Right-hand side, a vector or a matrix.

    Returns:
        L⁻¹ B.
    """
    return jax.scipy.linalg.solve_triangular(L, B, lower=True)


def _gram(U: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """Gram matrix U Uᵀ of a factor, symmetrised.

    Args:
        U: Factor.

    Returns:
        ½ (U Uᵀ + (U Uᵀ)ᵀ).
    """
    # U Uᵀ
    gram = U @ _t(U)
    return 0.5 * (gram + _t(gram))


# ----------------------------------------------------------------
# Elements
# ----------------------------------------------------------------


def _update_factors(
    predicted_factor: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"],
    obs_noise_factor: Float[Array, "M M"],
) -> tuple[Float[Array, "M M"], Float[Array, "N M"], Float[Array, "N N"]]:
    """Factored Kalman update from one QR of a stacked factor.

        Ψ = tria([[H L_x, L_ε], [L_x, 0]])
        Ψ Ψᵀ = [[Σ_y, H Σ_x], [Σ_x Hᵀ, Σ_x]],   Σ_x = L_x L_xᵀ

    with Σ_y = H Σ_x Hᵀ + Σ_ε, so Ψ₁₁ = L_y, Ψ₂₁ Ψ₁₁⁻¹ = G (the gain) and
    Ψ₂₂ Ψ₂₂ᵀ = Σ_x − G Σ_y Gᵀ (the updated covariance).

    Args:
        predicted_factor: Factor L_x of the covariance before the update.
        obs_operator: Observation operator H, masked rows zeroed.
        obs_noise_factor: Factor L_ε of the observation-error covariance.

    Returns:
        Tuple ``(obs_factor, gain, updated_factor)`` holding L_y, G and
        Ψ₂₂.
    """
    obs_size, state_size = obs_operator.shape
    # H L_x
    obs_projection = einsum(
        obs_operator, predicted_factor, "obs state, state col -> obs col"
    )
    zeros = jnp.zeros((state_size, obs_size), dtype=predicted_factor.dtype)
    # [[H L_x, L_ε], [L_x, 0]]
    stack = jnp.block([[obs_projection, obs_noise_factor], [predicted_factor, zeros]])
    # Ψ = tria([[H L_x, L_ε], [L_x, 0]])
    psi = _tria(stack)
    # L_y = Ψ₁₁
    obs_factor = psi[:obs_size, :obs_size]
    # Ψ₂₁
    gain_block = psi[obs_size:, :obs_size]
    # Ψ₂₂
    updated_factor = psi[obs_size:, obs_size:]
    # Gᵀ = Ψ₁₁⁻ᵀ Ψ₂₁ᵀ
    gain_transpose = jax.scipy.linalg.solve_triangular(
        obs_factor, _t(gain_block), lower=True, trans=1
    )
    # G = Ψ₂₁ Ψ₁₁⁻¹
    gain = _t(gain_transpose)
    return obs_factor, gain, updated_factor


def _generic_element(
    transition_t: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"],
    process_noise_factor: Float[Array, "N N"],
    obs_noise_factor: Float[Array, "M M"],
    obs_t: Float[Array, " M"],
) -> FactorElement:
    """Factored filtering element of step t ≥ 2: x_t−1 given, then update.

        F_cond,t = F_t − G_t H_t F_t,   μ_cond,t = G_t y_t,
        L_cond,t = Ψ₂₂ of `_update_factors` on L_η,t
        Λμ_lik,t = F_tᵀ H_tᵀ Σ_y∣x,t⁻¹ y_t,   L_lik,t = F_tᵀ H_tᵀ L_y,t⁻ᵀ

    where Σ_y∣x,t = H_t Σ_η,t H_tᵀ + Σ_ε,t = L_y,t L_y,tᵀ is the covariance
    of y_t given x_t−1.

    Args:
        transition_t: Dynamics F_t.
        obs_operator: Observation operator H_t, masked rows zeroed.
        process_noise_factor: Factor L_η,t of Σ_η,t.
        obs_noise_factor: Factor L_ε,t of Σ_ε,t.
        obs_t: Observation y_t, masked entries zeroed.

    Returns:
        The element ``(F_cond, μ_cond, L_cond, Λμ_lik, L_lik)`` of step t.
    """
    state_size = transition_t.shape[0]
    obs_size = obs_operator.shape[0]
    obs_factor, gain, cond_factor = _update_factors(
        process_noise_factor, obs_operator, obs_noise_factor
    )
    # H_t F_t
    obs_transition = einsum(
        obs_operator, transition_t, "obs state, state prev -> obs prev"
    )
    # G_t H_t F_t
    correction = einsum(gain, obs_transition, "state obs, obs prev -> state prev")
    # F_cond,t = F_t − G_t H_t F_t
    cond_transition = transition_t - correction
    # μ_cond,t = G_t y_t
    cond_mean = gain @ obs_t
    # L_y,t⁻¹ H_t F_t
    whitened_obs_transition = _solve_lower(obs_factor, obs_transition)
    # F_tᵀ H_tᵀ L_y,t⁻ᵀ, a factor of Λ_lik,t with D_y columns
    lik_factor_raw = _t(whitened_obs_transition)
    # L_y,t⁻¹ y_t
    whitened_obs = _solve_lower(obs_factor, obs_t)
    # Λμ_lik,t = F_tᵀ H_tᵀ L_y,t⁻ᵀ L_y,t⁻¹ y_t
    lik_information = lik_factor_raw @ whitened_obs
    # L_lik,t, square: reduced by QR when D_y > D_x, zero-padded otherwise
    if obs_size > state_size:
        lik_factor = _factor(lik_factor_raw)
    else:
        lik_factor = _pad_columns(lik_factor_raw)
    return cond_transition, cond_mean, cond_factor, lik_information, lik_factor


def _first_element(
    transition_t: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"],
    process_noise_factor: Float[Array, "N N"],
    obs_noise_factor: Float[Array, "M M"],
    obs_t: Float[Array, " M"],
    init_mean: Float[Array, " N"],
    init_factor: Float[Array, "N N"],
) -> FactorElement:
    """First factored element: absorb 𝒩(μ_x,0, L_x,0 L_x,0ᵀ), predict, update.

        F_cond,1 = 0,   μ_cond,1 = μ_x,1|0 + G₁ r₁,   L_cond,1 = L_x,1|1,
        Λμ_lik,1 = 0,   L_lik,1 = 0

    A fully masked step 1 arrives with H₁ = 0, Σ_ε,1 = I and y₁ = 0
    (`_masked_obs_inputs`), which makes G₁ = 0: the update is then the
    identity and the element is predict-only, without branching.

    Args:
        transition_t: Dynamics F₁.
        obs_operator: Observation operator H₁, masked rows zeroed.
        process_noise_factor: Factor L_η,1 of Σ_η,1.
        obs_noise_factor: Factor L_ε,1 of Σ_ε,1.
        obs_t: Observation y₁, masked entries zeroed.
        init_mean: Prior mean μ_x,0.
        init_factor: Factor L_x,0 of the prior covariance Σ_x,0.

    Returns:
        The element ``(F_cond, μ_cond, L_cond, Λμ_lik, L_lik)`` of step 1.
    """
    state_size = transition_t.shape[0]
    # μ_x,1|0 = F₁ μ_x,0
    predicted_mean = transition_t @ init_mean
    # F₁ L_x,0
    propagated_factor = einsum(
        transition_t, init_factor, "next prev, prev col -> next col"
    )
    # L_x,1|0 = tria([F₁ L_x,0, L_η,1])
    predicted_factor = _factor(jnp.block([[propagated_factor, process_noise_factor]]))
    _, gain, cond_factor = _update_factors(
        predicted_factor, obs_operator, obs_noise_factor
    )
    # H₁ μ_x,1|0
    predicted_obs = obs_operator @ predicted_mean
    # r₁ = y₁ − H₁ μ_x,1|0
    residual = obs_t - predicted_obs
    # μ_cond,1 = μ_x,1|0 + G₁ r₁
    cond_mean = predicted_mean + gain @ residual
    zeros = jnp.zeros((state_size, state_size), dtype=transition_t.dtype)
    lik_information = jnp.zeros(state_size, dtype=transition_t.dtype)
    return zeros, cond_mean, cond_factor, lik_information, zeros


# ----------------------------------------------------------------
# Combination
# ----------------------------------------------------------------


def _combine_one(elem_i: FactorElement, elem_j: FactorElement) -> FactorElement:
    """Combine an earlier factored element i with a later element j.

        Ξ₁₁ = tria([L_cond,iᵀ L_lik,j, I]),   Ξ₁₁ Ξ₁₁ᵀ = I + L_cond,iᵀ Λ_lik,j L_cond,i
        W = L_cond,i Ξ₁₁⁻ᵀ,   V = L_lik,jᵀ W
        (I + Σ_cond,i Λ_lik,j)⁻¹ = I − W Vᵀ L_lik,jᵀ
        (I + Λ_lik,j Σ_cond,i)⁻¹ = I − L_lik,j V Wᵀ
        Γ = tria([L_lik,jᵀ L_cond,i, I]),   Ξ₂₂ = L_lik,j Γ⁻ᵀ

    so that (I + Σ_cond,i Λ_lik,j)⁻¹ Σ_cond,i = W Wᵀ and
    (I + Λ_lik,j Σ_cond,i)⁻¹ Λ_lik,j = Ξ₂₂ Ξ₂₂ᵀ. Then

        F_cond,ij = F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ F_cond,i
        μ_cond,ij = F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ (μ_cond,i + Σ_cond,i Λμ_lik,j)
                    + μ_cond,j
        L_cond,ij = tria([F_cond,j W, L_cond,j])
        Λμ_lik,ij = F_cond,iᵀ (I + Λ_lik,j Σ_cond,i)⁻¹ (Λμ_lik,j − Λ_lik,j μ_cond,i)
                    + Λμ_lik,i
        L_lik,ij = tria([F_cond,iᵀ Ξ₂₂, L_lik,i])

    after Yaghoobi et al. (2022). V replaces the original's Pₓ. Every
    inverse is a triangular solve against Ξ₁₁ or Γ, whose stacks contain an
    identity block and so are never singular.

    Args:
        elem_i: Element i, ``(F_cond, μ_cond, L_cond, Λμ_lik, L_lik)``.
        elem_j: Element j, in the same layout.

    Returns:
        The combined element ij.
    """
    cond_transition_i, cond_mean_i, cond_factor_i, lik_information_i, lik_factor_i = (
        elem_i
    )
    cond_transition_j, cond_mean_j, cond_factor_j, lik_information_j, lik_factor_j = (
        elem_j
    )
    identity = jnp.eye(cond_transition_i.shape[0], dtype=cond_transition_i.dtype)
    # L_cond,iᵀ
    cond_factor_i_transpose = _t(cond_factor_i)
    # L_lik,jᵀ
    lik_factor_j_transpose = _t(lik_factor_j)

    # --- forward weights ---
    # L_cond,iᵀ L_lik,j
    cross_factor = einsum(
        cond_factor_i, lik_factor_j, "state row, state col -> row col"
    )
    # Ξ₁₁ = tria([L_cond,iᵀ L_lik,j, I])
    xi11 = _tria(jnp.block([[cross_factor, identity]]))
    # Ξ₁₁⁻¹ L_cond,iᵀ
    weight_transpose = _solve_lower(xi11, cond_factor_i_transpose)
    # W = L_cond,i Ξ₁₁⁻ᵀ
    weight = _t(weight_transpose)
    # V = L_lik,jᵀ W
    projected_weight = einsum(lik_factor_j, weight, "state row, state col -> row col")
    # Vᵀ
    projected_weight_transpose = _t(projected_weight)

    def forward_inverse(rhs):
        # (I + Σ_cond,i Λ_lik,j)⁻¹ X = X − W Vᵀ L_lik,jᵀ X, X a vector or matrix
        # L_lik,jᵀ X
        projected = lik_factor_j_transpose @ rhs
        # Vᵀ L_lik,jᵀ X
        reweighted = projected_weight_transpose @ projected
        # W Vᵀ L_lik,jᵀ X
        correction = weight @ reweighted
        return rhs - correction

    def backward_inverse(rhs):
        # (I + Λ_lik,j Σ_cond,i)⁻¹ x = x − L_lik,j V Wᵀ x
        # Wᵀ x
        projected = weight_transpose @ rhs
        # V Wᵀ x
        reweighted = projected_weight @ projected
        # L_lik,j V Wᵀ x
        correction = lik_factor_j @ reweighted
        return rhs - correction

    # (I + Σ_cond,i Λ_lik,j)⁻¹ F_cond,i
    solved_transition = forward_inverse(cond_transition_i)
    # F_cond,ij = F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ F_cond,i
    cond_transition = einsum(
        cond_transition_j, solved_transition, "next state, state prev -> next prev"
    )
    # L_cond,iᵀ Λμ_lik,j
    projected_information = cond_factor_i_transpose @ lik_information_j
    # Σ_cond,i Λμ_lik,j = L_cond,i L_cond,iᵀ Λμ_lik,j
    information_shift = cond_factor_i @ projected_information
    # (I + Σ_cond,i Λ_lik,j)⁻¹ (μ_cond,i + Σ_cond,i Λμ_lik,j)
    solved_mean = forward_inverse(cond_mean_i + information_shift)
    # F_cond,j (I + Σ_cond,i Λ_lik,j)⁻¹ (μ_cond,i + Σ_cond,i Λμ_lik,j)
    propagated_mean = cond_transition_j @ solved_mean
    # μ_cond,ij = … + μ_cond,j
    cond_mean = propagated_mean + cond_mean_j
    # F_cond,j W
    propagated_weight = einsum(
        cond_transition_j, weight, "next state, state col -> next col"
    )
    # L_cond,ij = tria([F_cond,j W, L_cond,j])
    cond_factor = _factor(jnp.block([[propagated_weight, cond_factor_j]]))

    # --- backward weights ---
    # L_lik,jᵀ L_cond,i
    cross_factor_transpose = einsum(
        lik_factor_j, cond_factor_i, "state row, state col -> row col"
    )
    # Γ = tria([L_lik,jᵀ L_cond,i, I])
    gamma = _tria(jnp.block([[cross_factor_transpose, identity]]))
    # Γ⁻¹ L_lik,jᵀ
    xi22_transpose = _solve_lower(gamma, lik_factor_j_transpose)
    # Ξ₂₂ = L_lik,j Γ⁻ᵀ
    xi22 = _t(xi22_transpose)
    # L_lik,jᵀ μ_cond,i
    projected_mean = lik_factor_j_transpose @ cond_mean_i
    # Λ_lik,j μ_cond,i = L_lik,j L_lik,jᵀ μ_cond,i
    precision_mean = lik_factor_j @ projected_mean
    # (I + Λ_lik,j Σ_cond,i)⁻¹ (Λμ_lik,j − Λ_lik,j μ_cond,i)
    solved_information = backward_inverse(lik_information_j - precision_mean)
    # F_cond,iᵀ
    cond_transition_i_transpose = _t(cond_transition_i)
    # F_cond,iᵀ (I + Λ_lik,j Σ_cond,i)⁻¹ (Λμ_lik,j − Λ_lik,j μ_cond,i)
    propagated_information = cond_transition_i_transpose @ solved_information
    # Λμ_lik,ij = … + Λμ_lik,i
    lik_information = propagated_information + lik_information_i
    # F_cond,iᵀ Ξ₂₂
    propagated_xi22 = einsum(cond_transition_i, xi22, "next prev, next col -> prev col")
    # L_lik,ij = tria([F_cond,iᵀ Ξ₂₂, L_lik,i])
    lik_factor = _factor(jnp.block([[propagated_xi22, lik_factor_i]]))
    return cond_transition, cond_mean, cond_factor, lik_information, lik_factor


# ``lax.associative_scan`` hands the combinator chunks with a leading
# (possibly zero-length) scan axis; vmapping the single-element rule keeps
# the per-element einx patterns free of that axis.
_combine = jax.vmap(_combine_one)


# ----------------------------------------------------------------
# Filter
# ----------------------------------------------------------------


def _masked_factors(
    obs_operator: Float[Array, "M N"],
    obs_noise_covariance: Float[Array, "M M"],
    obs_t: Float[Array, " M"],
    mask_t: Bool[Array, " M"],
) -> tuple[
    Float[Array, "M N"], Float[Array, "M M"], Float[Array, " M"], Float[Array, ""]
]:
    """Masked observation inputs of one step, with Σ_ε,t factored.

    Args:
        obs_operator: Observation operator H_t.
        obs_noise_covariance: Observation-error covariance Σ_ε,t.
        obs_t: Observation y_t.
        mask_t: Per-channel mask of y_t; ``True`` keeps a channel.

    Returns:
        Tuple ``(obs_operator, obs_noise_factor, obs, n_missing)``: H_t and
        y_t with masked channels zeroed, L_ε,t with unit blocks on them
        (left unshifted), and the number of masked channels N_missing.
    """
    masked_obs_operator, masked_obs_noise, masked_obs, n_missing = _masked_obs_inputs(
        obs_operator, obs_noise_covariance, obs_t, mask_t
    )
    # L_ε,t, shifting only the observed channels
    obs_noise_factor = _input_factor(masked_obs_noise, shift_mask=mask_t)
    return masked_obs_operator, obs_noise_factor, masked_obs, n_missing


def _predict_and_score(
    transition_t: Float[Array, "N N"],
    filtered_mean: Float[Array, " N"],
    filtered_factor: Float[Array, "N N"],
    process_noise_factor: Float[Array, "N N"],
    obs_operator: Float[Array, "M N"],
    obs_noise_factor: Float[Array, "M M"],
    obs_t: Float[Array, " M"],
    n_missing: Float[Array, ""],
    active: Bool[Array, ""],
) -> tuple[Float[Array, " N"], Float[Array, "N N"], Float[Array, ""]]:
    """Factored predict step and the log-likelihood term of one step.

        L_x,t|t−1 = tria([F_t L_x,t−1|t−1, L_η,t])
        L_y,t = tria([H_t L_x,t|t−1, L_ε,t])
        log p(y_t ∣ y_1:t−1) = −½ (‖L_y,t⁻¹ r_t‖² + log det Σ_y,t + D_y log 2π)
                               + ½ N_missing log 2π

    Σ_y,t = H_t Σ_x,t|t−1 H_tᵀ + Σ_ε,t is never formed.

    Args:
        transition_t: Dynamics F_t.
        filtered_mean: Filtered mean μ_x,t−1|t−1.
        filtered_factor: Factor L_x,t−1|t−1 of the filtered covariance.
        process_noise_factor: Factor L_η,t of Σ_η,t.
        obs_operator: Observation operator H_t, masked rows zeroed.
        obs_noise_factor: Factor L_ε,t of Σ_ε,t.
        obs_t: Observation y_t, masked entries zeroed.
        n_missing: Number of masked channels N_missing.
        active: Whether the step observes any channel; an inactive step
            contributes 0.

    Returns:
        Tuple ``(predicted_mean, predicted_covariance, log_likelihood_step)``
        holding μ_x,t|t−1, Σ_x,t|t−1 and log p(y_t ∣ y_1:t−1).
    """
    obs_size = obs_t.shape[-1]
    # μ_x,t|t−1 = F_t μ_x,t−1|t−1
    predicted_mean = transition_t @ filtered_mean
    # F_t L_x,t−1|t−1
    propagated_factor = einsum(
        transition_t, filtered_factor, "next prev, prev col -> next col"
    )
    # L_x,t|t−1 = tria([F_t L_x,t−1|t−1, L_η,t])
    predicted_factor = _factor(jnp.block([[propagated_factor, process_noise_factor]]))
    # H_t L_x,t|t−1
    obs_projection = einsum(
        obs_operator, predicted_factor, "obs state, state col -> obs col"
    )
    # L_y,t = tria([H_t L_x,t|t−1, L_ε,t])
    obs_factor = _tria(jnp.block([[obs_projection, obs_noise_factor]]))
    # H_t μ_x,t|t−1
    predicted_obs = obs_operator @ predicted_mean
    # r_t = y_t − H_t μ_x,t|t−1
    residual = obs_t - predicted_obs
    # L_y,t⁻¹ r_t
    whitened_residual = _solve_lower(obs_factor, residual)
    # ‖L_y,t⁻¹ r_t‖² = r_tᵀ Σ_y,t⁻¹ r_t
    mahalanobis = whitened_residual @ whitened_residual
    # log det Σ_y,t = 2 ∑ log ∣diag(L_y,t)∣
    logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diagonal(obs_factor))))
    # log p(y_t ∣ y_1:t−1) = −½ (r_tᵀ Σ_y,t⁻¹ r_t + log det Σ_y,t
    #     + D_y log 2π) + ½ N_missing log 2π
    log_likelihood_step = (
        -0.5 * (mahalanobis + logdet + obs_size * _LOG_2PI) + 0.5 * n_missing * _LOG_2PI
    )
    log_likelihood_step = jnp.where(
        active, log_likelihood_step, jnp.zeros_like(log_likelihood_step)
    )
    # Σ_x,t|t−1 = L_x,t|t−1 L_x,t|t−1ᵀ
    predicted_covariance = _gram(predicted_factor)
    return predicted_mean, predicted_covariance, log_likelihood_step


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
    """Square-root parallel Kalman filter; see `parallel_kalman_filter`.

        Σ_x,t|t = L_x,t|t L_x,t|tᵀ,   Σ_x,t|t−1 = L_x,t|t−1 L_x,t|t−1ᵀ

    with every factor built by QR, so both are PSD by construction. This is
    ``parallel_kalman_filter(..., square_root=True)``.

    Algorithm:
        1. Factor Σ_η,t, Σ_ε,t (observed channels shifted only) and Σ_x,0
           once with `_input_factor`.
        2. Build the factored elements (F_cond, μ_cond, L_cond, Λμ_lik,
           L_lik) of every step, the first absorbing the prior.
        3. Run the associative scan; μ_cond,t and L_cond,t of the result
           are μ_x,t|t and L_x,t|t.
        4. Predict from L_x,t−1|t−1 and score each step through L_y,t.

    Args:
        transition: Linear dynamics F_t, one or a stack over time.
        obs_model: Observation operator H_t, one or a stack over time.
        process_noise: Model-error covariance Σ_η,t, one or a stack.
        obs_noise: Observation-error covariance Σ_ε,t, one or a stack.
        observations: Observations y_1:T, one row per step.
        init_mean: Prior mean μ_x,0 of the initial state x₀.
        init_cov: Prior covariance Σ_x,0 of x₀.
        mask: Optional per-step ``(T,)`` or per-channel ``(T, M)`` mask, as
            in `gaussx.kalman_filter`.

    Returns:
        A `FilterState` with the filtered and predicted means and
        covariances and the total log-likelihood.

    Examples:
        The factored filter matches the covariance form:

        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> model = (jnp.eye(2), jnp.array([[1.0, 0.0]]), 0.1 * jnp.eye(2))
        >>> observations = jnp.array([[0.1], [0.3], [0.2]])
        >>> data = (jnp.array([[0.5]]), observations, jnp.zeros(2), jnp.eye(2))
        >>> state = parallel_kalman_filter_factor(*model, *data)
        >>> state.filtered_covs.shape
        (3, 2, 2)
        >>> reference = gaussx.kalman_filter(*model, *data)
        >>> bool(jnp.allclose(state.log_likelihood, reference.log_likelihood))
        True

        A per-channel mask is supported:

        >>> mask = jnp.array([[True], [False], [True]])
        >>> masked = parallel_kalman_filter_factor(*model, *data, mask=mask)
        >>> bool(jnp.allclose(masked.filtered_means[1], masked.predicted_means[1]))
        True
    """
    obs_size = observations.shape[-1]
    num_steps = observations.shape[0]
    state_size = init_mean.shape[0]
    dtype = jnp.result_type(init_mean, init_cov, observations)

    if num_steps == 0:
        return FilterState(
            filtered_means=jnp.zeros((0, state_size), dtype=init_mean.dtype),
            filtered_covs=jnp.zeros((0, state_size, state_size), dtype=init_cov.dtype),
            predicted_means=jnp.zeros((0, state_size), dtype=init_mean.dtype),
            predicted_covs=jnp.zeros((0, state_size, state_size), dtype=init_cov.dtype),
            log_likelihood=jnp.zeros((), dtype=init_mean.dtype),
        )

    transitions, obs_operators, process_noises, obs_noises, masks, _ = (
        _normalise_tv_inputs(
            transition,
            obs_model,
            process_noise,
            obs_noise,
            T=num_steps,
            mask=mask,
            M=obs_size,
        )
    )
    # Per channel throughout; a (T,) gate broadcasts to every channel.
    if masks.ndim == 2:
        channel_masks = masks
    else:
        channel_masks = repeat(masks, "t -> t obs", obs=obs_size)
    step_active = reduce(channel_masks, "t obs -> t", "any")

    # --- input factors ---
    masked_obs_operators, obs_noise_factors, masked_obs, n_missing = jax.vmap(
        _masked_factors
    )(obs_operators, obs_noises, observations, channel_masks)
    # L_η,t
    process_noise_factors = jax.vmap(_input_factor)(process_noises)
    # L_x,0, factored in the prior's own dtype (its rounding sets the
    # shift), then promoted
    init_factor = _input_factor(init_cov).astype(dtype)

    # --- elements and scan ---
    elements = jax.vmap(_generic_element)(
        transitions,
        masked_obs_operators,
        process_noise_factors,
        obs_noise_factors,
        masked_obs,
    )
    first = _first_element(
        transitions[0],
        masked_obs_operators[0],
        process_noise_factors[0],
        obs_noise_factors[0],
        masked_obs[0],
        init_mean,
        init_factor,
    )
    elements = tuple(
        stack.at[0].set(value) for stack, value in zip(elements, first, strict=True)
    )
    # (F_cond, μ_x,t|t, L_x,t|t, Λμ_lik, L_lik) for every t
    _, filtered_means, filtered_factors, _, _ = jax.lax.associative_scan(
        _combine, elements
    )

    # --- predict and score ---
    # μ_x,0:T−1|0:T−1, the prior first
    previous_means = jnp.concatenate(
        [rearrange(init_mean, "state -> 1 state"), filtered_means[:-1]]
    )
    # L_x,0:T−1|0:T−1, the prior first
    previous_factors = jnp.concatenate(
        [rearrange(init_factor, "state col -> 1 state col"), filtered_factors[:-1]]
    )
    predicted_means, predicted_covs, log_likelihood_steps = jax.vmap(
        _predict_and_score
    )(
        transitions,
        previous_means,
        previous_factors,
        process_noise_factors,
        masked_obs_operators,
        obs_noise_factors,
        masked_obs,
        n_missing,
        step_active,
    )

    return FilterState(
        filtered_means=filtered_means,
        # Σ_x,t|t = L_x,t|t L_x,t|tᵀ
        filtered_covs=jax.vmap(_gram)(filtered_factors),
        predicted_means=predicted_means,
        predicted_covs=predicted_covs,
        # log p(y_1:T) = ∑_t log p(y_t ∣ y_1:t−1)
        log_likelihood=jnp.sum(log_likelihood_steps),
    )
