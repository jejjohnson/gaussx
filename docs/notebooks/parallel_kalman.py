# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parallel Kalman Filter & SSM Natural Parameters
#
# This notebook demonstrates the parallel (associative-scan) Kalman filter
# and smoother, the Joseph-form covariance update, and the natural/expectation
# parameter representations of linear-Gaussian state-space models.
#
# **What you'll learn:**
#
# 1. Running `gaussx.parallel_kalman_filter` and `gaussx.parallel_rts_smoother`
# 2. Verifying equivalence with the sequential Kalman filter
# 3. The Joseph-form covariance update for numerical stability
# 4. Converting SSM parameters to natural parameters (`BlockTriDiag` precision)
# 5. Converting smoother output to expectation parameters and back

# %% [markdown]
# ## Background: why parallelism matters
#
# The standard Kalman filter is inherently sequential: each step depends on
# the previous filtered state. On modern accelerators (GPU/TPU) this
# sequential bottleneck limits throughput for long time series.
#
# The **associative-scan** formulation (Sarkka & Garcia-Fernandez, 2021)
# reformulates the Kalman recursion as a parallel prefix sum over a
# semigroup of affine transformations. This enables $O(\log T)$ parallel
# depth instead of $O(T)$ sequential steps, while producing identical
# results up to floating-point precision.
#
# In gaussx, `parallel_kalman_filter` and `parallel_rts_smoother` provide
# the same interface as their sequential counterparts but use dense array
# operations that are faster on GPU/TPU for long sequences.

# %%
from __future__ import annotations

import warnings


warnings.filterwarnings("ignore", message=r".*IProgress.*")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import gaussx


jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Define the model
#
# We reuse the **damped oscillator** from the sequential Kalman filter
# notebook. The 2D state ($D_x = 2$) $\mathbf{x}_t = [\text{position}, \text{velocity}]^\top$
# follows a discretized spring-mass-damper system, and only position is
# observed with noise ($D_y = 1$), with $\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \boldsymbol{\eta}_t$ and $\mathbf{y}_t = \mathbf{H} \mathbf{x}_t + \boldsymbol{\varepsilon}_t$.
#
# - $\mathbf{F}$: discretized spring-mass-damper transition
# - $\mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}$ (observe position only)
# - $\boldsymbol{\Sigma}_\eta$: model noise driving the oscillator
# - $\boldsymbol{\Sigma}_\varepsilon = [0.5]$: observation noise variance

# %%
# Δt
time_step = 0.1
# T
num_steps = 200

# Damped oscillator: omega=1.0 rad/s, damping gamma=0.15
angular_frequency, damping = 1.0, 0.15
# F = [[1, Δt], [−ω² Δt, 1 − γ Δt]]
transition = jnp.array(
    [
        [1.0, time_step],
        [-(angular_frequency**2) * time_step, 1.0 - damping * time_step],
    ]
)

# Observation matrix (observe position only)
# H ∈ ℝ^{D_y×D_x}
obs_operator = jnp.array([[1.0, 0.0]])

# Model noise covariance
# Σ_η = q_c [[Δt³/3, Δt²/2], [Δt²/2, Δt]]
noise_spectral_density = 0.3
model_noise_covariance = noise_spectral_density * jnp.array(
    [[time_step**3 / 3, time_step**2 / 2], [time_step**2 / 2, time_step]]
)

# Observation noise covariance
# Σ_ε ∈ ℝ^{D_y×D_y}
obs_noise_covariance = jnp.array([[0.5]])

print("F =\n", transition)
print("H =", obs_operator)
print("Σ_η =\n", model_noise_covariance)
print("Σ_ε =", obs_noise_covariance)

# %% [markdown]
# ## Simulate data
#
# Generate $T = 200$ time steps from the model. The true trajectory is
# sampled from the generative process so the Kalman filter assumptions
# are satisfied.

# %%
key = jax.random.PRNGKey(42)


def simulate_step(carry, key_t):
    state = carry
    k1, k2 = jax.random.split(key_t)
    # η_t ~ 𝒩(0, Σ_η)
    model_noise = jax.random.multivariate_normal(
        k1, jnp.zeros(2), model_noise_covariance
    )
    # x_t = F x_{t−1} + η_t
    new_state = transition @ state + model_noise
    # ε_t ~ 𝒩(0, Σ_ε)
    obs_noise = jax.random.multivariate_normal(k2, jnp.zeros(1), obs_noise_covariance)
    # y_t = H x_t + ε_t
    obs_t = obs_operator @ new_state + obs_noise
    return new_state, (new_state, obs_t)


# x_0
true_init_state = jnp.array([3.0, 0.0])  # displaced from equilibrium, at rest
keys = jax.random.split(key, num_steps)
_, (true_states, observations) = jax.lax.scan(simulate_step, true_init_state, keys)

times = jnp.arange(num_steps) * time_step
true_position = true_states[:, 0]

print("true_states shape:", true_states.shape)
print("observations shape:", observations.shape)

# %% [markdown]
# ## Run parallel Kalman filter and smoother

# %%
# μ_x,0 = 0
prior_mean = jnp.zeros(2)
# Σ_x,0 = 4 I
prior_covariance = jnp.eye(2) * 4.0

# Parallel filter
par_filter_state = gaussx.parallel_kalman_filter(
    transition,
    obs_operator,
    model_noise_covariance,
    obs_noise_covariance,
    observations,
    prior_mean,
    prior_covariance,
)

print("Parallel filtered means shape:", par_filter_state.filtered_means.shape)
print("Parallel log-likelihood:", par_filter_state.log_likelihood)

# %%
# Parallel smoother
par_smooth_means, par_smooth_covs = gaussx.parallel_rts_smoother(
    par_filter_state, transition
)

print("Parallel smoothed means shape:", par_smooth_means.shape)
print("Parallel smoothed covs shape:", par_smooth_covs.shape)

# %% [markdown]
# ## Compare with sequential Kalman filter
#
# The parallel and sequential implementations should produce identical
# results (up to floating-point precision). We verify this by comparing
# filtered means, covariances, and the log-likelihood.

# %%
# Sequential filter
seq_filter_state = gaussx.kalman_filter(
    transition,
    obs_operator,
    model_noise_covariance,
    obs_noise_covariance,
    observations,
    prior_mean,
    prior_covariance,
)

# Sequential smoother
seq_smooth_means, seq_smooth_covs = gaussx.rts_smoother(seq_filter_state, transition)

# Compare filtered means
mean_diff = jnp.max(
    jnp.abs(par_filter_state.filtered_means - seq_filter_state.filtered_means)
)
print(f"Max abs diff (filtered means): {mean_diff:.2e}")

# Compare filtered covariances
cov_diff = jnp.max(
    jnp.abs(par_filter_state.filtered_covs - seq_filter_state.filtered_covs)
)
print(f"Max abs diff (filtered covs):  {cov_diff:.2e}")

# Compare log-likelihoods
ll_diff = jnp.abs(par_filter_state.log_likelihood - seq_filter_state.log_likelihood)
print(f"Abs diff (log-likelihood):     {ll_diff:.2e}")

# Compare smoothed means
smooth_mean_diff = jnp.max(jnp.abs(par_smooth_means - seq_smooth_means))
print(f"Max abs diff (smoothed means): {smooth_mean_diff:.2e}")

# Compare smoothed covariances
smooth_cov_diff = jnp.max(jnp.abs(par_smooth_covs - seq_smooth_covs))
print(f"Max abs diff (smoothed covs):  {smooth_cov_diff:.2e}")

# %% [markdown]
# ## Joseph-form covariance update
#
# The standard Kalman update computes the posterior covariance as
# $\boldsymbol{\Sigma}_{x,t \mid t} = (\mathbf{I} - \mathbf{G}_t \mathbf{H}) \boldsymbol{\Sigma}_{x,t \mid t-1}$, but this can lose symmetry or positive
# definiteness due to round-off. The **Joseph form** is more robust:
#
# $$\boldsymbol{\Sigma}_{x,t \mid t} = (\mathbf{I} - \mathbf{G}_t \mathbf{H}) \boldsymbol{\Sigma}_{x,t \mid t-1} (\mathbf{I} - \mathbf{G}_t \mathbf{H})^\top + \mathbf{G}_t \boldsymbol{\Sigma}_\varepsilon \mathbf{G}_t^\top$$
#
# We manually compute the Kalman gain $\mathbf{G}_t$ at one time step and compare both
# formulas.

# %%
# Pick a time step to illustrate
step = 50
# Σ_x,t|t−1 ∈ ℝ^{D_x×D_x}
predicted_covariance = par_filter_state.predicted_covs[step]  # (2, 2)
# μ_x,t|t−1 ∈ ℝ^{D_x}
predicted_mean = par_filter_state.predicted_means[step]  # (2,)

# Compute Kalman gain G = Σ_x,t|t−1 Hᵀ (H Σ_x,t|t−1 Hᵀ + Σ_ε)⁻¹
# Σ_y,t = H Σ_x,t|t−1 Hᵀ + Σ_ε
obs_covariance = (
    obs_operator @ predicted_covariance @ obs_operator.T + obs_noise_covariance
)
# G_t = Σ_x,t|t−1 Hᵀ Σ_y,t⁻¹
gain = predicted_covariance @ obs_operator.T @ jnp.linalg.inv(obs_covariance)

print(f"Time step {step}:")
print(f"Kalman gain G =\n{gain}")

# Standard update: (I - GH) Σ_x,t|t−1
# Σ_x,t|t = (I − G_t H) Σ_x,t|t−1
state_size = predicted_covariance.shape[0]
standard_covariance = (jnp.eye(state_size) - gain @ obs_operator) @ predicted_covariance

# Joseph-form update
# Σ_x,t|t = (I − G_t H) Σ_x,t|t−1 (I − G_t H)ᵀ + G_t Σ_ε G_tᵀ
joseph_covariance = gaussx.joseph_update(
    predicted_covariance, gain, obs_operator, obs_noise_covariance
)

print(f"\nStandard update Σ_x,t|t =\n{standard_covariance}")
print(f"\nJoseph update Σ_x,t|t =\n{joseph_covariance}")
max_difference = jnp.max(jnp.abs(standard_covariance - joseph_covariance))
print(f"\nMax abs difference: {max_difference:.2e}")

# The Joseph form guarantees symmetry
standard_asymmetry = jnp.max(jnp.abs(standard_covariance - standard_covariance.T))
joseph_asymmetry = jnp.max(jnp.abs(joseph_covariance - joseph_covariance.T))
print(f"\nStandard symmetry error: {standard_asymmetry:.2e}")
print(f"Joseph symmetry error:   {joseph_asymmetry:.2e}")

# %% [markdown]
# ## SSM natural parameters
#
# A linear-Gaussian state-space model defines a joint Gaussian
# $p(\mathbf{x}_0, \ldots, \mathbf{x}_{T-1})$ on the stacked state $\mathbf{x} \in \mathbb{R}^{T D_x}$ whose precision matrix $\boldsymbol{\Lambda}_x$ is **block-tridiagonal**.
# The function `ssm_to_naturals` extracts the natural parameters
# $(\boldsymbol{\Lambda}_x \boldsymbol{\mu}_x, -\frac{1}{2} \boldsymbol{\Lambda}_x)$ where $-\frac{1}{2} \boldsymbol{\Lambda}_x$ is
# stored as a `BlockTriDiag` operator.
#
# The API takes the chain in the same layout as `gaussx.MarkovGaussian`:
# `A` and `Q` of shape `(T-1, D_x, D_x)`, where `Q[t]` is the model-noise covariance $\boldsymbol{\Sigma}_{\eta,t+1}$ of the
# transition $\mathbf{x}_t \to \mathbf{x}_{t+1}$, and the initial covariance $\boldsymbol{\Sigma}_{x,0}$ separately.

# %%
# F_t = F for t = 0, …, T − 2
transition_full = jnp.tile(transition, (num_steps - 1, 1, 1))
# Σ_η,t+1 = Σ_η for t = 0, …, T − 2
model_noise_full = jnp.tile(model_noise_covariance, (num_steps - 1, 1, 1))

print("F_full shape:", transition_full.shape)
print("Σ_η_full shape:", model_noise_full.shape)

# %%
# Convert to natural parameters
# (Λ_x μ_x, −½ Λ_x)
natural_linear, natural_precision = gaussx.ssm_to_naturals(
    transition_full, model_noise_full, prior_mean, prior_covariance
)

print("natural_linear shape:", natural_linear.shape)
print("natural_precision type:", type(natural_precision).__name__)
print("natural_precision diagonal shape:", natural_precision.diagonal.shape)
print("natural_precision sub_diagonal shape:", natural_precision.sub_diagonal.shape)

# %% [markdown]
# ### Inspect the `BlockTriDiag` structure
#
# The `BlockTriDiag` operator stores `(T, D_x, D_x)` diagonal blocks and
# `(T-1, D_x, D_x)` sub-diagonal blocks. Its dense representation is a
# $(T D_x \times T D_x)$ matrix with the characteristic banded sparsity
# pattern of a Gauss-Markov chain.

# %%
# Materialize the dense precision matrix (small enough for T=200, D_x=2)
dense_precision = natural_precision.as_matrix()
print("Dense precision shape:", dense_precision.shape)

# %%
# Round-trip: convert back to SSM parameters
transition_rt, model_noise_rt, prior_mean_rt, prior_covariance_rt = (
    gaussx.naturals_to_ssm(natural_linear, natural_precision, initial_in_q=False)
)

print(
    "Round-trip F max error:", jnp.max(jnp.abs(transition_rt - transition_full)).item()
)
print(
    "Round-trip Σ_η max error:",
    jnp.max(jnp.abs(model_noise_rt - model_noise_full)).item(),
)
print(
    "Round-trip μ_x,0 max error:", jnp.max(jnp.abs(prior_mean_rt - prior_mean)).item()
)
print(
    "Round-trip Σ_x,0 max error:",
    jnp.max(jnp.abs(prior_covariance_rt - prior_covariance)).item(),
)

# %% [markdown]
# ## SSM expectation parameters
#
# Given smoothed marginals $(\boldsymbol{\mu}_{x,t \mid T}, \boldsymbol{\Sigma}_{x,t \mid T})$ and cross-covariances
# $[\boldsymbol{\Sigma}_x]_{t+1,t} = \mathrm{Cov}(\mathbf{x}_{t+1}, \mathbf{x}_t \mid \mathbf{y}_{1:T})$, we can compute the
# expectation parameters $(\mathbb{E}[\mathbf{x}], \mathbb{E}[\mathbf{x} \mathbf{x}^\top])$ of the joint Gaussian:
#
# - $\mathbb{E}[\mathbf{x}]$ (concatenated means)
# - $\mathbb{E}[\mathbf{x} \mathbf{x}^\top]$: a `BlockTriDiag` with diagonal blocks $\mathbb{E}[\mathbf{x}_t \mathbf{x}_t^\top]$
#   and sub-diagonal blocks $\mathbb{E}[\mathbf{x}_{t+1} \mathbf{x}_t^\top]$
#
# The cross-covariance is computed from the smoother gain:
# $[\boldsymbol{\Sigma}_x]_{t+1,t} = \boldsymbol{\Sigma}_{x,t+1 \mid T} \mathbf{G}_{\text{smooth},t}^\top$ where $\mathbf{G}_{\text{smooth},t} = \boldsymbol{\Sigma}_{x,t \mid t} \mathbf{F}^\top \boldsymbol{\Sigma}_{x,t+1 \mid t}^{-1}$.

# %%
# Compute cross-covariances from filter/smoother outputs
# G_smooth,t = Σ_x,t|t Fᵀ Σ_x,t+1|t⁻¹
# [Σ_x]_t+1,t = Σ_x,t+1|T G_smooth,tᵀ


def compute_cross_covs(filter_state, smoothed_covs, transition):
    """Compute cross-covariances [Σ_x]_{t+1,t} = Σ_x,t+1|T G_smooth,tᵀ."""
    filtered_covs = filter_state.filtered_covs[:-1]  # (T-1, D_x, D_x)
    predicted_covs = filter_state.predicted_covs[1:]  # (T-1, D_x, D_x)
    smoothed_covs_next = smoothed_covs[1:]  # (T-1, D_x, D_x)

    def _one_step(filtered_cov, predicted_cov, smoothed_cov_next):
        # G_smooth,t = Σ_x,t|t Fᵀ Σ_x,t+1|t⁻¹
        smoother_gain = jnp.linalg.solve(
            predicted_cov.T, (filtered_cov @ transition.T).T
        ).T
        # Σ_x,t+1|T G_smooth,tᵀ
        return smoothed_cov_next @ smoother_gain.T

    return jax.vmap(_one_step)(filtered_covs, predicted_covs, smoothed_covs_next)


cross_covs = compute_cross_covs(par_filter_state, par_smooth_covs, transition)
print("Cross-covariances shape:", cross_covs.shape)

# %%
# Convert to expectation parameters
# (𝔼[x], 𝔼[x xᵀ])
expectation_mean, expectation_second_moment = gaussx.ssm_to_expectations(
    par_smooth_means, par_smooth_covs, cross_covs
)

print("expectation_mean shape:", expectation_mean.shape)
print("expectation_second_moment type:", type(expectation_second_moment).__name__)
print(
    "expectation_second_moment diagonal shape:",
    expectation_second_moment.diagonal.shape,
)
print(
    "expectation_second_moment sub_diagonal shape:",
    expectation_second_moment.sub_diagonal.shape,
)

# %%
# Round-trip: convert back to SSM marginals
means_rt, covs_rt, cross_covs_rt = gaussx.expectations_to_ssm(
    expectation_mean, expectation_second_moment
)

err_m = jnp.max(jnp.abs(means_rt - par_smooth_means)).item()
err_c = jnp.max(jnp.abs(covs_rt - par_smooth_covs)).item()
err_cc = jnp.max(jnp.abs(cross_covs_rt - cross_covs)).item()
print("Round-trip means max error:", err_m)
print("Round-trip covs max error:", err_c)
print("Round-trip cross_covs max error:", err_cc)

# %% [markdown]
# ## Visualizations
#
# ### BlockTriDiag sparsity pattern
#
# The precision matrix of a Gauss-Markov chain has a characteristic
# block-tridiagonal sparsity pattern. We visualize the first few blocks
# to see this structure clearly.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sparsity pattern of the precision (show first 20 blocks = 40x40)
n_show = min(20, num_steps)
d_show = n_show * state_size
prec_sub = dense_precision[:d_show, :d_show]

ax = axes[0]
ax.spy(jnp.abs(prec_sub) > 1e-12, markersize=1, color="C0")
ax.set_title(f"BlockTriDiag sparsity (first {n_show} blocks)")
ax.set_xlabel("Column index")
ax.set_ylabel("Row index")

# Zoom into a 3-block region to see the structure
n_zoom = 3
d_zoom = n_zoom * state_size
prec_zoom = dense_precision[:d_zoom, :d_zoom]

ax = axes[1]
im = ax.imshow(prec_zoom, cmap="RdBu_r", aspect="equal")
ax.set_title(f"Precision blocks (first {n_zoom} blocks)")
ax.set_xlabel("Column index")
ax.set_ylabel("Row index")
# Draw block boundaries
for i in range(1, n_zoom):
    ax.axhline(i * state_size - 0.5, color="k", lw=0.5, ls="--")
    ax.axvline(i * state_size - 0.5, color="k", lw=0.5, ls="--")
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Filtered vs smoothed estimates

# %%
filt_pos_mean = par_filter_state.filtered_means[:, 0]
filt_pos_std = jnp.sqrt(par_filter_state.filtered_covs[:, 0, 0])
smooth_pos_mean = par_smooth_means[:, 0]
smooth_pos_std = jnp.sqrt(par_smooth_covs[:, 0, 0])

filt_vel_mean = par_filter_state.filtered_means[:, 1]
filt_vel_std = jnp.sqrt(par_filter_state.filtered_covs[:, 1, 1])
smooth_vel_mean = par_smooth_means[:, 1]
smooth_vel_std = jnp.sqrt(par_smooth_covs[:, 1, 1])

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Position
ax = axes[0]
ax.scatter(
    times,
    observations[:, 0],
    s=30,
    c="C0",
    edgecolors="k",
    linewidths=0.5,
    alpha=0.25,
    label="Observations",
    zorder=5,
)
ax.plot(times, true_position, "k--", lw=1.5, label="True", zorder=4)
ax.fill_between(
    times,
    filt_pos_mean - 2 * filt_pos_std,
    filt_pos_mean + 2 * filt_pos_std,
    color="C1",
    alpha=0.15,
    label=r"Filtered $\pm 2\sigma$",
)
ax.plot(times, filt_pos_mean, "C1-", lw=2, alpha=0.6, label="Filtered", zorder=3)
ax.fill_between(
    times,
    smooth_pos_mean - 2 * smooth_pos_std,
    smooth_pos_mean + 2 * smooth_pos_std,
    color="C2",
    alpha=0.25,
    label=r"Smoothed $\pm 2\sigma$",
)
ax.plot(times, smooth_pos_mean, "C2-", lw=2, label="Smoothed", zorder=3)
ax.set_ylabel("Position")
ax.set_title("Parallel Kalman: Filtered vs Smoothed (Position)")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

# Velocity
ax = axes[1]
ax.plot(times, true_states[:, 1], "k--", lw=1.5, label="True", zorder=4)
ax.fill_between(
    times,
    filt_vel_mean - 2 * filt_vel_std,
    filt_vel_mean + 2 * filt_vel_std,
    color="C1",
    alpha=0.15,
    label=r"Filtered $\pm 2\sigma$",
)
ax.plot(times, filt_vel_mean, "C1-", lw=2, alpha=0.6, label="Filtered", zorder=3)
ax.fill_between(
    times,
    smooth_vel_mean - 2 * smooth_vel_std,
    smooth_vel_mean + 2 * smooth_vel_std,
    color="C3",
    alpha=0.25,
    label=r"Smoothed $\pm 2\sigma$",
)
ax.plot(times, smooth_vel_mean, "C3-", lw=2, label="Smoothed", zorder=3)
ax.set_xlabel("Time")
ax.set_ylabel("Velocity (latent)")
ax.set_title("Parallel Kalman: Filtered vs Smoothed (Velocity)")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - **Parallel Kalman filter**: `gaussx.parallel_kalman_filter` produces
#   identical results to the sequential `gaussx.kalman_filter` but uses
#   dense array operations suited for GPU/TPU acceleration.
# - **Parallel RTS smoother**: `gaussx.parallel_rts_smoother` likewise
#   matches the sequential smoother.
# - **Joseph-form update**: `gaussx.joseph_update` computes the Kalman
#   covariance update in a numerically stable form that preserves symmetry
#   and positive-definiteness.
# - **Natural parameters**: `gaussx.ssm_to_naturals` extracts the
#   block-tridiagonal precision (as a `BlockTriDiag` operator) and the
#   linear natural parameter from SSM matrices. `gaussx.naturals_to_ssm`
#   inverts this exactly.
# - **Expectation parameters**: `gaussx.ssm_to_expectations` converts
#   smoother marginals (means, covariances, cross-covariances) to
#   expectation parameters with `BlockTriDiag` second moments.
#   `gaussx.expectations_to_ssm` inverts this exactly.
# - The `BlockTriDiag` operator exploits Gauss-Markov sparsity for
#   $O(T D_x^3)$ operations instead of $O((T D_x)^3)$ dense cost.

# %% [markdown]
# ## References
#
# - Sarkka, S. & Garcia-Fernandez, A. F. (2021). Temporal parallelization
#   of Bayesian smoothers. *IEEE Trans. Automatic Control*, 66(1), 299--306.
# - Kalman, R. E. (1960). A new approach to linear filtering and prediction
#   problems. *J. Basic Engineering*, 82(1), 35--45.
# - Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum likelihood
#   estimates of linear dynamic systems. *AIAA Journal*, 3(8), 1445--1450.
# - Sarkka, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge
#   University Press.
