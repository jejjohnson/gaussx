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
# # Kalman Filter and RTS Smoother
#
# This notebook demonstrates the gaussx Kalman filter and Rauch-Tung-Striebel
# (RTS) smoother recipes on a simple linear dynamical system.
#
# **What you'll learn:**
#
# 1. Setting up a linear-Gaussian state-space model
# 2. Running `gaussx.kalman_filter` for online state estimation
# 3. Running `gaussx.rts_smoother` for offline smoothing
# 4. Comparing filtered vs smoothed estimates
# 5. The filter is fully differentiable via JAX

# %% [markdown]
# ## Background
#
# A linear-Gaussian state-space model (SSM) is defined by:
#
# $$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\eta)$$
#
# $$\mathbf{y}_t = \mathbf{H} \mathbf{x}_t + \boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\varepsilon)$$
#
# Here $t = 1, \ldots, T$ indexes the time steps, $\mathbf{x}_t \in \mathbb{R}^{D_x}$ is the state, $\mathbf{y}_t \in \mathbb{R}^{D_y}$ the observation, $\mathbf{F} \in \mathbb{R}^{D_x \times D_x}$ the transition, $\mathbf{H} \in \mathbb{R}^{D_y \times D_x}$ the observation operator, $\boldsymbol{\Sigma}_\eta \in \mathbb{R}^{D_x \times D_x}$ the model-noise covariance and $\boldsymbol{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$ the observation-noise covariance.
#
# The **Kalman filter** computes the filtering distribution
# $p(\mathbf{x}_t \mid \mathbf{y}_{1:t})$ in a single forward pass.
# The **RTS smoother** then refines these estimates using future observations,
# giving the smoothing distribution $p(\mathbf{x}_t \mid \mathbf{y}_{1:T})$.
#
# The smoother always has lower (or equal) posterior variance than the filter,
# because it conditions on strictly more data.
#
# ### Kalman filter equations
#
# The filter alternates between a **predict** step and an **update** step:
#
# **Predict:**
#
# $$\boldsymbol{\mu}_{x,t \mid t-1} = \mathbf{F} \boldsymbol{\mu}_{x,t-1 \mid t-1}$$
#
# $$\boldsymbol{\Sigma}_{x,t \mid t-1} = \mathbf{F} \boldsymbol{\Sigma}_{x,t-1 \mid t-1} \mathbf{F}^\top + \boldsymbol{\Sigma}_\eta$$
#
# **Update:**
#
# $$\boldsymbol{\Sigma}_{y,t} = \mathbf{H} \boldsymbol{\Sigma}_{x,t \mid t-1} \mathbf{H}^\top + \boldsymbol{\Sigma}_\varepsilon \quad \text{(innovation covariance)}$$
#
# $$\mathbf{G}_t = \boldsymbol{\Sigma}_{x,t \mid t-1} \mathbf{H}^\top \boldsymbol{\Sigma}_{y,t}^{-1} \quad \text{(Kalman gain)}$$
#
# $$\mathbf{r}_t = \mathbf{y}_t - \mathbf{H} \boldsymbol{\mu}_{x,t \mid t-1} \quad \text{(residual)}$$
#
# $$\boldsymbol{\mu}_{x,t \mid t} = \boldsymbol{\mu}_{x,t \mid t-1} + \mathbf{G}_t \mathbf{r}_t$$
#
# $$\boldsymbol{\Sigma}_{x,t \mid t} = (\mathbf{I} - \mathbf{G}_t \mathbf{H}) \boldsymbol{\Sigma}_{x,t \mid t-1}$$
#
# Note that each step involves a linear system solve (for $\mathbf{G}_t$), which is
# where gaussx primitives are used.

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
# We use a 2D state ($D_x = 2$) $\mathbf{x}_t = [\text{position}, \text{velocity}]^\top$ with
# a **damped oscillator** transition (discretized spring-mass-damper).
# Only position is observed with noise ($D_y = 1$).
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
# ## Generate data
#
# We simulate the true trajectory **from the model itself**, so the
# Kalman filter's assumptions are satisfied. This shows the filter
# performing as designed — tracking a randomly evolving state from
# noisy observations.

# %%
key = jax.random.PRNGKey(42)


# Simulate states from the model
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
# ## Run Kalman filter
#
# `gaussx.kalman_filter` takes the model matrices and observations,
# returning a `FilterState` with filtered means, covariances, and the
# total log-likelihood.

# %%
# Initial state: zero mean, moderate uncertainty
# μ_x,0 = 0
prior_mean = jnp.zeros(2)
# Σ_x,0 = 4 I
prior_covariance = jnp.eye(2) * 4.0

filter_state = gaussx.kalman_filter(
    transition,
    obs_operator,
    model_noise_covariance,
    obs_noise_covariance,
    observations,
    prior_mean,
    prior_covariance,
)

print("Filtered means shape:", filter_state.filtered_means.shape)
print("Filtered covs shape:", filter_state.filtered_covs.shape)
print("Log-likelihood:", filter_state.log_likelihood)

# %% [markdown]
# ## Plot filtered results
#
# The filtered estimate tracks the true position closely despite the noisy
# observations. The shaded band shows the $\pm 2\sigma$ credible interval.

# %%
filt_pos_mean = filter_state.filtered_means[:, 0]
filt_pos_std = jnp.sqrt(filter_state.filtered_covs[:, 0, 0])
filt_vel_mean = filter_state.filtered_means[:, 1]
filt_vel_std = jnp.sqrt(filter_state.filtered_covs[:, 1, 1])

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Position
ax = axes[0]
ax.scatter(
    times,
    observations[:, 0],
    s=30,
    c="C0",
    alpha=0.35,
    edgecolors="k",
    linewidths=0.5,
    label="Observations",
    zorder=5,
)
ax.plot(times, true_position, "k--", lw=1.5, label="True position", zorder=4)
ax.plot(times, filt_pos_mean, "C1-", lw=2, label="Filtered mean", zorder=3)
ax.fill_between(
    times,
    filt_pos_mean - 2 * filt_pos_std,
    filt_pos_mean + 2 * filt_pos_std,
    color="C1",
    alpha=0.2,
    label=r"$\pm 2\sigma$",
)
ax.set_ylabel("Position")
ax.set_title("Kalman Filter: Filtered Estimates")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

# Velocity (inferred, never directly observed)
ax = axes[1]
ax.plot(times, true_states[:, 1], "k--", lw=1.5, label="True velocity", zorder=4)
ax.plot(times, filt_vel_mean, "C3-", lw=2, label="Filtered mean", zorder=3)
ax.fill_between(
    times,
    filt_vel_mean - 2 * filt_vel_std,
    filt_vel_mean + 2 * filt_vel_std,
    color="C3",
    alpha=0.2,
    label=r"$\pm 2\sigma$",
)
ax.set_xlabel("Time")
ax.set_ylabel("Velocity (latent)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Run RTS smoother
#
# The RTS smoother takes the filter output and refines the estimates
# using a backward pass. This produces tighter credible intervals,
# especially in the middle of the time series.
#
# The smoother computes a backward recursion from $t = T-1$ down to $t = 1$:
#
# $$\mathbf{G}_{\text{smooth},t} = \boldsymbol{\Sigma}_{x,t \mid t} \mathbf{F}^\top \boldsymbol{\Sigma}_{x,t+1 \mid t}^{-1}$$
#
# $$\boldsymbol{\mu}_{x,t \mid T} = \boldsymbol{\mu}_{x,t \mid t} + \mathbf{G}_{\text{smooth},t} (\boldsymbol{\mu}_{x,t+1 \mid T} - \mathbf{F} \boldsymbol{\mu}_{x,t \mid t})$$
#
# $$\boldsymbol{\Sigma}_{x,t \mid T} = \boldsymbol{\Sigma}_{x,t \mid t} + \mathbf{G}_{\text{smooth},t} (\boldsymbol{\Sigma}_{x,t+1 \mid T} - \boldsymbol{\Sigma}_{x,t+1 \mid t}) \mathbf{G}_{\text{smooth},t}^\top$$
#
# The smoother refines filter estimates by incorporating future observations.
# This optimal smoother was derived by Rauch, Tung, & Striebel (1965).

# %%
smoothed_means, smoothed_covs = gaussx.rts_smoother(filter_state, transition)

print("Smoothed means shape:", smoothed_means.shape)
print("Smoothed covs shape:", smoothed_covs.shape)

# %% [markdown]
# ## Plot filtered vs smoothed
#
# The smoother uncertainty is everywhere less than or equal to the filter
# uncertainty, because it conditions on all observations $\mathbf{y}_{1:T}$.

# %%
smooth_pos_mean = smoothed_means[:, 0]
smooth_pos_std = jnp.sqrt(smoothed_covs[:, 0, 0])
smooth_vel_mean = smoothed_means[:, 1]
smooth_vel_std = jnp.sqrt(smoothed_covs[:, 1, 1])

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Top: position — overlay filtered and smoothed
ax = axes[0]
ax.scatter(
    times,
    observations[:, 0],
    s=30,
    c="C0",
    alpha=0.25,
    edgecolors="k",
    linewidths=0.5,
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
ax.plot(times, filt_pos_mean, "C1-", lw=1, alpha=0.6, label="Filtered", zorder=3)
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
ax.set_title("Filter vs Smoother: Position")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

# Bottom: velocity — overlay filtered and smoothed
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
ax.plot(times, filt_vel_mean, "C1-", lw=1, alpha=0.6, label="Filtered", zorder=3)
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
ax.set_title("Filter vs Smoother: Velocity")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Differentiability
#
# The Kalman filter is implemented via `jax.lax.scan`, so it is fully
# differentiable. We can compute gradients of the log-likelihood with
# respect to model parameters — useful for learning SSM parameters
# via gradient-based optimization.


# %%
def neg_log_likelihood(log_obs_noise_var):
    """Negative log-likelihood as a function of log observation noise variance."""
    # Σ_ε = exp(log σ_ε²) I
    obs_noise_covariance_param = jnp.exp(log_obs_noise_var) * jnp.eye(1)
    filter_state_param = gaussx.kalman_filter(
        transition,
        obs_operator,
        model_noise_covariance,
        obs_noise_covariance_param,
        observations,
        prior_mean,
        prior_covariance,
    )
    return -filter_state_param.log_likelihood


# Evaluate at the true value
log_obs_noise_true = jnp.log(obs_noise_covariance[0, 0])
nll = neg_log_likelihood(log_obs_noise_true)
grad_nll = jax.grad(neg_log_likelihood)(log_obs_noise_true)

print(f"log(Σ_ε) = {log_obs_noise_true:.4f}")
print(f"Negative log-likelihood = {nll:.4f}")
print(f"Gradient d(-LL)/d(log Σ_ε) = {grad_nll:.4f}")

# %% [markdown]
# ## Summary
#
# - `gaussx.kalman_filter` implements the standard Kalman filter forward pass,
#   returning filtered means, covariances, and the total log-likelihood.
# - `gaussx.rts_smoother` refines filtered estimates using a backward pass,
#   producing tighter credible intervals.
# - Both are implemented with `jax.lax.scan` and are fully compatible with
#   JAX transforms: `jit`, `vmap`, and `grad` all work out of the box.
# - This makes gaussx suitable for learning SSM parameters via gradient-based
#   optimization of the log-likelihood.

# %% [markdown]
# ## References
#
# - Kalman, R. E. (1960). A new approach to linear filtering and prediction
#   problems. *J. Basic Engineering*, 82(1), 35--45.
# - Rauch, H. E., Tung, F., & Striebel, C. T. (1965). Maximum likelihood
#   estimates of linear dynamic systems. *AIAA Journal*, 3(8), 1445--1450.
# - Sarkka, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge
#   University Press.
# - Anderson, B. D. O. & Moore, J. B. (1979). *Optimal Filtering*.
#   Prentice-Hall.
