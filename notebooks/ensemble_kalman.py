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
# # Ensemble Kalman Filter from Scratch
#
# This notebook builds an **Ensemble Kalman Filter (EnKF)** step by step,
# using gaussx ensemble primitives on the chaotic Lorenz-63 system.
#
# **What you'll learn:**
#
# 1. Computing empirical covariance as a structured `LowRankUpdate` operator
#    with `gaussx.ensemble_covariance`
# 2. Computing cross-covariance between state and observation ensembles
#    with `gaussx.ensemble_cross_covariance`
# 3. Numerically stable covariance updates with `gaussx.joseph_update`
# 4. Deriving process noise from a stationary covariance with
#    `gaussx.process_noise_covariance`

# %% [markdown]
# ## Background
#
# The **Ensemble Kalman Filter** (Evensen, 1994) is a Monte Carlo
# approximation to the Kalman filter designed for high-dimensional,
# nonlinear systems. Instead of propagating a mean and covariance matrix
# explicitly, it maintains an *ensemble* of $N_{\mathcal{E}}$ members
# $\{\mathbf{x}^{(e)}\}_{e \in \mathcal{E}}$, $\mathbf{x}^{(e)} \in \mathbb{R}^{D_x}$, that collectively represent the state
# distribution.
#
# The key idea: at each step the forecast covariance is estimated
# *empirically* from the ensemble spread, so no linearization (tangent
# linear model) is needed. The update step then applies the standard
# Kalman gain formula using these empirical statistics.
#
# ### EnKF algorithm sketch
#
# **Forecast:**
# Propagate each member through the (possibly nonlinear) dynamics $f$, with model noise $\boldsymbol{\eta}^{(e)}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\eta)$:
#
# $$\mathbf{x}^{(e)}_{t \mid t-1} = f(\mathbf{x}^{(e)}_{t-1 \mid t-1}) + \boldsymbol{\eta}^{(e)}_t$$
#
# **Analysis:** with the observation operator $\mathbf{H} \in \mathbb{R}^{D_y \times D_x}$, the predicted observation $\mathbf{y}^{(e)} = \mathbf{H} \mathbf{x}^{(e)}_{t \mid t-1}$, the ensemble means $\bar{\mathbf{x}}$, $\bar{\mathbf{y}}$ and the anomalies $\tilde{\mathbf{x}}^{(e)} = \mathbf{x}^{(e)} - \bar{\mathbf{x}}$, $\tilde{\mathbf{y}}^{(e)} = \mathbf{y}^{(e)} - \bar{\mathbf{y}}$:
#
# 1. Compute ensemble covariance:
#    $\hat{\boldsymbol{\Sigma}}_x = \frac{1}{N_{\mathcal{E}}} \sum_{e \in \mathcal{E}} \tilde{\mathbf{x}}^{(e)} \tilde{\mathbf{x}}^{(e) \top}$
# 2. Compute cross-covariance between state and predicted observations:
#    $\hat{\boldsymbol{\Sigma}}_{xy} = \frac{1}{N_{\mathcal{E}}} \sum_{e \in \mathcal{E}} \tilde{\mathbf{x}}^{(e)} \tilde{\mathbf{y}}^{(e) \top}$
# 3. Compute innovation covariance:
#    $\hat{\boldsymbol{\Sigma}}_y = \frac{1}{N_{\mathcal{E}}} \sum_{e \in \mathcal{E}} \tilde{\mathbf{y}}^{(e)} \tilde{\mathbf{y}}^{(e) \top} + \boldsymbol{\Sigma}_\varepsilon$
# 4. Kalman gain: $\mathbf{G} = \hat{\boldsymbol{\Sigma}}_{xy} \hat{\boldsymbol{\Sigma}}_y^{-1}$
# 5. Perturbed-observation update, with $\boldsymbol{\varepsilon}^{(e)} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\varepsilon)$:
#    $\mathbf{x}^{(e)}_{t \mid t} = \mathbf{x}^{(e)}_{t \mid t-1} + \mathbf{G} (\mathbf{y}_t + \boldsymbol{\varepsilon}^{(e)} - \mathbf{H} \mathbf{x}^{(e)}_{t \mid t-1})$

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
# ## Define the Lorenz-63 system
#
# The Lorenz-63 system is a classic chaotic dynamical system with state $\mathbf{x} = (x_1, x_2, x_3)$, so $D_x = 3$:
#
# $$\frac{\mathrm{d} x_1}{\mathrm{d} t} = \sigma (x_2 - x_1)$$
#
# $$\frac{\mathrm{d} x_2}{\mathrm{d} t} = x_1 (\rho - x_3) - x_2$$
#
# $$\frac{\mathrm{d} x_3}{\mathrm{d} t} = x_1 x_2 - \beta x_3$$
#
# With the standard parameters $\sigma = 10$, $\rho = 28$, $\beta = 8/3$,
# the system exhibits the famous butterfly attractor.
# We discretize with a 4th-order Runge-Kutta integrator.

# %%
# Lorenz-63 parameters
sigma_l = 10.0
rho = 28.0
beta = 8.0 / 3.0
# Δt
time_step = 0.01


def lorenz63(state):
    """Lorenz-63 right-hand side."""
    x1, x2, x3 = state[0], state[1], state[2]
    # dx₁/dt = σ (x₂ − x₁)
    dx1 = sigma_l * (x2 - x1)
    # dx₂/dt = x₁ (ρ − x₃) − x₂
    dx2 = x1 * (rho - x3) - x2
    # dx₃/dt = x₁ x₂ − β x₃
    dx3 = x1 * x2 - beta * x3
    return jnp.array([dx1, dx2, dx3])


def rk4_step(state, _dt=time_step):
    """Single RK4 integration step."""
    k1 = lorenz63(state)
    k2 = lorenz63(state + 0.5 * _dt * k1)
    k3 = lorenz63(state + 0.5 * _dt * k2)
    k4 = lorenz63(state + _dt * k3)
    return state + (_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# Quick sanity check: integrate a short trajectory
test_state = jnp.array([1.0, 1.0, 1.0])
test_state_next = rk4_step(test_state)
print("One RK4 step: ", test_state, " -> ", test_state_next)

# %% [markdown]
# ## Generate the true trajectory and observations
#
# We observe only the $x_1$ and $x_2$ components of the Lorenz state with
# additive Gaussian noise ($D_y = 2$). The $x_3$ component must be inferred.

# %%
# T
num_steps = 500  # number of time steps
# σ_ε
obs_noise_std = 2.0

# Observation model: observe x₁ and x₂ only
# H ∈ ℝ^{D_y×D_x}
obs_operator = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# Σ_ε = σ_ε² I
obs_noise_covariance = obs_noise_std**2 * jnp.eye(2)

# True initial condition (on the attractor)
true_init_state = jnp.array([1.508, -1.531, 25.46])

key = jax.random.PRNGKey(42)


def generate_truth(carry, key_t):
    """Propagate true state and generate noisy observation."""
    true_state = carry
    next_true_state = rk4_step(true_state)
    # ε_t ~ 𝒩(0, Σ_ε)
    noise = obs_noise_std * jax.random.normal(key_t, shape=(2,))
    # y_t = H x_t + ε_t
    obs_t = obs_operator @ next_true_state + noise
    return next_true_state, (next_true_state, obs_t)


keys_truth = jax.random.split(key, num_steps)
_, (true_states, observations) = jax.lax.scan(
    generate_truth, true_init_state, keys_truth
)

times = jnp.arange(num_steps) * time_step
print("True trajectory shape:", true_states.shape)
print("Observations shape:", observations.shape)

# %% [markdown]
# ## Initialize the ensemble
#
# We draw $N_{\mathcal{E}} = 50$ members from a Gaussian centered at the true initial
# state with moderate uncertainty. In practice the initial ensemble
# would come from a climatological prior.

# %%
# N_ℰ
num_members = 50  # ensemble size
# D_x
state_size = 3  # state dimension

# Σ_x,0 = 2 I
init_covariance = jnp.diag(jnp.array([2.0, 2.0, 2.0]))

key, subkey = jax.random.split(key)
# L Lᵀ = Σ_x,0
init_cholesky = jnp.linalg.cholesky(init_covariance)
# x⁽ᵉ⁾ = x_true,0 + L z⁽ᵉ⁾,  z⁽ᵉ⁾ ~ 𝒩(0, I)
ensemble = (
    true_init_state[None, :]
    + jax.random.normal(subkey, shape=(num_members, state_size)) @ init_cholesky.T
)

print("Initial ensemble shape:", ensemble.shape)
print("Ensemble mean:", jnp.mean(ensemble, axis=0))

# %% [markdown]
# ## Inspect the ensemble covariance operator
#
# Before running the filter, let's see what `gaussx.ensemble_covariance`
# returns. It produces a `LowRankUpdate` operator -- a structured
# representation of $\hat{\boldsymbol{\Sigma}}_x = (1/N_{\mathcal{E}}) \sum_{e \in \mathcal{E}} \tilde{\mathbf{x}}^{(e)} \tilde{\mathbf{x}}^{(e) \top}$
# where $\tilde{\mathbf{x}}^{(e)} = \mathbf{x}^{(e)} - \bar{\mathbf{x}}$. This is rank $\leq N_{\mathcal{E}}$ and
# avoids materializing the full $D_x \times D_x$ matrix (crucial when $D_x$
# is large).

# %%
ensemble_cov_op = gaussx.ensemble_covariance(ensemble)
print("Type:", type(ensemble_cov_op).__name__)
print("Materialized covariance:\n", ensemble_cov_op.as_matrix())

# Compare with naive computation
ensemble_mean = jnp.mean(ensemble, axis=0)
# x̃⁽ᵉ⁾ = x⁽ᵉ⁾ − x̄
anomalies = ensemble - ensemble_mean[None, :]
# Σ̂_x = (1/N_ℰ) ∑ₑ x̃⁽ᵉ⁾ x̃⁽ᵉ⁾ᵀ
naive_covariance = (anomalies.T @ anomalies) / num_members
print("\nNaive covariance:\n", naive_covariance)
print(
    "Max difference:", jnp.max(jnp.abs(ensemble_cov_op.as_matrix() - naive_covariance))
)

# %% [markdown]
# ## Run the Ensemble Kalman Filter
#
# The main loop: forecast each member through the nonlinear Lorenz
# dynamics, then perform the EnKF analysis update using gaussx ensemble
# primitives.
#
# We add a small amount of model noise (additive inflation) to prevent
# ensemble collapse, a common practical technique.

# %%
# σ_η
model_noise_std = 0.1
# Σ_η = σ_η² I
model_noise_covariance = model_noise_std**2 * jnp.eye(state_size)


def enkf_step(carry, inputs):
    """Single EnKF forecast + analysis step."""
    ensemble_prev, _key_prev = carry
    obs_t, key_t = inputs

    k1, k2, k3 = jax.random.split(key_t, 3)

    # --- Forecast: propagate each member through nonlinear dynamics ---
    ensemble_pred = jax.vmap(rk4_step)(ensemble_prev)

    # Add model noise (additive inflation)
    # η⁽ᵉ⁾ ~ 𝒩(0, Σ_η)
    model_noise = model_noise_std * jax.random.normal(
        k1, shape=(num_members, state_size)
    )
    # x⁽ᵉ⁾_t|t−1 = f(x⁽ᵉ⁾_t−1|t−1) + η⁽ᵉ⁾
    ensemble_pred = ensemble_pred + model_noise

    # --- Analysis: compute ensemble statistics using gaussx ---

    # Predicted observation ensemble: y⁽ᵉ⁾ = H x⁽ᵉ⁾
    obs_pred = ensemble_pred @ obs_operator.T  # (N_ℰ, D_y)

    # Cross-covariance Σ̂_xy between state and predicted observations
    cross_covariance = gaussx.ensemble_cross_covariance(
        ensemble_pred, obs_pred
    )  # (D_x, D_y)

    # Innovation covariance Σ̂_y = (1/N_ℰ) ∑ₑ ỹ⁽ᵉ⁾ ỹ⁽ᵉ⁾ᵀ + Σ_ε
    obs_cov_op = gaussx.ensemble_covariance(obs_pred)
    obs_covariance = obs_cov_op.as_matrix() + obs_noise_covariance  # (D_y, D_y)

    # Kalman gain G = Σ̂_xy Σ̂_y⁻¹
    gain = cross_covariance @ jnp.linalg.inv(obs_covariance)  # (D_x, D_y)

    # Perturbed-observation update
    # ε⁽ᵉ⁾ ~ 𝒩(0, Σ_ε)
    obs_noise = obs_noise_std * jax.random.normal(k2, shape=(num_members, 2))
    # y_t + ε⁽ᵉ⁾ − H x⁽ᵉ⁾_t|t−1
    innovations = obs_t[None, :] + obs_noise - obs_pred  # (N_ℰ, D_y)
    # x⁽ᵉ⁾_t|t = x⁽ᵉ⁾_t|t−1 + G (y_t + ε⁽ᵉ⁾ − H x⁽ᵉ⁾_t|t−1)
    ensemble_upd = ensemble_pred + innovations @ gain.T  # (N_ℰ, D_x)

    # Ensemble mean for diagnostics
    mean_upd = jnp.mean(ensemble_upd, axis=0)

    return (ensemble_upd, k3), (mean_upd, ensemble_upd)


# Prepare inputs
keys_enkf = jax.random.split(jax.random.PRNGKey(123), num_steps)
inputs = (observations, keys_enkf)

# Run the filter
init_carry = (ensemble, jax.random.PRNGKey(0))
_, (enkf_means, enkf_ensembles) = jax.lax.scan(enkf_step, init_carry, inputs)

print("EnKF means shape:", enkf_means.shape)
print("EnKF ensembles shape:", enkf_ensembles.shape)

# %% [markdown]
# ## Joseph-form covariance update
#
# The standard EnKF implicitly updates the covariance via the ensemble.
# For diagnostics or hybrid filters, one sometimes needs the *explicit*
# updated covariance. The Joseph form
#
# $$\boldsymbol{\Sigma}_{x,t \mid t} = (\mathbf{I} - \mathbf{G} \mathbf{H}) \boldsymbol{\Sigma}_{x,t \mid t-1} (\mathbf{I} - \mathbf{G} \mathbf{H})^\top + \mathbf{G} \boldsymbol{\Sigma}_\varepsilon \mathbf{G}^\top$$
#
# is numerically more stable than the naive $(\mathbf{I} - \mathbf{G} \mathbf{H}) \boldsymbol{\Sigma}_{x,t \mid t-1}$
# formula. Let's demonstrate `gaussx.joseph_update` on the final step.

# %%
# Use the final forecast ensemble to get Σ_x,t|t−1
final_ensemble_pred = jax.vmap(rk4_step)(enkf_ensembles[-1])
pred_cov_op = gaussx.ensemble_covariance(final_ensemble_pred)
pred_covariance = pred_cov_op.as_matrix()

# Recompute the Kalman gain for this step
obs_pred_final = final_ensemble_pred @ obs_operator.T
cross_covariance_final = gaussx.ensemble_cross_covariance(
    final_ensemble_pred, obs_pred_final
)
obs_covariance_final = (
    gaussx.ensemble_covariance(obs_pred_final).as_matrix() + obs_noise_covariance
)
# G = Σ̂_xy Σ̂_y⁻¹
gain_final = cross_covariance_final @ jnp.linalg.inv(obs_covariance_final)

# Joseph-form update
# Σ_x,t|t = (I − G H) Σ_x,t|t−1 (I − G H)ᵀ + G Σ_ε Gᵀ
joseph_covariance = gaussx.joseph_update(
    pred_covariance, gain_final, obs_operator, obs_noise_covariance
)

# Compare with naive formula
# (I − G H)
identity_minus_gain_obs = jnp.eye(state_size) - gain_final @ obs_operator
# Σ_x,t|t = (I − G H) Σ_x,t|t−1
naive_updated_covariance = identity_minus_gain_obs @ pred_covariance

print("Joseph-form updated covariance:\n", joseph_covariance)
print("\nNaive updated covariance:\n", naive_updated_covariance)
print(
    "\nJoseph form is symmetric:", jnp.allclose(joseph_covariance, joseph_covariance.T)
)

# %% [markdown]
# ## Process noise from stationary covariance
#
# For a *linearized* version of the Lorenz system, we can compute the
# Jacobian at a fixed point and use `gaussx.process_noise_covariance`
# to derive the model noise $\boldsymbol{\Sigma}_\eta$ that is consistent with a given
# stationary covariance $\boldsymbol{\Sigma}_{x,\infty}$:
#
# $$\boldsymbol{\Sigma}_\eta = \boldsymbol{\Sigma}_{x,\infty} - \mathbf{F} \boldsymbol{\Sigma}_{x,\infty} \mathbf{F}^\top$$
#
# This is useful when you know the long-run statistics but need the
# step-to-step noise for a filter.

# %%
# Jacobian of the Lorenz system at the origin (one of three fixed points)
# F_c = ∂f/∂x at x = 0 = [[−σ, σ, 0], [ρ, −1, 0], [0, 0, −β]]
drift_jacobian = jnp.array(
    [[-sigma_l, sigma_l, 0.0], [rho, -1.0, 0.0], [0.0, 0.0, -beta]]
)

# Discretize the linearized system (first-order: F ~ I + Δt F_c)
# F = I + Δt F_c
linear_transition = jnp.eye(state_size) + time_step * drift_jacobian
print("Linearized transition F:\n", linear_transition)

# Choose a stationary covariance (e.g., from long-run ensemble statistics)
# Σ_x,∞
stationary_covariance = jnp.cov(true_states.T)
print("\nEmpirical stationary covariance Σ_x,∞:\n", stationary_covariance)

# Compute the implied model noise
# Σ_η = Σ_x,∞ − F Σ_x,∞ Fᵀ
implied_model_noise = gaussx.process_noise_covariance(
    linear_transition, stationary_covariance
)
print("\nImplied process noise Σ_η:\n", implied_model_noise)
print("Σ_η is symmetric:", jnp.allclose(implied_model_noise, implied_model_noise.T))

# %% [markdown]
# ## Plot: 3D Lorenz attractor
#
# The true trajectory, ensemble mean estimate, and ensemble spread
# projected onto the Lorenz attractor.

# %%
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    true_states[:, 0],
    true_states[:, 1],
    true_states[:, 2],
    "k-",
    lw=0.6,
    alpha=0.5,
    label="True trajectory",
)
ax.plot(
    enkf_means[:, 0],
    enkf_means[:, 1],
    enkf_means[:, 2],
    "C1-",
    lw=0.8,
    alpha=0.8,
    label="EnKF mean",
)

# Plot a few ensemble members at the final time
for e in range(min(10, num_members)):
    member_trajectory = enkf_ensembles[:, e, :]
    ax.plot(
        member_trajectory[:, 0],
        member_trajectory[:, 1],
        member_trajectory[:, 2],
        "C0-",
        lw=0.2,
        alpha=0.15,
    )

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
ax.set_title("Lorenz-63: True vs EnKF Estimate")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="major", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Plot: time series with ensemble spread
#
# Each state component over time, showing the true value, ensemble mean,
# and $\pm 2\sigma$ spread computed from the ensemble.

# %%
# Compute ensemble standard deviation at each time step
enkf_std = jnp.std(enkf_ensembles, axis=1)  # (T, 3)

labels = ["$x_1$", "$x_2$", "$x_3$"]
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for d, (ax, label) in enumerate(zip(axes, labels, strict=True)):
    ax.plot(times, true_states[:, d], "k--", lw=1.5, label="True", zorder=4)
    ax.plot(times, enkf_means[:, d], "C1-", lw=2, label="EnKF mean", zorder=3)
    ax.fill_between(
        times,
        enkf_means[:, d] - 2 * enkf_std[:, d],
        enkf_means[:, d] + 2 * enkf_std[:, d],
        color="C1",
        alpha=0.2,
        label=r"$\pm 2\sigma$",
    )
    if d < 2:
        ax.scatter(
            times,
            observations[:, d],
            s=30,
            c="C0",
            edgecolors="k",
            linewidths=0.5,
            alpha=0.2,
            label="Observations",
            zorder=5,
        )
    ax.set_ylabel(label)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.minorticks_on()

axes[-1].set_xlabel("Time")
axes[0].set_title("EnKF State Estimates with Ensemble Spread")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Plot: RMSE over time

# %%
rmse = jnp.sqrt(jnp.mean((enkf_means - true_states) ** 2, axis=1))

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(times, rmse, "C3-", lw=2)
ax.set_xlabel("Time")
ax.set_ylabel("RMSE")
ax.set_title("EnKF Root Mean Square Error over Time")
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()
plt.tight_layout()
plt.show()

print(f"Mean RMSE: {jnp.mean(rmse):.3f}")
print(f"Final RMSE: {rmse[-1]:.3f}")

# %% [markdown]
# ## Summary
#
# - **`gaussx.ensemble_covariance(particles)`** returns a `LowRankUpdate`
#   operator representing the empirical covariance. This structured
#   representation avoids materializing a full $D_x \times D_x$ matrix and
#   enables efficient Woodbury-based solves when the ensemble size
#   $N_{\mathcal{E}} \ll D_x$.
#
# - **`gaussx.ensemble_cross_covariance(particles_theta, particles_G)`**
#   computes the cross-covariance matrix between two ensembles, giving
#   the $D_x \times D_y$ array needed for the Kalman gain.
#
# - **`gaussx.joseph_update(P_pred, K, H, R)`** provides the numerically
#   stable Joseph-form covariance update, guaranteeing symmetry even
#   when the Kalman gain is approximate.
#
# - **`gaussx.process_noise_covariance(A, Pinf)`** derives the model
#   noise $\boldsymbol{\Sigma}_\eta = \boldsymbol{\Sigma}_{x,\infty} - \mathbf{F} \boldsymbol{\Sigma}_{x,\infty} \mathbf{F}^\top$ from a stationary covariance,
#   useful when calibrating a filter from long-run statistics.
#
# Together these primitives let you build ensemble Kalman methods that
# are fully compatible with JAX transforms (`jit`, `vmap`, `grad`)
# while preserving structured linear algebra where it matters.

# %% [markdown]
# ## References
#
# - Evensen, G. (1994). Sequential data assimilation with a nonlinear
#   quasi-geostrophic model using Monte Carlo methods to forecast error
#   statistics. *J. Geophys. Res.*, 99(C5), 10143--10162.
# - Lorenz, E. N. (1963). Deterministic nonperiodic flow.
#   *J. Atmos. Sci.*, 20(2), 130--141.
# - Bucy, R. S. & Joseph, P. D. (1968). *Filtering for Stochastic
#   Processes with Applications to Guidance*. Interscience.
# - Sarkka, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge
#   University Press.
