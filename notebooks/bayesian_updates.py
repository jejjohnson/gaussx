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
# # Bayesian Updates from Scratch
#
# This notebook derives and implements Bayes' rule for Gaussian
# distributions, showing how gaussx primitives keep the linear algebra
# clean.
#
# **What you'll learn:**
#
# 1. Bayes' rule for Gaussian distributions -- the math and the code
# 2. Sequential Bayesian updating with natural parameters
# 3. How gaussx primitives make the linear algebra clean
# 4. From single observations to full GP regression via Bayes' rule
#
# These are foundational results in Bayesian statistics and form the
# computational core of Kalman filtering, GP regression, and variational
# inference.

# %% [markdown]
# ## 1. The Gaussian Bayes Rule
#
# Suppose we have a Gaussian prior on the state $\mathbf{x} \in \mathbb{R}^{D_x}$ and a linear-Gaussian likelihood for the observation $\mathbf{y} \in \mathbb{R}^{D_y}$, with observation operator $\mathbf{H} \in \mathbb{R}^{D_y \times D_x}$ and precisions $\boldsymbol{\Lambda}_x = \boldsymbol{\Sigma}_x^{-1}$, $\boldsymbol{\Lambda}_\varepsilon = \boldsymbol{\Sigma}_\varepsilon^{-1}$:
#
# $$\text{Prior:} \quad p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_x, \boldsymbol{\Sigma}_x)$$
#
# $$\text{Likelihood:} \quad p(\mathbf{y} \mid \mathbf{x}) = \mathcal{N}(\mathbf{H} \mathbf{x}, \boldsymbol{\Sigma}_\varepsilon)$$
#
# Bayes' rule gives a Gaussian posterior $p(\mathbf{x} \mid \mathbf{y}) = \mathcal{N}(\boldsymbol{\mu}_{x \mid y}, \boldsymbol{\Sigma}_{x \mid y})$ with:
#
# $$\boldsymbol{\Lambda}_{x \mid y} = \boldsymbol{\Lambda}_x + \mathbf{H}^\top \boldsymbol{\Lambda}_\varepsilon \mathbf{H} \quad \text{(precisions add!)}$$
#
# $$\boldsymbol{\mu}_{x \mid y} = \boldsymbol{\Sigma}_{x \mid y} (\boldsymbol{\Lambda}_x \boldsymbol{\mu}_x + \mathbf{H}^\top \boldsymbol{\Lambda}_\varepsilon \mathbf{y})$$
#
# In **natural parameters** $(\boldsymbol{\Lambda}_x \boldsymbol{\mu}_x, -\frac{1}{2} \boldsymbol{\Lambda}_x)$ the update is pure addition of the likelihood's natural parameters $(\boldsymbol{\Lambda\mu}_\text{lik}, -\frac{1}{2} \boldsymbol{\Lambda}_\text{lik})$, with $\boldsymbol{\Lambda\mu}_\text{lik} = \mathbf{H}^\top \boldsymbol{\Lambda}_\varepsilon \mathbf{y}$ and $\boldsymbol{\Lambda}_\text{lik} = \mathbf{H}^\top \boldsymbol{\Lambda}_\varepsilon \mathbf{H}$:
#
# $$\boldsymbol{\Lambda}_{x \mid y} \boldsymbol{\mu}_{x \mid y} = \boldsymbol{\Lambda}_x \boldsymbol{\mu}_x + \boldsymbol{\Lambda\mu}_\text{lik}$$
#
# $$-\frac{1}{2} \boldsymbol{\Lambda}_{x \mid y} = -\frac{1}{2} \boldsymbol{\Lambda}_x - \frac{1}{2} \boldsymbol{\Lambda}_\text{lik}$$
#
# This is exactly why the exponential family form is so powerful:
# incorporating new evidence is just vector addition.

# %%
from __future__ import annotations

import warnings


warnings.filterwarnings("ignore", message=r".*IProgress.*")

import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import numpy as np

import gaussx
from gaussx import GaussianExpFam


jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## 2. Setup
#
# We set up a 2D problem ($D_x = 2$): estimating an unknown location
# $\mathbf{x} = (x_1, x_2)$ from noisy measurements.

# %%
key = jax.random.PRNGKey(42)

# Ground truth location we are trying to estimate
# x_true ∈ ℝ^{D_x}
x_true = jnp.array([3.0, -1.0])


def ellipse_points(mean, covariance, n_std=2, n_points=100):
    """Compute points on a confidence ellipse from mean and covariance."""
    angle = jnp.linspace(0, 2 * jnp.pi, n_points)
    circle = jnp.stack([jnp.cos(angle), jnp.sin(angle)])
    eigvals, eigvecs = jnp.linalg.eigh(covariance)
    transform = eigvecs @ jnp.diag(jnp.sqrt(eigvals)) * n_std
    return mean[:, None] + transform @ circle


# %% [markdown]
# ## 3. Prior
#
# We start with a vague prior centred at the origin:
#
# $$p(\mathbf{x}) = \mathcal{N}(\mathbf{0}, 4 \mathbf{I})$$

# %%
# μ_x = 0
prior_mean = jnp.zeros(2)
# Σ_x = 4 I
prior_covariance = 4.0 * jnp.eye(2)
prior_covariance_op = lx.MatrixLinearOperator(prior_covariance, lx.symmetric_tag)

# Convert to natural parameters using gaussx
# (Λ_x μ_x, −½ Λ_x)
prior_nat1, prior_nat2 = gaussx.mean_cov_to_natural(prior_mean, prior_covariance_op)

print("Prior natural parameters:")
print("  eta1 =", prior_nat1)
print("  eta2 =")
print(prior_nat2.as_matrix())

# %% [markdown]
# ## 4. Single Observation Update
#
# We observe the first coordinate through $\mathbf{H}_1 = \begin{bmatrix} 1 & 0 \end{bmatrix} \in \mathbb{R}^{1 \times D_x}$ with noise variance $\Sigma_{\varepsilon,1} = 1$, so $D_y = 1$:
#
# $$y_1 = \mathbf{H}_1 \mathbf{x}_\text{true} + \varepsilon_1, \quad \varepsilon_1 \sim \mathcal{N}(0, 1)$$

# %%
# Observation model
# H_1 ∈ ℝ^{1×D_x}
obs_operator = jnp.array([[1.0, 0.0]])  # observe x-coordinate
# Σ_ε,1 ∈ ℝ^{1×1}
obs_noise_covariance = jnp.array([[1.0]])  # noise covariance

# Simulate observation
# y_1 = H_1 x_true + ε_1
key, subkey = jax.random.split(key)
obs = obs_operator @ x_true + jax.random.normal(subkey, (1,)) * jnp.sqrt(
    obs_noise_covariance[0, 0]
)
print("Observed y_1 =", obs)

# --- Natural parameter update ---
# Λ_ε = Σ_ε⁻¹
obs_noise_precision = jnp.linalg.inv(obs_noise_covariance)
# Λμ_lik = H_1ᵀ Λ_ε y_1
lik_nat1 = obs_operator.T @ obs_noise_precision @ obs
# −½ Λ_lik = −½ H_1ᵀ Λ_ε H_1
lik_nat2_matrix = -0.5 * obs_operator.T @ obs_noise_precision @ obs_operator

# Add to prior (the key insight: natural params are additive)
post_nat1 = prior_nat1 + lik_nat1
post_nat2_matrix = prior_nat2.as_matrix() + lik_nat2_matrix
post_nat2 = lx.MatrixLinearOperator(post_nat2_matrix, lx.symmetric_tag)

# Convert back to mean and covariance
post_expfam = GaussianExpFam(eta1=post_nat1, eta2=post_nat2)
post_mean, post_covariance_op = gaussx.to_mean_cov(post_expfam)
post_covariance = post_covariance_op.as_matrix()

print("\nNatural parameter posterior:")
print("  mu_1 =", post_mean)
print("  Sigma_1 =")
print(post_covariance)

# --- Covariance-form update (classic Kalman formula) ---
# Σ_y = H Σ_x Hᵀ + Σ_ε  (innovation covariance)
obs_covariance = obs_operator @ prior_covariance @ obs_operator.T + obs_noise_covariance

# G = Σ_x Hᵀ Σ_y⁻¹  (Kalman gain)
gain = prior_covariance @ obs_operator.T @ jnp.linalg.inv(obs_covariance)

# Posterior
# μ_x|y = μ_x + G (y − H μ_x)
post_mean_cov_form = prior_mean + gain @ (obs - obs_operator @ prior_mean)
# Σ_x|y = Σ_x − G Σ_y Gᵀ
post_covariance_cov_form = prior_covariance - gain @ obs_covariance @ gain.T

print("\nCovariance-form posterior:")
print("  mu_1 =", post_mean_cov_form.squeeze())
print("  Sigma_1 =")
print(post_covariance_cov_form)

# Verify both forms agree
print(
    "\nMax difference (mu):",
    float(jnp.max(jnp.abs(post_mean - post_mean_cov_form.squeeze()))),
)
print(
    "Max difference (Sigma):",
    float(jnp.max(jnp.abs(post_covariance - post_covariance_cov_form))),
)

# %% [markdown]
# Both the natural-parameter and covariance-form updates give the same
# posterior -- the natural form just replaces matrix inversions with
# addition.

# %% [markdown]
# ## 5. Prior vs Posterior after One Observation

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# Prior ellipse
pts_prior = ellipse_points(prior_mean, prior_covariance)
ax.plot(
    np.asarray(pts_prior[0]),
    np.asarray(pts_prior[1]),
    "b-",
    linewidth=2,
    label="Prior (2$\\sigma$)",
)

# Posterior ellipse
pts_post = ellipse_points(post_mean, post_covariance)
ax.plot(
    np.asarray(pts_post[0]),
    np.asarray(pts_post[1]),
    "r-",
    linewidth=2,
    label="Posterior (2$\\sigma$)",
)

# True location and observation
ax.plot(*np.asarray(x_true), "k*", markersize=15, label="True location")
ax.axvline(
    float(obs[0]),
    color="green",
    linestyle="--",
    alpha=0.7,
    label=f"$y_1 = {float(obs[0]):.2f}$",
)

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Bayesian Update: Prior vs Posterior")
ax.legend(fontsize=9)
ax.set_aspect("equal")
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

plt.show()

# %% [markdown]
# ## 6. Sequential Updates
#
# We now observe 5 more data points, each time adding the likelihood
# natural parameters to the current posterior. The posterior should
# shrink around the true location with each new observation.

# %%
# Observation matrices: mix of x-only, y-only, and joint observations
obs_operators = [
    jnp.array([[0.0, 1.0]]),  # y-coordinate
    jnp.array([[1.0, 1.0]]) / jnp.sqrt(2.0),  # diagonal
    jnp.array([[1.0, 0.0]]),  # x-coordinate again
    jnp.array([[0.0, 1.0]]),  # y-coordinate again
    jnp.array([[1.0, -1.0]]) / jnp.sqrt(2.0),  # anti-diagonal
]

# Σ_ε
seq_noise_covariance = jnp.array([[0.5]])  # all observations share the same noise

# Simulate observations
# y_n = H_n x_true + ε_n
observations = []
for H_n in obs_operators:
    key, subkey = jax.random.split(key)
    y_n = H_n @ x_true + jax.random.normal(subkey, (1,)) * jnp.sqrt(
        seq_noise_covariance[0, 0]
    )
    observations.append(y_n)

# Sequential Bayesian updating starting from posterior after first obs
trajectory_means = [prior_mean, post_mean]
trajectory_covariances = [prior_covariance, post_covariance]

# Current natural parameters (after first observation)
curr_nat1 = post_nat1.copy()
curr_nat2_matrix = post_nat2_matrix.copy()

# Λ_ε = Σ_ε⁻¹
seq_noise_precision = jnp.linalg.inv(seq_noise_covariance)

for i, (H_n, y_n) in enumerate(zip(obs_operators, observations, strict=True)):
    # Likelihood natural parameters
    # Λμ_lik = H_nᵀ Λ_ε y_n
    lik_nat1_n = H_n.T @ seq_noise_precision @ y_n
    # −½ Λ_lik = −½ H_nᵀ Λ_ε H_n
    lik_nat2_n = -0.5 * H_n.T @ seq_noise_precision @ H_n

    # Update: just add!
    curr_nat1 = curr_nat1 + lik_nat1_n
    curr_nat2_matrix = curr_nat2_matrix + lik_nat2_n

    # Convert back for plotting
    curr_nat2 = lx.MatrixLinearOperator(curr_nat2_matrix, lx.symmetric_tag)
    post_n = GaussianExpFam(eta1=curr_nat1, eta2=curr_nat2)
    mean_n, covariance_n_op = gaussx.to_mean_cov(post_n)

    trajectory_means.append(mean_n)
    trajectory_covariances.append(covariance_n_op.as_matrix())

    print(f"After obs {i + 2}: mu = [{float(mean_n[0]):.3f}, {float(mean_n[1]):.3f}]")

print(f"\nTrue location: [{float(x_true[0]):.3f}, {float(x_true[1]):.3f}]")

# %% [markdown]
# ## 7. Sequential Posterior Ellipses

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 7))

n_steps = len(trajectory_means)
colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_steps))

for k in range(n_steps):
    mean_k = trajectory_means[k]
    covariance_k = trajectory_covariances[k]
    pts = ellipse_points(mean_k, covariance_k)
    label = "Prior" if k == 0 else f"After obs {k}"
    ax.plot(
        np.asarray(pts[0]),
        np.asarray(pts[1]),
        color=colors[k],
        linewidth=2,
        label=label,
    )
    ax.plot(float(mean_k[0]), float(mean_k[1]), "o", color=colors[k], markersize=4)

ax.plot(*np.asarray(x_true), "r*", markersize=18, label="True location", zorder=10)

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Sequential Bayesian Updates")
ax.legend(fontsize=9, loc="upper left")
ax.set_aspect("equal")
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

plt.show()

# %% [markdown]
# ## 8. Batch Update = Sequential Update
#
# Because natural parameters are additive, processing all observations
# at once must give the same answer as processing them one at a time.
# This is a fundamental property of the exponential family.

# %%
# Collect ALL observations (including the first one)
all_obs_operators = [obs_operator, *obs_operators]
all_observations = [obs, *observations]
all_noise_covariances = [
    obs_noise_covariance,
    *([seq_noise_covariance] * len(observations)),
]

# Batch: sum all likelihood natural parameters at once
batch_nat1 = prior_nat1.copy()
batch_nat2_matrix = prior_nat2.as_matrix().copy()

for H_n, y_n, noise_covariance_n in zip(
    all_obs_operators, all_observations, all_noise_covariances, strict=True
):
    # Λ_ε = Σ_ε⁻¹
    noise_precision_n = jnp.linalg.inv(noise_covariance_n)
    batch_nat1 = batch_nat1 + H_n.T @ noise_precision_n @ y_n
    batch_nat2_matrix = batch_nat2_matrix + (-0.5 * H_n.T @ noise_precision_n @ H_n)

# Convert to mean/cov
batch_nat2 = lx.MatrixLinearOperator(batch_nat2_matrix, lx.symmetric_tag)
batch_expfam = GaussianExpFam(eta1=batch_nat1, eta2=batch_nat2)
batch_mean, batch_covariance_op = gaussx.to_mean_cov(batch_expfam)
batch_covariance = batch_covariance_op.as_matrix()

# Compare with sequential result (last entry in trajectory)
seq_mean = trajectory_means[-1]
seq_covariance = trajectory_covariances[-1]

diff_mean = float(jnp.max(jnp.abs(batch_mean - seq_mean)))
diff_covariance = float(jnp.max(jnp.abs(batch_covariance - seq_covariance)))

print("Batch posterior mean:     ", np.asarray(batch_mean))
print("Sequential posterior mean:", np.asarray(seq_mean))
print(f"Max |mu_batch - mu_seq|     = {diff_mean:.2e}")
print(f"Max |Sigma_batch - Sigma_seq| = {diff_covariance:.2e}")

# %% [markdown]
# The batch and sequential results agree to machine precision ($\sim 10^{-15}$),
# confirming that natural parameter addition is order-independent.

# %% [markdown]
# ## 9. Connection to GP Regression
#
# Gaussian process regression **is** Bayes' rule applied to a
# function-valued prior. On $N$ training inputs $\mathbf{X}$, the function values $\mathbf{f} \in \mathbb{R}^N$ have kernel matrix $\mathbf{K}_{XX} \in \mathbb{R}^{N \times N}$:
#
# $$\text{Prior:} \quad \mathbf{f} \sim \mathcal{N}(\mathbf{0}, \mathbf{K}_{XX})$$
#
# $$\text{Likelihood:} \quad \mathbf{y} \mid \mathbf{f} \sim \mathcal{N}(\mathbf{f}, \sigma_\varepsilon^2 \mathbf{I})$$
#
# Applying the Gaussian Bayes rule with $\mathbf{H} = \mathbf{I}$ and $\boldsymbol{\Sigma}_\varepsilon = \sigma_\varepsilon^2 \mathbf{I}$:
#
# $$\boldsymbol{\Lambda}_{f \mid y} = \mathbf{K}_{XX}^{-1} + \sigma_\varepsilon^{-2} \mathbf{I}$$
#
# $$\boldsymbol{\mu}_{f \mid y} = \boldsymbol{\Lambda}_{f \mid y}^{-1} \sigma_\varepsilon^{-2} \mathbf{y} = \mathbf{K}_{XX} (\mathbf{K}_{XX} + \sigma_\varepsilon^2 \mathbf{I})^{-1} \mathbf{y}$$
#
# This is the standard GP posterior mean formula. We can compute it
# directly with `gaussx.solve`.

# %%
# Small GP regression example
# N = 30
num_train = 30
key, subkey = jax.random.split(key)
train_inputs = jnp.sort(
    jax.random.uniform(subkey, (num_train,), minval=0.0, maxval=5.0)
)

# RBF kernel
# K_XX = exp(−½ (x − x')² / ℓ²)
lengthscale = 1.0
diff = train_inputs[:, None] - train_inputs[None, :]
gram_train = jnp.exp(-0.5 * diff**2 / lengthscale**2)

# Observations with noise
# y = f + ε, ε ~ 𝒩(0, σ_ε² I)
noise_variance = 0.1
key, subkey = jax.random.split(key)
f_true = jnp.sin(train_inputs)
train_obs = f_true + jax.random.normal(subkey, (num_train,)) * jnp.sqrt(noise_variance)

# GP posterior via gaussx.solve (the Bayes rule result)
# K_XX + σ_ε² I
gram_plus_noise = gram_train + noise_variance * jnp.eye(num_train)
gram_plus_noise_op = lx.MatrixLinearOperator(gram_plus_noise, lx.symmetric_tag)

# μ_f|y = K_XX (K_XX + σ_ε² I)⁻¹ y
alpha = gaussx.solve(gram_plus_noise_op, train_obs)
post_mean_gp = gram_train @ alpha

# Also compute the log marginal likelihood
lml = gaussx.log_marginal_likelihood(
    loc=jnp.zeros(num_train),
    cov_operator=gram_plus_noise_op,
    y=train_obs,
)

print(f"GP log marginal likelihood: {float(lml):.3f}")
rmse = float(jnp.sqrt(jnp.mean((post_mean_gp - f_true) ** 2)))
print(f"Posterior mean RMSE vs truth: {rmse:.4f}")

# %% [markdown]
# The GP posterior mean is just the Bayes rule result computed via a
# single `gaussx.solve` call. The log marginal likelihood -- used for
# hyperparameter optimisation -- is computed by
# `gaussx.log_marginal_likelihood`, which internally uses `solve` and
# `logdet` on the same structured operator.

# %% [markdown]
# ## 10. Summary
#
# Bayes' rule for Gaussians is **addition in natural parameter space**:
#
# $$\boldsymbol{\Lambda}_{x \mid y} \boldsymbol{\mu}_{x \mid y} = \boldsymbol{\Lambda}_x \boldsymbol{\mu}_x + \boldsymbol{\Lambda\mu}_\text{lik}$$
#
# $$\boldsymbol{\Lambda}_{x \mid y} = \boldsymbol{\Lambda}_x + \boldsymbol{\Lambda}_\text{lik}$$
#
# This single insight underlies:
#
# - **Kalman filtering** (sequential state estimation)
# - **GP regression** (function-space inference)
# - **Variational inference** with Gaussian approximations
# - **Natural gradient** methods
#
# gaussx makes this clean with:
#
# - `mean_cov_to_natural` / `to_mean_cov` for parameter conversion
# - `GaussianExpFam` for natural parameter containers
# - `solve`, `inv`, `logdet` for structured linear algebra
# - `log_marginal_likelihood` for GP model selection
#
# All operations compose naturally with JAX's `jit`, `grad`, and `vmap`.

# %% [markdown]
# ## References
#
# - Bernardo, J. M. & Smith, A. F. M. (2000). *Bayesian Theory*. Wiley.
# - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
#   Springer. (Section 2.3)
# - Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for
#   Machine Learning*. MIT Press.
# - Minka, T. P. (2001). *A Family of Algorithms for Approximate Bayesian
#   Inference*. PhD thesis, MIT.
# - Opper, M. & Winther, O. (2005). Expectation consistent approximate
#   inference. *JMLR*, 6, 2177--2204.
