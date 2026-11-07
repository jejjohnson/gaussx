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
# # Bayesian Linear Regression with Precision Parameterization
#
# This notebook demonstrates `gaussx.MultivariateNormalPrecision` in a
# NumPyro model for Bayesian linear regression. The precision
# parameterization is natural when:
#
# - The prior is specified via a **precision matrix** (e.g. ridge
#   regularization, Gaussian Markov random fields)
# - You want to perform **conjugate updates** by adding precision
#   matrices directly
# - The posterior precision has structure you want to preserve
#   (e.g. banded, sparse, Kronecker)
#
# **What you'll learn:**
#
# 1. Bayesian linear regression with a precision-parameterized prior
# 2. Analytic conjugate posterior in precision form
# 3. Using `MultivariateNormalPrecision` in a NumPyro model with MCMC
# 4. Comparing MCMC posterior against the analytic solution
# 5. Posterior predictive with uncertainty quantification

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import warnings


warnings.filterwarnings("ignore", message=r".*IProgress.*")

import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import gaussx


jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Background
#
# In Bayesian linear regression on $N$ samples with known noise variance $\sigma_\varepsilon^2$, the observations $\mathbf{y} \in \mathbb{R}^N$ are a linear function of the parameters $\boldsymbol{\theta} \in \mathbb{R}^{D_\theta}$ through the design matrix $\boldsymbol{\Phi} \in \mathbb{R}^{N \times D_\theta}$:
#
# $$\mathbf{y} = \boldsymbol{\Phi} \boldsymbol{\theta} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_\varepsilon^2 \mathbf{I})$$
#
# With a Gaussian prior $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Lambda}_\theta^{-1})$
# specified via **precision** $\boldsymbol{\Lambda}_\theta \in \mathbb{R}^{D_\theta \times D_\theta}$, the posterior is:
#
# $$\boldsymbol{\theta} \mid \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_{\theta \mid y}, \boldsymbol{\Lambda}_{\theta \mid y}^{-1})$$
#
# where:
#
# $$\boldsymbol{\Lambda}_{\theta \mid y} = \boldsymbol{\Lambda}_\theta + \frac{1}{\sigma_\varepsilon^2} \boldsymbol{\Phi}^\top \boldsymbol{\Phi}$$
#
# $$\boldsymbol{\mu}_{\theta \mid y} = \boldsymbol{\Lambda}_{\theta \mid y}^{-1} \left( \frac{1}{\sigma_\varepsilon^2} \boldsymbol{\Phi}^\top \mathbf{y} \right)$$
#
# The posterior precision is the **sum** of prior precision and data
# precision -- this additivity is why precision parameterization is
# natural for conjugate updates.
#
# The precision form arises naturally in **Gaussian Markov random fields
# (GMRFs)** where the precision matrix is sparse (typically banded or
# block-tridiagonal), encoding conditional independence structure. In
# spatial statistics, the SPDE approach (Lindgren et al., 2011) constructs
# a GMRF approximation to a Matern GP, where the precision is a sparse
# matrix defined by the finite element mesh. gaussx's
# `MultivariateNormalPrecision` can work with structured precision
# operators (banded, block-diagonal, Kronecker) for efficient inference
# in these models.

# %% [markdown]
# ## Generate data
#
# We create a polynomial regression problem with known true parameters $\boldsymbol{\theta}_\text{true}$. The design matrix has rows $\boldsymbol{\phi}(x_n) = (1, x_n, x_n^2, \ldots, x_n^{D_\theta - 1})$ for the scalar inputs $x_n$, $n = 1, \ldots, N$.

# %%
key = jax.random.PRNGKey(0)
# N = 50
num_train = 50
# σ_ε
noise_std = 0.5
degree = 5

# True parameters
# θ_true ∈ ℝ^{D_θ}
params_true = jnp.array([0.5, -1.2, 0.3, 0.8, -0.4, 0.1])

# Design matrix: polynomial features [1, x, x^2, ..., x^degree]
# Φ ∈ ℝ^{N×D_θ}
key, subkey = jax.random.split(key)
train_inputs = jnp.sort(jax.random.uniform(subkey, (num_train,), minval=-2, maxval=2))
train_features = jnp.column_stack([train_inputs**p for p in range(degree + 1)])

# Observations
# y = Φ θ_true + ε
key, subkey = jax.random.split(key)
train_obs = train_features @ params_true + noise_std * jax.random.normal(
    subkey, (num_train,)
)

# Test grid
# Φ_* ∈ ℝ^{M×D_θ}
test_inputs = jnp.linspace(-2.5, 2.5, 200)
test_features = jnp.column_stack([test_inputs**p for p in range(degree + 1)])

# D_θ = degree + 1
num_params = degree + 1
print(f"Design matrix shape: {train_features.shape}")
print(f"True weights: {params_true}")

# %% [markdown]
# ## Analytic posterior (conjugate)
#
# We compute the exact posterior using the conjugate precision-form
# update.

# %%
# Prior: isotropic precision (ridge regularization)
# Λ_θ = α I
prior_precision_scale = 1.0  # prior precision scalar
prior_precision = prior_precision_scale * jnp.eye(num_params)

# Data precision contribution
# Λ_data = σ_ε⁻² Φᵀ Φ
data_precision = (1 / noise_std**2) * train_features.T @ train_features

# Posterior precision and mean
# Λ_θ|y = Λ_θ + Λ_data
post_precision = prior_precision + data_precision
# μ_θ|y = Λ_θ|y⁻¹ σ_ε⁻² Φᵀ y
post_mean = jnp.linalg.solve(
    post_precision, (1 / noise_std**2) * train_features.T @ train_obs
)
# Σ_θ|y = Λ_θ|y⁻¹
post_covariance = jnp.linalg.inv(post_precision)

print("Posterior precision (diagonal):", jnp.diag(post_precision))
print("Posterior mean:", post_mean)
print("True weights:  ", params_true)

# %% [markdown]
# ## NumPyro model with precision prior
#
# We define the model using `gaussx.MultivariateNormalPrecision` for
# the prior on the parameters. This directly encodes the ridge
# regularization as a precision matrix.


# %%
def blr_model(features, noise_std, obs=None):
    num_params = features.shape[1]

    # Prior: θ ~ 𝒩(0, Λ_θ⁻¹) with Λ_θ = α I
    prior_precision = prior_precision_scale * jnp.eye(num_params)
    prior_precision_op = lx.MatrixLinearOperator(
        prior_precision, lx.positive_semidefinite_tag
    )
    params = numpyro.sample(
        "w",
        gaussx.MultivariateNormalPrecision(jnp.zeros(num_params), prior_precision_op),
    )

    # Likelihood: y | θ ~ 𝒩(Φ θ, σ_ε² I)
    mean = features @ params
    with numpyro.plate("data", features.shape[0]):
        numpyro.sample("obs", dist.Normal(mean, noise_std), obs=obs)


# %% [markdown]
# ## Run NUTS
#
# NumPyro's NUTS sampler works directly with
# `MultivariateNormalPrecision` — the log-prob and its gradients
# are computed via gaussx structural dispatch.

# %%
kernel = NUTS(blr_model)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, progress_bar=False)
mcmc.run(jax.random.PRNGKey(1), train_features, noise_std, obs=train_obs)
mcmc.print_summary()

# %%
param_samples = mcmc.get_samples()["w"]
print("MCMC weight samples shape:", param_samples.shape)

# %% [markdown]
# ## Compare MCMC vs analytic posterior
#
# The MCMC samples should match the analytic conjugate posterior.
# We compare marginal distributions for each parameter $\theta_d$, $d = 0, \ldots, D_\theta - 1$.

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

for d, ax in enumerate(axes.flat):
    # MCMC histogram
    ax.hist(
        param_samples[:, d],
        bins=35,
        density=True,
        alpha=0.6,
        color="steelblue",
        label="MCMC",
    )

    # Analytic posterior
    param_grid = jnp.linspace(
        post_mean[d] - 4 * jnp.sqrt(post_covariance[d, d]),
        post_mean[d] + 4 * jnp.sqrt(post_covariance[d, d]),
        200,
    )
    pdf = jnp.exp(
        dist.Normal(post_mean[d], jnp.sqrt(post_covariance[d, d])).log_prob(param_grid)
    )
    ax.plot(param_grid, pdf, "C3-", lw=2, label="Analytic")

    # True value
    ax.axvline(params_true[d], color="k", ls="--", lw=1.5, label="True")

    ax.set_title(f"$\\theta_{d}$")
    if d == 0:
        ax.legend(fontsize=9)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.minorticks_on()

fig.suptitle("MCMC vs Analytic Posterior (per weight)", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Posterior predictive
#
# We compute the predictive distribution by pushing MCMC parameter
# samples through the model.

# %%
# Predictive mean for each sample
# y_*⁽ᵉ⁾ = Φ_* θ⁽ᵉ⁾
pred_samples = param_samples @ test_features.T  # (n_samples, n_test)

pred_mean = jnp.mean(pred_samples, axis=0)
# Total predictive std: epistemic (parameter uncertainty) + aleatoric (noise)
pred_std = jnp.sqrt(jnp.var(pred_samples, axis=0) + noise_std**2)

# Also compute analytic predictive
# μ_y* = Φ_* μ_θ|y
analytic_mean = test_features @ post_mean
# σ_y*² = diag(Φ_* Σ_θ|y Φ_*ᵀ) + σ_ε²
analytic_var = (
    jnp.sum(test_features @ post_covariance * test_features, axis=1) + noise_std**2
)
analytic_std = jnp.sqrt(analytic_var)

print(f"MCMC pred mean shape: {pred_mean.shape}")
print(f"Analytic pred mean shape: {analytic_mean.shape}")

# %% [markdown]
# ## Plot predictions
#
# The MCMC posterior predictive should closely match the analytic
# solution. Both include observation noise in the uncertainty bands.

# %%
fig, ax = plt.subplots(figsize=(12, 5))

# True function
# f_true = Φ_* θ_true
true_test = test_features @ params_true
ax.plot(test_inputs, true_test, "k--", lw=1.5, label="True function", zorder=4)

# MCMC predictive
ax.plot(test_inputs, pred_mean, "C0-", lw=2, label="MCMC mean", zorder=3)
ax.fill_between(
    test_inputs,
    pred_mean - 2 * pred_std,
    pred_mean + 2 * pred_std,
    color="C0",
    alpha=0.15,
    label=r"MCMC $\pm 2\sigma$",
)

# Analytic predictive
ax.plot(test_inputs, analytic_mean, "C3--", lw=1.5, label="Analytic mean", zorder=3)
ax.fill_between(
    test_inputs,
    analytic_mean - 2 * analytic_std,
    analytic_mean + 2 * analytic_std,
    color="C3",
    alpha=0.1,
    label=r"Analytic $\pm 2\sigma$",
)

# Training data
ax.scatter(
    train_inputs,
    train_obs,
    s=30,
    c="C1",
    edgecolors="k",
    linewidths=0.5,
    label="Training data",
    zorder=5,
)

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Bayesian Linear Regression: MCMC vs Analytic Posterior")
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.set_ylim(-5, 5)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Precision additivity
#
# The key advantage of precision parameterization: posterior
# precision = prior precision + data precision. Let's verify this
# with gaussx.
#
# Each observation adds information (in the Fisher sense) to the
# posterior. The posterior precision
# $\boldsymbol{\Lambda}_{\theta \mid y} = \boldsymbol{\Lambda}_\theta + \boldsymbol{\Lambda}_\text{data}$ directly
# reflects this: the total information is the sum of prior information
# and data information. This additivity is the defining property of the
# natural parameters of the exponential family.

# %%
# Build operators
prior_precision_op = lx.MatrixLinearOperator(
    prior_precision, lx.positive_semidefinite_tag
)
data_precision_op = lx.MatrixLinearOperator(
    data_precision, lx.positive_semidefinite_tag
)
post_precision_op = lx.MatrixLinearOperator(
    post_precision, lx.positive_semidefinite_tag
)

# Prior
prior_dist = gaussx.MultivariateNormalPrecision(
    jnp.zeros(num_params), prior_precision_op
)
print("Prior entropy:", prior_dist.entropy())

# Posterior
post_dist = gaussx.MultivariateNormalPrecision(post_mean, post_precision_op)
print("Posterior entropy:", post_dist.entropy())

# Entropy should decrease (more information)
print("Entropy decreased:", float(prior_dist.entropy()) > float(post_dist.entropy()))

# Verify log-prob at posterior mean
print("log p(mu_post | posterior):", post_dist.log_prob(post_mean))

# %% [markdown]
# ## Summary
#
# - `gaussx.MultivariateNormalPrecision` naturally encodes ridge
#   regularization and conjugate Bayesian updates.
# - It plugs directly into NumPyro models — NUTS computes gradients
#   through the precision-parameterized log-prob.
# - For linear-Gaussian models, the MCMC posterior matches the
#   analytic conjugate solution, validating the implementation.
# - **Precision additivity** ($\boldsymbol{\Lambda}_{\theta \mid y} = \boldsymbol{\Lambda}_\theta + \boldsymbol{\Lambda}_\text{data}$) makes sequential updates
#   trivial — just add precision operators.
# - The same pattern extends to structured precisions (banded,
#   Kronecker, block-diagonal) for scalable models like GMRFs.

# %% [markdown]
# ## References
#
# - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
#   Springer. (Section 3.3 on Bayesian linear regression)
# - Lindgren, F., Rue, H., & Lindstrom, J. (2011). An explicit link
#   between Gaussian fields and Gaussian Markov random fields: the
#   stochastic partial differential equation approach. *JRSS-B*, 73(4),
#   423--498.
# - Rue, H. & Held, L. (2005). *Gaussian Markov Random Fields: Theory
#   and Applications*. Chapman & Hall/CRC.
