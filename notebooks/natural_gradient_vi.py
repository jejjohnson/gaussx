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
# # Natural Gradient Variational Inference from Scratch
#
# This notebook derives and implements natural gradient variational
# inference for Gaussian approximate posteriors, showing how gaussx
# primitives make the natural parameterization clean and efficient.
#
# **What you'll learn:**
#
# 1. Why Euclidean gradients are slow for distribution optimization
# 2. How natural parameters and the Fisher metric fix this
# 3. Using `damped_natural_update` for stable natural gradient steps
# 4. Computing the ELBO via `variational_elbo_gaussian` and `variational_elbo_mc`
# 5. The Bayesian Learning Rule with `blr_full_update`
# 6. Building Gauss-Newton Hessian approximations with `gauss_newton_precision`

# %% [markdown]
# ## 1. Background: Natural Gradient Descent
#
# Standard gradient descent updates parameters along the steepest
# direction in Euclidean space. But when the parameters describe a
# probability distribution, Euclidean distance is the wrong metric.
# A small step in $(\boldsymbol{\mu}_f, \boldsymbol{\Sigma}_f)$ can cause a huge change in
# KL-divergence, or vice versa.
#
# The **Fisher information metric** measures distance between nearby
# distributions $q(\mathbf{f}; \boldsymbol{\nu})$ and $q(\mathbf{f}; \boldsymbol{\nu} + \mathrm{d}\boldsymbol{\nu})$ with variational parameters $\boldsymbol{\nu} \in \mathbb{R}^{D_\nu}$:
#
# $$\mathrm{d}_F^2 = \mathrm{d}\boldsymbol{\nu}^\top \mathbf{F}(\boldsymbol{\nu}) \mathrm{d}\boldsymbol{\nu}$$
#
# where $\mathbf{F}(\boldsymbol{\nu}) = \mathbb{E}_{q(\mathbf{f}; \boldsymbol{\nu})}[\nabla_{\boldsymbol{\nu}} \log q (\nabla_{\boldsymbol{\nu}} \log q)^\top]$ is the Fisher information
# matrix.
#
# The **natural gradient** is the steepest descent direction under
# this metric:
#
# $$\tilde{\nabla}_{\boldsymbol{\nu}} \mathcal{L} = \mathbf{F}(\boldsymbol{\nu})^{-1} \nabla_{\boldsymbol{\nu}} \mathcal{L}$$
#
# For the multivariate Gaussian $q(\mathbf{f}) = \mathcal{N}(\boldsymbol{\mu}_f, \boldsymbol{\Sigma}_f)$
# in **natural parameters** $\boldsymbol{\nu}_1 = \boldsymbol{\Sigma}_f^{-1} \boldsymbol{\mu}_f$ and
# $\boldsymbol{\nu}_2 = -\frac{1}{2} \boldsymbol{\Sigma}_f^{-1}$, the natural gradient with respect to $\boldsymbol{\nu}$ equals the ordinary gradient with respect to the expectation parameters, so a step needs no Fisher inverse. This is why natural parameterization is so
# powerful.

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
# ## 2. A Non-Conjugate Model: GP Classification
#
# We set up a 1D binary classification problem with a probit
# likelihood. The posterior $p(\mathbf{f} \mid \mathbf{y})$ is non-Gaussian, so we
# must use approximate inference.
#
# **Model**, for $N$ samples with signed labels $y_n \in \{-1, +1\}$ and latent values $\mathbf{f} \in \mathbb{R}^N$:
# - Prior: $p(\mathbf{f}) = \mathcal{N}(\mathbf{0}, \mathbf{K}_{XX})$ with squared-exponential kernel
# - Likelihood: $p(y_n \mid f_n) = \Phi(y_n f_n)$ (probit)
#
# We approximate the posterior with $q(\mathbf{f}) = \mathcal{N}(\boldsymbol{\mu}_f, \boldsymbol{\Sigma}_f)$.

# %%
key = jax.random.PRNGKey(42)

# Generate 1D classification data
# N
num_samples = 30
key, subkey = jax.random.split(key)
train_inputs = jnp.sort(
    jax.random.uniform(subkey, (num_samples,), minval=-3.0, maxval=3.0)
)
y_class = jnp.where(train_inputs > 0.3, 1.0, -1.0)
# Flip some labels for noise
key, subkey = jax.random.split(key)
flip = jax.random.bernoulli(subkey, 0.1, (num_samples,))
y_class = jnp.where(flip, -y_class, y_class)

# Squared exponential kernel
lengthscale = 1.0
kernel_variance = 1.5


def se_kernel(x1, x2):
    # k(x, x') = σ_k² exp(−(x − x')² / (2 ℓ²))
    sq_dist = (x1[:, None] - x2[None, :]) ** 2
    return kernel_variance * jnp.exp(-0.5 * sq_dist / lengthscale**2)


# K_XX + jitter I
gram_train = se_kernel(train_inputs, train_inputs) + 1e-6 * jnp.eye(num_samples)


# Probit log-likelihood
def probit_log_lik(f, obs):
    # log p(y ∣ f) = ∑ₙ log Φ(y_n f_n)
    return jnp.sum(jax.scipy.stats.norm.logcdf(obs * f))


fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(
    train_inputs,
    y_class,
    s=30,
    c=y_class,
    cmap="bwr",
    edgecolors="k",
    linewidths=0.5,
    zorder=5,
)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Binary classification data")
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Euclidean Gradient Descent on the ELBO
#
# First we try the naive approach: parameterize $q(\mathbf{f})$ via
# $(\boldsymbol{\mu}_f, \mathbf{L}_{\boldsymbol{\Sigma}_f})$ where $\boldsymbol{\Sigma}_f = \mathbf{L}_{\boldsymbol{\Sigma}_f} \mathbf{L}_{\boldsymbol{\Sigma}_f}^\top$, and optimize the ELBO
# with plain gradient descent.
#
# The ELBO for non-conjugate likelihoods requires Monte Carlo:
#
# $$\mathrm{ELBO} = \mathbb{E}_{q(\mathbf{f})}[\log p(\mathbf{y} \mid \mathbf{f})] - \mathrm{KL}[q \Vert p]$$


# %%
def kl_gaussian(mean, covariance_cholesky, gram):
    """KL[q ‖ p] where q = 𝒩(μ_f, L Lᵀ) and p = 𝒩(0, K_XX)."""
    size = mean.shape[0]
    # L_K L_Kᵀ = K_XX
    gram_cholesky = jnp.linalg.cholesky(gram)
    # V = L_K⁻¹ L
    whitened_factor = jax.scipy.linalg.solve_triangular(
        gram_cholesky, covariance_cholesky, lower=True
    )
    # tr(K_XX⁻¹ Σ_f) = ‖V‖_F²
    trace_term = jnp.sum(whitened_factor**2)
    # μ_fᵀ K_XX⁻¹ μ_f
    gram_solve_mean = jax.scipy.linalg.cho_solve((gram_cholesky, True), mean)
    quad_term = mean @ gram_solve_mean
    # log det K_XX − log det Σ_f
    logdet_gram = 2.0 * jnp.sum(jnp.log(jnp.diag(gram_cholesky)))
    logdet_covariance = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(covariance_cholesky))))
    return 0.5 * (trace_term + quad_term - size + logdet_gram - logdet_covariance)


def euclidean_elbo(mean, chol_flat, gram, obs, key):
    """ELBO with reparameterization trick."""
    covariance_cholesky = (
        jnp.zeros((num_samples, num_samples))
        .at[jnp.tril_indices(num_samples)]
        .set(chol_flat)
    )
    kl = kl_gaussian(mean, covariance_cholesky, gram)
    # Reparameterized samples
    # z⁽ᵉ⁾ ~ 𝒩(0, I)
    base_draws = jax.random.normal(key, (8, num_samples))
    # f⁽ᵉ⁾ = μ_f + L z⁽ᵉ⁾
    f_samples = mean[None, :] + base_draws @ covariance_cholesky.T
    log_lik_fn = lambda f: probit_log_lik(f, obs)
    elbo = gaussx.variational_elbo_mc(log_lik_fn, f_samples, kl)
    return elbo


# Initialize
mean_init = jnp.zeros(num_samples)
cholesky_init = 0.5 * jnp.eye(num_samples)
chol_flat_init = cholesky_init[jnp.tril_indices(num_samples)]


# Adam optimizer (simple implementation)
def adam_step(
    params,
    grads,
    first_moment,
    second_moment,
    step,
    lr=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    # m ← β₁ m + (1 − β₁) g
    first_new = [
        beta1 * mi + (1 - beta1) * gi
        for mi, gi in zip(first_moment, grads, strict=True)
    ]
    # v ← β₂ v + (1 − β₂) g²
    second_new = [
        beta2 * vi + (1 - beta2) * gi**2
        for vi, gi in zip(second_moment, grads, strict=True)
    ]
    # m̂ = m / (1 − β₁^{step+1})
    first_hat = [mi / (1 - beta1 ** (step + 1)) for mi in first_new]
    # v̂ = v / (1 − β₂^{step+1})
    second_hat = [vi / (1 - beta2 ** (step + 1)) for vi in second_new]
    # p ← p − lr m̂ / (√v̂ + ε)
    params_new = [
        p - lr * mh / (jnp.sqrt(vh) + epsilon)
        for p, mh, vh in zip(params, first_hat, second_hat, strict=True)
    ]
    return params_new, first_new, second_new


n_steps_euclidean = 300
mean_euclid = mean_init.copy()
chol_flat_euclid = chol_flat_init.copy()
first_moment_adam = [jnp.zeros_like(mean_euclid), jnp.zeros_like(chol_flat_euclid)]
second_moment_adam = [jnp.zeros_like(mean_euclid), jnp.zeros_like(chol_flat_euclid)]
elbo_history_euclidean = []

for _i in range(n_steps_euclidean):
    key, subkey = jax.random.split(key)
    loss_fn = lambda mean, chol_flat, _k=subkey: (
        -euclidean_elbo(mean, chol_flat, gram_train, y_class, _k)
    )
    loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(
        mean_euclid, chol_flat_euclid
    )
    [mean_euclid, chol_flat_euclid], first_moment_adam, second_moment_adam = adam_step(
        [mean_euclid, chol_flat_euclid],
        grads,
        first_moment_adam,
        second_moment_adam,
        _i,
        lr=0.01,
    )
    elbo_history_euclidean.append(-float(loss))

print(f"Euclidean (Adam) — final ELBO: {elbo_history_euclidean[-1]:.3f}")

# %% [markdown]
# ## 4. Natural Gradient Descent
#
# Now we parameterize $q(\mathbf{f})$ via **natural parameters**, written as $\boldsymbol{\nu} = (\boldsymbol{\nu}_1, \boldsymbol{\nu}_2)$ with
#
# $$\boldsymbol{\nu}_1 = \boldsymbol{\Sigma}_f^{-1} \boldsymbol{\mu}_f$$
#
# $$\boldsymbol{\nu}_2 = -\frac{1}{2} \boldsymbol{\Sigma}_f^{-1}$$
#
# The natural gradient update becomes a simple convex combination, with $i$ counting optimiser iterations:
#
# $$\boldsymbol{\nu}^{(i+1)} = (1 - \gamma) \boldsymbol{\nu}^{(i)} + \gamma \boldsymbol{\nu}_\text{target}$$
#
# where $\boldsymbol{\nu}_\text{target}$ are the target natural parameters from the
# Newton/Laplace step and $\gamma$ is the learning rate.
#
# This is exactly what `gaussx.damped_natural_update` computes.

# %%
# Natural parameterization: ν_1 = Σ_f⁻¹ μ_f, ν_2 = −½ Σ_f⁻¹
# K_XX⁻¹
gram_inverse = jnp.linalg.inv(gram_train)

# Initialize at the prior
nat1 = jnp.zeros(num_samples)  # prior: ν_1 = K⁻¹ 0 = 0
nat2 = -0.5 * gram_inverse  # prior: ν_2 = −½ K⁻¹

n_steps_natural = 60
# γ
step_size_nat = 0.5
elbo_history_natural = []

for _i in range(n_steps_natural):
    # Recover moment parameters from natural
    # Λ_f = −2 ν_2
    precision = -2.0 * nat2
    # Σ_f = Λ_f⁻¹
    covariance = jnp.linalg.inv(precision)
    # μ_f = Σ_f ν_1
    mean = covariance @ nat1

    # Compute log-likelihood gradient and Hessian at current mean
    loglik_grad = jax.grad(lambda f: probit_log_lik(f, y_class))(mean)
    loglik_hessian = jax.hessian(lambda f: probit_log_lik(f, y_class))(mean)

    # Newton target natural parameters (likelihood site)
    # newton_update returns the site precision +Λ_site, so convert it to ν_2
    nat1_target, site_precision = gaussx.newton_update(
        mean, loglik_grad, loglik_hessian
    )
    # Add prior precision contribution: ν_2,prior = −½ K_XX⁻¹
    # ν_2,target = −½ Λ_site − ½ K_XX⁻¹
    nat2_target = -0.5 * site_precision + (-0.5 * gram_inverse)

    # Damped natural update
    nat1, nat2 = gaussx.damped_natural_update(
        nat1, nat2, nat1_target, nat2_target, lr=step_size_nat
    )

    # Evaluate ELBO for tracking
    precision_new = -2.0 * nat2
    covariance_new = jnp.linalg.inv(precision_new)
    mean_new = covariance_new @ nat1
    cholesky_new = jnp.linalg.cholesky(covariance_new)
    kl = kl_gaussian(mean_new, cholesky_new, gram_train)
    key, subkey = jax.random.split(key)
    base_draws = jax.random.normal(subkey, (32, num_samples))
    f_samples = mean_new[None, :] + base_draws @ cholesky_new.T
    log_lik_fn = lambda f: probit_log_lik(f, y_class)
    elbo_val = gaussx.variational_elbo_mc(log_lik_fn, f_samples, kl)
    elbo_history_natural.append(float(elbo_val))

print(f"Natural gradient — final ELBO: {elbo_history_natural[-1]:.3f}")

# %% [markdown]
# ## 5. ELBO with Gaussian Likelihood
#
# For a Gaussian observation model $p(\mathbf{y} \mid \mathbf{f}) = \mathcal{N}(\mathbf{y}; \mathbf{f}, \sigma_\varepsilon^2 \mathbf{I})$,
# the expected log-likelihood has a closed form.
# `gaussx.variational_elbo_gaussian` computes this without Monte Carlo
# sampling, with the marginals $q(f_n) = \mathcal{N}(\mu_n, \sigma_n^2)$:
#
# $$\mathrm{ELBO} = -\frac{N}{2} \log(2\pi \sigma_\varepsilon^2) - \frac{1}{2\sigma_\varepsilon^2} \left( \Vert \mathbf{y} - \boldsymbol{\mu}_f \Vert_2^2 + \sum_{n=1}^N \sigma_n^2 \right) - \mathrm{KL}[q \Vert p]$$

# %%
# Demonstrate variational_elbo_gaussian on a regression problem
key, subkey = jax.random.split(key)
obs_reg = jnp.sin(train_inputs) + 0.3 * jax.random.normal(subkey, (num_samples,))
# σ_ε²
noise_variance = 0.09

# Use the posterior mean/variance from a simple variational fit
# K_XX + σ_ε² I
gram_plus_noise = se_kernel(train_inputs, train_inputs) + noise_variance * jnp.eye(
    num_samples
)
gram_plus_noise_cholesky = jnp.linalg.cholesky(gram_plus_noise)
alpha_reg = jax.scipy.linalg.cho_solve((gram_plus_noise_cholesky, True), obs_reg)
f_loc = gram_train @ alpha_reg  # approximate posterior mean
f_var = jnp.diag(gram_train) - jnp.sum(
    jax.scipy.linalg.solve_triangular(gram_plus_noise_cholesky, gram_train, lower=True)
    ** 2,
    axis=0,
)

# KL divergence (using the approximate posterior)
posterior_covariance = gram_train - gram_train @ jax.scipy.linalg.cho_solve(
    (gram_plus_noise_cholesky, True), gram_train
)
posterior_cholesky = jnp.linalg.cholesky(
    posterior_covariance + 1e-6 * jnp.eye(num_samples)
)
kl_reg = kl_gaussian(f_loc, posterior_cholesky, gram_train)

elbo_gauss = gaussx.variational_elbo_gaussian(
    obs_reg, f_loc, f_var, noise_variance, kl_reg
)
print(f"Gaussian ELBO (closed form): {float(elbo_gauss):.3f}")

# %% [markdown]
# ## 6. Monte Carlo ELBO for Non-Conjugate Likelihoods
#
# For non-Gaussian likelihoods (Bernoulli, Poisson, etc.), we use
# `gaussx.variational_elbo_mc`. It takes a callable `log_likelihood_fn`
# and draws $N_{\mathcal{E}}$ members $\mathbf{f}^{(e)}$, $e \in \mathcal{E}$, from the variational posterior:
#
# $$\mathrm{ELBO}_{\mathrm{MC}} = \frac{1}{N_{\mathcal{E}}} \sum_{e \in \mathcal{E}} \log p(\mathbf{y} \mid \mathbf{f}^{(e)}) - \mathrm{KL}[q \Vert p]$$

# %%
# Using the natural gradient posterior from Section 4
# Λ_f = −2 ν_2
precision_final = -2.0 * nat2
# Σ_f = Λ_f⁻¹
covariance_final = jnp.linalg.inv(precision_final)
# μ_f = Σ_f ν_1
mean_final = covariance_final @ nat1
cholesky_final = jnp.linalg.cholesky(covariance_final)

kl_final = kl_gaussian(mean_final, cholesky_final, gram_train)

# Draw samples from q(f)
key, subkey = jax.random.split(key)
base_draws = jax.random.normal(subkey, (64, num_samples))
f_samples = mean_final[None, :] + base_draws @ cholesky_final.T

# Bernoulli probit log-likelihood
log_lik_fn = lambda f: probit_log_lik(f, y_class)

elbo_mc = gaussx.variational_elbo_mc(log_lik_fn, f_samples, kl_final)
print(f"MC ELBO (N_ℰ=64): {float(elbo_mc):.3f}")

# More samples for better estimate
key, subkey = jax.random.split(key)
base_draws_large = jax.random.normal(subkey, (512, num_samples))
f_samples_large = mean_final[None, :] + base_draws_large @ cholesky_final.T
elbo_mc_large = gaussx.variational_elbo_mc(log_lik_fn, f_samples_large, kl_final)
print(f"MC ELBO (N_ℰ=512): {float(elbo_mc_large):.3f}")

# %% [markdown]
# ## 7. Bayesian Learning Rule Update
#
# The **Bayesian Learning Rule** (Khan & Rue, 2023) unifies many
# approximate inference algorithms as natural gradient descent on the
# variational objective. For a Gaussian variational family, each step
# takes the form
#
# $$\boldsymbol{\nu}_1^{(i+1)} = (1 - \gamma) \boldsymbol{\nu}_1^{(i)} + \gamma \left( \nabla_{\mathbf{f}} \log p(\mathbf{y} \mid \boldsymbol{\mu}_f) - \nabla^2_{\mathbf{f}} \log p(\mathbf{y} \mid \boldsymbol{\mu}_f) \boldsymbol{\mu}_f \right)$$
#
# $$\boldsymbol{\nu}_2^{(i+1)} = (1 - \gamma) \boldsymbol{\nu}_2^{(i)} + \frac{\gamma}{2} \nabla^2_{\mathbf{f}} \log p(\mathbf{y} \mid \boldsymbol{\mu}_f)$$
#
# where the gradient $\nabla_{\mathbf{f}}$ and the Hessian $\nabla^2_{\mathbf{f}}$ of the
# log-likelihood are evaluated at the current mean $\boldsymbol{\mu}_f$.
#
# `gaussx.blr_full_update` implements this directly.

# %%
# BLR update starting from the prior
nat1_blr = jnp.zeros(num_samples)
nat2_blr = -0.5 * gram_inverse
# γ
step_size_blr = 0.5
elbo_history_blr = []

for _i in range(n_steps_natural):
    # Λ_f = −2 ν_2
    precision_blr = -2.0 * nat2_blr
    # Σ_f = Λ_f⁻¹
    covariance_blr = jnp.linalg.inv(precision_blr)
    # μ_f = Σ_f ν_1
    mean_blr = covariance_blr @ nat1_blr

    grad_blr = jax.grad(lambda f: probit_log_lik(f, y_class))(mean_blr)
    hessian_blr = jax.hessian(lambda f: probit_log_lik(f, y_class))(mean_blr)

    # BLR full-rank update (includes prior contribution already in nat2)
    nat1_blr, nat2_blr = gaussx.blr_full_update(
        nat1_blr, nat2_blr, grad_blr, hessian_blr, step_size_blr
    )
    # Re-add prior precision (BLR updates the likelihood site only)
    # We handle this by noting nat2 already has the prior from init;
    # blr_full_update damps toward the likelihood target, so we
    # add back the damped prior contribution
    # ν_2 ← ν_2 + γ (−½ K_XX⁻¹)
    nat2_blr = nat2_blr + step_size_blr * (-0.5 * gram_inverse)

    # Track ELBO
    precision_cur = -2.0 * nat2_blr
    covariance_cur = jnp.linalg.inv(precision_cur)
    mean_cur = covariance_cur @ nat1_blr
    cholesky_cur = jnp.linalg.cholesky(covariance_cur)
    kl_cur = kl_gaussian(mean_cur, cholesky_cur, gram_train)
    key, subkey = jax.random.split(key)
    base_draws = jax.random.normal(subkey, (32, num_samples))
    f_samp = mean_cur[None, :] + base_draws @ cholesky_cur.T
    elbo_blr = gaussx.variational_elbo_mc(
        lambda f: probit_log_lik(f, y_class), f_samp, kl_cur
    )
    elbo_history_blr.append(float(elbo_blr))

print(f"BLR — final ELBO: {elbo_history_blr[-1]:.3f}")

# %% [markdown]
# ## 8. Gauss-Newton Precision via `gauss_newton_precision`
#
# When the Hessian is expensive or indefinite, the **Gauss-Newton
# approximation** provides a guaranteed PSD alternative:
#
# $$\boldsymbol{\Lambda}_{\mathrm{GN}} = \mathbf{J}_r^\top \mathbf{J}_r$$
#
# where $\mathbf{J}_r \in \mathbb{R}^{D_y \times D_z}$ is the Jacobian of the residual mapping $\mathbf{r}$ with respect to the latent $\mathbf{z} \in \mathbb{R}^{D_z}$. This is always
# positive semi-definite, making it safe for precision updates.
#
# `gaussx.gauss_newton_precision` returns a lineax operator; when
# $D_y < D_z$, it returns a
# `LowRankUpdate` for efficient Woodbury solves.

# %%
# Illustrate gauss_newton_precision
obs_size, latent_size = 10, 30
key, subkey = jax.random.split(key)
# J_r ∈ ℝ^{D_y×D_z}
residual_jacobian = jax.random.normal(subkey, (obs_size, latent_size))

# Build GN precision operator
# Λ_GN = J_rᵀ J_r
precision_gn = gaussx.gauss_newton_precision(residual_jacobian)
print(f"Jacobian shape: ({obs_size}, {latent_size})")
print(f"Operator type: {type(precision_gn).__name__}")
print(f"Operator shape: {precision_gn.as_matrix().shape}")

# Verify it matches J^T J
jacobian_gram = residual_jacobian.T @ residual_jacobian
error = jnp.max(jnp.abs(precision_gn.as_matrix() - jacobian_gram))
print(f"Max error vs J_r^T J_r: {float(error):.2e}")

# When D_y >= D_z, returns a dense MatrixLinearOperator
tall_jacobian = jax.random.normal(subkey, (50, 10))
precision_gn_dense = gaussx.gauss_newton_precision(tall_jacobian)
print(f"\nTall Jacobian ({50}, {10}) -> {type(precision_gn_dense).__name__}")

# Eigenvalues are all non-negative (PSD guarantee)
eigvals = jnp.linalg.eigvalsh(precision_gn.as_matrix())
print(f"Min eigenvalue of GN precision: {float(jnp.min(eigvals)):.2e} (>= 0)")

# %% [markdown]
# ## 9. Convergence Comparison and Posterior
#
# Let's compare the ELBO convergence of:
# - **Euclidean** (Adam on $\boldsymbol{\mu}_f, \mathbf{L}_{\boldsymbol{\Sigma}_f}$) — 300 steps
# - **Natural gradient** (`damped_natural_update`) — 60 steps
# - **BLR** (`blr_full_update`) — 60 steps

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ELBO convergence
ax = axes[0]
ax.plot(elbo_history_euclidean, label="Euclidean (Adam)", alpha=0.8)
ax.plot(elbo_history_natural, label="Natural gradient", linewidth=2)
ax.plot(elbo_history_blr, label="BLR", linestyle="--", linewidth=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("ELBO")
ax.set_yscale("symlog")
ax.set_title("ELBO convergence")
ax.legend(fontsize=9)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

# Posterior approximation
ax = axes[1]
test_inputs = jnp.linspace(-3.5, 3.5, 200)
# K_*X
gram_test_train = se_kernel(test_inputs, train_inputs)

# Posterior predictive from natural gradient result
# K_*X K_XX⁻¹
projection = gram_test_train @ gram_inverse
# μ_* = K_*X K_XX⁻¹ μ_f
f_pred = projection @ jnp.linalg.solve(precision_final, nat1)
# k_** − diag(K_*X K_XX⁻¹ K_X*)
conditional_var = kernel_variance - jnp.sum(projection * gram_test_train, axis=1)
# σ²_* = k_** − diag(K_*X K_XX⁻¹ K_X*) + diag(K_*X K_XX⁻¹ Σ_f K_XX⁻¹ K_X*)
f_var_pred = conditional_var + jnp.sum(
    projection * (projection @ covariance_final), axis=1
)
f_std = jnp.sqrt(jnp.clip(f_var_pred, 0.0))

# Plot predictive probabilities via probit
prob_pred = jax.scipy.stats.norm.cdf(f_pred / jnp.sqrt(1.0 + f_std**2))

ax.fill_between(
    test_inputs,
    jax.scipy.stats.norm.cdf(f_pred - 2 * f_std),
    jax.scipy.stats.norm.cdf(f_pred + 2 * f_std),
    alpha=0.2,
    color="C1",
    label="95% CI",
)
ax.plot(test_inputs, prob_pred, "C1-", linewidth=2, label="$p(y=1 \\mid x)$")
ax.scatter(
    train_inputs,
    0.5 * (y_class + 1),
    s=30,
    c=y_class,
    cmap="bwr",
    edgecolors="k",
    linewidths=0.5,
    zorder=5,
)
ax.set_xlabel("$x$")
ax.set_ylabel("$p(y = 1 \\mid x)$")
ax.set_title("Posterior predictive (natural gradient)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Summary
#
# | Function | Purpose |
# |----------|---------|
# | `damped_natural_update` | Convex combination in natural parameter space |
# | `newton_update` | Convert Newton derivatives to natural site parameters |
# | `blr_full_update` | Bayesian Learning Rule step (full-rank) |
# | `blr_diag_update` | Bayesian Learning Rule step (diagonal) |
# | `gauss_newton_precision` | PSD Hessian approximation $\mathbf{J}_r^\top \mathbf{J}_r$ |
# | `riemannian_psd_correction` | Stabilize indefinite Hessians |
# | `variational_elbo_gaussian` | Closed-form ELBO for Gaussian likelihoods |
# | `variational_elbo_mc` | Monte Carlo ELBO for any likelihood |
#
# **Key takeaways:**
#
# - Natural gradient descent converges in ~60 steps where Euclidean
#   gradient descent needs 300+ steps (and still hasn't fully converged).
# - The natural parameterization $(\boldsymbol{\nu}_1, \boldsymbol{\nu}_2)$ makes the update
#   a simple damped linear combination via `damped_natural_update`.
# - The Bayesian Learning Rule (`blr_full_update`) wraps the Newton
#   target computation and damped update into a single call.
# - `gauss_newton_precision` gives a guaranteed PSD precision operator,
#   using `LowRankUpdate` when the Jacobian is fat for efficient
#   downstream Woodbury solves.
# - `variational_elbo_gaussian` avoids Monte Carlo noise for Gaussian
#   likelihoods; `variational_elbo_mc` handles arbitrary likelihoods.
