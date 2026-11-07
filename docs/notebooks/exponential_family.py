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
# # Exponential Family Gaussians
#
# This notebook demonstrates the gaussx exponential family module --
# working with Gaussians in natural parameter form.
#
# **What you'll learn:**
#
# 1. The exponential family form of the Gaussian: natural parameters $\boldsymbol{\eta}_1$, $\boldsymbol{\eta}_2$
# 2. Converting between mean/covariance and natural parameters
# 3. Log-partition function, sufficient statistics, Fisher information
# 4. KL divergence via Bregman divergence in natural parameter space
# 5. Why natural parameters are useful (conjugate updates, message passing)

# %% [markdown]
# ## 1. Background
#
# Any member of the exponential family can be written as:
#
# $$q(\mathbf{x}; \boldsymbol{\eta}) = h(\mathbf{x}) \exp(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta}))$$
#
# For a multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}_x, \boldsymbol{\Sigma}_x)$ on $\mathbf{x} \in \mathbb{R}^{D_x}$ with precision $\boldsymbol{\Lambda}_x = \boldsymbol{\Sigma}_x^{-1}$, the natural parameters are:
#
# $$\boldsymbol{\eta}_1 = \boldsymbol{\Lambda}_x \boldsymbol{\mu}_x \in \mathbb{R}^{D_x}$$
#
# $$\boldsymbol{\eta}_2 = -\frac{1}{2} \boldsymbol{\Lambda}_x \in \mathbb{R}^{D_x \times D_x}$$
#
# The sufficient statistics are $T(\mathbf{x}) = (\mathbf{x}, \mathbf{x} \mathbf{x}^\top)$, and the log-partition function is:
#
# $$A(\boldsymbol{\eta}) = -\frac{1}{4} \boldsymbol{\eta}_1^\top \boldsymbol{\eta}_2^{-1} \boldsymbol{\eta}_1 - \frac{1}{2} \log \det(-2 \boldsymbol{\eta}_2) + \frac{D_x}{2} \log 2\pi$$
#
# This encodes the normalization constant.  Everything in gaussx's
# `_expfam` module is built on these identities.
#
# The Gaussian is the maximum-entropy distribution for a given mean and
# covariance (Jaynes, 1957), which makes its exponential family form
# central to several inference frameworks:
#
# - **Expectation propagation** (Minka, 2001) — message passing operates
#   directly in natural parameter space, where site updates are additive.
# - **Natural gradient methods** (Amari, 1998) — the Fisher information
#   metric gives the steepest descent direction in distribution space,
#   and natural parameters make the Fisher matrix readily available.
# - **Variational inference** (Wainwright & Jordan, 2008) — conjugate-
#   computation variational inference exploits natural parameter additivity
#   to perform closed-form coordinate updates.

# %% [markdown]
# ## 2. Setup

# %%
from __future__ import annotations

import warnings


warnings.filterwarnings("ignore", message=r".*IProgress.*")

import jax
import jax.numpy as jnp
import lineax as lx

import gaussx


jax.config.update("jax_enable_x64", True)

# %% [markdown]
# We will work with a simple 3D Gaussian throughout, so $D_x = 3$.

# %%
# D_x = 3
state_size = 3

# μ_x ∈ ℝ^{D_x}
mean = jnp.array([1.0, -0.5, 2.0])

# A small positive-definite covariance matrix
# Σ_x ∈ ℝ^{D_x×D_x}
covariance_matrix = jnp.array(
    [
        [2.0, 0.5, 0.3],
        [0.5, 1.0, 0.1],
        [0.3, 0.1, 1.5],
    ]
)
covariance = lx.MatrixLinearOperator(covariance_matrix)

# Λ_x = Σ_x⁻¹
precision_matrix = jnp.linalg.inv(covariance_matrix)
precision = lx.MatrixLinearOperator(precision_matrix)

print("mu =", mean)
print("Sigma =\n", covariance_matrix)
print("Lambda =\n", precision_matrix)

# %% [markdown]
# ## 3. Construction
#
# There are three ways to build a `GaussianExpFam`.

# %%
# Method 1: from mean and covariance
expfam_from_cov = gaussx.GaussianExpFam.from_mean_cov(mean, covariance)

# Method 2: from mean and precision
expfam_from_prec = gaussx.GaussianExpFam.from_mean_prec(mean, precision)

# Method 3: directly from natural parameters
# η_1 = Λ_x μ_x
eta1_manual = precision_matrix @ mean
# η_2 = −½ Λ_x
eta2_manual = -0.5 * precision_matrix
expfam_manual = gaussx.GaussianExpFam(
    eta1=eta1_manual,
    eta2=lx.MatrixLinearOperator(eta2_manual),
)

print("eta1 (from_mean_cov) =", expfam_from_cov.eta1)
print("eta1 (from_mean_prec) =", expfam_from_prec.eta1)
print("eta1 (manual)         =", expfam_manual.eta1)
print()
print("eta2 (from_mean_cov) =\n", expfam_from_cov.eta2.as_matrix())
print("eta2 (from_mean_prec) =\n", expfam_from_prec.eta2.as_matrix())
print("eta2 (manual)         =\n", expfam_manual.eta2.as_matrix())

# %%
# Verify all three agree
assert jnp.allclose(expfam_from_cov.eta1, expfam_from_prec.eta1, atol=1e-12)
assert jnp.allclose(expfam_from_cov.eta1, expfam_manual.eta1, atol=1e-12)
assert jnp.allclose(
    expfam_from_cov.eta2.as_matrix(), expfam_from_prec.eta2.as_matrix(), atol=1e-12
)
assert jnp.allclose(
    expfam_from_cov.eta2.as_matrix(), expfam_manual.eta2.as_matrix(), atol=1e-12
)
print("All three constructors produce identical natural parameters.")

# %% [markdown]
# ## 4. Conversions
#
# Round-trip: natural $\to$ expectation $\to$ natural.

# %%
# Recover mean and covariance from the exp-fam object
mean_recovered, covariance_recovered = gaussx.to_mean_cov(expfam_from_cov)

print("Original mu   =", mean)
print("Recovered mu  =", mean_recovered)
print("Match:", jnp.allclose(mean, mean_recovered, atol=1e-12))
print()
print("Original Sigma =\n", covariance_matrix)
print("Recovered Sigma =\n", covariance_recovered.as_matrix())
print(
    "Match:",
    jnp.allclose(covariance_matrix, covariance_recovered.as_matrix(), atol=1e-10),
)

# %%
# Also verify mean_cov_to_natural produces matching eta1, eta2
eta1_fn, eta2_fn = gaussx.mean_cov_to_natural(mean, covariance)

print("eta1 (mean_cov_to_natural) =", eta1_fn)
print("eta1 (object)     =", expfam_from_cov.eta1)
print("Match:", jnp.allclose(eta1_fn, expfam_from_cov.eta1, atol=1e-12))
print()
print("eta2 (mean_cov_to_natural) =\n", eta2_fn.as_matrix())
print("eta2 (object)     =\n", expfam_from_cov.eta2.as_matrix())
print(
    "Match:",
    jnp.allclose(eta2_fn.as_matrix(), expfam_from_cov.eta2.as_matrix(), atol=1e-12),
)

# %% [markdown]
# ## 5. Log-partition function
#
# The log-partition $A(\boldsymbol{\eta})$ encodes the normalization constant.
# We verify the gaussx implementation against a manual computation.

# %%
log_partition_gaussx = gaussx.log_partition(expfam_from_cov)

# Manual computation:
# A(η) = −¼ η_1ᵀ η_2⁻¹ η_1 − ½ log det(−2 η_2) + ½ D_x log 2π
# η_2⁻¹
eta2_inv = jnp.linalg.inv(expfam_from_cov.eta2.as_matrix())
# −¼ η_1ᵀ η_2⁻¹ η_1
quad_term = -0.25 * expfam_from_cov.eta1 @ eta2_inv @ expfam_from_cov.eta1
# −2 η_2
neg2_eta2 = -2.0 * expfam_from_cov.eta2.as_matrix()
# −½ log det(−2 η_2)
logdet_term = -0.5 * jnp.linalg.slogdet(neg2_eta2)[1]
# ½ D_x log 2π
base_term = 0.5 * state_size * jnp.log(2.0 * jnp.pi)
log_partition_manual = quad_term + logdet_term + base_term

print(f"A(eta) [gaussx]: {log_partition_gaussx:.10f}")
print(f"A(eta) [manual]: {log_partition_manual:.10f}")
print(f"Match: {jnp.allclose(log_partition_gaussx, log_partition_manual, atol=1e-10)}")

# %% [markdown]
# ## 6. Sufficient statistics
#
# For the Gaussian, $T(\mathbf{x}) = (\mathbf{x}, \mathbf{x} \mathbf{x}^\top)$.

# %%
# Single vector
# x ∈ ℝ^{D_x}
point = jnp.array([0.5, -1.0, 0.3])
# T(x) = (x, x xᵀ)
stat_1, stat_2 = gaussx.sufficient_stats(point)

print("x =", point)
print("T_1(x) = x =", stat_1)
print("T_2(x) = x x^T =\n", stat_2)
print("Matches outer product:", jnp.allclose(stat_2, jnp.outer(point, point)))

# %%
# Batch of vectors
# X ∈ ℝ^{3×D_x}
points = jnp.array(
    [
        [0.5, -1.0, 0.3],
        [1.0, 0.0, -0.5],
        [2.0, 1.0, 1.0],
    ]
)
stat_1_batch, stat_2_batch = gaussx.sufficient_stats(points)

print(f"Batch input shape: {points.shape}")
print(f"T_1 shape: {stat_1_batch.shape}")
print(f"T_2 shape: {stat_2_batch.shape}")
for i in range(points.shape[0]):
    ok = jnp.allclose(stat_2_batch[i], jnp.outer(points[i], points[i]))
    print(f"  Sample {i}: outer product match = {ok}")

# %% [markdown]
# ## 7. Fisher information
#
# For a Gaussian, the Fisher information of the mean $\boldsymbol{\mu}_x$ is the precision matrix $\boldsymbol{\Lambda}_x = \boldsymbol{\Sigma}_x^{-1}$, which is what `gaussx.fisher_info` returns.
#
# The Fisher information with respect to the natural parameters is a different matrix. For any exponential family it is the Hessian of the log-partition function:
#
# $$F_{ij} = \frac{\partial^2 A(\boldsymbol{\eta})}{\partial \eta_i \partial \eta_j}$$
#
# This connects the geometry of the natural parameter space to the
# curvature of $A(\boldsymbol{\eta})$, which is always convex
# (Barndorff-Nielsen, 1978).

# %%
# F = Λ_x
fisher = gaussx.fisher_info(expfam_from_cov)

print("Fisher info =\n", fisher.as_matrix())
print("Precision   =\n", precision_matrix)
print("Match:", jnp.allclose(fisher.as_matrix(), precision_matrix, atol=1e-10))

# %% [markdown]
# ## 8. KL divergence
#
# `gaussx.kl_divergence(q, p)` computes $\mathrm{KL}[q \Vert p]$ via the
# Bregman divergence in natural parameter space:
#
# $$\mathrm{KL}[q \Vert p] = A(\boldsymbol{\eta}_p) - A(\boldsymbol{\eta}_q) - (\boldsymbol{\eta}_p - \boldsymbol{\eta}_q)^\top \nabla A(\boldsymbol{\eta}_q)$$
#
# The Bregman divergence interpretation means KL can be computed using
# only the log-partition function and its gradient — no explicit matrix
# inversions needed beyond what is in the natural-to-expectation
# conversion. This is computationally advantageous when the precision
# has structure (e.g. Kronecker, block-diagonal, or sparse).
#
# We verify against the standard formula, with $q = \mathcal{N}(\boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q)$, $p = \mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$ and $\boldsymbol{\Lambda}_p = \boldsymbol{\Sigma}_p^{-1}$:
#
# $$\mathrm{KL}[q \Vert p] = \frac{1}{2} \left[ \mathrm{tr}(\boldsymbol{\Lambda}_p \boldsymbol{\Sigma}_q) + (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^\top \boldsymbol{\Lambda}_p (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q) - D_x + \log \det \boldsymbol{\Sigma}_p - \log \det \boldsymbol{\Sigma}_q \right]$$

# %%
# Create a second Gaussian
# μ_p ∈ ℝ^{D_x}
mean_p = jnp.array([0.0, 1.0, -1.0])
# Σ_p ∈ ℝ^{D_x×D_x}
covariance_p_matrix = jnp.array(
    [
        [1.5, 0.2, 0.0],
        [0.2, 2.0, 0.4],
        [0.0, 0.4, 1.0],
    ]
)
covariance_p = lx.MatrixLinearOperator(covariance_p_matrix)

expfam_p = gaussx.GaussianExpFam.from_mean_cov(mean_p, covariance_p)

# gaussx KL
kl_gaussx = gaussx.kl_divergence(expfam_from_cov, expfam_p)

# Standard formula KL[q ‖ p]
# Λ_p = Σ_p⁻¹
precision_p = jnp.linalg.inv(covariance_p_matrix)
# μ_p − μ_q
mean_diff = mean_p - mean
# KL = ½ [tr(Λ_p Σ_q) + (μ_p − μ_q)ᵀ Λ_p (μ_p − μ_q) − D_x + log det Σ_p − log det Σ_q]
kl_standard = 0.5 * (
    jnp.trace(precision_p @ covariance_matrix)
    + mean_diff @ precision_p @ mean_diff
    - state_size
    + jnp.linalg.slogdet(covariance_p_matrix)[1]
    - jnp.linalg.slogdet(covariance_matrix)[1]
)

print(f"KL(q || p) [gaussx]:   {kl_gaussx:.10f}")
print(f"KL(q || p) [standard]: {kl_standard:.10f}")
print(f"Match: {jnp.allclose(kl_gaussx, kl_standard, atol=1e-8)}")

# %% [markdown]
# ## 9. Why natural parameters?
#
# In natural parameter space, **conjugate updates are just addition**.
# When we combine a Gaussian prior with a Gaussian likelihood site, the
# posterior natural parameters are the sum of the prior and site natural
# parameters:
#
# $$\boldsymbol{\eta}_\text{post} = \boldsymbol{\eta}_\text{prior} + \boldsymbol{\eta}_\text{site}$$
#
# This makes message passing and variational inference extremely simple:
# no matrix inversions are needed for the update itself.
#
# The site comes from a single noisy observation $\mathbf{y} = \mathbf{x} + \boldsymbol{\varepsilon}$ with $\mathbf{y} \in \mathbb{R}^{D_y}$, $D_y = D_x$ and $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\varepsilon)$, so its natural parameters are $\boldsymbol{\eta}_{1,\text{site}} = \boldsymbol{\Lambda}_\varepsilon \mathbf{y}$ and $\boldsymbol{\eta}_{2,\text{site}} = -\frac{1}{2} \boldsymbol{\Lambda}_\varepsilon$ with $\boldsymbol{\Lambda}_\varepsilon = \boldsymbol{\Sigma}_\varepsilon^{-1}$.

# %%
# Prior: our original Gaussian
prior = expfam_from_cov
print("Prior mean =", mean)

# A likelihood "site" -- e.g. from a single noisy observation
# Observation model: y = x + ε, ε ~ 𝒩(0, Σ_ε)
# Σ_ε = 0.1 I
obs_noise_covariance = 0.1 * jnp.eye(state_size)  # observation noise
# y ∈ ℝ^{D_y}
obs = jnp.array([1.2, -0.3, 2.1])  # observed value

# Site natural parameters: η_1,site = Λ_ε y, η_2,site = −½ Λ_ε
# Λ_ε = Σ_ε⁻¹
obs_noise_precision = jnp.linalg.inv(obs_noise_covariance)
site_eta1 = obs_noise_precision @ obs
site_eta2_matrix = -0.5 * obs_noise_precision

print(f"Site eta1 = {site_eta1}")

# %%
# Posterior = prior + site (just addition in natural parameter space!)
# η_1,post = η_1,prior + η_1,site
post_eta1 = prior.eta1 + site_eta1
# η_2,post = η_2,prior + η_2,site
post_eta2_matrix = prior.eta2.as_matrix() + site_eta2_matrix

posterior = gaussx.GaussianExpFam(
    eta1=post_eta1,
    eta2=lx.MatrixLinearOperator(post_eta2_matrix),
)

# Recover posterior mean and covariance
posterior_mean, posterior_covariance = gaussx.to_mean_cov(posterior)

print("Posterior mean =", posterior_mean)
print("Posterior cov  =\n", posterior_covariance.as_matrix())

# %%
# Verify against the standard Gaussian conditioning formula:
# Λ_post = Λ_x + Λ_ε
# η_1,post = Λ_x μ_x + Λ_ε y
# μ_post = Σ_post η_1,post
expected_posterior_precision = precision_matrix + obs_noise_precision
expected_posterior_covariance = jnp.linalg.inv(expected_posterior_precision)
expected_posterior_mean = expected_posterior_covariance @ (
    precision_matrix @ mean + obs_noise_precision @ obs
)

print("Expected posterior mean =", expected_posterior_mean)
print("Mean match:", jnp.allclose(posterior_mean, expected_posterior_mean, atol=1e-10))
print()
print("Expected posterior cov =\n", expected_posterior_covariance)
print(
    "Cov match:",
    jnp.allclose(
        posterior_covariance.as_matrix(), expected_posterior_covariance, atol=1e-10
    ),
)

# %% [markdown]
# ## 10. Summary
#
# | Concept | gaussx function |
# |---|---|
# | Build from mean/cov | `GaussianExpFam.from_mean_cov(mu, Sigma)` |
# | Build from mean/prec | `GaussianExpFam.from_mean_prec(mu, Lambda)` |
# | Natural $\to$ mean, covariance | `gaussx.to_mean_cov(q)` |
# | Mean, covariance $\to$ natural | `gaussx.mean_cov_to_natural(mu, Sigma)` |
# | Log-partition $A(\boldsymbol{\eta})$ | `gaussx.log_partition(q)` |
# | Sufficient statistics | `gaussx.sufficient_stats(x)` |
# | Fisher information | `gaussx.fisher_info(q)` |
# | KL divergence | `gaussx.kl_divergence(q, p)` |
#
# The key takeaway: natural parameters turn Bayesian updates into
# **simple addition**, making them the natural choice for message passing,
# variational inference, and conjugate update algorithms.

# %% [markdown]
# ## References
#
# - Amari, S. (1998). Natural gradient works efficiently in learning.
#   *Neural Computation*, 10(2), 251-276.
# - Barndorff-Nielsen, O. (1978). *Information and Exponential Families
#   in Statistical Theory*. Wiley.
# - Jaynes, E. T. (1957). Information theory and statistical mechanics.
#   *Physical Review*, 106(4), 620-630.
# - Minka, T. P. (2001). *A Family of Algorithms for Approximate
#   Bayesian Inference*. PhD thesis, MIT.
# - Wainwright, M. J. & Jordan, M. I. (2008). Graphical models,
#   exponential families, and variational inference. *Foundations and
#   Trends in Machine Learning*, 1(1-2), 1-305.
