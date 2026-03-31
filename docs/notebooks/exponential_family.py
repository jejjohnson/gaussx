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
# 1. The exponential family form of the Gaussian: natural parameters $\eta_1$, $\eta_2$
# 2. Converting between mean/covariance and natural parameters
# 3. Log-partition function, sufficient statistics, Fisher information
# 4. KL divergence via Bregman divergence in natural parameter space
# 5. Why natural parameters are useful (conjugate updates, message passing)

# %% [markdown]
# ## 1. Background
#
# Any member of the exponential family can be written as:
#
# $$
# q(x \mid \eta) = h(x) \exp\!\bigl(\eta^\top T(x) - A(\eta)\bigr)
# $$
#
# For a multivariate Gaussian with mean $\mu$ and precision
# $\Lambda = \Sigma^{-1}$, the natural parameters are:
#
# $$
# \eta_1 = \Lambda \mu, \qquad \eta_2 = -\tfrac{1}{2}\Lambda
# $$
#
# The sufficient statistics are $T(x) = [x,\; x x^\top]$, and the
# log-partition function is:
#
# $$
# A(\eta) = -\tfrac{1}{4}\,\eta_1^\top \eta_2^{-1} \eta_1
#           - \tfrac{1}{2}\log\lvert -2\eta_2 \rvert
#           + \tfrac{N}{2}\log(2\pi)
# $$
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

import jax
import jax.numpy as jnp
import lineax as lx

import gaussx


jax.config.update("jax_enable_x64", True)

# %% [markdown]
# We will work with a simple 3D Gaussian throughout.

# %%
N = 3

mu = jnp.array([1.0, -0.5, 2.0])

# A small positive-definite covariance matrix
Sigma_mat = jnp.array(
    [
        [2.0, 0.5, 0.3],
        [0.5, 1.0, 0.1],
        [0.3, 0.1, 1.5],
    ]
)
Sigma_op = lx.MatrixLinearOperator(Sigma_mat)

# Precision = Sigma^{-1}
Lambda_mat = jnp.linalg.inv(Sigma_mat)
Lambda_op = lx.MatrixLinearOperator(Lambda_mat)

print("mu =", mu)
print("Sigma =\n", Sigma_mat)
print("Lambda =\n", Lambda_mat)

# %% [markdown]
# ## 3. Construction
#
# There are three ways to build a `GaussianExpFam`.

# %%
# Method 1: from mean and covariance
q1 = gaussx.GaussianExpFam.from_mean_cov(mu, Sigma_op)

# Method 2: from mean and precision
q2 = gaussx.GaussianExpFam.from_mean_prec(mu, Lambda_op)

# Method 3: directly from natural parameters
eta1_manual = Lambda_mat @ mu
eta2_manual = -0.5 * Lambda_mat
q3 = gaussx.GaussianExpFam(
    eta1=eta1_manual,
    eta2=lx.MatrixLinearOperator(eta2_manual),
)

print("eta1 (from_mean_cov) =", q1.eta1)
print("eta1 (from_mean_prec) =", q2.eta1)
print("eta1 (manual)         =", q3.eta1)
print()
print("eta2 (from_mean_cov) =\n", q1.eta2.as_matrix())
print("eta2 (from_mean_prec) =\n", q2.eta2.as_matrix())
print("eta2 (manual)         =\n", q3.eta2.as_matrix())

# %%
# Verify all three agree
assert jnp.allclose(q1.eta1, q2.eta1, atol=1e-12)
assert jnp.allclose(q1.eta1, q3.eta1, atol=1e-12)
assert jnp.allclose(q1.eta2.as_matrix(), q2.eta2.as_matrix(), atol=1e-12)
assert jnp.allclose(q1.eta2.as_matrix(), q3.eta2.as_matrix(), atol=1e-12)
print("All three constructors produce identical natural parameters.")

# %% [markdown]
# ## 4. Conversions
#
# Round-trip: natural $\to$ expectation $\to$ natural.

# %%
# Recover mean and covariance from the exp-fam object
mu_recovered, Sigma_recovered = gaussx.to_expectation(q1)

print("Original mu   =", mu)
print("Recovered mu  =", mu_recovered)
print("Match:", jnp.allclose(mu, mu_recovered, atol=1e-12))
print()
print("Original Sigma =\n", Sigma_mat)
print("Recovered Sigma =\n", Sigma_recovered.as_matrix())
print("Match:", jnp.allclose(Sigma_mat, Sigma_recovered.as_matrix(), atol=1e-10))

# %%
# Also verify to_natural produces matching eta1, eta2
eta1_fn, eta2_fn = gaussx.to_natural(mu, Sigma_op)

print("eta1 (to_natural) =", eta1_fn)
print("eta1 (object)     =", q1.eta1)
print("Match:", jnp.allclose(eta1_fn, q1.eta1, atol=1e-12))
print()
print("eta2 (to_natural) =\n", eta2_fn.as_matrix())
print("eta2 (object)     =\n", q1.eta2.as_matrix())
print("Match:", jnp.allclose(eta2_fn.as_matrix(), q1.eta2.as_matrix(), atol=1e-12))

# %% [markdown]
# ## 5. Log-partition function
#
# The log-partition $A(\eta)$ encodes the normalization constant.
# We verify the gaussx implementation against a manual computation.

# %%
A_gaussx = gaussx.log_partition(q1)

# Manual computation:
# A(eta) = -0.25 * eta1^T eta2^{-1} eta1 - 0.5 * log|-2 eta2| + N/2 * log(2pi)
eta2_inv = jnp.linalg.inv(q1.eta2.as_matrix())
quad_term = -0.25 * q1.eta1 @ eta2_inv @ q1.eta1
neg2_eta2 = -2.0 * q1.eta2.as_matrix()
logdet_term = -0.5 * jnp.linalg.slogdet(neg2_eta2)[1]
base_term = 0.5 * N * jnp.log(2.0 * jnp.pi)
A_manual = quad_term + logdet_term + base_term

print(f"A(eta) [gaussx]: {A_gaussx:.10f}")
print(f"A(eta) [manual]: {A_manual:.10f}")
print(f"Match: {jnp.allclose(A_gaussx, A_manual, atol=1e-10)}")

# %% [markdown]
# ## 6. Sufficient statistics
#
# For the Gaussian, $T(x) = [x,\; x x^\top]$.

# %%
# Single vector
x = jnp.array([0.5, -1.0, 0.3])
t1, t2 = gaussx.sufficient_stats(x)

print("x =", x)
print("T_1(x) = x =", t1)
print("T_2(x) = x x^T =\n", t2)
print("Matches outer product:", jnp.allclose(t2, jnp.outer(x, x)))

# %%
# Batch of vectors
X = jnp.array(
    [
        [0.5, -1.0, 0.3],
        [1.0, 0.0, -0.5],
        [2.0, 1.0, 1.0],
    ]
)
t1_batch, t2_batch = gaussx.sufficient_stats(X)

print(f"Batch input shape: {X.shape}")
print(f"T_1 shape: {t1_batch.shape}")
print(f"T_2 shape: {t2_batch.shape}")
for i in range(X.shape[0]):
    ok = jnp.allclose(t2_batch[i], jnp.outer(X[i], X[i]))
    print(f"  Sample {i}: outer product match = {ok}")

# %% [markdown]
# ## 7. Fisher information
#
# For a Gaussian, the Fisher information in the natural parameterization
# equals the precision matrix $\Lambda = \Sigma^{-1}$.
#
# More generally, for any exponential family the Fisher information is
# the Hessian of the log-partition function:
#
# $$
# F_{ij} = \frac{\partial^2 A(\eta)}{\partial \eta_i \,\partial \eta_j}
# $$
#
# This connects the geometry of the natural parameter space to the
# curvature of $A(\eta)$, which is always convex
# (Barndorff-Nielsen, 1978).

# %%
F = gaussx.fisher_info(q1)

print("Fisher info =\n", F.as_matrix())
print("Precision   =\n", Lambda_mat)
print("Match:", jnp.allclose(F.as_matrix(), Lambda_mat, atol=1e-10))

# %% [markdown]
# ## 8. KL divergence
#
# `gaussx.kl_divergence(q, p)` computes $\text{KL}(q \| p)$ via the
# Bregman divergence in natural parameter space:
#
# $$
# \text{KL}(q \| p) = A(\eta_p) - A(\eta_q)
#     - (\eta_p - \eta_q)^\top \nabla A(\eta_q)
# $$
#
# The Bregman divergence interpretation means KL can be computed using
# only the log-partition function and its gradient — no explicit matrix
# inversions needed beyond what is in the natural-to-expectation
# conversion. This is computationally advantageous when the precision
# has structure (e.g. Kronecker, block-diagonal, or sparse).
#
# We verify against the standard formula:
#
# $$
# \text{KL}(q \| p) = \tfrac{1}{2}\bigl[
#     \text{tr}(\Sigma_p^{-1}\Sigma_q)
#     + (\mu_p - \mu_q)^\top \Sigma_p^{-1}(\mu_p - \mu_q)
#     - N + \log\tfrac{|\Sigma_p|}{|\Sigma_q|}
# \bigr]
# $$

# %%
# Create a second Gaussian
mu_p = jnp.array([0.0, 1.0, -1.0])
Sigma_p_mat = jnp.array(
    [
        [1.5, 0.2, 0.0],
        [0.2, 2.0, 0.4],
        [0.0, 0.4, 1.0],
    ]
)
Sigma_p_op = lx.MatrixLinearOperator(Sigma_p_mat)

p = gaussx.GaussianExpFam.from_mean_cov(mu_p, Sigma_p_op)

# gaussx KL
kl_gaussx = gaussx.kl_divergence(q1, p)

# Standard formula KL(q || p)
Lambda_p = jnp.linalg.inv(Sigma_p_mat)
diff = mu_p - mu
kl_standard = 0.5 * (
    jnp.trace(Lambda_p @ Sigma_mat)
    + diff @ Lambda_p @ diff
    - N
    + jnp.linalg.slogdet(Sigma_p_mat)[1]
    - jnp.linalg.slogdet(Sigma_mat)[1]
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
# $$
# \eta_{\text{post}} = \eta_{\text{prior}} + \eta_{\text{site}}
# $$
#
# This makes message passing and variational inference extremely simple:
# no matrix inversions are needed for the update itself.

# %%
# Prior: our original Gaussian q1
prior = q1
print("Prior mean =", mu)

# A likelihood "site" -- e.g. from a single noisy observation
# Observation model: y = x + noise, noise ~ N(0, R)
R_mat = 0.1 * jnp.eye(N)  # observation noise
y_obs = jnp.array([1.2, -0.3, 2.1])  # observed value

# Site natural parameters: eta1_site = R^{-1} y, eta2_site = -0.5 R^{-1}
R_inv = jnp.linalg.inv(R_mat)
site_eta1 = R_inv @ y_obs
site_eta2_mat = -0.5 * R_inv

print(f"Site eta1 = {site_eta1}")

# %%
# Posterior = prior + site (just addition in natural parameter space!)
post_eta1 = prior.eta1 + site_eta1
post_eta2_mat = prior.eta2.as_matrix() + site_eta2_mat

posterior = gaussx.GaussianExpFam(
    eta1=post_eta1,
    eta2=lx.MatrixLinearOperator(post_eta2_mat),
)

# Recover posterior mean and covariance
mu_post, Sigma_post = gaussx.to_expectation(posterior)

print("Posterior mean =", mu_post)
print("Posterior cov  =\n", Sigma_post.as_matrix())

# %%
# Verify against the standard Gaussian conditioning formula:
# Lambda_post = Lambda_prior + R^{-1}
# eta1_post = Lambda_prior @ mu_prior + R^{-1} @ y
# mu_post = Sigma_post @ eta1_post
Lambda_post_expected = Lambda_mat + R_inv
Sigma_post_expected = jnp.linalg.inv(Lambda_post_expected)
mu_post_expected = Sigma_post_expected @ (Lambda_mat @ mu + R_inv @ y_obs)

print("Expected posterior mean =", mu_post_expected)
print("Mean match:", jnp.allclose(mu_post, mu_post_expected, atol=1e-10))
print()
print("Expected posterior cov =\n", Sigma_post_expected)
print(
    "Cov match:", jnp.allclose(Sigma_post.as_matrix(), Sigma_post_expected, atol=1e-10)
)

# %% [markdown]
# ## 10. Summary
#
# | Concept | gaussx function |
# |---|---|
# | Build from mean/cov | `GaussianExpFam.from_mean_cov(mu, Sigma)` |
# | Build from mean/prec | `GaussianExpFam.from_mean_prec(mu, Lambda)` |
# | Natural $\to$ expectation | `gaussx.to_expectation(q)` |
# | Expectation $\to$ natural | `gaussx.to_natural(mu, Sigma)` |
# | Log-partition $A(\eta)$ | `gaussx.log_partition(q)` |
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
