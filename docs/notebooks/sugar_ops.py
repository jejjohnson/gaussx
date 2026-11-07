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
# # Sugar Operations for Gaussians and GPs
#
# gaussx provides "sugar" functions that combine primitives (`solve`,
# `logdet`, `trace`, etc.) into compound operations common in Gaussian
# and GP workflows.
#
# **What you'll learn:**
#
# 1. `quadratic_form` and `gaussian_log_prob` for evaluating distributions
# 2. `gaussian_entropy` and `kl_standard_normal` for information-theoretic quantities
# 3. `log_marginal_likelihood` for GP model selection
# 4. `schur_complement` and `cov_transform` for conditional distributions
# 5. `whiten_covariance` and `unwhiten` for reparameterization
#
# These compound operations appear repeatedly in GP inference
# (Rasmussen & Williams, 2006), variational inference (Blei et al.,
# 2017), and Bayesian neural networks. gaussx sugar functions combine
# primitives into numerically stable implementations — for example,
# `log_marginal_likelihood` avoids separate computation of the
# quadratic form and logdet, which can be more efficient with certain
# solver strategies.

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
# ## 1. Setup
#
# We build a small PSD kernel matrix from an RBF kernel on 20 points
# and wrap it as a lineax operator.

# %%
N = 20
key = jax.random.PRNGKey(42)
inputs = jnp.linspace(0.0, 5.0, N)

# RBF kernel: k(x, x′) = exp(−½ ‖x − x′‖² / ℓ²)
lengthscale = 1.0
difference = inputs[:, None] - inputs[None, :]
kernel_matrix = jnp.exp(-0.5 * difference**2 / lengthscale**2)

# Wrap as lineax operator with a bit of jitter for numerical stability
# Σ = K_XX + 10⁻⁶ I
covariance = kernel_matrix + 1e-6 * jnp.eye(N)
covariance_op = lx.MatrixLinearOperator(covariance, lx.positive_semidefinite_tag)
print("Sigma shape:", covariance.shape)

# %% [markdown]
# ## 2. Quadratic forms
#
# `quadratic_form(Sigma_op, x)` computes $\mathbf{x}^\top \boldsymbol{\Sigma}^{-1} \mathbf{x}$ via a single solve -- the squared Mahalanobis distance.

# %%
key, vector_key = jax.random.split(key)
vector = jax.random.normal(vector_key, (N,))

quadratic = gaussx.quadratic_form(covariance_op, vector)

# Manual verification: solve then dot
# α = Σ⁻¹ v
alpha = gaussx.solve(covariance_op, vector)
# q = vᵀ α
quadratic_manual = vector @ alpha
print("quadratic_form:", quadratic)
print("manual v^T Sigma^{-1} v:", quadratic_manual)
print("match:", jnp.allclose(quadratic, quadratic_manual))

# %% [markdown]
# ## 3. Log-probability
#
# `gaussian_log_prob(mu, Sigma_op, x)` evaluates the multivariate normal log-pdf:
#
# $$\log \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{1}{2} (N \log 2\pi + \log \det \boldsymbol{\Sigma} + (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}))$$

# %%
mean = jnp.zeros(N)
log_prob = gaussx.gaussian_log_prob(mean, covariance_op, vector)

# Manual verification
logdet = gaussx.logdet(covariance_op)
# r = x − μ
residual = vector - mean
# rᵀ Σ⁻¹ r
quadratic_residual = residual @ gaussx.solve(covariance_op, residual)
# log 𝒩(x; μ, Σ) = −½ (N log 2π + log det Σ + rᵀ Σ⁻¹ r)
log_prob_manual = -0.5 * (N * jnp.log(2.0 * jnp.pi) + logdet + quadratic_residual)
print("gaussian_log_prob:", log_prob)
print("manual log-prob:  ", log_prob_manual)
print("match:", jnp.allclose(log_prob, log_prob_manual))

# %% [markdown]
# It also works with structured operators. Here we use a Kronecker product
# to build a 2D covariance:

# %%
factor_a = jnp.eye(3) + 0.5 * jnp.ones((3, 3))
factor_b = jnp.diag(jnp.array([1.0, 2.0]))
factor_a_op = lx.MatrixLinearOperator(factor_a, lx.positive_semidefinite_tag)
factor_b_op = lx.MatrixLinearOperator(factor_b, lx.positive_semidefinite_tag)
kronecker_op = gaussx.Kronecker(factor_a_op, factor_b_op)

key, kron_vector_key = jax.random.split(key)
kron_vector = jax.random.normal(kron_vector_key, (6,))
log_prob_kron = gaussx.gaussian_log_prob(jnp.zeros(6), kronecker_op, kron_vector)
print("log-prob (Kronecker):", log_prob_kron)

# %% [markdown]
# ## 4. Entropy and KL divergence
#
# Entropy of $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:
#
# $$\mathbb{H}[\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})] = \frac{1}{2} (N (1 + \log 2\pi) + \log \det \boldsymbol{\Sigma})$$
#
# KL to the standard normal $\mathrm{KL}[\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \Vert \mathcal{N}(\mathbf{0}, \mathbf{I})]$:
#
# $$\mathrm{KL}[\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \Vert \mathcal{N}(\mathbf{0}, \mathbf{I})] = \frac{1}{2} (\mathrm{tr}(\boldsymbol{\Sigma}) + \boldsymbol{\mu}^\top \boldsymbol{\mu} - N - \log \det \boldsymbol{\Sigma})$$

# %%
entropy = gaussx.gaussian_entropy(covariance_op)
print("entropy:", entropy)

variational_mean = 0.1 * jnp.ones(N)
kl = gaussx.kl_standard_normal(variational_mean, covariance_op)

# Manual KL verification
# tr(Σ)
trace_covariance = gaussx.trace(covariance_op)
# μᵀ μ
mean_norm_sq = variational_mean @ variational_mean
# log det Σ
logdet_covariance = gaussx.logdet(covariance_op)
# KL = ½ (tr(Σ) + μᵀ μ − N − log det Σ)
kl_manual = 0.5 * (trace_covariance + mean_norm_sq - N - logdet_covariance)
print("KL(q || p):", kl)
print("manual KL: ", kl_manual)
print("match:", jnp.allclose(kl, kl_manual))

# %% [markdown]
# ## 5. GP log-marginal likelihood
#
# For GP regression with observations $\mathbf{y}$, prior mean $\boldsymbol{\mu}_y$, and covariance $\boldsymbol{\Sigma}_y = \mathbf{K}_{XX} + \sigma_\varepsilon^2 \mathbf{I}$:
#
# $$\log p(\mathbf{y}) = -\frac{1}{2} ((\mathbf{y} - \boldsymbol{\mu}_y)^\top \boldsymbol{\Sigma}_y^{-1} (\mathbf{y} - \boldsymbol{\mu}_y) + \log \det \boldsymbol{\Sigma}_y + N \log 2\pi)$$

# %%
# Generate data from a sine function + noise
noise_variance = 0.1
key, noise_key = jax.random.split(key)
obs = jnp.sin(inputs) + jax.random.normal(noise_key, (N,)) * jnp.sqrt(noise_variance)

# Build Σ_y = K_XX + σ_ε² I
obs_covariance = kernel_matrix + noise_variance * jnp.eye(N)
obs_covariance_op = lx.MatrixLinearOperator(
    obs_covariance, lx.positive_semidefinite_tag
)

log_marginal = gaussx.log_marginal_likelihood(jnp.zeros(N), obs_covariance_op, obs)
print("log marginal likelihood:", log_marginal)

# %% [markdown]
# Differentiating the LML with `jax.grad` for hyperparameter optimization:


# %%
def neg_lml(log_lengthscale, log_noise_variance, inputs, obs):
    lengthscale = jnp.exp(log_lengthscale)
    noise_variance = jnp.exp(log_noise_variance)
    difference = inputs[:, None] - inputs[None, :]
    # Σ_y = K_XX + σ_ε² I
    obs_covariance = jnp.exp(-0.5 * difference**2 / lengthscale**2)
    obs_covariance = obs_covariance + noise_variance * jnp.eye(len(obs))
    operator = lx.MatrixLinearOperator(obs_covariance, lx.positive_semidefinite_tag)
    return -gaussx.log_marginal_likelihood(jnp.zeros_like(obs), operator, obs)


grad_fn = jax.grad(neg_lml, argnums=(0, 1))
grads = grad_fn(jnp.log(1.0), jnp.log(0.1), inputs, obs)
print("grad w.r.t. log-lengthscale:", grads[0])
print("grad w.r.t. log-noise-var:  ", grads[1])

# %% [markdown]
# ## 6. Schur complement
#
# For GP conditional distributions, the Schur complement gives the conditional covariance:
#
# $$\mathbf{K}_{X \mid Z} = \mathbf{K}_{XX} - \mathbf{K}_{XZ} \mathbf{K}_{ZZ}^{-1} \mathbf{K}_{ZX}$$
#
# `schur_complement` returns a `LowRankUpdate` operator that
# preserves the low-rank structure.
#
# The Schur complement is the key identity behind GP conditional distributions. Given the joint $\begin{pmatrix} \mathbf{f}_X \\ \mathbf{f}_Z \end{pmatrix} \sim \mathcal{N}\left(\mathbf{0}, \begin{pmatrix} \mathbf{K}_{XX} & \mathbf{K}_{XZ} \\ \mathbf{K}_{ZX} & \mathbf{K}_{ZZ} \end{pmatrix}\right)$, the conditional $\mathbf{f}_X \mid \mathbf{f}_Z$ has covariance $\mathbf{K}_{XX} - \mathbf{K}_{XZ} \mathbf{K}_{ZZ}^{-1} \mathbf{K}_{ZX}$. When $M \ll N$, this is a rank-$M$ correction — exactly a `LowRankUpdate`.

# %%
M = 8
inducing_inputs = jnp.linspace(0.5, 4.5, M)

# Build kernel blocks
difference_xx = inputs[:, None] - inputs[None, :]
gram_train = jnp.exp(-0.5 * difference_xx**2) + 1e-6 * jnp.eye(N)
difference_xz = inputs[:, None] - inducing_inputs[None, :]
gram_train_inducing = jnp.exp(-0.5 * difference_xz**2)
difference_zz = inducing_inputs[:, None] - inducing_inputs[None, :]
gram_inducing = jnp.exp(-0.5 * difference_zz**2) + 1e-6 * jnp.eye(M)

gram_train_op = lx.MatrixLinearOperator(gram_train, lx.positive_semidefinite_tag)
gram_inducing_op = lx.MatrixLinearOperator(gram_inducing, lx.positive_semidefinite_tag)

schur = gaussx.schur_complement(gram_train_op, gram_train_inducing, gram_inducing_op)
print("type:", type(schur).__name__)

# Verify against manual computation
# K_ZZ⁻¹ K_ZX
inducing_solve = jnp.linalg.solve(gram_inducing, gram_train_inducing.T)
# K_XX − K_XZ K_ZZ⁻¹ K_ZX
schur_manual = gram_train - gram_train_inducing @ inducing_solve
schur_matrix = schur.as_matrix()
print("max error:", jnp.max(jnp.abs(schur_matrix - schur_manual)))

# %% [markdown]
# ## 7. Covariance propagation
#
# `cov_transform(J, Sigma_op)` propagates uncertainty through a linear map: $\mathbf{J} \boldsymbol{\Sigma} \mathbf{J}^\top$ with $\mathbf{J} \in \mathbb{R}^{M \times N}$.

# %%
num_outputs = 5
key, linear_map_key = jax.random.split(key)
linear_map = jax.random.normal(linear_map_key, (num_outputs, N))

propagated_op = gaussx.cov_transform(linear_map, covariance_op)

# Verify J Σ Jᵀ
propagated_manual = linear_map @ covariance @ linear_map.T
print("max error:", jnp.max(jnp.abs(propagated_op.as_matrix() - propagated_manual)))

# %% [markdown]
# ## 8. Whitening and unwhitening
#
# The reparameterization trick for Gaussians: sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, then compute $\mathbf{x} = \mathbf{L} \mathbf{z}$ to get $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ where $\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\top$.
#
# `whiten_covariance(L, S_tilde)` computes $\mathbf{L} \boldsymbol{\Sigma}_z \mathbf{L}^\top$, and `unwhiten(m_tilde, L)` computes $\mathbf{L} \boldsymbol{\mu}_z$.
#
# The reparameterization trick (Kingma & Welling, 2014) enables gradient-based optimization of variational objectives by expressing samples $\mathbf{x} = \mathbf{L} \mathbf{z} + \boldsymbol{\mu}$ where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, making the sampling differentiable w.r.t. the distribution parameters $\boldsymbol{\mu}$ and $\mathbf{L}$.

# %%
# Cholesky factor of our covariance
# Σ = L Lᵀ
cholesky_factor = jnp.linalg.cholesky(covariance)
cholesky_op = lx.MatrixLinearOperator(cholesky_factor)

# Whitened parameters (identity covariance in whitened space)
whitened_covariance_op = lx.IdentityLinearOperator(
    jax.ShapeDtypeStruct((N,), jnp.float64)
)

# Unwhiten covariance: L I Lᵀ = L Lᵀ = Σ
recovered_covariance = gaussx.whiten_covariance(cholesky_op, whitened_covariance_op)
print(
    "max error (covariance):",
    jnp.max(jnp.abs(recovered_covariance.as_matrix() - covariance)),
)

# Unwhiten a sample: z ~ 𝒩(0, I), x = L z ~ 𝒩(0, Σ)
key, sample_key = jax.random.split(key)
whitened_sample = jax.random.normal(sample_key, (N,))
unwhitened_sample = gaussx.unwhiten(whitened_sample, cholesky_op)
unwhitened_manual = cholesky_factor @ whitened_sample
print("max error (sample):", jnp.max(jnp.abs(unwhitened_sample - unwhitened_manual)))

# Verify empirically: many samples through L should have covariance ~ Σ
key, empirical_key = jax.random.split(key)
whitened_samples = jax.random.normal(empirical_key, (10_000, N))
# each row is L zᵢ
samples = whitened_samples @ cholesky_factor.T
empirical_covariance = jnp.cov(samples.T)
print(
    "mean abs error (empirical cov):",
    jnp.mean(jnp.abs(empirical_covariance - covariance)),
)

# %% [markdown]
# ## Summary
#
# This notebook demonstrated the main gaussx sugar operations:
#
# | Function | What it computes |
# |----------|-----------------|
# | `quadratic_form` | $\mathbf{x}^\top \boldsymbol{\Sigma}^{-1} \mathbf{x}$ (Mahalanobis distance squared) |
# | `gaussian_log_prob` | Multivariate normal log-pdf |
# | `gaussian_entropy` | $\mathbb{H}[\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})]$ |
# | `kl_standard_normal` | $\mathrm{KL}[\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \Vert \mathcal{N}(\mathbf{0}, \mathbf{I})]$ |
# | `log_marginal_likelihood` | GP log evidence (differentiable) |
# | `schur_complement` | $\mathbf{K}_{XX} - \mathbf{K}_{XZ} \mathbf{K}_{ZZ}^{-1} \mathbf{K}_{ZX}$ as `LowRankUpdate` |
# | `cov_transform` | $\mathbf{J} \boldsymbol{\Sigma} \mathbf{J}^\top$ (uncertainty propagation) |
# | `whiten_covariance` / `unwhiten` | Reparameterization trick |
#
# All operations dispatch on operator structure (diagonal, Kronecker,
# block-diagonal, low-rank) for efficient computation, and compose
# seamlessly with `jax.jit`, `jax.vmap`, and `jax.grad`.

# %% [markdown]
# ## References
#
# - Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational
#   inference: a review for statisticians. *JASA*, 112(518), 859-877.
# - Kingma, D. P. & Welling, M. (2014). Auto-encoding variational Bayes.
#   *Proc. ICLR*.
# - Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes
#   for Machine Learning*. MIT Press.
