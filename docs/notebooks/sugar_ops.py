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
x_pts = jnp.linspace(0.0, 5.0, N)

# RBF kernel: k(x, x') = exp(-0.5 * |x - x'|^2 / l^2)
lengthscale = 1.0
diff = x_pts[:, None] - x_pts[None, :]
K = jnp.exp(-0.5 * diff**2 / lengthscale**2)

# Wrap as lineax operator with a bit of jitter for numerical stability
Sigma = K + 1e-6 * jnp.eye(N)
Sigma_op = lx.MatrixLinearOperator(Sigma, lx.positive_semidefinite_tag)
print("Sigma shape:", Sigma.shape)

# %% [markdown]
# ## 2. Quadratic forms
#
# `quadratic_form(Sigma_op, x)` computes $x^\top \Sigma^{-1} x$ via a
# single solve -- the squared Mahalanobis distance.

# %%
key, subkey = jax.random.split(key)
v = jax.random.normal(subkey, (N,))

qf = gaussx.quadratic_form(Sigma_op, v)

# Manual verification: solve then dot
alpha = gaussx.solve(Sigma_op, v)
qf_manual = v @ alpha
print("quadratic_form:", qf)
print("manual v^T Sigma^{-1} v:", qf_manual)
print("match:", jnp.allclose(qf, qf_manual))

# %% [markdown]
# ## 3. Log-probability
#
# `gaussian_log_prob(mu, Sigma_op, x)` evaluates the multivariate normal
# log-pdf:
#
# $$\log \mathcal{N}(x \mid \mu, \Sigma)
# = -\tfrac{1}{2}\bigl(N \log 2\pi + \log|\Sigma|
# + (x - \mu)^\top \Sigma^{-1}(x - \mu)\bigr)$$

# %%
mu = jnp.zeros(N)
lp = gaussx.gaussian_log_prob(mu, Sigma_op, v)

# Manual verification
ld = gaussx.logdet(Sigma_op)
residual = v - mu
qf_val = residual @ gaussx.solve(Sigma_op, residual)
lp_manual = -0.5 * (N * jnp.log(2.0 * jnp.pi) + ld + qf_val)
print("gaussian_log_prob:", lp)
print("manual log-prob:  ", lp_manual)
print("match:", jnp.allclose(lp, lp_manual))

# %% [markdown]
# It also works with structured operators. Here we use a Kronecker product
# to build a 2D covariance:

# %%
A_mat = jnp.eye(3) + 0.5 * jnp.ones((3, 3))
B_mat = jnp.diag(jnp.array([1.0, 2.0]))
A_op = lx.MatrixLinearOperator(A_mat, lx.positive_semidefinite_tag)
B_op = lx.MatrixLinearOperator(B_mat, lx.positive_semidefinite_tag)
kron_op = gaussx.Kronecker(A_op, B_op)

key, subkey = jax.random.split(key)
v_kron = jax.random.normal(subkey, (6,))
lp_kron = gaussx.gaussian_log_prob(jnp.zeros(6), kron_op, v_kron)
print("log-prob (Kronecker):", lp_kron)

# %% [markdown]
# ## 4. Entropy and KL divergence
#
# Entropy of $\mathcal{N}(\mu, \Sigma)$:
#
# $$H = \tfrac{1}{2}\bigl(N(1 + \log 2\pi) + \log|\Sigma|\bigr)$$
#
# KL to the standard normal
# $\text{KL}(\mathcal{N}(m, S) \| \mathcal{N}(0, I))$:
#
# $$\text{KL} = \tfrac{1}{2}\bigl(\text{tr}(S) + m^\top m - N - \log|S|\bigr)$$

# %%
entropy = gaussx.gaussian_entropy(Sigma_op)
print("entropy:", entropy)

m = 0.1 * jnp.ones(N)
kl = gaussx.kl_standard_normal(m, Sigma_op)

# Manual KL verification
tr_S = gaussx.trace(Sigma_op)
mTm = m @ m
ld_S = gaussx.logdet(Sigma_op)
kl_manual = 0.5 * (tr_S + mTm - N - ld_S)
print("KL(q || p):", kl)
print("manual KL: ", kl_manual)
print("match:", jnp.allclose(kl, kl_manual))

# %% [markdown]
# ## 5. GP log-marginal likelihood
#
# For GP regression with observations $y$, prior mean $\mu$, and
# covariance $K_y = K + \sigma^2 I$:
#
# $$\log p(y) = -\tfrac{1}{2}\bigl((y-\mu)^\top K_y^{-1}(y-\mu)
# + \log|K_y| + N\log 2\pi\bigr)$$

# %%
# Generate data from a sine function + noise
noise_var = 0.1
key, subkey = jax.random.split(key)
y = jnp.sin(x_pts) + jax.random.normal(subkey, (N,)) * jnp.sqrt(noise_var)

# Build K_y = K + sigma^2 * I
K_y = K + noise_var * jnp.eye(N)
K_y_op = lx.MatrixLinearOperator(K_y, lx.positive_semidefinite_tag)

lml = gaussx.log_marginal_likelihood(jnp.zeros(N), K_y_op, y)
print("log marginal likelihood:", lml)

# %% [markdown]
# Differentiating the LML with `jax.grad` for hyperparameter optimization:


# %%
def neg_lml(log_lengthscale, log_noise_var, x_pts, y):
    ls = jnp.exp(log_lengthscale)
    nv = jnp.exp(log_noise_var)
    diff = x_pts[:, None] - x_pts[None, :]
    K_i = jnp.exp(-0.5 * diff**2 / ls**2) + nv * jnp.eye(len(y))
    K_op = lx.MatrixLinearOperator(K_i, lx.positive_semidefinite_tag)
    return -gaussx.log_marginal_likelihood(jnp.zeros_like(y), K_op, y)


grad_fn = jax.grad(neg_lml, argnums=(0, 1))
grads = grad_fn(jnp.log(1.0), jnp.log(0.1), x_pts, y)
print("grad w.r.t. log-lengthscale:", grads[0])
print("grad w.r.t. log-noise-var:  ", grads[1])

# %% [markdown]
# ## 6. Schur complement
#
# For GP conditional distributions, the Schur complement gives the
# conditional covariance:
#
# $$K_{X|Z} = K_{XX} - K_{XZ}\, K_{ZZ}^{-1}\, K_{ZX}$$
#
# `schur_complement` returns a `LowRankUpdate` operator that
# preserves the low-rank structure.
#
# The Schur complement is the key identity behind GP conditional
# distributions. Given the joint
# $\begin{pmatrix} f_X \\ f_Z \end{pmatrix} \sim \mathcal{N}\!\left(0,
# \begin{pmatrix} K_{XX} & K_{XZ} \\ K_{ZX} & K_{ZZ}
# \end{pmatrix}\right)$,
# the conditional $f_X \mid f_Z$ has covariance
# $K_{XX} - K_{XZ}\, K_{ZZ}^{-1}\, K_{ZX}$.
# When $M \ll N$, this is a rank-$M$ correction — exactly a
# `LowRankUpdate`.

# %%
M = 8
z_pts = jnp.linspace(0.5, 4.5, M)

# Build kernel blocks
diff_XX = x_pts[:, None] - x_pts[None, :]
K_XX = jnp.exp(-0.5 * diff_XX**2) + 1e-6 * jnp.eye(N)
diff_XZ = x_pts[:, None] - z_pts[None, :]
K_XZ = jnp.exp(-0.5 * diff_XZ**2)
diff_ZZ = z_pts[:, None] - z_pts[None, :]
K_ZZ = jnp.exp(-0.5 * diff_ZZ**2) + 1e-6 * jnp.eye(M)

K_XX_op = lx.MatrixLinearOperator(K_XX, lx.positive_semidefinite_tag)
K_ZZ_op = lx.MatrixLinearOperator(K_ZZ, lx.positive_semidefinite_tag)

schur = gaussx.schur_complement(K_XX_op, K_XZ, K_ZZ_op)
print("type:", type(schur).__name__)

# Verify against manual computation
K_ZZ_inv_KZX = jnp.linalg.solve(K_ZZ, K_XZ.T)
schur_manual = K_XX - K_XZ @ K_ZZ_inv_KZX
schur_mat = schur.as_matrix()
print("max error:", jnp.max(jnp.abs(schur_mat - schur_manual)))

# %% [markdown]
# ## 7. Covariance propagation
#
# `cov_transform(J, Sigma_op)` propagates uncertainty through a linear
# map: $\Sigma' = J\,\Sigma\,J^\top$.

# %%
M_out = 5
key, subkey = jax.random.split(key)
J = jax.random.normal(subkey, (M_out, N))

Sigma_prime_op = gaussx.cov_transform(J, Sigma_op)

# Verify J @ Sigma @ J^T
Sigma_prime_manual = J @ Sigma @ J.T
print("max error:", jnp.max(jnp.abs(Sigma_prime_op.as_matrix() - Sigma_prime_manual)))

# %% [markdown]
# ## 8. Whitening and unwhitening
#
# The reparameterization trick for Gaussians: sample $z \sim \mathcal{N}(0, I)$,
# then compute $x = L\,z$ to get $x \sim \mathcal{N}(0, \Sigma)$ where
# $\Sigma = L L^\top$.
#
# `whiten_covariance(L, S_tilde)` computes $L\,\tilde{S}\,L^\top$, and
# `unwhiten(m_tilde, L)` computes $L\,\tilde{m}$.
#
# The reparameterization trick (Kingma & Welling, 2014) enables
# gradient-based optimization of variational objectives by expressing
# samples $x = L\,z + \mu$ where $z \sim \mathcal{N}(0, I)$, making
# the sampling differentiable w.r.t. the distribution parameters
# $\mu$ and $L$.

# %%
# Cholesky factor of our covariance
L = jnp.linalg.cholesky(Sigma)
L_op = lx.MatrixLinearOperator(L)

# Whitened parameters (identity covariance in whitened space)
S_tilde_op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), jnp.float64))

# Unwhiten covariance: L @ I @ L^T = L @ L^T = Sigma
S_recovered = gaussx.whiten_covariance(L_op, S_tilde_op)
print("max error (covariance):", jnp.max(jnp.abs(S_recovered.as_matrix() - Sigma)))

# Unwhiten a sample: z ~ N(0,I), x = L @ z ~ N(0, Sigma)
key, subkey = jax.random.split(key)
z = jax.random.normal(subkey, (N,))
x_sample = gaussx.unwhiten(z, L_op)
x_manual = L @ z
print("max error (sample):", jnp.max(jnp.abs(x_sample - x_manual)))

# Verify empirically: many samples through L should have covariance ~ Sigma
key, subkey = jax.random.split(key)
Z = jax.random.normal(subkey, (10_000, N))
X = Z @ L.T  # each row is L @ z_i
cov_empirical = jnp.cov(X.T)
print("mean abs error (empirical cov):", jnp.mean(jnp.abs(cov_empirical - Sigma)))

# %% [markdown]
# ## Summary
#
# This notebook demonstrated the main gaussx sugar operations:
#
# | Function | What it computes |
# |----------|-----------------|
# | `quadratic_form` | $x^\top \Sigma^{-1} x$ (Mahalanobis distance squared) |
# | `gaussian_log_prob` | Multivariate normal log-pdf |
# | `gaussian_entropy` | $H[\mathcal{N}(\mu, \Sigma)]$ |
# | `kl_standard_normal` | $\text{KL}(\mathcal{N}(m,S) \| \mathcal{N}(0,I))$ |
# | `log_marginal_likelihood` | GP log evidence (differentiable) |
# | `schur_complement` | $K_{XX} - K_{XZ} K_{ZZ}^{-1} K_{ZX}$ as `LowRankUpdate` |
# | `cov_transform` | $J \Sigma J^\top$ (uncertainty propagation) |
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
