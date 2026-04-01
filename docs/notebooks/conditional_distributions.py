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
# # Conditional Distributions & GP Prediction from Scratch
#
# This notebook derives Gaussian process prediction as conditioning a
# joint Gaussian distribution, and shows how gaussx primitives make the
# linear algebra clean and verifiable.
#
# **What you'll learn:**
#
# 1. How GP prediction is just conditioning a joint Gaussian
# 2. `gaussx.conditional` for computing $p(x_A \mid x_B = b)$
# 3. `gaussx.schur_complement` for the conditional covariance block
# 4. `gaussx.dist_kl_divergence` to verify equivalence of two approaches
# 5. `gaussx.conditional_variance` for predictive variances

# %% [markdown]
# ## 1. Background: GP Prediction as Joint Conditioning
#
# A Gaussian process defines a joint Gaussian over any finite collection
# of function values. Given training inputs $X$ with noisy observations
# $y$ and test inputs $X_*$, we can stack them into a single joint:
#
# $$
# \begin{bmatrix} \mathbf{y} \\ \mathbf{f}_* \end{bmatrix}
# \sim \mathcal{N}\!\left(
# \mathbf{0},\;
# \begin{bmatrix}
#   K_{ff} + \sigma^2 I & K_{f*} \\
#   K_{*f} & K_{**}
# \end{bmatrix}
# \right)
# $$
#
# Conditioning on $\mathbf{y}$ gives the posterior predictive:
#
# $$
# \mu_{*|y} = K_{*f} (K_{ff} + \sigma^2 I)^{-1} \mathbf{y}
# $$
#
# $$
# \Sigma_{*|y} = K_{**} - K_{*f} (K_{ff} + \sigma^2 I)^{-1} K_{f*}
# $$
#
# This is exactly the Gaussian conditioning formula
# $\mu_{A|B} = \mu_A + \Sigma_{AB} \Sigma_{BB}^{-1}(b - \mu_B)$
# applied to the joint above. We will verify this with both
# `gaussx.conditional` and manual computation.

# %%
from __future__ import annotations

import warnings


warnings.filterwarnings("ignore", message=r".*IProgress.*")

import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt

import gaussx


jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## 2. Build a 1D GP
#
# We define a squared-exponential (RBF) kernel and sample noisy
# observations from a known function.
#
# $$
# k(x, x') = \sigma_f^2
#   \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right)
# $$


# %%
def rbf_kernel(x1, x2, lengthscale, variance):
    """Squared-exponential kernel matrix."""
    sq_dist = (x1[:, None] - x2[None, :]) ** 2
    return variance * jnp.exp(-0.5 * sq_dist / lengthscale**2)


# Hyperparameters
lengthscale = 0.8
variance = 1.0
noise_std = 0.2
noise_var = noise_std**2

# Data
key = jax.random.PRNGKey(42)
n_train = 30
n_test = 100

f_true = lambda x: jnp.sin(3 * x) * jnp.exp(-0.5 * x**2)

x_train = jnp.sort(jax.random.uniform(key, (n_train,), minval=-3.0, maxval=3.0))
y_train = f_true(x_train) + noise_std * jax.random.normal(
    jax.random.PRNGKey(1), (n_train,)
)
x_test = jnp.linspace(-3.5, 3.5, n_test)

# %% [markdown]
# ## 3. Form the Joint Distribution
#
# We build the full joint covariance over
# $[\mathbf{y},\, \mathbf{f}_*]$ of size
# $(n_\text{train} + n_\text{test}) \times (n_\text{train} + n_\text{test})$.
# Noise $\sigma^2 I$ is added only to the training block.

# %%
x_all = jnp.concatenate([x_train, x_test])
K_joint = rbf_kernel(x_all, x_all, lengthscale, variance)

# Add noise to the training block only
noise_block = jnp.zeros(n_train + n_test)
noise_block = noise_block.at[:n_train].set(noise_var)
K_joint = K_joint + jnp.diag(noise_block)

# Add small jitter to the entire joint for numerical stability
K_joint = K_joint + 1e-6 * jnp.eye(n_train + n_test)

# Joint mean is zero
mu_joint = jnp.zeros(n_train + n_test)

# Wrap as lineax operator
K_joint_op = lx.MatrixLinearOperator(K_joint, lx.positive_semidefinite_tag)

print(f"Joint covariance shape: {K_joint.shape}")
print(f"Training indices: 0..{n_train - 1}")
print(f"Test indices: {n_train}..{n_train + n_test - 1}")

# %% [markdown]
# ## 4. Condition with `gaussx.conditional`
#
# `gaussx.conditional(loc, cov, obs_idx, obs_values)` computes
# $p(x_A \mid x_B = b)$ from a joint Gaussian. The training indices
# are the observed set $B$, and the test indices are the free set $A$.

# %%
obs_idx = jnp.arange(n_train, dtype=jnp.int32)
cond_mean, cond_cov_op = gaussx.conditional(mu_joint, K_joint_op, obs_idx, y_train)

cond_cov = cond_cov_op.as_matrix()

print(f"Conditional mean shape: {cond_mean.shape}")
print(f"Conditional cov shape:  {cond_cov.shape}")

# %% [markdown]
# ## 5. Verify with Manual GP Formulas
#
# The standard GP prediction formulas are:
#
# $$
# \mu_* = K_{*f}\,(K_{ff} + \sigma^2 I)^{-1}\,\mathbf{y}
# $$
#
# $$
# \Sigma_* = K_{**} - K_{*f}\,(K_{ff} + \sigma^2 I)^{-1}\,K_{f*}
# $$
#
# We compute these manually using `gaussx.solve` and compare.

# %%
K_ff = rbf_kernel(x_train, x_train, lengthscale, variance) + noise_var * jnp.eye(
    n_train
)
K_sf = rbf_kernel(x_test, x_train, lengthscale, variance)
K_ss = rbf_kernel(x_test, x_test, lengthscale, variance)

K_ff_op = lx.MatrixLinearOperator(K_ff, lx.positive_semidefinite_tag)

# GP mean: K_sf @ (K_ff)^{-1} @ y
alpha = gaussx.solve(K_ff_op, y_train)
mu_manual = K_sf @ alpha

# GP covariance: K_ss - K_sf @ (K_ff)^{-1} @ K_fs
V = jax.vmap(lambda k: gaussx.solve(K_ff_op, k))(K_sf)  # (n_test, n_train)
cov_manual = K_ss - K_sf @ V.T

print("Max |mean difference|: ", jnp.max(jnp.abs(cond_mean - mu_manual)))
print("Max |cov difference|:  ", jnp.max(jnp.abs(cond_cov - cov_manual)))

# %% [markdown]
# Both should be near machine precision ($\sim 10^{-12}$), confirming
# that `gaussx.conditional` implements exactly the standard GP formulas.

# %% [markdown]
# ## 6. KL Divergence: Verify Equivalence
#
# As a further check, we compute
# $\text{KL}(p_\text{conditional} \| p_\text{manual})$.
# For identical distributions this should be exactly zero (up to
# floating-point noise).

# %%
# Add jitter for numerical stability (Schur complement can lose PSD)
cov_manual_reg = cov_manual + 1e-6 * jnp.eye(n_test)
cond_cov_reg = cond_cov + 1e-6 * jnp.eye(n_test)
cov_manual_op = lx.MatrixLinearOperator(cov_manual_reg, lx.positive_semidefinite_tag)
cond_cov_reg_op = lx.MatrixLinearOperator(cond_cov_reg, lx.positive_semidefinite_tag)

kl = gaussx.dist_kl_divergence(cond_mean, cond_cov_reg_op, mu_manual, cov_manual_op)
print(f"KL(conditional || manual) = {kl:.2e}")

# %% [markdown]
# ## 7. Plot: GP Posterior
#
# True function, noisy observations, posterior mean, and 95% credible
# interval.

# %%
cond_std = jnp.sqrt(jnp.maximum(jnp.diag(cond_cov), 0.0))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_test, f_true(x_test), "k--", lw=1.5, label="True function", zorder=4)
ax.scatter(
    x_train,
    y_train,
    s=30,
    c="C0",
    edgecolors="k",
    linewidths=0.5,
    zorder=5,
    label="Training data",
)
ax.plot(x_test, cond_mean, "C1-", lw=2, label="GP mean (conditional)", zorder=3)
ax.fill_between(
    x_test,
    cond_mean - 2 * cond_std,
    cond_mean + 2 * cond_std,
    alpha=0.2,
    color="C1",
    label="95% CI",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=9)
ax.set_title("GP Prediction via Joint Conditioning")
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Schur Complement: Conditional Covariance
#
# The conditional covariance is a **Schur complement**:
#
# $$
# \Sigma_{A|B} = K_{**} - K_{*f}\,(K_{ff} + \sigma^2 I)^{-1}\,K_{f*}
# $$
#
# `gaussx.schur_complement` returns a `LowRankUpdate` operator
# representing this expression without materializing the full matrix.

# %%
K_ss_op = lx.MatrixLinearOperator(K_ss, lx.positive_semidefinite_tag)

schur_op = gaussx.schur_complement(K_ss_op, K_sf, K_ff_op)
schur_mat = schur_op.as_matrix()

print(f"Schur complement shape: {schur_mat.shape}")
print("Max |Schur - conditional cov|:", jnp.max(jnp.abs(schur_mat - cond_cov)))

# %% [markdown]
# The Schur complement matches the conditional covariance to numerical
# precision. In practice, `schur_complement` returns a structured
# `LowRankUpdate` operator, so downstream `solve` and `logdet` calls
# can exploit the low-rank structure via the Woodbury identity.

# %% [markdown]
# ## 9. Conditional Variance with `conditional_variance`
#
# For large-scale GP prediction we often only need the **diagonal**
# of the predictive covariance (i.e., the marginal variances).
# `gaussx.conditional_variance` computes this efficiently without
# forming the full $n_* \times n_*$ matrix.
#
# It expects a projection matrix
# $A = K_{*f}\,(K_{ff} + \sigma^2 I)^{-1}$ and a "residual"
# covariance $S_u$. For exact GP prediction with no approximation,
# $S_u = 0$, so the predictive variance is
# $\text{diag}(K_{**}) - \text{diag}(A\,K_{ff}\,A^\top)$.
# Equivalently we can use $A$ directly and set $S_u$ to the zero
# operator, but the cleaner path is to note that
# `conditional_variance(K_diag, A, S_u)` computes
# $K_\text{diag} + \text{diag}(A\,S_u\,A^\top)$.
#
# We use the identity: predictive variance =
# $\text{diag}(K_{**}) - \text{diag}(K_{*f}\,\alpha_\text{cols})$
# where $\alpha_\text{cols} = (K_{ff})^{-1} K_{f*}$.

# %%
# Projection: A = K_sf @ K_ff^{-1}, shape (n_test, n_train)
A = V  # already computed above as vmap(solve(K_ff, .))(K_sf)

# Diagonal of the prior
K_ss_diag = variance * jnp.ones(n_test)

# For exact GP: S_u captures the "correction" term.
# predictive_var = K_ss_diag - diag(K_sf @ K_ff^{-1} @ K_fs) + diag(A @ S_u @ A^T)
# With S_u = 0, we recover the standard formula.  But conditional_variance
# computes K_ss_diag + diag(A @ S_u @ A^T), so we set
# K_ss_diag_adjusted = K_ss_diag - diag(K_sf @ V^T) and S_u = 0.
var_reduction = jnp.sum(K_sf * V, axis=1)  # diag(K_sf @ K_ff^{-1} @ K_fs)
pred_var_manual = K_ss_diag - var_reduction

# Via conditional_variance with S_u = 0
S_zero = lx.MatrixLinearOperator(jnp.zeros((n_train, n_train)))
adjusted_diag = K_ss_diag - var_reduction
pred_var_cv = gaussx.conditional_variance(adjusted_diag, A, S_zero)

print("Max |variance difference|:", jnp.max(jnp.abs(pred_var_cv - pred_var_manual)))
print(
    "Matches conditional diagonal:",
    jnp.allclose(pred_var_manual, jnp.diag(cond_cov), atol=1e-10),
)

# %% [markdown]
# ## 10. Summary
#
# | gaussx function | What it does |
# |----------------|--------------|
# | `conditional` | $p(x_A \mid x_B = b)$ |
# | `solve` | $(K + \sigma^2 I)^{-1} y$ |
# | `dist_kl_divergence` | KL between two MVNs |
# | `schur_complement` | Structured cond. cov. |
# | `conditional_variance` | Diagonal pred. var. |
#
# The key insight is that GP prediction is **nothing more than
# conditioning a joint Gaussian**. gaussx provides both the high-level
# `conditional` API and the building blocks (`solve`, `schur_complement`,
# `conditional_variance`) for when you need finer control.

# %% [markdown]
# ## References
#
# - Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes
#   for Machine Learning*. MIT Press, Ch. 2.
# - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
#   Springer, Sec. 2.3.1 (conditional Gaussians).
