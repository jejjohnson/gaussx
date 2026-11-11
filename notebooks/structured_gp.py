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
# # Structured GP: Kronecker and Low-Rank
#
# This notebook shows where gaussx shines: exploiting structure in
# kernel matrices for Gaussian process inference.
#
# We cover two common patterns:
#
# 1. **Kronecker structure** — GP on a grid (e.g. spatial data on a regular mesh)
# 2. **Low-rank + diagonal** — sparse GP with inducing points (Nystrom approximation)
#
# The two most common forms of kernel structure in GP models are (1) Kronecker products from separable kernels on grids, and (2) low-rank approximations from inducing point methods. Both reduce the $O(N^3)$ cost of dense GP inference. gaussx supports both through its operator type system, dispatching to the optimal algorithm automatically.

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
# ---
# ## Part 1: Kronecker GP on a grid
#
# When input data lies on a grid, the kernel matrix factorizes as a Kronecker product: $\mathbf{K} = \mathbf{K}_1 \otimes \mathbf{K}_2$. This turns $O(N^3)$ operations into $O(N_1^3 + N_2^3)$ where $N = N_1 N_2$.
#
# For a separable kernel on 2D inputs the factorization is:
#
# $$k(\mathbf{s}, \mathbf{s}') = k_1(s_1, s_1') k_2(s_2, s_2')$$
#
# This includes products of any 1D stationary kernels (e.g. RBF, Matern).
# See Saatci (2012) for a comprehensive treatment of Kronecker methods
# in GP models.
#
# ### Setup: 2D grid data


# %%
def rbf_kernel_1d(x, lengthscale, variance):
    """1D RBF kernel matrix."""
    sq_dist = (x[:, None] - x[None, :]) ** 2
    return variance * jnp.exp(-0.5 * sq_dist / lengthscale**2)


# Grid dimensions
n1, n2 = 10, 12
N = n1 * n2  # Total points = 120

x1 = jnp.linspace(0, 5, n1)
x2 = jnp.linspace(0, 5, n2)

print(f"Grid: {n1} x {n2} = {N} points")
print(f"Dense kernel would be {N} x {N} = {N**2:,} entries")
print(f"Kronecker stores {n1}x{n1} + {n2}x{n2} = {n1**2 + n2**2} entries")

# %% [markdown]
# ### Build the Kronecker kernel

# %%
# Per-dimension kernel matrices (lengthscale ~ grid spacing for good conditioning)
# K₁  (N₁, N₁)
kernel_1 = rbf_kernel_1d(x1, lengthscale=1.0, variance=1.0)
# K₂  (N₂, N₂)
kernel_2 = rbf_kernel_1d(x2, lengthscale=1.0, variance=1.0)

# Wrap as PSD lineax operators
kernel_1_op = lx.MatrixLinearOperator(kernel_1, lx.positive_semidefinite_tag)
kernel_2_op = lx.MatrixLinearOperator(kernel_2, lx.positive_semidefinite_tag)

# Kronecker product: K = K₁ ⊗ K₂  (N, N)
kernel_kron = gaussx.Kronecker(kernel_1_op, kernel_2_op)

# Add noise: (K₁ ⊗ K₂) + σ_ε² I
# For now, materialize noise addition — a full structured version
# would use a LowRankUpdate or custom operator.
noise_var = 0.01

print(f"Kronecker operator size: {kernel_kron.in_size()}")

# %% [markdown]
# ### Kronecker logdet
#
# The key identity: $\log \det (\mathbf{K}_1 \otimes \mathbf{K}_2) = N_2 \log \det \mathbf{K}_1 + N_1 \log \det \mathbf{K}_2$
#
# gaussx computes this automatically — no need to form the $N \times N$ matrix.

# %%
# Structured logdet: O(n1^3 + n2^3)
ld_structured = gaussx.logdet(kernel_kron)
print(f"logdet (structured): {ld_structured:.6f}")

# Verify against dense: O(N^3)
ld_dense = jnp.linalg.slogdet(kernel_kron.as_matrix())[1]
print(f"logdet (dense):      {ld_dense:.6f}")
print(f"Match: {jnp.allclose(ld_structured, ld_dense)}")

# %% [markdown]
# ### Kronecker solve and Cholesky
#
# Solve and Cholesky also decompose per-factor.

# %%
# Generate observations on the grid
key = jax.random.PRNGKey(0)
y = jax.random.normal(key, (N,))

# Structured solve of K α = y
# α = K⁻¹ y  (N,)
alpha = gaussx.solve(kernel_kron, y)
print(f"solve residual: {jnp.max(jnp.abs(kernel_kron.mv(alpha) - y)):.2e}")

# Structured Cholesky: returns Kronecker(chol(K₁), chol(K₂))
chol_kron = gaussx.cholesky(kernel_kron)
print(f"cholesky type: {type(chol_kron).__name__}")
print(f"L factors: {[type(f).__name__ for f in chol_kron.operators]}")

# Verify: L Lᵀ = K
recon_err = jnp.max(
    jnp.abs(chol_kron.as_matrix() @ chol_kron.as_matrix().T - kernel_kron.as_matrix())
)
print(f"Cholesky reconstruction error: {recon_err:.2e}")

# %% [markdown]
# ---
# ## Part 2: Low-rank GP (Nystrom / inducing points)
#
# For non-grid data, a common approximation is $\mathbf{K}_{XX} \approx \mathbf{K}_{X\mathcal{I}} \mathbf{K}_{\mathcal{I}\mathcal{I}}^{-1} \mathbf{K}_{\mathcal{I}X}$. The resulting covariance is $\mathrm{diag}(\sigma_\varepsilon^2) + \mathbf{U} \mathbf{U}^\top$ — a low-rank update.
#
# The Woodbury identity gives an efficient solve:
#
# $$(\sigma_\varepsilon^2 \mathbf{I} + \mathbf{U} \mathbf{U}^\top)^{-1} = \sigma_\varepsilon^{-2} \mathbf{I} - \sigma_\varepsilon^{-2} \mathbf{U} (\mathbf{I} + \sigma_\varepsilon^{-2} \mathbf{U}^\top \mathbf{U})^{-1} \mathbf{U}^\top \sigma_\varepsilon^{-2}$$
#
# and the matrix determinant lemma gives an efficient logdet:
#
# $$\log \det (\sigma_\varepsilon^2 \mathbf{I} + \mathbf{U} \mathbf{U}^\top) = N \log \sigma_\varepsilon^2 + \log \det (\mathbf{I} + \sigma_\varepsilon^{-2} \mathbf{U}^\top \mathbf{U})$$
#
# Both reduce the cost from $O(N^3)$ to $O(N N_{\mathcal{I}}^2 + N_{\mathcal{I}}^3)$, with $\mathbf{U} \in \mathbb{R}^{N \times N_{\mathcal{I}}}$ and $N_{\mathcal{I}}$ the number of inducing inputs.
#
# ### Setup: random 1D data with inducing points

# %%
n_data = 200
n_inducing = 15

key = jax.random.PRNGKey(7)
k1, k2 = jax.random.split(key)

x_data = jax.random.uniform(k1, (n_data,), minval=-3.0, maxval=3.0)
x_inducing = jnp.linspace(-2.5, 2.5, n_inducing)

print(f"Data points: {n_data}")
print(f"Inducing points: {n_inducing}")
print(f"Dense kernel: {n_data}x{n_data} = {n_data**2:,} entries")
print(
    f"Low-rank: {n_data}x{n_inducing} + {n_inducing}x{n_inducing} "
    f"= {n_data * n_inducing + n_inducing**2:,} entries"
)

# %% [markdown]
# ### Build the low-rank approximation

# %%
ls, var = 1.0, 1.5
noise = 0.1


def rbf_kernel_2d(x1, x2, ls, var):
    sq_dist = (x1[:, None] - x2[None, :]) ** 2
    return var * jnp.exp(-0.5 * sq_dist / ls**2)


# K_XI: data-to-inducing kernel  (N, N_I)
kernel_data_inducing = rbf_kernel_2d(x_data, x_inducing, ls, var)

# K_II: inducing-to-inducing kernel  (N_I, N_I)
kernel_inducing = rbf_kernel_2d(x_inducing, x_inducing, ls, var)

# Nystrom factor: U = K_XI L_IIᵀ⁻¹ gives K ≈ U Uᵀ, with L_II L_IIᵀ = K_II
chol_inducing = jnp.linalg.cholesky(kernel_inducing + 1e-6 * jnp.eye(n_inducing))
nystrom_factor = jax.scipy.linalg.solve_triangular(
    chol_inducing, kernel_data_inducing.T, lower=True
).T

print(f"U shape: {nystrom_factor.shape}")  # (200, 15)

# %% [markdown]
# ### Construct gaussx LowRankUpdate operator
#
# $\boldsymbol{\Sigma}_y = \sigma_\varepsilon^2 \mathbf{I} + \mathbf{U} \mathbf{U}^\top$

# %%
diag_noise = noise * jnp.ones(n_data)
# Σ_y = σ_ε² I + U Uᵀ  (N, N)
sigma = gaussx.low_rank_plus_diag(diag_noise, nystrom_factor)

print(f"LowRankUpdate rank: {sigma.rank}")
print(f"Operator size: {sigma.in_size()}")
print(f"gaussx.is_low_rank: {gaussx.is_low_rank(sigma)}")

# %% [markdown]
# ### Solve and logdet via Woodbury
#
# gaussx automatically uses the Woodbury identity for solve
# and the matrix determinant lemma for logdet — O(nk^2 + k^3)
# instead of O(n^3).

# %%
# Generate observations
y_data = jnp.sin(2 * x_data) + noise * jax.random.normal(k2, (n_data,))

# Woodbury solve: α = (σ_ε² I + U Uᵀ)⁻¹ y
alpha = gaussx.solve(sigma, y_data)
print(f"solve residual: {jnp.max(jnp.abs(sigma.mv(alpha) - y_data)):.2e}")

# Matrix determinant lemma logdet
ld = gaussx.logdet(sigma)
ld_dense = jnp.linalg.slogdet(sigma.as_matrix())[1]
print(f"logdet (Woodbury):  {ld:.6f}")
print(f"logdet (dense):     {ld_dense:.6f}")
print(f"Match: {jnp.allclose(ld, ld_dense, rtol=1e-4)}")

# %% [markdown]
# ### Log-marginal likelihood


# %%
def sparse_gp_lml(y, sigma_op):
    """Sparse GP log-marginal likelihood."""
    # α = Σ_y⁻¹ y
    alpha = gaussx.solve(sigma_op, y)
    # −½ yᵀ α
    data_fit = -0.5 * jnp.dot(y, alpha)
    # −½ log det Σ_y
    complexity = -0.5 * gaussx.logdet(sigma_op)
    # −(N/2) log 2π
    constant = -0.5 * len(y) * jnp.log(2 * jnp.pi)
    return data_fit + complexity + constant


lml = sparse_gp_lml(y_data, sigma)
print(f"Sparse GP LML: {lml:.4f}")

# %% [markdown]
# ### Visualize the sparse GP fit

# %%
# Predict on a test grid using the sparse approximation
x_plot = jnp.linspace(-3.5, 3.5, 300)
# K_*I  (N_plot, N_I)
kernel_plot_inducing = rbf_kernel_2d(x_plot, x_inducing, ls, var)
inducing_op = lx.MatrixLinearOperator(
    kernel_inducing + 1e-6 * jnp.eye(n_inducing), lx.positive_semidefinite_tag
)
# Nystrom prediction weights
# α = Σ_y⁻¹ y  (N,)
alpha_sparse = gaussx.solve(sigma, y_data)
# Project through inducing points
w = nystrom_factor.T @ alpha_sparse  # (k,)
# μ_* = K_*I (K_II + jitter)⁻¹ K_IX α  (N_plot,)
y_plot = kernel_plot_inducing @ jnp.linalg.solve(
    kernel_inducing + 1e-6 * jnp.eye(n_inducing),
    kernel_data_inducing.T @ alpha_sparse,
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_plot, jnp.sin(2 * x_plot), "k--", lw=1.5, label="True", zorder=4)
ax.scatter(
    x_data,
    y_data,
    s=30,
    c="C0",
    alpha=0.4,
    edgecolors="k",
    linewidths=0.5,
    label="Data",
    zorder=5,
)
ax.plot(x_plot, y_plot, "C1-", lw=2, label="Sparse GP mean", zorder=3)
ax.axvline(x_inducing[0], color="gray", lw=0.5, alpha=0.3)
for xi in x_inducing[1:]:
    ax.axvline(xi, color="gray", lw=0.5, alpha=0.3)
ax.scatter(
    x_inducing,
    jnp.zeros_like(x_inducing) - 1.2,
    marker="^",
    s=40,
    c="C2",
    zorder=5,
    label=f"Inducing pts ({n_inducing})",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Sparse GP: {n_data} data, {n_inducing} inducing points")
ax.legend(fontsize=9)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | Structure | When to use | Solve cost | Logdet cost |
# |-----------|------------|------------|-------------|
# | `Kronecker(K1, K2)` | Grid data | O(n1^3 + n2^3) | O(n1^3 + n2^3) |
# | `low_rank_plus_diag` | Inducing points | O(nk^2 + k^3) | O(nk^2 + k^3) |
# | Dense | Unstructured | O(n^3) | O(n^3) |
#
# gaussx dispatches to the right algorithm automatically based
# on the operator type — write the math once, get efficient
# computation for free.

# %% [markdown]
# ## References
#
# - Quinonero-Candela, J. & Rasmussen, C. E. (2005). A unifying view of sparse
#   approximate Gaussian process regression. *JMLR*, 6, 1939--1959.
# - Saatci, Y. (2012). *Scalable Inference for Structured Gaussian Process
#   Models*. PhD thesis, University of Cambridge.
# - Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for
#   Machine Learning*. MIT Press.
