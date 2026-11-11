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
# # GP on a 2D Grid with Kronecker Structure
#
# When data lives on a regular grid, the kernel matrix factorizes as $\mathbf{K} = \mathbf{K}_1 \otimes \mathbf{K}_2$. This enables GP inference on grids with thousands of points that would be intractable with a dense kernel.
#
# This example fits a 2D GP to noisy observations on a grid using
# Kronecker structure for solve, logdet, and prediction.

# %% [markdown]
# ## Background
#
# When inputs lie on a Cartesian product grid $\mathcal{S}_1 \times \mathcal{S}_2$ with $\mathcal{S}_1 = \{s_{1,1}, \ldots, s_{1,N_1}\}$ and $\mathcal{S}_2 = \{s_{2,1}, \ldots, s_{2,N_2}\}$ and the kernel is separable,
#
# $$k(\mathbf{s}, \mathbf{s}') = k_1(s_1, s_1') k_2(s_2, s_2'),$$
#
# the kernel matrix factorizes as a Kronecker product $\mathbf{K} = \mathbf{K}_1 \otimes \mathbf{K}_2$ where $\mathbf{K}_1 \in \mathbb{R}^{N_1 \times N_1}$ and $\mathbf{K}_2 \in \mathbb{R}^{N_2 \times N_2}$. With $N = N_1 N_2$, this reduces:
#
# - **Storage** from $O(N^2)$ to $O(N_1^2 + N_2^2)$.
# - **Solve** from $O(N^3)$ to $O(N_1^3 + N_2^3)$.
# - **Log-determinant** via $\log \det (\mathbf{K}_1 \otimes \mathbf{K}_2) = N_2 \log \det \mathbf{K}_1 + N_1 \log \det \mathbf{K}_2$.
#
# These savings make GP inference feasible on grids with thousands of
# points that would be intractable with a dense kernel matrix.
# See Saatci (2012) for a comprehensive treatment and Gilboa et al.
# (2015) for extensions to higher-dimensional grids.

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
# ## Generate 2D grid data

# %%
# Grid
nx, ny = 15, 15
N = nx * ny
x = jnp.linspace(0, 4, nx)
y = jnp.linspace(0, 4, ny)
xx, yy = jnp.meshgrid(x, y, indexing="ij")


# True function: a smooth 2D surface
def f_true(x, y):
    return jnp.sin(x) * jnp.cos(1.5 * y)


z_true = f_true(xx, yy)

# Noisy observations
noise_std = 0.15
key = jax.random.PRNGKey(7)
z_obs = z_true + noise_std * jax.random.normal(key, z_true.shape)

print(f"Grid: {nx}x{ny} = {N} points")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].pcolormesh(xx, yy, z_true, cmap="RdBu_r", shading="auto")
axes[0].set_title("True function")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(xx, yy, z_obs, cmap="RdBu_r", shading="auto")
axes[1].set_title(f"Noisy observations ($\\sigma_\\varepsilon$={noise_std})")
plt.colorbar(im1, ax=axes[1])

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Build Kronecker kernel
#
# Separable RBF: $k(\mathbf{s}, \mathbf{s}') = k_1(s_1, s_1') k_2(s_2, s_2')$
#
# Separability holds for any product of 1D stationary kernels (RBF, Matern, periodic). For non-separable kernels, sums of Kronecker products $\sum_r \mathbf{K}_1^{(r)} \otimes \mathbf{K}_2^{(r)}$ can be used as approximations.


# %%
def rbf_1d(coords, lengthscale, variance):
    sq_dist = (coords[:, None] - coords[None, :]) ** 2
    return variance * jnp.exp(-0.5 * sq_dist / lengthscale**2)


ls_x, ls_y = 1.0, 1.0
var_x, var_y = 1.0, 1.0
noise_var = noise_std**2

# K₁ = k₁(S₁, S₁)  (N₁, N₁)
kernel_x = rbf_1d(x, ls_x, var_x)
# K₂ = k₂(S₂, S₂)  (N₂, N₂)
kernel_y = rbf_1d(y, ls_y, var_y)

kernel_x_op = lx.MatrixLinearOperator(kernel_x, lx.positive_semidefinite_tag)
kernel_y_op = lx.MatrixLinearOperator(kernel_y, lx.positive_semidefinite_tag)

# Full kernel: K₁ ⊗ K₂ + noise * I
# K = K₁ ⊗ K₂  (N, N)
kernel_kron = gaussx.Kronecker(kernel_x_op, kernel_y_op)

print(f"Kronecker kernel: {kernel_kron.in_size()}x{kernel_kron.out_size()}")
print(f"Storage: {nx}x{nx} + {ny}x{ny} = {nx**2 + ny**2}")
print(f"Dense would be: {N}x{N} = {N**2:,}")

# %% [markdown]
# ## Add noise and solve
#
# We need $(\mathbf{K}_1 \otimes \mathbf{K}_2 + \sigma_\varepsilon^2 \mathbf{I}) \boldsymbol{\alpha} = \mathbf{y}$, with $\mathbf{y} \in \mathbb{R}^N$ the flattened observations. For now we add noise to the dense matrix; a full implementation would keep this structured.

# %%
# Σ_y = K + σ_ε² I  (N, N)
noisy_kernel = kernel_kron.as_matrix() + noise_var * jnp.eye(N)
op = lx.MatrixLinearOperator(noisy_kernel, lx.positive_semidefinite_tag)

# Flatten observations for solve
y_vec = z_obs.ravel()
# α = Σ_y⁻¹ y  (N,)
alpha = gaussx.solve(op, y_vec)

# Posterior mean on the same grid
# μ = K α  (N,) reshaped to (N₁, N₂)
z_pred = (kernel_kron.as_matrix() @ alpha).reshape(nx, ny)

print(f"Solve residual: {jnp.max(jnp.abs(op.mv(alpha) - y_vec)):.2e}")

# %% [markdown]
# ## Log-marginal likelihood
#
# The log-marginal likelihood (LML) for GP regression is
#
# $$\log p(\mathbf{y}) = -\frac{1}{2} \mathbf{y}^\top (\mathbf{K} + \sigma_\varepsilon^2 \mathbf{I})^{-1} \mathbf{y} - \frac{1}{2} \log \det (\mathbf{K} + \sigma_\varepsilon^2 \mathbf{I}) - \frac{N}{2} \log 2\pi.$$
#
# Note that the noise term $\sigma_\varepsilon^2 \mathbf{I}$ breaks the pure Kronecker structure of $\mathbf{K}_1 \otimes \mathbf{K}_2$. However, eigendecomposition-based approaches (Saatci, 2012) or Krylov methods can handle $\mathbf{K}_1 \otimes \mathbf{K}_2 + \sigma_\varepsilon^2 \mathbf{I}$ efficiently by working with the per-factor eigenvalues.


# %%
def lml(y, op):
    # α = Σ_y⁻¹ y
    alpha = gaussx.solve(op, y)
    n = len(y)
    # log det Σ_y
    ld = gaussx.logdet(op)
    # −½ yᵀ α − ½ log det Σ_y − (N/2) log 2π
    return -0.5 * jnp.dot(y, alpha) - 0.5 * ld - 0.5 * n * jnp.log(2 * jnp.pi)


print(f"Log-marginal likelihood: {lml(y_vec, op):.2f}")

# %% [markdown]
# ## Visualize results

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].pcolormesh(xx, yy, z_obs, cmap="RdBu_r", shading="auto")
axes[0].set_title("Observations")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(xx, yy, z_pred, cmap="RdBu_r", shading="auto")
axes[1].set_title("GP posterior mean")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(xx, yy, jnp.abs(z_true - z_pred), cmap="Reds", shading="auto")
axes[2].set_title("|True - Predicted|")
plt.colorbar(im2, ax=axes[2])

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Kronecker-only operations
#
# Even though we fell back to dense for the noisy solve,
# the Kronecker structure enables cheap operations on the
# kernel itself.

# %%
print("logdet(Kx kron Ky):")
print(f"  Structured: {gaussx.logdet(kernel_kron):.4f}")
print(f"  Dense:      {jnp.linalg.slogdet(kernel_kron.as_matrix())[1]:.4f}")

print("\ntrace(Kx kron Ky):")
print(f"  Structured: {gaussx.trace(kernel_kron):.4f}")
print(f"  Dense:      {jnp.trace(kernel_kron.as_matrix()):.4f}")

# L L^T = K₁ ⊗ K₂
L_kron = gaussx.cholesky(kernel_kron)
print(f"\ncholesky type: {type(L_kron).__name__}")

# %% [markdown]
# ## References
#
# - Gilboa, E., Saatci, Y., & Cunningham, J. P. (2015). Scaling
#   multidimensional inference for structured Gaussian processes.
#   *IEEE TPAMI*, 37(2), 424--436.
# - Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes
#   for Machine Learning*. MIT Press.
# - Saatci, Y. (2012). *Scalable Inference for Structured Gaussian
#   Process Models*. PhD thesis, University of Cambridge.
