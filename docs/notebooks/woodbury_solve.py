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
# # Woodbury Solve Step by Step
#
# The Woodbury matrix identity turns an $N \times N$ solve into a cheap $k \times k$ solve when the matrix has low-rank structure:
#
# $$(\mathbf{A}_0 + \mathbf{U} \mathbf{D} \mathbf{V}^\top)^{-1} = \mathbf{A}_0^{-1} - \mathbf{A}_0^{-1} \mathbf{U} \mathbf{C}^{-1} \mathbf{V}^\top \mathbf{A}_0^{-1}$$
#
# where $\mathbf{C} = \mathbf{D}^{-1} + \mathbf{V}^\top \mathbf{A}_0^{-1} \mathbf{U} \in \mathbb{R}^{k \times k}$ is the **capacitance matrix**, $\mathbf{U}, \mathbf{V} \in \mathbb{R}^{N \times k}$ and $\mathbf{D} \in \mathbb{R}^{k \times k}$.
#
# This notebook walks through each step and shows that `gaussx.solve`
# does it automatically.

# %% [markdown]
# ## Background
#
# The Woodbury identity (also known as the matrix inversion lemma or
# Sherman-Morrison-Woodbury formula) is one of the most important identities
# in computational linear algebra. It appears in:
#
# - **Sparse GP inference** — the Nystrom approximation yields a low-rank + diagonal covariance, and the Woodbury identity turns the $N \times N$ inversion into a $k \times k$ problem where $k \ll N$.
# - **Kalman filtering** — sequential rank-1 updates to the state covariance
#   are applied via the matrix inversion lemma at each time step.
# - **Online learning** — rank-1 covariance updates (e.g. recursive least squares)
#   use the Sherman-Morrison special case.
# - **Ridge regression with feature space formulations** — the kernel trick relates the $N \times N$ and $D_x \times D_x$ solutions through Woodbury.
#
# The identity was first published by Woodbury (1950) and independently by
# Sherman & Morrison (1950) for the rank-1 case.

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
# ## Setup
#
# We construct $\mathbf{A} = \mathrm{diag}(\mathbf{a}) + \mathbf{U} \mathbf{U}^\top$ where $\mathbf{a} \in \mathbb{R}^{100}$ and $\mathbf{U} \in \mathbb{R}^{100 \times 3}$ (rank 3).

# %%
key = jax.random.PRNGKey(0)
key_diagonal, key_factor, key_rhs = jax.random.split(key, 3)

size, rank = 100, 3
# positive diagonal
base_diagonal = jnp.abs(jax.random.normal(key_diagonal, (size,))) + 0.5
low_rank_factor = jax.random.normal(key_factor, (size, rank)) * 0.5
rhs = jax.random.normal(key_rhs, (size,))

# Build operator
# A = diag(a) + U Uᵀ
operator = gaussx.low_rank_plus_diag(base_diagonal, low_rank_factor)
print(f"Operator: diag({size}) + rank-{rank} update")
print(f"  Full matrix size: {size}x{size} = {size**2:,} entries")
print(f"  Low-rank storage: {size} + {size}x{rank} = {size + size * rank:,} entries")

# %% [markdown]
# ## Step-by-step Woodbury

# %%
# Step 1: A₀⁻¹ b  (cheap — diagonal solve)
base_inv_rhs = rhs / base_diagonal
print("Step 1: A_0^{-1} b  — diagonal solve, O(N)")

# Step 2: A₀⁻¹ U  (k diagonal solves)
base_inv_factor = low_rank_factor / base_diagonal[:, None]
print(f"Step 2: A_0^{{-1}} U  — {rank} diagonal solves, O(Nk)")

# Step 3: Capacitance matrix C = I + Uᵀ A₀⁻¹ U  (k x k)
capacitance = jnp.eye(rank) + low_rank_factor.T @ base_inv_factor
print(f"Step 3: C = I + U^T A_0^{{-1}} U  — {rank}x{rank} matrix")

# Step 4: Solve C z = Uᵀ A₀⁻¹ b  (k x k dense solve)
capacitance_solution = jnp.linalg.solve(capacitance, low_rank_factor.T @ base_inv_rhs)
print(f"Step 4: C^{{-1}} (V^T A_0^{{-1}} b)  — {rank}x{rank} solve")

# Step 5: x = A₀⁻¹ b − A₀⁻¹ U z
solution_woodbury = base_inv_rhs - base_inv_factor @ capacitance_solution
print("Step 5: x = A_0^{-1} b - A_0^{-1} U z  — final answer")

# %% [markdown]
# ## Verify against gaussx.solve

# %%
solution_gaussx = gaussx.solve(operator, rhs)
solution_dense = jnp.linalg.solve(operator.as_matrix(), rhs)

woodbury_error = jnp.max(jnp.abs(solution_woodbury - solution_dense))
print(f"Woodbury vs dense:  max|diff| = {woodbury_error:.2e}")
gaussx_error = jnp.max(jnp.abs(solution_gaussx - solution_dense))
print(f"gaussx  vs dense:   max|diff| = {gaussx_error:.2e}")
cross_error = jnp.max(jnp.abs(solution_woodbury - solution_gaussx))
print(f"Woodbury vs gaussx: max|diff| = {cross_error:.2e}")

# %% [markdown]
# ## Matrix determinant lemma
#
# Similarly, the logdet decomposes. This follows from the identity
#
# $$\det (\mathbf{A}_0 + \mathbf{U} \mathbf{D} \mathbf{V}^\top) = \det (\mathbf{D}^{-1} + \mathbf{V}^\top \mathbf{A}_0^{-1} \mathbf{U}) \det \mathbf{D} \det \mathbf{A}_0$$
#
# which is a direct consequence of the Schur complement determinant identity. Taking logarithms:
#
# $$\log \det (\mathbf{A}_0 + \mathbf{U} \mathbf{D} \mathbf{V}^\top) = \log \det \mathbf{A}_0 + \log \det \mathbf{C} + \log \det \mathbf{D}$$
#
# where $\mathbf{C} = \mathbf{D}^{-1} + \mathbf{V}^\top \mathbf{A}_0^{-1} \mathbf{U}$ is the capacitance matrix. In our case ($\mathbf{D} = \mathbf{I}$, $\mathbf{V} = \mathbf{U}$):

# %%
logdet_gaussx = gaussx.logdet(operator)
logdet_dense = jnp.linalg.slogdet(operator.as_matrix())[1]

print(f"gaussx logdet:  {logdet_gaussx:.6f}")
print(f"dense logdet:   {logdet_dense:.6f}")
print(f"match: {jnp.allclose(logdet_gaussx, logdet_dense, rtol=1e-10)}")

# %% [markdown]
# ## Visualizing the decomposition

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 3))

axes[0].imshow(jnp.diag(base_diagonal[:20]), cmap="Blues", interpolation="nearest")
axes[0].set_title(r"$\mathbf{A}_0 = \mathrm{diag}(\mathbf{a})$" + "\n(first 20x20)")

axes[1].imshow(
    low_rank_factor[:20], cmap="RdBu", interpolation="nearest", aspect="auto"
)
axes[1].set_title(r"$\mathbf{U}$" + f"\n({size}x{rank}, first 20 rows)")

axes[2].imshow(capacitance, cmap="Blues", interpolation="nearest")
axes[2].set_title(r"$\mathbf{C}$ (capacitance)" + f"\n({rank}x{rank})")

matrix_20 = jnp.abs(operator.as_matrix()[:20, :20])
axes[3].imshow(matrix_20, cmap="Blues", interpolation="nearest")
axes[3].set_title(
    r"$\mathbf{A} = \mathbf{A}_0 + \mathbf{U} \mathbf{U}^\top$" + "\n(first 20x20)"
)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Cost comparison
#
# | Operation | Dense | Woodbury |
# |-----------|-------|----------|
# | Solve | $O(N^3)$ | $O(Nk^2 + k^3)$ |
# | Logdet | $O(N^3)$ | $O(Nk^2 + k^3)$ |
# | Storage | $O(N^2)$ | $O(Nk)$ |
#
# For $N = 100$ and $k = 3$, dense does about $10^6$ operations and Woodbury about 900.

# %% [markdown]
# ## References
#
# - Hager, W. W. (1989). Updating the inverse of a matrix.
#   *SIAM Review*, 31(2), 221--239.
# - Henderson, H. V. & Searle, S. R. (1981). On deriving the inverse of a
#   sum of matrices. *SIAM Review*, 23(1), 53--60.
# - Woodbury, M. A. (1950). *Inverting Modified Matrices*. Statistical Research
#   Group Memo Report 42, Princeton University.
