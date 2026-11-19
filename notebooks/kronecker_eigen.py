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
# # Kronecker Eigendecomposition
#
# For a Kronecker product $\mathbf{A} \otimes \mathbf{B}$, all eigenvalues and eigenvectors decompose per-factor:
#
# $$\gamma_{ij} = \gamma^A_i \gamma^B_j$$
#
# $$\mathbf{v}_{ij} = \mathbf{v}^A_i \otimes \mathbf{v}^B_j$$
#
# This means we can compute the full spectrum of an $N \times N$ matrix ($N = N_1 N_2$) by only decomposing two small matrices, where $\mathbf{A} \in \mathbb{R}^{N_1 \times N_1}$, $\mathbf{B} \in \mathbb{R}^{N_2 \times N_2}$, and $i = 1, \ldots, N_1$ and $j = 1, \ldots, N_2$ index the factor eigenpairs.
# gaussx exploits this for `cholesky`, `sqrt`, `logdet`, and `inv`.

# %% [markdown]
# ## Context
#
# The Kronecker eigendecomposition is a special case of the tensor product spectral theorem. If $\mathbf{A} = \mathbf{Q}_A \mathrm{diag}(\boldsymbol{\gamma}_A) \mathbf{Q}_A^\top$ and $\mathbf{B} = \mathbf{Q}_B \mathrm{diag}(\boldsymbol{\gamma}_B) \mathbf{Q}_B^\top$, then
#
# $$\mathbf{A} \otimes \mathbf{B} = (\mathbf{Q}_A \otimes \mathbf{Q}_B) (\mathrm{diag}(\boldsymbol{\gamma}_A) \otimes \mathrm{diag}(\boldsymbol{\gamma}_B)) (\mathbf{Q}_A \otimes \mathbf{Q}_B)^\top$$
#
# This result is foundational for:
#
# - **GP inference on grids** -- computing solves and log-determinants
#   without forming the full $N \times N$ matrix (Saatci, 2012).
# - **Solving Kronecker-structured linear systems** -- reducing an
#   $N$-dimensional problem to two smaller per-factor problems.
# - **Stochastic PDEs with separable operators** -- exploiting product
#   structure in spatial and temporal discretizations.

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
# ## Setup: PSD Kronecker product

# %%
key = jax.random.PRNGKey(42)
key_a, key_b = jax.random.split(key)

size_a, size_b = 8, 10
size = size_a * size_b

# Random PSD matrices
random_a = jax.random.normal(key_a, (size_a, size_a))
factor_a = random_a @ random_a.T + 0.1 * jnp.eye(size_a)

random_b = jax.random.normal(key_b, (size_b, size_b))
factor_b = random_b @ random_b.T + 0.1 * jnp.eye(size_b)

factor_a_op = lx.MatrixLinearOperator(factor_a, lx.positive_semidefinite_tag)
factor_b_op = lx.MatrixLinearOperator(factor_b, lx.positive_semidefinite_tag)
kronecker_op = gaussx.Kronecker(factor_a_op, factor_b_op)

print(f"A: {size_a}x{size_a}, B: {size_b}x{size_b}")
print(f"A kron B: {size}x{size} = {size**2:,} entries")

# %% [markdown]
# ## Per-factor eigenvalues

# %%
eigenvalues_a = jnp.linalg.eigvalsh(factor_a)
eigenvalues_b = jnp.linalg.eigvalsh(factor_b)

# γ_ij = γ^A_i γ^B_j, the outer product of the per-factor eigenvalues
eigenvalues_kron = jnp.sort(jnp.outer(eigenvalues_a, eigenvalues_b).ravel())

# Dense eigenvalues (for verification)
eigenvalues_dense = jnp.linalg.eigvalsh(kronecker_op.as_matrix())

print(f"Per-factor eigenvalues: {size_a} + {size_b} = {size_a + size_b} eigh calls")
print(f"Dense eigenvalues: one {size}x{size} eigh call")
eigenvalue_error = jnp.max(jnp.abs(eigenvalues_kron - eigenvalues_dense))
print(f"Max eigenvalue error: {eigenvalue_error:.2e}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

axes[0].stem(range(size_a), eigenvalues_a, linefmt="C0-", markerfmt="C0o", basefmt="k-")
axes[0].set_title(f"Eigenvalues of $\\mathbf{{A}}$ ({size_a}x{size_a})")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("$\\gamma$")
axes[0].grid(True, which="major", alpha=0.3)
axes[0].grid(True, which="minor", alpha=0.1)
axes[0].minorticks_on()

axes[1].stem(range(size_b), eigenvalues_b, linefmt="C1-", markerfmt="C1o", basefmt="k-")
axes[1].set_title(f"Eigenvalues of $\\mathbf{{B}}$ ({size_b}x{size_b})")
axes[1].set_xlabel("Index")
axes[1].grid(True, which="major", alpha=0.3)
axes[1].grid(True, which="minor", alpha=0.1)
axes[1].minorticks_on()

axes[2].semilogy(eigenvalues_kron, "C2-", lw=2, label="Kronecker (per-factor)")
axes[2].semilogy(eigenvalues_dense, "k--", lw=1.5, label="Dense (verification)")
axes[2].set_title(
    f"Eigenvalues of $\\mathbf{{A}} \\otimes \\mathbf{{B}}$ ({size}x{size})"
)
axes[2].set_xlabel("Index")
axes[2].legend(fontsize=9)
axes[2].grid(True, which="major", alpha=0.3)
axes[2].grid(True, which="minor", alpha=0.1)
axes[2].minorticks_on()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Structured Cholesky
#
# $\mathrm{chol}(\mathbf{A} \otimes \mathbf{B}) = \mathrm{chol}(\mathbf{A}) \otimes \mathrm{chol}(\mathbf{B})$

# %%
cholesky_factor = gaussx.cholesky(kronecker_op)
print(f"cholesky type: {type(cholesky_factor).__name__}")

# Reconstruction error
# L Lᵀ ≈ A ⊗ B
cholesky_matrix = cholesky_factor.as_matrix()
reconstruction = cholesky_matrix @ cholesky_factor.as_matrix().T
reconstruction_error = jnp.max(jnp.abs(reconstruction - kronecker_op.as_matrix()))
print(f"||L L^T - K||_max: {reconstruction_error:.2e}")

# %% [markdown]
# ## Structured sqrt
#
# $(\mathbf{A} \otimes \mathbf{B})^{1/2} = \mathbf{A}^{1/2} \otimes \mathbf{B}^{1/2}$

# %%
sqrt_op = gaussx.sqrt(kronecker_op)
print(f"sqrt type: {type(sqrt_op).__name__}")

# Verify S S = A ⊗ B
sqrt_matrix = sqrt_op.as_matrix()
reconstruction_sqrt = sqrt_matrix @ sqrt_op.as_matrix()
sqrt_error = jnp.max(jnp.abs(reconstruction_sqrt - kronecker_op.as_matrix()))
print(f"||S S - K||_max: {sqrt_error:.2e}")

# %% [markdown]
# ## Structured logdet
#
# $\log \det (\mathbf{A} \otimes \mathbf{B}) = N_2 \log \det \mathbf{A} + N_1 \log \det \mathbf{B}$
#
# **Derivation.** Using the Kronecker eigenvalues $\gamma_{ij} = \gamma^A_i \gamma^B_j$:
#
# $$\log \det (\mathbf{A} \otimes \mathbf{B}) = \sum_{i,j} \log (\gamma^A_i \gamma^B_j) = \sum_{i,j} (\log \gamma^A_i + \log \gamma^B_j)$$
#
# $$\sum_{i,j} (\log \gamma^A_i + \log \gamma^B_j) = N_2 \sum_i \log \gamma^A_i + N_1 \sum_j \log \gamma^B_j = N_2 \log \det \mathbf{A} + N_1 \log \det \mathbf{B}$$

# %%
logdet_structured = gaussx.logdet(kronecker_op)
logdet_dense = jnp.linalg.slogdet(kronecker_op.as_matrix())[1]
# log det (A ⊗ B) = ∑ᵢⱼ log γ_ij
logdet_from_eigs = jnp.sum(jnp.log(eigenvalues_kron))

print(f"Structured logdet:  {logdet_structured:.6f}")
print(f"Dense logdet:       {logdet_dense:.6f}")
print(f"From eigenvalues:   {logdet_from_eigs:.6f}")

# %% [markdown]
# ## Summary
#
# | Operation | Dense cost | Kronecker cost | Speedup |
# |-----------|-----------|---------------|---------|
# | Eigenvalues | $O(N^3)$ | $O(N_1^3 + N_2^3)$ | $\approx N / \max(N_1, N_2)$ |
# | Cholesky | $O(N^3)$ | $O(N_1^3 + N_2^3)$ | $\approx N / \max(N_1, N_2)$ |
# | Logdet | $O(N^3)$ | $O(N_1^3 + N_2^3)$ | $\approx N / \max(N_1, N_2)$ |
# | Solve | $O(N^3)$ | $O(N_1^3 + N_2^3)$ | $\approx N / \max(N_1, N_2)$ |

# %% [markdown]
# ## References
#
# - Van Loan, C. F. (2000). The ubiquitous Kronecker product. *Journal
#   of Computational and Applied Mathematics*, 123, 85--100.
# - Saatci, Y. (2012). *Scalable Inference for Structured Gaussian
#   Process Models*. PhD thesis, University of Cambridge.
# - Steeb, W.-H. (2011). *Matrix Calculus and Kronecker Product*.
#   2nd edition, World Scientific.
