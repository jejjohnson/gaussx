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
# # Operator Zoo
#
# A tour of every operator type in gaussx, showing construction,
# sizes, matvec, and how they compose.
#
# Structured linear operators arise throughout computational mathematics:
# Kronecker products in multi-dimensional PDEs and separable covariance
# kernels, block-diagonal matrices in independent subsystems and mixture
# models, and low-rank updates in Kalman filtering and spatial statistics.
# By representing these operators in factored form, we avoid forming and factorizing dense $N \times N$ matrices -- often reducing $O(N^3)$ operations to $O(N)$ or $O(N \log N)$.

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
# ## 1. Diagonal
#
# The simplest operator. Lineax provides `DiagonalLinearOperator` and gaussx gives it $O(N)$ fast paths for every primitive.

# %%
diagonal = jnp.array([1.0, 4.0, 9.0, 16.0])
diagonal_op = lx.DiagonalLinearOperator(diagonal)

print(f"Type:    {type(diagonal_op).__name__}")
print(f"Size:    {diagonal_op.in_size()} x {diagonal_op.out_size()}")
print(f"mv([1,1,1,1]): {diagonal_op.mv(jnp.ones(4))}")
print(f"solve:   {gaussx.solve(diagonal_op, jnp.ones(4))}")
print(f"logdet:  {gaussx.logdet(diagonal_op):.4f}")
print(f"trace:   {gaussx.trace(diagonal_op):.1f}")

# %% [markdown]
# ## 2. BlockDiag
#
# Block diagonal from independent sub-operators.
# Every primitive decomposes per-block.

# %%
block_a = lx.MatrixLinearOperator(jnp.array([[2.0, 0.5], [0.5, 3.0]]))
block_b = lx.DiagonalLinearOperator(jnp.array([1.0, 5.0, 7.0]))

block_diag_op = gaussx.BlockDiag(block_a, block_b)

print(f"Type:    {type(block_diag_op).__name__}")
print(f"Size:    {block_diag_op.in_size()} x {block_diag_op.out_size()}")
print(f"Blocks:  {len(block_diag_op.operators)}")
print(f"logdet:  {gaussx.logdet(block_diag_op):.4f}")
print(f"trace:   {gaussx.trace(block_diag_op):.1f}")

# %% [markdown]
# ## 3. Kronecker
#
# $\mathbf{A} \otimes \mathbf{B}$ stored as two small matrices. Matvec via Roth's column lemma, logdet/solve/cholesky per-factor.
#
# The eigenvalue decomposition of $\mathbf{A} \otimes \mathbf{B}$ inherits from its factors: if $\mathbf{A} \mathbf{v}^A_i = \gamma^A_i \mathbf{v}^A_i$ and $\mathbf{B} \mathbf{v}^B_j = \gamma^B_j \mathbf{v}^B_j$, then
#
# $$\gamma_{ij} = \gamma^A_i \gamma^B_j$$
#
# $$\mathbf{v}_{ij} = \mathbf{v}^A_i \otimes \mathbf{v}^B_j$$
#
# This means all spectral operations on the product decompose into operations on the factors, reducing $O((N_1 N_2)^3)$ to $O(N_1^3 + N_2^3)$ (Van Loan, 2000).

# %%
factor_1 = lx.MatrixLinearOperator(jnp.array([[2.0, 0.5], [0.5, 3.0]]))
factor_2 = lx.MatrixLinearOperator(jnp.array([[1.0, 0.3], [0.3, 2.0]]))

kronecker_op = gaussx.Kronecker(factor_1, factor_2)

print(f"Type:    {type(kronecker_op).__name__}")
print(f"Size:    {kronecker_op.in_size()} x {kronecker_op.out_size()}")
print(f"Factors: {len(kronecker_op.operators)}")
print(f"logdet:  {gaussx.logdet(kronecker_op):.4f}")
print(f"trace:   {gaussx.trace(kronecker_op):.1f}")

# 3-factor Kronecker
kronecker_3_op = gaussx.Kronecker(
    factor_1, factor_2, lx.DiagonalLinearOperator(jnp.array([1.0, 2.0]))
)
print(f"\n3-factor Kronecker size: {kronecker_3_op.in_size()}")
print(f"3-factor logdet: {gaussx.logdet(kronecker_3_op):.4f}")

# %% [markdown]
# ## 4. LowRankUpdate
#
# $\mathbf{A}_0 + \mathbf{U} \mathrm{diag}(\mathbf{d}) \mathbf{V}^\top$. Woodbury solve, determinant lemma logdet.
#
# **Woodbury identity** (Hager, 1989):
# $$(\mathbf{A} + \mathbf{U} \mathbf{C} \mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1} \mathbf{U} (\mathbf{C}^{-1} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U})^{-1} \mathbf{V} \mathbf{A}^{-1}$$
#
# **Matrix determinant lemma:**
# $$\log \det (\mathbf{A} + \mathbf{U} \mathbf{C} \mathbf{V}) = \log \det (\mathbf{C}^{-1} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U}) + \log \det \mathbf{C} + \log \det \mathbf{A}$$
#
# Both reduce an $N \times N$ problem to a $k \times k$ problem on the capacitance matrix, where $k$ is the rank of the update.

# %%
base_op = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 3.0, 4.0]))
low_rank_factor = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.0]])
update_diagonal = jnp.array([0.5, 0.3])

# A = A₀ + U diag(d) Uᵀ
low_rank_op = gaussx.LowRankUpdate(base_op, low_rank_factor, update_diagonal)

print(f"Type:    {type(low_rank_op).__name__}")
print(f"Size:    {low_rank_op.in_size()} x {low_rank_op.out_size()}")
print(f"Rank:    {low_rank_op.rank}")
print(f"logdet:  {gaussx.logdet(low_rank_op):.4f}")

# Convenience constructors
low_rank_op_2 = gaussx.low_rank_plus_diag(jnp.ones(4), low_rank_factor)
print(f"\nlow_rank_plus_diag rank: {low_rank_op_2.rank}")
print(f"is_low_rank: {gaussx.is_low_rank(low_rank_op_2)}")

# %% [markdown]
# ## 5. Visualizing operator structure
#
# Let's visualize the sparsity pattern of each operator type.

# %%
operators = {
    "Diagonal": diagonal_op,
    "BlockDiag": block_diag_op,
    "Kronecker": kronecker_op,
    "LowRankUpdate": low_rank_op,
}

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, (name, op) in zip(axes, operators.items(), strict=False):
    matrix = op.as_matrix()
    im = ax.imshow(jnp.abs(matrix), cmap="Blues", interpolation="nearest")
    ax.set_title(f"{name}\n({op.in_size()}x{op.out_size()})")
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle("Operator Structure (|entries|)", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Composition with lineax arithmetic
#
# gaussx operators compose with lineax's `+`, `@`, `*` operators.

# %%
# Kronecker + diagonal perturbation
small_diagonal_op = lx.DiagonalLinearOperator(0.1 * jnp.ones(4))
perturbed = kronecker_op + small_diagonal_op
print(f"K + 0.1*I type: {type(perturbed).__name__}")
sum_mv = (kronecker_op + small_diagonal_op).mv(jnp.ones(4))
split_mv = kronecker_op.mv(jnp.ones(4)) + 0.1
print(f"(K + 0.1*I).mv == K.mv + 0.1*v: {jnp.allclose(sum_mv, split_mv)}")

# Scalar multiplication
scaled = kronecker_op * 2.0
print(f"2*K type: {type(scaled).__name__}")

# %% [markdown]
# ## 7. Tag queries
#
# Every operator carries structural tags that gaussx uses for dispatch.

# %%
print("Diagonal:")
print(f"  is_symmetric:  {gaussx.is_symmetric(diagonal_op)}")
print(f"  is_diagonal:   {gaussx.is_diagonal(diagonal_op)}")

print("\nKronecker:")
print(f"  is_kronecker:  {gaussx.is_kronecker(kronecker_op)}")
print(f"  is_symmetric:  {gaussx.is_symmetric(kronecker_op)}")

print("\nBlockDiag:")
print(f"  is_block_diagonal: {gaussx.is_block_diagonal(block_diag_op)}")

print("\nLowRankUpdate:")
print(f"  is_low_rank:   {gaussx.is_low_rank(low_rank_op)}")

# %% [markdown]
# ## Summary
#
# | Operator | Storage | mv cost | Key identity |
# |----------|---------|---------|-------------|
# | `Diagonal` | $O(N)$ | $O(N)$ | Built into lineax |
# | `BlockDiag` | sum $O(N_i^2)$ | sum $O(N_i^2)$ | Per-block decomposition |
# | `Kronecker` | sum $O(N_i^2)$ | $O(N \sum_i N_i)$ | Roth's column lemma |
# | `LowRankUpdate` | $O(Nk)$ | $O(Nk)$ | Woodbury identity |

# %% [markdown]
# ## References
#
# - Golub, G. H. & Van Loan, C. F. (2013).
#   *Matrix Computations*. 4th ed., Johns Hopkins.
# - Hager, W. W. (1989). Updating the inverse of a
#   matrix. *SIAM Review*, 31(2), 221--239.
# - Van Loan, C. F. (2000). The ubiquitous Kronecker
#   product. *J. Comput. Appl. Math.*, 123, 85--100.
