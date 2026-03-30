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

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt

import gaussx


jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## 1. Diagonal
#
# The simplest operator. Lineax provides `DiagonalLinearOperator`
# and gaussx gives it O(n) fast paths for every primitive.

# %%
d = jnp.array([1.0, 4.0, 9.0, 16.0])
D = lx.DiagonalLinearOperator(d)

print(f"Type:    {type(D).__name__}")
print(f"Size:    {D.in_size()} x {D.out_size()}")
print(f"mv([1,1,1,1]): {D.mv(jnp.ones(4))}")
print(f"solve:   {gaussx.solve(D, jnp.ones(4))}")
print(f"logdet:  {gaussx.logdet(D):.4f}")
print(f"trace:   {gaussx.trace(D):.1f}")

# %% [markdown]
# ## 2. BlockDiag
#
# Block diagonal from independent sub-operators.
# Every primitive decomposes per-block.

# %%
A = lx.MatrixLinearOperator(jnp.array([[2.0, 0.5], [0.5, 3.0]]))
B = lx.DiagonalLinearOperator(jnp.array([1.0, 5.0, 7.0]))

BD = gaussx.BlockDiag(A, B)

print(f"Type:    {type(BD).__name__}")
print(f"Size:    {BD.in_size()} x {BD.out_size()}")
print(f"Blocks:  {len(BD.operators)}")
print(f"logdet:  {gaussx.logdet(BD):.4f}")
print(f"trace:   {gaussx.trace(BD):.1f}")

# %% [markdown]
# ## 3. Kronecker
#
# $A \otimes B$ stored as two small matrices.
# Matvec via Roth's column lemma, logdet/solve/cholesky per-factor.

# %%
K1 = lx.MatrixLinearOperator(jnp.array([[2.0, 0.5], [0.5, 3.0]]))
K2 = lx.MatrixLinearOperator(jnp.array([[1.0, 0.3], [0.3, 2.0]]))

K = gaussx.Kronecker(K1, K2)

print(f"Type:    {type(K).__name__}")
print(f"Size:    {K.in_size()} x {K.out_size()}")
print(f"Factors: {len(K.operators)}")
print(f"logdet:  {gaussx.logdet(K):.4f}")
print(f"trace:   {gaussx.trace(K):.1f}")

# 3-factor Kronecker
K3 = gaussx.Kronecker(K1, K2, lx.DiagonalLinearOperator(jnp.array([1.0, 2.0])))
print(f"\n3-factor Kronecker size: {K3.in_size()}")
print(f"3-factor logdet: {gaussx.logdet(K3):.4f}")

# %% [markdown]
# ## 4. LowRankUpdate
#
# $L + U \mathrm{diag}(d) V^\top$. Woodbury solve, determinant lemma logdet.

# %%
base = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 3.0, 4.0]))
U = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.0]])
d_vals = jnp.array([0.5, 0.3])

lr = gaussx.LowRankUpdate(base, U, d_vals)

print(f"Type:    {type(lr).__name__}")
print(f"Size:    {lr.in_size()} x {lr.out_size()}")
print(f"Rank:    {lr.rank}")
print(f"logdet:  {gaussx.logdet(lr):.4f}")

# Convenience constructors
lr2 = gaussx.low_rank_plus_diag(jnp.ones(4), U)
print(f"\nlow_rank_plus_diag rank: {lr2.rank}")
print(f"is_low_rank: {gaussx.is_low_rank(lr2)}")

# %% [markdown]
# ## 5. Visualizing operator structure
#
# Let's visualize the sparsity pattern of each operator type.

# %%
operators = {
    "Diagonal": D,
    "BlockDiag": BD,
    "Kronecker": K,
    "LowRankUpdate": lr,
}

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, (name, op) in zip(axes, operators.items(), strict=False):
    mat = op.as_matrix()
    im = ax.imshow(jnp.abs(mat), cmap="Blues", interpolation="nearest")
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
D_small = lx.DiagonalLinearOperator(0.1 * jnp.ones(4))
perturbed = K + D_small
print(f"K + 0.1*I type: {type(perturbed).__name__}")
print(
    f"(K + 0.1*I).mv == K.mv + 0.1*v: "
    f"{jnp.allclose((K + D_small).mv(jnp.ones(4)), K.mv(jnp.ones(4)) + 0.1)}"
)

# Scalar multiplication
scaled = K * 2.0
print(f"2*K type: {type(scaled).__name__}")

# %% [markdown]
# ## 7. Tag queries
#
# Every operator carries structural tags that gaussx uses for dispatch.

# %%
print("Diagonal:")
print(f"  is_symmetric:  {gaussx.is_symmetric(D)}")
print(f"  is_diagonal:   {gaussx.is_diagonal(D)}")

print("\nKronecker:")
print(f"  is_kronecker:  {gaussx.is_kronecker(K)}")
print(f"  is_symmetric:  {gaussx.is_symmetric(K)}")

print("\nBlockDiag:")
print(f"  is_block_diagonal: {gaussx.is_block_diagonal(BD)}")

print("\nLowRankUpdate:")
print(f"  is_low_rank:   {gaussx.is_low_rank(lr)}")

# %% [markdown]
# ## Summary
#
# | Operator | Storage | mv cost | Key identity |
# |----------|---------|---------|-------------|
# | `Diagonal` | O(n) | O(n) | Built into lineax |
# | `BlockDiag` | sum O(n_i^2) | sum O(n_i^2) | Per-block decomposition |
# | `Kronecker` | sum O(n_i^2) | O(sum n_i^3) | Roth's column lemma |
# | `LowRankUpdate` | O(nk) | O(nk) | Woodbury identity |
