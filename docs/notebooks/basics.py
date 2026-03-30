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
# # GaussX Basics
#
# This notebook introduces the core building blocks of gaussx:
# structured linear operators and primitives that exploit their structure.
#
# **What you'll learn:**
#
# 1. How to construct `Kronecker`, `BlockDiag`, and `LowRankUpdate` operators
# 2. How primitives (`solve`, `logdet`, `cholesky`, etc.) dispatch on structure
# 3. How everything composes with JAX transforms (`jit`, `vmap`, `grad`)

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx

import gaussx


jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## 1. Diagonal operators
#
# The simplest structured operator. All gaussx primitives have
# O(n) fast paths for diagonals.

# %%
d = jnp.array([1.0, 4.0, 9.0])
D = lx.DiagonalLinearOperator(d)

# Solve D x = b is just elementwise division
b = jnp.array([2.0, 8.0, 27.0])
x = gaussx.solve(D, b)
print("solve:", x)  # [2.0, 2.0, 3.0]

# logdet is sum of log|d_i|
print("logdet:", gaussx.logdet(D))  # log(1) + log(4) + log(9) = log(36)
print("trace:", gaussx.trace(D))  # 1 + 4 + 9 = 14

# %% [markdown]
# ## 2. Kronecker products
#
# A Kronecker product $A \otimes B$ has size $(m_A m_B) \times (n_A n_B)$
# but is stored as two small matrices. gaussx exploits this structure
# for O(n_A^3 + n_B^3) operations instead of O((n_A n_B)^3).

# %%
A = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
B = lx.MatrixLinearOperator(jnp.array([[1.0, 0.5], [0.5, 2.0]]))

K = gaussx.Kronecker(A, B)
print("Kronecker size:", K.in_size(), "x", K.out_size())  # 4 x 4

# Efficient matvec via Roth's column lemma (no 4x4 matrix formed)
v = jnp.ones(4)
print("K @ v:", K.mv(v))

# logdet decomposes: logdet(A kron B) = n_B * logdet(A) + n_A * logdet(B)
print("logdet(K):", gaussx.logdet(K))

# Verify against dense
print("logdet(dense):", jnp.linalg.slogdet(K.as_matrix())[1])

# %% [markdown]
# ## 3. Block diagonal operators
#
# Block diagonal operators act independently on each block.
# All primitives decompose per-block.

# %%
block1 = lx.MatrixLinearOperator(jnp.array([[4.0, 1.0], [1.0, 3.0]]))
block2 = lx.DiagonalLinearOperator(jnp.array([2.0, 5.0, 7.0]))

BD = gaussx.BlockDiag(block1, block2)
print("BlockDiag size:", BD.in_size())  # 5

b = jnp.ones(5)
x = gaussx.solve(BD, b)
print("solve:", x)

# logdet is sum of per-block logdets
print("logdet:", gaussx.logdet(BD))

# %% [markdown]
# ## 4. Low-rank updates
#
# `LowRankUpdate` represents $L + U \mathrm{diag}(d) V^\top$ where $L$ is
# any operator. Solve uses the Woodbury identity, logdet uses the
# matrix determinant lemma — both O(nk^2 + k^3) for rank k.

# %%
# Diagonal + rank-2 update
diag_vals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
U = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

lr = gaussx.low_rank_plus_diag(diag_vals, U)
print("LowRankUpdate rank:", lr.rank)

v = jnp.ones(5)
x = gaussx.solve(lr, v)
print("solve:", x)

# Verify
x_dense = jnp.linalg.solve(lr.as_matrix(), v)
print("max error:", jnp.max(jnp.abs(x - x_dense)))

# %% [markdown]
# ## 5. Cholesky and square root
#
# Both preserve structure: `cholesky(Kronecker(A, B))` returns
# `Kronecker(cholesky(A), cholesky(B))`.

# %%
# PSD Kronecker product
A_pd = jnp.array([[2.0, 0.5], [0.5, 3.0]])
B_pd = jnp.array([[4.0, 1.0], [1.0, 2.0]])
K_pd = gaussx.Kronecker(
    lx.MatrixLinearOperator(A_pd, lx.positive_semidefinite_tag),
    lx.MatrixLinearOperator(B_pd, lx.positive_semidefinite_tag),
)

L = gaussx.cholesky(K_pd)
print("cholesky type:", type(L).__name__)  # Kronecker!

# Verify: L @ L^T = K
reconstructed = L.as_matrix() @ L.as_matrix().T
print("reconstruction error:", jnp.max(jnp.abs(reconstructed - K_pd.as_matrix())))

# %% [markdown]
# ## 6. Lazy inverse
#
# `inv` returns a new operator that computes $A^{-1} v$ via solve on demand.

# %%
D_inv = gaussx.inv(D)
print("D^{-1} diagonal:", gaussx.diag(D_inv))  # [1, 0.25, 0.111...]

# For structured types, inv preserves structure
K_inv = gaussx.inv(K)
print("inv(Kronecker) type:", type(K_inv).__name__)  # Kronecker

# %% [markdown]
# ## 7. JAX transforms
#
# Everything is compatible with `jit`, `vmap`, and `grad` because
# all operators are equinox modules (PyTrees).

# %%
import equinox as eqx


# JIT
@eqx.filter_jit
def neg_log_marginal(op, y):
    """Negative log marginal likelihood (up to constants)."""
    alpha = gaussx.solve(op, y)
    return 0.5 * jnp.dot(y, alpha) + 0.5 * gaussx.logdet(op)


y = jnp.array([1.0, 2.0, 3.0, 4.0])
nll = neg_log_marginal(K, y)
print("neg log marginal:", nll)

# vmap over multiple right-hand sides
solve_batch = jax.vmap(lambda v: gaussx.solve(D, v))
vs = jnp.ones((3, 3))
print("batched solve shape:", solve_batch(vs).shape)

# %% [markdown]
# ## Summary
#
# | Primitive | Diagonal | BlockDiag | Kronecker | LowRankUpdate | Dense |
# |-----------|----------|-----------|-----------|---------------|-------|
# | `solve` | O(n) | per-block | per-factor | Woodbury | lineax |
# | `logdet` | O(n) | sum | scaled sum | det lemma | slogdet |
# | `cholesky` | O(n) | per-block | per-factor | -- | Cholesky |
# | `trace` | O(n) | sum | product | -- | O(n^2) |
# | `inv` | O(n) | per-block | per-factor | -- | lazy |
