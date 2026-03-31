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

# %% [markdown]
# ### Why structured linear algebra?
#
# Gaussian process (GP) inference and many probabilistic models require
# computing $\alpha = K^{-1} y$ (solve) and $\log |K|$ (log-determinant)
# where $K$ is an $n \times n$ covariance matrix. With a dense
# representation both operations cost $O(n^3)$ time and $O(n^2)$ memory,
# which becomes prohibitive for $n > 10^4$.
#
# When the covariance matrix has *structure* -- Kronecker, block-diagonal,
# low-rank-plus-diagonal -- each operation can be reduced dramatically.
# For example, a Kronecker product $A \otimes B$ with factors of size
# $n_A$ and $n_B$ reduces $O((n_A n_B)^3)$ to $O(n_A^3 + n_B^3)$.
# gaussx encodes this structure in lineax-compatible operators so that
# `solve`, `logdet`, `cholesky`, and other primitives automatically
# dispatch to the most efficient algorithm.

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
#
# The key identities that make this possible (Van Loan, 2000):
#
# **Matvec** via Roth's column lemma:
# $$(A \otimes B)\operatorname{vec}(X) = \operatorname{vec}(B X A^\top)$$
#
# **Solve:**
# $$(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$$
#
# **Log-determinant:**
# $$\log|A \otimes B| = n_B \log|A| + n_A \log|B|$$
#
# **Cholesky:**
# $$\operatorname{chol}(A \otimes B)
# = \operatorname{chol}(A) \otimes \operatorname{chol}(B)$$

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
# matrix determinant lemma -- both $O(nk^2 + k^3)$ for rank $k$.
#
# **Woodbury identity** (Hager, 1989):
# $$(A + UCV)^{-1} = A^{-1} - A^{-1}U\bigl(C^{-1} + VA^{-1}U\bigr)^{-1}VA^{-1}$$
#
# **Matrix determinant lemma:**
# $$\log|A + UCV| = \log|C^{-1} + VA^{-1}U| + \log|C| + \log|A|$$
#
# These reduce an $n \times n$ solve/logdet to operations on the
# $k \times k$ capacitance matrix $C^{-1} + V A^{-1} U$.

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
# ## 7. JAX transforms: jit and grad
#
# Everything is compatible with `jit`, `vmap`, and `grad` because
# all operators are equinox modules (PyTrees).
#
# Differentiability is essential for GP hyperparameter optimization:
# we minimize the negative log-marginal likelihood
# $-\log p(y|\theta) = \tfrac{1}{2}y^\top K_\theta^{-1} y
# + \tfrac{1}{2}\log|K_\theta| + \text{const}$
# with respect to kernel hyperparameters $\theta$ via gradient descent
# (Rasmussen & Williams, 2006, Ch. 5). Because gaussx primitives are
# differentiable through JAX, gradients flow through `solve` and `logdet`
# automatically.

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


# %%
# grad through solve and logdet
def loss(log_diag, y):
    op = lx.DiagonalLinearOperator(jnp.exp(log_diag))
    alpha = gaussx.solve(op, y)
    return 0.5 * jnp.dot(y, alpha) + 0.5 * gaussx.logdet(op)


log_d = jnp.log(jnp.array([1.0, 2.0, 3.0]))
g = jax.grad(loss)(log_d, jnp.ones(3))
print("grad shape:", g.shape)
print("gradient:", g)

# %% [markdown]
# ## 8. vmap: batching over vectors and operators
#
# gaussx primitives are **vector** operations — `solve(op, b)` takes a
# single vector `b`, not a matrix. To solve for multiple right-hand
# sides or multiple operators, use `jax.vmap`. This is the JAX way:
# write scalar/vector code, then batch it.
#
# > **Key rule:** gaussx primitives work on single vectors. Use
# > `jax.vmap` to batch over vectors, operators, or both.

# %%
# --- vmap over a batch of vectors (fixed operator) ---
# Solve D x = b for 5 different b vectors
b_batch = jax.random.normal(jax.random.PRNGKey(0), (5, 3))

x_batch = jax.vmap(lambda b: gaussx.solve(D, b))(b_batch)
print("vmap over vectors:", x_batch.shape)  # (5, 3)

# Verify against manual loop
for i in range(5):
    assert jnp.allclose(x_batch[i], gaussx.solve(D, b_batch[i]))
print("all match ✓")

# %%
# --- vmap over columns (matrix RHS) ---
# gaussx.solve expects a vector, but you can solve A X = B
# by vmapping over the columns of B
n = 4
M = jnp.array(
    [
        [2.0, 1.0, 0.5, 0.2],
        [1.0, 3.0, 0.5, 0.1],
        [0.5, 0.5, 4.0, 0.3],
        [0.2, 0.1, 0.3, 2.0],
    ]
)
M_op = lx.MatrixLinearOperator(M, lx.positive_semidefinite_tag)

B_rhs = jax.random.normal(jax.random.PRNGKey(1), (n, 3))  # 3 right-hand sides

X = jax.vmap(lambda col: gaussx.solve(M_op, col), in_axes=1, out_axes=1)(B_rhs)
print("solve matrix RHS:", X.shape)  # (4, 3)

# Verify
X_ref = jnp.linalg.solve(M, B_rhs)
print("max error:", jnp.max(jnp.abs(X - X_ref)))

# %%
# --- vmap over operators (e.g. hyperparameter sweep) ---
# Solve for different diagonal scalings
scales = jnp.array([0.5, 1.0, 2.0, 4.0])
b_fixed = jnp.ones(3)


def solve_scaled(scale):
    op = lx.DiagonalLinearOperator(scale * d)
    return gaussx.solve(op, b_fixed)


x_scaled = jax.vmap(solve_scaled)(scales)
print("vmap over operators:", x_scaled.shape)  # (4, 3)
print("scale=0.5:", x_scaled[0])  # b / (0.5 * d)
print("scale=4.0:", x_scaled[3])  # b / (4.0 * d)

# %%
# --- vmap over operators AND vectors simultaneously ---
# Common in GP: different kernel matrix per hyperparameter sample


def _make_psd(key, n):
    A = jax.random.normal(key, (n, n))
    return A @ A.T + 0.5 * jnp.eye(n)


keys = jax.random.split(jax.random.PRNGKey(2), 4)
matrices = jax.vmap(lambda k: _make_psd(k, 3))(keys)
vectors = jax.random.normal(jax.random.PRNGKey(3), (4, 3))


def solve_one(K_i, b_i):
    op = lx.MatrixLinearOperator(K_i, lx.positive_semidefinite_tag)
    return gaussx.solve(op, b_i)


xs = jax.vmap(solve_one)(matrices, vectors)
print("vmap over (K, b):", xs.shape)  # (4, 3)


# %%
# --- vmap works with all primitives ---
def all_primitives(K_i):
    op = lx.MatrixLinearOperator(K_i, lx.positive_semidefinite_tag)
    return (
        gaussx.logdet(op),
        gaussx.trace(op),
        gaussx.diag(op),
    )


lds, trs, diags = jax.vmap(all_primitives)(matrices)
print("vmapped logdet:", lds.shape)  # (4,)
print("vmapped trace:", trs.shape)  # (4,)
print("vmapped diag:", diags.shape)  # (4, 3)

# %%
# --- vmap with structured operators ---
# Kronecker solve batched over right-hand sides
A_pd = jnp.array([[2.0, 0.5], [0.5, 3.0]])
B_pd = jnp.array([[4.0, 1.0], [1.0, 2.0]])
K_pd = gaussx.Kronecker(
    lx.MatrixLinearOperator(A_pd, lx.positive_semidefinite_tag),
    lx.MatrixLinearOperator(B_pd, lx.positive_semidefinite_tag),
)

b_batch_4 = jax.random.normal(jax.random.PRNGKey(4), (10, 4))
x_kron = jax.vmap(lambda b: gaussx.solve(K_pd, b))(b_batch_4)
print("Kronecker batched solve:", x_kron.shape)  # (10, 4)

# Verify against dense
x_ref = jnp.linalg.solve(K_pd.as_matrix(), b_batch_4.T).T
print("max error:", jnp.max(jnp.abs(x_kron - x_ref)))

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

# %% [markdown]
# ## References
#
# - Rasmussen, C. E. & Williams, C. K. I. (2006).
#   *Gaussian Processes for Machine Learning*. MIT Press.
# - Van Loan, C. F. (2000). The ubiquitous Kronecker
#   product. *J. Comput. Appl. Math.*, 123, 85--100.
# - Hager, W. W. (1989). Updating the inverse of a
#   matrix. *SIAM Review*, 31(2), 221--239.
# - Saatci, Y. (2012). *Scalable Inference for Structured
#   Gaussian Process Models*. PhD thesis, Cambridge.
