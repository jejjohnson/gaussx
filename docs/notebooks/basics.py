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
# Gaussian process (GP) inference and many probabilistic models require computing $\boldsymbol{\alpha} = \mathbf{K}_{XX}^{-1} \mathbf{y}$ (solve) and $\log \det \mathbf{K}_{XX}$ (log-determinant), where $\mathbf{K}_{XX} \in \mathbb{R}^{N \times N}$ is the kernel matrix of $N$ training inputs. With a dense representation both operations cost $O(N^3)$ time and $O(N^2)$ memory, which becomes prohibitive for $N > 10^4$.
#
# When the kernel matrix has *structure* -- Kronecker, block-diagonal, low-rank-plus-diagonal -- each operation can be reduced dramatically. For example, a Kronecker product $\mathbf{A} \otimes \mathbf{B}$ with factors of size $N_A$ and $N_B$ reduces $O((N_A N_B)^3)$ to $O(N_A^3 + N_B^3)$. gaussx encodes this structure in lineax-compatible operators so that
# `solve`, `logdet`, `cholesky`, and other primitives automatically
# dispatch to the most efficient algorithm.

# %%
from __future__ import annotations

import warnings


warnings.filterwarnings("ignore", message=r".*IProgress.*")

import jax
import jax.numpy as jnp
import lineax as lx

import gaussx


jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## 1. Diagonal operators
#
# The simplest structured operator. All gaussx primitives have $O(N)$ fast paths for diagonals.

# %%
diagonal = jnp.array([1.0, 4.0, 9.0])
diagonal_op = lx.DiagonalLinearOperator(diagonal)

# D x = b is elementwise division: xᵢ = bᵢ / dᵢ
rhs = jnp.array([2.0, 8.0, 27.0])
solution = gaussx.solve(diagonal_op, rhs)
# Expected: [2.0, 2.0, 3.0]
print("solve:", solution)

# log det D = ∑ᵢ log |dᵢ| = log(1) + log(4) + log(9) = log(36)
print("logdet:", gaussx.logdet(diagonal_op))
# tr D = ∑ᵢ dᵢ = 1 + 4 + 9 = 14
print("trace:", gaussx.trace(diagonal_op))

# %% [markdown]
# ## 2. Kronecker products
#
# A Kronecker product $\mathbf{A} \otimes \mathbf{B}$ with $\mathbf{A} \in \mathbb{R}^{M_A \times N_A}$ and $\mathbf{B} \in \mathbb{R}^{M_B \times N_B}$ has size $(M_A M_B) \times (N_A N_B)$ but is stored as two small matrices. gaussx exploits this structure for $O(N_A^3 + N_B^3)$ operations instead of $O((N_A N_B)^3)$.
#
# The key identities that make this possible (Van Loan, 2000):
#
# **Matvec** via Roth's column lemma:
# $$(\mathbf{A} \otimes \mathbf{B}) \mathrm{vec}(\mathbf{X}) = \mathrm{vec}(\mathbf{B} \mathbf{X} \mathbf{A}^\top)$$
#
# **Solve:**
# $$(\mathbf{A} \otimes \mathbf{B})^{-1} = \mathbf{A}^{-1} \otimes \mathbf{B}^{-1}$$
#
# **Log-determinant:**
# $$\log \det (\mathbf{A} \otimes \mathbf{B}) = N_B \log \det \mathbf{A} + N_A \log \det \mathbf{B}$$
#
# **Cholesky:**
# $$\mathrm{chol}(\mathbf{A} \otimes \mathbf{B}) = \mathrm{chol}(\mathbf{A}) \otimes \mathrm{chol}(\mathbf{B})$$

# %%
factor_a = lx.MatrixLinearOperator(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
factor_b = lx.MatrixLinearOperator(jnp.array([[1.0, 0.5], [0.5, 2.0]]))

kronecker_op = gaussx.Kronecker(factor_a, factor_b)
# 4 x 4
print("Kronecker size:", kronecker_op.in_size(), "x", kronecker_op.out_size())

# Efficient matvec via Roth's column lemma (no 4x4 matrix formed)
ones_vector = jnp.ones(4)
print("K @ v:", kronecker_op.mv(ones_vector))

# log det (A ⊗ B) = N_B log det A + N_A log det B
print("logdet(K):", gaussx.logdet(kronecker_op))

# Verify against dense
print("logdet(dense):", jnp.linalg.slogdet(kronecker_op.as_matrix())[1])

# %% [markdown]
# ## 3. Block diagonal operators
#
# Block diagonal operators act independently on each block.
# All primitives decompose per-block.

# %%
block_a = lx.MatrixLinearOperator(jnp.array([[4.0, 1.0], [1.0, 3.0]]))
block_b = lx.DiagonalLinearOperator(jnp.array([2.0, 5.0, 7.0]))

block_diag_op = gaussx.BlockDiag(block_a, block_b)
# 5
print("BlockDiag size:", block_diag_op.in_size())

rhs = jnp.ones(5)
solution = gaussx.solve(block_diag_op, rhs)
print("solve:", solution)

# log det blockdiag(A₁, A₂) = log det A₁ + log det A₂
print("logdet:", gaussx.logdet(block_diag_op))

# %% [markdown]
# ## 4. Low-rank updates
#
# `LowRankUpdate` represents $\mathbf{A}_0 + \mathbf{U} \mathrm{diag}(\mathbf{d}) \mathbf{V}^\top$ where $\mathbf{A}_0$ is any operator. Solve uses the Woodbury identity, logdet uses the matrix determinant lemma -- both $O(Nk^2 + k^3)$ for rank $k$.
#
# **Woodbury identity** (Hager, 1989):
# $$(\mathbf{A} + \mathbf{U} \mathbf{C} \mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1} \mathbf{U} (\mathbf{C}^{-1} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U})^{-1} \mathbf{V} \mathbf{A}^{-1}$$
#
# **Matrix determinant lemma:**
# $$\log \det (\mathbf{A} + \mathbf{U} \mathbf{C} \mathbf{V}) = \log \det (\mathbf{C}^{-1} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U}) + \log \det \mathbf{C} + \log \det \mathbf{A}$$
#
# These reduce an $N \times N$ solve/logdet to operations on the $k \times k$ capacitance matrix $\mathbf{C}^{-1} + \mathbf{V} \mathbf{A}^{-1} \mathbf{U}$.

# %%
# Diagonal + rank-2 update
diag_values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
low_rank_factor = jnp.array(
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
)

low_rank_op = gaussx.low_rank_plus_diag(diag_values, low_rank_factor)
print("LowRankUpdate rank:", low_rank_op.rank)

vector = jnp.ones(5)
solution = gaussx.solve(low_rank_op, vector)
print("solve:", solution)

# Verify
solution_dense = jnp.linalg.solve(low_rank_op.as_matrix(), vector)
print("max error:", jnp.max(jnp.abs(solution - solution_dense)))

# %% [markdown]
# ## 5. Cholesky and square root
#
# Both preserve structure: `cholesky(Kronecker(A, B))` returns
# `Kronecker(cholesky(A), cholesky(B))`.

# %%
# PSD Kronecker product
factor_a_psd = jnp.array([[2.0, 0.5], [0.5, 3.0]])
factor_b_psd = jnp.array([[4.0, 1.0], [1.0, 2.0]])
kronecker_psd_op = gaussx.Kronecker(
    lx.MatrixLinearOperator(factor_a_psd, lx.positive_semidefinite_tag),
    lx.MatrixLinearOperator(factor_b_psd, lx.positive_semidefinite_tag),
)

cholesky_factor = gaussx.cholesky(kronecker_psd_op)
# Kronecker!
print("cholesky type:", type(cholesky_factor).__name__)

# Verify: L Lᵀ = A ⊗ B
cholesky_matrix = cholesky_factor.as_matrix()
reconstructed = cholesky_matrix @ cholesky_factor.as_matrix().T
print(
    "reconstruction error:",
    jnp.max(jnp.abs(reconstructed - kronecker_psd_op.as_matrix())),
)

# %% [markdown]
# ## 6. Lazy inverse
#
# `inv` returns a new operator that computes $\mathbf{A}^{-1} \mathbf{v}$ via solve on demand.

# %%
diagonal_inv_op = gaussx.inv(diagonal_op)
# Expected: [1, 0.25, 0.111...]
print("D^{-1} diagonal:", gaussx.diag(diagonal_inv_op))

# For structured types, inv preserves structure
kronecker_inv_op = gaussx.inv(kronecker_op)
# Kronecker
print("inv(Kronecker) type:", type(kronecker_inv_op).__name__)

# %% [markdown]
# ## 7. JAX transforms: jit and grad
#
# Everything is compatible with `jit`, `vmap`, and `grad` because
# all operators are equinox modules (PyTrees).
#
# Differentiability is essential for GP hyperparameter optimization: we minimize the negative log marginal likelihood $\mathcal{L}_\text{NLL}(\boldsymbol{\theta}) = -\log p(\mathbf{y}; \boldsymbol{\theta}) = \frac{1}{2} \mathbf{y}^\top \mathbf{K}_{XX}(\boldsymbol{\theta})^{-1} \mathbf{y} + \frac{1}{2} \log \det \mathbf{K}_{XX}(\boldsymbol{\theta}) + \text{const}$ with respect to kernel hyperparameters $\boldsymbol{\theta}$ via gradient descent (Rasmussen & Williams, 2006, Ch. 5). Because gaussx primitives are
# differentiable through JAX, gradients flow through `solve` and `logdet`
# automatically.

# %%
import equinox as eqx


# JIT
@eqx.filter_jit
def neg_log_marginal(operator, obs):
    """Negative log marginal likelihood (up to constants)."""
    # α = K⁻¹ y
    alpha = gaussx.solve(operator, obs)
    # ½ yᵀ α
    data_fit = 0.5 * jnp.dot(obs, alpha)
    # ½ log det K
    complexity = 0.5 * gaussx.logdet(operator)
    # ℒ_NLL = ½ yᵀ K⁻¹ y + ½ log det K
    return data_fit + complexity


obs = jnp.array([1.0, 2.0, 3.0, 4.0])
nll = neg_log_marginal(kronecker_op, obs)
print("neg log marginal:", nll)


# %%
# grad through solve and logdet
def loss(log_diagonal, obs):
    # K = diag(exp(log d))
    operator = lx.DiagonalLinearOperator(jnp.exp(log_diagonal))
    # α = K⁻¹ y
    alpha = gaussx.solve(operator, obs)
    # ℒ_NLL = ½ yᵀ α + ½ log det K
    data_fit = 0.5 * jnp.dot(obs, alpha)
    complexity = 0.5 * gaussx.logdet(operator)
    return data_fit + complexity


log_diagonal = jnp.log(jnp.array([1.0, 2.0, 3.0]))
gradient = jax.grad(loss)(log_diagonal, jnp.ones(3))
print("grad shape:", gradient.shape)
print("gradient:", gradient)

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
rhs_batch = jax.random.normal(jax.random.PRNGKey(0), (5, 3))

solutions = jax.vmap(lambda rhs: gaussx.solve(diagonal_op, rhs))(rhs_batch)
# (5, 3)
print("vmap over vectors:", solutions.shape)

# Verify against manual loop
for i in range(5):
    assert jnp.allclose(solutions[i], gaussx.solve(diagonal_op, rhs_batch[i]))
print("all match ✓")

# %%
# --- vmap over columns (matrix RHS) ---
# gaussx.solve expects a vector, but you can solve A X = B
# by vmapping over the columns of B
size = 4
psd_matrix = jnp.array(
    [
        [2.0, 1.0, 0.5, 0.2],
        [1.0, 3.0, 0.5, 0.1],
        [0.5, 0.5, 4.0, 0.3],
        [0.2, 0.1, 0.3, 2.0],
    ]
)
psd_op = lx.MatrixLinearOperator(psd_matrix, lx.positive_semidefinite_tag)

# 3 right-hand sides
rhs_matrix = jax.random.normal(jax.random.PRNGKey(1), (size, 3))

solution_matrix = jax.vmap(
    lambda column: gaussx.solve(psd_op, column), in_axes=1, out_axes=1
)(rhs_matrix)
# (4, 3)
print("solve matrix RHS:", solution_matrix.shape)

# Verify
solution_matrix_ref = jnp.linalg.solve(psd_matrix, rhs_matrix)
print("max error:", jnp.max(jnp.abs(solution_matrix - solution_matrix_ref)))

# %%
# --- vmap over operators (e.g. hyperparameter sweep) ---
# Solve for different diagonal scalings
scales = jnp.array([0.5, 1.0, 2.0, 4.0])
rhs_fixed = jnp.ones(3)


def solve_scaled(scale):
    # D_s = s D
    operator = lx.DiagonalLinearOperator(scale * diagonal)
    return gaussx.solve(operator, rhs_fixed)


solutions_scaled = jax.vmap(solve_scaled)(scales)
# (4, 3)
print("vmap over operators:", solutions_scaled.shape)
# b / (0.5 d)
print("scale=0.5:", solutions_scaled[0])
# b / (4.0 d)
print("scale=4.0:", solutions_scaled[3])

# %%
# --- vmap over operators AND vectors simultaneously ---
# Common in GP: different kernel matrix per hyperparameter sample


def _make_psd(key, size):
    random_factor = jax.random.normal(key, (size, size))
    # K = A Aᵀ + ½ I
    return random_factor @ random_factor.T + 0.5 * jnp.eye(size)


keys = jax.random.split(jax.random.PRNGKey(2), 4)
matrices = jax.vmap(lambda key: _make_psd(key, 3))(keys)
vectors = jax.random.normal(jax.random.PRNGKey(3), (4, 3))


def solve_one(matrix, rhs):
    operator = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
    return gaussx.solve(operator, rhs)


solutions_pairs = jax.vmap(solve_one)(matrices, vectors)
# (4, 3)
print("vmap over (K, b):", solutions_pairs.shape)


# %%
# --- vmap works with all primitives ---
def all_primitives(matrix):
    operator = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
    return (
        gaussx.logdet(operator),
        gaussx.trace(operator),
        gaussx.diag(operator),
    )


logdets, traces, diagonals = jax.vmap(all_primitives)(matrices)
# (4,)
print("vmapped logdet:", logdets.shape)
# (4,)
print("vmapped trace:", traces.shape)
# (4, 3)
print("vmapped diag:", diagonals.shape)

# %%
# --- vmap with structured operators ---
# Kronecker solve batched over right-hand sides
factor_a_psd = jnp.array([[2.0, 0.5], [0.5, 3.0]])
factor_b_psd = jnp.array([[4.0, 1.0], [1.0, 2.0]])
kronecker_psd_op = gaussx.Kronecker(
    lx.MatrixLinearOperator(factor_a_psd, lx.positive_semidefinite_tag),
    lx.MatrixLinearOperator(factor_b_psd, lx.positive_semidefinite_tag),
)

rhs_batch_kron = jax.random.normal(jax.random.PRNGKey(4), (10, 4))
solutions_kron = jax.vmap(lambda rhs: gaussx.solve(kronecker_psd_op, rhs))(
    rhs_batch_kron
)
# (10, 4)
print("Kronecker batched solve:", solutions_kron.shape)

# Verify against dense
solutions_kron_ref = jnp.linalg.solve(kronecker_psd_op.as_matrix(), rhs_batch_kron.T).T
print("max error:", jnp.max(jnp.abs(solutions_kron - solutions_kron_ref)))

# %% [markdown]
# ## Summary
#
# | Primitive | Diagonal | BlockDiag | Kronecker | LowRankUpdate | Dense |
# |-----------|----------|-----------|-----------|---------------|-------|
# | `solve` | $O(N)$ | per-block | per-factor | Woodbury | lineax |
# | `logdet` | $O(N)$ | sum | scaled sum | det lemma | slogdet |
# | `cholesky` | $O(N)$ | per-block | per-factor | -- | Cholesky |
# | `trace` | $O(N)$ | sum | product | -- | $O(N^2)$ |
# | `inv` | $O(N)$ | per-block | per-factor | -- | lazy |

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
