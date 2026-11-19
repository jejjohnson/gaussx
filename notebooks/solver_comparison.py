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
# # Solver Strategy Comparison
#
# gaussx provides two solver strategies that pair `solve` + `logdet`:
#
# - **DenseSolver** — structural dispatch (Cholesky for PSD, etc.)
# - **CGSolver** — iterative CG solve + stochastic Lanczos logdet
#
# This notebook compares them on the same problem.
#
# For small-to-medium problems ($N \lesssim 5000$), direct factorization
# (Cholesky for PSD systems) is optimal: $O(N^3/3)$ flops with
# machine-precision accuracy. For larger problems, iterative methods like
# conjugate gradients (CG) achieve useful accuracy in far fewer flops,
# especially when the matrix is well-conditioned or a good preconditioner
# is available.

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
# ## Setup: PSD kernel matrix

# %%
key = jax.random.PRNGKey(0)
size = 50

# RBF kernel + noise
# A = K_XX + 0.1 I, with k(x, x′) = exp(−½ ‖x − x′‖² / ℓ²) and ℓ = 1
inputs = jnp.linspace(0, 5, size)
sq_dist = (inputs[:, None] - inputs[None, :]) ** 2
matrix = jnp.exp(-0.5 * sq_dist / 1.0**2) + 0.1 * jnp.eye(size)

operator = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
rhs = jax.random.normal(key, (size,))

print(f"Problem size: {size}x{size}")

# %% [markdown]
# ## DenseSolver

# %%
dense = gaussx.DenseSolver()

solution_dense = dense.solve(operator, rhs)
logdet_dense = dense.logdet(operator)

# max |A x − b|
print("DenseSolver:")
print(f"  solve residual: {jnp.max(jnp.abs(operator.mv(solution_dense) - rhs)):.2e}")
print(f"  logdet: {logdet_dense:.6f}")

# %% [markdown]
# ## CGSolver
#
# The CGSolver pairs conjugate gradients for the linear solve with
# stochastic Lanczos quadrature (SLQ) for the log-determinant. SLQ
# exploits the identity $\log \det \mathbf{A} = \mathrm{tr}(\log \mathbf{A})$ and then uses Hutchinson's trace estimator: draw random probe vectors $\mathbf{z}$ and approximate $\mathrm{tr}(\log \mathbf{A}) \approx \frac{1}{P} \sum_{p=1}^P \mathbf{z}_p^\top \log(\mathbf{A}) \mathbf{z}_p$. Each quadratic form $\mathbf{z}^\top \log(\mathbf{A}) \mathbf{z}$ is evaluated via a short
# Lanczos decomposition, which produces a tridiagonal matrix whose
# eigenvalues give accurate Gauss quadrature nodes for the spectral
# integral. See Ubaru et al. (2017) for convergence analysis.

# %%
cg = gaussx.CGSolver(rtol=1e-8, atol=1e-8, max_steps=200, num_probes=50)

solution_cg = cg.solve(operator, rhs)
logdet_cg = cg.logdet(operator, key=jax.random.PRNGKey(42))

# max |A x − b|
print("CGSolver:")
print(f"  solve residual: {jnp.max(jnp.abs(operator.mv(solution_cg) - rhs)):.2e}")
print(f"  logdet: {logdet_cg:.6f}")

# %% [markdown]
# ## Comparison

# %%
print(f"Solve difference: {jnp.max(jnp.abs(solution_dense - solution_cg)):.2e}")
print(f"Logdet difference: {jnp.abs(logdet_dense - logdet_cg):.4f}")

# True logdet for reference
logdet_true = jnp.linalg.slogdet(matrix)[1]
print(f"\nTrue logdet:    {logdet_true:.6f}")
dense_error = jnp.abs(logdet_dense - logdet_true)
print(f"Dense logdet:   {logdet_dense:.6f}  (error: {dense_error:.2e})")
cg_error = jnp.abs(logdet_cg - logdet_true)
print(f"CG logdet:      {logdet_cg:.6f}  (error: {cg_error:.4f})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Solve comparison
axes[0].plot(solution_dense, "C0-", lw=2, label="DenseSolver", alpha=0.8)
axes[0].plot(solution_cg, "C1--", lw=2, label="CGSolver", alpha=0.8)
axes[0].set_xlabel("Index")
axes[0].set_ylabel("Solution")
axes[0].set_title("Solve comparison")
axes[0].legend(fontsize=9)
axes[0].grid(True, which="major", alpha=0.3)
axes[0].grid(True, which="minor", alpha=0.1)
axes[0].minorticks_on()

# Solve difference
axes[1].semilogy(jnp.abs(solution_dense - solution_cg), "C2-")
axes[1].set_xlabel("Index")
axes[1].set_ylabel("|Dense - CG|")
axes[1].set_title("Pointwise solve difference")
axes[1].grid(True, which="major", alpha=0.3)
axes[1].grid(True, which="minor", alpha=0.1)
axes[1].minorticks_on()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## When to use which
#
# | Strategy | Best for | Solve | Logdet |
# |----------|----------|-------|--------|
# | `DenseSolver` | Small-medium, structured | Exact (structural dispatch) | Exact |
# | `CGSolver` | Large PSD, matrix-free | Iterative | Stochastic |
#
# The `DenseSolver` is exact and exploits gaussx structural dispatch
# (Kronecker, BlockDiag, LowRank, Diagonal fast paths).
# The `CGSolver` works for any PSD operator, even matrix-free ones
# where `as_matrix()` is unavailable, but the logdet is approximate.
#
# The crossover point depends on hardware (GPU memory, FLOP throughput) and
# matrix conditioning. On modern GPUs, Cholesky can handle
# $N \sim 10{,}000$--$50{,}000$; beyond that, CG-based methods become
# necessary.

# %% [markdown]
# ## References
#
# - Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*. 4th edition,
#   Johns Hopkins University Press.
# - Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of
#   $\mathrm{tr}(f(\mathbf{A}))$ via stochastic Lanczos quadrature. *SIAM J. Matrix Analysis*, 38(4), 1075--1099.
# - Hestenes, M. R. & Stiefel, E. (1952). Methods of conjugate gradients for
#   solving linear systems. *Journal of Research of the National Bureau of
#   Standards*, 49(6), 409--436.
