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

# %%
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt

import gaussx


try:
    _here = Path(__file__).resolve().parent
except NameError:
    _here = Path.cwd()
IMG_DIR = _here.parent / "images" / "solver_comparison"
IMG_DIR.mkdir(parents=True, exist_ok=True)

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Setup: PSD kernel matrix

# %%
key = jax.random.PRNGKey(0)
n = 50

# RBF kernel + noise
x = jnp.linspace(0, 5, n)
sq_dist = (x[:, None] - x[None, :]) ** 2
K = jnp.exp(-0.5 * sq_dist / 1.0**2) + 0.1 * jnp.eye(n)

op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
b = jax.random.normal(key, (n,))

print(f"Problem size: {n}x{n}")

# %% [markdown]
# ## DenseSolver

# %%
dense = gaussx.DenseSolver()

x_dense = dense.solve(op, b)
ld_dense = dense.logdet(op)

print("DenseSolver:")
print(f"  solve residual: {jnp.max(jnp.abs(op.mv(x_dense) - b)):.2e}")
print(f"  logdet: {ld_dense:.6f}")

# %% [markdown]
# ## CGSolver

# %%
cg = gaussx.CGSolver(rtol=1e-8, atol=1e-8, max_steps=200, num_probes=50)

x_cg = cg.solve(op, b)
ld_cg = cg.logdet(op, key=jax.random.PRNGKey(42))

print("CGSolver:")
print(f"  solve residual: {jnp.max(jnp.abs(op.mv(x_cg) - b)):.2e}")
print(f"  logdet: {ld_cg:.6f}")

# %% [markdown]
# ## Comparison

# %%
print(f"Solve difference: {jnp.max(jnp.abs(x_dense - x_cg)):.2e}")
print(f"Logdet difference: {jnp.abs(ld_dense - ld_cg):.4f}")

# True logdet for reference
ld_true = jnp.linalg.slogdet(K)[1]
print(f"\nTrue logdet:    {ld_true:.6f}")
print(f"Dense logdet:   {ld_dense:.6f}  (error: {jnp.abs(ld_dense - ld_true):.2e})")
print(f"CG logdet:      {ld_cg:.6f}  (error: {jnp.abs(ld_cg - ld_true):.4f})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Solve comparison
axes[0].plot(x_dense, "C0-", lw=1.5, label="DenseSolver", alpha=0.8)
axes[0].plot(x_cg, "C1--", lw=1.5, label="CGSolver", alpha=0.8)
axes[0].set_xlabel("Index")
axes[0].set_ylabel("Solution")
axes[0].set_title("Solve comparison")
axes[0].legend()

# Solve difference
axes[1].semilogy(jnp.abs(x_dense - x_cg), "C2-")
axes[1].set_xlabel("Index")
axes[1].set_ylabel("|Dense - CG|")
axes[1].set_title("Pointwise solve difference")

plt.tight_layout()
fig.savefig(IMG_DIR / "comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Solver comparison](../images/solver_comparison/comparison.png)

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
