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
# # Solver Strategies
#
# gaussx decouples *what* you compute (solve, logdet) from *how* it is
# computed (dense factorization, iterative CG, LSMR, etc.) via **solver
# strategies**.
#
# **What you'll learn:**
#
# 1. How solver strategies decouple algorithms from models
# 2. Comparing all 7 full strategies: `DenseSolver`, `CGSolver`, `AutoSolver`,
#    `BBMMSolver`, `PreconditionedCGSolver`, `LSMRSolver`, `ComposedSolver`
# 3. Standalone logdet strategies: `SLQLogdet`, `DenseLogdet`,
#    `IndefiniteSLQLogdet`
# 4. Fine-grained composition via `ComposedSolver`
# 5. When to use which strategy

# %% [markdown]
# ## Background
#
# gaussx separates solve and logdet into **independent protocol axes**:
#
# - **`AbstractSolveStrategy`** -- just `.solve(op, b)`
# - **`AbstractLogdetStrategy`** -- just `.logdet(op)`
# - **`AbstractSolverStrategy`** -- bundles both (inherits from both)
#
# Convenience classes like `CGSolver` implement both methods, but you can
# also use **standalone logdet strategies** (`SLQLogdet`, `DenseLogdet`,
# `IndefiniteSLQLogdet`) and compose them freely via `ComposedSolver`.
#
# For example, a `MultivariateNormal` distribution accepts a strategy
# object and delegates all linear algebra to it. During prototyping you
# might use `DenseSolver`; at scale you switch to `CGSolver` or
# `BBMMSolver` with a single-line change.
#
# **Conjugate Gradients (CG):** For a PSD system $\mathbf{A} \mathbf{x} = \mathbf{b}$, CG minimizes $\Vert \mathbf{x} - \mathbf{x}^\star \Vert_{\mathbf{A}}$ over Krylov subspaces $\mathcal{K}_j(\mathbf{A}, \mathbf{b})$, where $j$ counts CG iterations. After $j$ iterations, the error satisfies
#
# $$\Vert \mathbf{x}^{(j)} - \mathbf{x}^\star \Vert_{\mathbf{A}} \le 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^j \Vert \mathbf{x}^{(0)} - \mathbf{x}^\star \Vert_{\mathbf{A}}$$
#
# where $\kappa = \gamma_{\max} / \gamma_{\min}$ is the condition number (Hestenes & Stiefel, 1952).
#
# **Stochastic Lanczos Quadrature (SLQ):** Approximates $\log \det \mathbf{A} = \mathrm{tr}(\log \mathbf{A})$ using Hutchinson's trace estimator:
#
# $$\log \det \mathbf{A} \approx \frac{1}{S} \sum_{s=1}^S \mathbf{z}_s^\top \log(\mathbf{A}) \mathbf{z}_s$$
#
# where each $\mathbf{z}_s^\top \log(\mathbf{A}) \mathbf{z}_s$ is evaluated via the Lanczos
# decomposition (Ubaru et al., 2017).

# %% [markdown]
# ## Setup

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

# %%
# Create a 100x100 PSD kernel matrix (RBF + noise)
key = jax.random.PRNGKey(0)
size = 100

# A = K_XX + 0.1 I, with k(x, x′) = exp(−½ ‖x − x′‖² / ℓ²) and ℓ = 1
inputs = jnp.linspace(0, 5, size)
sq_dist = (inputs[:, None] - inputs[None, :]) ** 2
matrix = jnp.exp(-0.5 * sq_dist / 1.0**2) + 0.1 * jnp.eye(size)

operator = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
rhs = jax.random.normal(key, (size,))

# True logdet for reference
logdet_true = jnp.linalg.slogdet(matrix)[1]

print(f"Problem size: {size}x{size}")
print(f"True logdet:  {logdet_true:.6f}")

# %% [markdown]
# ## DenseSolver
#
# `DenseSolver` delegates to gaussx structural dispatch (`gaussx.solve`,
# `gaussx.logdet`). It automatically picks the best dense algorithm based
# on operator tags: Cholesky for PSD, block-diagonal factorization for
# `BlockDiag`, Kronecker eigendecomposition for `Kronecker`, etc.

# %%
dense = gaussx.DenseSolver()

solution_dense = dense.solve(operator, rhs)
logdet_dense = dense.logdet(operator)

# max |A x − b|
residual_dense = jnp.max(jnp.abs(operator.mv(solution_dense) - rhs))
print("DenseSolver:")
print(f"  solve residual: {residual_dense:.2e}")
print(f"  logdet:         {logdet_dense:.6f}")
print(f"  logdet error:   {jnp.abs(logdet_dense - logdet_true):.2e}")

# %% [markdown]
# ## CGSolver
#
# `CGSolver` uses lineax CG for the solve and matfree stochastic Lanczos
# quadrature (SLQ) for the log-determinant. It works for any PSD operator,
# even matrix-free ones where `as_matrix()` is unavailable.

# %%
cg = gaussx.CGSolver(rtol=1e-8, atol=1e-8, max_steps=500, num_probes=50)

solution_cg = cg.solve(operator, rhs)
logdet_cg = cg.logdet(operator, key=jax.random.PRNGKey(42))

# max |A x − b|
residual_cg = jnp.max(jnp.abs(operator.mv(solution_cg) - rhs))
print("CGSolver:")
print(f"  solve residual: {residual_cg:.2e}")
print(f"  logdet:         {logdet_cg:.6f}")
print(f"  logdet error:   {jnp.abs(logdet_cg - logdet_true):.2e}")

# %% [markdown]
# ## AutoSolver
#
# `AutoSolver` inspects the operator type and size to automatically select
# the right backend. For structured operators (Diagonal, BlockDiag,
# Kronecker, LowRankUpdate) and small dense matrices it picks `DenseSolver`.
# For large PSD operators it switches to `CGSolver`.

# %%
auto = gaussx.AutoSolver(size_threshold=1000)

solution_auto = auto.solve(operator, rhs)
logdet_auto = auto.logdet(operator)

# max |A x − b|
residual_auto = jnp.max(jnp.abs(operator.mv(solution_auto) - rhs))
print("AutoSolver (size_threshold=1000):")
print(f"  solve residual: {residual_auto:.2e}")
print(f"  logdet:         {logdet_auto:.6f}")
print(f"  logdet error:   {jnp.abs(logdet_auto - logdet_true):.2e}")
print(f"  (N={size} < 1000, so AutoSolver selects DenseSolver internally)")

# %% [markdown]
# ## BBMMSolver
#
# `BBMMSolver` carries the settings of the Black-Box Matrix-Matrix method
# (Gardner et al. 2018). Its `solve` is a CG solve and its `logdet` a
# stochastic Lanczos quadrature (SLQ) estimate, like `CGSolver`.
# `solve_and_logdet` is a convenience that calls the two in turn; it shares no
# matvecs.
#
# The BBMM method (Gardner et al., 2018) was introduced in GPyTorch. It runs
# the right-hand sides and the SLQ probe vectors through one modified batched
# CG (mBCG) pass and reads the Lanczos tridiagonals for the logdet off the CG
# coefficients. In gaussx that shared pass is
# `gaussx.inv_quad_logdet(op, rhs, strategy=BBMMSolver(...))`. It is not
# automatically cheaper: it runs the probe columns to CG convergence rather
# than stopping after `lanczos_order` Lanczos steps, so on a well-conditioned
# kernel it can apply the operator to more columns than a separate solve and
# SLQ.

# %%
bbmm = gaussx.BBMMSolver(
    max_steps=500, rtol=1e-6, atol=1e-6, lanczos_order=50, num_probes=20
)

solution_bbmm = bbmm.solve(operator, rhs)
logdet_bbmm = bbmm.logdet(operator)

# max |A x − b|
residual_bbmm = jnp.max(jnp.abs(operator.mv(solution_bbmm) - rhs))
print("BBMMSolver:")
print(f"  solve residual: {residual_bbmm:.2e}")
print(f"  logdet:         {logdet_bbmm:.6f}")
print(f"  logdet error:   {jnp.abs(logdet_bbmm - logdet_true):.2e}")

# %%
# solve_and_logdet computes both in one call (no shared matvecs)
solution_bbmm_joint, logdet_bbmm_joint = bbmm.solve_and_logdet(operator, rhs)

print("\nBBMMSolver.solve_and_logdet:")
print(
    f"  solve residual: {jnp.max(jnp.abs(operator.mv(solution_bbmm_joint) - rhs)):.2e}"
)
print(f"  logdet:         {logdet_bbmm_joint:.6f}")

# %% [markdown]
# ## PreconditionedCGSolver
#
# `PreconditionedCGSolver` builds a low-rank partial Cholesky preconditioner via matfree, then runs preconditioned CG. For operators of the form $\mathbf{K} + \sigma^2 \mathbf{I}$ (common in GP regression), preconditioning dramatically reduces the number of CG iterations required for convergence.
#
# Preconditioning transforms the system to $\mathbf{M}^{-1} \mathbf{A} \mathbf{x} = \mathbf{M}^{-1} \mathbf{b}$ where $\mathbf{M} \approx \mathbf{A}$ is cheap to invert. For GP kernels of the form $\mathbf{K} + \sigma^2 \mathbf{I}$, a rank-$r$ partial Cholesky approximation $\mathbf{R} \mathbf{R}^\top \approx \mathbf{K}$ gives $\mathbf{M} = \mathbf{R} \mathbf{R}^\top + \sigma^2 \mathbf{I}$, which can be inverted via the Woodbury identity in $O(Nr^2)$. This dramatically improves the effective condition number.
#
# The `shift` parameter should be set to the noise variance $\sigma^2$.

# %%
precond_cg = gaussx.PreconditionedCGSolver(
    preconditioner_rank=20, shift=0.1, rtol=1e-8, atol=1e-8
)

solution_pcg = precond_cg.solve(operator, rhs)
logdet_pcg = precond_cg.logdet(operator)

# max |A x − b|
residual_pcg = jnp.max(jnp.abs(operator.mv(solution_pcg) - rhs))
print("PreconditionedCGSolver:")
print(f"  solve residual: {residual_pcg:.2e}")
print(f"  logdet:         {logdet_pcg:.6f}")
print(f"  logdet error:   {jnp.abs(logdet_pcg - logdet_true):.2e}")

# %% [markdown]
# ## LSMRSolver
#
# `LSMRSolver` implements the LSMR algorithm (Fong & Saunders 2011). Unlike
# CG, LSMR works for **rectangular** and **ill-conditioned** systems too. It only requires matvec and transpose-matvec, and supports Tikhonov regularization via the `damp` parameter (minimizes $\Vert \mathbf{A} \mathbf{x} - \mathbf{b} \Vert_2^2 + \beta \Vert \mathbf{x} \Vert_2^2$ with $\beta = \mathrm{damp}^2$).
#
# LSMR (Fong & Saunders, 2011) is mathematically equivalent to MINRES applied to the normal equations $\mathbf{A}^\top \mathbf{A} \mathbf{x} = \mathbf{A}^\top \mathbf{b}$, but is numerically more stable. It is the natural choice when $\mathbf{A}$ is rectangular or when Tikhonov regularization ($\min \Vert \mathbf{A} \mathbf{x} - \mathbf{b} \Vert_2^2 + \beta \Vert \mathbf{x} \Vert_2^2$) is desired.

# %%
lsmr = gaussx.LSMRSolver(atol=1e-8, btol=1e-8, max_steps=500)

solution_lsmr = lsmr.solve(operator, rhs)
logdet_lsmr = lsmr.logdet(operator)

# max |A x − b|
residual_lsmr = jnp.max(jnp.abs(operator.mv(solution_lsmr) - rhs))
print("LSMRSolver:")
print(f"  solve residual: {residual_lsmr:.2e}")
print(f"  logdet:         {logdet_lsmr:.6f}")
print(f"  logdet error:   {jnp.abs(logdet_lsmr - logdet_true):.2e}")

# %% [markdown]
# ## ComposedSolver
#
# `ComposedSolver` lets you mix-and-match solve and logdet from different
# strategies. It accepts **fine-grained protocols**: any
# `AbstractSolveStrategy` for the solve side and any
# `AbstractLogdetStrategy` for the logdet side.
#
# This means you can pair a full solver (like `DenseSolver`) with a
# **standalone logdet strategy** (like `SLQLogdet`) -- no need to wrap
# the logdet in a full solver class.

# %%
composed = gaussx.ComposedSolver(
    solve_strategy=gaussx.DenseSolver(),
    logdet_strategy=gaussx.SLQLogdet(num_probes=50, lanczos_order=50),
)

solution_composed = composed.solve(operator, rhs)
logdet_composed = composed.logdet(operator, key=jax.random.PRNGKey(99))

# max |A x − b|
residual_composed = jnp.max(jnp.abs(operator.mv(solution_composed) - rhs))
print("ComposedSolver (DenseSolver solve + SLQLogdet):")
print(f"  solve residual: {residual_composed:.2e}")
print(f"  logdet:         {logdet_composed:.6f}")
print(f"  logdet error:   {jnp.abs(logdet_composed - logdet_true):.2e}")

# %% [markdown]
# Notice the solve residual matches DenseSolver (exact) while the logdet
# uses stochastic Lanczos quadrature directly. This is the main use case:
# exact solve for stable gradients, stochastic logdet where noise is
# acceptable.

# %% [markdown]
# ## Standalone Logdet Strategies
#
# gaussx provides three standalone logdet strategies that implement
# `AbstractLogdetStrategy`:
#
# - **`SLQLogdet`** -- stochastic Lanczos quadrature for PSD operators
# - **`IndefiniteSLQLogdet`** -- SLQ with $\log \vert \gamma \vert$ for symmetric indefinite operators, with optional diagonal shift
# - **`DenseLogdet`** -- delegates to `gaussx.logdet` (structural dispatch)
#
# These are the building blocks that `CGSolver`, `BBMMSolver`, etc. use
# internally for their logdet computation. By extracting them as
# standalone strategies, you can compose them freely.

# %%
# SLQLogdet: stochastic logdet for PSD operators
slq = gaussx.SLQLogdet(num_probes=50, lanczos_order=50)
logdet_slq = slq.logdet(operator, key=jax.random.PRNGKey(0))

# DenseLogdet: exact logdet via structural dispatch
dense_logdet = gaussx.DenseLogdet()
logdet_exact = dense_logdet.logdet(operator)

# IndefiniteSLQLogdet: handles indefinite matrices
indefinite_slq = gaussx.IndefiniteSLQLogdet(num_probes=50, lanczos_order=50)
logdet_indef = indefinite_slq.logdet(operator, key=jax.random.PRNGKey(0))

print("Standalone logdet strategies:")
slq_error = jnp.abs(logdet_slq - logdet_true)
exact_error = jnp.abs(logdet_exact - logdet_true)
indefinite_error = jnp.abs(logdet_indef - logdet_true)
print(f"  SLQLogdet:           {logdet_slq:.6f}  (err {slq_error:.2e})")
print(f"  DenseLogdet:         {logdet_exact:.6f}  (err {exact_error:.2e})")
print(f"  IndefiniteSLQLogdet: {logdet_indef:.6f}  (err {indefinite_error:.2e})")

# %% [markdown]
# ## Comparison Table
#
# Collect all results and compare side by side.

# %%
results = {
    "DenseSolver": {"residual": residual_dense, "logdet": logdet_dense},
    "CGSolver": {"residual": residual_cg, "logdet": logdet_cg},
    "AutoSolver": {"residual": residual_auto, "logdet": logdet_auto},
    "BBMMSolver": {"residual": residual_bbmm, "logdet": logdet_bbmm},
    "PrecondCG": {"residual": residual_pcg, "logdet": logdet_pcg},
    "LSMRSolver": {"residual": residual_lsmr, "logdet": logdet_lsmr},
    "ComposedSolver": {"residual": residual_composed, "logdet": logdet_composed},
}

print(f"{'Strategy':<20s}  {'Residual':>12s}  {'Logdet':>12s}  {'Logdet Err':>12s}")
print("-" * 62)
for name, stats in results.items():
    logdet_error = jnp.abs(stats["logdet"] - logdet_true)
    residual = stats["residual"]
    logdet_value = stats["logdet"]
    print(f"{name:<20s}  {residual:12.2e}  {logdet_value:12.6f}  {logdet_error:12.2e}")

# %% [markdown]
# | Strategy | Solve Residual | Logdet | Logdet Error vs True |
# |----------|---------------|--------|---------------------|
# | DenseSolver | ~1e-15 | exact | ~1e-13 |
# | CGSolver | ~1e-9 | stochastic | ~1e-1 |
# | AutoSolver | ~1e-15 | exact | ~1e-13 |
# | BBMMSolver | ~1e-8 | stochastic | ~1e-1 |
# | PrecondCG | ~1e-9 | stochastic | ~1e-1 |
# | LSMRSolver | ~1e-8 | stochastic | ~1e-1 |

# %% [markdown]
# ## Comparison Figure
#
# Bar chart comparing solve residuals and logdet errors across all
# strategies.

# %%
names = list(results.keys())
residuals = [float(results[name]["residual"]) for name in names]
logdet_errors = [
    float(jnp.abs(results[name]["logdet"] - logdet_true)) for name in names
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#795548"]

# Solve residuals (log scale)
axes[0].bar(names, residuals, color=colors)
axes[0].set_yscale("log")
axes[0].set_ylabel("Max solve residual")
axes[0].set_title("Solve accuracy")
axes[0].tick_params(axis="x", rotation=30)
axes[0].grid(True, which="major", alpha=0.3)
axes[0].grid(True, which="minor", alpha=0.1)
axes[0].minorticks_on()

# Logdet errors (log scale)
# Clamp zeros to a small value for log scale
logdet_errors_plot = [max(e, 1e-16) for e in logdet_errors]
axes[1].bar(names, logdet_errors_plot, color=colors)
axes[1].set_yscale("log")
axes[1].set_ylabel("|logdet - true logdet|")
axes[1].set_title("Log-determinant accuracy")
axes[1].tick_params(axis="x", rotation=30)
axes[1].grid(True, which="major", alpha=0.3)
axes[1].grid(True, which="minor", alpha=0.1)
axes[1].minorticks_on()

fig.suptitle(f"Solver strategy comparison (N={size})", fontsize=14)
plt.tight_layout()

plt.show()

# %% [markdown]
# ## When to Use Which
#
# ### Full solver strategies (solve + logdet)
#
# | Strategy | Best for | Solve | Logdet |
# |----------|----------|-------|--------|
# | `DenseSolver` | Small-medium, structured | Exact | Exact |
# | `CGSolver` | Large PSD, matrix-free | CG | SLQ |
# | `AutoSolver` | Don't want to choose | Auto | Auto |
# | `BBMMSolver` | Large PSD; mBCG via `inv_quad_logdet` | CG | SLQ |
# | `PrecondCGSolver` | Noisy GP kernels | Precond CG | SLQ |
# | `LSMRSolver` | Rectangular, ill-cond. | LSMR | SLQ |
# | `ComposedSolver` | Mix-and-match | Any | Any |
#
# ### Standalone logdet strategies
#
# | Strategy | Best for |
# |----------|----------|
# | `SLQLogdet` | Large PSD operators |
# | `IndefiniteSLQLogdet` | Symmetric indefinite / shifted |
# | `DenseLogdet` | Small-medium, structured |
#
# **Details:**
#
# - **DenseSolver**: Exploits Kronecker, BlockDiag, Diagonal
#   fast paths via structural dispatch.
# - **CGSolver**: Works without `as_matrix()`; stochastic
#   Lanczos quadrature (SLQ) for logdet.
# - **AutoSolver**: Inspects operator type and size, delegates
#   to DenseSolver or CGSolver automatically.
# - **BBMMSolver**: GPyTorch-style settings; `inv_quad_logdet` runs its
#   shared mBCG pass, while `solve_and_logdet` is a separate solve and SLQ.
# - **PreconditionedCGSolver**: Low-rank Cholesky
#   preconditioner accelerates convergence on $\mathbf{K} + \sigma^2 \mathbf{I}$.
# - **LSMRSolver**: Supports Tikhonov damping; only needs
#   matvec and transpose-matvec.
# - **ComposedSolver**: Delegates solve and logdet to two
#   independent strategies; pair exact solve with stochastic logdet, etc.
# - **SLQLogdet / IndefiniteSLQLogdet / DenseLogdet**: Standalone logdet
#   strategies usable with `ComposedSolver` for fine-grained control.

# %% [markdown]
# ## Summary
#
# - **Solver strategies** decouple algorithm choice from model code. Swap
#   a single strategy object to change how `solve` and `logdet` are computed.
# - **Fine-grained protocols** (`AbstractSolveStrategy`,
#   `AbstractLogdetStrategy`) let you compose solve and logdet independently
#   via `ComposedSolver`.
# - **Standalone logdet strategies** (`SLQLogdet`, `IndefiniteSLQLogdet`,
#   `DenseLogdet`) can be used directly with `ComposedSolver` for
#   maximum flexibility -- no need to wrap logdet in a full solver.
# - **`DenseSolver`** is exact and fast for small-medium problems; it
#   leverages gaussx structural dispatch for operators like `Kronecker`,
#   `BlockDiag`, and `LowRankUpdate`.
# - **`CGSolver`** and **`BBMMSolver`** are iterative solvers for large PSD
#   systems. `BBMMSolver` also drives the shared mBCG pass of
#   `gaussx.inv_quad_logdet`.
# - **`PreconditionedCGSolver`** dramatically improves CG convergence on
#   noisy kernel matrices by using a low-rank Cholesky preconditioner.
# - **`LSMRSolver`** handles rectangular and ill-conditioned systems that
#   other PSD-only solvers cannot.
# - **`ComposedSolver`** decouples solve and logdet: pair any solve strategy
#   with any logdet strategy for maximum flexibility.
# - **`AutoSolver`** removes the need to choose: it inspects operator type
#   and size and delegates to the appropriate backend.

# %% [markdown]
# ## References
#
# - Fong, D. C.-L. & Saunders, M. A. (2011). LSMR: An iterative algorithm
#   for sparse least-squares problems. *SIAM J. Scientific Computing*,
#   33(5), 2950--2971.
# - Gardner, J. R., Pleiss, G., Weinberger, K. Q., Bindel, D., &
#   Wilson, A. G. (2018). GPyTorch: Blackbox matrix-matrix Gaussian process
#   inference with GPU acceleration. *Proc. NeurIPS*.
# - Hestenes, M. R. & Stiefel, E. (1952). Methods of conjugate gradients
#   for solving linear systems. *Journal of Research of the National Bureau
#   of Standards*, 49(6), 409--436.
# - Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of
#   $\mathrm{tr}(f(\mathbf{A}))$ via stochastic Lanczos quadrature. *SIAM J. Matrix Analysis*, 38(4), 1075--1099.
