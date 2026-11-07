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
# # Sparse Variational GP with Inducing Points
#
# Sparse GPs approximate the full kernel with $N_{\mathcal{I}} \ll N$ inducing points:
#
# $$\mathbf{K}_{XX} \approx \mathbf{K}_{X\mathcal{I}} \mathbf{K}_{\mathcal{I}\mathcal{I}}^{-1} \mathbf{K}_{\mathcal{I}X}$$
#
# The resulting covariance is $\sigma_\varepsilon^2 \mathbf{I} + \mathbf{U} \mathbf{U}^\top$ — a `LowRankUpdate` in gaussx. We optimize the ELBO with respect to inducing point locations and kernel hyperparameters via `jax.grad`.

# %% [markdown]
# ## Background
#
# Sparse GP methods address the $O(N^3)$ cost of exact GP inference by introducing $N_{\mathcal{I}} \ll N$ inducing inputs $\mathbf{X}_{\mathcal{I}} \in \mathbb{R}^{N_{\mathcal{I}} \times D_x}$. The key idea is to approximate the full GP posterior via a variational distribution that conditions on the inducing values $\mathbf{f}_{\mathcal{I}} = f(\mathbf{X}_{\mathcal{I}})$. The Nystrom approximation gives:
#
# $$\mathbf{K}_{XX} \approx \mathbf{Q}_{XX} = \mathbf{K}_{X\mathcal{I}} \mathbf{K}_{\mathcal{I}\mathcal{I}}^{-1} \mathbf{K}_{\mathcal{I}X}$$
#
# The variational free energy (VFE) framework (Titsias, 2009) optimizes
# a lower bound on the log-marginal likelihood:
#
# $$\mathrm{ELBO}_\text{VFE} = \log \mathcal{N}(\mathbf{y}; \mathbf{0}, \mathbf{Q}_{XX} + \sigma_\varepsilon^2 \mathbf{I}) - \frac{1}{2 \sigma_\varepsilon^2} \mathrm{tr}(\mathbf{K}_{XX} - \mathbf{Q}_{XX})$$
#
# The trace correction penalizes the approximation error, and the bound
# is tight when the inducing points explain all the data.

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
# ## Generate data

# %%
key = jax.random.PRNGKey(123)
k1, k2 = jax.random.split(key)

n_data = 300
noise_std = 0.2

f_true = lambda x: jnp.sin(2 * x) + 0.3 * jnp.cos(5 * x)

x_data = jnp.sort(jax.random.uniform(k1, (n_data,), minval=-4.0, maxval=4.0))
y_data = f_true(x_data) + noise_std * jax.random.normal(k2, (n_data,))

x_plot = jnp.linspace(-4.5, 4.5, 500)

# %% [markdown]
# ## Build Nystrom approximation
#
# The Nystrom factor $\mathbf{U} = \mathbf{K}_{X\mathcal{I}} \mathbf{L}_{\mathcal{I}\mathcal{I}}^{-\top} \in \mathbb{R}^{N \times N_{\mathcal{I}}}$ gives $\mathbf{Q}_{XX} = \mathbf{U} \mathbf{U}^\top$, so the noisy covariance $\sigma_\varepsilon^2 \mathbf{I} + \mathbf{U} \mathbf{U}^\top$ is a rank-$N_{\mathcal{I}}$ update of a diagonal -- exactly a `LowRankUpdate`. This is why gaussx can apply the Woodbury identity for $O(N N_{\mathcal{I}}^2 + N_{\mathcal{I}}^3)$ solve and the matrix determinant lemma for $O(N N_{\mathcal{I}}^2 + N_{\mathcal{I}}^3)$ logdet.

# %%
n_inducing = 12


def rbf(x1, x2, ls, var):
    return var * jnp.exp(-0.5 * (x1[:, None] - x2[None, :]) ** 2 / ls**2)


def build_sparse_gp(x_data, x_inducing, ls, var, noise):
    """Build the sparse GP covariance as a LowRankUpdate."""
    # K_XI  (N, N_I)
    kernel_data_inducing = rbf(x_data, x_inducing, ls, var)
    # K_II  (N_I, N_I)
    kernel_inducing = rbf(x_inducing, x_inducing, ls, var)

    # Nystrom: U = K_XI L_IIᵀ⁻¹ with L_II L_IIᵀ = K_II
    jitter = 1e-5 * jnp.eye(len(x_inducing))
    chol_inducing = jnp.linalg.cholesky(kernel_inducing + jitter)
    nystrom_factor = jax.scipy.linalg.solve_triangular(
        chol_inducing, kernel_data_inducing.T, lower=True
    ).T

    # Σ_y = σ_ε² I + U Uᵀ
    return gaussx.low_rank_plus_diag(noise * jnp.ones(len(x_data)), nystrom_factor)


# Initial setup
x_inducing = jnp.linspace(-3.5, 3.5, n_inducing)
ls, var, noise = 1.0, 1.0, noise_std**2

sigma = build_sparse_gp(x_data, x_inducing, ls, var, noise)
print(f"Data: {n_data}, Inducing: {n_inducing}")
print(f"Operator rank: {sigma.rank}")

# %% [markdown]
# ## ELBO (variational lower bound)
#
# $$\mathrm{ELBO} = \log \mathcal{N}(\mathbf{y}; \mathbf{0}, \sigma_\varepsilon^2 \mathbf{I} + \mathbf{U} \mathbf{U}^\top) - \frac{1}{2 \sigma_\varepsilon^2} \mathrm{tr}(\mathbf{K}_{XX} - \mathbf{Q}_{XX})$$
#
# The first term uses gaussx solve + logdet. The trace correction
# penalizes information lost by the approximation.


# %%
def elbo(log_params, x_data, y_data, x_inducing):
    ls = jnp.exp(log_params[0])
    var = jnp.exp(log_params[1])
    noise = jnp.exp(log_params[2])

    sigma_op = build_sparse_gp(x_data, x_inducing, ls, var, noise)

    # Log-likelihood term
    # α = Σ_y⁻¹ y
    alpha = gaussx.solve(sigma_op, y_data)
    # −½ yᵀ α
    data_fit = -0.5 * jnp.dot(y_data, alpha)
    # −½ log det Σ_y
    complexity = -0.5 * gaussx.logdet(sigma_op)
    # −(N/2) log 2π
    const = -0.5 * len(y_data) * jnp.log(2 * jnp.pi)

    # Trace correction: tr(K_XX − Q_XX) / (2 σ_ε²)
    # K_XX diagonal = σ_k² (for RBF with same point)
    # Q_XX diagonal = diag(U Uᵀ)
    q_diag = jnp.sum(sigma_op.U**2, axis=1)
    # −(1 / 2σ_ε²) tr(K_XX − Q_XX)
    trace_correction = -0.5 * jnp.sum(var - q_diag) / noise

    return data_fit + complexity + const + trace_correction


log_params = jnp.log(jnp.array([ls, var, noise]))
print(f"Initial ELBO: {elbo(log_params, x_data, y_data, x_inducing):.2f}")

# %% [markdown]
# ## Optimize hyperparameters

# %%
grad_fn = jax.jit(jax.grad(lambda p: -elbo(p, x_data, y_data, x_inducing)))

lr = 0.01
for _i in range(100):
    g = grad_fn(log_params)
    # Clip gradients for stability
    g = jnp.clip(g, -10.0, 10.0)
    log_params = log_params - lr * g

ls_opt, var_opt, noise_opt = jnp.exp(log_params)
print(f"Optimized: ls={ls_opt:.3f}, var={var_opt:.3f}, noise={noise_opt:.4f}")
print(f"Final ELBO: {elbo(log_params, x_data, y_data, x_inducing):.2f}")

# %% [markdown]
# ## Predict

# %%
sigma_opt = build_sparse_gp(x_data, x_inducing, ls_opt, var_opt, noise_opt)
# α = Σ_y⁻¹ y  (N,)
alpha_opt = gaussx.solve(sigma_opt, y_data)

# Prediction via inducing points
# K_*I  (N_plot, N_I)
kernel_plot_inducing = rbf(x_plot, x_inducing, ls_opt, var_opt)
# K_II  (N_I, N_I)
kernel_inducing = rbf(x_inducing, x_inducing, ls_opt, var_opt)
# K_XI  (N, N_I)
kernel_data_inducing = rbf(x_data, x_inducing, ls_opt, var_opt)

# w = K_II⁻¹ K_IX α  (N_I,)
weights = jnp.linalg.solve(
    kernel_inducing + 1e-5 * jnp.eye(n_inducing),
    kernel_data_inducing.T @ alpha_opt,
)
# μ_* = K_*I w  (N_plot,)
y_pred = kernel_plot_inducing @ weights

# %% [markdown]
# ## Visualize

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_plot, f_true(x_plot), "k--", lw=1.5, label="True function", zorder=4)
ax.scatter(
    x_data,
    y_data,
    s=30,
    c="C0",
    alpha=0.3,
    edgecolors="k",
    linewidths=0.5,
    label="Data",
    zorder=5,
)
ax.plot(x_plot, y_pred, "C1-", lw=2, label="Sparse GP mean", zorder=3)
ax.scatter(
    x_inducing,
    jnp.full_like(x_inducing, -1.8),
    marker="^",
    s=50,
    c="C2",
    zorder=5,
    label=f"Inducing pts ({n_inducing})",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(
    f"Sparse Variational GP: {n_data} data, {n_inducing} inducing "
    f"(ls={ls_opt:.2f}, var={var_opt:.2f})"
)
ax.legend(fontsize=9)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | Component | gaussx role |
# |-----------|------------|
# | `low_rank_plus_diag` | Build Nystrom covariance |
# | `gaussx.solve` | Woodbury solve for weights |
# | `gaussx.logdet` | Matrix determinant lemma for ELBO |
# | `jax.grad` | Differentiate through everything |

# %% [markdown]
# ## References
#
# - Quinonero-Candela, J. & Rasmussen, C. E. (2005). A unifying view of sparse
#   approximate Gaussian process regression. *JMLR*, 6, 1939--1959.
# - Titsias, M. (2009). Variational learning of inducing variables in sparse
#   Gaussian processes. *Proc. AISTATS*.
# - Hensman, J., Fusi, N., & Lawrence, N. D. (2013). Gaussian processes for
#   big data. *Proc. UAI*.
