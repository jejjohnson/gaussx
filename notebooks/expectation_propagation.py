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
# # Expectation Propagation from Scratch
#
# This notebook implements expectation propagation (EP) for Gaussian
# process classification using gaussx primitives. EP approximates
# non-Gaussian likelihoods with local Gaussian "sites" in natural
# parameter form, iterating a cavity-update-project loop until
# convergence.
#
# **What you'll learn:**
#
# 1. The EP algorithm for GP classification with a probit likelihood
# 2. How `GaussianSites` stores per-observation natural parameters
# 3. Computing cavity distributions and tilted moments
# 4. Damped site updates via `cvi_update_sites`
# 5. How sites produce block-diagonal precision via `sites_to_precision`

# %% [markdown]
# ## 1. Background: EP for GP Classification
#
# In GP classification we place a GP prior over a latent function
# $f(\mathbf{x})$ and connect it to binary labels $y_n$ at the $N$ training inputs ($n = 1, \ldots, N$) via a probit
# likelihood:
#
# $$p(y_n = 1 \mid f_n) = \Phi(f_n)$$
#
# where $f_n = f(\mathbf{x}_n)$ and $\Phi$ is the standard normal CDF. The posterior over $\mathbf{f} \in \mathbb{R}^N$ is:
#
# $$p(\mathbf{f} \mid \mathbf{y}) \propto p(\mathbf{f}) \prod_{n=1}^N p(y_n \mid f_n)$$
#
# This posterior is intractable because the probit factors break
# conjugacy. EP approximates each likelihood factor with a Gaussian
# *site* in natural parameter form:
#
# $$\tilde{t}_n(f_n) \propto \exp \left( \Lambda\mu_{\text{site},n} f_n - \frac{1}{2} \Lambda_{\text{site},n} f_n^2 \right)$$
#
# where $\Lambda\mu_{\text{site},n}$ is the precision-weighted site mean, so that the natural parameters of the site are $(\Lambda\mu_{\text{site},n}, -\frac{1}{2} \Lambda_{\text{site},n})$ with site precision $\Lambda_{\text{site},n}$.
#
# The EP loop repeats three steps for each site $n$:
#
# 1. **Cavity**: Remove site $n$ from the current posterior to get the
#    cavity distribution $q_{\text{cav},n}(f_n)$.
# 2. **Tilt**: Form the tilted distribution
#    $p_{\text{tilt},n}(f_n) \propto p(y_n \mid f_n) q_{\text{cav},n}(f_n)$
#    and compute its mean and variance.
# 3. **Update**: Project the tilted moments back to obtain new site
#    natural parameters, then apply a damped update.

# %% [markdown]
# ## 2. Setup and Data Generation

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
# Generate 1D binary classification data. The true decision boundary
# comes from a latent function $f_\text{true}(x) = \sin(2x)$; labels are drawn
# from $y_n \sim \text{Bernoulli}(\Phi(f_\text{true}(x_n)))$.

# %%
key = jax.random.PRNGKey(42)
# N
num_samples = 30

key, subkey = jax.random.split(key)
train_inputs = jnp.sort(
    jax.random.uniform(subkey, (num_samples,), minval=-3.0, maxval=3.0)
)

# f_true(x) = sin(2x)
f_true = jnp.sin(2.0 * train_inputs)

# Draw binary labels from probit likelihood
key, subkey = jax.random.split(key)
# p(y_n = 1 ∣ f_n) = Φ(f_n)
probs = jax.scipy.stats.norm.cdf(f_true)
labels = jax.random.bernoulli(subkey, probs).astype(jnp.float64)
y_signed = 2.0 * labels - 1.0  # convert to {-1, +1}

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(
    train_inputs[labels == 1],
    labels[labels == 1],
    s=30,
    c="C0",
    edgecolors="k",
    linewidths=0.5,
    label="y = +1",
    zorder=5,
)
ax.scatter(
    train_inputs[labels == 0],
    labels[labels == 0],
    s=30,
    c="C3",
    edgecolors="k",
    linewidths=0.5,
    label="y = -1",
    zorder=5,
)
ax.plot(
    train_inputs,
    jax.scipy.stats.norm.cdf(f_true),
    "k--",
    lw=1.5,
    label=r"$\Phi(f_\mathrm{true})$",
    zorder=4,
)
ax.set(xlabel="$x$", ylabel=r"$y$ / $p(y=1 \mid f_\mathrm{true})$")
ax.legend(fontsize=9)
ax.set_title("Binary classification data (probit likelihood)")
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. GP Prior
#
# We use an RBF (squared exponential) kernel with kernel variance $\sigma_k^2$ and lengthscale $\ell$:
#
# $$k(x, x') = \sigma_k^2 \exp \left( -\frac{(x - x')^2}{2 \ell^2} \right)$$

# %%
lengthscale = 1.0
kernel_variance = 1.0


def rbf_kernel(x1, x2, lengthscale=lengthscale, variance=kernel_variance):
    """Squared exponential kernel."""
    # k(x, x') = σ_k² exp(−(x − x')² / (2 ℓ²))
    sq_dist = (x1[:, None] - x2[None, :]) ** 2
    return variance * jnp.exp(-0.5 * sq_dist / lengthscale**2)


# K_XX ∈ ℝ^{N×N}
gram_train = rbf_kernel(train_inputs, train_inputs)
jitter = 1e-6
# K_XX + jitter I
gram_train = gram_train + jitter * jnp.eye(num_samples)
gram_train_op = lx.MatrixLinearOperator(gram_train)

# μ_f = 0
prior_mean = jnp.zeros(num_samples)

# %% [markdown]
# ## 4. Initialize Gaussian Sites
#
# `GaussianSites(nat1, nat2)` stores per-observation natural parameters.
# For scalar EP (one latent value per site), `nat1` has shape `(N, 1)` and `nat2` has
# shape `(N, 1, 1)`. We initialize both to zero, corresponding to an
# uninformative (improper flat) site.

# %%
nat1_init = jnp.zeros((num_samples, 1))
nat2_init = jnp.zeros((num_samples, 1, 1))
sites = gaussx.GaussianSites(nat1=nat1_init, nat2=nat2_init)

print(f"Sites nat1 shape: {sites.nat1.shape}")
print(f"Sites nat2 shape: {sites.nat2.shape}")

# %% [markdown]
# ## 5. EP Helper Functions
#
# ### Tilted moments via Gauss-Hermite quadrature
#
# The tilted distribution for site $n$ is:
#
# $$p_{\text{tilt},n}(f_n) \propto p(y_n \mid f_n) q_{\text{cav},n}(f_n)$$
#
# where $q_{\text{cav},n}(f_n) = \mathcal{N}(f_n; \mu_{\text{cav},n}, \sigma_{\text{cav},n}^2)$
# is the cavity marginal. We need the mean $\mu_{\text{tilt},n}$ and variance $\sigma_{\text{tilt},n}^2$ of $p_{\text{tilt},n}$.
#
# Using the change of variable $f_n = \mu_{\text{cav},n} + \sigma_{\text{cav},n} z$ with
# $z \sim \mathcal{N}(0, 1)$, we can compute the moments via
# Gauss-Hermite quadrature using `gaussx.gauss_hermite_points`.

# %%
gh_points, gh_weights = gaussx.gauss_hermite_points(order=30, dim=1)
gh_nodes = gh_points[:, 0]  # (P,)

print(f"Gauss-Hermite: {gh_nodes.shape[0]} quadrature points")


def probit_log_lik(f, y_signed_n):
    """Log probit likelihood: log Φ(y_n f_n) with the signed label y_n."""
    return jax.scipy.stats.norm.logcdf(y_signed_n * f)


def tilted_moments(cav_mean_n, cav_var_n, y_signed_n):
    """Compute mean and variance of the tilted distribution via GH quadrature.

    Args:
        cav_mean_n: Cavity mean μ_cav,n (scalar).
        cav_var_n: Cavity variance σ²_cav,n (scalar, positive).
        y_signed_n: Signed label in {-1, +1}.

    Returns:
        (tilt_mean, tilt_var): Mean μ_tilt,n and variance σ²_tilt,n of the
        tilted distribution.
    """
    # σ_cav,n
    cav_std_n = jnp.sqrt(cav_var_n)

    # Transform quadrature points to cavity distribution
    # f = μ_cav,n + σ_cav,n z
    f_pts = cav_mean_n + cav_std_n * gh_nodes  # (P,)

    # Log-weights: GH weights already account for the Gaussian measure,
    # so we only need the log-likelihood part
    log_lik = jax.vmap(lambda f: probit_log_lik(f, y_signed_n))(f_pts)  # (P,)

    # Normalize in log-space for numerical stability
    log_w = jnp.log(gh_weights) + log_lik
    log_z = jax.scipy.special.logsumexp(log_w)
    w_normalized = jnp.exp(log_w - log_z)

    # Tilted moments
    # μ_tilt,n = ∑ₚ w̃ₚ fₚ
    tilt_mean = jnp.sum(w_normalized * f_pts)
    # σ²_tilt,n = ∑ₚ w̃ₚ (fₚ − μ_tilt,n)²
    tilt_var = jnp.sum(w_normalized * (f_pts - tilt_mean) ** 2)

    # Clamp variance to stay positive
    tilt_var = jnp.maximum(tilt_var, 1e-10)

    return tilt_mean, tilt_var


# %% [markdown]
# ## 6. The EP Loop
#
# Each EP iteration:
#
# 1. **Posterior from prior + sites.** The site precision is
#    $\Lambda_{\text{site},n} = -2 [\text{nat2}]_n$ (diagonal), and the
#    posterior precision is $\boldsymbol{\Lambda}_{f \mid y} = \mathbf{K}_{XX}^{-1} + \boldsymbol{\Lambda}_\text{site}$ with the posterior marginals $q(f_n) = \mathcal{N}(\mu_n, \sigma_n^2)$.
#
# 2. **Cavity.** For each $n$, remove site $n$:
#    $\Lambda_{\text{cav},n} = 1/\sigma_n^2 - \Lambda_{\text{site},n}$,
#    $\Lambda\mu_{\text{cav},n} = \mu_n/\sigma_n^2 - \Lambda\mu_{\text{site},n}$.
#
# 3. **Tilted moments.** Compute mean and variance of
#    $p(y_n \mid f_n) q_{\text{cav},n}(f_n)$ by quadrature.
#
# 4. **New site params.** Convert tilted moments to natural parameters: the site precision is
#    $\Lambda_{\text{site},n}^{\text{new}} = 1/\sigma_{\text{tilt},n}^2 - 1/\sigma_{\text{cav},n}^2$
#    and $\Lambda\mu_{\text{site},n}^{\text{new}} = \mu_{\text{tilt},n}/\sigma_{\text{tilt},n}^2 - \mu_{\text{cav},n}/\sigma_{\text{cav},n}^2$.
#
# 5. **Damped update** via `cvi_update_sites`, with step size $\gamma$.

# %%
n_iterations = 15
# γ
step_size = 0.5  # damping factor
log_z_history = []


def compute_posterior(prior_mean, gram_train_op, sites):
    """Compute the Gaussian posterior from prior + sites."""
    # Site precisions: -2 * nat2, shape (N, 1, 1) -> (N,)
    # Λ_site,n = −2 [nat2]_n
    site_precisions = -2.0 * sites.nat2[:, 0, 0]  # (N,)
    site_info = sites.nat1[:, 0]  # (N,)

    # Λ_f|y = K_XX⁻¹ + Λ_site
    gram_inverse = gaussx.inv(gram_train_op).as_matrix()
    post_prec_mat = gram_inverse + jnp.diag(site_precisions)
    post_prec_op = lx.MatrixLinearOperator(post_prec_mat)

    # Posterior covariance and mean
    # Σ_f|y = Λ_f|y⁻¹
    post_cov_mat = gaussx.inv(post_prec_op).as_matrix()
    post_cov_op = lx.MatrixLinearOperator(post_cov_mat)

    # Posterior natural params: Λμ_f|y = K_XX⁻¹ μ_f + Λμ_site
    prior_info = gaussx.solve(gram_train_op, prior_mean)
    post_info = prior_info + site_info

    # μ_f|y = Σ_f|y Λμ_f|y
    post_mean = post_cov_op.mv(post_info)

    return post_mean, post_cov_op, post_prec_mat


for iteration in range(n_iterations):
    # Step 1: compute current posterior
    post_mean, post_cov_op, post_prec_mat = compute_posterior(
        prior_mean, gram_train_op, sites
    )
    post_var = jnp.diag(post_cov_op.as_matrix())  # marginal variances

    # Current site params as flat vectors
    site_precisions = -2.0 * sites.nat2[:, 0, 0]  # (N,)
    site_info = sites.nat1[:, 0]  # (N,)

    # Step 2-4: for each site, compute cavity -> tilted -> new site
    new_nat1 = jnp.zeros(num_samples)
    new_nat2 = jnp.zeros(num_samples)
    log_z_sum = 0.0

    for n in range(num_samples):
        # Cavity distribution (scalar shortcut)
        # Λ_cav,n = 1/σ_n² − Λ_site,n
        cav_prec_n = 1.0 / post_var[n] - site_precisions[n]
        cav_prec_n = jnp.maximum(cav_prec_n, 1e-10)
        # σ²_cav,n = 1/Λ_cav,n
        cav_var_n = 1.0 / cav_prec_n
        # μ_cav,n = σ²_cav,n (μ_n/σ_n² − Λμ_site,n)
        cav_mean_n = cav_var_n * (post_mean[n] / post_var[n] - site_info[n])

        # Tilted moments
        tilt_mean_n, tilt_var_n = tilted_moments(cav_mean_n, cav_var_n, y_signed[n])

        # New site from moment matching:
        # Λ_site,n = 1/σ²_tilt,n − Λ_cav,n
        site_prec_new_n = 1.0 / tilt_var_n - cav_prec_n
        site_prec_new_n = jnp.maximum(site_prec_new_n, 1e-10)
        # Λμ_site,n = μ_tilt,n/σ²_tilt,n − μ_cav,n Λ_cav,n
        site_info_new_n = tilt_mean_n / tilt_var_n - cav_mean_n * cav_prec_n

        new_nat1 = new_nat1.at[n].set(site_info_new_n)
        new_nat2 = new_nat2.at[n].set(-0.5 * site_prec_new_n)

        # Log normalizer for convergence monitoring
        # Z_n = ∑ₚ wₚ Φ(y_n fₚ)
        z_n = jnp.exp(
            jax.scipy.special.logsumexp(
                jnp.log(gh_weights)
                + jax.vmap(lambda f, y_n=y_signed[n]: probit_log_lik(f, y_n))(
                    cav_mean_n + jnp.sqrt(cav_var_n) * gh_nodes
                )
            )
        )
        log_z_sum = log_z_sum + jnp.log(jnp.maximum(z_n, 1e-30))

    # Step 5: damped update via cvi_update_sites
    target_sites = gaussx.GaussianSites(
        nat1=new_nat1[:, None],
        nat2=new_nat2[:, None, None],
    )
    sites = gaussx.cvi_update_sites(
        sites, target_sites.nat1, target_sites.nat2, step_size
    )

    log_z_history.append(float(log_z_sum))
    print(f"Iteration {iteration + 1:2d}: log Z sum = {log_z_sum:.4f}")

# %% [markdown]
# ## 7. Convergence
#
# The sum of log-normalizer terms $\sum_n \log Z_n$ should stabilize
# as EP converges.

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, n_iterations + 1), log_z_history, "o-", markersize=4)
ax.set(xlabel="EP iteration", ylabel=r"$\sum_n \log Z_n$")
ax.set_title("EP convergence")
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Posterior Predictions
#
# Compute the final EP posterior and plot classification probabilities
# $\Phi(\mu_n / \sqrt{1 + \sigma_n^2})$ with uncertainty bands.

# %%
# Final posterior
post_mean, post_cov_op, post_prec_mat = compute_posterior(
    prior_mean, gram_train_op, sites
)
post_var = jnp.diag(post_cov_op.as_matrix())
post_std = jnp.sqrt(post_var)

# Classification probabilities (integrating out latent f)
# p(y_n = 1) = 𝔼_q(f_n)[Φ(f_n)] = Φ(μ_n / √(1 + σ_n²))
pred_probs = jax.scipy.stats.norm.cdf(post_mean / jnp.sqrt(1.0 + post_var))

# Confidence bands on latent function
f_upper = post_mean + 2.0 * post_std
f_lower = post_mean - 2.0 * post_std

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Top: latent function posterior
ax = axes[0]
ax.fill_between(
    train_inputs,
    f_lower,
    f_upper,
    alpha=0.2,
    color="C0",
    label=r"$\mu_n \pm 2\sigma_n$",
)
ax.plot(train_inputs, post_mean, "C0-", lw=2, label="EP posterior mean", zorder=3)
ax.plot(train_inputs, f_true, "k--", lw=1.5, label="True $f(x)$", zorder=4)
ax.scatter(
    train_inputs[labels == 1],
    jnp.ones_like(train_inputs[labels == 1]) * 1.5,
    c="C0",
    marker="+",
    s=40,
    zorder=5,
)
ax.scatter(
    train_inputs[labels == 0],
    jnp.ones_like(train_inputs[labels == 0]) * -1.5,
    c="C3",
    marker="_",
    s=40,
    zorder=5,
)
ax.set(ylabel="$f(x)$")
ax.legend(fontsize=9)
ax.set_title("EP posterior over latent function")
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

# Bottom: predictive probabilities
ax = axes[1]
ax.plot(train_inputs, pred_probs, "C0-", lw=2, label=r"$p(y=1 \mid x)$", zorder=3)
ax.plot(
    train_inputs,
    jax.scipy.stats.norm.cdf(f_true),
    "k--",
    lw=1.5,
    label="True probability",
    zorder=4,
)
ax.scatter(
    train_inputs[labels == 1],
    labels[labels == 1],
    s=30,
    c="C0",
    edgecolors="k",
    linewidths=0.5,
    zorder=5,
)
ax.scatter(
    train_inputs[labels == 0],
    labels[labels == 0],
    s=30,
    c="C3",
    edgecolors="k",
    linewidths=0.5,
    zorder=5,
)
ax.set(xlabel="$x$", ylabel=r"$p(y = 1 \mid x)$")
ax.legend(fontsize=9)
ax.set_title("Predictive classification probabilities")
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Precision Structure via `sites_to_precision`
#
# The site natural parameters form a diagonal precision contribution.
# `sites_to_precision` wraps these into a `BlockTriDiag` operator with
# zero sub-diagonals --- a block-diagonal structure. In a state-space
# GP model, combining this with the prior's block-tridiagonal precision
# yields a full block-tridiagonal posterior precision, enabling
# $O(N D_x^3)$ inference.

# %%
# Λ_site as a BlockTriDiag
site_precision = gaussx.sites_to_precision(sites)
print(f"Type: {type(site_precision).__name__}")
print(f"Diagonal blocks shape: {site_precision.diagonal.shape}")
print(f"Sub-diagonal blocks shape: {site_precision.sub_diagonal.shape}")

# Verify sub-diagonals are zero (block-diagonal structure)
print(f"Sub-diagonal norm: {jnp.linalg.norm(site_precision.sub_diagonal):.2e}")

# %%
# Visualize the site precisions (diagonal entries)
# Λ_site,n = −2 [nat2]_n
site_prec_values = -2.0 * sites.nat2[:, 0, 0]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(num_samples), site_prec_values, color="C1", alpha=0.7)
ax.set(xlabel="Observation index", ylabel=r"Site precision $\Lambda_{\mathrm{site},n}$")
ax.set_title("EP site precisions")
ax.set_axisbelow(True)
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)
ax.minorticks_on()
plt.tight_layout()
plt.show()

# %% [markdown]
# Sites near the decision boundary have higher precision --- EP
# concentrates its approximation effort where the likelihood is most
# informative.

# %% [markdown]
# ## 10. Using `cavity_distribution` and `newton_update`
#
# For completeness, let us verify the gaussx API functions on the final
# posterior. `cavity_distribution` removes a site from the full
# posterior, and `newton_update` converts derivatives into natural
# parameters.

# %%
# Pick site n = N // 2 (near the middle of the domain)
n = num_samples // 2

# Wrap site n's natural params as lineax operators for cavity_distribution
site_nat1_n = sites.nat1[n, 0]  # scalar
site_nat2_n = -2.0 * sites.nat2[n, 0, 0]  # site precision scalar

# Build full-size vectors/operators for site n
site_nat1_vec = jnp.zeros(num_samples).at[n].set(site_nat1_n)
site_nat2_mat = jnp.diag(jnp.zeros(num_samples).at[n].set(site_nat2_n))
site_nat2_op = lx.MatrixLinearOperator(site_nat2_mat)

cav_mean, cav_cov = gaussx.cavity_distribution(
    post_mean, post_cov_op, site_nat1_vec, site_nat2_op
)

print(f"Cavity mean at n={n}: {cav_mean[n]:.4f}")
print(f"Cavity var  at n={n}: {jnp.diag(cav_cov.as_matrix())[n]:.4f}")

# %%
# newton_update: convert Hessian of log-likelihood to site natural params
# For probit, at the posterior mean:
f_n = post_mean[n]
y_n = y_signed[n]

# Derivatives of log Phi(y*f) w.r.t. f
# ratio = 𝒩(y_n f_n; 0, 1) / Φ(y_n f_n)
pdf_val = jax.scipy.stats.norm.pdf(y_n * f_n)
cdf_val = jax.scipy.stats.norm.cdf(y_n * f_n)
ratio = pdf_val / jnp.maximum(cdf_val, 1e-10)
# ∂ log Φ(y_n f_n) / ∂f_n = y_n ratio
jacobian_n = y_n * ratio
# ∂² log Φ(y_n f_n) / ∂f_n² = −(ratio² + y_n f_n ratio)
hessian_n = -(ratio**2 + y_n * f_n * ratio)

# newton_update expects arrays
mean_arr = jnp.array([f_n])
jac_arr = jnp.array([jacobian_n])
hess_arr = jnp.array([[hessian_n]])

nat1_newton, nat2_newton = gaussx.newton_update(mean_arr, jac_arr, hess_arr)
print(f"Newton nat1: {nat1_newton[0]:.4f}")
print(f"Newton nat2 (= -hessian): {nat2_newton[0, 0]:.4f}")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated EP for GP classification from scratch using
# gaussx building blocks:
#
# | gaussx API | Role in EP |
# |---|---|
# | `GaussianSites(nat1, nat2)` | Store per-site natural parameters |
# | `cvi_update_sites(sites, ...)` | Damped natural parameter update |
# | `sites_to_precision(sites)` | Convert sites to `BlockTriDiag` precision |
# | `cavity_distribution(...)` | Remove a site from the posterior |
# | `newton_update(mean, jac, hess)` | Derivatives to natural parameters |
# | `gauss_hermite_points(order, dim)` | Quadrature for tilted moments |
# | `BlockTriDiag` | Structured block-tridiagonal operator |
#
# Key takeaways:
#
# - EP iterates cavity-tilt-project to approximate non-Gaussian
#   likelihoods with local Gaussian sites.
# - The natural parameterization
#   $(\boldsymbol{\Lambda\mu}, -\frac{1}{2} \boldsymbol{\Lambda})$ makes
#   site addition and removal simple additions in parameter space.
# - `sites_to_precision` produces a `BlockTriDiag` that can be combined
#   with state-space GP priors for scalable temporal inference.
# - Damping (`step_size` $\gamma < 1$) in `cvi_update_sites` is essential for EP
#   stability.
