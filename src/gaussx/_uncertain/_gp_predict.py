"""Uncertain GP predictions."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._uncertain._integrator import AbstractIntegrator
from gaussx._uncertain._types import GaussianState


def kernel_expectations(
    kernel_fn: Callable[[Float[Array, " D"], Float[Array, " D"]], Float[Array, ""]],
    state: GaussianState,
    X_train: Float[Array, "N_train D"],
    integrator: AbstractIntegrator,
) -> tuple[Float[Array, ""], Float[Array, " N_train"], Float[Array, "N_train N_train"]]:
    r"""Compute kernel expectations Psi_0, Psi_1, Psi_2 for uncertain inputs.

    These are the core quantities for GP inference with uncertain inputs::

        Psi_0 = E[k(x, x)]                      scalar
        Psi_1_i = E[k(x, x_i)]                  (N_train,)
        Psi_2_{ij} = E[k(x, x_i) k(x, x_j)]    (N_train, N_train)

    Args:
        kernel_fn: Kernel function ``k(x, x') -> scalar``.
        state: Uncertain input distribution ``x ~ N(mu, Sigma)``.
        X_train: Training points, shape ``(N_train, D)``.
        integrator: Integration method.

    Returns:
        Tuple ``(Psi_0, Psi_1, Psi_2)``.
    """
    from gaussx._uncertain._expectations import mean_expectation

    # Psi_0 = E[k(x, x)]
    Psi_0 = mean_expectation(
        lambda x: jnp.atleast_1d(kernel_fn(x, x)),
        state,
        integrator,
    )[0]

    # Psi_1_i = E[k(x, x_i)]
    def psi1_fn(x: Float[Array, " D"]) -> Float[Array, " N_train"]:
        return jax.vmap(lambda xi: kernel_fn(x, xi))(X_train)

    Psi_1 = mean_expectation(psi1_fn, state, integrator)

    # Psi_2_{ij} = E[k(x, x_i) * k(x, x_j)]
    def psi2_fn(
        x: Float[Array, " D"],
    ) -> Float[Array, "N_train N_train"]:
        k_vec = jax.vmap(lambda xi: kernel_fn(x, xi))(X_train)
        return jnp.outer(k_vec, k_vec)

    from einops import rearrange

    N_train = X_train.shape[0]
    psi2_flat_fn = lambda x: rearrange(psi2_fn(x), "i j -> (i j)")
    Psi_2_flat = mean_expectation(psi2_flat_fn, state, integrator)
    Psi_2 = rearrange(Psi_2_flat, "(i j) -> i j", i=N_train, j=N_train)

    return Psi_0, Psi_1, Psi_2


def uncertain_gp_predict(
    kernel_fn: Callable,
    X_train: Float[Array, "N_train D"],
    alpha: Float[Array, " N_train"],
    K_inv: lx.AbstractLinearOperator,
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Predictive mean and variance for GP with uncertain inputs.

    Uses kernel expectations::

        mu_pred = Psi_1 @ alpha
        var_pred = Psi_0 - tr(K_inv @ Psi_2) + alpha^T @ Psi_2 @ alpha - mu_pred^2

    Args:
        kernel_fn: Kernel function ``k(x, x') -> scalar``.
        X_train: Training points, shape ``(N_train, D)``.
        alpha: Precomputed weights ``K^{-1} y``, shape ``(N_train,)``.
        K_inv: Inverse of training kernel matrix operator.
        state: Uncertain test input ``x ~ N(mu, Sigma)``.
        integrator: Integration method.

    Returns:
        Tuple ``(mean, variance)`` — scalar predictive moments.
    """
    from gaussx._sugar._linalg import trace_product

    Psi_0, Psi_1, Psi_2 = kernel_expectations(kernel_fn, state, X_train, integrator)

    mu_pred = jnp.dot(Psi_1, alpha)

    Psi_2_op = lx.MatrixLinearOperator(Psi_2)
    tr_term = trace_product(K_inv, Psi_2_op)
    quad_term = alpha @ Psi_2 @ alpha

    var_pred = Psi_0 - tr_term + quad_term - mu_pred**2
    var_pred = jnp.clip(var_pred, 0.0)

    return mu_pred, var_pred


def uncertain_gp_predict_mc(
    predict_fn: Callable[
        [Float[Array, " D"]], tuple[Float[Array, ""], Float[Array, ""]]
    ],
    state: GaussianState,
    n_particles: int = 100,
    key: jax.Array | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Monte Carlo GP prediction with uncertain inputs.

    Alternative to analytic kernel expectations when Psi integrals are
    intractable. Uses law of total variance::

        mu = mean(particle_means)
        var = var(particle_means) + mean(particle_vars)

    Args:
        predict_fn: GP predictor mapping ``(D,) -> (mean, var)``.
        state: Uncertain test input ``x ~ N(mu, Sigma)``.
        n_particles: Number of Monte Carlo particles. Default ``100``.
        key: PRNG key. If ``None``, uses ``jax.random.key(0)``.

    Returns:
        Tuple ``(mean, variance)`` — scalar predictive moments.
    """
    mu = state.mean
    Sigma = state.cov.as_matrix()

    if key is None:
        key = jr.key(0)

    # Sample inputs
    L = jnp.linalg.cholesky(Sigma)
    eps = jr.normal(key, (n_particles, mu.shape[0]))
    x_samples = mu[None, :] + eps @ L.T

    # Predict at each sample
    means, variances = jax.vmap(predict_fn)(x_samples)

    # Law of total variance
    pred_mean = jnp.mean(means)
    pred_var = jnp.var(means) + jnp.mean(variances)

    return pred_mean, pred_var
