"""GP predictions at uncertain inputs, from the kernel expectations Ψ₀, Ψ₁ and Ψ₂."""

from __future__ import annotations

from collections.abc import Callable

import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._linalg._linalg import trace_product
from gaussx._primitives._cholesky import cholesky
from gaussx._quadrature._expectations import (
    _points_and_weights_or_none,
    mean_expectation,
)
from gaussx._quadrature._integrator import AbstractIntegrator
from gaussx._quadrature._types import GaussianState


def kernel_expectations(
    kernel_fn: Callable[[Float[Array, " D"], Float[Array, " D"]], Float[Array, ""]],
    state: GaussianState,
    X_train: Float[Array, "N_train D"],
    integrator: AbstractIntegrator,
) -> tuple[Float[Array, ""], Float[Array, " N_train"], Float[Array, "N_train N_train"]]:
    """Kernel expectations Ψ₀, Ψ₁ and Ψ₂ for an uncertain input.

        Ψ₀ = 𝔼[k(x, x)]
        Ψ₁ᵢ = 𝔼[k(x, xᵢ)]
        Ψ₂ᵢⱼ = 𝔼[k(x, xᵢ) k(x, xⱼ)],    x ~ 𝒩(μ_x, Σ_x)

    These are the core quantities for GP inference with uncertain inputs. For
    a point-based rule with points χₚ and mean weights wₚ they are computed
    directly from the kernel evaluations K_pi = k(χₚ, xᵢ):

        Ψ₀ = ∑ₚ wₚ k(χₚ, χₚ)
        Ψ₁ᵢ = ∑ₚ wₚ K_pi
        Ψ₂ᵢⱼ = ∑ₚ wₚ K_pi K_pj

    so no N² × N² output covariance or per-point outer product is ever built.
    Integrators without points (e.g. Taylor) go through `mean_expectation` on
    the flattened outer product.

    Algorithm:
        1. Ask the rule for its points χₚ and mean weights wₚ.
        2. With points, evaluate k on them once and take the weighted sums.
        3. Without points, take `mean_expectation` of k(x, x), of the kernel
           row and of its flattened outer product.

    Args:
        kernel_fn: Kernel k(x, x′), scalar-valued.
        state: Uncertain input 𝒩(μ_x, Σ_x).
        X_train: Training inputs X, whose rows are the xᵢ.
        integrator: Integration method.

    Returns:
        Tuple ``(Ψ₀, Ψ₁, Ψ₂)`` of Ψ statistics.

    Examples:
        A constant kernel has Ψ₀ = 1 and all-ones Ψ₁ and Ψ₂:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> X_train = jnp.zeros((3, 2))
        >>> kernel_fn = lambda x, x_prime: jnp.ones(())
        >>> rule = gaussx.CubatureIntegrator()
        >>> psi_0, psi_1, psi_2 = kernel_expectations(kernel_fn, state, X_train, rule)
        >>> bool(jnp.allclose(psi_0, 1.0)), psi_1.shape, psi_2.shape
        (True, (3,), (3, 3))

        A rule without points returns the same shapes:

        >>> taylor = gaussx.TaylorIntegrator()
        >>> _, psi_1, psi_2 = kernel_expectations(kernel_fn, state, X_train, taylor)
        >>> psi_1.shape, psi_2.shape
        ((3,), (3, 3))
    """

    def kernel_row(x: Float[Array, " D"]) -> Float[Array, " N_train"]:
        def kernel_at_input(x_train: Float[Array, " D"]) -> Float[Array, ""]:
            return kernel_fn(x, x_train)

        # k(x, xᵢ)
        return jax.vmap(kernel_at_input)(X_train)

    def kernel_self(x: Float[Array, " D"]) -> Float[Array, ""]:
        return kernel_fn(x, x)

    # The patterns that an empty X_train reaches keep their axis names: einx
    # prints the pattern in its AxisSizeError.
    rule = _points_and_weights_or_none(integrator, state)
    if rule is not None:
        points, mean_weights, _ = rule
        # k(χₚ, χₚ)
        self_kernels = jax.vmap(kernel_self)(points)
        # K_pi = k(χₚ, xᵢ)
        kernel_rows = jax.vmap(kernel_row)(points)
        # Ψ₀ = ∑ₚ wₚ k(χₚ, χₚ)
        self_kernel_mean = einsum(mean_weights, self_kernels, "p, p ->")
        # Ψ₁ᵢ = ∑ₚ wₚ K_pi
        cross_kernel_mean = einsum(mean_weights, kernel_rows, "p, p i -> i")
        # wₚ K_pi
        weighted_rows = einx.multiply("p, p i -> p i", mean_weights, kernel_rows)
        # Ψ₂ᵢⱼ = ∑ₚ wₚ K_pi K_pj
        kernel_outer_mean = einsum(
            weighted_rows, kernel_rows, "p row, p col -> row col"
        )
        return self_kernel_mean, cross_kernel_mean, kernel_outer_mean

    def vector_kernel_self(x: Float[Array, " D"]) -> Float[Array, " 1"]:
        return jnp.atleast_1d(kernel_fn(x, x))

    # Ψ₀ = 𝔼[k(x, x)]
    self_kernel_expectation = mean_expectation(vector_kernel_self, state, integrator)
    self_kernel_mean = self_kernel_expectation[0]

    # Ψ₁ᵢ = 𝔼[k(x, xᵢ)]
    cross_kernel_mean = mean_expectation(kernel_row, state, integrator)

    train_size = X_train.shape[0]

    def flat_kernel_outer(x: Float[Array, " D"]) -> Float[Array, " NN"]:
        # k(x, xᵢ)
        row = kernel_row(x)
        # k(x, xᵢ) k(x, xⱼ)
        outer = jnp.outer(row, row)
        return rearrange(outer, "i j -> (i j)")

    # Ψ₂ᵢⱼ = 𝔼[k(x, xᵢ) k(x, xⱼ)], flattened
    flat_outer_mean = mean_expectation(flat_kernel_outer, state, integrator)
    kernel_outer_mean = rearrange(
        flat_outer_mean, "(i j) -> i j", i=train_size, j=train_size
    )

    return self_kernel_mean, cross_kernel_mean, kernel_outer_mean


def uncertain_gp_predict(
    kernel_fn: Callable,
    X_train: Float[Array, "N_train D"],
    alpha: Float[Array, " N_train"],
    K_inv: lx.AbstractLinearOperator,
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Predictive mean and variance of a GP at an uncertain input.

        μ_* = Ψ₁ᵀ α
        σ²_* = Ψ₀ − tr(K_XX⁻¹ Ψ₂) + αᵀ Ψ₂ α − μ_*²

    Algorithm:
        1. Take the kernel expectations Ψ₀, Ψ₁ and Ψ₂ under the input.
        2. Contract Ψ₁ with α for the mean.
        3. Assemble the variance and clip it at zero.

    Args:
        kernel_fn: Kernel k(x, x′), scalar-valued.
        X_train: Training inputs X.
        alpha: Precomputed weights α = K_XX⁻¹ y.
        K_inv: Inverse K_XX⁻¹ of the training kernel matrix, as an operator.
        state: Uncertain test input 𝒩(μ_x, Σ_x).
        integrator: Integration method.

    Returns:
        Tuple ``(mean, variance)`` holding the scalar moments μ_* and σ²_*.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> def kernel_fn(x, x_prime):
        ...     return jnp.exp(-0.5 * jnp.sum((x - x_prime) ** 2))
        >>> covariance = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> X_train = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> K_inv = lx.MatrixLinearOperator(jnp.eye(2))
        >>> mean, variance = uncertain_gp_predict(
        ...     kernel_fn,
        ...     X_train,
        ...     jnp.ones(2),
        ...     K_inv,
        ...     state,
        ...     gaussx.CubatureIntegrator(),
        ... )
        >>> mean.shape, bool(variance >= 0.0)
        ((), True)
    """
    self_kernel_mean, cross_kernel_mean, kernel_outer_mean = kernel_expectations(
        kernel_fn, state, X_train, integrator
    )

    # μ_* = Ψ₁ᵀ α
    predictive_mean = jnp.dot(cross_kernel_mean, alpha)

    kernel_outer_op = lx.MatrixLinearOperator(kernel_outer_mean)
    # tr(K_XX⁻¹ Ψ₂)
    trace_term = trace_product(K_inv, kernel_outer_op)
    # αᵀ Ψ₂
    weights_outer = alpha @ kernel_outer_mean
    # αᵀ Ψ₂ α
    quadratic_term = weights_outer @ alpha

    # σ²_* = Ψ₀ − tr(K_XX⁻¹ Ψ₂) + αᵀ Ψ₂ α − μ_*²
    predictive_variance = (
        self_kernel_mean - trace_term + quadratic_term - predictive_mean**2
    )
    predictive_variance = jnp.clip(predictive_variance, 0.0)

    return predictive_mean, predictive_variance


def uncertain_svgp_predict(
    kernel_fn: Callable,
    Z: Float[Array, "M D"],
    alpha: Float[Array, " M"],
    Q: lx.AbstractLinearOperator,
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Predictive mean and variance of an SVGP at an uncertain input.

        μ_* = Ψ₁ᵀ α
        σ²_* = Ψ₀ + tr(Q Ψ₂) + αᵀ Ψ₂ α − μ_*²

    with Ψ taken over the inducing inputs X_ℐ, and
    Q = K_ℐℐ⁻¹ Σ_ℐ K_ℐℐ⁻¹ − K_ℐℐ⁻¹ the variance adjustment operator (see
    `gaussx.svgp_variance_adjustment`). The exact uncertain GP trace correction
    is recovered by setting Q = −K_ℐℐ⁻¹; if X_ℐ equals the training inputs, this
    matches `gaussx.uncertain_gp_predict`.

    Algorithm:
        1. Take the kernel expectations Ψ₀, Ψ₁ and Ψ₂ over X_ℐ.
        2. Contract Ψ₁ with α for the mean.
        3. Assemble the variance and clip it at zero.

    Args:
        kernel_fn: Kernel k(x, x′), scalar-valued.
        Z: Inducing inputs X_ℐ.
        alpha: Effective weights α.
        Q: Variance adjustment operator Q = K_ℐℐ⁻¹ Σ_ℐ K_ℐℐ⁻¹ − K_ℐℐ⁻¹.
        state: Uncertain test input 𝒩(μ_x, Σ_x).
        integrator: Integration method.

    Returns:
        Tuple ``(mean, variance)`` holding the scalar moments μ_* and σ²_*.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> def kernel_fn(x, x_prime):
        ...     return jnp.exp(-0.5 * jnp.sum((x - x_prime) ** 2))
        >>> covariance = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> inducing = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> Q = lx.MatrixLinearOperator(-jnp.eye(2))
        >>> mean, variance = uncertain_svgp_predict(
        ...     kernel_fn, inducing, jnp.ones(2), Q, state, gaussx.CubatureIntegrator()
        ... )
        >>> mean.shape, bool(variance >= 0.0)
        ((), True)
    """
    self_kernel_mean, cross_kernel_mean, kernel_outer_mean = kernel_expectations(
        kernel_fn, state, Z, integrator
    )

    # μ_* = Ψ₁ᵀ α
    predictive_mean = jnp.dot(cross_kernel_mean, alpha)

    kernel_outer_op = lx.MatrixLinearOperator(kernel_outer_mean)
    # tr(Q Ψ₂)
    trace_term = trace_product(Q, kernel_outer_op)
    # αᵀ Ψ₂
    weights_outer = alpha @ kernel_outer_mean
    # αᵀ Ψ₂ α
    quadratic_term = weights_outer @ alpha

    # σ²_* = Ψ₀ + tr(Q Ψ₂) + αᵀ Ψ₂ α − μ_*²
    predictive_variance = (
        self_kernel_mean + trace_term + quadratic_term - predictive_mean**2
    )
    predictive_variance = jnp.clip(predictive_variance, 0.0)

    return predictive_mean, predictive_variance


def uncertain_vgp_predict(
    kernel_fn: Callable,
    X_train: Float[Array, "N_train D"],
    alpha: Float[Array, " N_train"],
    Q: lx.AbstractLinearOperator,
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Predictive mean and variance of a dense VGP at an uncertain input.

        μ_* = Ψ₁ᵀ α
        σ²_* = Ψ₀ + tr(Q Ψ₂) + αᵀ Ψ₂ α − μ_*²

    with Q = K_XX⁻¹ Σ_f K_XX⁻¹ − K_XX⁻¹ and α = K_XX⁻¹ μ_f for the variational
    q(f) = 𝒩(μ_f, Σ_f). The exact uncertain GP is the special case Q = −K_XX⁻¹.
    By contrast, Q = 0 corresponds to Σ_f = K_XX and removes only the trace
    correction.

    Algorithm:
        1. Take the kernel expectations Ψ₀, Ψ₁ and Ψ₂ over X.
        2. Contract Ψ₁ with α for the mean.
        3. Assemble the variance and clip it at zero.

    Args:
        kernel_fn: Kernel k(x, x′), scalar-valued.
        X_train: Training inputs X.
        alpha: Precomputed weights α = K_XX⁻¹ μ_f.
        Q: Variance adjustment operator Q = K_XX⁻¹ Σ_f K_XX⁻¹ − K_XX⁻¹.
        state: Uncertain test input 𝒩(μ_x, Σ_x).
        integrator: Integration method.

    Returns:
        Tuple ``(mean, variance)`` holding the scalar moments μ_* and σ²_*.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> def kernel_fn(x, x_prime):
        ...     return jnp.exp(-0.5 * jnp.sum((x - x_prime) ** 2))
        >>> covariance = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> X_train = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> Q = lx.MatrixLinearOperator(-jnp.eye(2))
        >>> mean, variance = uncertain_vgp_predict(
        ...     kernel_fn, X_train, jnp.ones(2), Q, state, gaussx.CubatureIntegrator()
        ... )
        >>> mean.shape, bool(variance >= 0.0)
        ((), True)
    """
    self_kernel_mean, cross_kernel_mean, kernel_outer_mean = kernel_expectations(
        kernel_fn, state, X_train, integrator
    )

    # μ_* = Ψ₁ᵀ α
    predictive_mean = jnp.dot(cross_kernel_mean, alpha)

    kernel_outer_op = lx.MatrixLinearOperator(kernel_outer_mean)
    # tr(Q Ψ₂)
    trace_term = trace_product(Q, kernel_outer_op)
    # αᵀ Ψ₂
    weights_outer = alpha @ kernel_outer_mean
    # αᵀ Ψ₂ α
    quadratic_term = weights_outer @ alpha

    # σ²_* = Ψ₀ + tr(Q Ψ₂) + αᵀ Ψ₂ α − μ_*²
    predictive_variance = (
        self_kernel_mean + trace_term + quadratic_term - predictive_mean**2
    )
    predictive_variance = jnp.clip(predictive_variance, 0.0)

    return predictive_mean, predictive_variance


def uncertain_bgplvm_predict(
    kernel_fn: Callable,
    X_train: Float[Array, "N_train D"],
    alpha: Float[Array, "N_train D_out"],
    K_inv: lx.AbstractLinearOperator,
    state: GaussianState,
    integrator: AbstractIntegrator,
) -> tuple[Float[Array, " D_out"], Float[Array, " D_out"]]:
    """Multi-output uncertain GP prediction for a Bayesian GP-LVM.

        μ_*,d = Ψ₁ᵀ α_d
        σ²_*,d = Ψ₀ − tr(K_XX⁻¹ Ψ₂) + α_dᵀ Ψ₂ α_d − μ_*,d²

    Maps a latent variable z ~ 𝒩(μ_z, Σ_z) to a high-dimensional reconstruction
    y using per-output GP weights α_d.

    This intentionally uses the exact GP trace term for every output dimension.
    Unlike `gaussx.uncertain_vgp_predict` and `gaussx.uncertain_svgp_predict`,
    there is no separate variational covariance correction operator.

    Algorithm:
        1. Take the kernel expectations Ψ₀, Ψ₁ and Ψ₂ under the latent.
        2. Contract Ψ₁ with each α_d for the means.
        3. Assemble the variances and clip them at zero.

    Args:
        kernel_fn: Kernel k(x, x′), scalar-valued.
        X_train: Training inputs X.
        alpha: Multi-output weights K_XX⁻¹ Y, one column α_d per output.
        K_inv: Inverse K_XX⁻¹ of the training kernel matrix, as an operator.
        state: Uncertain latent input 𝒩(μ_z, Σ_z).
        integrator: Integration method.

    Returns:
        Tuple ``(mean, variance)`` holding μ_*,d and σ²_*,d for each output d.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> def kernel_fn(x, x_prime):
        ...     return jnp.exp(-0.5 * jnp.sum((x - x_prime) ** 2))
        >>> covariance = lx.DiagonalLinearOperator(0.1 * jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> X_train = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        >>> K_inv = lx.MatrixLinearOperator(jnp.eye(2))
        >>> mean, variance = uncertain_bgplvm_predict(
        ...     kernel_fn, X_train, jnp.ones((2, 3)), K_inv, state,
        ...     gaussx.CubatureIntegrator(),
        ... )
        >>> mean.shape, variance.shape
        ((3,), (3,))
    """
    self_kernel_mean, cross_kernel_mean, kernel_outer_mean = kernel_expectations(
        kernel_fn, state, X_train, integrator
    )

    # μ_*,d = Ψ₁ᵀ α_d
    predictive_mean = cross_kernel_mean @ alpha

    kernel_outer_op = lx.MatrixLinearOperator(kernel_outer_mean)
    # tr(K_XX⁻¹ Ψ₂)
    trace_term = trace_product(K_inv, kernel_outer_op)

    # Ψ₂ α_d
    outer_weights = kernel_outer_mean @ alpha
    # α_d ⊙ Ψ₂ α_d
    weighted_outer_weights = alpha * outer_weights
    # α_dᵀ Ψ₂ α_d
    # jnp, not einx: einx cannot resolve zero outputs (D_y = 0), which return
    # empty moments here today.
    quadratic_term = jnp.sum(weighted_outer_weights, axis=0)

    # σ²_*,d = Ψ₀ − tr(K_XX⁻¹ Ψ₂) + α_dᵀ Ψ₂ α_d − μ_*,d²
    predictive_variance = (
        self_kernel_mean - trace_term + quadratic_term - predictive_mean**2
    )
    predictive_variance = jnp.clip(predictive_variance, 0.0)

    return predictive_mean, predictive_variance


def uncertain_gp_predict_mc(
    predict_fn: Callable[
        [Float[Array, " D"]], tuple[Float[Array, ""], Float[Array, ""]]
    ],
    state: GaussianState,
    n_particles: int = 100,
    key: jax.Array | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Monte Carlo GP prediction at an uncertain input, by total variance.

        μ_* = (1 / |ℰ|) ∑ₑ μ_*⁽ᵉ⁾
        σ²_* = 𝕍[μ_*⁽ᵉ⁾] + (1 / |ℰ|) ∑ₑ σ²_*⁽ᵉ⁾

    with (μ_*⁽ᵉ⁾, σ²_*⁽ᵉ⁾) the GP's moments at the draw x⁽ᵉ⁾ ~ 𝒩(μ_x, Σ_x). An
    alternative to analytic kernel expectations when the Ψ integrals are
    intractable.

    Algorithm:
        1. Draw inputs x⁽ᵉ⁾ = μ_x + L z⁽ᵉ⁾ with L Lᵀ = Σ_x.
        2. Evaluate the GP's moments at each draw.
        3. Combine them with the law of total variance.

    Args:
        predict_fn: GP predictor mapping an input to its scalar mean and variance.
        state: Uncertain test input 𝒩(μ_x, Σ_x).
        n_particles: Number of Monte Carlo particles |ℰ|. Default ``100``.
        key: PRNG key. If ``None``, uses ``jax.random.key(0)``.

    Returns:
        Tuple ``(mean, variance)`` holding the scalar moments μ_* and σ²_*.

    Examples:
        A constant predictor has no spread between the particles:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import lineax as lx
        >>> import gaussx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(2))
        >>> state = gaussx.GaussianState(jnp.zeros(2), covariance)
        >>> mean, variance = uncertain_gp_predict_mc(
        ...     lambda x: (jnp.ones(()), 0.5 * jnp.ones(())),
        ...     state,
        ...     n_particles=8,
        ...     key=jr.key(0),
        ... )
        >>> bool(jnp.allclose(mean, 1.0)), bool(jnp.allclose(variance, 0.5))
        (True, True)
    """
    input_mean = state.mean
    input_size = input_mean.shape[0]

    if key is None:
        key = jr.key(0)

    # --- sample inputs ---
    # L Lᵀ = Σ_x
    cholesky_op = cholesky(state.cov)
    cholesky_factor = cholesky_op.as_matrix()
    # z⁽ᵉ⁾ ~ 𝒩(0, I)
    standard_draws = jr.normal(
        key,
        (n_particles, input_size),
        dtype=jnp.result_type(input_mean, cholesky_factor),
    )
    # jnp, not einx: einx cannot resolve no particles (|ℰ| = 0) or an empty state
    # (D_x = 0), which both return here today.
    input_mean_row = input_mean[None, :]
    cholesky_factor_transpose = cholesky_factor.T
    # L z⁽ᵉ⁾
    shifts = standard_draws @ cholesky_factor_transpose
    # x⁽ᵉ⁾ = μ_x + L z⁽ᵉ⁾
    input_samples = input_mean_row + shifts

    # --- predict at each sample ---
    particle_means, particle_variances = jax.vmap(predict_fn)(input_samples)

    # --- law of total variance ---
    # μ_* = (1 / |ℰ|) ∑ₑ μ_*⁽ᵉ⁾
    predictive_mean = jnp.mean(particle_means)
    # σ²_* = 𝕍[μ_*⁽ᵉ⁾] + (1 / |ℰ|) ∑ₑ σ²_*⁽ᵉ⁾
    predictive_variance = jnp.var(particle_means) + jnp.mean(particle_variances)

    return predictive_mean, predictive_variance
