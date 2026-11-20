"""Joint draws of a linear-Gaussian state path by backward sampling.

`rts_sample` draws x_1:T from the smoothing distribution p(x_1:T | y_1:T)
of a `kalman_filter` run, forward-filtering backward-sampling (FFBS), with
the smoother gain of `rts_smoother`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from gaussx._einx import einsum, rearrange
from gaussx._linalg._safe_cholesky import safe_cholesky
from gaussx._linalg._symmetrize import symmetrize
from gaussx._ssm._kalman import FilterState, _smoother_gain, _transition_stack
from gaussx._strategies._base import AbstractSolverStrategy


def _psd_root(covariance: Float[Array, "N N"]) -> Float[Array, "N N"]:
    """Cholesky factor of a covariance, jittered if round-off broke it."""
    return safe_cholesky(
        lx.MatrixLinearOperator(covariance, lx.positive_semidefinite_tag)
    )


def rts_sample(
    filter_state: FilterState,
    transition: Float[Array, "*T N N"] | lx.AbstractLinearOperator,
    key: PRNGKeyArray,
    num_samples: int = 1,
    *,
    solver: AbstractSolverStrategy | None = None,
) -> Float[Array, "S T N"]:
    """Joint draws of the state path x_1:T from p(x_1:T | y_1:T).

        x_T ~ 𝒩(μ_x,T|T, Σ_x,T|T)
        x_t | x_t+1 ~ 𝒩(μ_x,t|t + G_t (x_t+1 − μ_x,t+1|t),
                        Σ_x,t|t − G_t Σ_x,t+1|t G_tᵀ)

    with the smoother gain G_t = Σ_x,t|t F_t+1ᵀ Σ_x,t+1|t⁻¹ that
    `rts_smoother` forms. This is forward-filtering backward-sampling
    (Carter and Kohn, 1994; Frühwirth-Schnatter, 1994): the filter has
    already run, and the draws walk backwards from the last filtered state.
    Each draw is a whole path, so a quantity of several steps (a maximum
    over a window, the length of an exceedance, a sum) gets its joint
    uncertainty, which the smoother's marginals cannot give. The per-step
    moments of the draws converge to `rts_smoother`'s.

    It takes what `rts_smoother` takes, the `FilterState` and the same
    time-invariant, time-varying or operator ``transition``, plus a key and
    the number of draws. The process noise enters only through the
    predicted covariances Σ_x,t+1|t in the state, so it is not an argument.
    A step masked off in the filter has μ_x,t|t = μ_x,t|t−1, so the
    recursion passes through it, as the smoother's does.

    The conditional covariance Σ_x,t|t − G_t Σ_x,t+1|t G_tᵀ is a difference
    of two PSD matrices and can lose definiteness to round-off, most often
    in float32 on a nearly deterministic step; its square root is taken
    with `safe_cholesky`, which adds diagonal jitter when the plain
    factorisation fails. The draws are reparameterised: for a fixed key
    they are a differentiable function of the model's parameters.

    Algorithm:
        1. For t = T − 1, …, 1, form G_t and the Cholesky factor L_t of the
           conditional covariance once, from the filtered and predicted
           moments (these do not depend on the draw).
        2. For each draw, set x_T = μ_x,T|T + L_T z_T, then walk backwards,
           x_t = μ_x,t|t + G_t (x_t+1 − μ_x,t+1|t) + L_t z_t, with z_t ~ 𝒩(0, I).

    Cost: one N × N solve and one Cholesky per step, O(T N³), then
    O(S T N²) for the S draws.

    Args:
        filter_state: Output of `kalman_filter`.
        transition: Linear dynamics F_t, as a matrix, a stack over time or
            an operator; the same one passed to the filter.
        key: PRNG key for the draws.
        num_samples: Number of joint draws S.
        solver: Solver strategy for Σ_x,t+1|t; ``None`` uses structural
            dispatch.

    Returns:
        The draws, shaped ``(S, T, N)``, row ``t`` of each holding the state
        observed by ``observations[t]``, as in `FilterState`.

    Raises:
        TypeError: If an operator-typed ``transition`` has a time axis.
        ValueError: If ``transition`` is neither a matrix nor a stack.

    Examples:
        Draw paths of a local-level model with its second step missing:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import gaussx
        >>> transition = jnp.eye(1)
        >>> state = gaussx.kalman_filter(
        ...     transition, jnp.eye(1), 0.1 * jnp.eye(1), 0.5 * jnp.eye(1),
        ...     jnp.array([[0.1], [0.0], [0.2], [0.4]]), jnp.zeros(1), jnp.eye(1),
        ...     mask=jnp.array([True, False, True, True]),
        ... )
        >>> paths = gaussx.rts_sample(state, transition, jr.key(0), num_samples=8)
        >>> paths.shape
        (8, 4, 1)

        The last step of every draw comes from the last filtered moments,
        and the mean of many draws approaches the smoothed means:

        >>> many = gaussx.rts_sample(state, transition, jr.key(1), num_samples=4000)
        >>> means, _ = gaussx.rts_smoother(state, transition)
        >>> bool(jnp.allclose(many.mean(axis=0), means, atol=0.05))
        True
    """
    num_steps, state_size = filter_state.filtered_means.shape
    dtype = filter_state.filtered_means.dtype
    if num_steps == 0:
        return jnp.zeros((num_samples, 0, state_size), dtype=dtype)
    transitions = _transition_stack(transition, num_steps)

    def gain_and_root(filtered_covariance, predicted_covariance_next, transition_next):
        gain = _smoother_gain(
            filtered_covariance, predicted_covariance_next, transition_next, solver
        )
        # G_t Σ_x,t+1|t G_tᵀ
        explained = einsum(
            einsum(
                gain, predicted_covariance_next, "state next, next col -> state col"
            ),
            gain,
            "state col, other col -> state other",
        )
        # Σ_x,t|t − G_t Σ_x,t+1|t G_tᵀ, the covariance of x_t given x_t+1
        conditional = symmetrize(filtered_covariance - explained)
        return gain, _psd_root(conditional)

    # Step 1: the gains and conditional roots for t = 1, …, T − 1, which the
    # draws share. ``transitions[t + 1]`` maps step t to step t + 1.
    gains, roots = jax.vmap(gain_and_root)(
        filter_state.filtered_covs[:-1],
        filter_state.predicted_covs[1:],
        transitions[1:],
    )
    last_root = _psd_root(filter_state.filtered_covs[-1])

    def one_path(path_key):
        last_key, steps_key = jr.split(path_key)
        noise_last = jr.normal(last_key, (state_size,), dtype=dtype)
        noise = jr.normal(steps_key, (num_steps - 1, state_size), dtype=dtype)
        # x_T = μ_x,T|T + L_T z_T
        last = filter_state.filtered_means[-1] + last_root @ noise_last

        def step(next_state, inputs):
            filtered_mean, predicted_mean_next, gain, root, noise_t = inputs
            # x_t = μ_x,t|t + G_t (x_t+1 − μ_x,t+1|t) + L_t z_t
            state = (
                filtered_mean
                + gain @ (next_state - predicted_mean_next)
                + root @ noise_t
            )
            return state, state

        inputs = (
            filter_state.filtered_means[:-1][::-1],
            filter_state.predicted_means[1:][::-1],
            gains[::-1],
            roots[::-1],
            noise,
        )
        _, earlier_rev = jax.lax.scan(step, last, inputs)
        return jnp.concatenate([earlier_rev[::-1], rearrange(last, "n -> 1 n")])

    return jax.vmap(one_path)(jr.split(key, num_samples))
