"""Gauss-Markov chain over a state trajectory x_0:T−1 as a NumPyro density.

The chain has T states x_t ∈ ℝ^{D_x}, t = 0, …, T − 1, a prior
x₀ ~ 𝒩(μ_x,0, Σ_x,0) and transitions x_t+1 = F_t x_t + c_t + η_t+1 with
model noise η_t+1 ~ 𝒩(0, Σ_η,t+1). As in `gaussx.udl_from_ssm_params`, F_t
maps x_t to x_t+1, so it is the array ``A[t]``; the offset c_t is ``b[t]``
and Σ_η,t+1 is ``Q[t]``. L_x,0 and L_η,t+1 are the lower Cholesky factors of
Σ_x,0 and Σ_η,t+1, and z ~ 𝒩(0, I) is the standard normal noise of a draw.
The joint precision Λ_x = Σ_x⁻¹ of x_0:T−1 is block tridiagonal.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.distributions.util import lazy_property, validate_sample

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._distributions._utils import _reshape_batch, _reshape_series
from gaussx._einx import einsum, rearrange
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._ssm._pairwise_marginals import pairwise_marginals
from gaussx._ssm._udl import (
    UDLDecomposition,
    udl_decomposition,
    udl_from_ssm_params,
    udl_to_ssm_params,
)


def _chol_log_prob(
    chol: Float[Array, "d d"],
    residual: Float[Array, " d"],
) -> Float[Array, ""]:
    """Zero-mean Gaussian log-density from a lower Cholesky factor L.

        log 𝒩(r; 0, L Lᵀ) = −½ ‖L⁻¹ r‖² − ∑_i log L_ii − ½ D_x log 2π

    Args:
        chol: Lower Cholesky factor L of the covariance L Lᵀ.
        residual: Residual r.

    Returns:
        The scalar log 𝒩(r; 0, L Lᵀ).
    """
    # L⁻¹ r
    whitened_residual = jsl.solve_triangular(chol, residual, lower=True)
    # L_ii
    chol_diagonal = jnp.diagonal(chol)
    # log L_ii
    log_chol_diagonal = jnp.log(chol_diagonal)
    # ½ log det(L Lᵀ) = ∑_i log L_ii, a full reduction
    half_logdet = jnp.sum(log_chol_diagonal)
    # ‖L⁻¹ r‖²
    mahalanobis = whitened_residual @ whitened_residual
    event_size = residual.shape[0]
    # log 𝒩(r; 0, L Lᵀ) = −½ ‖L⁻¹ r‖² − ½ log det(L Lᵀ) − ½ D_x log 2π
    return -0.5 * mahalanobis - half_logdet - 0.5 * event_size * _LOG_2PI


class MarkovGaussian(dist.Distribution):
    """Gauss-Markov chain over the states x_0:T−1 as a density over ``(T, d)``.

        x₀ ~ 𝒩(μ_x,0, Σ_x,0)
        x_t+1 = F_t x_t + c_t + η_t+1,   η_t+1 ~ 𝒩(0, Σ_η,t+1)

    for t = 0, …, T − 2 and states x_t ∈ ℝ^{D_x}. This generative
    state-space form is the canonical parameterisation; the joint
    covariance Σ_x of x_0:T−1 has a block-tridiagonal inverse Λ_x.
    ``event_shape`` is ``(T, d)``, so ``log_prob`` of a trajectory is a
    scalar and the distribution can be used directly as a NumPyro site: a
    prior over a latent path, or a structured variational guide. Every view
    is derived from the state-space parameters on demand:

    | View | Cost | Route |
    |---|---|---|
    | `marginals`, `mean`, `variance` | O(T D_x³) | forward moment propagation |
    | `sample`, `log_prob` | O(T D_x³) | chain factorisation, no joint Cholesky |
    | `precision`, `to_precision_form` | O(T D_x³) | `udl_from_ssm_params` |
    | `from_precision_form` | O(T D_x³) | `udl_decomposition` |
    | `covariance_matrix` | O(T² D_x³) | banded solves against I; small T |

    Where `LGSSM` is a density over the *observations* of a chain with
    the states marginalised out, this is the density over the *states*
    themselves. Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).

    Args:
        A: Transition matrices F_t, with ``A[t]`` mapping x_t to x_t+1.
        Q: Model-noise covariances Σ_η,t+1, with ``Q[t]`` driving the step
            from x_t to x_t+1.
        mu0: Prior mean μ_x,0 of the initial state x₀.
        P0: Prior covariance Σ_x,0 of x₀.
        b: Optional transition offsets c_t, with ``b[t]`` added in the step
            from x_t to x_t+1. Defaults to zero.
        validate_args: Whether to validate input arguments.

    Raises:
        ValueError: If ``A`` is not a stack of square matrices, or ``Q``,
            ``mu0``, ``P0`` or ``b`` does not match its shape.

    Examples:
        A ten-step chain in two dimensions:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> T, d = 10, 2
        >>> A = jnp.broadcast_to(0.9 * jnp.eye(d), (T - 1, d, d))
        >>> Q = jnp.broadcast_to(0.1 * jnp.eye(d), (T - 1, d, d))
        >>> chain = gaussx.MarkovGaussian(A, Q, jnp.zeros(d), jnp.eye(d))
        >>> xs = chain.sample(jax.random.key(0), (4,))
        >>> xs.shape, chain.log_prob(xs).shape
        ((4, 10, 2), (4,))

        The precision form round-trips to the same chain:

        >>> mean, precision = chain.to_precision_form()
        >>> mean.shape, precision.as_matrix().shape
        ((20,), (20, 20))
        >>> rebuilt = gaussx.MarkovGaussian.from_precision_form(mean, precision)
        >>> bool(jnp.allclose(rebuilt.A, A, atol=1e-4))
        True

        A single state has no transitions:

        >>> single = gaussx.MarkovGaussian(A[:0], Q[:0], jnp.zeros(d), jnp.eye(d))
        >>> single.event_shape, single.cross_covariances().shape
        ((1, 2), (0, 2, 2))
    """

    arg_constraints = {}  # noqa: RUF012
    support = dist.constraints.real_matrix
    reparametrized_params = ["A", "b", "Q", "mu0", "P0"]  # noqa: RUF012
    pytree_data_fields = ("A", "b", "Q", "mu0", "P0")

    def __init__(
        self,
        A: Float[Array, "Tm1 d d"],
        Q: Float[Array, "Tm1 d d"],
        mu0: Float[Array, " d"],
        P0: Float[Array, "d d"],
        *,
        b: Float[Array, "Tm1 d"] | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ValueError(f"A must have shape (T-1, d, d), got {A.shape}.")
        Tm1, d, _ = A.shape
        if Q.shape != (Tm1, d, d):
            raise ValueError(f"Q must have shape {(Tm1, d, d)}, got {Q.shape}.")
        if mu0.shape != (d,):
            raise ValueError(f"mu0 must have shape {(d,)}, got {mu0.shape}.")
        if P0.shape != (d, d):
            raise ValueError(f"P0 must have shape {(d, d)}, got {P0.shape}.")
        if b is None:
            b = jnp.zeros((Tm1, d), dtype=A.dtype)
        elif b.shape != (Tm1, d):
            raise ValueError(f"b must have shape {(Tm1, d)}, got {b.shape}.")
        self.A = A
        self.b = b
        self.Q = Q
        self.mu0 = mu0
        self.P0 = P0
        super().__init__(
            batch_shape=(),
            event_shape=(Tm1 + 1, d),
            validate_args=validate_args,
        )

    @property
    def horizon(self) -> int:
        """Number of states T in the chain."""
        return self.event_shape[0]

    @property
    def state_dim(self) -> int:
        """State size D_x."""
        return self.event_shape[1]

    # ------------------------------------------------------------------
    # Moment views
    # ------------------------------------------------------------------

    def marginals(self) -> tuple[Float[Array, "T d"], Float[Array, "T d d"]]:
        """Marginal means μ_x,t and covariances Σ_x,t of every state.

            μ_x,t+1 = F_t μ_x,t + c_t
            Σ_x,t+1 = F_t Σ_x,t F_tᵀ + Σ_η,t+1

        Algorithm:
            1. Start from the prior moments μ_x,0 and Σ_x,0.
            2. Push the moments through each transition with a scan.
            3. Prepend the prior moments to the propagated ones.

        Returns:
            Tuple ``(means, covs)`` holding μ_x,t and Σ_x,t for
            t = 0, …, T − 1.
        """

        def _step(carry, inputs):
            mean, covariance = carry
            transition, offset, model_noise_covariance = inputs
            # F_t μ_x,t
            propagated_mean = transition @ mean
            # μ_x,t+1 = F_t μ_x,t + c_t
            mean_next = propagated_mean + offset
            # F_t Σ_x,t F_tᵀ, one three-operand contraction
            propagated_covariance = einsum(
                transition,
                covariance,
                transition,
                "next prev, prev col, other col -> next other",
            )
            # Σ_x,t+1 = F_t Σ_x,t F_tᵀ + Σ_η,t+1
            covariance_next = propagated_covariance + model_noise_covariance
            return (mean_next, covariance_next), (mean_next, covariance_next)

        _, (means_rest, covs_rest) = jax.lax.scan(
            _step, (self.mu0, self.P0), (self.A, self.b, self.Q)
        )
        # μ_x,0 as the first row
        prior_mean_row = rearrange(self.mu0, "state -> 1 state")
        # Σ_x,0 as the first slab
        prior_covariance_slab = rearrange(self.P0, "row col -> 1 row col")
        # μ_x,0:T−1, prior first
        means = jnp.concatenate([prior_mean_row, means_rest])
        # Σ_x,0:T−1, prior first
        covs = jnp.concatenate([prior_covariance_slab, covs_rest])
        return means, covs

    @lazy_property
    def mean(self) -> Float[Array, "T d"]:
        """Marginal means μ_x,t, one row per state."""
        return self.marginals()[0]

    @lazy_property
    def variance(self) -> Float[Array, "T d"]:
        """Marginal variances diag(Σ_x,t), one row per state."""
        _, covs = self.marginals()
        # diag(Σ_x,t)
        return rearrange(covs, "step state state -> step state")

    def cross_covariances(self) -> Float[Array, "Tm1 d d"]:
        """Lag-one cross-covariances Σ_x,t+1,t = Cov[x_t+1, x_t] = F_t Σ_x,t.

        Returns:
            One D_x × D_x block per transition, t = 0, …, T − 2.
        """
        _, covs = self.marginals()
        return self._cross(covs)

    def _cross(self, covs: Float[Array, "T d d"]) -> Float[Array, "Tm1 d d"]:
        """Lag-one cross-covariances Σ_x,t+1,t = F_t Σ_x,t from the marginals.

        Args:
            covs: Marginal covariances Σ_x,t, one per state.

        Returns:
            One block Σ_x,t+1,t per transition.
        """
        if self.A.shape[0] == 0:
            # One step, no transitions: einx cannot contract the empty
            # transition axis (gh-348).
            return self.A
        # Σ_x,t+1,t = F_t Σ_x,t
        return einsum(
            self.A, covs[:-1], "step next prev, step prev col -> step next col"
        )

    def pairwise_marginals(
        self,
    ) -> tuple[Float[Array, "Tm1 two_d"], Float[Array, "Tm1 two_d two_d"]]:
        """Joint Gaussian p(x_t, x_t+1) of each pair of consecutive states.

            p(x_t, x_t+1) = 𝒩([μ_x,t; μ_x,t+1], [[Σ_x,t, Σ_x,t+1,tᵀ],
                                                [Σ_x,t+1,t, Σ_x,t+1]])

        Returns:
            Tuple ``(joint_means, joint_covs)`` with one pair per
            transition, each of length 2 D_x, in the layout of
            `gaussx.pairwise_marginals`.
        """
        means, covs = self.marginals()
        if self.A.shape[0] == 0:
            # One state, no pairs.
            # 2 D_x
            joint_size = 2 * self.state_dim
            return (
                jnp.zeros((0, joint_size), means.dtype),
                jnp.zeros((0, joint_size, joint_size), covs.dtype),
            )
        return pairwise_marginals(means, covs, self._cross(covs))

    @lazy_property
    def covariance_matrix(self) -> Float[Array, "Td Td"]:
        """Dense joint covariance Σ_x = Λ_x⁻¹ of the flattened trajectory.

        Built column by column through `UDLDecomposition.solve`, so it
        costs O(T² D_x³) and is meant for small T (reference checks, dense
        downstream consumers).
        """
        # T D_x
        joint_size = self.horizon * self.state_dim
        identity = jnp.eye(joint_size, dtype=self.A.dtype)
        # Λ_x = U D Uᵀ
        udl = self.udl()
        # Σ_x = Λ_x⁻¹ I, one column at a time
        return jax.vmap(udl.solve, in_axes=1, out_axes=1)(identity)

    # ------------------------------------------------------------------
    # Precision views
    # ------------------------------------------------------------------

    def udl(self) -> UDLDecomposition:
        """The chain's precision, already factorised as Λ_x = U D Uᵀ.

        Returns:
            The `UDLDecomposition` with U_tᵀ = −F_t, D_t = Σ_η,t⁻¹ for
            t ≥ 1 and D_0 = Σ_x,0⁻¹.
        """
        return udl_from_ssm_params(self.A, self.Q, self.P0)

    @property
    def precision(self) -> BlockTriDiag:
        """Joint precision Λ_x = Σ_x⁻¹ as a `BlockTriDiag`.

        Raw Λ_x, not the −½ Λ_x natural-parameter convention of
        `ssm_to_naturals`.
        """
        return self.udl().as_block_tridiag()

    def to_precision_form(self) -> tuple[Float[Array, " Td"], BlockTriDiag]:
        """The chain as its flattened mean and joint precision ``(μ_x, Λ_x)``.

        This is the layout `spingp_posterior` consumes and returns: pass
        ``mean`` as its ``prior_mean`` (it assumes a zero-mean prior
        otherwise) and hand its output to `from_precision_form`. The
        information vector Λ_x μ_x, if needed, is ``precision.mv(mean)``.

        Returns:
            Tuple ``(mean, precision)``: the marginal means μ_x,t
            flattened state by state, and the joint precision Λ_x.
        """
        # μ_x = [μ_x,0; …; μ_x,T−1]
        mean = rearrange(self.mean, "step state -> (step state)")
        return mean, self.precision

    @classmethod
    def from_precision_form(
        cls,
        mean: Float[Array, " Td"] | Float[Array, "T d"],
        precision: BlockTriDiag,
    ) -> MarkovGaussian:
        """Recover the generative chain from its mean and precision.

            F_t = −U_tᵀ,   Σ_η,t+1 = D_t+1⁻¹,   c_t = μ_x,t+1 − F_t μ_x,t

        Round-trips `to_precision_form` exactly (to floating point).

        Algorithm:
            1. Factorise Λ_x = U D Uᵀ with one `udl_decomposition` pass.
            2. Read off F_t, Σ_η,t+1 and Σ_x,0 = D_0⁻¹ with
               `udl_to_ssm_params`.
            3. Recover the offsets c_t from the mean chain.

        Args:
            mean: Marginal means μ_x,t, flattened state by state or one row
                per state.
            precision: Symmetric positive-definite joint precision Λ_x.

        Returns:
            The equivalent `MarkovGaussian`.
        """
        num_blocks, block_size = precision._num_blocks, precision._block_size
        if mean.ndim == 2:
            means = mean
        else:
            # μ_x,t, one row per state
            means = rearrange(
                mean,
                "(step state) -> step state",
                step=num_blocks,
                state=block_size,
            )
        # Λ_x = U D Uᵀ
        udl = udl_decomposition(precision)
        # F_t = −U_tᵀ and the stacked covariances [Σ_x,0; Σ_η,1:T−1]
        transitions, covariances, _ = udl_to_ssm_params(udl)
        if num_blocks == 1:
            # One state, no transitions: einx cannot contract the empty axis.
            offsets = means[1:]
        else:
            # F_t μ_x,t
            propagated_means = einsum(
                transitions,
                means[:-1],
                "step next prev, step prev -> step next",
            )
            # c_t = μ_x,t+1 − F_t μ_x,t
            offsets = means[1:] - propagated_means
        return cls(transitions, covariances[1:], means[0], covariances[0], b=offsets)

    # ------------------------------------------------------------------
    # Density and sampling
    # ------------------------------------------------------------------

    def _chol_factors(self) -> tuple[Float[Array, "d d"], Float[Array, "Tm1 d d"]]:
        """Lower Cholesky factors L_x,0 of Σ_x,0 and L_η,t+1 of each Σ_η,t+1.

        Returns:
            Tuple ``(prior_factor, model_noise_factors)``.
        """
        # L_x,0 L_x,0ᵀ = Σ_x,0
        prior_factor = jnp.linalg.cholesky(self.P0)
        # L_η,t+1 L_η,t+1ᵀ = Σ_η,t+1
        model_noise_factors = jnp.linalg.cholesky(self.Q)
        return prior_factor, model_noise_factors

    def _log_prob_single(self, xs: Float[Array, "T d"]) -> Float[Array, ""]:
        """Log-density of one trajectory by the chain factorisation, O(T D_x³).

            log p(x_0:T−1) = log 𝒩(x₀; μ_x,0, Σ_x,0)
                + ∑_t log 𝒩(x_t+1; F_t x_t + c_t, Σ_η,t+1)

        Args:
            xs: Trajectory x_0:T−1, one row per state.

        Returns:
            The scalar log p(x_0:T−1).
        """
        prior_factor, model_noise_factors = self._chol_factors()
        # x₀ − μ_x,0
        prior_residual = xs[0] - self.mu0
        # log 𝒩(x₀; μ_x,0, Σ_x,0)
        prior_log_prob = _chol_log_prob(prior_factor, prior_residual)
        if self.A.shape[0] == 0:
            return prior_log_prob
        # F_t x_t
        propagated = einsum(self.A, xs[:-1], "step next prev, step prev -> step next")
        # x_t+1 − F_t x_t
        innovations = xs[1:] - propagated
        # r_t+1 = x_t+1 − F_t x_t − c_t
        residuals = innovations - self.b
        # log 𝒩(x_t+1; F_t x_t + c_t, Σ_η,t+1), one per transition
        transition_log_probs = jax.vmap(_chol_log_prob)(model_noise_factors, residuals)
        # ∑_t log 𝒩(x_t+1; F_t x_t + c_t, Σ_η,t+1), a full reduction
        transition_log_prob = jnp.sum(transition_log_probs)
        # log p(x_0:T−1)
        return prior_log_prob + transition_log_prob

    @validate_sample
    def log_prob(self, value: Float[Array, "*batch T d"]) -> Float[Array, "*batch"]:
        """Log-density log p(x_0:T−1) of each trajectory in ``value``.

        Args:
            value: Trajectories x_0:T−1, with leading batch axes.

        Returns:
            One log-density per trajectory.
        """
        leading_shape = value.shape[:-2]
        value_flat = rearrange(value, "... step state -> (...) step state")
        log_prob_flat = jax.vmap(self._log_prob_single)(value_flat)
        return _reshape_batch(log_prob_flat, leading_shape)

    def _sample_single(self, key: jax.Array) -> Float[Array, "T d"]:
        """One ancestral draw of the trajectory x_0:T−1, O(T D_x³).

            x₀ = μ_x,0 + L_x,0 z₀
            x_t+1 = F_t x_t + c_t + L_η,t+1 z_t+1

        Args:
            key: PRNG key.

        Returns:
            The trajectory x_0:T−1, one row per state.
        """
        prior_factor, model_noise_factors = self._chol_factors()
        num_states, state_size = self.horizon, self.state_dim
        prior_key, transition_key = jax.random.split(key)
        # Draw in the parameters' dtype, not JAX's default float.
        dtype = jnp.result_type(self.mu0, self.P0, self.Q)
        # z₀ ~ 𝒩(0, I)
        prior_noise = jax.random.normal(prior_key, (state_size,), dtype=dtype)
        # L_x,0 z₀
        prior_deviation = prior_factor @ prior_noise
        # x₀ = μ_x,0 + L_x,0 z₀
        initial_state = self.mu0 + prior_deviation
        # z_t+1 ~ 𝒩(0, I), one per transition
        transition_noise = jax.random.normal(
            transition_key, (num_states - 1, state_size), dtype=dtype
        )

        def _step(state, inputs):
            transition, offset, model_noise_factor, noise = inputs
            # F_t x_t
            propagated = transition @ state
            # F_t x_t + c_t
            mean_next = propagated + offset
            # L_η,t+1 z_t+1
            model_noise = model_noise_factor @ noise
            # x_t+1 = F_t x_t + c_t + L_η,t+1 z_t+1
            state_next = mean_next + model_noise
            return state_next, state_next

        _, states_rest = jax.lax.scan(
            _step,
            initial_state,
            (self.A, self.b, model_noise_factors, transition_noise),
        )
        # jnp, not einx: a draw with D_x = 0 works today, and einx rejects
        # a zero-length axis.
        # x₀ as the first row
        initial_state_row = initial_state[None]
        # x_0:T−1, x₀ first
        return jnp.concatenate([initial_state_row, states_rest])

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample T d"]:
        """Ancestral draws of the trajectory x_0:T−1.

        Args:
            key: PRNG key.
            sample_shape: Leading shape of the draws.

        Returns:
            Draws of shape ``sample_shape + (T, d)``.

        Raises:
            ValueError: If ``key`` is ``None``.
        """
        if key is None:
            raise ValueError("PRNG key must be provided to sample from MarkovGaussian.")
        num_draws = math.prod(sample_shape) if sample_shape else 1
        keys = jax.random.split(key, num_draws)
        samples_flat = jax.vmap(self._sample_single)(keys)
        return _reshape_series(samples_flat, sample_shape)
