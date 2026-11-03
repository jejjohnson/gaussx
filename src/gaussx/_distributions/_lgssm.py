"""Time-invariant linear-Gaussian state-space models as sampleable densities.

The notation is that of `gaussx.kalman_filter`: states x_t ∈ ℝ^{D_x} with
transition F and model-noise covariance Σ_η, observations y_t ∈ ℝ^{D_y}
with observation operator H and observation-noise covariance Σ_ε, a prior
x₀ ~ 𝒩(μ_x,0, Σ_x,0) and t = 1, …, T. L_x,0, L_η and L_ε are the lower
Cholesky factors of Σ_x,0, Σ_η and Σ_ε, and z ~ 𝒩(0, I) is the standard
normal noise of a draw.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float
from numpyro.distributions.util import lazy_property, validate_sample

from gaussx._distributions._utils import _reshape_batch, _reshape_series
from gaussx._einx import einsum, rearrange
from gaussx._linalg._linalg import sandwich
from gaussx._primitives._cholesky import cholesky as _cholesky
from gaussx._primitives._diag import diag as _diag
from gaussx._ssm._kalman import kalman_filter
from gaussx._ssm._utils import _as_operator, _materialise, _matvec


# Operator-or-array unions mirroring ``gaussx.kalman_filter``'s contract.
_Transition = Float[Array, "N N"] | lx.AbstractLinearOperator
_Emission = Float[Array, "M N"] | lx.AbstractLinearOperator
_StateCov = Float[Array, "N N"] | lx.AbstractLinearOperator
_ObsCov = Float[Array, "M M"] | lx.AbstractLinearOperator


def _psd(operand: Float[Array, ...] | lx.AbstractLinearOperator):
    """Operator view of a covariance, tagged PSD when wrapping an array.

    Args:
        operand: A covariance, as a dense array or a lineax operator.

    Returns:
        The operator; an operator passes through untagged.
    """
    return _as_operator(operand, lx.positive_semidefinite_tag)


def _out_size(operand: Float[Array, "M N"] | lx.AbstractLinearOperator) -> int:
    """Number of output rows, without materialising an operator.

    Args:
        operand: A matrix, as a dense array or a lineax operator.

    Returns:
        The number of rows, D_y for the observation operator H.
    """
    if isinstance(operand, lx.AbstractLinearOperator):
        return operand.out_size()
    return operand.shape[0]


class LGSSM(dist.Distribution):
    """Linear-Gaussian state-space model as a density over observations ``(T, M)``.

        x_t = F x_t−1 + η_t,   η_t ~ 𝒩(0, Σ_η)
        y_t = H x_t + ε_t,     ε_t ~ 𝒩(0, Σ_ε)

    with x₀ ~ 𝒩(μ_x,0, Σ_x,0), a state x_t of length D_x, an observation
    y_t of length D_y and t = 1, …, T. ``log_prob`` is the Kalman marginal
    likelihood log p(y_1:T), delegated to `gaussx.kalman_filter`; ``sample``
    is ancestral forward simulation. ``event_shape`` is ``(T, M)``, so
    ``log_prob`` returns a scalar and no ``.to_event()`` wrapping is needed.

    Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).

    ``A``, ``H``, ``Q`` and ``R`` each accept a dense array *or* a
    `lineax.AbstractLinearOperator`, exactly matching
    `gaussx.kalman_filter`'s contract. Structured Σ_η and Σ_ε keep their
    structure through the Cholesky in ``sample``, H through the sandwich in
    ``variance``, and F and H apply via structural matvec rather than a
    dense matmul. μ_x,0 and Σ_x,0 are dense, as they are for
    `gaussx.kalman_filter`.

    Args:
        A: Transition F, the same for every step.
        H: Observation operator H.
        Q: Model-noise covariance Σ_η.
        R: Observation-noise covariance Σ_ε.
        m0: Prior mean μ_x,0 of the initial state x₀.
        P0: Prior covariance Σ_x,0 of x₀.
        n_steps: Number of steps T.
        validate_args: Whether to validate input arguments.

    Examples:
        Ten steps of a two-dimensional state, observed in full:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gaussx import LGSSM
        >>> model = LGSSM(
        ...     0.9 * jnp.eye(2),
        ...     jnp.eye(2),
        ...     0.1 * jnp.eye(2),
        ...     0.2 * jnp.eye(2),
        ...     jnp.zeros(2),
        ...     jnp.eye(2),
        ...     n_steps=10,
        ... )
        >>> y = model.sample(jr.key(0))
        >>> y.shape, model.log_prob(y).shape
        ((10, 2), ())

        The marginal moments have the event shape:

        >>> model.mean.shape, model.variance.shape
        ((10, 2), (10, 2))

        A diagonal operator Σ_η gives the same density as its dense matrix:

        >>> import lineax as lx
        >>> structured = LGSSM(
        ...     0.9 * jnp.eye(2),
        ...     jnp.eye(2),
        ...     lx.DiagonalLinearOperator(0.1 * jnp.ones(2)),
        ...     0.2 * jnp.eye(2),
        ...     jnp.zeros(2),
        ...     jnp.eye(2),
        ...     n_steps=10,
        ... )
        >>> bool(jnp.allclose(structured.log_prob(y), model.log_prob(y)))
        True

    Notes:
        To use as a normalizing-flow base, wrap with
        ``gauss_flows.NumpyroBase``; no bespoke adapter is required.
    """

    arg_constraints = {}  # noqa: RUF012
    support = dist.constraints.real_matrix
    reparametrized_params = ["A", "H", "Q", "R", "m0", "P0"]  # noqa: RUF012
    pytree_data_fields = ("A", "H", "Q", "R", "m0", "P0")

    def __init__(
        self,
        A: _Transition,
        H: _Emission,
        Q: _StateCov,
        R: _ObsCov,
        m0: Float[Array, " N"],
        P0: Float[Array, "N N"],
        n_steps: int,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        self.m0 = m0
        self.P0 = P0
        super().__init__(
            batch_shape=(),
            event_shape=(n_steps, _out_size(H)),
            validate_args=validate_args,
        )

    @property
    def n_steps(self) -> int:
        """Number of steps T."""
        return self.event_shape[0]

    @property
    def _obs_mask(self) -> Bool[Array, "T M"] | None:
        """Observation mask handed to `gaussx.kalman_filter`."""
        return None

    def _log_prob_single(self, value: Float[Array, "T M"]) -> Float[Array, ""]:
        """Kalman marginal log-likelihood log p(y_1:T) of one series.

        Args:
            value: Observations y_1:T, one row per step.

        Returns:
            The scalar log p(y_1:T).
        """
        filter_state = kalman_filter(
            self.A,
            self.H,
            self.Q,
            self.R,
            value,
            self.m0,
            self.P0,
            mask=self._obs_mask,
        )
        return filter_state.log_likelihood

    @validate_sample
    def log_prob(self, value: Float[Array, "*batch T M"]) -> Float[Array, "*batch"]:
        """Kalman marginal log-likelihood log p(y_1:T) of each series.

        Args:
            value: Observations y_1:T, with leading batch axes.

        Returns:
            One log-likelihood per series.
        """
        leading_shape = value.shape[:-2]
        value_flat = rearrange(value, "... step obs -> (...) step obs")
        log_prob_flat = jax.vmap(self._log_prob_single)(value_flat)
        return _reshape_batch(log_prob_flat, leading_shape)

    def _sample_single(self, key: jax.Array) -> Float[Array, "T M"]:
        """One ancestral forward simulation of the series y_1:T.

            x₀ = μ_x,0 + L_x,0 z₀
            x_t = F x_t−1 + L_η z_η,t
            y_t = H x_t + L_ε z_ε,t

        Cholesky factors go through the structural `gaussx.cholesky`
        primitive, so a Kronecker, block-diagonal or diagonal covariance is
        never materialised.

        Args:
            key: PRNG key.

        Returns:
            The observations y_1:T, one row per step.
        """
        state_size = self.m0.shape[0]
        obs_size = self.event_shape[1]
        prior_key, model_noise_key, obs_noise_key = jax.random.split(key, 3)
        # L_x,0 L_x,0ᵀ = Σ_x,0
        prior_factor = _cholesky(_psd(self.P0))
        # L_η L_ηᵀ = Σ_η
        model_noise_factor = _cholesky(_psd(self.Q))
        # L_ε L_εᵀ = Σ_ε
        obs_noise_factor = _cholesky(_psd(self.R))

        # Each noise term is drawn in its factor's dtype, not JAX's default
        # float, so a float32 model stays float32 under x64.
        # z₀ ~ 𝒩(0, I)
        prior_noise = jax.random.normal(
            prior_key, (state_size,), dtype=prior_factor.in_structure().dtype
        )
        # L_x,0 z₀
        prior_deviation = prior_factor.mv(prior_noise)
        # x₀ = μ_x,0 + L_x,0 z₀
        initial_state = self.m0 + prior_deviation
        # z_η,t ~ 𝒩(0, I), one per step
        model_noise_draws = jax.random.normal(
            model_noise_key,
            (self.n_steps, state_size),
            dtype=model_noise_factor.in_structure().dtype,
        )
        # z_ε,t ~ 𝒩(0, I), one per step
        obs_noise_draws = jax.random.normal(
            obs_noise_key,
            (self.n_steps, obs_size),
            dtype=obs_noise_factor.in_structure().dtype,
        )

        def step(state, noise):
            model_noise, obs_noise = noise
            # F x_t−1
            propagated = _matvec(self.A, state)
            # L_η z_η,t
            model_deviation = model_noise_factor.mv(model_noise)
            # x_t = F x_t−1 + L_η z_η,t
            state_next = propagated + model_deviation
            # H x_t
            predicted_obs = _matvec(self.H, state_next)
            # L_ε z_ε,t
            obs_deviation = obs_noise_factor.mv(obs_noise)
            # y_t = H x_t + L_ε z_ε,t
            obs = predicted_obs + obs_deviation
            return state_next, obs

        _, observations = jax.lax.scan(
            step, initial_state, (model_noise_draws, obs_noise_draws)
        )
        return observations

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample T M"]:
        """Ancestral draws of the observation series y_1:T.

        Args:
            key: PRNG key.
            sample_shape: Leading shape of the draws.

        Returns:
            Draws of shape ``sample_shape + (T, M)``.

        Raises:
            ValueError: If ``key`` is ``None``.
        """
        if key is None:
            raise ValueError("PRNG key must be provided to sample from LGSSM.")
        num_draws = math.prod(sample_shape) if sample_shape else 1
        keys = jax.random.split(key, num_draws)
        samples_flat = jax.vmap(self._sample_single)(keys)
        return _reshape_series(samples_flat, sample_shape)

    @lazy_property
    def mean(self) -> Float[Array, "T M"]:
        """Marginal mean μ_y,t = H Fᵗ μ_x,0 of each observation."""

        def step(state_mean, _):
            # μ_x,t = F μ_x,t−1
            state_mean_next = _matvec(self.A, state_mean)
            # μ_y,t = H μ_x,t
            obs_mean = _matvec(self.H, state_mean_next)
            return state_mean_next, obs_mean

        _, obs_means = jax.lax.scan(step, self.m0, None, length=self.n_steps)
        return obs_means

    @lazy_property
    def variance(self) -> Float[Array, "T M"]:
        """Marginal variances diag(Σ_y,t) of each observation.

            diag(Σ_y,t) = diag(H Σ_x,t Hᵀ) + diag(Σ_ε)

        The state recursion Σ_x,t = F Σ_x,t−1 Fᵀ + Σ_η runs on a dense Σ_x,t,
        the evolving quantity, so no input structure survives it. The
        observation step avoids forming H Σ_x,t Hᵀ: operators go through
        `gaussx.sandwich` and `gaussx.diag`, arrays contract straight to
        the diagonal.
        """
        transition = _materialise(self.A)
        model_noise_covariance = _materialise(self.Q)
        # diag(Σ_ε)
        obs_noise_variance = _diag(_as_operator(self.R))
        obs_operator = self.H if isinstance(self.H, lx.AbstractLinearOperator) else None

        def step(state_covariance, _):
            # F Σ_x,t−1
            transition_covariance = einsum(
                transition, state_covariance, "next prev, prev col -> next col"
            )
            # F Σ_x,t−1 Fᵀ
            propagated = einsum(
                transition_covariance, transition, "next col, other col -> next other"
            )
            # Σ_x,t = F Σ_x,t−1 Fᵀ + Σ_η
            state_covariance_next = propagated + model_noise_covariance
            if obs_operator is not None:
                # H Σ_x,t Hᵀ, structure-preserving
                obs_projection = sandwich(obs_operator, _psd(state_covariance_next))
                # diag(H Σ_x,t Hᵀ)
                projected_variance = _diag(obs_projection)
            else:
                # diag(H Σ_x,t Hᵀ), one three-operand contraction
                projected_variance = einsum(
                    self.H,
                    state_covariance_next,
                    self.H,
                    "obs state, state col, obs col -> obs",
                )
            # diag(Σ_y,t) = diag(H Σ_x,t Hᵀ) + diag(Σ_ε)
            obs_variance = projected_variance + obs_noise_variance
            return state_covariance_next, obs_variance

        _, obs_variances = jax.lax.scan(step, self.P0, None, length=self.n_steps)
        return obs_variances


class MaskedLGSSM(LGSSM):
    """`LGSSM` whose ``log_prob`` marginalises unobserved channels exactly.

        log p(y_obs) = log ∫ p(y_obs, y_miss) dy_miss

    Takes a ``(T, M)`` boolean mask at construction. ``log_prob`` returns
    the exact marginal log p(y_obs), not a bound: the conditional
    p(y_miss ∣ y_obs) is closed-form Gaussian, so dropping entries costs
    nothing in exactness. Masked entries of ``value`` are never read and
    may be ``NaN``.

    ``sample`` is inherited unchanged and simulates the **complete**
    ``(T, M)`` series: the mask governs which entries the density scores,
    not which are generated.

    Args:
        A: Transition F, the same for every step.
        H: Observation operator H.
        Q: Model-noise covariance Σ_η.
        R: Observation-noise covariance Σ_ε.
        m0: Prior mean μ_x,0 of the initial state x₀.
        P0: Prior covariance Σ_x,0 of x₀.
        n_steps: Number of steps T.
        obs_mask: Observation mask, one row per step. ``True`` marks an
            observed channel. An all-``False`` row leaves the state at
            its predicted value and contributes nothing. Named
            ``obs_mask`` so it does not shadow numpyro's inherited
            ``Distribution.mask()`` method.
        validate_args: Whether to validate input arguments.

    Raises:
        ValueError: If ``obs_mask`` does not have shape ``(n_steps, M)``.

    Examples:
        Score a partially observed series:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gaussx import LGSSM, MaskedLGSSM
        >>> params = (
        ...     0.9 * jnp.eye(2),
        ...     jnp.eye(2),
        ...     0.1 * jnp.eye(2),
        ...     0.2 * jnp.eye(2),
        ...     jnp.zeros(2),
        ...     jnp.eye(2),
        ... )
        >>> mask = jr.bernoulli(jr.key(0), 0.7, (10, 2))
        >>> masked = MaskedLGSSM(*params, n_steps=10, obs_mask=mask)
        >>> masked.log_prob(masked.sample(jr.key(1))).shape
        ()

        An all-``True`` mask gives the `LGSSM` density:

        >>> y = masked.sample(jr.key(2))
        >>> full = MaskedLGSSM(*params, n_steps=10, obs_mask=jnp.ones((10, 2)))
        >>> plain = LGSSM(*params, n_steps=10)
        >>> bool(jnp.allclose(full.log_prob(y), plain.log_prob(y)))
        True
    """

    pytree_data_fields = ("A", "H", "Q", "R", "m0", "P0", "obs_mask")

    # Named ``obs_mask`` rather than ``mask``: ``numpyro.distributions.
    # Distribution.mask`` is an inherited *method* returning a
    # ``MaskedDistribution`` (it backs ``numpyro.sample(..., obs_mask=)``),
    # and shadowing it with an array would silently break that API on
    # this subclass alone.
    obs_mask: Bool[Array, "T M"]

    def __init__(
        self,
        A: _Transition,
        H: _Emission,
        Q: _StateCov,
        R: _ObsCov,
        m0: Float[Array, " N"],
        P0: Float[Array, "N N"],
        n_steps: int,
        obs_mask: Bool[Array, "T M"],
        *,
        validate_args: bool | None = None,
    ) -> None:
        obs_mask = jnp.asarray(obs_mask, dtype=bool)
        expected = (n_steps, _out_size(H))
        if obs_mask.shape != expected:
            raise ValueError(
                f"obs_mask must have shape {expected}; got shape {obs_mask.shape}."
            )
        self.obs_mask = obs_mask
        super().__init__(A, H, Q, R, m0, P0, n_steps, validate_args=validate_args)

    @property
    def _obs_mask(self) -> Bool[Array, "T M"]:
        """Observation mask handed to `gaussx.kalman_filter`."""
        return self.obs_mask

    def _validate_sample(self, value: ArrayLike) -> ArrayLike:
        # Masked entries are never read and may be NaN, so check the support
        # on the observed entries only. numpyro >= 0.22 validates by default,
        # and a NaN anywhere would otherwise turn log_prob into -inf.
        return super()._validate_sample(jnp.where(self.obs_mask, value, 0.0))


class LGSSMFactory(eqx.Module):
    """Callable ``obs_mask -> MaskedLGSSM``, for conditional use.

        obs_mask ↦ MaskedLGSSM(F, H, Σ_η, Σ_ε, μ_x,0, Σ_x,0, T, obs_mask)

    An `equinox.Module` rather than a closure so the state-space
    parameters stay visible to `equinox.filter_grad` when passed as
    ``gauss_flows.NumpyroBase(dist_factory=...)``.

    Attributes:
        A: Transition F, the same for every step.
        H: Observation operator H.
        Q: Model-noise covariance Σ_η.
        R: Observation-noise covariance Σ_ε.
        m0: Prior mean μ_x,0 of the initial state x₀.
        P0: Prior covariance Σ_x,0 of x₀.
        n_steps: Number of steps T.

    Examples:
        Build a masked model for one mask:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from gaussx import LGSSMFactory
        >>> factory = LGSSMFactory(
        ...     0.9 * jnp.eye(2),
        ...     jnp.eye(2),
        ...     0.1 * jnp.eye(2),
        ...     0.2 * jnp.eye(2),
        ...     jnp.zeros(2),
        ...     jnp.eye(2),
        ...     n_steps=10,
        ... )
        >>> model = factory(jr.bernoulli(jr.key(0), 0.7, (10, 2)))
        >>> model.event_shape
        (10, 2)

        The model carries the factory's parameters:

        >>> bool(jnp.allclose(model.A, factory.A))
        True
    """

    A: _Transition
    H: _Emission
    Q: _StateCov
    R: _ObsCov
    m0: Float[Array, " N"]
    P0: Float[Array, "N N"]
    n_steps: int = eqx.field(static=True)

    def __call__(self, obs_mask: Bool[Array, "T M"]) -> MaskedLGSSM:
        """Build the `MaskedLGSSM` for a given observation mask.

        Args:
            obs_mask: Observation mask, one row per step; ``True`` marks an
                observed channel.

        Returns:
            A `MaskedLGSSM` carrying this factory's parameters.
        """
        return MaskedLGSSM(
            self.A,
            self.H,
            self.Q,
            self.R,
            self.m0,
            self.P0,
            self.n_steps,
            obs_mask,
        )
