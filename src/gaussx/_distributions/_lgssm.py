"""Linear-Gaussian state-space models as sampleable densities."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as dist
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
    """Operator view of a covariance, tagged PSD when wrapping an array."""
    return _as_operator(operand, lx.positive_semidefinite_tag)


def _out_size(operand: Float[Array, "M N"] | lx.AbstractLinearOperator) -> int:
    """Number of output rows, without materialising an operator."""
    if isinstance(operand, lx.AbstractLinearOperator):
        return operand.out_size()
    return operand.shape[0]


class LGSSM(dist.Distribution):
    r"""Linear-Gaussian state-space model as a density over ``(T, M)``.

    Models

    $$
    x_t = A x_{t-1} + q_t,\quad q_t \sim N(0, Q), \qquad
    y_t = H x_t + r_t,\quad r_t \sim N(0, R),
    $$

    with $x_0 \sim N(m_0, P_0)$ and $t = 1, \dots, T$. ``log_prob`` is
    the Kalman marginal likelihood $\log p(y_{1:T})$, delegated to
    `gaussx.kalman_filter`; ``sample`` is ancestral forward
    simulation. ``event_shape`` is ``(T, M)``, so ``log_prob`` returns a
    scalar and no ``.to_event()`` wrapping is needed.

    Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).

    ``A``, ``H``, ``Q`` and ``R`` each accept a dense array *or* a
    `lineax.AbstractLinearOperator`, exactly matching
    `gaussx.kalman_filter`'s contract. Structured ``Q`` / ``R`` keep
    their structure through the Cholesky in ``sample``, ``H`` through
    the sandwich in ``variance``, and ``A`` / ``H`` apply via
    structural matvec rather than a dense matmul. ``m0`` and ``P0`` are
    dense, as they are for `gaussx.kalman_filter`.

    Args:
        A: Transition matrix or operator, shape ``(N, N)``.
        H: Emission matrix or operator, shape ``(M, N)``.
        Q: Process noise covariance or operator, shape ``(N, N)``.
        R: Observation noise covariance or operator, shape ``(M, M)``.
        m0: Initial state mean, shape ``(N,)``.
        P0: Initial state covariance, shape ``(N, N)``.
        n_steps: Sequence length ``T``.
        validate_args: Whether to validate input arguments.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from gaussx import LGSSM
        >>> d = LGSSM(0.9 * jnp.eye(2), jnp.eye(2), 0.1 * jnp.eye(2),
        ...           0.2 * jnp.eye(2), jnp.zeros(2), jnp.eye(2), n_steps=10)
        >>> y = d.sample(jr.key(0))
        >>> y.shape
        (10, 2)
        >>> d.log_prob(y).shape
        ()

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
        """Sequence length ``T``."""
        return self.event_shape[0]

    @property
    def _obs_mask(self) -> Bool[Array, "T M"] | None:
        """Observation mask handed to `gaussx.kalman_filter`."""
        return None

    def _log_prob_single(self, value: Float[Array, "T M"]) -> Float[Array, ""]:
        return kalman_filter(
            self.A,
            self.H,
            self.Q,
            self.R,
            value,
            self.m0,
            self.P0,
            mask=self._obs_mask,
        ).log_likelihood

    @validate_sample
    def log_prob(self, value: Float[Array, "*batch T M"]) -> Float[Array, "*batch"]:
        leading_shape = value.shape[:-2]
        value_flat = rearrange(value, "... T M -> (...) T M")
        log_prob_flat = jax.vmap(self._log_prob_single)(value_flat)
        return _reshape_batch(log_prob_flat, leading_shape)

    def _sample_single(self, key: jax.Array) -> Float[Array, "T M"]:
        """One ancestral forward simulation of the full ``(T, M)`` series.

        Cholesky factors go through the structural
        `gaussx.cholesky` primitive, so a Kronecker / block-diagonal
        / diagonal covariance is never materialised.
        """
        N = self.m0.shape[0]
        M = self.event_shape[1]
        key_init, key_q, key_r = jax.random.split(key, 3)
        L_P0 = _cholesky(_psd(self.P0))
        L_Q = _cholesky(_psd(self.Q))
        L_R = _cholesky(_psd(self.R))

        x0 = self.m0 + L_P0.mv(jax.random.normal(key_init, (N,)))
        eps_q = jax.random.normal(key_q, (self.n_steps, N))
        eps_r = jax.random.normal(key_r, (self.n_steps, M))

        def step(x, noise):
            e_q, e_r = noise
            x_next = _matvec(self.A, x) + L_Q.mv(e_q)
            y = _matvec(self.H, x_next) + L_R.mv(e_r)
            return x_next, y

        _, ys = jax.lax.scan(step, x0, (eps_q, eps_r))
        return ys

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample T M"]:
        if key is None:
            raise ValueError("PRNG key must be provided to sample from LGSSM.")
        n_samples = math.prod(sample_shape) if sample_shape else 1
        keys = jax.random.split(key, n_samples)
        samples_flat = jax.vmap(self._sample_single)(keys)
        return _reshape_series(samples_flat, sample_shape)

    @lazy_property
    def mean(self) -> Float[Array, "T M"]:
        """Marginal mean $H A^t m_0$ of each observation, shape ``(T, M)``."""

        def step(m, _):
            m_next = _matvec(self.A, m)
            return m_next, _matvec(self.H, m_next)

        _, means = jax.lax.scan(step, self.m0, None, length=self.n_steps)
        return means

    @lazy_property
    def variance(self) -> Float[Array, "T M"]:
        """Marginal variance $\\mathrm{diag}(H P_t H^\\top + R)$, shape ``(T, M)``.

        The state recursion runs on dense ``P_t`` — it is the evolving
        quantity, so no input structure survives it — but the emission
        step avoids forming ``H P H^T``: operators go through
        `gaussx.sandwich` + `gaussx.diag`, arrays contract straight
        to the diagonal.
        """
        A_dense = _materialise(self.A)
        Q_dense = _materialise(self.Q)
        diag_R = _diag(_as_operator(self.R))
        H_op = self.H if isinstance(self.H, lx.AbstractLinearOperator) else None

        def step(P, _):
            P_next = A_dense @ P @ A_dense.T + Q_dense
            if H_op is not None:
                diag_HPH = _diag(sandwich(H_op, _psd(P_next)))
            else:
                diag_HPH = einsum(self.H, P_next, self.H, "m n, n k, m k -> m")
            return P_next, diag_HPH + diag_R

        _, variances = jax.lax.scan(step, self.P0, None, length=self.n_steps)
        return variances


class MaskedLGSSM(LGSSM):
    r"""`LGSSM` whose ``log_prob`` marginalises unobserved channels exactly.

    Takes a ``(T, M)`` boolean mask at construction. ``log_prob`` returns
    the exact marginal $\log p(y_{\mathrm{obs}})$, not a bound: the
    conditional $p(y_{\mathrm{miss}} \mid y_{\mathrm{obs}})$ is
    closed-form Gaussian, so dropping dimensions costs nothing in
    exactness. Masked entries of ``value`` are never read and may be
    ``NaN``.

    ``sample`` is inherited unchanged and simulates the **complete**
    ``(T, M)`` series — the mask governs which entries the density
    scores, not which are generated.

    Args:
        A: Transition matrix or operator, shape ``(N, N)``.
        H: Emission matrix or operator, shape ``(M, N)``.
        Q: Process noise covariance or operator, shape ``(N, N)``.
        R: Observation noise covariance or operator, shape ``(M, M)``.
        m0: Initial state mean, shape ``(N,)``.
        P0: Initial state covariance, shape ``(N, N)``.
        n_steps: Sequence length ``T``.
        obs_mask: Observation mask, shape ``(T, M)``. ``True`` marks an
            observed channel. An all-``False`` row leaves the state at
            its predicted value and contributes nothing. Named
            ``obs_mask`` so it does not shadow numpyro's inherited
            ``Distribution.mask()`` method.
        validate_args: Whether to validate input arguments.

    Raises:
        ValueError: If ``obs_mask`` does not have shape ``(n_steps, M)``.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from gaussx import MaskedLGSSM
        >>> mask = jr.bernoulli(jr.key(0), 0.7, (10, 2))
        >>> d = MaskedLGSSM(0.9 * jnp.eye(2), jnp.eye(2), 0.1 * jnp.eye(2),
        ...                 0.2 * jnp.eye(2), jnp.zeros(2), jnp.eye(2),
        ...                 n_steps=10, obs_mask=mask)
        >>> d.log_prob(d.sample(jr.key(1))).shape
        ()
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
        return self.obs_mask


class LGSSMFactory(eqx.Module):
    """Callable ``obs_mask -> MaskedLGSSM``, for conditional use.

    An `equinox.Module` rather than a closure so the state-space
    parameters stay visible to `equinox.filter_grad` when passed as
    ``gauss_flows.NumpyroBase(dist_factory=...)``.

    Attributes:
        A: Transition matrix or operator, shape ``(N, N)``.
        H: Emission matrix or operator, shape ``(M, N)``.
        Q: Process noise covariance or operator, shape ``(N, N)``.
        R: Observation noise covariance or operator, shape ``(M, M)``.
        m0: Initial state mean, shape ``(N,)``.
        P0: Initial state covariance, shape ``(N, N)``.
        n_steps: Sequence length ``T``.

    Examples:
        >>> import jax.numpy as jnp, jax.random as jr
        >>> from gaussx import LGSSMFactory
        >>> factory = LGSSMFactory(0.9 * jnp.eye(2), jnp.eye(2),
        ...                        0.1 * jnp.eye(2), 0.2 * jnp.eye(2),
        ...                        jnp.zeros(2), jnp.eye(2), n_steps=10)
        >>> d = factory(jr.bernoulli(jr.key(0), 0.7, (10, 2)))
        >>> d.event_shape
        (10, 2)
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
            obs_mask: Observation mask, shape ``(T, M)``.

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
