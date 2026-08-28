"""Chain-Markov Gaussian over a state trajectory as a NumPyro density."""

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
    """``log N(residual; 0, L L^T)`` from a lower Cholesky factor ``L``."""
    z = jsl.solve_triangular(chol, residual, lower=True)
    log_det = jnp.sum(jnp.log(jnp.diagonal(chol)))
    return -0.5 * (z @ z) - log_det - 0.5 * residual.shape[0] * _LOG_2PI


class MarkovGaussian(dist.Distribution):
    r"""Chain-Markov Gaussian over $x_{0:T-1}$ as a density over ``(T, d)``.

    The canonical parameterisation is the generative state-space form

    $$
    x_0 \sim \mathcal{N}(\mu_0, P_0), \qquad
    x_{k+1} = A_k x_k + b_k + \varepsilon_k, \quad
    \varepsilon_k \sim \mathcal{N}(0, Q_k),
    $$

    whose joint covariance has a block-tridiagonal inverse. ``event_shape``
    is ``(T, d)``, so ``log_prob`` of a trajectory is a scalar and the
    distribution can be used directly as a NumPyro site — a prior over a
    latent path, or a structured variational guide. Every view is derived
    from the SSM tuple on demand:

    | view | cost | route |
    |---|---|---|
    | `marginals`, `mean`, `variance` | $O(T d^3)$ | forward moment propagation |
    | `sample`, `log_prob` | $O(T d^3)$ | chain factorisation, no joint Cholesky |
    | `precision`, `to_precision_form` | $O(T d^3)$ | `udl_from_ssm_params` |
    | `from_precision_form` | $O(T d^3)$ | `udl_decomposition` |
    | `covariance_matrix` | $O(T^2 d^3)$ | banded solves vs. identity; small $T$ |

    Where `LGSSM` is a density over the *observations* of a chain with
    the states marginalised out, this is the density over the *states*
    themselves. Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).

    Args:
        A: Transition matrices, shape ``(T-1, d, d)``.
        Q: Process noise covariances, shape ``(T-1, d, d)``; ``Q[k]``
            drives the step from $x_k$ to $x_{k+1}$.
        mu0: Initial mean, shape ``(d,)``.
        P0: Initial covariance, shape ``(d, d)``.
        b: Optional transition offsets, shape ``(T-1, d)``. Defaults to
            zero.
        validate_args: Whether to validate input arguments.

    Examples:
        >>> import jax, jax.numpy as jnp, gaussx
        >>> T, d = 10, 2
        >>> A = jnp.broadcast_to(0.9 * jnp.eye(d), (T - 1, d, d))
        >>> Q = jnp.broadcast_to(0.1 * jnp.eye(d), (T - 1, d, d))
        >>> chain = gaussx.MarkovGaussian(A, Q, jnp.zeros(d), jnp.eye(d))
        >>> xs = chain.sample(jax.random.key(0), (4,))
        >>> xs.shape, chain.log_prob(xs).shape
        ((4, 10, 2), (4,))
        >>> mean, precision = chain.to_precision_form()
        >>> mean.shape, precision.as_matrix().shape
        ((20,), (20, 20))
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
        """Number of states ``T`` in the chain."""
        return self.event_shape[0]

    @property
    def state_dim(self) -> int:
        """State dimension ``d``."""
        return self.event_shape[1]

    # ------------------------------------------------------------------
    # Moment views
    # ------------------------------------------------------------------

    def marginals(self) -> tuple[Float[Array, "T d"], Float[Array, "T d d"]]:
        r"""Marginal means and covariances of every state.

        Forward propagation $m_{k+1} = A_k m_k + b_k$,
        $P_{k+1} = A_k P_k A_k^{\top} + Q_k$.

        Returns:
            Tuple ``(means, covs)`` of shapes ``(T, d)`` and ``(T, d, d)``.
        """

        def _step(carry, inputs):
            m, P = carry
            A_k, b_k, Q_k = inputs
            m_next = A_k @ m + b_k
            P_next = einsum(A_k, P, A_k, "i j, j k, l k -> i l") + Q_k
            return (m_next, P_next), (m_next, P_next)

        _, (means_rest, covs_rest) = jax.lax.scan(
            _step, (self.mu0, self.P0), (self.A, self.b, self.Q)
        )
        means = jnp.concatenate([self.mu0[None], means_rest], axis=0)
        covs = jnp.concatenate([self.P0[None], covs_rest], axis=0)
        return means, covs

    @lazy_property
    def mean(self) -> Float[Array, "T d"]:
        """Marginal means, shape ``(T, d)``."""
        return self.marginals()[0]

    @lazy_property
    def variance(self) -> Float[Array, "T d"]:
        """Marginal variances, shape ``(T, d)``."""
        _, covs = self.marginals()
        return jnp.diagonal(covs, axis1=-2, axis2=-1)

    def cross_covariances(self) -> Float[Array, "Tm1 d d"]:
        r"""$\mathrm{Cov}(x_{k+1}, x_k) = A_k P_k$ for each consecutive pair."""
        _, covs = self.marginals()
        return einsum(self.A, covs[:-1], "T i j, T j k -> T i k")

    def pairwise_marginals(
        self,
    ) -> tuple[Float[Array, "Tm1 two_d"], Float[Array, "Tm1 two_d two_d"]]:
        r"""Joint $p(x_k, x_{k+1})$ for each consecutive pair.

        Returns:
            Tuple ``(joint_means, joint_covs)`` of shapes ``(T-1, 2d)``
            and ``(T-1, 2d, 2d)``, in the layout of
            `gaussx.pairwise_marginals`.
        """
        means, covs = self.marginals()
        cross = einsum(self.A, covs[:-1], "T i j, T j k -> T i k")
        return pairwise_marginals(means, covs, cross)

    @lazy_property
    def covariance_matrix(self) -> Float[Array, "Td Td"]:
        r"""Dense joint covariance $\Lambda^{-1}$, shape ``(Td, Td)``.

        Built column by column through `UDLDecomposition.solve`, so it
        costs $O(T^2 d^3)$ and is meant for small $T$ (reference checks,
        dense downstream consumers).
        """
        n = self.horizon * self.state_dim
        eye = jnp.eye(n, dtype=self.A.dtype)
        return jax.vmap(self.udl().solve, in_axes=1, out_axes=1)(eye)

    # ------------------------------------------------------------------
    # Precision views
    # ------------------------------------------------------------------

    def _Q_with_initial(self) -> Float[Array, "T d d"]:
        """``Q`` in the ``Q[0] == P0`` layout used by the UDL helpers."""
        return jnp.concatenate([self.P0[None], self.Q], axis=0)

    def udl(self) -> UDLDecomposition:
        r"""The chain's precision, already factorised as $U \tilde{D} U^{\top}$."""
        return udl_from_ssm_params(self.A, self._Q_with_initial())

    @property
    def precision(self) -> BlockTriDiag:
        r"""Joint precision $\Lambda$ as a `BlockTriDiag`.

        Raw $\Lambda$, not the $-\tfrac12 \Lambda$ natural-parameter
        convention of `ssm_to_naturals`.
        """
        return self.udl().as_block_tridiag()

    def to_precision_form(self) -> tuple[Float[Array, " Td"], BlockTriDiag]:
        r"""The chain as ``(mean, precision)``.

        This is the layout `spingp_posterior` consumes and returns: pass
        ``mean`` as its ``prior_mean`` (it assumes a zero-mean prior
        otherwise) and hand its output to `from_precision_form`. The
        natural location parameter, if needed, is $\eta = \Lambda \mu$,
        i.e. ``precision.mv(mean)``.

        Returns:
            Tuple ``(mean, precision)``: the flattened marginal means of
            shape ``(T * d,)`` and the joint precision.
        """
        mean = rearrange(self.mean, "T d -> (T d)")
        return mean, self.precision

    @classmethod
    def from_precision_form(
        cls,
        mean: Float[Array, " Td"] | Float[Array, "T d"],
        precision: BlockTriDiag,
    ) -> MarkovGaussian:
        r"""Recover the generative chain from ``(mean, precision)``.

        One `udl_decomposition` pass gives $A_k = -U_k^{\top}$ and
        $Q_k = \tilde{D}_k^{-1}$; the offsets follow from the mean chain
        as $b_k = m_{k+1} - A_k m_k$. Round-trips `to_precision_form`
        exactly (to floating point).

        Args:
            mean: Marginal means, flattened ``(T * d,)`` or ``(T, d)``.
            precision: Symmetric positive-definite joint precision.

        Returns:
            The equivalent `MarkovGaussian`.
        """
        T, d = precision._num_blocks, precision._block_size
        means = mean if mean.ndim == 2 else rearrange(mean, "(T d) -> T d", T=T, d=d)
        A, Q, _ = udl_to_ssm_params(udl_decomposition(precision))
        b = means[1:] - einsum(A, means[:-1], "T i j, T j -> T i")
        return cls(A, Q[1:], means[0], Q[0], b=b)

    # ------------------------------------------------------------------
    # Density and sampling
    # ------------------------------------------------------------------

    def _chol_factors(self) -> tuple[Float[Array, "d d"], Float[Array, "Tm1 d d"]]:
        return jnp.linalg.cholesky(self.P0), jnp.linalg.cholesky(self.Q)

    def _log_prob_single(self, xs: Float[Array, "T d"]) -> Float[Array, ""]:
        r"""$\log \mathcal{N}(x_0; \mu_0, P_0)
        + \sum_k \log \mathcal{N}(x_{k+1}; A_k x_k + b_k, Q_k)$ at $O(T d^3)$."""
        chol_P0, chol_Q = self._chol_factors()
        residuals = xs[1:] - einsum(self.A, xs[:-1], "T i j, T j -> T i") - self.b
        lp_0 = _chol_log_prob(chol_P0, xs[0] - self.mu0)
        lp_rest = jax.vmap(_chol_log_prob)(chol_Q, residuals)
        return lp_0 + jnp.sum(lp_rest)

    @validate_sample
    def log_prob(self, value: Float[Array, "*batch T d"]) -> Float[Array, "*batch"]:
        leading_shape = value.shape[:-2]
        value_flat = rearrange(value, "... T d -> (...) T d")
        log_prob_flat = jax.vmap(self._log_prob_single)(value_flat)
        return _reshape_batch(log_prob_flat, leading_shape)

    def _sample_single(self, key: jax.Array) -> Float[Array, "T d"]:
        r"""One ancestral draw $x_0 \sim \mathcal{N}(\mu_0, P_0)$,
        $x_{k+1} = A_k x_k + b_k + L_k z_k$ at $O(T d^3)$."""
        chol_P0, chol_Q = self._chol_factors()
        T, d = self.horizon, self.state_dim
        k0, k_rest = jax.random.split(key)
        x0 = self.mu0 + chol_P0 @ jax.random.normal(k0, (d,))
        eps = jax.random.normal(k_rest, (T - 1, d))

        def _step(x, inputs):
            A_k, b_k, L_k, e_k = inputs
            x_next = A_k @ x + b_k + L_k @ e_k
            return x_next, x_next

        _, xs_rest = jax.lax.scan(_step, x0, (self.A, self.b, chol_Q, eps))
        return jnp.concatenate([x0[None], xs_rest], axis=0)

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample T d"]:
        if key is None:
            raise ValueError("PRNG key must be provided to sample from MarkovGaussian.")
        n_samples = math.prod(sample_shape) if sample_shape else 1
        keys = jax.random.split(key, n_samples)
        samples_flat = jax.vmap(self._sample_single)(keys)
        return _reshape_series(samples_flat, sample_shape)
