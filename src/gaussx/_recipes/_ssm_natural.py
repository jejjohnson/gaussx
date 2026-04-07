"""SSM <-> natural/expectation parameter transformations for Gauss-Markov models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._operators._block_tridiag import BlockTriDiag


def ssm_to_naturals(
    A: Float[Array, "Nm1 d d"],
    Q: Float[Array, "N d d"],
    mu_0: Float[Array, " d"],
    P_0: Float[Array, "d d"],
) -> tuple[Float[Array, " Nd"], BlockTriDiag]:
    r"""Convert SSM parameters to natural parameters.

    For a linear-Gaussian state-space model::

        x_0 \sim N(\mu_0, P_0)
        x_{k+1} = A_k x_k + \epsilon_k,\quad \epsilon_k \sim N(0, Q_{k+1})

    the joint prior ``p(x_0, \ldots, x_{N-1})`` has a block-tridiagonal
    precision matrix. This function returns its natural parameters
    ``(\theta_1, \theta_2)`` where ``\theta_2 = -\tfrac{1}{2}\Lambda``
    (matching the convention in :func:`gaussx.mean_cov_to_natural`).

    Args:
        A: Transition matrices, shape ``(N-1, d, d)``.
        Q: Process noise covariances, shape ``(N, d, d)``.
            ``Q[0]`` must equal ``P_0`` and ``Q[k]`` for ``k >= 1`` is the
            process noise at step ``k``.
        mu_0: Initial mean, shape ``(d,)``.
        P_0: Initial covariance, shape ``(d, d)``.

    Returns:
        Tuple ``(theta_linear, theta_precision)`` where
        ``theta_linear`` has shape ``(N*d,)`` and
        ``theta_precision`` is a :class:`~gaussx.BlockTriDiag`
        in the ``eta_2 = -0.5 * Lambda`` convention.
    """
    N = Q.shape[0]
    d = Q.shape[1]

    try:
        q0_matches_p0 = bool(jnp.allclose(Q[0], P_0))
    except jax.errors.TracerBoolConversionError:
        q0_matches_p0 = True  # skip validation under jax.jit

    if not q0_matches_p0:
        msg = "Q[0] must match P_0 so the returned natural parameters are consistent"
        raise ValueError(msg)

    # Invert all process noise covariances
    Q_inv = jnp.linalg.inv(Q)  # (N, d, d)
    P_0_inv = jnp.linalg.inv(P_0)

    # Future contributions: A_k^T Q_{k+1}^{-1} A_k for k = 0..N-2
    future = jax.vmap(lambda Ak, Qinv_kp1: Ak.T @ Qinv_kp1 @ Ak)(
        A, Q_inv[1:]
    )  # (N-1, d, d)

    # Precision diagonal blocks (raw Lambda, not eta2)
    # D[0] = P_0^{-1} + A[0]^T Q[1]^{-1} A[0]
    # D[k] = Q[k]^{-1} + A[k]^T Q[k+1]^{-1} A[k]  for k=1..N-2
    # D[N-1] = Q[N-1]^{-1}
    diag = jnp.zeros((N, d, d), dtype=Q.dtype)
    diag = diag.at[0].set(P_0_inv + future[0] if N > 1 else P_0_inv)
    if N > 2:
        diag = diag.at[1:-1].set(Q_inv[1:-1] + future[1:])
    diag = diag.at[-1].set(Q_inv[-1])

    # Sub-diagonal blocks (raw precision off-diagonal)
    # S[k] = -Q[k+1]^{-1} A[k]  for k=0..N-2
    # (negative because precision cross-terms are negative for transitions)
    sub_diag = jax.vmap(lambda Qinv_kp1, Ak: -Qinv_kp1 @ Ak)(
        Q_inv[1:], A
    )  # (N-1, d, d)

    # Convert to eta2 convention: theta_precision = -0.5 * Lambda
    theta_precision = BlockTriDiag(-0.5 * diag, -0.5 * sub_diag)

    # Linear natural parameter: eta1 = Lambda @ mu
    # For zero-mean transitions, only the initial condition contributes
    theta_linear = jnp.zeros(N * d, dtype=Q.dtype)
    eta1_0 = jnp.linalg.solve(P_0, mu_0)
    theta_linear = theta_linear.at[:d].set(eta1_0)

    return theta_linear, theta_precision


def naturals_to_ssm(
    theta_linear: Float[Array, " Nd"],
    theta_precision: BlockTriDiag,
) -> tuple[
    Float[Array, "Nm1 d d"],
    Float[Array, "N d d"],
    Float[Array, " d"],
    Float[Array, "d d"],
]:
    r"""Convert natural parameters back to SSM parameters.

    Recovers ``(A, Q, \mu_0, P_0)`` from the block-tridiagonal natural
    parameters via a backward recurrence on the precision blocks.

    Args:
        theta_linear: Natural location parameter, shape ``(N*d,)``.
        theta_precision: Natural precision parameter as
            :class:`~gaussx.BlockTriDiag` (eta2 convention).

    Returns:
        Tuple ``(A, Q, mu_0, P_0)`` where:
        - ``A``: Transition matrices, shape ``(N-1, d, d)``.
        - ``Q``: Process noise covariances, shape ``(N, d, d)``.
        - ``mu_0``: Initial mean, shape ``(d,)``.
        - ``P_0``: Initial covariance, shape ``(d, d)``.
    """
    d = theta_precision._block_size

    # Convert from eta2 to raw precision
    prec_diag = -2.0 * theta_precision.diagonal  # (N, d, d)
    prec_sub = -2.0 * theta_precision.sub_diagonal  # (N-1, d, d)

    # Backward recurrence to recover Q and A
    # Start from last block: Q[N-1] = inv(prec_diag[N-1])
    # Then for k = N-2 down to 0:
    #   A[k] = Q[k+1] @ (-prec_sub[k])  (sub-diag was -Q_{k+1}^{-1} A_k)
    #   Q[k] = inv(prec_diag[k] - A[k]^T @ Q[k+1]^{-1} @ A[k])

    def _backward_step(Q_next_inv, inputs):
        diag_k, sub_k = inputs
        Q_next = jnp.linalg.inv(Q_next_inv)
        # sub_k = -Q_{k+1}^{-1} A_k, so A_k = -Q_{k+1} @ sub_k
        A_k = -Q_next @ sub_k
        # Q_k^{-1} = diag_k - A_k^T @ Q_next_inv @ A_k
        Q_k_inv = diag_k - A_k.T @ Q_next_inv @ A_k
        return Q_k_inv, (A_k, Q_k_inv)

    Q_last_inv = prec_diag[-1]

    # Reverse scan: iterate from k=N-2 down to 0
    _, (A_rev, Q_inv_rev) = jax.lax.scan(
        _backward_step,
        Q_last_inv,
        (prec_diag[:-1], prec_sub),
        reverse=True,
    )

    # A_rev is (N-1, d, d), Q_inv_rev is (N-1, d, d) for k=0..N-2
    A = A_rev

    # Q: invert all Q_inv values
    Q_inv_all = jnp.concatenate([Q_inv_rev, Q_last_inv[None]], axis=0)
    Q = jnp.linalg.inv(Q_inv_all)

    # Recover initial conditions
    P_0 = Q[0]
    mu_0 = P_0 @ theta_linear[:d]

    return A, Q, mu_0, P_0


def ssm_to_expectations(
    means: Float[Array, "N d"],
    covs: Float[Array, "N d d"],
    cross_covs: Float[Array, "Nm1 d d"],
) -> tuple[Float[Array, " Nd"], BlockTriDiag]:
    r"""Convert SSM marginals to expectation parameters.

    Given filtered or smoothed marginals, computes the expectation
    parameters ``(eta1, eta2)`` of the joint Gaussian where:

    - ``eta1 = E[x]`` (concatenated means)
    - ``eta2`` is a :class:`~gaussx.BlockTriDiag` storing the
      block-tridiagonal subset of ``E[xx^T]`` (second moments matching
      the Gauss-Markov sparsity pattern, not the full dense matrix)

    The diagonal blocks of ``eta2`` are ``E[x_k x_k^T] = P_k + m_k m_k^T``
    and the sub-diagonal blocks are
    ``E[x_{k+1} x_k^T] = C_k + m_{k+1} m_k^T`` where ``C_k`` is the
    cross-covariance ``Cov(x_{k+1}, x_k)``.

    Args:
        means: Marginal means, shape ``(N, d)``.
        covs: Marginal covariances, shape ``(N, d, d)``.
        cross_covs: Cross-covariances ``Cov(x_{k+1}, x_k)``,
            shape ``(N-1, d, d)``.

    Returns:
        Tuple ``(eta1, eta2)`` where ``eta1`` has shape ``(N*d,)``
        and ``eta2`` is a :class:`~gaussx.BlockTriDiag`.
    """
    N, d = means.shape

    # eta1 = concatenated means
    eta1 = means.reshape(N * d)

    # Diagonal blocks: E[x_k x_k^T] = P_k + m_k m_k^T
    diag = covs + jax.vmap(jnp.outer)(means, means)  # (N, d, d)

    # Sub-diagonal blocks: E[x_{k+1} x_k^T] = C_k + m_{k+1} m_k^T
    sub_diag = cross_covs + jax.vmap(jnp.outer)(means[1:], means[:-1])  # (N-1, d, d)

    eta2 = BlockTriDiag(diag, sub_diag)
    return eta1, eta2


def expectations_to_ssm(
    eta1: Float[Array, " Nd"],
    eta2: BlockTriDiag,
) -> tuple[
    Float[Array, "N d"],
    Float[Array, "N d d"],
    Float[Array, "Nm1 d d"],
]:
    r"""Convert expectation parameters back to SSM marginals.

    Recovers ``(means, covs, cross_covs)`` from the expectation
    parameters of the joint Gaussian.

    Args:
        eta1: Concatenated means, shape ``(N*d,)``.
        eta2: Second-moment :class:`~gaussx.BlockTriDiag`.

    Returns:
        Tuple ``(means, covs, cross_covs)`` where:

        - ``means``: shape ``(N, d)``
        - ``covs``: shape ``(N, d, d)``
        - ``cross_covs``: shape ``(N-1, d, d)``
    """
    d = eta2._block_size
    N = eta2._num_blocks

    means = eta1.reshape(N, d)

    # covs = E[x_k x_k^T] - m_k m_k^T
    covs = eta2.diagonal - jax.vmap(jnp.outer)(means, means)

    # cross_covs = E[x_{k+1} x_k^T] - m_{k+1} m_k^T
    cross_covs = eta2.sub_diagonal - jax.vmap(jnp.outer)(means[1:], means[:-1])

    return means, covs, cross_covs
