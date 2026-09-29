"""SSM <-> natural/expectation parameter transformations for Gauss-Markov models.

These functions convert between Gauss-Markov state-space model (SSM) parameters
and natural/expectation parameterizations of the joint Gaussian, exploiting the
block-tridiagonal sparsity of the precision matrix.

For **general-purpose** (dense or operator-based) Gaussian parameterization
conversions see `gaussx._expfam._natural`. For **per-site** (scalar/
diagonal EP) conversions see `gaussx._ssm._site_natural`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange
from gaussx._linalg._linalg import solve_matrix
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._primitives._inv import inv
from gaussx._strategies._base import AbstractSolverStrategy


def ssm_to_naturals(
    A: Float[Array, "Nm1 d d"],
    Q: Float[Array, "N d d"],
    mu_0: Float[Array, " d"],
    P_0: Float[Array, "d d"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " Nd"], BlockTriDiag]:
    r"""Convert SSM parameters to natural parameters.

    For a linear-Gaussian state-space model:

        x_0 \sim N(\mu_0, P_0)
        x_{k+1} = A_k x_k + \epsilon_k,\quad \epsilon_k \sim N(0, Q_{k+1})

    the joint prior ``p(x_0, \ldots, x_{N-1})`` has a block-tridiagonal
    precision matrix. This function returns its natural parameters
    ``(\theta_1, \theta_2)`` where ``\theta_2 = -\tfrac{1}{2}\Lambda``
    (matching the convention in `gaussx.mean_cov_to_natural`).

    Args:
        A: Transition matrices, shape ``(N-1, d, d)``.
        Q: Process noise covariances, shape ``(N, d, d)``.
            ``Q[0]`` must equal ``P_0`` (checked with `equinox.error_if`,
            so also under ``jax.jit`` / ``jax.vmap``) and ``Q[k]`` for
            ``k >= 1`` is the process noise at step ``k``.
        mu_0: Initial mean, shape ``(d,)``.
        P_0: Initial covariance, shape ``(d, d)``.
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch.

    Returns:
        Tuple ``(theta_linear, theta_precision)`` where
        ``theta_linear`` has shape ``(N*d,)`` and
        ``theta_precision`` is a `BlockTriDiag`
        in the ``eta_2 = -0.5 * Lambda`` convention.
    """
    N = Q.shape[0]
    d = Q.shape[1]

    # ``eqx.error_if`` so the check also runs under jit / vmap / grad; a
    # Python ``bool()`` could only run eagerly (gh-359). Attached to both
    # P_0 and Q so every output depends on it and a jitted projection cannot
    # dead-code-eliminate it: theta_linear and the first diagonal block use
    # P_0 (the only live input when N = 1), the sub-diagonal uses only
    # A and Q[1:].
    P_0, Q = eqx.error_if(
        (P_0, Q),
        ~jnp.allclose(Q[0], P_0),
        "Q[0] must match P_0 so the returned natural parameters are consistent",
    )

    # Invert the transition noise Q[1:] only; Q[0] is P_0, handled below
    # with a single factorisation (gh-403).
    def _inv_single(q):
        return inv(lx.MatrixLinearOperator(q, lx.positive_semidefinite_tag)).as_matrix()

    Q_inv = jax.vmap(_inv_single)(Q[1:])  # (N-1, d, d): Q_inv[k] = Q[k+1]^{-1}

    # P_0^{-1} and P_0^{-1} mu_0 from one solve against [I | mu_0].
    P_0_op = lx.MatrixLinearOperator(P_0, lx.positive_semidefinite_tag)
    rhs = jnp.concatenate([jnp.eye(d, dtype=Q.dtype), mu_0[:, None]], axis=1)
    P_0_solved = solve_matrix(P_0_op, rhs, solver=solver)
    P_0_inv, eta1_0 = P_0_solved[:, :d], P_0_solved[:, d]

    # Future contributions: A_k^T Q_{k+1}^{-1} A_k for k = 0..N-2
    future = jax.vmap(lambda Ak, Qinv_kp1: Ak.T @ Qinv_kp1 @ Ak)(
        A, Q_inv
    )  # (N-1, d, d)

    # Precision diagonal blocks (raw Lambda, not eta2)
    # D[0] = P_0^{-1} + A[0]^T Q[1]^{-1} A[0]
    # D[k] = Q[k]^{-1} + A[k]^T Q[k+1]^{-1} A[k]  for k=1..N-2
    # D[N-1] = Q[N-1]^{-1}
    diag = jnp.zeros((N, d, d), dtype=Q.dtype)
    diag = diag.at[0].set(P_0_inv + future[0] if N > 1 else P_0_inv)
    if N > 2:
        diag = diag.at[1:-1].set(Q_inv[:-1] + future[1:])
    if N > 1:
        # For N = 1 the only block is the initial one, P_0^{-1}.
        diag = diag.at[-1].set(Q_inv[-1])

    # Sub-diagonal blocks (raw precision off-diagonal)
    # S[k] = -Q[k+1]^{-1} A[k]  for k=0..N-2
    # (negative because precision cross-terms are negative for transitions)
    sub_diag = jax.vmap(lambda Qinv_kp1, Ak: -Qinv_kp1 @ Ak)(Q_inv, A)  # (N-1, d, d)

    # Convert to eta2 convention: theta_precision = -0.5 * Lambda
    theta_precision = BlockTriDiag(-0.5 * diag, -0.5 * sub_diag)

    # Linear natural parameter: eta1 = Lambda @ mu
    # For zero-mean transitions, only the initial condition contributes
    theta_linear = jnp.zeros(N * d, dtype=Q.dtype)
    theta_linear = theta_linear.at[:d].set(eta1_0)

    return theta_linear, theta_precision


def naturals_to_ssm(
    theta_linear: Float[Array, " Nd"],
    theta_precision: BlockTriDiag,
    *,
    solver: AbstractSolverStrategy | None = None,
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
            `BlockTriDiag` (eta2 convention).
        solver: Optional solver strategy for structured linear algebra.
            When ``None``, falls back to structural dispatch. This parameter
            is accepted for API consistency but is not currently used by the
            matrix inverse operations in this function.

    Returns:
        Tuple ``(A, Q, mu_0, P_0)`` where:
        - ``A``: Transition matrices, shape ``(N-1, d, d)``.
        - ``Q``: Process noise covariances, shape ``(N, d, d)``.
        - ``mu_0``: Initial mean, shape ``(d,)``.
        - ``P_0``: Initial covariance, shape ``(d, d)``.
    """
    del solver  # inv does not accept a solver; parameter reserved for future use
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
        Q_next = inv(
            lx.MatrixLinearOperator(Q_next_inv, lx.positive_semidefinite_tag)
        ).as_matrix()
        # sub_k = -Q_{k+1}^{-1} A_k, so A_k = -Q_{k+1} @ sub_k
        A_k = -Q_next @ sub_k
        # Q_k^{-1} = diag_k - A_k^T @ Q_next_inv @ A_k
        Q_k_inv = diag_k - A_k.T @ Q_next_inv @ A_k
        # Emit Q_next (= Q[k+1]) so it is not inverted again below (gh-403).
        return Q_k_inv, (A_k, Q_next)

    Q_last_inv = prec_diag[-1]

    # Reverse scan: iterate from k=N-2 down to 0
    Q_0_inv, (A, Q_rest) = jax.lax.scan(
        _backward_step,
        Q_last_inv,
        (prec_diag[:-1], prec_sub),
        reverse=True,
    )

    # The scan inverted Q[1..N-1]; only Q[0] (the final carry) is left:
    # N factorisations in total, not 2N - 1.
    Q_0 = inv(lx.MatrixLinearOperator(Q_0_inv, lx.positive_semidefinite_tag))
    Q = jnp.concatenate([Q_0.as_matrix()[None], Q_rest], axis=0)

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
    - ``eta2`` is a `BlockTriDiag` storing the
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
        and ``eta2`` is a `BlockTriDiag`.
    """
    _N, _d = means.shape

    # eta1 = concatenated means
    eta1 = rearrange(means, "N d -> (N d)")

    # Diagonal blocks: E[xₖ xₖᵀ] = Pₖ + mₖ mₖᵀ
    diag = covs + einsum(means, means, "N i, N j -> N i j")  # (N, d, d)

    # Sub-diagonal blocks: E[xₖ₊₁ xₖᵀ] = Cₖ + mₖ₊₁ mₖᵀ
    sub_diag = cross_covs + einsum(
        means[1:], means[:-1], "N i, N j -> N i j"
    )  # (N-1, d, d)

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
        eta2: Second-moment `BlockTriDiag`.

    Returns:
        Tuple ``(means, covs, cross_covs)`` where:

        - ``means``: shape ``(N, d)``
        - ``covs``: shape ``(N, d, d)``
        - ``cross_covs``: shape ``(N-1, d, d)``
    """
    d = eta2._block_size
    N = eta2._num_blocks

    means = rearrange(eta1, "(N d) -> N d", N=N, d=d)

    # covs = E[xₖ xₖᵀ] − mₖ mₖᵀ
    covs = eta2.diagonal - einsum(means, means, "N i, N j -> N i j")

    # cross_covs = E[xₖ₊₁ xₖᵀ] − mₖ₊₁ mₖᵀ
    cross_covs = eta2.sub_diagonal - einsum(means[1:], means[:-1], "N i, N j -> N i j")

    return means, covs, cross_covs
