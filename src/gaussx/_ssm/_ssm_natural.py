"""Natural and expectation parameters of a Gauss-Markov chain.

The chain

    x₀ ~ 𝒩(μ_x,0, Σ_x,0),    x_t+1 = F_t x_t + η_t+1,    η_t+1 ~ 𝒩(0, Σ_η,t+1),

for t = 0, …, T − 2 (T blocks of size D_x, the ``N`` axis of the shapes)
has a block-tridiagonal joint precision Λ_x. Its natural parameters are
(Λ_x μ_x, −½ Λ_x), in the convention of `gaussx.mean_cov_to_natural`, and
its expectation parameters (𝔼[x], 𝔼[x xᵀ]) restricted to the same band.
Both are held as a `BlockTriDiag`, so no conversion forms a dense
T D_x × T D_x matrix. F_t maps x_t to x_t+1, so it is the array ``A[t]``,
as in ``_udl.py``.

Dense Gaussian conversions are in ``_expfam/_natural.py`` and the per-site
EP ones in ``_site_natural.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._deprecation import warn_deprecated
from gaussx._einx import einsum, rearrange
from gaussx._linalg._linalg import solve_matrix
from gaussx._operators._block_tridiag import BlockTriDiag
from gaussx._primitives._inv import inv
from gaussx._strategies._base import AbstractSolverStrategy


def _transition_noise(
    A: Float[Array, "Nm1 d d"],
    Q: Float[Array, "Nm1 d d"] | Float[Array, "N d d"],
    P_0: Float[Array, "d d"],
    function: str,
) -> tuple[Float[Array, "d d"], Float[Array, "Nm1 d d"]]:
    """Normalise ``Q`` to the transition-noise layout (gh-364).

    The `gaussx.MarkovGaussian` layout, ``Q`` of shape ``(N-1, d, d)`` with
    ``Q[t]`` = Σ_η,t+1 driving x_t → x_t+1 and Σ_x,0 passed separately, is
    the one to use. The older stacked layout of shape ``(N, d, d)`` with
    ``Q[0]`` = Σ_x,0 is told apart by its length, still accepted with a
    ``DeprecationWarning``, and checked for consistency.

    Args:
        A: Transitions F_t, one per transition.
        Q: Model-error covariances Σ_η,t+1, in either layout.
        P_0: Prior covariance Σ_x,0.
        function: Name of the caller, for the messages.

    Returns:
        Tuple ``(P_0, Q)`` holding Σ_x,0 and the transition noise only.

    Raises:
        ValueError: If ``Q`` matches neither layout.
    """
    n_transitions = A.shape[0]
    if Q.shape[0] == n_transitions:
        return P_0, Q
    if Q.shape[0] != n_transitions + 1:
        msg = (
            f"{function}: Q must have {n_transitions} transition-noise blocks "
            f"to match A, got {Q.shape[0]}."
        )
        raise ValueError(msg)
    warn_deprecated(
        f"{function} with Q of shape (N, d, d) and Q[0] == P_0 is deprecated; "
        "pass only the transition noise Q[1:] (shape (N-1, d, d)), as "
        "MarkovGaussian does. The stacked layout will be removed in gaussx 0.7.0."
    )
    # ``eqx.error_if`` so the check also runs under jit / vmap / grad; a
    # Python ``bool()`` could only run eagerly (gh-359). Attached to both
    # P_0 and Q so every output depends on it and a jitted projection cannot
    # dead-code-eliminate it.
    P_0, Q = eqx.error_if(
        (P_0, Q),
        ~jnp.allclose(Q[0], P_0),
        "Q[0] must match P_0 so the returned natural parameters are consistent",
    )
    return P_0, Q[1:]


def _dense_precision(covariance: Float[Array, "d d"]) -> Float[Array, "d d"]:
    """Dense inverse of one covariance block through `gaussx.inv`.

    Args:
        covariance: Symmetric positive-definite block Σ.

    Returns:
        The dense precision Σ⁻¹.
    """
    covariance_op = lx.MatrixLinearOperator(covariance, lx.positive_semidefinite_tag)
    # Σ⁻¹
    return inv(covariance_op).as_matrix()


def _chain_precision_blocks(
    diag: Float[Array, "N d d"],
    prior_precision: Float[Array, "d d"],
    noise_precisions: Float[Array, "Nm1 d d"],
    A: Float[Array, "Nm1 d d"],
) -> tuple[Float[Array, "N d d"], Float[Array, "Nm1 d d"]]:
    """Diagonal and sub-diagonal blocks of a chain precision Λ_x, for T ≥ 2.

        [Λ_x]_0,0 = Σ_x,0⁻¹ + F_0ᵀ Σ_η,1⁻¹ F_0
        [Λ_x]_t,t = Σ_η,t⁻¹ + F_tᵀ Σ_η,t+1⁻¹ F_t,    0 < t < T − 1
        [Λ_x]_T−1,T−1 = Σ_η,T−1⁻¹
        [Λ_x]_t+1,t = −Σ_η,t+1⁻¹ F_t

    Args:
        diag: Zero blocks to fill, one per state.
        prior_precision: Prior precision Σ_x,0⁻¹.
        noise_precisions: Model-error precisions Σ_η,t+1⁻¹.
        A: Transitions F_t, with ``A[t]`` mapping x_t to x_t+1.

    Returns:
        Tuple ``(diag, sub_diag)`` holding [Λ_x]_t,t and [Λ_x]_t+1,t.
    """
    num_blocks = diag.shape[0]
    # F_tᵀ Σ_η,t+1⁻¹
    transition_precision = einsum(
        A, noise_precisions, "block next prev, block next col -> block prev col"
    )
    # F_tᵀ Σ_η,t+1⁻¹ F_t
    future = einsum(
        transition_precision, A, "block prev next, block next col -> block prev col"
    )
    # [Λ_x]_0,0 = Σ_x,0⁻¹ + F_0ᵀ Σ_η,1⁻¹ F_0
    diag = diag.at[0].set(prior_precision + future[0])
    if num_blocks > 2:
        # [Λ_x]_t,t = Σ_η,t⁻¹ + F_tᵀ Σ_η,t+1⁻¹ F_t, for 0 < t < T − 1
        diag = diag.at[1:-1].set(noise_precisions[:-1] + future[1:])
    # [Λ_x]_T−1,T−1 = Σ_η,T−1⁻¹
    diag = diag.at[-1].set(noise_precisions[-1])

    # −Σ_η,t+1⁻¹
    negated_noise_precisions = -noise_precisions
    # [Λ_x]_t+1,t = −Σ_η,t+1⁻¹ F_t
    sub_diag = einsum(
        negated_noise_precisions,
        A,
        "block next mid, block mid prev -> block next prev",
    )
    return diag, sub_diag


def ssm_to_naturals(
    A: Float[Array, "Nm1 d d"],
    Q: Float[Array, "Nm1 d d"] | Float[Array, "N d d"],
    mu_0: Float[Array, " d"],
    P_0: Float[Array, "d d"],
    *,
    solver: AbstractSolverStrategy | None = None,
) -> tuple[Float[Array, " Nd"], BlockTriDiag]:
    """Natural parameters (Λ_x μ_x, −½ Λ_x) of a Gauss-Markov chain.

        [Λ_x]_0,0 = Σ_x,0⁻¹ + F_0ᵀ Σ_η,1⁻¹ F_0
        [Λ_x]_t,t = Σ_η,t⁻¹ + F_tᵀ Σ_η,t+1⁻¹ F_t,    0 < t < T − 1
        [Λ_x]_T−1,T−1 = Σ_η,T−1⁻¹
        [Λ_x]_t+1,t = −Σ_η,t+1⁻¹ F_t
        Λ_x μ_x = (Σ_x,0⁻¹ μ_x,0, 0, …, 0)

    for the chain of the module docstring with T blocks; the transitions
    have zero mean, so only the first block of Λ_x μ_x is non-zero. With
    T = 1 the only block is [Λ_x]_0,0 = Σ_x,0⁻¹.

    Algorithm:
        1. Invert each model-error covariance Σ_η,t+1.
        2. Solve Σ_x,0 against [I | μ_x,0] once for Σ_x,0⁻¹ and
           Σ_x,0⁻¹ μ_x,0 (gh-403).
        3. Assemble the diagonal and sub-diagonal blocks of Λ_x.
        4. Scale both by −½ and place Σ_x,0⁻¹ μ_x,0 in the first block of
           Λ_x μ_x.

    Args:
        A: Transitions F_t, with ``A[t]`` mapping x_t to x_t+1.
        Q: Model-error covariances, with ``Q[t]`` = Σ_η,t+1 driving
            x_t → x_t+1, as in `gaussx.MarkovGaussian`. The older stacked
            layout of shape ``(N, d, d)`` with ``Q[0]`` = Σ_x,0 (checked with
            `equinox.error_if`, so also under ``jax.jit``) is deprecated and
            will be removed in gaussx 0.7.0.
        mu_0: Prior mean μ_x,0 of the initial state x₀.
        P_0: Prior covariance Σ_x,0 of x₀.
        solver: Solver strategy for the solve against Σ_x,0; ``None`` uses
            structural dispatch.

    Returns:
        Tuple ``(theta_linear, theta_precision)`` holding Λ_x μ_x, stacked
        over the T blocks, and −½ Λ_x as a `BlockTriDiag`.

    Raises:
        ValueError: If ``Q`` has neither T − 1 nor T blocks.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transitions = jnp.broadcast_to(0.9 * jnp.eye(2), (3, 2, 2))
        >>> model_noise = jnp.broadcast_to(0.1 * jnp.eye(2), (3, 2, 2))
        >>> prior = (jnp.array([1.0, -1.0]), jnp.eye(2))
        >>> linear, precision = gaussx.ssm_to_naturals(
        ...     transitions, model_noise, *prior
        ... )
        >>> linear.shape, precision.as_matrix().shape
        ((8,), (8, 8))

        The last diagonal block of −½ Λ_x is −½ Σ_η,T−1⁻¹:

        >>> bool(jnp.allclose(precision.diagonal[-1], -5.0 * jnp.eye(2), atol=1e-5))
        True

        The chain round-trips through `naturals_to_ssm`:

        >>> rebuilt = gaussx.naturals_to_ssm(linear, precision, initial_in_q=False)
        >>> bool(jnp.allclose(rebuilt[0], transitions, atol=1e-5))
        True
    """
    P_0, Q = _transition_noise(A, Q, P_0, "ssm_to_naturals")
    num_blocks = Q.shape[0] + 1
    state_size = P_0.shape[0]

    # --- model-error precisions ---
    # Only the transition noise is inverted here; Σ_x,0 gets a single
    # factorisation below (gh-403).
    # Σ_η,t+1⁻¹
    noise_precisions = jax.vmap(_dense_precision)(Q)

    # --- prior precision and precision-weighted mean ---
    prior_covariance_op = lx.MatrixLinearOperator(P_0, lx.positive_semidefinite_tag)
    identity = jnp.eye(state_size, dtype=Q.dtype)
    # [I | μ_x,0]ᵀ, stacked as rows
    rhs_transpose = jnp.concatenate([identity, rearrange(mu_0, "state -> 1 state")])
    # [I | μ_x,0]
    rhs = rearrange(rhs_transpose, "col state -> state col")
    # Σ_x,0⁻¹ [I | μ_x,0]
    prior_solved = solve_matrix(prior_covariance_op, rhs, solver=solver)
    # Σ_x,0⁻¹
    prior_precision = prior_solved[:, :state_size]
    # Σ_x,0⁻¹ μ_x,0
    prior_precision_mean = prior_solved[:, state_size]

    diag = jnp.zeros((num_blocks, state_size, state_size), dtype=Q.dtype)
    if num_blocks == 1:
        # One block, so no transition to contract (einx rejects the empty
        # block axis).
        # [Λ_x]_0,0 = Σ_x,0⁻¹
        diag = diag.at[0].set(prior_precision)
        sub_diag = jnp.zeros(A.shape, dtype=jnp.result_type(noise_precisions, A))
    else:
        diag, sub_diag = _chain_precision_blocks(
            diag, prior_precision, noise_precisions, A
        )

    # --- natural parameters ---
    # −½ Λ_x
    theta_precision = BlockTriDiag(-0.5 * diag, -0.5 * sub_diag)
    # Λ_x μ_x = (Σ_x,0⁻¹ μ_x,0, 0, …, 0)
    theta_linear = jnp.zeros(num_blocks * state_size, dtype=Q.dtype)
    theta_linear = theta_linear.at[:state_size].set(prior_precision_mean)

    return theta_linear, theta_precision


def naturals_to_ssm(
    theta_linear: Float[Array, " Nd"],
    theta_precision: BlockTriDiag,
    *,
    solver: AbstractSolverStrategy | None = None,
    initial_in_q: bool | None = None,
) -> tuple[
    Float[Array, "Nm1 d d"],
    Float[Array, "Nm1 d d"] | Float[Array, "N d d"],
    Float[Array, " d"],
    Float[Array, "d d"],
]:
    """Gauss-Markov chain (F_t, Σ_η,t, μ_x,0, Σ_x,0) from its natural parameters.

        Σ_η,T−1⁻¹ = [Λ_x]_T−1,T−1
        F_t = −Σ_η,t+1 [Λ_x]_t+1,t
        Σ_η,t⁻¹ = [Λ_x]_t,t − F_tᵀ Σ_η,t+1⁻¹ F_t,    t ≥ 1
        Σ_x,0⁻¹ = [Λ_x]_0,0 − F_0ᵀ Σ_η,1⁻¹ F_0
        μ_x,0 = Σ_x,0 [Λ_x μ_x]_0

    The inverse of `ssm_to_naturals`: a backward recurrence through the
    Schur complements of Λ_x, as in `gaussx.udl_decomposition`. Only the
    first block [Λ_x μ_x]_0 of the linear parameter is read.

    Algorithm:
        1. Recover the precision Λ_x = −2 (−½ Λ_x).
        2. Start from Σ_η,T−1⁻¹ = [Λ_x]_T−1,T−1.
        3. For t = T − 2, …, 0, invert the carried precision to Σ_η,t+1,
           read off F_t and form the next Schur complement.
        4. Invert the last carry to Σ_x,0, for T factorisations in all
           (gh-403), and set μ_x,0 = Σ_x,0 [Λ_x μ_x]_0.

    Args:
        theta_linear: Precision-weighted mean Λ_x μ_x, stacked over the T
            blocks.
        theta_precision: Scaled precision −½ Λ_x as a `BlockTriDiag`.
        solver: Deprecated and ignored (the block inverses are dense);
            passing it warns, and it will be removed in gaussx 0.7.0.
        initial_in_q: Layout of the returned ``Q``. ``False`` returns only
            the transition noise Σ_η,t+1, shape ``(N-1, d, d)``, as
            `gaussx.MarkovGaussian` takes it; ``True`` the older stacked
            layout of shape ``(N, d, d)`` with ``Q[0]`` = Σ_x,0. The default
            ``None`` means ``True`` with a ``DeprecationWarning``; it becomes
            ``False`` in gaussx 0.7.0.

    Returns:
        Tuple ``(A, Q, mu_0, P_0)`` holding the transitions F_t, the
        model-error covariances in the layout chosen by ``initial_in_q``,
        the prior mean μ_x,0 and the prior covariance Σ_x,0.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> transitions = jnp.broadcast_to(0.9 * jnp.eye(2), (3, 2, 2))
        >>> model_noise = jnp.broadcast_to(0.1 * jnp.eye(2), (3, 2, 2))
        >>> prior = (jnp.array([1.0, -1.0]), jnp.eye(2))
        >>> naturals = gaussx.ssm_to_naturals(transitions, model_noise, *prior)
        >>> A, Q, mu_0, P_0 = gaussx.naturals_to_ssm(*naturals, initial_in_q=False)
        >>> A.shape, Q.shape
        ((3, 2, 2), (3, 2, 2))
        >>> bool(jnp.allclose(Q, model_noise, atol=1e-5))
        True
        >>> bool(jnp.allclose(mu_0, prior[0], atol=1e-5))
        True

        The stacked layout puts Σ_x,0 in front:

        >>> _, stacked, _, _ = gaussx.naturals_to_ssm(*naturals, initial_in_q=True)
        >>> stacked.shape
        (4, 2, 2)
    """
    if solver is not None:
        warn_deprecated(
            "naturals_to_ssm(solver=...) is deprecated and ignored; it will be "
            "removed in gaussx 0.7.0."
        )
    if initial_in_q is None:
        warn_deprecated(
            "naturals_to_ssm returns Q with P_0 stacked as Q[0]; in gaussx 0.7.0 it "
            "will return only the transition noise (shape (N-1, d, d)), as "
            "MarkovGaussian and ssm_to_naturals take it. Pass "
            "initial_in_q=False to adopt that layout now, or initial_in_q=True "
            "to keep the current one until then."
        )
        initial_in_q = True
    state_size = theta_precision._block_size

    # --- precision blocks ---
    # [Λ_x]_t,t = −2 (−½ Λ_x)_t,t
    prec_diag = -2.0 * theta_precision.diagonal
    # [Λ_x]_t+1,t = −2 (−½ Λ_x)_t+1,t
    prec_sub = -2.0 * theta_precision.sub_diagonal

    # --- backward recurrence ---
    def _backward_step(noise_precision_next, inputs):
        diag_block, sub_block = inputs
        noise_precision_next_op = lx.MatrixLinearOperator(
            noise_precision_next, lx.positive_semidefinite_tag
        )
        # Σ_η,t+1 = (Σ_η,t+1⁻¹)⁻¹
        noise_covariance_next = inv(noise_precision_next_op).as_matrix()
        # −Σ_η,t+1
        negated_noise_covariance = -noise_covariance_next
        # F_t = −Σ_η,t+1 [Λ_x]_t+1,t, since [Λ_x]_t+1,t = −Σ_η,t+1⁻¹ F_t
        transition = einsum(
            negated_noise_covariance, sub_block, "next mid, mid prev -> next prev"
        )
        # F_tᵀ Σ_η,t+1⁻¹
        transition_precision = einsum(
            transition, noise_precision_next, "next prev, next col -> prev col"
        )
        # F_tᵀ Σ_η,t+1⁻¹ F_t
        future = einsum(
            transition_precision, transition, "prev next, next col -> prev col"
        )
        # Σ_η,t⁻¹ = [Λ_x]_t,t − F_tᵀ Σ_η,t+1⁻¹ F_t (Σ_x,0⁻¹ at t = 0)
        noise_precision = diag_block - future
        # Emit Σ_η,t+1 so it is not inverted again below (gh-403).
        return noise_precision, (transition, noise_covariance_next)

    # Σ_η,T−1⁻¹ = [Λ_x]_T−1,T−1
    last_noise_precision = prec_diag[-1]
    prior_precision, (A, noise_covariances) = jax.lax.scan(
        _backward_step,
        last_noise_precision,
        (prec_diag[:-1], prec_sub),
        reverse=True,
    )

    # The scan inverted Σ_η,1:T−1; only Σ_x,0 (the final carry) is left:
    # T factorisations in total, not 2T − 1.
    # Σ_x,0 = (Σ_x,0⁻¹)⁻¹
    prior_precision_op = lx.MatrixLinearOperator(
        prior_precision, lx.positive_semidefinite_tag
    )
    prior_covariance = inv(prior_precision_op).as_matrix()
    # [Σ_x,0; Σ_η,1:T−1]
    Q = jnp.concatenate(
        [rearrange(prior_covariance, "row col -> 1 row col"), noise_covariances]
    )

    # --- initial conditions ---
    P_0 = Q[0]
    # μ_x,0 = Σ_x,0 [Λ_x μ_x]_0
    mu_0 = P_0 @ theta_linear[:state_size]

    return A, (Q if initial_in_q else Q[1:]), mu_0, P_0


def ssm_to_expectations(
    means: Float[Array, "N d"],
    covs: Float[Array, "N d d"],
    cross_covs: Float[Array, "Nm1 d d"],
) -> tuple[Float[Array, " Nd"], BlockTriDiag]:
    """Expectation parameters (𝔼[x], 𝔼[x xᵀ]) of a chain from its marginals.

        𝔼[x_t] = μ_x,t
        𝔼[x_t x_tᵀ] = Σ_x,t + μ_x,t μ_x,tᵀ
        𝔼[x_t+1 x_tᵀ] = [Σ_x]_t+1,t + μ_x,t+1 μ_x,tᵀ

    with the marginals 𝒩(μ_x,t, Σ_x,t) and the cross-covariances
    [Σ_x]_t+1,t = Cov(x_t+1, x_t) of filtered or smoothed states. The
    second moment is kept only on the block-tridiagonal band of the chain,
    not as the full dense matrix.

    Args:
        means: Marginal means μ_x,t.
        covs: Marginal covariances Σ_x,t.
        cross_covs: Cross-covariances [Σ_x]_t+1,t = Cov(x_t+1, x_t).

    Returns:
        Tuple ``(eta1, eta2)`` holding 𝔼[x], stacked over the T blocks, and
        the band of 𝔼[x xᵀ] as a `BlockTriDiag`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> means = jnp.ones((3, 2))
        >>> covs = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
        >>> cross_covs = jnp.zeros((2, 2, 2))
        >>> first, second = gaussx.ssm_to_expectations(means, covs, cross_covs)
        >>> first.shape, second.as_matrix().shape
        ((6,), (6, 6))
        >>> bool(jnp.allclose(second.diagonal[0], jnp.eye(2) + 1.0))
        True

        Independent states with unit means have 𝔼[x_t+1 x_tᵀ] = 1:

        >>> bool(jnp.allclose(second.sub_diagonal, 1.0))
        True
    """
    # 𝔼[x] = (μ_x,0, …, μ_x,T−1)
    eta1 = rearrange(means, "N d -> (N d)")

    # μ_x,t μ_x,tᵀ
    mean_outer = einsum(means, means, "block row, block col -> block row col")
    # 𝔼[x_t x_tᵀ] = Σ_x,t + μ_x,t μ_x,tᵀ
    diag = covs + mean_outer

    # μ_x,t+1 μ_x,tᵀ
    lagged_mean_outer = einsum(
        means[1:], means[:-1], "block row, block col -> block row col"
    )
    # 𝔼[x_t+1 x_tᵀ] = [Σ_x]_t+1,t + μ_x,t+1 μ_x,tᵀ
    sub_diag = cross_covs + lagged_mean_outer

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
    """Marginals of a chain from its expectation parameters (𝔼[x], 𝔼[x xᵀ]).

        μ_x,t = 𝔼[x_t]
        Σ_x,t = 𝔼[x_t x_tᵀ] − μ_x,t μ_x,tᵀ
        [Σ_x]_t+1,t = 𝔼[x_t+1 x_tᵀ] − μ_x,t+1 μ_x,tᵀ

    The inverse of `ssm_to_expectations`.

    Args:
        eta1: First moment 𝔼[x], stacked over the T blocks.
        eta2: Band of the second moment 𝔼[x xᵀ] as a `BlockTriDiag`.

    Returns:
        Tuple ``(means, covs, cross_covs)`` holding μ_x,t, Σ_x,t and
        [Σ_x]_t+1,t = Cov(x_t+1, x_t).

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> means = jnp.array([[0.5, -1.0], [1.0, 0.0], [0.0, 2.0]])
        >>> covs = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
        >>> cross_covs = jnp.broadcast_to(0.3 * jnp.eye(2), (2, 2, 2))
        >>> expectations = gaussx.ssm_to_expectations(means, covs, cross_covs)
        >>> rebuilt = gaussx.expectations_to_ssm(*expectations)
        >>> [r.shape for r in rebuilt]
        [(3, 2), (3, 2, 2), (2, 2, 2)]
        >>> bool(jnp.allclose(rebuilt[2], cross_covs, atol=1e-6))
        True
    """
    state_size = eta2._block_size
    num_blocks = eta2._num_blocks

    # μ_x,t = 𝔼[x_t]
    means = rearrange(eta1, "(N d) -> N d", N=num_blocks, d=state_size)

    # μ_x,t μ_x,tᵀ
    mean_outer = einsum(means, means, "block row, block col -> block row col")
    # Σ_x,t = 𝔼[x_t x_tᵀ] − μ_x,t μ_x,tᵀ
    covs = eta2.diagonal - mean_outer

    # μ_x,t+1 μ_x,tᵀ
    lagged_mean_outer = einsum(
        means[1:], means[:-1], "block row, block col -> block row col"
    )
    # [Σ_x]_t+1,t = 𝔼[x_t+1 x_tᵀ] − μ_x,t+1 μ_x,tᵀ
    cross_covs = eta2.sub_diagonal - lagged_mean_outer

    return means, covs, cross_covs
