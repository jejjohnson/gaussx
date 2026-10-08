r"""Gaussian-path linearisation of a nonlinear SDE and the drift path-KL.

Variational treatments of a nonlinear Itô SDE

$$
dx = f(x)\,dt + \Sigma^{1/2}\,dW
$$

approximate its posterior by a Gaussian process $q$ with marginals
$q(x_t) = \mathcal{N}(m_t, S_t)$ and a linear drift $f_L(x) = A_t x + b_t$
(Archambeau, Cornford, Opper & Shawe-Taylor, 2007). Two quantities recur in
every such method: the linear drift that is optimal along the path, and the
path-KL between the linear and the nonlinear SDE. Both are expectations under
the Gaussian marginals, so both are evaluated with a gaussx integrator.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import rearrange, repeat
from gaussx._linalg._linalg import solve_rows
from gaussx._quadrature._expectations import mean_expectation
from gaussx._quadrature._integrator import AbstractIntegrator, moment_transform
from gaussx._quadrature._taylor import TaylorIntegrator
from gaussx._quadrature._types import GaussianState
from gaussx._quadrature._unscented import UnscentedIntegrator


class LinearizedSDE(eqx.Module):
    r"""Linear surrogate $dx = (A_t x + b_t)\,dt + Q_t^{1/2}\,dW$ of an SDE.

    Attributes:
        A: Linearised drift matrices $A_t$, shape ``(T, d, d)``.
        b: Linearised drift offsets $b_t$, shape ``(T, d)``.
        Q: Diffusion covariances $Q_t = \Sigma$ (unchanged by the
            linearisation), shape ``(T, d, d)``.
    """

    A: Float[Array, "T d d"]
    b: Float[Array, "T d"]
    Q: Float[Array, "T d d"]


def _default_integrator(integrator: AbstractIntegrator | None) -> AbstractIntegrator:
    if integrator is not None:
        return integrator
    # alpha=1.0 for the reason given in `moment_transform`: the 1e-3 default
    # recovers the moments by cancellation and is unsafe in float32.
    return UnscentedIntegrator(alpha=1.0)


def _check_path(
    path_means: Float[Array, "T d"], path_covs: Float[Array, "T d d"]
) -> tuple[int, int]:
    if path_means.ndim != 2:
        raise ValueError(f"path_means must have shape (T, d), got {path_means.shape}.")
    T, d = path_means.shape
    if path_covs.shape != (T, d, d):
        raise ValueError(
            f"path_covs must have shape (T, d, d) = {(T, d, d)}, got {path_covs.shape}."
        )
    return T, d


def linearize_sde(
    drift_fn: Callable[[Float[Array, " d"]], Float[Array, " d"]],
    diffusion: Float[Array, "d d"] | Float[Array, "T d d"],
    path_means: Float[Array, "T d"],
    path_covs: Float[Array, "T d d"],
    integrator: AbstractIntegrator | None = None,
) -> LinearizedSDE:
    r"""Optimal linear drift of a nonlinear SDE along a Gaussian path.

    For each marginal $q(x_t) = \mathcal{N}(m_t, S_t)$ of the path, the
    linear drift $f_L(x) = A_t x + b_t$ that minimises
    $\mathbb{E}_q\lVert f(x) - f_L(x)\rVert^2_{\Sigma^{-1}}$ (Archambeau
    et al., 2007) is

    $$
    A_t = \mathbb{E}_q\!\left[\frac{\partial f}{\partial x}\right]
        = \mathrm{Cov}_q\big(f(x), x\big)\, S_t^{-1},
    \qquad
    b_t = \mathbb{E}_q[f(x)] - A_t m_t.
    $$

    The second form of $A_t$ is Stein's lemma for a Gaussian $q$, so only
    $f$ is evaluated, never its Jacobian. Both moments come from one pass
    of ``integrator`` over the marginal (`gaussx.moment_transform`), which
    makes $(A_t, b_t)$ the statistical linear regression of $f$ under
    $q(x_t)$ — the same surrogate `gaussx.statistical_linear_regression`
    builds for an observation model. The minimiser does not depend on
    $\Sigma$, which is only carried through to the result.

    A linear drift $f(x) = Mx + c$ is reproduced exactly ($A_t = M$,
    $b_t = c$) by any rule that integrates quadratics exactly.

    Pseudocode:

        for t in 1..T:                                  (vmapped)
            E_f, _, C_xf = moment_transform(f, m_t, S_t)
            A_t = C_xfᵀ S_t⁻¹                           (row solves)
            b_t = E_f − A_t m_t
        Q_t = Σ for every t

    Args:
        drift_fn: Drift $f$, mapping ``(d,) -> (d,)``.
        diffusion: Diffusion covariance $\Sigma$, shape ``(d, d)``, or a
            per-step ``(T, d, d)``.
        path_means: Marginal means $m_t$, shape ``(T, d)``.
        path_covs: Marginal covariances $S_t$, shape ``(T, d, d)``; must
            be positive definite.
        integrator: Gaussian integration rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``, as in
            `gaussx.moment_transform`.

    Returns:
        `LinearizedSDE` with ``A`` ``(T, d, d)``, ``b`` ``(T, d)`` and
        ``Q`` ``(T, d, d)``.

    Raises:
        ValueError: If the path or diffusion shapes are inconsistent.

    References:
        Archambeau, C., Cornford, D., Opper, M. & Shawe-Taylor, J. (2007).
        Gaussian process approximations of stochastic differential
        equations. *JMLR Workshop and Conference Proceedings* 1, 1-16.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> lin = gaussx.linearize_sde(
        ...     lambda x: x - x**3,
        ...     jnp.eye(1),
        ...     jnp.zeros((5, 1)),
        ...     jnp.full((5, 1, 1), 0.1),
        ...     gaussx.GaussHermiteIntegrator(order=5),
        ... )
        >>> lin.A.shape
        (5, 1, 1)
        >>> round(float(lin.A[0, 0, 0]), 6)  # E[1 - 3x^2] = 1 - 3 * 0.1
        0.7
    """
    T, d = _check_path(path_means, path_covs)
    integrator = _default_integrator(integrator)

    def one(m: Float[Array, " d"], S: Float[Array, "d d"]):
        mean_f, _, cross_cov = moment_transform(drift_fn, m, S, integrator=integrator)
        S_op = lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag)
        # A = C_xfᵀ S⁻¹: S is symmetric, so solving S a = row for each row
        # of C_xfᵀ gives the rows of A (as in statistical_linear_regression).
        A = solve_rows(S_op, rearrange(cross_cov, "i j -> j i"))
        return A, mean_f - A @ m

    A, b = jax.vmap(one)(path_means, path_covs)
    Q = _broadcast_diffusion(jnp.asarray(diffusion), T, d)
    return LinearizedSDE(A=A, b=b, Q=Q)


def _broadcast_diffusion(
    diffusion: Float[Array, ...], T: int, d: int
) -> Float[Array, "T d d"]:
    if diffusion.shape == (d, d):
        return repeat(diffusion, "i j -> t i j", t=T)
    if diffusion.shape == (T, d, d):
        return diffusion
    raise ValueError(
        f"diffusion must have shape (d, d) = {(d, d)} or (T, d, d) = "
        f"{(T, d, d)}, got {diffusion.shape}."
    )


def sde_kl_divergence(
    drift_fn: Callable[[Float[Array, " d"]], Float[Array, " d"]],
    linear_drift: LinearizedSDE,
    path_means: Float[Array, "T d"],
    path_covs: Float[Array, "T d d"],
    dt: float | Float[Array, ""] | Float[Array, " T"],
    integrator: AbstractIntegrator | None = None,
) -> Float[Array, ""]:
    r"""Drift path-KL between a linear and a nonlinear SDE along a Gaussian path.

    Two SDEs with the same diffusion $\Sigma$ and drifts $f_L$ and $f$
    have, by Girsanov's theorem, the path-space divergence

    $$
    \mathrm{KL}[q \,\Vert\, p] = \frac{1}{2}\int_0^T
        \mathbb{E}_{q(x_t)}\!\left[
        \lVert f(x_t) - f_L(x_t) \rVert^2_{\Sigma^{-1}}\right] dt,
    $$

    where $q$ is the law of the linear SDE (Gaussian marginals
    $\mathcal{N}(m_t, S_t)$), $p$ the law of the nonlinear one, and
    $\lVert r\rVert^2_{\Sigma^{-1}} = r^\top \Sigma^{-1} r$ (Archambeau
    et al., 2007). The KL between the two initial-state distributions is
    not included; add it separately when $q(x_0) \ne p(x_0)$.

    **Discretisation.** This is the drift-only form with a left Riemann
    sum over the ``T`` path points,

    $$
    \mathrm{KL} \approx \frac{1}{2}\sum_{t=1}^{T} \Delta t_t\,
        \mathbb{E}_{q(x_t)}\!\left[
        \lVert f(x_t) - A_t x_t - b_t\rVert^2_{Q_t^{-1}}\right],
    $$

    which needs only the marginals. The transition-form alternative, a
    KL between Euler-Maruyama transition densities, needs the pairwise
    marginals $q(x_t, x_{t+1})$ as well and agrees with this one to
    $O(\Delta t)$; it is not what this function computes.

    Each expectation is the integrator's weighted sum over its points of
    the whitened residual's squared norm, so for a rule with non-negative
    mean weights the result is non-negative by construction.
    `gaussx.TaylorIntegrator` is rejected at every order: order 1 sees the
    residual only at the mean, where the drift of `linearize_sde` makes it
    vanish (a zero KL for every drift), and order 2 adds a Hessian term
    that can make the expectation of a square negative (drift ``x²`` at
    ``m = 0, S = 1`` gives ``E[(x² − 1)²] ≈ −1``). For the optimal
    drift and a linear $f$ the residual vanishes everywhere and the KL is
    zero.

    Pseudocode:

        for t in 1..T:                                  (vmapped)
            L_t = chol(Q_t)
            e_t = E_q[‖L_t⁻¹ (f(x) − A_t x − b_t)‖²]    (integrator)
        KL = ½ Σ_t Δt_t e_t

    Args:
        drift_fn: Nonlinear drift $f$, mapping ``(d,) -> (d,)``.
        linear_drift: Linear drift and diffusion, e.g. from
            `linearize_sde`. ``linear_drift.Q`` supplies $\Sigma$, which
            must be positive definite.
        path_means: Marginal means $m_t$, shape ``(T, d)``.
        path_covs: Marginal covariances $S_t$, shape ``(T, d, d)``.
        dt: Step size, a non-negative scalar or per-step ``(T,)`` array.
        integrator: Gaussian integration rule. Defaults to
            ``UnscentedIntegrator(alpha=1.0)``.

    Returns:
        Scalar path-KL.

    Raises:
        ValueError: If the path and ``linear_drift`` shapes disagree, or
            ``integrator`` is a `gaussx.TaylorIntegrator`.
        EquinoxRuntimeError: If any step size is negative (also under
            ``jit``).

    References:
        Archambeau, C., Cornford, D., Opper, M. & Shawe-Taylor, J. (2007).
        Gaussian process approximations of stochastic differential
        equations. *JMLR Workshop and Conference Proceedings* 1, 1-16.

    Examples:
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> drift = lambda x: -2.0 * x + 1.0
        >>> m, S = jnp.zeros((4, 1)), jnp.full((4, 1, 1), 0.3)
        >>> lin = gaussx.linearize_sde(drift, jnp.eye(1), m, S)
        >>> abs(float(gaussx.sde_kl_divergence(drift, lin, m, S, dt=0.1))) < 1e-10
        True
    """
    T, d = _check_path(path_means, path_covs)
    if linear_drift.A.shape != (T, d, d) or linear_drift.b.shape != (T, d):
        raise ValueError(
            f"linear_drift must have A of shape {(T, d, d)} and b of shape "
            f"{(T, d)} to match the path, got {linear_drift.A.shape} and "
            f"{linear_drift.b.shape}."
        )
    Q = _broadcast_diffusion(linear_drift.Q, T, d)
    integrator = _default_integrator(integrator)
    if isinstance(integrator, TaylorIntegrator):
        raise ValueError(
            "sde_kl_divergence needs a rule that keeps the expectation of a "
            "squared residual non-negative: TaylorIntegrator returns zero for "
            "every drift at order 1 and can go negative at order 2. Use a "
            "point-based rule with non-negative weights (the default "
            "UnscentedIntegrator(alpha=1.0), cubature or Gauss-Hermite)."
        )

    def one(m, S, A, b, Q_t):
        L = jnp.linalg.cholesky(Q_t)

        def whitened_sq(x: Float[Array, " d"]) -> Float[Array, " 1"]:
            r = drift_fn(x) - (A @ x + b)
            w = jax.scipy.linalg.solve_triangular(L, r, lower=True)
            return jnp.atleast_1d(jnp.sum(w**2))

        state = GaussianState(
            mean=m, cov=lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag)
        )
        return mean_expectation(whitened_sq, state, integrator)[0]

    e = jax.vmap(one)(path_means, path_covs, linear_drift.A, linear_drift.b, Q)
    dt_arr = jnp.broadcast_to(jnp.asarray(dt, dtype=e.dtype), (T,))
    dt_arr = eqx.error_if(
        dt_arr, jnp.any(dt_arr < 0), "sde_kl_divergence: dt must be non-negative."
    )
    return 0.5 * jnp.sum(dt_arr * e)
