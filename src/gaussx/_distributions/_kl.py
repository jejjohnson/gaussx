"""Distribution-level KL divergence for MultivariateNormal variants."""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from gaussx._linalg._linalg import trace_product
from gaussx._primitives._inv import inv
from gaussx._primitives._logdet import logdet
from gaussx._primitives._solve import solve


def dist_kl_divergence(
    p_loc: Float[Array, " N"],
    p_cov: lx.AbstractLinearOperator,
    q_loc: Float[Array, " N"],
    q_cov: lx.AbstractLinearOperator,
) -> Float[Array, ""]:
    r"""KL divergence ``KL(p || q)`` between two multivariate normals.

    This is the **canonical KL implementation** for lineax-operator covariances.
    The specialised variants below all compute the same quantity but with
    different parameterisations suited to their use cases:

    - `kl_standard_normal` —
      special case ``KL(N(m, S) || N(0, I))``; avoids matrix inversion.
    - `gauss_kl` — Cholesky-parameterised form
      for GP/SVGP models; supports multi-output and diagonal ``q_sqrt``.
    - `kl_divergence` — Bregman-divergence
      form operating on natural parameters for the exponential family.

    $$
    KL(p \| q) = \frac{1}{2}\bigl(
        \operatorname{tr}(\Sigma_q^{-1} \Sigma_p)
        + (\mu_q - \mu_p)^T \Sigma_q^{-1} (\mu_q - \mu_p)
        - N
        + \log|\Sigma_q| - \log|\Sigma_p|
    \bigr)
    $$

    Exploits structured operators for the trace and logdet terms.

    Args:
        p_loc: Mean of distribution p, shape ``(N,)``.
        p_cov: Covariance operator of distribution p.
        q_loc: Mean of distribution q, shape ``(N,)``.
        q_cov: Covariance operator of distribution q.

    Returns:
        Scalar KL divergence.
    """
    N = p_loc.shape[-1]
    delta = q_loc - p_loc

    # tr(Sigma_q^{-1} Sigma_p)
    q_inv = inv(q_cov)
    trace_term = trace_product(q_inv, p_cov)

    # Quadratic term: delta^T Sigma_q^{-1} delta
    quad = jnp.sum(delta * solve(q_cov, delta))

    # Log-determinant difference
    ld_q = logdet(q_cov)
    ld_p = logdet(p_cov)

    return 0.5 * (trace_term + quad - N + ld_q - ld_p)
