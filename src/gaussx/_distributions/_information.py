"""Information diagnostics of a linear-Gaussian retrieval (Rodgers 2000).

The retrieval observes a state x ∈ ℝⁿ through y = K x + ε, with Jacobian K of
size m × n, noise ε ~ 𝒩(0, S_ε) and prior x ~ 𝒩(x_a, S_a). Ŝ is the posterior
covariance and A the averaging kernel.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Float

from gaussx._einx import einsum
from gaussx._linalg._linalg import solve_columns
from gaussx._linalg._symmetrize import symmetrize
from gaussx._primitives._cholesky import cholesky
from gaussx._primitives._logdet import logdet


class InformationDiagnostics(eqx.Module):
    """Output of `information_diagnostics`.

    Attributes:
        averaging_kernel: A = Ŝ Kᵀ S_ε⁻¹ K, of size n × n.
        dofs: Degrees of freedom for signal, d_s = tr A.
        information_content: Shannon information content H, in nats.
        posterior_cov: The posterior covariance Ŝ, of size n × n.
        retrieval_noise_cov: S_m = A Ŝ, the measurement-noise part of the
            retrieval error, of size n × n.
    """

    averaging_kernel: Float[Array, "n n"]
    dofs: Float[Array, ""]
    information_content: Float[Array, ""]
    posterior_cov: Float[Array, "n n"]
    retrieval_noise_cov: Float[Array, "n n"]


def information_diagnostics(
    jacobian: Float[Array, "m n"],
    noise_cov: lx.AbstractLinearOperator,
    prior_cov: lx.AbstractLinearOperator,
) -> InformationDiagnostics:
    """Averaging kernel, DOFS and information content of a linear retrieval.

        Ŝ⁻¹ = Kᵀ S_ε⁻¹ K + S_a⁻¹
        A = Ŝ Kᵀ S_ε⁻¹ K
        d_s = tr A
        H = ½ (ln det S_a + ln det Ŝ⁻¹)
        S_m = A Ŝ

    H is the mutual information between the state and the measurement, and
    0 ≤ d_s ≤ min(n, m). See Rodgers (2000), *Inverse Methods for Atmospheric
    Sounding: Theory and Practice*, World Scientific, doi:10.1142/3171,
    chapters 2 and 3.

    Algorithm:
        1. Whiten with structured factors, K̃ = L_ε⁻¹ K and L_a⁻¹, where
           L_ε L_εᵀ = S_ε and L_a L_aᵀ = S_a, so neither covariance is
           materialised beyond the n × n outputs.
        2. Factor the stacked [K̃; L_a⁻¹] = Q T by QR, so Ŝ⁻¹ = Tᵀ T without
           forming K̃ᵀ K̃, which would round away the prior when K̃ is large.
        3. Read the diagnostics off T: Ŝ = T⁻¹ T⁻ᵀ, A = T⁻¹ (K̃ T⁻¹)ᵀ K̃ and
           ln det Ŝ⁻¹ = 2 ∑ᵢ ln |T_ii|.

    Args:
        jacobian: The Jacobian K of the forward model at the retrieval.
        noise_cov: The measurement-noise covariance S_ε, m × m.
        prior_cov: The prior covariance S_a, n × n.

    Returns:
        The five diagnostics, as dense arrays.

    Examples:
        One unit measurement of one state with a unit prior carries half a
        degree of freedom and ½ ln 2 nats:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import gaussx
        >>> identity = lx.DiagonalLinearOperator(jnp.ones(1))
        >>> diagnostics = gaussx.information_diagnostics(
        ...     jnp.ones((1, 1)), identity, identity
        ... )
        >>> round(float(diagnostics.dofs), 4)
        0.5
        >>> round(float(diagnostics.information_content), 4)
        0.3466
        >>> round(float(diagnostics.retrieval_noise_cov[0, 0]), 4)
        0.25
    """
    n_state = jacobian.shape[1]
    identity = jnp.eye(n_state, dtype=jacobian.dtype)
    # K̃ = L_ε⁻¹ K
    whitened_jacobian = solve_columns(cholesky(noise_cov), jacobian)
    # L_a⁻¹
    prior_root_inverse = solve_columns(cholesky(prior_cov), identity)
    # [K̃; L_a⁻¹] = Q T, so Ŝ⁻¹ = Tᵀ T
    _, triangular = jnp.linalg.qr(jnp.vstack([whitened_jacobian, prior_root_inverse]))
    # T⁻¹
    triangular_inverse = jax.scipy.linalg.solve_triangular(
        triangular, identity, lower=False
    )
    # Ŝ = T⁻¹ T⁻ᵀ
    posterior_cov = symmetrize(
        einsum(triangular_inverse, triangular_inverse, "i k, j k -> i j")
    )
    # K̃ T⁻¹
    gain_factor = einsum(whitened_jacobian, triangular_inverse, "m k, k j -> m j")
    # A = Ŝ K̃ᵀ K̃ = T⁻¹ (K̃ T⁻¹)ᵀ K̃
    averaging_kernel = einsum(
        triangular_inverse, gain_factor, whitened_jacobian, "i k, m k, m j -> i j"
    )
    # S_m = A Ŝ
    retrieval_noise_cov = symmetrize(
        einsum(averaging_kernel, posterior_cov, "i k, k j -> i j")
    )
    # ln det Ŝ⁻¹ = 2 ∑ᵢ ln |T_ii|
    logdet_precision = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(triangular))))
    return InformationDiagnostics(
        averaging_kernel=averaging_kernel,
        # d_s = tr A
        dofs=jnp.trace(averaging_kernel),
        # H = ½ (ln det S_a + ln det Ŝ⁻¹)
        information_content=0.5 * (logdet(prior_cov) + logdet_precision),
        posterior_cov=posterior_cov,
        retrieval_noise_cov=retrieval_noise_cov,
    )
