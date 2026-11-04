"""OILMM projection for multi-output Gaussian processes.

In the orthogonal instantaneous linear mixing model (OILMM) of Bruinsma et
al. (2020), the P outputs at each of the N inputs mix L independent latent
GPs through a mixing matrix W with orthonormal columns, WᵀW = I_L, plus
observation noise with variances σ_ε² per output. Projecting the data onto
W decouples the latent GPs, so each can be fitted on its own.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from gaussx._einx import einsum


def oilmm_project(
    Y: Float[Array, "N P"],
    W: Float[Array, "P L"],
    noise_var: Float[Array, " P"] | float,
    *,
    check_orthonormal: bool = False,
    orthonormal_tol: float | None = None,
) -> tuple[Float[Array, "N L"], Float[Array, " L"]]:
    """Project multi-output data to independent latent GPs via OILMM.

        Y_latent = Y W
        σ²_latent = diag(Wᵀ diag(σ_ε²) W) = (W ⊙ W)ᵀ σ_ε²

    For isotropic noise Wᵀ diag(σ_ε²) W = σ_ε² I_L is exactly diagonal.
    For heteroscedastic noise it has off-diagonal terms; only its diagonal
    is returned, which is the OILMM independence approximation (the latent
    channels are treated as independent).

    WᵀW = I_L is a precondition and is **not** checked by default: with a
    non-orthonormal W the projection Y W is not the least-squares one (that
    would be Y W (WᵀW)⁻¹) and the latent noise is wrong.

    Algorithm:
        1. Optionally check max |WᵀW − I_L| against the tolerance, jit-safe
           through `equinox.error_if`.
        2. Project the data, Y_latent = Y W.
        3. Broadcast σ_ε² to one variance per output and project it,
           σ²_latent = (W ⊙ W)ᵀ σ_ε².

    Args:
        Y: Observations Y, one row per input and one column per output.
        W: Mixing matrix W with orthonormal columns, WᵀW = I_L.
        noise_var: Observation noise variances σ_ε²: a scalar for isotropic
            noise, or one per output for heteroscedastic noise.
        check_orthonormal: If ``True``, raise (jit-safe, via
            `equinox.error_if`) when max |WᵀW − I_L| > ``orthonormal_tol``.
        orthonormal_tol: Tolerance for ``check_orthonormal``. Default
            ``sqrt(eps)`` of ``W``'s dtype.

    Returns:
        Tuple ``(Y_latent, noise_latent)`` holding the projected data
        Y_latent and the latent noise variances σ²_latent.

    Raises:
        equinox.EquinoxRuntimeError: If ``check_orthonormal`` is set and
            WᵀW deviates from I_L by more than ``orthonormal_tol``.

    Examples:
        Two outputs mixed from one latent channel:

        >>> import math
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> W = jnp.array([[1.0], [1.0]]) / math.sqrt(2.0)
        >>> Y = jnp.array([[1.0, 1.0], [2.0, 0.0], [0.0, -1.0]])
        >>> Y_latent, noise_latent = gaussx.oilmm_project(Y, W, 0.5)
        >>> Y_latent.shape, noise_latent.shape
        ((3, 1), (1,))
        >>> bool(jnp.allclose(noise_latent, 0.5))
        True

        Heteroscedastic noise keeps the diagonal of Wᵀ diag(σ_ε²) W:

        >>> _, noise_latent = gaussx.oilmm_project(Y, W, jnp.array([0.2, 0.4]))
        >>> bool(jnp.allclose(noise_latent, 0.3))
        True

        Projecting and back-projecting round-trips data in the span of W:

        >>> Y_span = jnp.array([[2.0], [-1.0]]) @ W.T
        >>> Y_latent, _ = gaussx.oilmm_project(Y_span, W, 0.5)
        >>> means, _ = gaussx.oilmm_back_project(
        ...     Y_latent, jnp.zeros_like(Y_latent), W
        ... )
        >>> bool(jnp.allclose(means, Y_span))
        True

    References:
        Bruinsma, W. P., Perim, E., Tebbutt, W., Hosking, J. S., Solin, A.
        and Turner, R. E. (2020). Scalable exact inference in multi-output
        Gaussian processes. *ICML*.
    """
    if check_orthonormal:
        # WᵀW
        gram = einsum(W, W, "output row, output col -> row col")
        tol = (
            math.sqrt(float(jnp.finfo(W.dtype).eps))
            if orthonormal_tol is None
            else orthonormal_tol
        )
        identity = jnp.eye(W.shape[1], dtype=W.dtype)
        # WᵀW − I_L
        deviation = gram - identity
        # |WᵀW − I_L|
        abs_deviation = jnp.abs(deviation)
        # max |WᵀW − I_L|
        max_deviation = jnp.max(abs_deviation)
        W = eqx.error_if(
            W, max_deviation > tol, "oilmm_project: W is not orthonormal (WᵀW != I)."
        )
    # Y_latent = Y W
    # jnp, not einx: N may be 0, which einx rejects and `@` accepts
    Y_latent = Y @ W
    num_outputs = W.shape[0]
    noise_variance = jnp.asarray(noise_var)
    # σ_ε², one per output
    # jnp, not einx: noise_var is either a scalar or already one per output
    output_noise = jnp.broadcast_to(noise_variance, (num_outputs,))
    # W ⊙ W
    mixing_squared = W**2
    # σ²_latent = (W ⊙ W)ᵀ σ_ε²
    noise_latent = einsum(
        mixing_squared, output_noise, "output latent, output -> latent"
    )
    return Y_latent, noise_latent


def oilmm_back_project(
    f_means: Float[Array, "N L"],
    f_vars: Float[Array, "N L"],
    W: Float[Array, "P L"],
) -> tuple[Float[Array, "N P"], Float[Array, "N P"]]:
    """Back-project latent GP predictions to the observation space.

        y_means = f_means Wᵀ
        y_vars = f_vars (W ⊙ W)ᵀ

    Args:
        f_means: Latent predictive means, one column per latent channel.
        f_vars: Latent predictive variances, one column per latent channel.
        W: Mixing matrix W with orthonormal columns, WᵀW = I_L.

    Returns:
        Tuple ``(y_means, y_vars)`` holding the predictive means and
        variances of the P outputs at each input.

    Examples:
        One latent channel spread over two outputs:

        >>> import math
        >>> import jax.numpy as jnp
        >>> import gaussx
        >>> W = jnp.array([[1.0], [1.0]]) / math.sqrt(2.0)
        >>> y_means, y_vars = gaussx.oilmm_back_project(
        ...     jnp.array([[2.0], [0.0]]), jnp.array([[1.0], [4.0]]), W
        ... )
        >>> y_means.shape, y_vars.shape
        ((2, 2), (2, 2))
        >>> bool(jnp.allclose(y_vars, jnp.array([[0.5, 0.5], [2.0, 2.0]])))
        True
    """
    # jnp, not einx, throughout: N, L and P may each be 0, which einx
    # rejects and these products accept.
    # y_means = f_means Wᵀ
    y_means = f_means @ W.T
    # W ⊙ W
    mixing_squared = W**2
    # y_vars = f_vars (W ⊙ W)ᵀ
    y_vars = f_vars @ mixing_squared.T
    return y_means, y_vars
