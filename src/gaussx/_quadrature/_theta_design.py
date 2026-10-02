r"""θ-designs: integration points over INLA hyperparameters.

Rue, Martino & Chopin (2009), *Approximate Bayesian inference for latent
Gaussian models by using integrated nested Laplace approximations*, JRSS-B
71(2), §6.5.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Literal

import einx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from jaxtyping import Array, Float

from gaussx._einx import einsum, rearrange


# Hard cap on the axis-wise grid walk, so a flat ``log_post`` cannot loop
# forever. In ``z`` a well-posed posterior drops by ``z^2 / 2``, so this is
# never reached for any sensible ``grid_step`` / ``grid_threshold``.
_MAX_GRID_STEPS = 100


def theta_design(
    log_post: Callable[[Array], Array],
    mode: Float[Array, " m"],
    *,
    method: Literal["eb", "grid", "ccd"] | None = None,
    hessian: Float[Array, "m m"] | None = None,
    grid_step: float = 1.0,
    grid_threshold: float = 2.5,
    ccd_f0: float = 1.1,
) -> tuple[Float[Array, "K m"], Float[Array, " K"]]:
    r"""Integration points and log-weights over hyperparameters $\theta$.

    Builds the design R-INLA uses to integrate over $\theta$. With
    $-\nabla^2 \log\tilde\pi(\theta^\ast \mid y) = V \Lambda V^\top$ at the
    mode $\theta^\ast$, points are placed in the standardised coordinates
    $z$ of

    $$
    \theta(z) = \theta^\ast + V \Lambda^{-1/2} z ,
    $$

    in which the posterior is approximately $\mathcal N(0, I)$. Each point
    gets a design weight $\Delta_k$, which is corrected by the evaluated
    posterior, $\log w_k = \log\Delta_k + \log\tilde\pi(\theta_k \mid y)$,
    and normalised so that $\sum_k w_k = 1$. Then
    $\tilde\pi(x_i \mid y) \approx \sum_k w_k\,\tilde\pi(x_i \mid \theta_k, y)$.

    Methods:

    - `"eb"`: empirical Bayes, the mode alone.
    - `"grid"`: walk each $z$-axis in both directions in steps of
      `grid_step` while $\log\tilde\pi$ stays within `grid_threshold`
      log-units of the mode, take the product grid of those axis ranges,
      and keep its points that are within `grid_threshold` of the mode.
      $\Delta_k$ is constant. The point count depends on `log_post`, so
      this method cannot be traced under `jax.jit`.
    - `"ccd"`: a central composite design, $O(m^2)$ points: the centre,
      $2m$ axial points and a resolution-V fractional factorial
      $2^{m-p}$, all non-centre points on the sphere of radius
      $f_0\sqrt m$ (15 points at $m = 3$, 27 at $m = 5$). The centre and
      shell weights are fixed, as in Rue et al. (2009) §6.5, so that the
      design integrates a Gaussian posterior exactly up to its second
      moments:
      $\Delta_0 \propto 1 - f_0^{-2}$ and
      $\Delta_{\text{shell}} \propto e^{m f_0^2/2} / ((K - 1) f_0^2)$.

    Args:
        log_post: Unnormalised log-posterior $\log\tilde\pi(\theta \mid y)$,
            mapping a ``(m,)`` array to a scalar.
        mode: Its mode $\theta^\ast$, shape ``(m,)``. Finding it is the
            caller's job.
        method: ``"eb"``, ``"grid"`` or ``"ccd"``. ``None`` (default)
            picks ``"grid"`` for ``m <= 2`` and ``"ccd"`` for ``m > 2``.
        hessian: Hessian of `log_post` at `mode`, shape ``(m, m)``
            (negative definite). Computed with `jax.hessian` if ``None``.
        grid_step: Step size in $z$ for ``"grid"``.
        grid_threshold: Log-density drop from the mode beyond which
            ``"grid"`` stops and discards points.
        ccd_f0: Shell radius factor $f_0 > 1$ for ``"ccd"``.

    Returns:
        Tuple ``(points, log_weights)``: points $\theta_k$ of shape
        ``(K, m)`` and normalised log-weights of shape ``(K,)``
        (``logsumexp(log_weights) == 0``). The first point is the mode.

    Example:
        ```python
        def log_post(theta):
            return -0.5 * jnp.sum((theta - 1.0) ** 2 / jnp.array([1.0, 4.0, 9.0]))


        pts, logw = gaussx.theta_design(log_post, jnp.ones(3), method="ccd")
        pts.shape, logw.shape  # (15, 3), (15,)
        post_mean = einx.dot("k, k m -> m", jnp.exp(logw), pts)  # == 1.0

        # In INLA, one batched Laplace fit per design point:
        fits = jax.vmap(lambda th: gaussx.laplace_mode(prior_at(th), lik, y))(pts)
        x_mean = einx.dot("k, k n -> n", jnp.exp(logw), fits.mode)
        ```
    """
    mode = jnp.asarray(mode)
    if mode.ndim != 1:
        raise ValueError(f"mode must be 1-D, got shape {mode.shape}")
    m = mode.shape[0]
    if method is None:
        method = "grid" if m <= 2 else "ccd"
    dtype = mode.dtype

    if method == "eb":
        return rearrange(mode, "m -> 1 m"), jnp.zeros((1,), dtype=dtype)

    if hessian is None:
        hessian = jax.hessian(log_post)(mode)
    evals, evecs = jnp.linalg.eigh(-jnp.asarray(hessian, dtype=dtype))
    # theta(z) = mode + V Lambda^{-1/2} z
    scale = einx.multiply("i j, j -> i j", evecs, jax.lax.rsqrt(evals))

    def to_theta(z: Float[Array, "K m"]) -> Float[Array, "K m"]:
        return einx.add("k i, i -> k i", einsum(scale, z, "i j, k j -> k i"), mode)

    if method == "grid":
        if grid_step <= 0 or grid_threshold <= 0:
            raise ValueError("grid_step and grid_threshold must be positive")
        z = _grid_z(log_post, scale, mode, grid_step, grid_threshold)
        points = to_theta(jnp.asarray(z, dtype=dtype))
        lp = jax.vmap(log_post)(points)
        keep = np.asarray(lp >= log_post(mode) - grid_threshold)
        points, lp = points[keep], lp[keep]
        return points, lp - jsp.logsumexp(lp)

    if method == "ccd":
        if ccd_f0 <= 1:
            raise ValueError(f"ccd_f0 must be > 1, got {ccd_f0}")
        z_unit = _ccd_design(m)  # centre first; shell points on radius sqrt(m)
        n_points = z_unit.shape[0]
        log_delta = np.full(
            n_points,
            m * ccd_f0**2 / 2 - math.log((n_points - 1) * ccd_f0**2),
        )
        log_delta[0] = math.log1p(-(ccd_f0**-2))
        points = to_theta(jnp.asarray(ccd_f0 * z_unit, dtype=dtype))
        logw = jnp.asarray(log_delta, dtype=dtype) + jax.vmap(log_post)(points)
        return points, logw - jsp.logsumexp(logw)

    raise ValueError(f"method must be 'eb', 'grid' or 'ccd', got {method!r}")


def _grid_z(
    log_post: Callable[[Array], Array],
    scale: Float[Array, "m m"],
    mode: Float[Array, " m"],
    step: float,
    threshold: float,
) -> np.ndarray:
    """Product grid in ``z`` of the axis-wise explored ranges, centre first."""
    log_post = jax.jit(log_post)  # one compile for the whole walk
    floor = float(log_post(mode)) - threshold
    axes = []
    for j in range(mode.shape[0]):
        ks = [0]
        for sign in (1, -1):
            k = 1
            while k <= _MAX_GRID_STEPS:
                theta = mode + (sign * k * step) * scale[:, j]
                if float(log_post(theta)) < floor:
                    break
                ks.append(sign * k)
                k += 1
        axes.append(step * np.asarray(ks, dtype=float))
    return np.array(list(itertools.product(*axes)))


def _resolution_v_columns(m: int) -> tuple[int, list[int]]:
    """Walsh column indices of a minimal resolution-V ``2^(m-p)`` design.

    Greedy construction of Sanchez & Sanchez (2005): a new column index may
    not equal the XOR of at most three chosen ones, so no defining word has
    length below five (main effects and two-factor interactions unaliased).
    Returns ``(n_runs, columns)``.
    """
    cols: list[int] = []
    forbidden = {0}
    c = 0
    while len(cols) < m:
        c += 1
        if c in forbidden:
            continue
        sums = {c}
        sums |= {c ^ a for a in cols}
        sums |= {c ^ a ^ b for i, a in enumerate(cols) for b in cols[i + 1 :]}
        forbidden |= sums
        cols.append(c)
    return 1 << max(cols).bit_length(), cols


def _ccd_design(m: int) -> np.ndarray:
    """CCD in ``z`` with shell radius ``sqrt(m)``: centre, axial, factorial."""
    axial = math.sqrt(m) * np.concatenate([np.eye(m), -np.eye(m)])
    if m == 1:  # the factorial points would duplicate the axial ones
        return np.concatenate([np.zeros((1, 1)), axial])
    n_runs, cols = _resolution_v_columns(m)
    runs = np.arange(n_runs)
    parity = np.array([[bin(r & c).count("1") % 2 for c in cols] for r in runs])
    factorial = 1.0 - 2.0 * parity
    return np.concatenate([np.zeros((1, m)), axial, factorial])
