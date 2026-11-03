"""Multivariate normal x ~ 𝒩(μ, Λ⁻¹) with the precision Λ = Σ⁻¹ as a lineax operator.

The residual is r = x − μ, D_x the event size and z ~ 𝒩(0, I) the standard
normal noise of a draw. L_Λ is the lower Cholesky factor of Λ = L_Λ L_Λᵀ, and
Λ = Φ diag(γ) Φᵀ its eigendecomposition, with eigenvalues γ and orthonormal
eigenvectors Φ.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float
from numpyro.distributions.util import lazy_property

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._distributions._mvn_base import AbstractMultivariateNormal
from gaussx._distributions._utils import _reshape_samples
from gaussx._einx import einsum, rearrange
from gaussx._linalg._diag_inv import _diag_inv_structured
from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._sparse import SparseOperator
from gaussx._primitives._cholesky import cholesky as _cholesky
from gaussx._primitives._diag import diag as _diag
from gaussx._primitives._inv import inv as _inv
from gaussx._primitives._solve import solve as _solve
from gaussx._sparse._factor import SparseCholeskyFactor
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import AbstractSolverStrategy


class MultivariateNormalPrecision(AbstractMultivariateNormal):
    """Multivariate normal 𝒩(μ, Λ⁻¹) whose precision Λ = Σ⁻¹ is a lineax operator.

        x ~ 𝒩(μ, Λ⁻¹)
        log 𝒩(x; μ, Λ⁻¹) = −½ (D_x log 2π − log det Λ + (x − μ)ᵀ Λ (x − μ))

    The natural parameterisation for many inference algorithms (message
    passing, variational inference in natural coordinates): the log-density
    needs a matrix-vector product with Λ and no solve. Precision-parameterised
    member of `AbstractMultivariateNormal`, which supplies the shared
    accessors (``covariance_operator``, ``precision_operator``,
    ``covariance_matrix``, ``scale_tril``, `kl`, …).

    Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).

    Args:
        loc: Mean μ; leading axes are the batch shape.
        prec_operator: Precision Λ = Σ⁻¹, shared by every batch element.
        solver: Solver strategy for the log-determinant of Λ. Defaults to
            ``AutoSolver()``.
        validate_args: Whether to validate input arguments.

    Examples:
        A precision of 2 I in three dimensions:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._distributions import MultivariateNormalPrecision
        >>> precision = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(3), lx.positive_semidefinite_tag
        ... )
        >>> d = MultivariateNormalPrecision(jnp.zeros(3), precision)
        >>> round(float(d.log_prob(jnp.ones(3))), 4)
        -4.7171

        The marginal variances are diag(Λ⁻¹):

        >>> bool(jnp.allclose(d.variance, 0.5 * jnp.ones(3)))
        True

        Draws carry the sample shape in front of the event shape:

        >>> d.sample(jax.random.key(0), (4,)).shape
        (4, 3)
    """

    pytree_data_fields = ("loc", "prec_operator", "solver")

    def __init__(
        self,
        loc: Float[Array, "*batch N"],
        prec_operator: lx.AbstractLinearOperator,
        solver: AbstractSolverStrategy | None = None,
        *,
        validate_args: bool | None = None,
    ) -> None:
        if solver is None:
            solver = AutoSolver()
        self.loc = loc
        self.prec_operator = prec_operator
        self.solver = solver
        event_shape = loc.shape[-1:]
        batch_shape = loc.shape[:-1]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @property
    def covariance_operator(self) -> lx.AbstractLinearOperator:
        """The covariance Σ = Λ⁻¹: the lazy `gaussx.inv` of ``prec_operator``."""
        return _inv(self.prec_operator)

    @property
    def precision_operator(self) -> lx.AbstractLinearOperator:
        """The precision Λ (native; the same object as ``prec_operator``)."""
        return self.prec_operator

    def _log_prob_single(self, residual: Float[Array, " N"]) -> Float[Array, ""]:
        """Log-density of one event from its residual r = x − μ.

            log 𝒩(x; μ, Λ⁻¹) = −½ (D_x log 2π − log det Λ + rᵀ Λ r)

        Args:
            residual: Residual r = x − μ of a single event.

        Returns:
            The scalar log 𝒩(x; μ, Λ⁻¹).
        """
        # Λ r
        precision_residual = self.prec_operator.mv(residual)
        # rᵀ Λ r, a full reduction of one event's vector
        mahalanobis = jnp.sum(residual * precision_residual)
        # log det Λ
        logdet = self.solver.logdet(self.prec_operator)
        event_size = self.loc.shape[-1]
        # log 𝒩(x; μ, Λ⁻¹) = −½ (D_x log 2π − log det Λ + rᵀ Λ r)
        return -0.5 * (event_size * _LOG_2PI - logdet + mahalanobis)

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch N"]:
        """Draw x = μ + L_Λ⁻ᵀ z with z ~ 𝒩(0, I), by a triangular solve.

            x = μ + L_Λ⁻ᵀ z,   z ~ 𝒩(0, I),   Λ = L_Λ L_Λᵀ

        Algorithm:
            1. Draw z ~ 𝒩(0, I) in the parameters' dtype.
            2. Strip tags and scalar multiples, Λ = c Λ₀, so a structured Λ₀
               keeps its structured Cholesky factor L_Λ₀.
            3. Solve L_Λ₀ᵀ x₀ = z and rescale, x − μ = c^(−½) x₀.
            4. For a dense Λ whose Cholesky fails (non-finite draws), use
               the symmetric inverse square root x − μ = Λ^(−½) z instead.
            5. Add the mean μ.

        Args:
            key: PRNG key.
            sample_shape: Leading shape of the draws.

        Returns:
            Draws of shape ``sample_shape + batch_shape + event_shape``.

        Raises:
            ValueError: If ``key`` is ``None``.
        """
        if key is None:
            raise ValueError(
                "PRNG key must be provided to sample from MultivariateNormalPrecision."
            )
        shape = sample_shape + self.batch_shape + self.event_shape
        # Draw in the parameters' dtype: default-float noise meets a float32
        # factor in the solve below, which lineax rejects under x64.
        dtype = jnp.result_type(self.loc, self.prec_operator.in_structure().dtype)
        # z ~ 𝒩(0, I)
        noise = jax.random.normal(key, shape=shape, dtype=dtype)  # type: ignore[arg-type]
        noise_flat = rearrange(noise, "... D -> (...) D")

        # Look through tags and scalar multiples so a structured precision
        # (e.g. 2 * Kronecker) keeps its structured Cholesky factor.
        # Λ = c Λ₀, s = c^(−½)
        precision, scale = _unwrap_scaled(self.prec_operator)
        # L_Λ₀ L_Λ₀ᵀ = Λ₀
        factor = _cholesky(precision)

        def _solve_one(standard_normal):
            if isinstance(factor, SparseCholeskyFactor):
                # A SparseOperator's factor is of P Λ₀ Pᵀ: x₀ = Pᵀ L⁻ᵀ z.
                return factor.solve_lower_transpose(standard_normal)
            # x₀ = L_Λ₀⁻ᵀ z
            return _solve(factor.T, standard_normal)

        # L_Λ₀⁻ᵀ z, one row per draw
        unscaled_flat = jax.vmap(_solve_one)(noise_flat)
        # x − μ = s L_Λ₀⁻ᵀ z
        draws_flat = scale * unscaled_flat
        if isinstance(precision, lx.MatrixLinearOperator):
            # A dense Cholesky that fails numerically (an ill-conditioned or
            # only semi-definite precision) gives non-finite draws; take the
            # dense symmetric inverse square root then. Dense precisions only:
            # lax.cond stages both branches under jit, and a dense branch
            # would put D_x × D_x buffers into every structured or sparse
            # sample.
            structured = draws_flat
            draws_flat = jax.lax.cond(
                jnp.all(jnp.isfinite(structured)),
                lambda: structured,
                lambda: _dense_precision_draws(self.prec_operator, noise_flat),
            )
        centred = _reshape_samples(draws_flat, shape[:-1])
        # x = μ + (x − μ)
        return self.loc + centred

    @lazy_property
    def variance(self) -> Float[Array, "*batch N"]:
        """Marginal variances diag(Λ⁻¹), broadcast over the batch shape.

        The exact structured paths of `gaussx.diag_inv` (e.g. the O(N d³)
        selected inverse of a `gaussx.BlockTriDiag`) apply; only an operator
        with none of them densifies (gh-360).

        Returns:
            The variances diag(Λ⁻¹), one per event entry and batch element.
        """
        # diag(Λ⁻¹), structured
        variance = _diag_inv_structured(
            self.prec_operator, pinv=False, num_probes=30, key=None, solver=None
        )
        if variance is None:
            # diag(Λ⁻¹) of the lazy inverse
            variance = _diag(_inv(self.prec_operator))
        return jnp.broadcast_to(variance, self.batch_shape + self.event_shape)

    def entropy(self) -> Float[Array, "*batch"]:
        """Entropy ℍ[𝒩(μ, Λ⁻¹)], one entry per batch element.

            ℍ[𝒩(μ, Λ⁻¹)] = ½ (D_x (1 + log 2π) − log det Λ)

        Returns:
            The entropy, broadcast over the batch shape.
        """
        event_size = self.loc.shape[-1]
        # log det Λ
        logdet = self.solver.logdet(self.prec_operator)
        # ℍ[𝒩(μ, Λ⁻¹)] = ½ (D_x (1 + log 2π) − log det Λ)
        entropy = 0.5 * (event_size * (1.0 + _LOG_2PI) - logdet)
        # One entry per batch element, as numpyro's contract requires.
        return jnp.broadcast_to(entropy, self.batch_shape)


def _unwrap_scaled(
    operator: lx.AbstractLinearOperator,
) -> tuple[lx.AbstractLinearOperator, Float[Array, ""] | float]:
    """Strip tags and scalar multiples from a precision Λ = c Λ₀.

        Λ = c Λ₀,   s = c^(−½),   s L_Λ₀⁻ᵀ z ~ 𝒩(0, Λ⁻¹)

    A non-positive c gives a non-finite s, which the dense fallback in
    ``sample`` then handles.

    Args:
        operator: The precision Λ, possibly tagged or scaled.

    Returns:
        Tuple ``(precision, scale)`` holding Λ₀ and s. A composite with a
        sparse block is returned unchanged with s = 1, since its sparse
        factor is not a lineax operator.
    """
    original = operator
    scale: Float[Array, ""] | float = 1.0
    while True:
        if isinstance(operator, lx.TaggedLinearOperator):
            operator = operator.operator
        elif isinstance(operator, lx.MulLinearOperator):
            # √c
            root = jnp.sqrt(operator.scalar)
            # s ← s / √c, for Λ = c Λ₀
            scale = scale / root
            operator = operator.operator
        elif isinstance(operator, lx.DivLinearOperator):
            # √c
            root = jnp.sqrt(operator.scalar)
            # s ← s √c, for Λ = Λ₀ / c
            scale = scale * root
            operator = operator.operator
        else:
            break
    if not isinstance(operator, SparseOperator) and _contains_sparse(operator):
        # A composite with a sparse block (e.g. BlockDiag(sparse, dense)) has
        # no structured Cholesky (its sparse factor is not a lineax
        # operator); keep the wrapper, which takes the dense Cholesky.
        return original, 1.0
    return operator, scale


def _contains_sparse(operator: lx.AbstractLinearOperator) -> bool:
    """Whether a `SparseOperator` sits anywhere inside ``operator``.

    Args:
        operator: The operator to search.

    Returns:
        ``True`` if any pytree leaf of ``operator`` is a `SparseOperator`.
    """
    leaves = jax.tree.leaves(operator, is_leaf=lambda x: isinstance(x, SparseOperator))
    return any(isinstance(leaf, SparseOperator) for leaf in leaves)


def _dense_precision_draws(
    precision: lx.AbstractLinearOperator,
    eps: Float[Array, "S N"],
) -> Float[Array, "S N"]:
    """Draws Λ^(−½) z through the dense eigendecomposition of Λ.

        Λ = Φ diag(γ) Φᵀ,   Λ^(−½) = Φ diag(γ^(−½)) Φᵀ

    The fallback of ``sample`` for a dense Λ whose Cholesky fails.

    Args:
        precision: The precision Λ, densified here.
        eps: Standard normal noise z, one row per draw.

    Returns:
        The draws Λ^(−½) z, one row per draw.
    """
    # ½ (Λ + Λᵀ)
    symmetric_precision = symmetrize(precision.as_matrix())
    # Λ = Φ diag(γ) Φᵀ
    eigenvalues, eigenvectors = jnp.linalg.eigh(symmetric_precision)
    # γ^(−½)
    inverse_root_eigenvalues = jax.lax.rsqrt(eigenvalues)
    # Φ diag(γ^(−½))
    scaled = einx.multiply(
        "row eig, eig -> row eig", eigenvectors, inverse_root_eigenvalues
    )
    # Λ^(−½) = Φ diag(γ^(−½)) Φᵀ
    inverse_root = einsum(scaled, eigenvectors, "row eig, col eig -> row col")
    # Λ^(−½) z, per draw
    return einsum(inverse_root, eps, "row col, draw col -> draw row")
