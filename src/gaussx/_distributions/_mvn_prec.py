"""Multivariate normal distribution parameterized by precision operator."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.distributions.util import lazy_property, validate_sample

from gaussx._distributions._gaussian import _LOG_2PI
from gaussx._distributions._utils import _reshape_batch, _reshape_samples
from gaussx._einx import einsum, rearrange
from gaussx._linalg._symmetrize import symmetrize
from gaussx._operators._sparse import SparseOperator
from gaussx._primitives._cholesky import cholesky as _cholesky
from gaussx._primitives._diag import diag as _diag
from gaussx._primitives._inv import inv as _inv
from gaussx._primitives._solve import solve as _solve
from gaussx._sparse._factor import SparseCholeskyFactor
from gaussx._strategies._auto import AutoSolver
from gaussx._strategies._base import AbstractSolverStrategy


class MultivariateNormalPrecision(dist.Distribution):
    """Multivariate normal parameterized by a precision (inverse covariance) operator.

    This is the natural parameterization for many inference algorithms
    (e.g. message passing, variational inference in natural coordinates).
    The precision operator ``Lambda`` satisfies ``Lambda = Sigma^{-1}``.

    Requires the ``numpyro`` optional extra
    (``pip install "gaussx[numpyro]"``).

    Args:
        loc: Mean vector of shape ``(N,)``.
        prec_operator: Precision matrix as a lineax linear operator of
            shape ``(N, N)``.
        solver: Solver strategy for ``solve`` and ``logdet``. Defaults
            to ``AutoSolver()``.
        validate_args: Whether to validate input arguments.

    Examples:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from gaussx._distributions import MultivariateNormalPrecision
        >>> Lambda = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(3), lx.positive_semidefinite_tag
        ... )
        >>> d = MultivariateNormalPrecision(jnp.zeros(3), Lambda)
        >>> d.log_prob(jnp.ones(3))
    """

    arg_constraints = {"loc": dist.constraints.real_vector}  # noqa: RUF012
    support = dist.constraints.real_vector
    reparametrized_params = ["loc"]  # noqa: RUF012
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

    def _log_prob_single(self, residual: Float[Array, " N"]) -> Float[Array, ""]:
        quad = jnp.sum(residual * self.prec_operator.mv(residual), axis=-1)
        ld = self.solver.logdet(self.prec_operator)
        n = self.loc.shape[-1]
        return -0.5 * (n * _LOG_2PI - ld + quad)

    @validate_sample
    def log_prob(self, value: Float[Array, "*batch N"]) -> Float[Array, "*batch"]:
        residual = value - self.loc
        leading_shape = residual.shape[:-1]
        residual_flat = rearrange(residual, "... D -> (...) D")
        log_prob_flat = jax.vmap(self._log_prob_single)(residual_flat)
        return _reshape_batch(log_prob_flat, leading_shape)

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*batch N"]:
        if key is None:
            raise ValueError(
                "PRNG key must be provided to sample from MultivariateNormalPrecision."
            )
        shape = sample_shape + self.batch_shape + self.event_shape
        # Draw in the parameters' dtype: default-float noise meets a float32
        # factor in the solve below, which lineax rejects under x64.
        dtype = jnp.result_type(self.loc, self.prec_operator.in_structure().dtype)
        eps = jax.random.normal(key, shape=shape, dtype=dtype)  # type: ignore[arg-type]
        eps_flat = rearrange(eps, "... D -> (...) D")

        # Look through tags and scalar multiples so a structured precision
        # (e.g. 2 * Kronecker) keeps its structured Cholesky factor.
        precision, scale = _unwrap_scaled(self.prec_operator)
        L = _cholesky(precision)

        def _solve_one(z):
            if isinstance(L, SparseCholeskyFactor):
                # A SparseOperator's factor is of P Λ Pᵀ: x = Pᵀ L⁻ᵀ z.
                return L.solve_lower_transpose(z)
            return _solve(L.T, z)

        samples_flat = scale * jax.vmap(_solve_one)(eps_flat)
        if isinstance(precision, lx.MatrixLinearOperator):
            # A dense Cholesky that fails numerically (an ill-conditioned or
            # only semi-definite precision) gives non-finite draws; take the
            # dense symmetric inverse square root then. Dense precisions only:
            # lax.cond stages both branches under jit, and a dense branch
            # would put N x N buffers into every structured or sparse sample.
            structured = samples_flat
            samples_flat = jax.lax.cond(
                jnp.all(jnp.isfinite(structured)),
                lambda: structured,
                lambda: _dense_precision_draws(self.prec_operator, eps_flat),
            )
        return self.loc + _reshape_samples(samples_flat, shape[:-1])

    @lazy_property
    def mean(self) -> Float[Array, "*batch N"]:
        return self.loc

    @lazy_property
    def variance(self) -> Float[Array, "*batch N"]:
        return jnp.broadcast_to(
            _diag(_inv(self.prec_operator)), self.batch_shape + self.event_shape
        )

    def entropy(self) -> Float[Array, ""]:
        n = self.loc.shape[-1]
        ld = self.solver.logdet(self.prec_operator)
        return 0.5 * (n * (1.0 + _LOG_2PI) - ld)


def _unwrap_scaled(
    operator: lx.AbstractLinearOperator,
) -> tuple[lx.AbstractLinearOperator, Float[Array, ""] | float]:
    """Strip tags and scalar multiples from a precision ``c Λ``.

    Returns ``(Λ, s)`` with ``s = 1 / sqrt(c)``, so that ``L_Λ⁻ᵀ z * s`` is a
    draw from ``N(0, (c Λ)⁻¹)``. A non-positive ``c`` gives a non-finite
    ``s``, which the dense fallback in ``sample`` then handles.
    """
    original = operator
    scale: Float[Array, ""] | float = 1.0
    while True:
        if isinstance(operator, lx.TaggedLinearOperator):
            operator = operator.operator
        elif isinstance(operator, lx.MulLinearOperator):
            scale = scale / jnp.sqrt(operator.scalar)
            operator = operator.operator
        elif isinstance(operator, lx.DivLinearOperator):
            scale = scale * jnp.sqrt(operator.scalar)
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
    """Whether a `SparseOperator` sits anywhere inside ``operator``."""
    leaves = jax.tree.leaves(operator, is_leaf=lambda x: isinstance(x, SparseOperator))
    return any(isinstance(leaf, SparseOperator) for leaf in leaves)


def _dense_precision_draws(
    precision: lx.AbstractLinearOperator,
    eps: Float[Array, "S N"],
) -> Float[Array, "S N"]:
    """``Λ^{-1/2} z`` through the dense eigendecomposition of ``Λ``."""
    eigvals, eigvecs = jnp.linalg.eigh(symmetrize(precision.as_matrix()))
    scaled = einx.multiply("i k, k -> i k", eigvecs, jax.lax.rsqrt(eigvals))
    root = einsum(scaled, eigvecs, "i k, j k -> i j")
    return einsum(root, eps, "i j, s j -> s i")
