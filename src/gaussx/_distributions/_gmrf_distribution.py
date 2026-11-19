"""NumPyro face of a GMRF: one `Distribution` that wraps any of the four fields.

`GaussianMRF`, `ConstrainedGMRF`, `IntrinsicGMRF` and `BYM2GMRF` are equinox
modules with no NumPyro in them, so gaussx's Laplace and VB code and argminax
take them as linear-algebra objects. `GMRFDistribution` hands one to
``numpyro.sample``; it delegates ``log_prob``, ``sample``, ``mean`` and
``variance`` to the field and adds nothing to the numerics.
"""

from __future__ import annotations

import jax
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.distributions.util import validate_sample

from gaussx._distributions._gmrf import (
    BYM2GMRF,
    ConstrainedGMRF,
    GaussianMRF,
    IntrinsicGMRF,
)


class GMRFDistribution(dist.Distribution):
    """A GMRF field as a NumPyro distribution, for ``numpyro.sample``.

        x ~ field,   log p(x) = field.log_prob(x)

    The field keeps every structural dispatch (sparse Cholesky, factored
    eigenbasis, perturbation-optimisation, kriging); this class only gives
    it NumPyro's support, event shape and seeding. Draws are differentiable
    in the field's parameters, so the distribution is reparameterised.

    Requires the ``numpyro`` optional extra.

    Args:
        field: The field, a `GaussianMRF`, `ConstrainedGMRF`, `IntrinsicGMRF`
            or `BYM2GMRF`.
        validate_args: Whether to validate input arguments.

    Examples:
        A proper field on four iid nodes, sampled inside a NumPyro model:

        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import numpyro
        >>> import gaussx as gx
        >>> precision = lx.MatrixLinearOperator(
        ...     2.0 * jnp.eye(4), lx.positive_semidefinite_tag
        ... )
        >>> field = gx.GaussianMRF(jnp.zeros(4), precision)
        >>> def model():
        ...     return numpyro.sample("x", gx.GMRFDistribution(field))
        >>> numpyro.handlers.seed(model, 0)().shape
        (4,)
    """

    arg_constraints = {}  # noqa: RUF012
    support = dist.constraints.real_vector
    pytree_data_fields = ("field",)

    def __init__(
        self,
        field: GaussianMRF | ConstrainedGMRF | IntrinsicGMRF | BYM2GMRF,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.field = field
        super().__init__(
            batch_shape=(), event_shape=field.event_shape, validate_args=validate_args
        )

    @validate_sample
    def log_prob(self, value: Float[Array, "*sample N"]) -> Float[Array, "*sample"]:
        """The field's log-density.

        Args:
            value: Points x, shape ``(..., N)``.

        Returns:
            Log-densities, shape ``(...)``.
        """
        return self.field.log_prob(value)

    def sample(
        self,
        key: jax.Array | None,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample N"]:
        """The field's draws.

        Args:
            key: PRNG key. ``None`` means ``jax.random.PRNGKey(0)``.
            sample_shape: Leading sample shape.

        Returns:
            Draws, shape ``sample_shape + (N,)``.
        """
        return self.field.sample(key, sample_shape)

    @property
    def mean(self) -> Float[Array, " N"]:
        """The field's mean."""
        return self.field.mean

    @property
    def variance(self) -> Float[Array, " N"]:
        """The field's marginal variances."""
        return self.field.variance
