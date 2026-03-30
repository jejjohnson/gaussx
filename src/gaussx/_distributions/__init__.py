"""GaussX distributions -- NumPyro-compatible Gaussian distributions."""

from gaussx._distributions._mvn import MultivariateNormal
from gaussx._distributions._mvn_prec import MultivariateNormalPrecision


__all__ = [
    "MultivariateNormal",
    "MultivariateNormalPrecision",
]
