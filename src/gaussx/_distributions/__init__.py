"""GaussX distributions -- Gaussian objects + log-prob / KL / jitter helpers."""

from __future__ import annotations

from gaussx._distributions._conditional import conditional
from gaussx._distributions._gaussian import (
    add_jitter,
    gaussian_entropy,
    gaussian_log_prob,
    kl_standard_normal,
    quadratic_form,
)
from gaussx._distributions._joseph import joseph_update
from gaussx._distributions._kl import dist_kl_divergence
from gaussx._distributions._project import project


__all__ = [
    "add_jitter",
    "conditional",
    "dist_kl_divergence",
    "gaussian_entropy",
    "gaussian_log_prob",
    "joseph_update",
    "kl_standard_normal",
    "project",
    "quadratic_form",
]


try:
    from gaussx._distributions._mvn import MultivariateNormal
    from gaussx._distributions._mvn_prec import MultivariateNormalPrecision

    __all__ += ["MultivariateNormal", "MultivariateNormalPrecision"]
except ModuleNotFoundError as _e:
    if _e.name != "numpyro":
        raise
