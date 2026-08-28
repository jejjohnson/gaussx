"""GaussX distributions -- Gaussian objects + log-prob / KL / jitter helpers."""

from __future__ import annotations

from typing import Any

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
    "LGSSM",
    "LGSSMFactory",
    "MarkovGaussian",
    "MaskedLGSSM",
    "MultivariateNormal",
    "MultivariateNormalPrecision",
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


# MultivariateNormal / MultivariateNormalPrecision / MarkovGaussian / the LGSSM family
# require the optional ``numpyro`` dependency. Exposing them via PEP 562
# ``__getattr__`` means:
#
# - Importing ``gaussx._distributions`` always succeeds (so the non-numpyro
#   helpers above are available in a base install).
# - ``from gaussx._distributions import MultivariateNormal`` raises
#   ``ModuleNotFoundError`` with ``name='numpyro'`` when numpyro is absent,
#   which the optional-dependency guard in ``gaussx/__init__.py`` catches.
# - Users get the expected ``ModuleNotFoundError`` (not a confusing
#   ``ImportError``) if they directly access the name without the extra.
def __getattr__(name: str) -> Any:
    if name in ("LGSSM", "MaskedLGSSM", "LGSSMFactory"):
        from gaussx._distributions import _lgssm

        return getattr(_lgssm, name)
    if name == "MarkovGaussian":
        from gaussx._distributions._markov_gaussian import MarkovGaussian

        return MarkovGaussian
    if name == "MultivariateNormal":
        from gaussx._distributions._mvn import MultivariateNormal

        return MultivariateNormal
    if name == "MultivariateNormalPrecision":
        from gaussx._distributions._mvn_prec import MultivariateNormalPrecision

        return MultivariateNormalPrecision
    raise AttributeError(f"module 'gaussx._distributions' has no attribute {name!r}")
