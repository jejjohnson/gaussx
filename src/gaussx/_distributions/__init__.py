"""Gaussian distributions x ~ 𝒩(μ, Σ) with log-density, entropy and KL helpers.

The array-level helpers (`gaussian_log_prob`, `gaussian_kl`, `conditional`, …)
import eagerly; the numpyro-backed distribution classes load lazily.
"""

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
from gaussx._distributions._kl import gaussian_kl
from gaussx._distributions._project import project
from gaussx._distributions._sample import sample_joint_conditional, sample_mvn


__all__ = [
    "BYM2GMRF",
    "LGSSM",
    "AbstractMultivariateNormal",
    "ConstrainedGMRF",
    "GaussianMRF",
    "IntrinsicGMRF",
    "LGSSMFactory",
    "MarkovGaussian",
    "MaskedLGSSM",
    "MultivariateNormal",
    "MultivariateNormalPrecision",
    "add_jitter",
    "conditional",
    "gaussian_entropy",
    "gaussian_kl",
    "gaussian_log_prob",
    "joseph_update",
    "kl_standard_normal",
    "project",
    "quadratic_form",
    "sample_joint_conditional",
    "sample_mvn",
]


# The distribution classes (MultivariateNormal, MultivariateNormalPrecision,
# MarkovGaussian, the LGSSM and GMRF families) need the optional numpyro
# extra, so they load through the PEP 562 ``__getattr__`` below. Importing
# ``gaussx._distributions`` then always succeeds, which keeps the helpers
# above available in a base install. Accessing a class without numpyro raises
# ``ModuleNotFoundError`` with ``name='numpyro'`` rather than a confusing
# ``ImportError``, and the optional-dependency guard in
# ``gaussx/__init__.py`` catches exactly that error.
def __getattr__(name: str) -> Any:
    if name in ("LGSSM", "MaskedLGSSM", "LGSSMFactory"):
        from gaussx._distributions import _lgssm

        return getattr(_lgssm, name)
    if name in ("GaussianMRF", "IntrinsicGMRF", "ConstrainedGMRF", "BYM2GMRF"):
        from gaussx._distributions import _gmrf

        return getattr(_gmrf, name)
    if name == "MarkovGaussian":
        from gaussx._distributions._markov_gaussian import MarkovGaussian

        return MarkovGaussian
    if name == "AbstractMultivariateNormal":
        from gaussx._distributions._mvn_base import AbstractMultivariateNormal

        return AbstractMultivariateNormal
    if name == "MultivariateNormal":
        from gaussx._distributions._mvn import MultivariateNormal

        return MultivariateNormal
    if name == "MultivariateNormalPrecision":
        from gaussx._distributions._mvn_prec import MultivariateNormalPrecision

        return MultivariateNormalPrecision
    raise AttributeError(f"module 'gaussx._distributions' has no attribute {name!r}")
