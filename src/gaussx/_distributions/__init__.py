"""Gaussian distributions x ~ 𝒩(μ, Σ) with log-density, entropy and KL helpers.

The array-level helpers (`gaussian_log_prob`, `gaussian_kl`, `conditional`, …)
import eagerly; the GMRF fields load lazily. The NumPyro distribution classes
(`MultivariateNormal`, `LGSSM`, …) live in the pyrox-gauss package.
"""

from __future__ import annotations

from typing import Any

from gaussx._distributions._conditional import (
    LinearGaussianPosterior,
    conditional,
    linear_gaussian_update,
)
from gaussx._distributions._gaussian import (
    add_jitter,
    gaussian_entropy,
    gaussian_log_prob,
    kl_standard_normal,
    quadratic_form,
)
from gaussx._distributions._information import (
    InformationDiagnostics,
    information_diagnostics,
)
from gaussx._distributions._joseph import joseph_update
from gaussx._distributions._kl import gaussian_kl
from gaussx._distributions._project import project
from gaussx._distributions._sample import sample_joint_conditional, sample_mvn


__all__ = [
    "BYM2GMRF",
    "ConstrainedGMRF",
    "GaussianMRF",
    "InformationDiagnostics",
    "IntrinsicGMRF",
    "LinearGaussianPosterior",
    "add_jitter",
    "conditional",
    "gaussian_entropy",
    "gaussian_kl",
    "gaussian_log_prob",
    "information_diagnostics",
    "joseph_update",
    "kl_standard_normal",
    "linear_gaussian_update",
    "project",
    "quadratic_form",
    "sample_joint_conditional",
    "sample_mvn",
]


# The GMRF fields load through the PEP 562 ``__getattr__`` below.
def __getattr__(name: str) -> Any:
    if name in ("GaussianMRF", "IntrinsicGMRF", "ConstrainedGMRF", "BYM2GMRF"):
        from gaussx._distributions import _gmrf

        return getattr(_gmrf, name)
    raise AttributeError(f"module 'gaussx._distributions' has no attribute {name!r}")
