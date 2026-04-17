"""GaussX distributions -- NumPyro-compatible Gaussian distributions.

These require ``numpyro`` at import time.  When numpyro is not installed
the distribution classes are unavailable, but the rest of gaussx
(primitives, operators, strategies) works fine.
"""

from __future__ import annotations

__all__: list[str] = []

try:
    from gaussx._distributions._mvn import MultivariateNormal
    from gaussx._distributions._mvn_prec import MultivariateNormalPrecision

    __all__ += ["MultivariateNormal", "MultivariateNormalPrecision"]
except ImportError:  # numpyro not installed
    pass
