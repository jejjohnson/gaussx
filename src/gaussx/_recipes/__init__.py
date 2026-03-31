"""GaussX recipes -- Layer 3 cross-library patterns."""

from gaussx._recipes._ensemble import (
    ensemble_covariance,
    ensemble_cross_covariance,
)
from gaussx._recipes._kalman import (
    FilterState,
    kalman_filter,
    kalman_gain,
    rts_smoother,
)
from gaussx._recipes._natural import (
    expectation_to_natural,
    natural_to_expectation,
)
from gaussx._recipes._spingp import spingp_log_likelihood, spingp_posterior


__all__ = [
    "FilterState",
    "ensemble_covariance",
    "ensemble_cross_covariance",
    "expectation_to_natural",
    "kalman_filter",
    "kalman_gain",
    "natural_to_expectation",
    "rts_smoother",
    "spingp_log_likelihood",
    "spingp_posterior",
]
