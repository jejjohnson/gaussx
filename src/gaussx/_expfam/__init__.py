"""GaussX exponential family -- Gaussian in natural parameter form."""

from gaussx._expfam._gaussian import (
    GaussianExpFam,
    fisher_info,
    kl_divergence,
    log_partition,
    sufficient_stats,
    to_expectation,
    to_natural,
)


__all__ = [
    "GaussianExpFam",
    "fisher_info",
    "kl_divergence",
    "log_partition",
    "sufficient_stats",
    "to_expectation",
    "to_natural",
]
