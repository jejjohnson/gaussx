"""GaussX recipes -- Layer 3 cross-library patterns."""

from gaussx._recipes._conditional import base_conditional
from gaussx._recipes._cvi import (
    GaussianSites,
    cvi_update_sites,
    sites_to_precision,
)
from gaussx._recipes._dare import DAREResult, dare
from gaussx._recipes._ensemble import (
    ensemble_covariance,
    ensemble_cross_covariance,
)
from gaussx._recipes._interpolation import conditional_interpolate
from gaussx._recipes._kalman import (
    FilterState,
    kalman_filter,
    kalman_gain,
    rts_smoother,
)
from gaussx._recipes._kl_divergence import gauss_kl
from gaussx._recipes._kronecker_gp import (
    kronecker_mll,
    kronecker_posterior_predictive,
)
from gaussx._recipes._love import LOVECache, love_cache, love_variance
from gaussx._recipes._natural import (
    mean_cov_to_natural,
    natural_to_mean_cov,
)
from gaussx._recipes._pairwise_marginals import pairwise_marginals
from gaussx._recipes._parameterizations import (
    expectation_to_meanvar,
    expectation_to_natural,
    meanvar_to_expectation,
    meanvar_to_natural,
    natural_to_expectation,
    natural_to_meanvar,
)
from gaussx._recipes._spingp import spingp_log_likelihood, spingp_posterior
from gaussx._recipes._ssm_natural import (
    expectations_to_ssm,
    naturals_to_ssm,
    ssm_to_expectations,
    ssm_to_naturals,
)


__all__ = [
    "DAREResult",
    "FilterState",
    "GaussianSites",
    "LOVECache",
    "base_conditional",
    "conditional_interpolate",
    "cvi_update_sites",
    "dare",
    "ensemble_covariance",
    "ensemble_cross_covariance",
    "expectation_to_meanvar",
    "expectation_to_natural",
    "expectations_to_ssm",
    "gauss_kl",
    "kalman_filter",
    "kalman_gain",
    "kronecker_mll",
    "kronecker_posterior_predictive",
    "love_cache",
    "love_variance",
    "mean_cov_to_natural",
    "meanvar_to_expectation",
    "meanvar_to_natural",
    "natural_to_expectation",
    "natural_to_mean_cov",
    "natural_to_meanvar",
    "naturals_to_ssm",
    "pairwise_marginals",
    "rts_smoother",
    "sites_to_precision",
    "spingp_log_likelihood",
    "spingp_posterior",
    "ssm_to_expectations",
    "ssm_to_naturals",
]
