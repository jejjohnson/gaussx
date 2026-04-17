"""GaussX state-space models -- Kalman family, SpInGP, CVI sites."""

from gaussx._ssm._cvi import (
    GaussianSites,
    cvi_update_sites,
    sites_to_precision,
)
from gaussx._ssm._dare import DAREResult, dare
from gaussx._ssm._emission import EmissionModel
from gaussx._ssm._infinite_horizon_kalman import (
    InfiniteHorizonState,
    infinite_horizon_filter,
    infinite_horizon_smoother,
)
from gaussx._ssm._kalman import (
    FilterState,
    kalman_filter,
    kalman_gain,
    rts_smoother,
)
from gaussx._ssm._pairwise_marginals import pairwise_marginals
from gaussx._ssm._parallel_kalman import (
    parallel_kalman_filter,
    parallel_rts_smoother,
)
from gaussx._ssm._spingp import spingp_log_likelihood, spingp_posterior
from gaussx._ssm._ssm_natural import (
    expectations_to_ssm,
    naturals_to_ssm,
    ssm_to_expectations,
    ssm_to_naturals,
)


__all__ = [
    "DAREResult",
    "EmissionModel",
    "FilterState",
    "GaussianSites",
    "InfiniteHorizonState",
    "cvi_update_sites",
    "dare",
    "expectations_to_ssm",
    "infinite_horizon_filter",
    "infinite_horizon_smoother",
    "kalman_filter",
    "kalman_gain",
    "naturals_to_ssm",
    "pairwise_marginals",
    "parallel_kalman_filter",
    "parallel_rts_smoother",
    "rts_smoother",
    "sites_to_precision",
    "spingp_log_likelihood",
    "spingp_posterior",
    "ssm_to_expectations",
    "ssm_to_naturals",
]
