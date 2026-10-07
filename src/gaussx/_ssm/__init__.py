"""GaussX state-space models -- Kalman family, SpInGP, CVI sites, SDE kernels."""

from gaussx._ssm._autocovariance import sde_autocovariance
from gaussx._ssm._composition import ProductSDE, QuasiPeriodicSDE, SumSDE
from gaussx._ssm._constant import ConstantSDE
from gaussx._ssm._cvi import (
    GaussianSites,
    cvi_update_sites,
    sites_to_precision,
)
from gaussx._ssm._dare import DAREResult, dare
from gaussx._ssm._discretise import (
    discretize_mfd,
    discretize_mfd_sequence,
    process_noise_covariance,
)
from gaussx._ssm._emission import EmissionModel
from gaussx._ssm._infinite_horizon_kalman import (
    infinite_horizon_filter,
    infinite_horizon_smoother,
)
from gaussx._ssm._kalman import (
    FilterState,
    kalman_filter,
    kalman_gain,
    rts_smoother,
)
from gaussx._ssm._matern import MaternSDE
from gaussx._ssm._meanfield_kalman import (
    meanfield_kalman_filter,
    meanfield_rts_smoother,
)
from gaussx._ssm._nonlinear_filter import (
    nonlinear_kalman_filter,
    nonlinear_rts_smoother,
    nonlinear_rts_step,
)
from gaussx._ssm._nonlinear_update import (
    masked_moment_inputs,
    nonlinear_kalman_predict,
    nonlinear_kalman_update,
)
from gaussx._ssm._pairwise_marginals import pairwise_marginals
from gaussx._ssm._parallel_kalman import (
    parallel_kalman_filter,
    parallel_rts_smoother,
)
from gaussx._ssm._periodic import CosineSDE, PeriodicSDE
from gaussx._ssm._sde_kernel import SDEKernel, SDEParams
from gaussx._ssm._sde_kl import LinearizedSDE, linearize_sde, sde_kl_divergence
from gaussx._ssm._site_natural import (
    cavity_from_marginal,
    site_mean_var_from_natural,
    site_natural_from_tilted,
)
from gaussx._ssm._spingp import spingp_log_likelihood, spingp_posterior
from gaussx._ssm._ssm_natural import (
    expectations_to_ssm,
    naturals_to_ssm,
    ssm_to_expectations,
    ssm_to_naturals,
)
from gaussx._ssm._udl import (
    UDLDecomposition,
    udl_decomposition,
    udl_from_ssm_params,
    udl_to_ssm_params,
)
from gaussx._ssm._wiener import IntegratedWienerSDE


__all__ = [
    "ConstantSDE",
    "CosineSDE",
    "DAREResult",
    "EmissionModel",
    "FilterState",
    "GaussianSites",
    "IntegratedWienerSDE",
    "LinearizedSDE",
    "MaternSDE",
    "PeriodicSDE",
    "ProductSDE",
    "QuasiPeriodicSDE",
    "SDEKernel",
    "SDEParams",
    "SumSDE",
    "UDLDecomposition",
    "cavity_from_marginal",
    "cvi_update_sites",
    "dare",
    "discretize_mfd",
    "discretize_mfd_sequence",
    "expectations_to_ssm",
    "infinite_horizon_filter",
    "infinite_horizon_smoother",
    "kalman_filter",
    "kalman_gain",
    "linearize_sde",
    "masked_moment_inputs",
    "meanfield_kalman_filter",
    "meanfield_rts_smoother",
    "naturals_to_ssm",
    "nonlinear_kalman_filter",
    "nonlinear_kalman_predict",
    "nonlinear_kalman_update",
    "nonlinear_rts_smoother",
    "nonlinear_rts_step",
    "pairwise_marginals",
    "parallel_kalman_filter",
    "parallel_rts_smoother",
    "process_noise_covariance",
    "rts_smoother",
    "sde_autocovariance",
    "sde_kl_divergence",
    "site_mean_var_from_natural",
    "site_natural_from_tilted",
    "sites_to_precision",
    "spingp_log_likelihood",
    "spingp_posterior",
    "ssm_to_expectations",
    "ssm_to_naturals",
    "udl_decomposition",
    "udl_from_ssm_params",
    "udl_to_ssm_params",
]


def __getattr__(name: str):
    # Deprecated aliases warn on access (gh-364).
    from gaussx._ssm._infinite_horizon_kalman import (
        _DEPRECATED_ALIASES,
        _deprecated_alias,
    )

    if name in _DEPRECATED_ALIASES:
        return _deprecated_alias(name)
    raise AttributeError(f"module 'gaussx._ssm' has no attribute {name!r}")
