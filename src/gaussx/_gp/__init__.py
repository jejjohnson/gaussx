"""GaussX GP-specific helpers -- ELBO, whitening, prediction caches, SVGP.

Sparse-GP convention (gh-353): new helpers follow it, existing ones are
being moved onto it with one-release deprecations.

| Object | Name | Shape / type |
|---|---|---|
| Inducing covariance (first argument) | ``K_zz`` | ``(M, M)`` array or PSD operator |
| Cross-covariance (data x inducing) | ``K_xz`` | ``(N, M)`` |
| Prior at the evaluation points | ``K_xx`` / ``K_xx_diag`` | ``(N, N)`` / ``(N,)`` |
| Variational mean | ``q_mu`` | ``(M, R)`` or ``(M,)`` |
| Variational root | ``q_sqrt`` | ``(R, M, M)`` or ``(M, R)`` |

Functions take no ``solver=`` they would ignore.
"""

from gaussx._gp._base_conditional import base_conditional, sparse_conditional
from gaussx._gp._collapsed_elbo import collapsed_elbo
from gaussx._gp._elbo import variational_elbo_gaussian, variational_elbo_mc
from gaussx._gp._gauss_kl import gauss_kl
from gaussx._gp._interpolation import conditional_interpolate, rts_interpolate
from gaussx._gp._kronecker_gp import (
    kronecker_mll,
    kronecker_posterior_predictive,
)
from gaussx._gp._loo import LOOResult, leave_one_out_cv
from gaussx._gp._love import (
    LOVECache,
    love_cache,
    love_residual,
    love_variance,
    love_variance_error_bound,
)
from gaussx._gp._matheron import matheron_update
from gaussx._gp._oilmm import oilmm_back_project, oilmm_project
from gaussx._gp._prediction_cache import (
    PredictionCache,
    build_prediction_cache,
    predict_mean,
    predict_variance,
)
from gaussx._gp._svgp import whitened_svgp_predict
from gaussx._gp._svgp_variance import svgp_variance_adjustment
from gaussx._gp._unwhiten import unwhiten, unwhiten_covariance, whiten_covariance


__all__ = [
    "LOOResult",
    "LOVECache",
    "PredictionCache",
    "base_conditional",
    "build_prediction_cache",
    "collapsed_elbo",
    "conditional_interpolate",
    "gauss_kl",
    "kronecker_mll",
    "kronecker_posterior_predictive",
    "leave_one_out_cv",
    "love_cache",
    "love_residual",
    "love_variance",
    "love_variance_error_bound",
    "matheron_update",
    "oilmm_back_project",
    "oilmm_project",
    "predict_mean",
    "predict_variance",
    "rts_interpolate",
    "sparse_conditional",
    "svgp_variance_adjustment",
    "unwhiten",
    "unwhiten_covariance",
    "variational_elbo_gaussian",
    "variational_elbo_mc",
    "whiten_covariance",
    "whitened_svgp_predict",
]
