# Golden fixture for gaussx.vb_mean_correction (roadmap G10).
#
# Rare binary detections along a transect, a Bernoulli (logit) model with an
# intercept and an RW2 effect,
#
#   y_i ~ Bernoulli(plogis(eta_i)),  eta_i = beta_0 + u_i,  i = 1..n,
#
# at FIXED tau, with u ~ RW2(tau) (unscaled, no stabilising diagonal) under
# the hard constraints sum(u) = 0 and sum((t - mean(t)) u) = 0, and
# beta_0 ~ N(0, 1 / prec_fixed). The latent vector is (u_1..u_n, beta_0),
# nodes 0..n in R-INLA's (0-based) numbering. Four fits at the same theta:
#
#   - "gaussian" with the VB correction off: the Laplace mode;
#   - the VB mean correction (strategy = "mean") with f.enable.limit = 0:
#     the fixed effect only, subspace = {n};
#   - with f.enable.limit[1] = 10, below the RW2's size: R-INLA then adds
#     every 4th RW2 node, subspace = {2, 6, ..., 38, n} (read from the
#     verbose log, "Node[k]" lines; R does not return it);
#   - with f.enable.limit[1] = 100: all n + 1 latent nodes.
#
# R-INLA stops its VB iterations once max |dx| / sd < ~0.01, so its
# corrected means are converged to a few 1e-3 sd.
#
# R-INLA's latent field includes the linear predictor with noise precision
# 1e6 (its default); the corrected means differ from the noise-free model by
# about that much. The default tolerances are kept: raising the predictor
# precision to e^25 (as scotland_bym2.R does) or setting tolerance = 1e-10
# leaves R-INLA 26.08's Bernoulli mode far from converged.
#
# Run offline (CI never runs R):
#   Rscript scripts/golden/inla/bernoulli_rw2_vb.R
# Needs R-INLA and jsonlite (generated with INLA 26.08.07, the Ubuntu-22.04
# binary from inla.binary.install). Writes bernoulli_rw2_vb.json next to this
# script.

suppressMessages({
  library(INLA)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
out_dir <- if (length(script)) dirname(normalizePath(script)) else "."

set.seed(20241)
n <- 40
t <- seq_len(n)
y <- rbinom(n, 1, plogis(-2 + 1.5 * sin(t / 5)))
tau <- 2.0
prec_fixed <- 0.001
trend <- matrix(t - mean(t), 1, n)

fit <- function(vb) {
  inla(
    y ~ 1 + f(
      t,
      model = "rw2",
      constr = TRUE,
      extraconstr = list(A = trend, e = 0),
      scale.model = FALSE,
      diagonal = 0,
      hyper = list(prec = list(initial = log(tau), fixed = TRUE))
    ),
    family = "binomial",
    Ntrials = 1,
    data = data.frame(y = y, t = t),
    control.fixed = list(prec.intercept = prec_fixed),
    control.compute = list(config = TRUE),
    control.inla = list(
      strategy = "gaussian",
      int.strategy = "eb",
      control.vb = vb
    )
  )
}
means <- function(f) c(f$summary.random$t$mean, f$summary.fixed$mean)

laplace <- fit(list(enable = FALSE))
vb <- function(limit) fit(list(enable = TRUE, strategy = "mean", f.enable.limit = limit))
vb_fixed <- vb(c(0, 0, 0, 0))
vb_sparse <- vb(c(10, 25, 1024, 768))
vb_all <- vb(c(100, 25, 1024, 768))

fixture <- list(
  description = paste(
    "R-INLA Bernoulli / RW2 + intercept at fixed tau: latent means",
    "(u_1..u_n, beta_0) of the Laplace approximation and of the VB mean",
    "correction on {beta_0}, on subspace_sparse (0-based) and on all nodes."
  ),
  inla_version = as.character(packageVersion("INLA")),
  n = n,
  y = y,
  tau = tau,
  prec_fixed = prec_fixed,
  mean_laplace = means(laplace),
  sd_laplace = c(laplace$summary.random$t$sd, laplace$summary.fixed$sd),
  subspace_sparse = c(seq(2L, 38L, by = 4L), n),
  mean_vb_fixed = means(vb_fixed),
  mean_vb_sparse = means(vb_sparse),
  mean_vb_all = means(vb_all)
)
write_json(
  fixture,
  file.path(out_dir, "bernoulli_rw2_vb.json"),
  digits = NA,
  auto_unbox = TRUE,
  pretty = TRUE
)
cat(sprintf(
  "intercept: laplace %.6f, vb(fixed) %.6f, vb(sparse) %.6f, vb(all) %.6f\n",
  fixture$mean_laplace[n + 1], fixture$mean_vb_fixed[n + 1],
  fixture$mean_vb_sparse[n + 1], fixture$mean_vb_all[n + 1]
))
