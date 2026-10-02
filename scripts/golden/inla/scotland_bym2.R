# Golden fixture for gaussx.laplace_mode (roadmap G8).
#
# The Scotland lip cancer data (Clayton & Kaldor, 1987; R-INLA's Scotland
# dataset, 56 districts on demodata/scotland.graph) with a BYM2 Poisson model
#
#   Counts_i ~ Poisson(E_i exp(eta_i)),  eta_i = beta_0 + beta_1 X_i + b_i,
#
# at FIXED hyperparameters (tau, phi), so that R-INLA's latent mode is the
# mode of the Gaussian approximation that laplace_mode computes. The fixed
# effects get N(0, 1 / prec_fixed) priors. The latent mode (b, u*, beta_0,
# beta_1) is read from the configuration at the (single, fixed) theta. To make
# it the exact mode of gaussx's model: the VB mean correction is off, the
# bym2 block gets no stabilising diagonal (diagonal = 0; R-INLA adds ~1e-4 by
# default), and the linear predictor's own noise has precision e^25 (default
# 1e6).
#
# Run offline (CI never runs R):
#   Rscript scripts/golden/inla/scotland_bym2.R
# Needs R-INLA and jsonlite (generated with INLA 26.08.07, the Ubuntu-22.04
# binary from inla.binary.install). Writes scotland_bym2.json next to this script.

suppressMessages({
  library(INLA)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
out_dir <- if (length(script)) dirname(normalizePath(script)) else "."

data(Scotland)
graph_file <- system.file("demodata/scotland.graph", package = "INLA")
g <- inla.read.graph(graph_file)
n <- g$n

tau <- 2.0
phi <- 0.6
prec_fixed <- 0.001

Scotland$ID <- Scotland$Region
formula <- Counts ~ 1 + X + f(
  ID,
  model = "bym2",
  graph = graph_file,
  scale.model = TRUE,
  constr = TRUE,
  diagonal = 0,
  hyper = list(
    prec = list(initial = log(tau), fixed = TRUE),
    phi = list(initial = log(phi / (1 - phi)), fixed = TRUE)
  )
)
fit <- inla(
  formula,
  family = "poisson",
  data = Scotland,
  E = Scotland$E,
  control.fixed = list(prec.intercept = prec_fixed, prec = prec_fixed),
  control.predictor = list(hyper = list(prec = list(initial = 25, fixed = TRUE))),
  control.compute = list(config = TRUE),
  control.inla = list(
    strategy = "gaussian",
    int.strategy = "eb",
    control.vb = list(enable = FALSE),
    tolerance = 1e-8
  )
)

# config$mean holds the latent field only; contents' start indices also count
# the linear predictor, which comes first.
config <- fit$misc$configs$config[[1]]
contents <- fit$misc$configs$contents
predictor <- sum(contents$length[contents$tag %in% c("APredictor", "Predictor")])
slice <- function(tag) {
  k <- which(contents$tag == tag)
  config$mean[contents$start[k] - 1 - predictor + seq_len(contents$length[k])]
}
id <- slice("ID")

fixture <- list(
  description = paste(
    "R-INLA latent mode of the Scotland BYM2 Poisson model at fixed",
    "(tau, phi); regions in graph order (Region = node + 1)."
  ),
  inla_version = as.character(packageVersion("INLA")),
  n = n,
  region = Scotland$Region - 1L,
  counts = Scotland$Counts,
  expected = Scotland$E,
  covariate = Scotland$X,
  tau = tau,
  phi = phi,
  prec_fixed = prec_fixed,
  mode_b = id[1:n],
  mode_u = id[n + 1:n],
  mode_intercept = slice("(Intercept)"),
  mode_slope = slice("X")
)
write_json(
  fixture,
  file.path(out_dir, "scotland_bym2.json"),
  digits = NA,
  auto_unbox = TRUE,
  pretty = TRUE
)
cat(sprintf("intercept = %.10g, slope = %.10g\n", fixture$mode_intercept, fixture$mode_slope))
