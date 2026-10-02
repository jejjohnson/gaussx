# Golden fixture for gaussx.generalized_variance_scale (roadmap G7).
#
# R-INLA's scaling constant of the ICAR structure matrix of the Scotland lip
# cancer graph (56 districts, R-INLA's demodata/scotland.graph), under the
# sum-to-zero constraint, as computed by inla.scale.model (Sorbye & Rue, 2014).
#
# Run offline (CI never runs R):
#   Rscript scripts/golden/inla/scotland_scale.R
# Needs R-INLA (any version providing inla.scale.model; generated with
# 23.04.24) and jsonlite. Writes scotland_scale.json next to this script.

suppressMessages({
  library(INLA)
  library(Matrix)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
out_dir <- if (length(script)) dirname(normalizePath(script)) else "."

g <- inla.read.graph(system.file("demodata/scotland.graph", package = "INLA"))
n <- g$n

# Structure matrix R = D - W of the unweighted graph.
W <- inla.graph2matrix(g)
diag(W) <- 0
R <- Diagonal(n, rowSums(W)) - W

constr <- list(A = matrix(1, 1, n), e = 0)
R_scaled <- inla.scale.model(R, constr = constr)
scale <- R_scaled[1, 1] / R[1, 1]

# Edges once each (sender > receiver), 0-based for Python.
edges <- summary(as(W, "TsparseMatrix"))
edges <- edges[edges$i > edges$j, ]

fixture <- list(
  description = paste(
    "R-INLA inla.scale.model scaling constant of the Scotland ICAR",
    "structure matrix under sum-to-zero; edges are 0-based, each once."
  ),
  inla_version = as.character(packageVersion("INLA")),
  n = n,
  senders = edges$i - 1L,
  receivers = edges$j - 1L,
  scale = scale,
  scaled_diagonal = as.numeric(diag(R_scaled))
)
write_json(
  fixture,
  file.path(out_dir, "scotland_scale.json"),
  digits = NA,
  auto_unbox = TRUE,
  pretty = TRUE
)
cat(sprintf("scale = %.15g\n", scale))
