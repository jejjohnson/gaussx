# Changelog

## [0.0.8](https://github.com/jejjohnson/gaussx/compare/v0.0.7...v0.0.8) (2026-04-07)


### Features

* add 7 new lineax operators ([#37](https://github.com/jejjohnson/gaussx/issues/37), [#38](https://github.com/jejjohnson/gaussx/issues/38), [#41](https://github.com/jejjohnson/gaussx/issues/41), [#42](https://github.com/jejjohnson/gaussx/issues/42), [#44](https://github.com/jejjohnson/gaussx/issues/44)) ([#85](https://github.com/jejjohnson/gaussx/issues/85)) ([ccf012f](https://github.com/jejjohnson/gaussx/commit/ccf012ff9a60df8e118cd125a4ab283211e2bcf0))
* add correct_variance flag and ComposedSolver ([#81](https://github.com/jejjohnson/gaussx/issues/81)) ([901dd53](https://github.com/jejjohnson/gaussx/commit/901dd5337a83d75d1bb25d83d14b2f67f68f658a))
* integrator unification — GaussHermiteIntegrator, unified ELL & ELBO ([#83](https://github.com/jejjohnson/gaussx/issues/83)) ([47cee23](https://github.com/jejjohnson/gaussx/commit/47cee2319783a40b3c15f8764fcaf76a0aed4425))

## [0.0.7](https://github.com/jejjohnson/gaussx/compare/v0.0.6...v0.0.7) (2026-03-31)


### Features

* add SSM expectation params and Joseph-form covariance update ([#17](https://github.com/jejjohnson/gaussx/issues/17)) ([45ffc4e](https://github.com/jejjohnson/gaussx/commit/45ffc4ea5d531409fcb6f6c5aae7b8cb8d909d6d))
* **recipes:** add SSM expectation params and Joseph-form covariance update ([45ffc4e](https://github.com/jejjohnson/gaussx/commit/45ffc4ea5d531409fcb6f6c5aae7b8cb8d909d6d))

## [0.0.6](https://github.com/jejjohnson/gaussx/compare/v0.0.5...v0.0.6) (2026-03-31)


### Features

* add phases 12r, 15, 16, 17, 18.2-18.4 ([#15](https://github.com/jejjohnson/gaussx/issues/15)) ([be8d3a5](https://github.com/jejjohnson/gaussx/commit/be8d3a5723f57f0d6f1b6529f983cb0114aefc97))

## [0.0.5](https://github.com/jejjohnson/gaussx/compare/v0.0.4...v0.0.5) (2026-03-31)


### Features

* add new operators, distributions, uncertainty propagation, and enriched docs ([#13](https://github.com/jejjohnson/gaussx/issues/13)) ([6aada55](https://github.com/jejjohnson/gaussx/commit/6aada55dfe3fb88e395718f9f52f07169421c6e8))
* add structured operators, uncertainty propagation, and enriched docs ([6aada55](https://github.com/jejjohnson/gaussx/commit/6aada55dfe3fb88e395718f9f52f07169421c6e8))

## [0.0.4](https://github.com/jejjohnson/gaussx/compare/v0.0.3...v0.0.4) (2026-03-30)


### Features

* add NumPyro-compatible MultivariateNormal distributions ([#11](https://github.com/jejjohnson/gaussx/issues/11)) ([98248d7](https://github.com/jejjohnson/gaussx/commit/98248d76b2a5a4204280082c7d990aea0d3b5135))
* add v0.2–v0.4 layers (strategies, sugar, expfam, recipes, matfree backends) ([#8](https://github.com/jejjohnson/gaussx/issues/8)) ([a07d878](https://github.com/jejjohnson/gaussx/commit/a07d87831f0b5213d4458eaa63e9e46502a4538d))
* **distributions:** add NumPyro-compatible MultivariateNormal distributions ([98248d7](https://github.com/jejjohnson/gaussx/commit/98248d76b2a5a4204280082c7d990aea0d3b5135))

## [0.0.3](https://github.com/jejjohnson/gaussx/compare/v0.0.2...v0.0.3) (2026-03-30)


### Features

* add _testing.py utilities and deduplicate test helpers ([#7](https://github.com/jejjohnson/gaussx/issues/7)) ([de3e839](https://github.com/jejjohnson/gaussx/commit/de3e839b413c82507fa986d0fa41465ce6ff1fa9))


### Bug Fixes

* point docs nav to .ipynb notebooks for rendered outputs ([c4f16d3](https://github.com/jejjohnson/gaussx/commit/c4f16d3548f257b844b1243b1dc31f07eaacb150))
* render .ipynb notebooks instead of .py in docs ([#5](https://github.com/jejjohnson/gaussx/issues/5)) ([066b28e](https://github.com/jejjohnson/gaussx/commit/066b28e4db11926d301669ac8eff59c0a58aff5a))
* stabilize sparse variational GP optimization ([36f7910](https://github.com/jejjohnson/gaussx/commit/36f791090a8d03364ffde5bd26caa18173a77746))

## [0.0.2](https://github.com/jejjohnson/gaussx/compare/v0.0.1...v0.0.2) (2026-03-30)


### Features

* add BlockDiag, Kronecker, and LowRankUpdate operators ([0ee2b64](https://github.com/jejjohnson/gaussx/commit/0ee2b6482707c27af8bd273e957ece3272c74f34))
* add DenseSolver and CGSolver strategies ([91ccaed](https://github.com/jejjohnson/gaussx/commit/91ccaed9187d3da7ac620751cf5db6f07f117cdb))
* add JAX ecosystem dependencies and update project metadata ([cae329c](https://github.com/jejjohnson/gaussx/commit/cae329c9bd17e9ea0fb36084fb70616875cf057e))
* add Layer 0 primitives with structural dispatch ([f91da41](https://github.com/jejjohnson/gaussx/commit/f91da415d3922505ca9e9d14fd3eb121c088de3f))
* add structural tags and query helpers ([2e6ebe3](https://github.com/jejjohnson/gaussx/commit/2e6ebe348dc99183e56931dbeca454dbd3daf100))
* gaussx v0.0.1 — structured linear algebra foundations ([0518464](https://github.com/jejjohnson/gaussx/commit/051846498733099fee8b2bda512c95f4aa76b0bb))
* rename mypackage to gaussx ([e70a1aa](https://github.com/jejjohnson/gaussx/commit/e70a1aad6bae28bc0cf908382a895fb2c67d7571))
* scaffold gaussx package and test directory layout ([da4705e](https://github.com/jejjohnson/gaussx/commit/da4705ebf39771894f20441562be3cf3aca4ff37))
* wire up gaussx public API exports ([d2b39ec](https://github.com/jejjohnson/gaussx/commit/d2b39ec9d129595f6cdcb1c56ab54b8276582dff))


### Bug Fixes

* address operator review findings ([d8a8b12](https://github.com/jejjohnson/gaussx/commit/d8a8b12640e91dd59e0307de348d34032e5b6e9c))

## Changelog

All notable changes to this project will be documented in this file.

See [Conventional Commits](https://www.conventionalcommits.org/) for commit guidelines.
