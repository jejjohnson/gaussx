# Changelog

## [0.0.18](https://github.com/jejjohnson/gaussx/compare/v0.0.17...v0.0.18) (2026-06-10)


### ⚠ BREAKING CHANGES

* **core:** SumOperator, ScaledOperator, and ProductOperator are factory functions returning lineax-native operators, no longer classes; isinstance checks against them will break. is_diagonal of a product of diagonal operators now correctly reports True.

### Features

* **core:** consolidate on lineax 0.1.1 and matfree 0.6 with expanded structured dispatch ([#194](https://github.com/jejjohnson/gaussx/issues/194)) ([c56d40e](https://github.com/jejjohnson/gaussx/commit/c56d40e6fc0d5f92933b8041d54b6faf3865586c))
* **inference:** ensemble DA primitives — localization, inflation, ETKF ([#190](https://github.com/jejjohnson/gaussx/issues/190)) ([b1aa7bc](https://github.com/jejjohnson/gaussx/commit/b1aa7bc65af49f824f2ce882ddb4ce1d6aed9e2d))

## [0.0.17](https://github.com/jejjohnson/gaussx/compare/v0.0.16...v0.0.17) (2026-06-02)


### Features

* **solvers:** unified solver substrate — front door, preconditioners, capacitance, tridiagonal ([#188](https://github.com/jejjohnson/gaussx/issues/188)) ([e6633fb](https://github.com/jejjohnson/gaussx/commit/e6633fb5dbf18ec135217338436b51d9e525a754))

## [0.0.16](https://github.com/jejjohnson/gaussx/compare/v0.0.15...v0.0.16) (2026-05-29)


### Features

* **gp:** add Matheron-rule posterior sample updates ([#180](https://github.com/jejjohnson/gaussx/issues/180)) ([2369222](https://github.com/jejjohnson/gaussx/commit/23692225df394e0f22683cb0008035d24a4e4717))
* **kernels:** add EigenPro spectral preconditioning for kernel SGD ([#182](https://github.com/jejjohnson/gaussx/issues/182)) ([a66bab1](https://github.com/jejjohnson/gaussx/commit/a66bab1904d5cbb7dd5ef89cafcef9a9a3455896))
* **linalg:** add structured sandwich covariance transform ([#177](https://github.com/jejjohnson/gaussx/issues/177)) ([25392b4](https://github.com/jejjohnson/gaussx/commit/25392b4f76ffc79fe0367f962c875fa37a0d4592))
* **primitives:** add root and inverse-root decomposition primitives ([#181](https://github.com/jejjohnson/gaussx/issues/181)) ([c6809ce](https://github.com/jejjohnson/gaussx/commit/c6809ce2469ff4f2e72259520f93687e30ad14b9))
* **ssm:** add opt-in Woodbury innovation covariance ([#178](https://github.com/jejjohnson/gaussx/issues/178)) ([0b745df](https://github.com/jejjohnson/gaussx/commit/0b745dff31600795de7a97d595507ba9c57f0ba2))
* **ssm:** add square-root form for parallel Kalman filtering ([#179](https://github.com/jejjohnson/gaussx/issues/179)) ([3e7e1d4](https://github.com/jejjohnson/gaussx/commit/3e7e1d4153cd1fba540135348be2f7b4c436d595))

## [0.0.15](https://github.com/jejjohnson/gaussx/compare/v0.0.14...v0.0.15) (2026-05-12)


### Features

* **operators:** add FFT-based Toeplitz sampling via circulant embedding ([#172](https://github.com/jejjohnson/gaussx/issues/172)) ([8dc8376](https://github.com/jejjohnson/gaussx/commit/8dc837689fd9141b3d19bdccbc79a89ec5f58896))
* **operators:** add matrix-free Lanczos sampling for SumKronecker ([#174](https://github.com/jejjohnson/gaussx/issues/174)) ([b39deb3](https://github.com/jejjohnson/gaussx/commit/b39deb37f29a8e720dac061d2b1f970ea3c38927))
* **operators:** add structured KroneckerSum sqrt and sampling ([#173](https://github.com/jejjohnson/gaussx/issues/173)) ([3cdd9a0](https://github.com/jejjohnson/gaussx/commit/3cdd9a0118619b3a7c5727dc2557a78c6bd79159))
* **recipes:** add Bessel-corrected ensemble covariances and Kalman gain ([#175](https://github.com/jejjohnson/gaussx/issues/175)) ([e0b3218](https://github.com/jejjohnson/gaussx/commit/e0b32180c54d9452ac0648331caf44e6843eefa3))

## [0.0.14](https://github.com/jejjohnson/gaussx/compare/v0.0.13...v0.0.14) (2026-05-03)


### Bug Fixes

* **ssm:** real parallel Kalman filter / RTS smoother via associative_scan ([#166](https://github.com/jejjohnson/gaussx/issues/166)) ([990b7d5](https://github.com/jejjohnson/gaussx/commit/990b7d522ff7f34e2dd188fd6c75aba53bc89cc8))

## [0.0.13](https://github.com/jejjohnson/gaussx/compare/v0.0.12...v0.0.13) (2026-05-03)


### Features

* add batched input support to kernel operators ([#141](https://github.com/jejjohnson/gaussx/issues/141)) ([165a6f6](https://github.com/jejjohnson/gaussx/commit/165a6f6d9cd24e45f90677513aa8f5b05b488897))
* **ssm:** operator-typed Kalman family + time-varying generalisation ([#162](https://github.com/jejjohnson/gaussx/issues/162)) ([cdf898d](https://github.com/jejjohnson/gaussx/commit/cdf898d3c7252efbaa5c322ea897058a675a464d))

## [0.0.12](https://github.com/jejjohnson/gaussx/compare/v0.0.11...v0.0.12) (2026-05-02)


### Features

* route all linear algebra through dispatch infrastructure and expose solver parameter ([#154](https://github.com/jejjohnson/gaussx/issues/154)) ([400754a](https://github.com/jejjohnson/gaussx/commit/400754ac1639440ffea27d7b4239b95f5e617963))
* structural dispatch for eigh/submatrix/lyapunov; cleaner inverse + trace_product ([#158](https://github.com/jejjohnson/gaussx/issues/158)) ([98140fc](https://github.com/jejjohnson/gaussx/commit/98140fc52d292890a1e7648e4c37e5ea580059fe))

## [0.0.11](https://github.com/jejjohnson/gaussx/compare/v0.0.10...v0.0.11) (2026-05-02)


### Features

* promote pyrox primitives into gaussx ([#130](https://github.com/jejjohnson/gaussx/issues/130)) ([#131](https://github.com/jejjohnson/gaussx/issues/131)) ([dbb7ed5](https://github.com/jejjohnson/gaussx/commit/dbb7ed554c9a788ded4de782d1a13142cc4486ad))


### Bug Fixes

* address PR review comments ([f4a9371](https://github.com/jejjohnson/gaussx/commit/f4a937104cef3b46dd48fbe51f759504423d03d5))

## [0.0.10](https://github.com/jejjohnson/gaussx/compare/v0.0.9...v0.0.10) (2026-04-08)


### Features

* add 6 features — infinite-horizon KF, collapsed ELBO, OILMM, Psi stats, emission model, grid interpolation ([#96](https://github.com/jejjohnson/gaussx/issues/96)) ([b2b1e1f](https://github.com/jejjohnson/gaussx/commit/b2b1e1ff743bd479c6aa75b12a4ee725a6f1e7b0))
* add 6 standalone sugar/recipe features ([#27](https://github.com/jejjohnson/gaussx/issues/27), [#32](https://github.com/jejjohnson/gaussx/issues/32), [#33](https://github.com/jejjohnson/gaussx/issues/33), [#34](https://github.com/jejjohnson/gaussx/issues/34), [#35](https://github.com/jejjohnson/gaussx/issues/35), [#69](https://github.com/jejjohnson/gaussx/issues/69)) ([#93](https://github.com/jejjohnson/gaussx/issues/93)) ([cb40d2a](https://github.com/jejjohnson/gaussx/commit/cb40d2a2da343b53f8a78dd34158394bf3e8a61e))

## [0.0.9](https://github.com/jejjohnson/gaussx/compare/v0.0.8...v0.0.9) (2026-04-07)


### Features

* add 5 features — implicit op params, stable distances, batched matvec, gauss_kl, conditional ([#50](https://github.com/jejjohnson/gaussx/issues/50), [#51](https://github.com/jejjohnson/gaussx/issues/51), [#73](https://github.com/jejjohnson/gaussx/issues/73), [#74](https://github.com/jejjohnson/gaussx/issues/74), [#86](https://github.com/jejjohnson/gaussx/issues/86)) ([#89](https://github.com/jejjohnson/gaussx/issues/89)) ([92b37cc](https://github.com/jejjohnson/gaussx/commit/92b37ccfebc340d47a87163c041f4185d3827e71))
* add KernelOperator and ImplicitCrossKernelOperator ([#46](https://github.com/jejjohnson/gaussx/issues/46), [#48](https://github.com/jejjohnson/gaussx/issues/48)) ([#87](https://github.com/jejjohnson/gaussx/issues/87)) ([64b03b4](https://github.com/jejjohnson/gaussx/commit/64b03b419cd3ab7bf8c647006f4254eae6246791))
* add MINRESSolver and Gaussian 3-parameterization conversions ([#45](https://github.com/jejjohnson/gaussx/issues/45), [#75](https://github.com/jejjohnson/gaussx/issues/75)) ([#91](https://github.com/jejjohnson/gaussx/issues/91)) ([73a62b7](https://github.com/jejjohnson/gaussx/commit/73a62b7afa6a128c65077db04166ab491ea2ac80))

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
