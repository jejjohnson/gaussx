# Changelog

## [0.6.5](https://github.com/jejjohnson/gaussx/compare/v0.6.4...v0.6.5) (2026-10-08)


### Features

* **api:** adopt and document naming conventions, renaming with deprecated aliases ([#617](https://github.com/jejjohnson/gaussx/issues/617)) ([e97145c](https://github.com/jejjohnson/gaussx/commit/e97145c85ac46f4954d11442d4e7dde66f1c88ad))
* **api:** export is_eigen_reducible, add a top-level __all__, check doc page placement ([#616](https://github.com/jejjohnson/gaussx/issues/616)) ([4242879](https://github.com/jejjohnson/gaussx/commit/4242879bd420d7a3aee129f98b617a0f349c2ae2))
* **distributions:** shared AbstractMultivariateNormal base for both MVN classes ([#613](https://github.com/jejjohnson/gaussx/issues/613)) ([37bc8c2](https://github.com/jejjohnson/gaussx/commit/37bc8c28f8a657254e50c0b1bf12bb77103e1499))
* **inference:** add dense_innovation to ensemble_kalman_gain ([#608](https://github.com/jejjohnson/gaussx/issues/608)) ([caf9c94](https://github.com/jejjohnson/gaussx/commit/caf9c9434fb7373d25bf8b0ddc6dd9c052bf7042))
* **inference:** document natural-parameter conventions and add blr convention="precision" ([#611](https://github.com/jejjohnson/gaussx/issues/611)) ([66cdb3c](https://github.com/jejjohnson/gaussx/commit/66cdb3c735dbf93356ae5eb9c45bd9b8e01d5330))
* **linalg:** add diag_inv(method="xdiag") (G16) ([#628](https://github.com/jejjohnson/gaussx/issues/628)) ([9fb9c1e](https://github.com/jejjohnson/gaussx/commit/9fb9c1e6c6818c40604d2e4439f3b8073181166d))
* **operators:** finish the gh-136 rename with SumOfKroneckersSqrt and sum_of_kroneckers_sample ([#610](https://github.com/jejjohnson/gaussx/issues/610)) ([2397c4b](https://github.com/jejjohnson/gaussx/commit/2397c4be2f93e4811e81e4955faf49437b783f1d))
* **operators:** orthonormal fast path for LowRankUpdate on a scaled identity ([#606](https://github.com/jejjohnson/gaussx/issues/606)) ([475f169](https://github.com/jejjohnson/gaussx/commit/475f1692efe00fef3b83b43ddffae090d3d222d3))
* **primitives:** add Hutch++ to trace (G17) ([#629](https://github.com/jejjohnson/gaussx/issues/629)) ([e2e8fec](https://github.com/jejjohnson/gaussx/commit/e2e8fec7a1ca8e18820de796336438ee7e492f7b))
* **primitives:** add mixture_quantile and a Chandrupatla root finder ([#627](https://github.com/jejjohnson/gaussx/issues/627)) ([82b7b8f](https://github.com/jejjohnson/gaussx/commit/82b7b8f97c45b58b942871611e74015675970d11))
* **randomized:** add column_id and cur (G17) ([#632](https://github.com/jejjohnson/gaussx/issues/632)) ([651c9b1](https://github.com/jejjohnson/gaussx/commit/651c9b1ccb9291f1dffd0ed1c3e832c72c3bd96a)), closes [#486](https://github.com/jejjohnson/gaussx/issues/486) [#511](https://github.com/jejjohnson/gaussx/issues/511)
* **ssm:** add a square-root parallel Kalman filter ([#630](https://github.com/jejjohnson/gaussx/issues/630)) ([4cb317d](https://github.com/jejjohnson/gaussx/commit/4cb317d19c4715b1911c3acccbb6c65127b0cf49))
* **ssm:** add SDE linearization and drift path-KL helpers ([#623](https://github.com/jejjohnson/gaussx/issues/623)) ([e41dc42](https://github.com/jejjohnson/gaussx/commit/e41dc42c1ae35903f25aee454f0622f81f79d453))
* **ssm:** differentiate dare through an optimistix implicit adjoint ([#624](https://github.com/jejjohnson/gaussx/issues/624)) ([d0a7db0](https://github.com/jejjohnson/gaussx/commit/d0a7db0e4430d04ae51b11fbba4adbfa32279d96))
* **strategies:** add NystromLogdet (G17) ([#631](https://github.com/jejjohnson/gaussx/issues/631)) ([52abe44](https://github.com/jejjohnson/gaussx/commit/52abe4482546175da711ac74f8d8ef5f6d1e0b6b))
* **strategies:** add SketchAndPrecondLSMR and sketch_and_solve (G15) ([#625](https://github.com/jejjohnson/gaussx/issues/625)) ([f91cbff](https://github.com/jejjohnson/gaussx/commit/f91cbff74bd51f86702a7b27169d45b984baff9e))


### Bug Fixes

* **deps:** make the dense pseudo-inverse references robust to numpy 2.5 ([#600](https://github.com/jejjohnson/gaussx/issues/600)) ([fc3f05e](https://github.com/jejjohnson/gaussx/commit/fc3f05ef7c2f2b8da9b7e48bb4d50eab041ea152))
* **inference:** keep obs_noise structured in etkf_transform and take one eigh ([#607](https://github.com/jejjohnson/gaussx/issues/607)) ([051e843](https://github.com/jejjohnson/gaussx/commit/051e8439408d822fb968108fafe71b90dd80f5bc))
* **inference:** small-fixes checklist from the v0.2.0 review ([#612](https://github.com/jejjohnson/gaussx/issues/612)) ([041fafe](https://github.com/jejjohnson/gaussx/commit/041fafea83c66d720489b1528bde03734a60086f))
* land the review fixes for [#639](https://github.com/jejjohnson/gaussx/issues/639), [#623](https://github.com/jejjohnson/gaussx/issues/623) and [#627](https://github.com/jejjohnson/gaussx/issues/627) (CG atol rescaling, Taylor KL, quantile edge cases) ([#646](https://github.com/jejjohnson/gaussx/issues/646)) ([16837bf](https://github.com/jejjohnson/gaussx/commit/16837bfa3f5eb0a88ea296a9c500cdfe41f9ea04))
* **operators:** inherit symmetric/PSD tags in InterpolatedOperator ([#601](https://github.com/jejjohnson/gaussx/issues/601)) ([9b3917a](https://github.com/jejjohnson/gaussx/commit/9b3917af1ca4f827c7121e28dc20e6333f194186))


### Performance Improvements

* **sketching:** trace hadamard_transform's butterfly once ([#622](https://github.com/jejjohnson/gaussx/issues/622)) ([735e63d](https://github.com/jejjohnson/gaussx/commit/735e63d11d7b202d34aa210817b9dffe20998eff)), closes [#620](https://github.com/jejjohnson/gaussx/issues/620)

## [0.6.4](https://github.com/jejjohnson/gaussx/compare/v0.6.3...v0.6.4) (2026-10-07)


### Deprecations

Back-filled in [#300](https://github.com/jejjohnson/gaussx/issues/300): the deprecations in effect as of 0.6.4. Every one is removed in **0.7.0** (see the deprecation policy in `docs/api/index.md`). The items promised for 0.5.0 missed that date and move to 0.7.0.

* **operators:** `SumKronecker` → `SumOfKroneckers` ([#136](https://github.com/jejjohnson/gaussx/issues/136)); `SVDLowRankUpdate` → `LowRankUpdate(..., orthonormal=True)` or `svd_low_rank_plus_diag`; `Circulant(symmetric=None)` value-inferred symmetry (the default becomes `False`)
* **linalg:** the legacy three-positional `conditional_variance(base_diag, A_X, S_u)` → `conditional_variance(K_XX_diag, K_XZ, A_X, S_u=S_u)`
* **expfam:** `to_expectation` → `to_mean_cov`; `to_natural` → `mean_cov_to_natural`
* **gp:** `base_conditional` → `sparse_conditional`; `collapsed_elbo(K_diag=)` → `K_xx_diag=`; the ignored `solver=` of `collapsed_elbo` and `gauss_kl`; `predict_variance(K_cross, K_test_diag, operator)` → `predict_variance(cache, K_cross, K_test_diag)`
* **strategies:** `BBMMSolver.cg_tolerance` / `cg_max_iter` / `lanczos_iter` and `LSMRSolver.maxiter` (and their constructor keywords) → `rtol` / `max_steps` / `lanczos_order` ([#405](https://github.com/jejjohnson/gaussx/issues/405)); an SLQ logdet of a non-symmetric-tagged operator (will raise)
* **ssm:** `InfiniteHorizonState` → `FilterState`; the old `infinite_horizon_smoother` argument order; `dare(P_init=)`; `parallel_kalman_filter(solver=)` without `woodbury_innovation=True`; `form="sqrt"` → `psd_project=True`; the stacked `Q[0] = P_0` layout (`naturals_to_ssm`, `ssm_to_naturals`, `udl_from_ssm_params`) and `naturals_to_ssm(solver=)`; the ignored `process_noise` argument of the RTS smoothers


### Features

* **expfam:** add to_mean_cov and deprecate the misnamed to_expectation/to_natural ([#545](https://github.com/jejjohnson/gaussx/issues/545)) ([51cf276](https://github.com/jejjohnson/gaussx/commit/51cf276d5608a00e3155082baa1288090d56ee76))
* **quadrature:** let ep_tilted_moments take any point-based integrator ([#575](https://github.com/jejjohnson/gaussx/issues/575)) ([bc4d10a](https://github.com/jejjohnson/gaussx/commit/bc4d10a2fc5ae2bffc4dd2830d27bfbc55d75068))
* **strategies:** thread a PRNG key to stochastic logdets and seed CGSolver/MINRESSolver ([#590](https://github.com/jejjohnson/gaussx/issues/590)) ([8710dfa](https://github.com/jejjohnson/gaussx/commit/8710dfa3e99854c147a2b2f30890ac99a3c98fd6))


### Bug Fixes

* **distributions:** let project take an untagged Cholesky factor, and sufficient_stats any batch rank ([#347](https://github.com/jejjohnson/gaussx/issues/347)) ([#547](https://github.com/jejjohnson/gaussx/issues/547)) ([9fc76af](https://github.com/jejjohnson/gaussx/commit/9fc76af2dd2ca6bc8ce406bdb87c1833a7ec5ac1))
* **frontend:** default linear_solve to AutoSolver and attach preconditioners through Composed/Auto ([#596](https://github.com/jejjohnson/gaussx/issues/596)) ([efd32cf](https://github.com/jejjohnson/gaussx/commit/efd32cfeddc496099db7d606d9f859b81bd39b88))
* **gp, quadrature:** small-fixes checklist from the v0.2.0 review ([#589](https://github.com/jejjohnson/gaussx/issues/589)) ([6b31bad](https://github.com/jejjohnson/gaussx/commit/6b31badd00ef8cb873feaa848781a4a4aadac27a))
* **gp:** cache the Cholesky factor in PredictionCache and take the cache first in predict_variance ([#582](https://github.com/jejjohnson/gaussx/issues/582)) ([071d748](https://github.com/jejjohnson/gaussx/commit/071d7487155eaf3fece0a3a061c2f69865d7cc91))
* **gp:** document conditional_interpolate as two-filter fusion and add rts_interpolate ([#566](https://github.com/jejjohnson/gaussx/issues/566)) ([662253d](https://github.com/jejjohnson/gaussx/commit/662253dc45de56c9d17e1847ee0ddd11eb52568a))
* **gp:** document love_cache's one-signed Lanczos bias and add a convergence diagnostic ([#564](https://github.com/jejjohnson/gaussx/issues/564)) ([d57c2d9](https://github.com/jejjohnson/gaussx/commit/d57c2d9515cfaf2673b8a9192897e7ce627dbab0))
* **linalg:** diag_inv takes structured inverses and safe_cholesky densifies only on retry ([#573](https://github.com/jejjohnson/gaussx/issues/573)) ([a79d0a4](https://github.com/jejjohnson/gaussx/commit/a79d0a4f92549bae8a6b6813f29d8142d3cc864f))
* **linalg:** solve_matrix factors once only for dense PSD operators ([#570](https://github.com/jejjohnson/gaussx/issues/570)) ([56cbba7](https://github.com/jejjohnson/gaussx/commit/56cbba7dbdd90f1264baeacae20c6da93fbffecf))
* **operators:** add Circulant(symmetric=) and deprecate value-inferred symmetry ([#567](https://github.com/jejjohnson/gaussx/issues/567)) ([681900d](https://github.com/jejjohnson/gaussx/commit/681900d84cc8ccb20f2a71f09233ceb3b3e46119))
* **primitives:** cholesky and sqrt unwrap scalar Mul/Div and keep structure ([#563](https://github.com/jejjohnson/gaussx/issues/563)) ([398757e](https://github.com/jejjohnson/gaussx/commit/398757e32c61399dcd160a12cdee070a42c42a9e))
* **primitives:** close small dispatch gaps in trace, tagged MaskedOperator solve and linear_solve negation ([#585](https://github.com/jejjohnson/gaussx/issues/585)) ([8b2953a](https://github.com/jejjohnson/gaussx/commit/8b2953a329b44b04ca216d0776158702d7d020fa)), closes [#391](https://github.com/jejjohnson/gaussx/issues/391)
* **primitives:** flag silent densification with DenseFallbackWarning at the caller ([#587](https://github.com/jejjohnson/gaussx/issues/587)) ([4a8b52a](https://github.com/jejjohnson/gaussx/commit/4a8b52a9bb9b231bccfa44c3775215c4785ee969)), closes [#406](https://github.com/jejjohnson/gaussx/issues/406)
* **primitives:** forward stochastic options through every wrapper in trace/diag/frobenius_norm ([#581](https://github.com/jejjohnson/gaussx/issues/581)) ([6ac0201](https://github.com/jejjohnson/gaussx/commit/6ac0201fb53f0bf79f5e074a575ce43fb7eb02f4)), closes [#320](https://github.com/jejjohnson/gaussx/issues/320)
* **primitives:** logdet(KroneckerSum) uses general eigvals for non-symmetric factors ([#560](https://github.com/jejjohnson/gaussx/issues/560)) ([a670b31](https://github.com/jejjohnson/gaussx/commit/a670b317abf14ebe5f1afb80f42b4e5324629125))
* **primitives:** one rank= rule for eig, eigvals and svd ([#584](https://github.com/jejjohnson/gaussx/issues/584)) ([b708680](https://github.com/jejjohnson/gaussx/commit/b70868015c047e772ed6489a3ac404f788368744)), closes [#383](https://github.com/jejjohnson/gaussx/issues/383)
* **primitives:** sqrt_matmul shifts KroneckerSum and BlockTriDiag structurally ([#579](https://github.com/jejjohnson/gaussx/issues/579)) ([6eb2310](https://github.com/jejjohnson/gaussx/commit/6eb2310de9e82feab8c70d808f610964a9b370fe)), closes [#337](https://github.com/jejjohnson/gaussx/issues/337)
* **primitives:** structured diag/trace for Toeplitz and FFT-diagonalised operators ([#574](https://github.com/jejjohnson/gaussx/issues/574)) ([2d10a88](https://github.com/jejjohnson/gaussx/commit/2d10a88120520b616800d4343e47aaceefb09d4a)), closes [#373](https://github.com/jejjohnson/gaussx/issues/373)
* **primitives:** submatrix keeps LowRankUpdate and lineax wrapper structure ([#576](https://github.com/jejjohnson/gaussx/issues/576)) ([269162d](https://github.com/jejjohnson/gaussx/commit/269162d144b5ba44ba4a59ed9de9bd6fc4bc1021))
* **quadrature:** warn about the unscented alpha=1e-3 default in float32 ([#572](https://github.com/jejjohnson/gaussx/issues/572)) ([e2340c7](https://github.com/jejjohnson/gaussx/commit/e2340c7653d9b85cb71c3d15db27351666107c79))
* **solve:** accept a gaussx strategy in gaussx.solve and a lineax solver in linear_solve ([#595](https://github.com/jejjohnson/gaussx/issues/595)) ([1bd1540](https://github.com/jejjohnson/gaussx/commit/1bd1540bb13de1109103e5d0951230e9541c289a))
* **strategies:** check SLQ symmetry, make Jacobi gradient- and sign-safe, reject rectangular LSMR logdets ([#597](https://github.com/jejjohnson/gaussx/issues/597)) ([b7caaa9](https://github.com/jejjohnson/gaussx/commit/b7caaa962cfd8cb0984b158fa6dc2b06673c01e7))
* **strategies:** keep the SLQ logdet gradient finite through a Lanczos breakdown ([#593](https://github.com/jejjohnson/gaussx/issues/593)) ([8eb9c8d](https://github.com/jejjohnson/gaussx/commit/8eb9c8dae44c5d34d117a0648ecfd3672c863836))
* **strategies:** resolve CG tolerances from the dtype and add throw= to the CG family ([#571](https://github.com/jejjohnson/gaussx/issues/571)) ([1e2ad62](https://github.com/jejjohnson/gaussx/commit/1e2ad62f382c650c19eb159cf2cd18df8ae79fc7))
* **strategies:** stop MINRES at convergence, differentiate it implicitly, and raise on exhaustion ([#577](https://github.com/jejjohnson/gaussx/issues/577)) ([52452f3](https://github.com/jejjohnson/gaussx/commit/52452f31fde33381a18c22b6548191fb04f1bb02))


### Performance Improvements

* **preconditioners:** read the diagonal without materialising, and take a column function in from_operator ([#594](https://github.com/jejjohnson/gaussx/issues/594)) ([2325beb](https://github.com/jejjohnson/gaussx/commit/2325beb3aa643859508452b6f25416fd12b14611))
* **quadrature:** compute kernel expectations without the dead (P, N², N²) covariance ([#578](https://github.com/jejjohnson/gaussx/issues/578)) ([5e4b643](https://github.com/jejjohnson/gaussx/commit/5e4b6438fe37cfcf53491fb5e7f09cb97dcd1c2a))

## [0.6.3](https://github.com/jejjohnson/gaussx/compare/v0.6.2...v0.6.3) (2026-10-05)


### Features

* **distributions:** register numpyro kl_divergence for the gaussx MVN classes ([#540](https://github.com/jejjohnson/gaussx/issues/540)) ([5682677](https://github.com/jejjohnson/gaussx/commit/56826778bfffd620d87bcc75623cfab40aa9a847))


### Bug Fixes

* **distributions:** draw sample noise in the parameter dtype ([#381](https://github.com/jejjohnson/gaussx/issues/381)) ([#528](https://github.com/jejjohnson/gaussx/issues/528)) ([852dabc](https://github.com/jejjohnson/gaussx/commit/852dabc9eed93213fc9c0794090e00bb83f10037))
* **distributions:** give batched MVN entropies the batch shape ([#319](https://github.com/jejjohnson/gaussx/issues/319)) ([#542](https://github.com/jejjohnson/gaussx/issues/542)) ([bc189c0](https://github.com/jejjohnson/gaussx/commit/bc189c0b0dd892d3786a26f5ea9b6f4466eb0709))
* **distributions:** make log(2 pi) a Python float so it follows the active dtype ([#369](https://github.com/jejjohnson/gaussx/issues/369)) ([#527](https://github.com/jejjohnson/gaussx/issues/527)) ([7db7647](https://github.com/jejjohnson/gaussx/commit/7db76476ab23dbe0e21242863c7bceb515b5d2f9))
* **distributions:** route MultivariateNormal.sample through sample_mvn and keep structured precisions structured ([#538](https://github.com/jejjohnson/gaussx/issues/538)) ([1e12d54](https://github.com/jejjohnson/gaussx/commit/1e12d545a8c405a9d58612c63aa11e8408a0fd08))
* **distributions:** validate only the observed entries in MaskedLGSSM, and allow round-off in the Lanczos bracket test ([#469](https://github.com/jejjohnson/gaussx/issues/469)) ([#526](https://github.com/jejjohnson/gaussx/issues/526)) ([6d37c2c](https://github.com/jejjohnson/gaussx/commit/6d37c2c249a0329f0c2cd1f8329a23b493246f12))
* **linalg:** centre stable_squared_distances and accumulate its norms in the wide dtype ([#414](https://github.com/jejjohnson/gaussx/issues/414)) ([#533](https://github.com/jejjohnson/gaussx/issues/533)) ([ebd69e6](https://github.com/jejjohnson/gaussx/commit/ebd69e6cceb66f50187ba2312ab1987abc873938))
* **operators:** derive dtypes from inputs at the remaining default-dtype sites ([#408](https://github.com/jejjohnson/gaussx/issues/408)) ([#532](https://github.com/jejjohnson/gaussx/issues/532)) ([835981e](https://github.com/jejjohnson/gaussx/commit/835981e4733cef207d326bfb7df9fe1397470a95))
* **primitives:** return non-finite from a singular dense solve instead of raising ([#537](https://github.com/jejjohnson/gaussx/issues/537)) ([c91bf01](https://github.com/jejjohnson/gaussx/commit/c91bf0136beded909ea585010d2a1cfcc3306295))
* **quadrature:** keep ep_tilted_moments in the cavity dtype and floor its variance relative to the cavity ([#400](https://github.com/jejjohnson/gaussx/issues/400)) ([#531](https://github.com/jejjohnson/gaussx/issues/531)) ([526b94f](https://github.com/jejjohnson/gaussx/commit/526b94f16f896143ebfda0ba9fbc53ba841e5c06))
* **ssm:** build SumSDE's L with block_diag so it keeps the component dtype ([#397](https://github.com/jejjohnson/gaussx/issues/397)) ([#530](https://github.com/jejjohnson/gaussx/issues/530)) ([3f4fd86](https://github.com/jejjohnson/gaussx/commit/3f4fd86759f8272037ea3b2429aa78d28fb30678))
* **ssm:** give infinite_horizon_filter's carry the model dtype ([#386](https://github.com/jejjohnson/gaussx/issues/386)) ([#529](https://github.com/jejjohnson/gaussx/issues/529)) ([b15fd35](https://github.com/jejjohnson/gaussx/commit/b15fd350a416c9cc27d24558fe561b5ebf631d45))


### Performance Improvements

* **distributions:** factor a dense PSD covariance once in log_prob, entropy and KL ([#543](https://github.com/jejjohnson/gaussx/issues/543)) ([4fa71ea](https://github.com/jejjohnson/gaussx/commit/4fa71ea67f07179432422b30be3341dbceb65653))

## [0.6.2](https://github.com/jejjohnson/gaussx/compare/v0.6.1...v0.6.2) (2026-10-03)


### Bug Fixes

* **sparse:** takahashi on a bucket of diagonal-only columns ([#521](https://github.com/jejjohnson/gaussx/issues/521)) ([2efa446](https://github.com/jejjohnson/gaussx/commit/2efa4469d56b21f6dda413c57eb1110be4bb09e3))

## [0.6.1](https://github.com/jejjohnson/gaussx/compare/v0.6.0...v0.6.1) (2026-10-03)


### Bug Fixes

* add helpful error message for numpyro extras ([#512](https://github.com/jejjohnson/gaussx/issues/512)) ([a2870e1](https://github.com/jejjohnson/gaussx/commit/a2870e11ce094b27a2c5b3b18a5ec338c762e3d2))
* **distributions:** add BYM2GMRF, the BYM2 pair with its exact density and draws ([#518](https://github.com/jejjohnson/gaussx/issues/518)) ([e9f3201](https://github.com/jejjohnson/gaussx/commit/e9f3201b7c1f8c315d5e2bdf866d52188558bcad)), closes [#508](https://github.com/jejjohnson/gaussx/issues/508)
* **primitives:** inv_quad_logdet's default is exact for structured operators ([#517](https://github.com/jejjohnson/gaussx/issues/517)) ([2360f45](https://github.com/jejjohnson/gaussx/commit/2360f455c099253a496abee342c0eb4d4a733889)), closes [#340](https://github.com/jejjohnson/gaussx/issues/340)
* **strategies:** AutoSolver keeps every structured operator on its exact path ([#514](https://github.com/jejjohnson/gaussx/issues/514)) ([dfb0b89](https://github.com/jejjohnson/gaussx/commit/dfb0b89c6a9a940fa7b0014db89128ae464bb6cb)), closes [#321](https://github.com/jejjohnson/gaussx/issues/321)
* **strategies:** differentiate CG solves through data-dependent preconditioners ([#513](https://github.com/jejjohnson/gaussx/issues/513)) ([0da6c41](https://github.com/jejjohnson/gaussx/commit/0da6c416d690dafa790058be936e6eaa1f20d1b3))

## [0.6.0](https://github.com/jejjohnson/gaussx/compare/v0.5.0...v0.6.0) (2026-10-02)


### ⚠ BREAKING CHANGES

* **sparse:** cholesky(SparseOperator) now returns a SparseCholeskyFactor of the permuted matrix (solve, logdet, solve_lower_transpose, selected_inverse, diag_inv) instead of a dense lx.MatrixLinearOperator, and no longer raises above AutoSolver's size threshold.
* **randomized:** NystromPreconditioner.from_operator now requires a keyword-only shift (μ, e.g. the noise variance σ²) and its operator is the PSD part A (e.g. K), not the system A + μI; build it from K and pass σ² instead of building it from K + σ²I. Its fields are now basis, eigenvalues and shift (the old scale field is gone), and as_operator ignores its argument. For the old Rayleigh–Ritz projection use gaussx.randomized_eigh(op, rank, n_power_iter=0), which projects onto orth(AΩ) rather than orth(Ω) and so is more accurate, not identical.

### Features

* **distributions:** add GaussianMRF, IntrinsicGMRF and ConstrainedGMRF (G6) ([#505](https://github.com/jejjohnson/gaussx/issues/505)) ([99d39f5](https://github.com/jejjohnson/gaussx/commit/99d39f55ad45424d2aa57d917e4726841c40c68e))
* **gmrf:** add precision builders, SPDE on meshes and grids, fem_matrices, fem_projector and SpectralFunction (G7) ([#502](https://github.com/jejjohnson/gaussx/issues/502)) ([b888637](https://github.com/jejjohnson/gaussx/commit/b888637b8daca76e4d73947f82c72d888fd8d933))
* **inference:** add laplace_mode with implicit gradients, and binomial and negative-binomial likelihoods (G8) ([0c64f4e](https://github.com/jejjohnson/gaussx/commit/0c64f4ed3df36161818775467574fd49f5dd0830))
* **inference:** add vb_mean_correction, the low-rank variational-Bayes correction of the Laplace mean (G10) ([c65d82f](https://github.com/jejjohnson/gaussx/commit/c65d82f2f66f9c2ea5a10d4f9ded7ebed49e5f1f))
* **primitives:** add pseudo_logdet; theta_design's default Hessian works through sparse log-determinants (G5) ([#504](https://github.com/jejjohnson/gaussx/issues/504)) ([0018e75](https://github.com/jejjohnson/gaussx/commit/0018e7509c1d243ad4a9086ad1c43e517ff9bcfe))
* **randomized:** add randomized_nystrom and rebuild NystromPreconditioner on it with an explicit shift (G13) ([#501](https://github.com/jejjohnson/gaussx/issues/501)) ([46b248b](https://github.com/jejjohnson/gaussx/commit/46b248be96de942dd54983d7053e6b5b48c015b0))
* **randomized:** add range_finder, qb, randomized_svd and randomized_eigh with method="randomized" on svd/eig (G12) ([#500](https://github.com/jejjohnson/gaussx/issues/500)) ([a7a5ab2](https://github.com/jejjohnson/gaussx/commit/a7a5ab2b7d65c7a5468161f1882917c3e1f7bc03))
* **sparse:** add sparse cholesky with cached symbolic analysis, takahashi selected inverse and exact vjps (G4) ([a15380c](https://github.com/jejjohnson/gaussx/commit/a15380c7258d025fe94307bff2a9cc1050cf71b4))


### Bug Fixes

* **gmrf:** generalized_variance_scale takes diag(S⁻¹) and its correction from one sparse factor (G4) ([a15380c](https://github.com/jejjohnson/gaussx/commit/a15380c7258d025fe94307bff2a9cc1050cf71b4))
* **inference:** import gaussx without numpyro; laplace_mode sits behind the optional-numpyro guard (G8) ([0c64f4e](https://github.com/jejjohnson/gaussx/commit/0c64f4ed3df36161818775467574fd49f5dd0830))
* **inference:** vb_mean_correction sits behind the optional-numpyro guard too (G10) ([c65d82f](https://github.com/jejjohnson/gaussx/commit/c65d82f2f66f9c2ea5a10d4f9ded7ebed49e5f1f))
* **sparse:** second-order gradients through the sparse logdet and solve are correct (G4) ([a15380c](https://github.com/jejjohnson/gaussx/commit/a15380c7258d025fe94307bff2a9cc1050cf71b4))
* **sparse:** typecheck without the optional scikit-sparse installed (G4) ([a15380c](https://github.com/jejjohnson/gaussx/commit/a15380c7258d025fe94307bff2a9cc1050cf71b4))

## [0.5.0](https://github.com/jejjohnson/gaussx/compare/v0.4.1...v0.5.0) (2026-10-02)


### ⚠ BREAKING CHANGES

* **randomized:** PartialCholeskyPreconditioner(rank, shift).as_operator(op) now treats op as the system K + shift·I and factors op − shift·I (fixes #345). To factor a noiseless K directly, use PartialCholeskyPreconditioner.from_operator(K, rank, shift=σ²), which also builds the preconditioner once ([#371](https://github.com/jejjohnson/gaussx/issues/371)).

### Features

* **linalg:** structured selected inverses and shifted-Kronecker diag_inv; fix BlockTriDiag tags (G3) ([#489](https://github.com/jejjohnson/gaussx/issues/489)) ([6529589](https://github.com/jejjohnson/gaussx/commit/6529589b8360a433adf33489d4551912e19bfecb))
* **operators:** add SparseOperator with a static SparsityPattern (G1) ([#488](https://github.com/jejjohnson/gaussx/issues/488)) ([56799a0](https://github.com/jejjohnson/gaussx/commit/56799a0a0f5673f310d9fbc4a607c26c77b81aaa))
* **primitives:** add eigh_generalized with a schur-complement path for singular b (G2) ([#490](https://github.com/jejjohnson/gaussx/issues/490)) ([80db3c5](https://github.com/jejjohnson/gaussx/commit/80db3c5ebd7862fd9efe5f31a11099b59e8005fe)), closes [#471](https://github.com/jejjohnson/gaussx/issues/471)
* **quadrature:** add theta_design with eb, grid and ccd designs (G9) ([#492](https://github.com/jejjohnson/gaussx/issues/492)) ([f691dd4](https://github.com/jejjohnson/gaussx/commit/f691dd4649069a8a35ad2b149b5f82b066184bfc))
* **randomized:** add rp_cholesky and a build-once PartialCholeskyPreconditioner; stop counting noise twice (G14) ([14ea1f1](https://github.com/jejjohnson/gaussx/commit/14ea1f12ab3f01e38d59042fa6fed2df5e2fd9f5))
* **sketching:** add sketching operators and move hadamard_transform from kernellib (G11) ([#491](https://github.com/jejjohnson/gaussx/issues/491)) ([7275881](https://github.com/jejjohnson/gaussx/commit/72758814468d6f4b195c80ca13538ada79e241b1))

## [0.4.1](https://github.com/jejjohnson/gaussx/compare/v0.4.0...v0.4.1) (2026-10-02)


### Bug Fixes

* **release:** bump uv.lock's gaussx version in release PRs and check the lock in CI ([#458](https://github.com/jejjohnson/gaussx/issues/458)) ([aa8fdf7](https://github.com/jejjohnson/gaussx/commit/aa8fdf7d9edd6133cdcbf30c4dc51a64a4e9084b))

## [0.4.0](https://github.com/jejjohnson/gaussx/compare/v0.3.1...v0.4.0) (2026-09-29)


### ⚠ BREAKING CHANGES

* **periodic:** PeriodicSDE.state_dim is 2 * (n_harmonics + 1), was 2 * n_harmonics.

### Bug Fixes

* **dare:** solve by doubling so slow dynamics converge; refuse a non-converged steady state ([#294](https://github.com/jejjohnson/gaussx/issues/294)) ([#445](https://github.com/jejjohnson/gaussx/issues/445)) ([70b3f38](https://github.com/jejjohnson/gaussx/commit/70b3f3804cc4f1de3e48181c27c71c9e71892aa5))
* **kalman:** symmetrise the filter and smoother covariances ([#388](https://github.com/jejjohnson/gaussx/issues/388)) ([#447](https://github.com/jejjohnson/gaussx/issues/447)) ([b358060](https://github.com/jejjohnson/gaussx/commit/b35806098ed901569b2b64e9321fd390ea9c2e39))
* **periodic:** add the missing j = 0 harmonic so k(0) equals the variance ([#289](https://github.com/jejjohnson/gaussx/issues/289)) ([#444](https://github.com/jejjohnson/gaussx/issues/444)) ([e2aa904](https://github.com/jejjohnson/gaussx/commit/e2aa904624c1ba546075df85a30bb964465f03a8))
* **periodic:** exact scaled Bessel weights for any lengthscale ([#291](https://github.com/jejjohnson/gaussx/issues/291)) ([#443](https://github.com/jejjohnson/gaussx/issues/443)) ([7029e9a](https://github.com/jejjohnson/gaussx/commit/7029e9aeb635b9330e687c1b44f9964dd3da2866))


### Performance Improvements

* **composition:** SumSDE.discretise is the block diagonal of its components ([#318](https://github.com/jejjohnson/gaussx/issues/318)) ([#446](https://github.com/jejjohnson/gaussx/issues/446)) ([33f36ad](https://github.com/jejjohnson/gaussx/commit/33f36adb6b3252f5a2267217709add8c7c8095e1))
* **nonlinear_kalman:** validate only rules that can go indefinite; Cholesky solves ([#331](https://github.com/jejjohnson/gaussx/issues/331)) ([#451](https://github.com/jejjohnson/gaussx/issues/451)) ([418ce42](https://github.com/jejjohnson/gaussx/commit/418ce42be1579be6e21f83c0487e5ec61ae9d42f))
* **spingp,ssm_natural:** factor R once; N block factorisations in naturals_to_ssm ([#403](https://github.com/jejjohnson/gaussx/issues/403)) ([#449](https://github.com/jejjohnson/gaussx/issues/449)) ([f603059](https://github.com/jejjohnson/gaussx/commit/f60305994cb6a90dd857f28c30865fdc372835c5))

## [0.3.1](https://github.com/jejjohnson/gaussx/compare/v0.3.0...v0.3.1) (2026-09-29)


### Bug Fixes

* **base_conditional:** accept a 1-D f, validate shapes, clip diagonal variances at 0 ([#363](https://github.com/jejjohnson/gaussx/issues/363)) ([#438](https://github.com/jejjohnson/gaussx/issues/438)) ([dc8e1d2](https://github.com/jejjohnson/gaussx/commit/dc8e1d2303d83a6a0098fca9017a215722727f6f))
* **block_tridiag:** support a single block (N = 1) in mv, transpose, cholesky and solve ([#304](https://github.com/jejjohnson/gaussx/issues/304)) ([#429](https://github.com/jejjohnson/gaussx/issues/429)) ([9c4a5d8](https://github.com/jejjohnson/gaussx/commit/9c4a5d8f22dd51ec495fc58e96fff51ebd0cc710))
* **diagonalised:** complex transpose is Aᵀ, and only real outputs are auto-tagged symmetric ([#330](https://github.com/jejjohnson/gaussx/issues/330)) ([#433](https://github.com/jejjohnson/gaussx/issues/433)) ([9b03a08](https://github.com/jejjohnson/gaussx/commit/9b03a089873ae3b740767d1a829a31cc798ee906))
* **diagonalised:** reject a non-conjugate-even symbol for a real-output circulant ([#325](https://github.com/jejjohnson/gaussx/issues/325)) ([#432](https://github.com/jejjohnson/gaussx/issues/432)) ([0431898](https://github.com/jejjohnson/gaussx/commit/0431898dff0ea65136f194c9f939c34f288f98ff))
* **ensemble:** validate etkf_transform inputs like its siblings ([#341](https://github.com/jejjohnson/gaussx/issues/341)) ([#435](https://github.com/jejjohnson/gaussx/issues/435)) ([43b9560](https://github.com/jejjohnson/gaussx/commit/43b95600f49f70668898fcd5289d04d50e06cb47))
* **low_rank_update:** reciprocal-free Woodbury capacitance so zero weights stay exact ([#307](https://github.com/jejjohnson/gaussx/issues/307)) ([#430](https://github.com/jejjohnson/gaussx/issues/430)) ([0e00263](https://github.com/jejjohnson/gaussx/commit/0e00263995f3f8010fd2976ee07c7153019ba0f2))
* **markov_gaussian:** support a single step (zero transitions) in log_prob and the precision views ([#348](https://github.com/jejjohnson/gaussx/issues/348)) ([#436](https://github.com/jejjohnson/gaussx/issues/436)) ([a3d6b02](https://github.com/jejjohnson/gaussx/commit/a3d6b02bd14bb1bee4983b14eab8d41c216885f8))
* **natural_gradient:** gauss_newton_precision(J, base=prior) keeps the Woodbury structure ([#334](https://github.com/jejjohnson/gaussx/issues/334)) ([#434](https://github.com/jejjohnson/gaussx/issues/434)) ([b0a2bf0](https://github.com/jejjohnson/gaussx/commit/b0a2bf0a79785a989e80efd60038ac2be44a4091))
* **sum_kronecker:** never whiten by a non-positive diagonal anchor factor ([#317](https://github.com/jejjohnson/gaussx/issues/317)) ([#431](https://github.com/jejjohnson/gaussx/issues/431)) ([51bb045](https://github.com/jejjohnson/gaussx/commit/51bb045d17ac8932d7f3a31a17352353cca158ef))
* **tilted_moments:** pass a non-positive cavity variance through instead of returning NaN ([#357](https://github.com/jejjohnson/gaussx/issues/357)) ([#437](https://github.com/jejjohnson/gaussx/issues/437)) ([c732eac](https://github.com/jejjohnson/gaussx/commit/c732eac6ff16dd5b943fd4f34c794118ce85cecc))
* **toeplitz:** reject a complex column at construction ([#368](https://github.com/jejjohnson/gaussx/issues/368)) ([#439](https://github.com/jejjohnson/gaussx/issues/439)) ([75dd026](https://github.com/jejjohnson/gaussx/commit/75dd026f1c4f32b8577a1532d71fbcbcbc0d571c))

## [0.3.0](https://github.com/jejjohnson/gaussx/compare/v0.2.0...v0.3.0) (2026-09-29)


### ⚠ BREAKING CHANGES

* **low_rank_update:** low_rank_plus_diag(diag, U, ...) is no longer PSD-tagged from the sign of diag (pass psd=True), and value-equal but distinct factors (V = U.copy()) are no longer inferred symmetric (pass V=U or tags=lx.symmetric_tag).

### Bug Fixes

* **inv:** keep inv(LowRankUpdate) structured under jit; general Woodbury branch ([#328](https://github.com/jejjohnson/gaussx/issues/328)) ([#423](https://github.com/jejjohnson/gaussx/issues/423)) ([5570352](https://github.com/jejjohnson/gaussx/commit/557035214329ada38af125792c8cf15205eefb2d))
* **kronecker_sum:** finite gradients through solve/sqrt at repeated eigenvalues ([#295](https://github.com/jejjohnson/gaussx/issues/295)) ([#421](https://github.com/jejjohnson/gaussx/issues/421)) ([0fa19b3](https://github.com/jejjohnson/gaussx/commit/0fa19b32f38501cfd4c67278eaa3cbd5ab703659))
* **kronecker_sum:** make the KroneckerSumSqrt PSD guard jit-safe ([#420](https://github.com/jejjohnson/gaussx/issues/420)) ([19a1ba2](https://github.com/jejjohnson/gaussx/commit/19a1ba2d3eae33343e031b1fd405e0de53b32495)), closes [#292](https://github.com/jejjohnson/gaussx/issues/292)
* **low_rank_update:** infer tags from structure only, never array values ([#422](https://github.com/jejjohnson/gaussx/issues/422)) ([d7d891e](https://github.com/jejjohnson/gaussx/commit/d7d891e5ec7e06b04475b67086ca6cc10a4e9006)), closes [#343](https://github.com/jejjohnson/gaussx/issues/343)
* **masked:** differentiable, cache-stable capacitance solve; trace-safe transpose ([#290](https://github.com/jejjohnson/gaussx/issues/290)) ([#427](https://github.com/jejjohnson/gaussx/issues/427)) ([ca6fc00](https://github.com/jejjohnson/gaussx/commit/ca6fc007ef5ce12e94e5f0fc55fbf7589450b9a5))
* **ssm_natural:** check Q[0] == P_0 at run time so tracing cannot skip it ([#359](https://github.com/jejjohnson/gaussx/issues/359)) ([#426](https://github.com/jejjohnson/gaussx/issues/426)) ([4a328b2](https://github.com/jejjohnson/gaussx/commit/4a328b2161bd001513361cda694bddf79e31cb82))
* **strategies:** make strategy and preconditioner config static pytree data ([#419](https://github.com/jejjohnson/gaussx/issues/419)) ([bfc4bf4](https://github.com/jejjohnson/gaussx/commit/bfc4bf4728054a318936e20670109e046676d0c2)), closes [#301](https://github.com/jejjohnson/gaussx/issues/301)

## [0.2.0](https://github.com/jejjohnson/gaussx/compare/v0.1.0...v0.2.0) (2026-09-25)


### ⚠ BREAKING CHANGES

* the kernel operators, kernel approximations, kernel statistics, Falkon, EigenPro, batched kernel matvecs and stable_rbf_kernel are removed from gaussx. Import them from kernellib: `kernellib.<name>` for operators, Nystrom / RFF, Falkon and EigenPro; `kernellib.functional.<name>` for hsic, mmd_squared, center_kernel, centering_operator and stable_rbf_kernel.

### Features

* move the kernel layer to kernellib; add a low-rank trace_product path ([#272](https://github.com/jejjohnson/gaussx/issues/272)) ([53a0324](https://github.com/jejjohnson/gaussx/commit/53a0324d94666475544f2917fbf4c99968fd6bfa))

## [0.1.0](https://github.com/jejjohnson/gaussx/compare/v0.0.29...v0.1.0) (2026-09-24)


### ⚠ BREAKING CHANGES

* **operators:** the `green` and `capacitance_inv` attributes are removed (replaced by `capacitance_lu`). No downstream package reads them.

### Features

* **operators:** add DiagonalisedOperator and Circulant; diagonalised Kronecker-sum solves ([#268](https://github.com/jejjohnson/gaussx/issues/268)) ([a9b65e9](https://github.com/jejjohnson/gaussx/commit/a9b65e9eb0ccfe6272aaa5dbe9f44ca70e3f6c0a))
* **operators:** solve(MaskedOperator) via the capacitance method ([#269](https://github.com/jejjohnson/gaussx/issues/269)) ([d951677](https://github.com/jejjohnson/gaussx/commit/d95167779f2bf877dcfac4804caa18b058abca28))


### Bug Fixes

* **capacitance:** solve the right PDE for singular bases; drop the Green's table ([#267](https://github.com/jejjohnson/gaussx/issues/267)) ([bff418c](https://github.com/jejjohnson/gaussx/commit/bff418c6710c245697c9367d4bcb4d35e12b4d68))

## [0.0.29](https://github.com/jejjohnson/gaussx/compare/v0.0.28...v0.0.29) (2026-09-24)


### Features

* **kernels:** falkon_preconditioner for Nyström KRR (gh-49, 1/3) ([#253](https://github.com/jejjohnson/gaussx/issues/253)) ([9e02c71](https://github.com/jejjohnson/gaussx/commit/9e02c71cd1ba8e54399efad4acf99c63506a8891))
* **kernels:** falkon_solve, Falkon's preconditioned CG (gh-49, 2/3) ([#254](https://github.com/jejjohnson/gaussx/issues/254)) ([8926d82](https://github.com/jejjohnson/gaussx/commit/8926d821acd7f1bf229599a18a993bdaa4ace031))
* **linalg:** add EigenFactorization and kronecker_sum_solve; fix non-symmetric KroneckerSum solve ([#264](https://github.com/jejjohnson/gaussx/issues/264)) ([074e540](https://github.com/jejjohnson/gaussx/commit/074e54035e0d71531e4413b922dd7529bc0bf99a))


### Bug Fixes

* **preconditioners:** guard partial-Cholesky pivots past numerical rank (gh-237) ([#252](https://github.com/jejjohnson/gaussx/issues/252)) ([1a2b87b](https://github.com/jejjohnson/gaussx/commit/1a2b87b772ce50eb29c1925992a8fef9111950a3))

## [0.0.28](https://github.com/jejjohnson/gaussx/compare/v0.0.27...v0.0.28) (2026-09-24)


### Features

* **distributions:** structure-aware sample_mvn dispatch (gh-78) ([#249](https://github.com/jejjohnson/gaussx/issues/249)) ([ca4fedd](https://github.com/jejjohnson/gaussx/commit/ca4fedd83ecad9b09ec44e4505d9761946d97908))
* **primitives:** contour-integral matrix square roots and joint inv-quad/logdet (gh-39, gh-43) ([#244](https://github.com/jejjohnson/gaussx/issues/244)) ([b6f5003](https://github.com/jejjohnson/gaussx/commit/b6f5003161849870b27714cecbaefe2451d40c4a))

## [0.0.27](https://github.com/jejjohnson/gaussx/compare/v0.0.26...v0.0.27) (2026-09-23)


### Bug Fixes

* **operators:** exact gradients through the SumOfKroneckers solve; cavity precision_floor ([#246](https://github.com/jejjohnson/gaussx/issues/246)) ([77f3382](https://github.com/jejjohnson/gaussx/commit/77f3382e6a0c72cb63687558e333a58a4c495daa))

## [0.0.26](https://github.com/jejjohnson/gaussx/compare/v0.0.25...v0.0.26) (2026-08-28)


### Features

* **ssm,distributions:** add UDL factorisation and MarkovGaussian chain distribution (gh-65, gh-76) ([#242](https://github.com/jejjohnson/gaussx/issues/242)) ([58e90a6](https://github.com/jejjohnson/gaussx/commit/58e90a68aa240bed1e0117d88fcd5b2813dadcb9))

## [0.0.25](https://github.com/jejjohnson/gaussx/compare/v0.0.24...v0.0.25) (2026-08-26)


### Features

* **inference:** ensemble Kalman inversion step (gh-230) ([#239](https://github.com/jejjohnson/gaussx/issues/239)) ([5348f18](https://github.com/jejjohnson/gaussx/commit/5348f18b66199d2ef04f08501f8d76405579c1dc))
* **ssm:** mean-field block-diagonal Kalman filter and smoother (gh-29) ([#240](https://github.com/jejjohnson/gaussx/issues/240)) ([15d82f3](https://github.com/jejjohnson/gaussx/commit/15d82f37e63040f9bb784795e05f6faa4e77d588))

## [0.0.24](https://github.com/jejjohnson/gaussx/compare/v0.0.23...v0.0.24) (2026-08-26)


### Bug Fixes

* **root:** guard pivoted-Cholesky against rank-deficient inf columns (gh-236) ([#238](https://github.com/jejjohnson/gaussx/issues/238)) ([71ff499](https://github.com/jejjohnson/gaussx/commit/71ff49975d95bcd8578c8bc28e95f5fac770ec27))
* safe_cholesky reverse-mode AD + SDE kernel dtype promotion (gh-229, gh-224) ([#232](https://github.com/jejjohnson/gaussx/issues/232)) ([7dc00a1](https://github.com/jejjohnson/gaussx/commit/7dc00a1c80bc4105175ac033223e7cc3f2304fa2))

## [0.0.23](https://github.com/jejjohnson/gaussx/compare/v0.0.22...v0.0.23) (2026-08-25)


### Features

* **operators:** structural solve/logdet dispatch for SumOfKroneckers ([#228](https://github.com/jejjohnson/gaussx/issues/228)) ([b7855b6](https://github.com/jejjohnson/gaussx/commit/b7855b6a57e86192b6ddff85313736770809f8eb))

## [0.0.22](https://github.com/jejjohnson/gaussx/compare/v0.0.21...v0.0.22) (2026-08-21)


### Bug Fixes

* **ssm,quadrature:** preserve input dtype under x64 (gh-219) ([#221](https://github.com/jejjohnson/gaussx/issues/221)) ([b29f4f8](https://github.com/jejjohnson/gaussx/commit/b29f4f863e491bd061752c461c5b22f8e0f3631c))
* **ssm:** report a Lyapunov-consistent diffusion for ProductSDE (gh-219) ([#222](https://github.com/jejjohnson/gaussx/issues/222)) ([f38ffe7](https://github.com/jejjohnson/gaussx/commit/f38ffe7e05721f5a2feb31be83bf39ffc9e63df5))

## [0.0.21](https://github.com/jejjohnson/gaussx/compare/v0.0.20...v0.0.21) (2026-08-20)


### Features

* **ssm:** nonlinear Kalman filter/smoother and matrix-fraction discretisation ([#216](https://github.com/jejjohnson/gaussx/issues/216)) ([a28b86a](https://github.com/jejjohnson/gaussx/commit/a28b86a92df98bbcbe8a613781dd30355185e8c6))


### Bug Fixes

* **inference:** draw enkf_analysis perturbations with a PSD square root ([#217](https://github.com/jejjohnson/gaussx/issues/217)) ([2c27c35](https://github.com/jejjohnson/gaussx/commit/2c27c358de0466063970fd12cb4defe0f4d86ee9))

## [0.0.20](https://github.com/jejjohnson/gaussx/compare/v0.0.19...v0.0.20) (2026-08-19)


### Features

* **inference:** enkf_analysis — perturbed-observation ensemble Kalman update ([#214](https://github.com/jejjohnson/gaussx/issues/214)) ([2328fa1](https://github.com/jejjohnson/gaussx/commit/2328fa10834731ec131fdc3542057e4a7826aa06))
* **quadrature,inference:** cubature moment matching, SLR, and diagonal sites ([#212](https://github.com/jejjohnson/gaussx/issues/212)) ([3bc48cf](https://github.com/jejjohnson/gaussx/commit/3bc48cf29c50950a435ae99df96fec7b432c1217))

## [0.0.19](https://github.com/jejjohnson/gaussx/compare/v0.0.18...v0.0.19) (2026-08-18)


### Features

* **ssm,distributions:** per-channel observation mask and LGSSM densities ([#210](https://github.com/jejjohnson/gaussx/issues/210)) ([0dc3e3a](https://github.com/jejjohnson/gaussx/commit/0dc3e3abf692f68d4a10d5fd9c264113a70da79e))

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
