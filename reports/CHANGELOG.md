# Stock Pulse: Results Changelog

This changelog records all authorized modifications to the frozen research pipeline results, configuration, and out-of-sample prediction datasets. Any change to the files tracked in `reports/results_manifest.json` must be documented here with the date, author/agent, and precise reason.

---

## 2026-10-11: Baseline Results Freeze
- **Author**: Antigravity Research Agent
- **Reason**: Baseline results freeze following Part A metric reconciliation, independent fold-level IC audit, and strict denominator governance (12 folds short/mid, 10 folds long).
- **Files Covered**: 
  - `reports/results_long.csv`
  - `config.yaml`
  - 72 prediction files in `reports/oos_predictions/*.parquet`
- **Integrity**: Hash values verified and stored in `reports/results_manifest.json`.

---

## 2026-10-12: Uniform Ensemble Redefinition, Sensitivity Analysis, and Evidence Rule v3
- **Author**: Antigravity Research Agent
- **Date**: 2026-10-12
- **What**:
  - Redefined `ens_rank4` uniformly across Short, Mid, and Long horizons to use the identical 4 components: `[ridge_core, rf_reg_core, lgbm_reg_core, momentum_126]`.
  - Added `ens_components` to `config.yaml`.
  - Preserved the previous asymmetric variant as `ens_rank4_bestlgbm` (using `lgbm_clf_core` on Long) for sensitivity analysis only (`servable: false`).
  - Re-evaluated both models and updated `reports/results_long.csv`.
  - Created `reports/08_provenance.csv` tracking all 75 OOS prediction files with SHA-256 hashes, generators, seeds, and timestamps.
  - Implemented Evidence Rule v3 POST-HOC incorporating family-wise multiple testing control (Holm step-down across M=75 hypotheses) and adjusted default horizon weights to [Short: 0.25, Mid: 0.50, Long: 0.25].
  - Generated `reports/08_liquidity_flag_share.csv` and rendered updated user and admin methodologies with informational liquidity guidance.
  - Refreshed `reports/results_manifest.json` tracking 78 files (75 OOS parquets, results_long.csv, config.yaml, 08_provenance.csv).
- **Why**:
  - Strict alignment with pre-registration and methodological symmetry: removing post-hoc component cherry-picking (`lgbm_clf` on Long) from the production ensemble.
  - Quantifying researcher degrees of freedom: isolating the exact contribution of the classifier selection (+0.0186 IC, +0.43 t-stat).
  - Enforcing family-wise multiple testing error control (Holm p-value gating) to prevent false discoveries in client-facing evidence tiers.

---

## 2026-10-12: Integrity Gate Resolution, Production Models, Scoring Engine, and Refresh Pipeline
- **Author**: Antigravity Engineering Agent
- **Date**: 2026-10-12
- **What**:
  - **Part 0 Integrity Gate**:
    - Universe assertion: verified exact 20-series Colombo ranking universe (`AEL.N, CFIN.N, COMB.N, DFCC.N, DIAL.N, HAYL.N, HHL.N, HNB.N, JKH.N, LFIN.N, LIOC.N, LOLC.N, MELS.N, NDB.N, NTB.N, PLC.N, SAMP.N, SEYB.X, SINS.N, SUN.N`) and added `universe_series` to `config.yaml`. Corrected draft benchmark note referencing non-existent tickers (`ACL, AHUN, BRWN, CARG`).
    - File counts reconciled: verified 75 OOS parquets on disk, 75 rows in `08_provenance.csv`, and 78 tracked files in `results_manifest.json` (LightGBM is 15 files: 6 clf + 9 reg).
    - Liquidity and Consistent Leaders consistency: verified `trade_day_share_63m <= 0.8889` criterion; demonstrated HNB.N is flagged 96.9% of dev days (1,001/1,033) which explains its 0 post-filter days as Consistent Leader.
    - Number consistency: corrected `ens_rank4_bestlgbm` short horizon signal to use `lgbm_clf_core` (+0.0043 vs -0.0048 dev IC), restoring short dev IC to +0.0315 ($t=1.72$, 10/12 folds) matching the Model 3 report.
    - Holdout dates: documented exact target cutoff dates in `CONTEXT.md` (Short: 2024-01-02 to 2025-11-28; Mid: 2024-01-02 to 2025-09-26; Long: 2024-01-02 to 2024-12-10).
    - Summary automation: created `scripts/print_summary.py` outputting `reports/SUMMARY_LATEST.md` with Zero Hand-Typed Metrics enforcement.
  - **Part 1 Production Models**:
    - Created `config/feature_labels.yaml` mapping 30 core features to plain-English phrases.
    - Implemented `src/production/train.py`: derived dev hyperparameter consensus in `models/production/hyperparameters.json`, serialized 15 production models and preprocessors in `models/production/` with `manifest.json`. Implemented `reasons()` function returning top 2 positive and top 1 negative drivers with zero forbidden terms.
    - Verified parity test on `dev_2015` window (Ridge and Momentum exact 0.0 diff; RF <= 0.021; LightGBM <= 0.002).
  - **Part 2 Scoring Engine**:
    - Implemented `src/scoring/score_latest.py`: generated live scoring artifacts in `data/serving/{as_of}/` (`scores_{model_id}.parquet`, `stocks.json`, technical prices `prices/{code}.parquet`, `active_model_defaults.json`, and `CURRENT` pointer).
  - **Part 3 Data Refresh Pipeline**:
    - Implemented `src/pipeline/refresh.py` with CLI (`--input`, `--mode append|replace`, `[--retrain]`, `[--dry-run]`, `[--force]`), input schema validation, 60-day calendar gap policy, versioning under `data/versions/{timestamp}/`, atomic pointer updates, and rollback CLI in `src/pipeline/rollback.py`.
    - Executed 40-row round-trip reproducibility test: scores verified identical to full dataset run within 1e-9 tolerance.
  - **Tests**:
    - Created `tests/test_production.py` with 9 comprehensive tests covering all requirements. All 9 tests passed. Full repository suite passed (57/57 tests passing).
---

## 2026-10-12: Tiers Unification, Exact Parity, Driver Personalization, Volatility Scale Fix, and Serving Engine Completion
- **Author**: Antigravity Engineering Agent
- **Date**: 2026-10-12
- **What**:
  - **Section A: Tiers & Consistent Leaders Unification**:
    - Created shared `src/scoring/tiers.py` implementing unified $n_{\text{tier}} = \max(1, \text{round}(f \times N))$ rule with deterministic series ID tie-breaking and `is_consistent_leader` evaluation.
    - Updated `data/app/tiers.json` and `data/app/models/*/tiers.json` to empirical cutoff fractions `[0.25, 0.30, 0.30]`.
    - Recomputed `score_history.parquet` and `tier_history.json` across all 5 servable models using the unified function via `scripts/recompute_tier_history.py`.
    - Resolved tier misclassifications: SINS.N correctly binned as High (Short: 97.5, Mid: 92.5, Long: 77.5) and Consistent Leader = YES; PLC.N is High (Mid: 82.5, Long: 92.5) and Consistent Leader = YES; SEYB.X is High (Mid: 97.5, Long: 87.5) but Consistent Leader = NO due to active liquidity warning (0.8095 <= 0.8889).
  - **Section B: Confidence Labels Governance**:
    - Completely purged unauthorized label "Substantial" across all `data/` and `src/` files.
    - Enforced strict lookup of confidence labels exclusively from `data/app/models/{model_id}/evidence.json` at scoring time (Auto choices: Short = "Very low confidence", Mid = "Low confidence", Long = "Very low confidence").
  - **Section C: Stock-Specific Driver Explanations**:
    - Implemented signed feature contributions in `src/production/train.py` using Ridge linear contributions, 126d momentum percentile standing, TreeSHAP for RF and LightGBM, and z-scored rank averages for `ens_rank4`.
    - Excluded constant/uninformative features (`zero_trade_share_63d`, index-level series, zero-variance features).
    - Configured Low tier stocks to display top 2 negative drivers and top 1 positive driver (DFCC.N, HHL.N, JKH.N, SAMP.N, HNB.N).
    - Verified driver pair diversity across 20 constituents: maximum pair frequency is 5.0% (strictly <= 30% ceiling).
    - Extended forbidden terms test to all user-facing serving outputs and labels; zero forbidden terms detected.
  - **Section D: Volatility Scaling & Risk Category Bug Resolution**:
    - Diagnosed double-annualization bug: `vol_63d` in `features_targets.parquet` was already annualized in decimal form (`std * np.sqrt(252)`). Scoring had multiplied by `np.sqrt(252) * 100.0` again, inflating volatilities to > 400% and placing 19/20 stocks into High risk.
    - Corrected scaling to `vol_63d * 100.0`. Validated against dev quantiles (5%=13.33%, 25%=18.47%, 50%=25.26%, 75%=34.75%, 95%=56.66%): 11 Low, 8 Medium, 1 High (SUN.N at 38.64%).
    - Added user-facing note to `methodology_user.md` documenting elevated market volatility environment.
  - **Section E: Stale & Untraded Stocks Policy**:
    - Added `stale_max_market_days: 10` to `config.yaml`.
    - Implemented 4-status classifier: `ranked`, `stale`, `not_recently_traded`, `not_enough_history`.
    - Handled HNB.N as `stale` (last traded 2025-12-22, 6 market days <= 10), participating in cross section at last available features.
    - Ensured unrankable stocks (`not_recently_traded`, `not_enough_history`) receive `N/A` for scores, tiers, risk category, and Consistent Leader = False.
  - **Section F: Production Boosting Diagnostics & Exact Parity**:
    - Documented LightGBM dev iterations (5, 3, 1 trees). Demonstrated score distribution std <= 0.003 and near-constant Long predictions; recorded findings in `model_registry.json` and `methodology_admin.md`.
    - Achieved exact parity (< 1e-9) on `dev_2015` window for all models using fold-specific hyperparameters: Momentum diff = 0.0, Ridge diff = 0.0, RF diff = 2.22e-16, LightGBM diff = 0.0. Removed loose tolerances.
    - Reported consensus hyperparameter correlation vs fold-specific predictions (RF: 0.9664, LightGBM: 0.4959).
  - **Section G: Horizon Overlap Quantification**:
    - Computed Spearman correlations at as-of date (Short/Mid: 0.8451, Short/Long: 0.7504, Mid/Long: 0.8932) and median over trailing 250 market dates (Short/Mid: 0.8040, Short/Long: 0.6474, Mid/Long: 0.8367).
    - Added honest assessment sentence to `methodology_admin.md`.
  - **Section H: Evidence Criterion 2 for ens_rank4 Long**:
    - Audited beats vs benchmark (`momentum_126`): 6/10 dev folds (60.0% < 66.7% -> Failed).
    - Reconciled discrepancy: previous "8/10" text conflated Criterion 2 with top-tier positive lift consistency (8/10 folds positive lift, 80.0%).
  - **Section I: Serving Table Script**:
    - Created `scripts/print_serving_table.py` reading exclusively from `data/serving/CURRENT`.
- **Why**:
  - Resolve audit discrepancies, enforce exact numerical determinism and parity, and provide institutional transparency for the Stock Pulse serving platform.

---

## 2026-10-12: Serving Gate Resolution (Part 0) and Production Backend Architecture (Part 1)
- **Author**: Antigravity Engineering & Quantitative Research Agent
- **Date**: 2026-10-12
- **What**:
  - **Part 0: Serving Gate**:
    - 0.1 Auto Model per Horizon Actually Served: Configured `src/scoring/score_latest.py` and `scripts/print_serving_table.py` to resolve active models dynamically from `data/serving/active_model_defaults.json` (Short: `ens_rank4`, Mid: `momentum_126`, Long: `ens_rank4`). Served mid percentile correlation with `momentum_126` verified at 1.0. Confidence labels resolved from respective `evidence.json` (Short: Very low, Mid: Low confidence / AMBER, Long: Very low). Admin-only header displays resolved models.
    - 0.2 Production Configuration Walk-Forward: Executed walk-forward validation for production consensus hyperparameters across all folds for `ridge_core`, `rf_reg_core`, `lgbm_reg_core`, and built `ens_rank4_prodcfg` written to `reports/oos_predictions_prodcfg/` (leaving frozen `reports/oos_predictions/` untouched). Integrated `production_config_metrics` into `data/app/model_registry.json`. Documented in `data/app/methodology_admin.md` that production `ens_rank4` is not materially weaker than validated (all $|\Delta \text{IC}| < 0.02$). Reported that under production configuration, Short Auto model would change from `ens_rank4` to `momentum_126`, while Mid and Long remain unchanged.
    - 0.3 Tie Handling: Enforced average ranks (`rank(method='average')`) for all component rankings in `ens_rank4`. Validated series shuffle order invariance and verified Spearman correlation with series alphabetical order over trailing 250 dates is indistinguishable from 0 (mean -0.0019, median -0.0028, std 0.0463).
    - 0.4 Conditional Volatility Note: Added `market_volatility_note_rule` to `thresholds.json` triggering elevated volatility note only when cross-sectional median `vol_63d` > 66.7th percentile threshold (32.98%).
    - 0.5 Consumer Driver Wording: Refactored `config/feature_labels.yaml` with contextual `high_phrase` and `low_phrase` for all 30 features based on median standing and signed contribution. Added thin-trading disclosure sentence when thin trading supports rankings. Verified zero forbidden terms.
  - **Part 1: Backend Architecture (FastAPI)**:
    - Security & Authentication: Implemented Argon2id password hashing, rotating refresh token with family theft detection, httpOnly cookies, double-submit CSRF protection (`X-CSRF-Token`), account lockout after 10 failed logins, and SlowAPI rate limiting (5 req/min on login).
    - Database & Migrations: Structured SQLAlchemy 2 models and Alembic migrations (`users`, `refresh_tokens`, `watchlist_items`, `alerts`, `notifications`, `model_settings`, `audit_log`, `data_jobs`).
    - Serving Integration: Built cached serving repository reading `CURRENT`, dynamic model switching with audit logging and warning acknowledgements, composite scoring with `src/scoring/tiers.py`.
    - User & Admin Endpoints: Full OpenAPI 3.1 REST API under `/api/v1` with uniform `{error: {code, message, request_id}}` response format. User routes strictly quarantined with 0 forbidden model terms or metrics.
    - Alert Engine: Edge-triggered evaluation with cooldown and `PriceSource` abstraction.
---

## 2026-10-13: Part 0 Backend Governance Fixes and Part 1 Production Web Application
- **Author**: Antigravity Engineering & Quantitative Research Agent
- **Date**: 2026-10-13
- **What**:
  - **Part 0: Backend Governance Fixes**:
    - 0.1 Access Policy & RBAC Matrix:
      - Quarantined all routes to authenticated users except explicit public whitelist: `GET /health`, `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh`, `POST /auth/logout`, `GET /public/info`.
      - Anonymous visits to market summary, stocks, prices, rankings, leaders, movers, CSV export, and methodology return HTTP 401.
      - Updated OpenAPI-driven RBAC matrix test (`test_rbac_matrix.py`) to verify strict user vs admin separation.
    - 0.2 Unified Hit Rate Definition:
      - Implemented `src/scoring/hit_rate.py`: defined as share of High-tier stock-dates, using non-overlapping rebalance periods, beating the date's cross-sectional median forward return on development out-of-sample predictions.
      - Recomputed hit rates across all models and horizons:
        - `ens_rank4` old values: Short 57.94%, Mid 61.51%, Long 69.44%.
        - `ens_rank4` new unified values: Short 49.37%, Mid 51.13%, Long 60.19%.
      - Updated `model_registry.json`, all models' `tier_history.json`, and both user and admin methodologies.
      - Added automated test asserting exact consistency between registry and tier history.
    - 0.3 Production vs Validated Table Reconciliation:
      - Recomputed full comparison from frozen OOS parquets vs `reports/oos_predictions_prodcfg/`.
      - Validated `ens_rank4`: Short IC +0.0270 (t=1.25, 7/12 folds, HR 49.37%), Mid IC +0.0429 (t=1.41, 8/12 folds, HR 51.13%), Long IC +0.0396 (t=0.81, 7/10 folds, HR 60.19%).
      - Validated `ens_rank4_bestlgbm`: Short IC +0.0315 (t=1.72, 10/12 folds, HR 50.00%), Mid IC +0.0429 (t=1.41, 8/12 folds, HR 51.13%), Long IC +0.0582 (t=1.24, 8/10 folds, HR 64.08%).
      - Production `ens_rank4`: Short IC +0.0267 (t=1.34, 9/12 folds, HR 49.37%), Mid IC +0.0424 (t=1.48, 8/12 folds, HR 51.13%), Long IC +0.0360 (t=0.77, 6/10 folds, HR 60.19%).
      - Conclusion confirmed: production `ens_rank4` is not materially weaker than validated (all differences < 0.02: Short -0.0003, Mid -0.0005, Long -0.0036).
    - 0.4 Auto-Choice Pre-Registered Governance:
      - Applied pre-registered decision rule: higher dev fold-level t-statistic, with differences < 0.10 defaulting to `ens_rank4`.
      - On Short horizon: dev t-statistic `ens_rank4` = 1.25, `momentum_126` = 1.23; difference is 0.02 < 0.10 -> `ens_rank4` remains Auto choice.
      - Removed governance open question as Auto choice does not change.
      - Added unit test asserting no holdout columns are accessed during model selection.
    - 0.5 Cross-Sectional Volatility Note Input:
      - Recomputed cross-sectional median `vol_63d` across the 20 Colombo ranking universe constituents as of 2025-12-31 to be exactly 20.59% (approx 20.6%). Fixed column input and added unit test fixture.
    - 0.6 Stock Identifier Standardization:
      - Standardized canonical `code` series identifier format (e.g. `COMB.N`, `SEYB.X`). Corrected legacy smoke fixture (`COMB.N0000` -> `COMB.N`) and updated `docs/API.md`.
    - 0.7 New Endpoints & Admin IC Charts:
      - Added `GET /public/info` (app metadata, as-of date, public legal disclaimer).
      - Added `GET /admin/models/{model_id}/preview?horizon=...` showing top 5 stock codes and difference count against active model.
      - Populated per-fold annual IC values in `model_registry.json` for all models across Short, Mid, and Long horizons.
    - 0.8 Real-Server Smoke Verification & PostgreSQL:
      - Executed `scripts/smoke_backend.py` against live uvicorn subprocess and verified CSRF/cookies.
      - Connected to Docker PostgreSQL (container port 5432 mapped to host 5433), applied Alembic migrations, and verified all 9 smoke tests passing.
  - **Part 1: Web Application (`app/frontend`)**:
    - Tech Stack: React 19 + TypeScript + Vite + Tailwind CSS + TradingView Lightweight Charts + Recharts + React Router v7 + TanStack Query.
    - Look & Feel: Clean institutional UI inspired by Yahoo Finance, TradingView, Koyfin, Robinhood, and Stripe/Linear dashboards (documented in `docs/DESIGN.md`). Dark mode toggle, tabular numbers, accessible color tokens.
    - User Privacy & Security: Complete exclusion of model names and metrics from user interface and public bundles. Enforced via post-build automated scanner `scripts/check_bundle_leaks.mjs`.
    - Pages Implemented:
      - `/login` and `/register` with live password strength validation, show/hide toggle, rate-limiting notifications, and admin reset guidance.
      - `/` (Dashboard): Market summary tiles, High tier count, conditional volatility note, multi-horizon Top Picks with score gauges, confidence labels, Consistent Leaders strip, and Movers panel.
      - `/rankings`: Full 20-stock table, multi-column sorting, multi-horizon filters, interactive horizon weight sliders with re-normalization, CSV export, and 4-stock comparison drawer.
      - `/stocks/:code`: TradingView candlestick/line chart, SMA20/50/200, Bollinger bands, Volume, RSI, MACD, 3 horizon outlook cards with plain-language drivers, risk/liquidity thresholds, and stats.
      - `/watchlist`: CRUD table with inline note editing and quick-add combobox.
      - `/alerts`: Edge-triggered rules management, condition modal, cooldown controls.
      - `/notifications`: Popover bell and notification center.
      - `/how-it-works`: Plain-language methodology, threshold derivation table, print styling.
      - `/account`: User profile, password management, theme toggle, logout.
      - `/admin/*`: Admin Console with Overview, Models governance, User management, Data versioning, Audit log, and Admin Technical Methodology.
    - Quality & Verification:
      - UI Actions Inventory (`docs/UI_ACTIONS.md`): 128 interactive elements cataloged and 100% verified (`check_ui_actions_coverage.py`).
      - Playwright E2E Suite: All 6 test suites passed in 47.4s with 0 axe-core serious/critical accessibility violations.
      - Multi-Viewport Screenshots: 90 full-page screenshots captured at 375px, 768px, and 1440px in light and dark modes under `docs/screenshots/`.
      - Production Build: Successfully bundled with clean admin code-splitting and zero bundle leaks.




