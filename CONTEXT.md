# Stock Pulse: Project Context & Architectural Rules

## Project Goal
"Stock Pulse" is a research-grade stock-upside ranking system for the Colombo Stock Exchange (S&P SL20, 20 companies).
The objective is to rank stocks by their forward upward potential across three investment horizons:
- **Short Horizon**: 21 trading days (~1 month)
- **Mid Horizon**: 63 trading days (~1 quarter)
- **Long Horizon**: 252 trading days (~1 year)

All candidate models will be evaluated with institutional rigor (leak-free cross-validation, Sharpe/hit-rate analysis, SHAP interpretability) and served through a backend API and an interactive web application.

---

## Non-Negotiable Core Rules
1. **Zero Information Leakage**:
   - Never leak future information into the past.
   - All dataset splits (train / validation / test) must strictly be walk-forward, expanding-window, or rolling time-based splits. Random train/test splits are strictly prohibited.
   - Target generation, moving averages, and rolling statistics must strictly respect company borders and `segment_id` boundaries (no cross-gap leakage).
   - Training folds must be rigorously **purged** (removing training rows whose forward target window overlaps the test period) and **embargoed** around boundaries.
2. **Deterministic & Reproducible Runs**:
   - Fixed random seeds across all scripts and configurations (`random_seed: 42`).
   - Every transformation step must be deterministic and logged.
3. **Partition Roles & Evaluation Governance**:
   - **Development Folds (Test Years 2008–2019)**: Used exclusively for feature selection, model exploration, hyperparameter tuning, and threshold derivation.
   - **Stress-Test Fold (Test Year 2020)**: Used exclusively as an out-of-sample shock test (COVID-19 pandemic regime; short and mid horizons only). Never used for parameter tuning.
   - **Final Holdout Fold (Test Years 2024–2025)**: Strictly preserved as the final out-of-sample holdout. Evaluated and reported for transparency, but NEVER accessed during model selection, feature engineering, tuning, or threshold derivation. Exact valid target dates: 2024-01-02 to 2025-11-28 (short targets), 2025-09-26 (mid targets), 2024-12-10 (long targets).
   - **Universe & Ranking**: Ranking universe is exactly the 20 series defined in `config.yaml` (`universe_series`): AEL.N, CFIN.N, COMB.N, DFCC.N, DIAL.N, HAYL.N, HHL.N, HNB.N, JKH.N, LFIN.N, LIOC.N, LOLC.N, MELS.N, NDB.N, NTB.N, PLC.N, SAMP.N, SEYB.X, SINS.N, SUN.N. All ranking is cross-sectional per date.
   - **Survivorship Bias Warning**: The 20 stocks represent today's index constituents (S&P SL20). Historical long-horizon momentum and winner heuristics carry survivorship bias.
4. **Honest Uncertainty & Evaluation Mechanics**:
   - **Random Baseline**: Uses independent RNG stream per (draw, date), shuffles series order, and averages draws to guarantee unbiased expectation (dev IC within +/-0.01, precision 48-52%).
   - **Portfolio Simulations**: Non-overlapping rebalancing every h trading days. `gross_excess = mean top-k return - mean benchmark return`. `net_excess = gross_excess - 2 x transaction_cost_per_side_bps / 10000` (200 bps round-trip friction is an ASSUMPTION). Strict identity: `|net - gross| == round_trip_cost` for every period. Aggregated over all periods in a partition.
   - **Ties & Precision**: Percentile ranks computed with average rank for ties (`(rank - 0.5) / N`). Precision counts exact median ties (`rankpct == 0.5`) as 0.5 hit. Classification label `outperform` drops exact median ties from training.
   - **Fold-Level Stats & Skew Correction**: Every model reports annual fold-level ICs, folds with IC > 0, mean fold IC, and fold t-stat ("indicative: small n"). Moving-block bootstrap CIs require `effective_blocks >= 5` (else N/A). Skew correction reports median excess, 1%/99% winsorized mean, and excess minus random baseline.
5. **Transparent Threshold Governance**:
   - Every operational, filtering, or detection threshold is documented in `thresholds.json` with a plain-English explanation detailing its financial or empirical derivation.
   - Zero hardcoded magic numbers in code; all thresholds and paths must be loaded directly from `config.yaml` or `thresholds.json`.
6. **Machine-Readable Standard Outputs**:
   - Every experiment writes out-of-sample predictions to `reports/oos_predictions/{model}_{horizon}_{feature_set}.parquet`.
   - All evaluation metrics append to a common long-format table at `reports/results_long.csv` using the schema:
     `model, feature_set, horizon, fold, split_type, metric, value`.
7. **Strict Summary Automation Rule (Zero Hand-Typed Metrics)**:
   - No performance number, metric, t-statistic, threshold, or ticker list in any summary or report finish may be typed by hand.
   - All summaries must be generated programmatically by `scripts/print_summary.py` directly querying `results_long.csv`, the frozen OOS parquets, `thresholds.json`, and `data/app/` JSON artifacts, and written to `reports/SUMMARY_LATEST.md`.

---

## Directory Layout
```
Group Project_1/
├── app/
│   ├── backend/         # API service (FastAPI) serving stock rankings & reports
│   └── frontend/        # Interactive dashboard
├── config.yaml          # Central configuration file for all paths and parameters
├── CONTEXT.md           # Master project context and architectural rules
├── thresholds.json      # Threshold derivation registry
├── requirements.txt     # Python environment requirements
├── data/
│   ├── raw/             # Original raw source files (Excel, CSE raw files)
│   └── processed/       # Parquet clean prices, features_targets, latest_features
├── models/              # Serialized trained model artifacts
├── reports/             # Audit reports, markdown documentation, figures, metrics
│   ├── figures/         # Audit and performance charts (PNG/SVG)
│   └── oos_predictions/# Standardized out-of-sample predictions per model fold
├── src/
│   ├── data/            # Data loading, cleaning, split adjustment, segmentation
│   ├── features/        # Feature engineering (momentum, risk, trend, volume, macro)
│   ├── models/          # Walk-forward fold builder, model contract, baselines, metrics
│   └── scoring/         # Stock upside ranking engine and portfolio scoring
└── tests/               # Pytest suite validating integrity, segmentation, zero-leakage
```
