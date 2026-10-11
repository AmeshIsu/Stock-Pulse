# Stock Pulse: Model 2 — Random Forest Benchmark & Regime Diagnostics

**Document**: `reports/05_model2_rf.md`  
**Purpose**: Rigorous walk-forward benchmark of non-linear tree ensembles (`RandomForestRegressor` and `RandomForestClassifier`), permutation importance analysis, factor tilt diagnostics, and realistic turnover-adjusted transaction frictions.

> [!WARNING]
> **Sample Limitations & Survivorship Caveat**:
> 1. **Index Survivorship Bias**: The S&P SL20 universe consists of companies that survived and prospered to be index members today. Backward evaluations—particularly over the long horizon—are inherently flattered by this survivorship selection.
> 2. **Macroeconomic Regime Shift**: The 2024–2025 holdout covers a single post-default recovery cycle characterized by aggressive disinflation and currency stabilization, comprising very few independent periods. It serves as a sanity check rather than conclusive proof of generalization.
> 3. **Statistical Sample Size**: With only ~10 independent years of development history on the Colombo Stock Exchange, long-horizon evidence is constrained by small sample degrees of freedom.

---

## 1. Usability Verdicts per Horizon

A model is classified as **usable** only if all three mandatory criteria are satisfied:
1. **Dev Rank IC is outside the random null band** (empirical permutation $p < 0.05$ vs. 1,000 single random scorers).
2. **Beats the best naive baseline (`momentum_126`) in at least two-thirds ($\ge 67\%$) of development folds**.
3. **Neutralized IC stays positive** when residualized against market volatility, liquidity, and momentum tilts.

### Evaluation of Criteria:
- **SHORT Horizon (21d)**: **AMBER** (1 of 3 criteria met)
  - *(1) Null Band*: **FAILED** (Dev IC `+0.007` lies inside the random null band `[-0.013, +0.014]`, $p = 0.073$).
  - *(2) Beats Best Baseline*: **FAILED** (Beats `momentum_126` in only **5 / 12** dev folds = 41.7%).
  - *(3) Neutralized IC*: **PASSED** (Residual IC `+0.012` > 0).
  - *Verdict*: Model is dominated by noise and transaction friction. Realized net excess return is negative after realistic trading costs.

- **MID Horizon (63d)**: **AMBER** (2 of 3 criteria met)
  - *(1) Null Band*: **PASSED** (Dev IC `+0.025` is outside the random null band `[-0.012, +0.013]`, $p < 0.001$).
  - *(2) Beats Best Baseline*: **FAILED** (Beats `momentum_126` in **6 / 12** dev folds = 50.0% < 66.7%). Best baseline `momentum_126` achieves higher mean Dev IC (`+0.066` vs `+0.025`).
  - *(3) Neutralized IC*: **PASSED** (Residual IC `+0.021` > 0).
  - *Verdict*: Statistically distinguishable from noise and retains positive skill after neutralization, but fails to consistently surpass simple 6-month intermediate momentum across economic regimes.

- **LONG Horizon (252d)**: **AMBER** (2 of 3 criteria met)
  - *(1) Null Band*: **PASSED** (Dev IC `+0.060` is outside the random null band `[-0.011, +0.012]`, $p < 0.001$).
  - *(2) Beats Best Baseline*: **FAILED** (Beats `momentum_126` in **6 / 10** dev folds = 60.0% < 66.7%). `momentum_126` achieves mean Dev IC `+0.068` vs RF `+0.060`.
  - *(3) Neutralized IC*: **PASSED** (Residual IC `+0.026` > 0).
  - *Verdict*: Shows positive rank skill, but fails the two-thirds consistency threshold against momentum. Furthermore, performance is subject to survivorship bias and small sample constraints.

> [!CAUTION]
> **No Horizon Achieves GREEN**: Across all three horizons, non-linear Random Forest models fail to beat naive 6-month momentum (`momentum_126`) in at least two-thirds of development years. The 2024–2025 holdout covers a single post-default recovery regime with few independent periods and serves as a sanity check only. Long-horizon findings remain severely constrained by index survivorship bias and only ~10 independent years of history.

---

## 2. Model Performance Comparison Tables

### Realistic Friction Assumption:
> [!IMPORTANT]
> **Transaction Cost Modeling**: Friction is evaluated two ways assuming **100 bps per side (200 bps round-trip)**:
> - **Full Turnover**: Conservative baseline assuming 100% of the portfolio is liquidated and replaced every rebalance.
> - **Actual Turnover**: Proportionate friction: `2 x 100 bps x (Share of Top-5 Replaced)`.

### Horizon: `SHORT` (21 Trading Days)

| Candidate Model | Split Type | Mean Rank IC | Outside Null Band? | Perm. p-val | Folds IC > 0 | Mean Fold IC | Top-5 Med. Excess | Excess vs Random | Net Excess (Full) | Net Excess (Actual) | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rf_reg_core` | **Dev (2008-19)** | +0.007 | No | 0.073 | 6 / 12 | +0.007 | +0.30% | -0.03% | -2.27% | -2.27% | 138 |
| `rf_reg_core` | *Stress (2020)* | +0.018 | No | 0.149 | 1 / 1 | +0.018 | +2.14% | +0.70% | -1.25% | -1.25% | 10 |
| `rf_reg_core` | *Holdout (2024-25)* | +0.207 | Yes | 0.000 | 1 / 1 | +0.207 | +2.75% | +2.10% | -0.00% | -0.00% | 16 |
| `rf_reg_full` | **Dev (2008-19)** | -0.001 | No | 0.557 | 5 / 12 | -0.001 | +0.15% | -0.17% | -2.28% | -2.28% | 138 |
| `rf_reg_full` | *Stress (2020)* | +0.036 | Yes | 0.022 | 1 / 1 | +0.036 | +1.70% | +0.26% | +0.54% | +0.54% | 10 |
| `rf_reg_full` | *Holdout (2024-25)* | +0.050 | Yes | 0.000 | 1 / 1 | +0.050 | +1.91% | +1.27% | -0.96% | -0.96% | 10 |
| `rf_clf_core` | **Dev (2008-19)** | +0.009 | No | 0.041 | 6 / 12 | +0.009 | +0.27% | -0.06% | -2.16% | -2.16% | 138 |
| `rf_clf_core` | *Stress (2020)* | -0.050 | Yes | 0.997 | 0 / 1 | -0.050 | +1.05% | -0.39% | -1.41% | -1.41% | 10 |
| `rf_clf_core` | *Holdout (2024-25)* | +0.160 | Yes | 0.000 | 1 / 1 | +0.160 | +2.20% | +1.56% | -0.28% | -0.28% | 16 |
| `rf_clf_full` | **Dev (2008-19)** | -0.005 | No | 0.810 | 6 / 12 | -0.005 | +0.08% | -0.25% | -2.29% | -2.29% | 138 |
| `rf_clf_full` | *Stress (2020)* | +0.017 | No | 0.165 | 1 / 1 | +0.017 | +1.69% | +0.25% | +1.23% | +1.23% | 10 |
| `rf_clf_full` | *Holdout (2024-25)* | +0.146 | Yes | 0.000 | 1 / 1 | +0.146 | +2.04% | +1.39% | +0.29% | +0.29% | 10 |
| `ridge_core` | **Dev (2008-19)** | +0.028 | Yes | 0.000 | 8 / 12 | +0.028 | +0.45% | +0.12% | -1.77% | -0.78% | 138 |
| `ridge_core` | *Stress (2020)* | -0.027 | No | 0.957 | 0 / 1 | -0.027 | +1.41% | -0.03% | -1.76% | -0.43% | 10 |
| `ridge_core` | *Holdout (2024-25)* | +0.191 | Yes | 0.000 | 1 / 1 | +0.191 | +2.06% | +1.42% | -0.72% | +0.24% | 16 |
| `logit_core` | **Dev (2008-19)** | +0.024 | Yes | 0.000 | 7 / 12 | +0.024 | +0.37% | +0.05% | -1.94% | -0.92% | 138 |
| `logit_core` | *Stress (2020)* | -0.032 | Yes | 0.978 | 0 / 1 | -0.032 | +1.29% | -0.15% | -2.49% | -1.25% | 10 |
| `logit_core` | *Holdout (2024-25)* | +0.180 | Yes | 0.000 | 1 / 1 | +0.180 | +2.18% | +1.54% | -0.52% | +0.39% | 16 |
| `momentum_126` | **Dev (2008-19)** | +0.036 | Yes | 0.000 | 8 / 12 | +0.036 | +0.60% | +0.28% | -1.60% | -1.60% | 138 |
| `momentum_126` | *Stress (2020)* | +0.092 | Yes | 0.000 | 1 / 1 | +0.092 | +3.49% | +2.06% | +1.38% | +1.38% | 10 |
| `momentum_126` | *Holdout (2024-25)* | +0.038 | Yes | 0.001 | 1 / 1 | +0.038 | +1.12% | +0.48% | -1.79% | -1.79% | 16 |
| `past_winner` | **Dev (2008-19)** | +0.010 | Yes | 0.025 | 5 / 12 | +0.010 | +0.27% | -0.06% | -1.66% | -1.66% | 138 |
| `past_winner` | *Stress (2020)* | -0.045 | Yes | 0.995 | 0 / 1 | -0.045 | +1.15% | -0.29% | -2.65% | -2.65% | 10 |
| `past_winner` | *Holdout (2024-25)* | +0.053 | Yes | 0.000 | 1 / 1 | +0.053 | +0.18% | -0.47% | -2.85% | -2.85% | 16 |

### Horizon: `MID` (63 Trading Days)

| Candidate Model | Split Type | Mean Rank IC | Outside Null Band? | Perm. p-val | Folds IC > 0 | Mean Fold IC | Top-5 Med. Excess | Excess vs Random | Net Excess (Full) | Net Excess (Actual) | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rf_reg_core` | **Dev (2008-19)** | +0.025 | Yes | 0.000 | 9 / 12 | +0.025 | +1.15% | +0.23% | -2.83% | -2.83% | 46 |
| `rf_reg_core` | *Stress (2020)* | +0.092 | Yes | 0.000 | 1 / 1 | +0.092 | +8.32% | +4.28% | -0.24% | -0.24% | 4 |
| `rf_reg_core` | *Holdout (2024-25)* | +0.298 | Yes | 0.000 | 1 / 1 | +0.298 | +7.13% | +6.38% | +5.10% | +5.10% | 5 |
| `rf_reg_full` | **Dev (2008-19)** | -0.003 | No | 0.761 | 5 / 12 | -0.003 | +1.04% | +0.13% | -2.50% | -2.50% | 46 |
| `rf_reg_full` | *Stress (2020)* | +0.095 | Yes | 0.000 | 1 / 1 | +0.095 | +12.71% | +8.67% | +3.97% | +3.97% | 4 |
| `rf_reg_full` | *Holdout (2024-25)* | +0.135 | Yes | 0.000 | 1 / 1 | +0.135 | +8.91% | +8.16% | -0.85% | -0.85% | 3 |
| `rf_clf_core` | **Dev (2008-19)** | -0.006 | No | 0.870 | 4 / 12 | -0.005 | +0.63% | -0.28% | -2.85% | -2.85% | 46 |
| `rf_clf_core` | *Stress (2020)* | -0.032 | No | 0.974 | 0 / 1 | -0.032 | +4.72% | +0.68% | +0.57% | +0.57% | 4 |
| `rf_clf_core` | *Holdout (2024-25)* | +0.273 | Yes | 0.000 | 1 / 1 | +0.273 | +7.36% | +6.61% | +3.99% | +3.99% | 5 |
| `rf_clf_full` | **Dev (2008-19)** | -0.011 | Yes | 0.991 | 4 / 12 | -0.011 | +0.76% | -0.16% | -2.40% | -2.40% | 46 |
| `rf_clf_full` | *Stress (2020)* | -0.041 | Yes | 0.994 | 0 / 1 | -0.041 | +3.97% | -0.08% | +4.84% | +4.84% | 4 |
| `rf_clf_full` | *Holdout (2024-25)* | +0.321 | Yes | 0.000 | 1 / 1 | +0.321 | +10.43% | +9.69% | +1.91% | +1.91% | 3 |
| `ridge_core` | **Dev (2008-19)** | +0.014 | Yes | 0.001 | 7 / 12 | +0.015 | +0.84% | -0.07% | -1.54% | -0.68% | 46 |
| `ridge_core` | *Stress (2020)* | +0.071 | Yes | 0.000 | 1 / 1 | +0.071 | +8.12% | +4.07% | +1.28% | +2.21% | 4 |
| `ridge_core` | *Holdout (2024-25)* | +0.330 | Yes | 0.000 | 1 / 1 | +0.330 | +5.26% | +4.51% | +2.44% | +3.54% | 5 |
| `logit_core` | **Dev (2008-19)** | +0.009 | No | 0.033 | 7 / 12 | +0.010 | +0.85% | -0.07% | -1.34% | -0.46% | 46 |
| `logit_core` | *Stress (2020)* | +0.038 | Yes | 0.014 | 1 / 1 | +0.038 | +6.97% | +2.92% | +1.56% | +2.63% | 4 |
| `logit_core` | *Holdout (2024-25)* | +0.268 | Yes | 0.000 | 1 / 1 | +0.268 | +3.93% | +3.18% | +3.64% | +4.94% | 5 |
| `momentum_126` | **Dev (2008-19)** | +0.066 | Yes | 0.000 | 9 / 12 | +0.066 | +1.60% | +0.69% | -0.75% | -0.75% | 46 |
| `momentum_126` | *Stress (2020)* | +0.064 | Yes | 0.000 | 1 / 1 | +0.064 | +11.68% | +7.63% | +2.46% | +2.46% | 4 |
| `momentum_126` | *Holdout (2024-25)* | +0.114 | Yes | 0.000 | 1 / 1 | +0.114 | +3.71% | +2.97% | -0.46% | -0.46% | 5 |
| `past_winner` | **Dev (2008-19)** | +0.023 | Yes | 0.000 | 6 / 12 | +0.023 | +0.93% | +0.02% | -0.41% | -0.41% | 46 |
| `past_winner` | *Stress (2020)* | -0.124 | Yes | 1.000 | 0 / 1 | -0.124 | -1.77% | -5.82% | -0.36% | -0.36% | 4 |
| `past_winner` | *Holdout (2024-25)* | +0.098 | Yes | 0.000 | 1 / 1 | +0.098 | -1.01% | -1.76% | -2.11% | -2.11% | 5 |

### Horizon: `LONG` (252 Trading Days)

| Candidate Model | Split Type | Mean Rank IC | Outside Null Band? | Perm. p-val | Folds IC > 0 | Mean Fold IC | Top-5 Med. Excess | Excess vs Random | Net Excess (Full) | Net Excess (Actual) | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rf_reg_core` | **Dev (2008-19)** | +0.060 | Yes | 0.000 | 7 / 10 | +0.059 | +4.66% | +2.19% | +4.59% | +4.59% | 10 |
| `rf_reg_core` | *Holdout (2024-25)* | +0.375 | Yes | 0.000 | 1 / 1 | +0.375 | +62.98% | +56.43% | +10.69% | +10.69% | 1 |
| `rf_reg_full` | **Dev (2008-19)** | +0.013 | Yes | 0.011 | 5 / 10 | +0.012 | +3.46% | +0.99% | +5.81% | +5.81% | 10 |
| `rf_reg_full` | *Holdout (2024-25)* | *Not Evaluable* | - | - | *0 / 0* | N/A | N/A (<90% full) | N/A | N/A | N/A | 0 |
| `rf_clf_core` | **Dev (2008-19)** | +0.035 | Yes | 0.000 | 6 / 10 | +0.035 | +4.02% | +1.56% | -9.61% | -9.61% | 10 |
| `rf_clf_core` | *Holdout (2024-25)* | +0.274 | Yes | 0.000 | 1 / 1 | +0.274 | +54.84% | +48.29% | +4.46% | +4.46% | 1 |
| `rf_clf_full` | **Dev (2008-19)** | +0.014 | Yes | 0.005 | 6 / 10 | +0.014 | +4.30% | +1.83% | +4.63% | +4.63% | 10 |
| `rf_clf_full` | *Holdout (2024-25)* | *Not Evaluable* | - | - | *0 / 0* | N/A | N/A (<90% full) | N/A | N/A | N/A | 0 |
| `ridge_core` | **Dev (2008-19)** | +0.069 | Yes | 0.000 | 7 / 10 | +0.070 | +4.71% | +2.25% | -8.84% | -8.13% | 10 |
| `ridge_core` | *Holdout (2024-25)* | +0.478 | Yes | 0.000 | 1 / 1 | +0.478 | +63.64% | +57.09% | +29.53% | +29.53% | 1 |
| `logit_core` | **Dev (2008-19)** | +0.046 | Yes | 0.000 | 7 / 10 | +0.047 | +3.43% | +0.96% | -8.34% | -7.71% | 10 |
| `logit_core` | *Holdout (2024-25)* | +0.456 | Yes | 0.000 | 1 / 1 | +0.456 | +58.65% | +52.10% | +29.53% | +29.53% | 1 |
| `momentum_126` | **Dev (2008-19)** | +0.068 | Yes | 0.000 | 7 / 10 | +0.069 | +4.83% | +2.37% | -6.87% | -6.87% | 10 |
| `momentum_126` | *Holdout (2024-25)* | -0.013 | No | 0.698 | 0 / 1 | -0.013 | +27.52% | +20.97% | -3.29% | -3.29% | 1 |
| `past_winner` | **Dev (2008-19)** | -0.045 | Yes | 1.000 | 4 / 10 | -0.044 | +4.27% | +1.80% | +15.40% | +15.40% | 10 |
| `past_winner` | *Holdout (2024-25)* | +0.108 | Yes | 0.000 | 1 / 1 | +0.108 | -8.72% | -15.27% | -12.57% | -12.57% | 1 |

---

## 3. Permutation Feature Importance & Grouped Family Breakdown

Evaluated on development test folds across 5 random shuffling repeats.

### Horizon: `SHORT` Top 10 Permutation Importance Features

| Rank | Feature Name | Family | Mean Drop in Daily Rank IC | Share of Positive Folds |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `parkinson_vol_21d` | risk | +8.84 x 1e-3 | 75.0% |
| 2 | `close_to_sma50` | trend | +4.89 x 1e-3 | 66.7% |
| 3 | `log_ret_21` | momentum | +4.59 x 1e-3 | 83.3% |
| 4 | `log_turnover_21d` | liquidity/volume | +4.40 x 1e-3 | 66.7% |
| 5 | `max_drawdown_63d` | risk | +3.70 x 1e-3 | 66.7% |
| 6 | `close_to_sma20` | trend | +2.69 x 1e-3 | 91.7% |
| 7 | `log_ret_5` | momentum | +1.81 x 1e-3 | 83.3% |
| 8 | `log_ret_10` | momentum | +1.38 x 1e-3 | 83.3% |
| 9 | `rsi_14` | trend | +1.37 x 1e-3 | 75.0% |
| 10 | `log_ret_63` | momentum | +1.19 x 1e-3 | 58.3% |

**Grouped Feature Family Importance (`SHORT`)**:

| Feature Family | Total Drop in Rank IC | Average Drop per Feature | Number of Features |
| :--- | :---: | :---: | :---: |
| `liquidity/volume` | -1.49 x 1e-3 | -0.30 x 1e-3 | 5 |
| `market-relative` | -1.47 x 1e-3 | -0.37 x 1e-3 | 4 |
| `momentum` | +5.68 x 1e-3 | +0.95 x 1e-3 | 6 |
| `risk` | +0.79 x 1e-3 | +0.10 x 1e-3 | 8 |
| `trend` | +10.07 x 1e-3 | +1.44 x 1e-3 | 7 |

### Horizon: `MID` Top 10 Permutation Importance Features

| Rank | Feature Name | Family | Mean Drop in Daily Rank IC | Share of Positive Folds |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `log_turnover_21d` | liquidity/volume | +27.94 x 1e-3 | 66.7% |
| 2 | `cs_rank_mom_63d` | momentum | +12.74 x 1e-3 | 66.7% |
| 3 | `downside_vol_63d` | risk | +5.35 x 1e-3 | 58.3% |
| 4 | `log_ret_63` | momentum | +2.58 x 1e-3 | 58.3% |
| 5 | `max_drawdown_63d` | risk | +2.50 x 1e-3 | 66.7% |
| 6 | `dist_low_126d` | trend | +1.87 x 1e-3 | 50.0% |
| 7 | `rel_ret_63d` | market-relative | +1.56 x 1e-3 | 50.0% |
| 8 | `close_to_sma50` | trend | +0.93 x 1e-3 | 75.0% |
| 9 | `parkinson_vol_21d` | risk | +0.92 x 1e-3 | 58.3% |
| 10 | `amihud_illiq_63d` | liquidity/volume | +0.64 x 1e-3 | 58.3% |

**Grouped Feature Family Importance (`MID`)**:

| Feature Family | Total Drop in Rank IC | Average Drop per Feature | Number of Features |
| :--- | :---: | :---: | :---: |
| `liquidity/volume` | +25.22 x 1e-3 | +5.04 x 1e-3 | 5 |
| `market-relative` | +1.32 x 1e-3 | +0.33 x 1e-3 | 4 |
| `momentum` | +13.95 x 1e-3 | +2.33 x 1e-3 | 6 |
| `risk` | -4.87 x 1e-3 | -0.61 x 1e-3 | 8 |
| `trend` | +3.52 x 1e-3 | +0.50 x 1e-3 | 7 |

### Horizon: `LONG` Top 10 Permutation Importance Features

| Rank | Feature Name | Family | Mean Drop in Daily Rank IC | Share of Positive Folds |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `log_turnover_21d` | liquidity/volume | +64.54 x 1e-3 | 60.0% |
| 2 | `vol_ratio_21_126` | liquidity/volume | +10.21 x 1e-3 | 70.0% |
| 3 | `cs_rank_mom_63d` | momentum | +4.33 x 1e-3 | 80.0% |
| 4 | `cs_rank_vol_63d` | risk | +1.74 x 1e-3 | 70.0% |
| 5 | `dist_low_126d` | trend | +1.20 x 1e-3 | 30.0% |
| 6 | `cs_rank_vol_21d` | risk | +0.45 x 1e-3 | 70.0% |
| 7 | `rel_ret_63d` | market-relative | +0.23 x 1e-3 | 50.0% |
| 8 | `log_ret_10` | momentum | +0.09 x 1e-3 | 60.0% |
| 9 | `rsi_14` | trend | +0.05 x 1e-3 | 50.0% |
| 10 | `log_ret_5` | momentum | +0.01 x 1e-3 | 20.0% |

**Grouped Feature Family Importance (`LONG`)**:

| Feature Family | Total Drop in Rank IC | Average Drop per Feature | Number of Features |
| :--- | :---: | :---: | :---: |
| `liquidity/volume` | +63.49 x 1e-3 | +12.70 x 1e-3 | 5 |
| `market-relative` | -2.47 x 1e-3 | -0.62 x 1e-3 | 4 |
| `momentum` | +3.43 x 1e-3 | +0.57 x 1e-3 | 6 |
| `risk` | -20.34 x 1e-3 | -2.54 x 1e-3 | 8 |
| `trend` | -2.28 x 1e-3 | -0.33 x 1e-3 | 7 |

---

## 4. Factor Tilt & Neutralization Diagnostics

Does model skill survive when controlling for market volatility, liquidity, and momentum tilts?

### Horizon: `SHORT` Factor Tilt & Neutralized IC Table

| Model Tag | Split Type | Volatility Tilt (Spearman) | Liquidity Tilt (Spearman) | Momentum Tilt (Spearman) | Raw Mean IC | Neutralized Mean IC | Up Market Top-5 Excess | Down Market Top-5 Excess |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge_core` | **Dev** | -0.238 | -0.344 | +0.090 | +0.028 | +0.040 | -0.30% | +0.72% |
| `ridge_core` | *Holdout* | -0.111 | -0.464 | +0.148 | +0.191 | +0.122 | +1.64% | +0.50% |
| `logit_core` | **Dev** | -0.326 | -0.361 | +0.122 | +0.024 | +0.049 | -0.43% | +0.51% |
| `logit_core` | *Holdout* | -0.161 | -0.494 | +0.271 | +0.180 | +0.119 | +1.75% | +0.88% |
| `rf_reg_core` | **Dev** | +0.001 | -0.484 | +0.005 | +0.007 | +0.012 | -0.41% | -0.14% |
| `rf_reg_core` | *Holdout* | -0.021 | -0.557 | +0.101 | +0.207 | +0.083 | +2.44% | +1.03% |
| `rf_clf_core` | **Dev** | -0.092 | -0.460 | +0.069 | +0.009 | +0.013 | -0.12% | -0.19% |
| `rf_clf_core` | *Holdout* | -0.005 | -0.504 | +0.301 | +0.160 | +0.077 | +2.03% | +1.05% |
| `momentum_126_core` | **Dev** | +0.065 | +0.078 | +0.615 | +0.036 | +0.026 | +0.15% | +0.63% |
| `momentum_126_core` | *Holdout* | +0.355 | -0.136 | +0.603 | +0.038 | +0.078 | +0.94% | -1.40% |

### Horizon: `MID` Factor Tilt & Neutralized IC Table

| Model Tag | Split Type | Volatility Tilt (Spearman) | Liquidity Tilt (Spearman) | Momentum Tilt (Spearman) | Raw Mean IC | Neutralized Mean IC | Up Market Top-5 Excess | Down Market Top-5 Excess |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge_core` | **Dev** | -0.113 | -0.371 | +0.097 | +0.014 | +0.044 | +0.58% | +0.33% |
| `ridge_core` | *Holdout* | -0.070 | -0.563 | +0.271 | +0.330 | +0.184 | +4.89% | +2.67% |
| `logit_core` | **Dev** | -0.217 | -0.336 | +0.079 | +0.009 | +0.042 | +0.52% | +0.82% |
| `logit_core` | *Holdout* | -0.239 | -0.472 | +0.166 | +0.268 | +0.156 | +6.17% | +3.52% |
| `rf_reg_core` | **Dev** | +0.131 | -0.389 | +0.137 | +0.025 | +0.021 | -1.85% | +0.38% |
| `rf_reg_core` | *Holdout* | -0.200 | -0.558 | +0.128 | +0.298 | +0.202 | +6.63% | +8.96% |
| `rf_clf_core` | **Dev** | +0.033 | -0.404 | +0.080 | -0.006 | +0.007 | -1.29% | -0.34% |
| `rf_clf_core` | *Holdout* | -0.344 | -0.485 | +0.033 | +0.273 | +0.210 | +5.31% | +8.73% |
| `momentum_126_core` | **Dev** | +0.068 | +0.077 | +0.615 | +0.066 | +0.055 | +1.63% | +0.80% |
| `momentum_126_core` | *Holdout* | +0.302 | -0.147 | +0.591 | +0.114 | +0.100 | +2.60% | -2.69% |

### Horizon: `LONG` Factor Tilt & Neutralized IC Table

| Model Tag | Split Type | Volatility Tilt (Spearman) | Liquidity Tilt (Spearman) | Momentum Tilt (Spearman) | Raw Mean IC | Neutralized Mean IC | Up Market Top-5 Excess | Down Market Top-5 Excess |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge_core` | **Dev** | -0.008 | -0.332 | +0.113 | +0.069 | +0.089 | -14.46% | +4.58% |
| `ridge_core` | *Holdout* | +0.022 | -0.682 | +0.313 | +0.478 | +0.220 | +31.53% | N/A |
| `logit_core` | **Dev** | -0.205 | -0.114 | -0.009 | +0.046 | +0.081 | -11.33% | +1.16% |
| `logit_core` | *Holdout* | +0.042 | -0.638 | +0.386 | +0.456 | +0.173 | +31.53% | N/A |
| `rf_reg_core` | **Dev** | +0.211 | -0.414 | +0.001 | +0.060 | +0.026 | +5.91% | +7.61% |
| `rf_reg_core` | *Holdout* | -0.075 | -0.707 | +0.322 | +0.375 | -0.012 | +12.69% | N/A |
| `rf_clf_core` | **Dev** | +0.092 | -0.200 | -0.023 | +0.035 | -0.014 | -13.93% | +1.88% |
| `rf_clf_core` | *Holdout* | +0.166 | -0.249 | +0.374 | +0.274 | +0.051 | +6.46% | N/A |
| `momentum_126_core` | **Dev** | +0.083 | +0.084 | +0.634 | +0.068 | +0.011 | -8.90% | +1.17% |
| `momentum_126_core` | *Holdout* | +0.001 | -0.101 | +0.417 | -0.013 | -0.099 | -1.29% | N/A |

---

## 5. Probability Calibration & Decile Lift (Random Forest Classifier)

Calibration analysis of predicted probabilities on development out-of-sample test blocks:

### Horizon: `SHORT` Decile Calibration Table (`rf_clf_core`)

| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4,547 | 0.427 | 50.5% | 50.1% | 1.01x |
| 2 | 4,546 | 0.458 | 47.9% | 50.1% | 0.96x |
| 3 | 4,546 | 0.474 | 49.1% | 50.1% | 0.98x |
| 4 | 4,546 | 0.487 | 49.8% | 50.1% | 0.99x |
| 5 | 4,546 | 0.497 | 49.6% | 50.1% | 0.99x |
| 6 | 4,546 | 0.506 | 52.7% | 50.1% | 1.05x |
| 7 | 4,546 | 0.515 | 50.0% | 50.1% | 1.00x |
| 8 | 4,546 | 0.527 | 50.5% | 50.1% | 1.01x |
| 9 | 4,546 | 0.541 | 49.8% | 50.1% | 0.99x |
| 10 | 4,546 | 0.566 | 51.3% | 50.1% | 1.02x |

**Top Decile Lift**: **1.02x** over random base rate.

### Horizon: `MID` Decile Calibration Table (`rf_clf_core`)

| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4,537 | 0.391 | 47.5% | 50.1% | 0.95x |
| 2 | 4,536 | 0.443 | 53.2% | 50.1% | 1.06x |
| 3 | 4,536 | 0.466 | 52.6% | 50.1% | 1.05x |
| 4 | 4,537 | 0.483 | 51.6% | 50.1% | 1.03x |
| 5 | 4,536 | 0.497 | 48.7% | 50.1% | 0.97x |
| 6 | 4,536 | 0.508 | 51.7% | 50.1% | 1.03x |
| 7 | 4,537 | 0.519 | 50.3% | 50.1% | 1.00x |
| 8 | 4,536 | 0.532 | 52.0% | 50.1% | 1.04x |
| 9 | 4,536 | 0.550 | 50.4% | 50.1% | 1.00x |
| 10 | 4,537 | 0.585 | 43.5% | 50.1% | 0.87x |

**Top Decile Lift**: **0.87x** over random base rate.

### Horizon: `LONG` Decile Calibration Table (`rf_clf_core`)

| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 3,909 | 0.366 | 52.4% | 50.1% | 1.05x |
| 2 | 3,908 | 0.419 | 51.7% | 50.1% | 1.03x |
| 3 | 3,908 | 0.450 | 44.2% | 50.1% | 0.88x |
| 4 | 3,908 | 0.472 | 44.0% | 50.1% | 0.88x |
| 5 | 3,908 | 0.492 | 47.5% | 50.1% | 0.95x |
| 6 | 3,908 | 0.517 | 53.1% | 50.1% | 1.06x |
| 7 | 3,908 | 0.534 | 51.0% | 50.1% | 1.02x |
| 8 | 3,908 | 0.550 | 56.8% | 50.1% | 1.13x |
| 9 | 3,908 | 0.562 | 55.3% | 50.1% | 1.10x |
| 10 | 3,908 | 0.604 | 45.3% | 50.1% | 0.90x |

**Top Decile Lift**: **0.90x** over random base rate.

---

## 6. Diagnostic Visualizations

### Annual Rank IC Comparison
![Annual Rank IC Comparison](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/rf_mean_ic_by_year.png)

### Permutation Feature Importance
![Permutation Importance](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/rf_permutation_importance.png)

### Raw vs. Neutralized IC
![Raw vs. Neutralized IC](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/rf_neutralized_vs_raw_ic.png)

---
*Report generated by Stock Pulse Model 2 (Random Forest) Benchmarking Suite.*  
*Total execution runtime: 1553.8s (25.90 min).*
