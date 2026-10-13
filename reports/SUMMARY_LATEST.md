# Stock Pulse: Automated Pipeline Audit & Governance Summary

**Generated Programmatically**: `scripts/print_summary.py`
**Results Manifest**: `reports/results_manifest.json`

## 1. File Count & Cryptographic Manifest Reconciliation

- **OOS Prediction Parquets on Disk**: `75`
- **Tracked Rows in `08_provenance.csv`**: `75`
- **Tracked Files in `results_manifest.json`**: `78` (75 parquets + results_long.csv + config.yaml + 08_provenance.csv)

| Prediction Model Category | Parquet Files on Disk | Tracked in Provenance | Generating Script |
| :--- | :---: | :---: | :--- |
| **Baselines** | 21 | Yes | `src/models/baselines.py` |
| **Linear** | 12 | Yes | `src/models/linear_models.py` |
| **Random Forest** | 12 | Yes | `src/models/forest_models.py` |
| **LightGBM** | 15 | Yes | `src/models/boosting_models.py` |
| **LSTM** | 3 | Yes | `src/models/deep_models.py` |
| **TabPFN** | 6 | Yes | `src/models/tabpfn_models.py` |
| **ens_rank4** | 3 | Yes | `scripts/update_ensemble_and_eval.py` |
| **ens_rank4_bestlgbm** | 3 | Yes | `scripts/update_ensemble_and_eval.py` |
| **Total Prediction Files** | **75** | **Yes** | All deterministic |

## 2. Core Model Comparison per Horizon (Dev & Holdout)

### Horizon: SHORT (21 Trading Days)

| Model | Dev Mean IC | Fold IC Mean | Fold t-stat | Folds IC > 0 | Top-5 Med. Excess | Net (Turnover) | Replaced Share | Holdout IC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `momentum_126` | +0.0356 | +0.0359 | +1.56 | 8/12 | +0.60% | -0.22% | 30.5% | +0.0381 |
| `ridge` | +0.0284 | +0.0284 | +1.62 | 8/12 | +0.45% | -0.79% | 50.4% | +0.1913 |
| `rf_reg` | +0.0069 | +0.0068 | +0.42 | 6/12 | +0.30% | -1.32% | 52.6% | +0.2068 |
| `lgbm_reg` | -0.0048 | -0.0044 | -0.28 | 6/12 | +0.20% | -1.31% | 50.8% | +0.1494 |
| `ens_rank4` | +0.0281 | +0.0282 | +1.68 | 9/12 | +0.39% | -1.20% | 50.4% | +0.2062 |
| `ens_rank4_bestlgbm` | +0.0315 | +0.0315 | +1.72 | 10/12 | +0.33% | -0.90% | 47.2% | +0.1966 |

### Horizon: MID (63 Trading Days)

| Model | Dev Mean IC | Fold IC Mean | Fold t-stat | Folds IC > 0 | Top-5 Med. Excess | Net (Turnover) | Replaced Share | Holdout IC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `momentum_126` | +0.0658 | +0.0662 | +1.74 | 9/12 | +1.60% | +0.30% | 46.2% | +0.1135 |
| `ridge` | +0.0144 | +0.0146 | +0.52 | 7/12 | +0.84% | -0.69% | 56.9% | +0.3304 |
| `rf_reg` | +0.0250 | +0.0252 | +1.16 | 9/12 | +1.15% | -2.01% | 58.2% | +0.2984 |
| `lgbm_reg` | +0.0179 | +0.0181 | +0.82 | 6/12 | +0.70% | -1.66% | 56.9% | +0.1834 |
| `ens_rank4` | +0.0429 | +0.0433 | +1.41 | 8/12 | +1.49% | -0.62% | 51.1% | +0.3176 |
| `ens_rank4_bestlgbm` | +0.0429 | +0.0433 | +1.41 | 8/12 | +1.49% | -0.62% | 51.1% | +0.3176 |

### Horizon: LONG (252 Trading Days)

| Model | Dev Mean IC | Fold IC Mean | Fold t-stat | Folds IC > 0 | Top-5 Med. Excess | Net (Turnover) | Replaced Share | Holdout IC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `momentum_126` | +0.0682 | +0.0686 | +1.32 | 7/10 | +4.83% | -6.43% | 75.6% | -0.0127 |
| `ridge` | +0.0694 | +0.0697 | +1.48 | 7/10 | +4.71% | -8.20% | 64.4% | +0.4780 |
| `rf_reg` | +0.0595 | +0.0594 | +2.22 | 7/10 | +4.66% | +5.47% | 51.1% | +0.3754 |
| `lgbm_reg` | +0.0451 | +0.0451 | +1.64 | 6/10 | +2.26% | -11.97% | 62.2% | +0.2929 |
| `ens_rank4` | +0.0957 | +0.0959 | +2.36 | 9/10 | +5.87% | +9.24% | 66.7% | +0.3591 |
| `ens_rank4_bestlgbm` | +0.1143 | +0.1145 | +2.79 | 9/10 | +6.76% | +5.88% | 68.9% | +0.3939 |

## 3. Auto Served Model Selection & Paired Comparison

| Horizon | ens_rank4 Fold t | momentum_126 Fold t | t-stat Diff | Auto Selection | Rationale |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SHORT** | +1.68 | +1.56 | +0.12 | **`ens_rank4`** | ens_rank4 t-stat (+1.68) beats momentum_126 (+1.56) by +0.12 > 0.10 |
| **MID** | +1.41 | +1.74 | -0.33 | **`momentum_126`** | momentum_126 t-stat (+1.74) beats ens_rank4 (+1.41) by +0.33 > 0.10 |
| **LONG** | +2.36 | +1.32 | +1.04 | **`ens_rank4`** | ens_rank4 t-stat (+2.36) beats momentum_126 (+1.32) by +1.04 > 0.10 |

## 4. Evidence Levels & Weights (Rule v3 Post-Hoc)

| Horizon | Served Model | Criterion 1 (Null Band) | Criterion 2 (Benchmark) | Criterion 3 (Neutralized IC) | Evidence Level | Evidence Points | Horizon Weight |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SHORT** | `ens_rank4` | Failed | Failed | Passed | **RED** | 1 | **25%** |
| **MID** | `momentum_126` | Passed | Failed | Passed | **AMBER** | 2 | **50%** |
| **LONG** | `ens_rank4` | Failed | Failed | Passed | **RED** | 1 | **25%** |

## 5. Operational Thresholds Registry

| Threshold Key | Value | Unit | Status / Derivation |
| :--- | :---: | :--- | :--- |
| `tier_cutoff_short` | `0.25` | fraction | convention - no evidence-based cutoff |
| `tier_cutoff_mid` | `0.3` | fraction | empirical |
| `tier_cutoff_long` | `0.3` | fraction | empirical |
| `risk_category_low_cutoff` | `21.76` | annualized_percent | empirical |
| `risk_category_high_cutoff` | `32.98` | annualized_percent | empirical |
| `liquidity_warning_cutoff` | `0.8889` | trade_day_share_63m | empirical |
| `overall_weight_short` | `0.25` | weight | derived_from_evidence |
| `overall_weight_mid` | `0.5` | weight | derived_from_evidence |
| `overall_weight_long` | `0.25` | weight | derived_from_evidence |

## 6. Liquidity Flag Share & Consistent Leaders (All 20 Constituents)

| Series ID | Total Dev Days | Warning Days | Warning Share (%) | Stretched Horizon? | CL Days Before | CL Days After |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `AEL.N` | 1,742 | 0 | 0.00% | No | 51 | 51 |
| `CFIN.N` | 2,337 | 61 | 2.61% | No | 187 | 185 |
| `COMB.N` | 2,111 | 1,136 | 53.81% | No | 41 | 37 |
| `DFCC.N` | 2,389 | 0 | 0.00% | No | 100 | 100 |
| `DIAL.N` | 2,401 | 0 | 0.00% | No | 55 | 55 |
| `HAYL.N` | 2,245 | 523 | 23.30% | No | 126 | 73 |
| `HHL.N` | 2,345 | 80 | 3.41% | No | 279 | 274 |
| `HNB.N` | 1,033 | 1,001 | 96.90% | **Yes** | 41 | 0 |
| `JKH.N` | 1,915 | 395 | 20.63% | No | 2 | 2 |
| `LFIN.N` | 2,163 | 368 | 17.01% | No | 360 | 319 |
| `LIOC.N` | 2,393 | 0 | 0.00% | No | 379 | 379 |
| `LOLC.N` | 2,352 | 53 | 2.25% | No | 458 | 457 |
| `MELS.N` | 573 | 6 | 1.05% | No | 2 | 2 |
| `NDB.N` | 2,390 | 0 | 0.00% | No | 36 | 36 |
| `NTB.N` | 2,110 | 185 | 8.77% | No | 96 | 90 |
| `PLC.N` | 1,825 | 0 | 0.00% | No | 66 | 66 |
| `SAMP.N` | 2,151 | 56 | 2.60% | No | 72 | 71 |
| `SEYB.X` | 1,929 | 1,416 | 73.41% | No | 45 | 14 |
| `SINS.N` | 2,134 | 965 | 45.22% | No | 134 | 61 |
| `SUN.N` | 1,639 | 890 | 54.30% | No | 249 | 107 |
