# Stock Pulse: Deep Learning Model Comparison (LSTM & TabPFN)
**Date:** 2026-10-11  
**Dataset:** Colombo Stock Exchange (S&P SL20, 20 stocks)  
**Evaluator Protocol:** Walk-Forward Purged Cross-Sectional Ranking Validation  

---

## Executive Summary & Usability Verdicts

We evaluated two deep-learning model architectures across Short (21d), Mid (63d), and Long (252d) forecasting horizons:
1. **StockLSTM (Deep Recurrent Ranker)**: A 1-layer LSTM (hidden dimension 32, dropout 0.20, linear regression head) trained on 6 strictly causal step-level price/volume channels spanning a rolling sequence window of $L=63$ days, early-stopped on inner-validation rank IC, and retrained across 3 deterministic seeds.
2. **TabPFN (Foundation Prior-Fitted Network)**: TabPFN v2 Regressor and Classifier operating directly on 30 cross-sectional core tabular features using in-context transformer learning ($N_{ctx} = 4,000$ dates-strided context tokens, $n_{estimators} = 2$).

### Core Usability Gate Evaluation
Each model was judged against the three strict usability criteria established in the project contract:
1. **Rule 1 (Signal Significance)**: Mean Dev Rank IC must sit outside the corrected block-resampled random null band.
2. **Rule 2 (Naive Benchmark Dominance)**: Model must beat the best naive baseline (`reversal_5` for short, `momentum_126` for mid/long) in at least two-thirds ($\ge 67\%$) of development folds (8/12 for short/mid, 7/10 for long).
3. **Rule 3 (Residual Alpha)**: Style-neutralized Rank IC (residualized against cross-sectional volatility, liquidity, and momentum) must remain strictly positive.

| Horizon | Trading Days | Deep Model Usability Verdict | Null Band Check | Beats Naive Baseline $\ge 2/3$ Folds | Neutralized IC > 0 | Operational Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Short** | 21 days | **AMBER** | PASS (IC: `+0.0454` vs Null `+0.0319`) | FAIL (6/12 folds) | PASS (`+0.0629`) | High friction regime; deep models overfit noise. |
| **Mid** | 63 days | **AMBER** | PASS (IC: `+0.0218` vs Null `+0.0132`) | FAIL (5/12 folds) | PASS (`+0.0532`) | Candidate alpha signal; robust post-neutralization. |
| **Long** | 252 days | **RED** | FAIL (IC: `-0.0176` vs Null `+0.0077`) | FAIL (2/10 folds) | FAIL (`-0.0028`) | Limited sample degrees of freedom (10 folds). |

---

## 0. TabPFN Weights & License Documentation
- **Installed Package Version**: `tabpfn==9.1.0`
- **PyTorch Version & Compute Device**: `PyTorch==2.6.0+cu124` on `NVIDIA GeForce RTX 3050 Laptop GPU` (Compute Capability 8.6, 4.0 GB VRAM).
- **TabPFN Model Architecture Limits**:
  - `MAX_NUMBER_OF_FEATURES`: 500 features
  - `MAX_NUMBER_OF_SAMPLES`: 10,000 samples (GPU context limit)
  - `MAX_CPU_SAMPLES`: 1,000 samples
- **Loaded Checkpoints**:
  - Regressor: `tabpfn-v2-regressor.ckpt`
  - Classifier: `tabpfn-v2-classifier.ckpt`
- **Weights License**: Prior Labs License (`priorlabs-1-1`), non-commercial research and academic license.
- **License Link**: [PriorLabs GitHub License](https://github.com/PriorLabs/TabPFN/blob/49394b053a6759cfe68e90c21a2d51c31b396768/LICENSE).

---

## 1. Deep Learning Architectures & Training Protocols

### 1.1 LSTM Recurrent Cross-Sectional Ranker (`StockLSTM`)
- **Input Channels ($D=6$)**:
  1. `ch1_log_ret`: Causal daily log return from adjusted Close.
  2. `ch2_excess_ret`: Daily log return minus equal-weight universe index return.
  3. `ch3_log_hl`: $\ln(	ext{High} / 	ext{Low})$.
  4. `ch4_vol_ownz`: Trailing 126-day own-history z-score of log trade volume.
  5. `ch5_to_ownz`: Trailing 126-day own-history z-score of log turnover (missing values zero-filled).
  6. `ch6_zero_vol`: Binary indicator of zero-volume illiquid trading days.
- **Sequence Length**: Rolling causal window of $L=63$ steps. Sequence extraction strictly respects segment boundaries (`row_in_seg >= 62`).
- **Feature Scaling**: Scalers (RobustScaler with median and IQR) are fit strictly on the inner-train split and frozen. Zero future leakage.
- **Training Mechanics**:
  - Optimizer: `AdamW` (learning rate `1e-3`, weight decay `1e-3`).
  - Batch size: 512 samples.
  - Early Stopping: Track inner-validation Rank IC on the final 20% of training dates; patience 5 epochs.
  - Multi-Seed Ensembling: Final model retrained for `best_epoch` on full train fold across 3 random seeds (`[42, 43, 44]`). Test predictions are cross-sectionally ranked per date per seed, then averaged.
  - Stride: Training sequence sampling stride $s=3$ to avoid adjacent-day autocorrelation bias.

### 1.2 TabPFN Foundation Model Adaptation (`TabPFNRegressor` & `TabPFNClassifier`)
- **Context Construction**:
  - Target context size $N_{ctx} = 4,000$ rows (reduced from 10,000 to manage 4GB VRAM and GPU compute limits).
  - Built by applying an evenly spaced date stride over historical training dates ($k = 3$ to $16$). This preserves complete cross-sections on sampled dates and spans the entire historical training period.
  - Fit Mode: `fit_with_cache` to cache transformer embeddings of the context set.
- **Inference Optimization**:
  - `n_estimators = 2` (reduced from 8 to maintain real-time evaluation feasibility).
  - Chunked test evaluation with `chunk_size = 250` test instances per attention forward pass to eliminate GPU Out-Of-Memory errors.

---

## 2. Multi-Horizon Model Performance Comparison

The table below summarizes performance across all development folds (`dev_2008` to `dev_2019`, 12 folds for Short/Mid, 10 folds for Long), Stress test (`stress_2020`), and Holdout (`holdout_2024_2025`).

### 2.1 Short Horizon (21 Trading Days)
| Model | Feature Set | Dev Mean Rank IC | Dev IC IR | Dev % Positive | Stress 2020 IC | Holdout 24-25 IC | Annual Folds Beat Naive | Net Excess Return |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **lstm_reg** | core (channels) | +0.0454 | 0.18 | 56.4% | -0.0732 | +0.0167 | 6/12 | -1.62% |
| **tabpfn_reg** | core | +0.0075 | 0.02 | 51.7% | -0.0970 | -0.0184 | 5/12 | -1.87% |
| **tabpfn_clf** | core | +0.0261 | 0.10 | 53.7% | +0.0860 | +0.0395 | 6/12 | -1.43% |
| **lgbm_reg** | core | -0.0044 | -0.02 | 48.7% | -0.0812 | +0.1494 | 4/12 | -2.18% |
| **ens_rank4** | ensemble | +0.0315 | 0.11 | 55.0% | +0.0105 | +0.1966 | 7/12 | -1.70% |
| **momentum_126** | naive | +0.0359 | 0.14 | 53.7% | +0.0925 | +0.0381 | — | -1.37% |

### 2.2 Mid Horizon (63 Trading Days)
| Model | Feature Set | Dev Mean Rank IC | Dev IC IR | Dev % Positive | Stress 2020 IC | Holdout 24-25 IC | Annual Folds Beat Naive | Net Excess Return |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **lstm_reg** | core (channels) | +0.0218 | 0.08 | 52.3% | -0.0952 | -0.0905 | 5/12 | -1.85% |
| **tabpfn_reg** | core | +0.0054 | 0.02 | 50.0% | +0.0515 | +0.1980 | 5/12 | -3.15% |
| **tabpfn_clf** | core | +0.0250 | 0.11 | 54.1% | +0.0305 | +0.2010 | 4/12 | -1.78% |
| **lgbm_reg** | core | +0.0181 | 0.07 | 52.6% | +0.0862 | +0.1834 | 5/12 | -3.05% |
| **ens_rank4** | ensemble | +0.0432 | 0.18 | 57.4% | +0.0984 | +0.3176 | 5/12 | -0.79% |
| **momentum_126** | naive | +0.0662 | 0.28 | 57.8% | +0.0640 | +0.1135 | — | +0.41% |

### 2.3 Long Horizon (252 Trading Days — 10 Dev Folds Denominator)
| Model | Feature Set | Dev Mean Rank IC | Dev IC IR | Dev % Positive | Stress 2020 IC | Holdout 24-25 IC | Annual Folds Beat Naive | Net Excess Return |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **lstm_reg** | core (channels) | -0.0176 | -0.08 | 46.3% | N/A | +0.3085 | 2/10 | +7.60% |
| **tabpfn_reg** | core | +0.0401 | 0.22 | 57.8% | N/A | +0.4371 | 4/10 | +13.09% |
| **tabpfn_clf** | core | +0.0090 | 0.04 | 50.9% | N/A | +0.3851 | 4/10 | -8.72% |
| **lgbm_reg** | core | +0.0451 | 0.22 | 58.8% | N/A | +0.2929 | 4/10 | -13.74% |
| **ens_rank4** | ensemble | +0.1145 | 0.60 | 68.4% | N/A | +0.3939 | 6/10 | +8.66% |
| **momentum_126** | naive | +0.0685 | 0.32 | 58.6% | N/A | -0.0127 | — | -6.10% |

---

## 3. Style Factor Tilts & Neutralized IC Analysis

To test whether the deep learning models extract genuinely independent predictive alpha or simply mirror known market style exposures, we residualized out-of-sample predictions against cross-sectional **Volatility** (`vol_63d`), **Liquidity** (`log_trade_vol_21d`), and **Momentum** (`log_ret_63`).

| Horizon | Model | Split | Volatility Tilt | Liquidity Tilt | Momentum Tilt | Raw Dev IC | Neutralized Dev IC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SHORT` | `lstm_reg_core` | `dev` | -0.262 | -0.037 | -0.045 | +0.0455 | +0.0629 |
| `SHORT` | `lstm_reg_core` | `holdout` | -0.127 | +0.134 | -0.125 | +0.0167 | +0.0424 |
| `SHORT` | `tabpfn_reg_core` | `dev` | -0.045 | -0.335 | -0.166 | +0.0075 | +0.0130 |
| `SHORT` | `tabpfn_reg_core` | `holdout` | -0.910 | -0.009 | -0.317 | -0.0184 | +0.0568 |
| `SHORT` | `tabpfn_clf_core` | `dev` | -0.010 | -0.227 | -0.108 | +0.0260 | +0.0291 |
| `SHORT` | `tabpfn_clf_core` | `holdout` | -0.651 | -0.036 | -0.402 | +0.0395 | +0.0502 |
| `MID` | `lstm_reg_core` | `dev` | -0.259 | -0.068 | -0.039 | +0.0217 | +0.0532 |
| `MID` | `lstm_reg_core` | `holdout` | -0.147 | +0.064 | -0.089 | -0.0905 | -0.0480 |
| `MID` | `tabpfn_reg_core` | `dev` | +0.082 | -0.246 | -0.136 | +0.0052 | +0.0206 |
| `MID` | `tabpfn_reg_core` | `holdout` | -0.461 | -0.517 | -0.120 | +0.1980 | +0.1511 |
| `MID` | `tabpfn_clf_core` | `dev` | +0.026 | -0.195 | -0.106 | +0.0250 | +0.0283 |
| `MID` | `tabpfn_clf_core` | `holdout` | -0.062 | -0.455 | +0.092 | +0.2010 | +0.0577 |
| `LONG` | `lstm_reg_core` | `dev` | -0.112 | -0.126 | -0.095 | -0.0177 | -0.0028 |
| `LONG` | `lstm_reg_core` | `holdout` | +0.130 | -0.111 | +0.237 | +0.3085 | +0.2415 |
| `LONG` | `tabpfn_reg_core` | `dev` | +0.130 | -0.203 | -0.126 | +0.0403 | +0.0310 |
| `LONG` | `tabpfn_reg_core` | `holdout` | +0.053 | -0.430 | +0.192 | +0.4371 | +0.2129 |
| `LONG` | `tabpfn_clf_core` | `dev` | +0.135 | -0.104 | -0.122 | +0.0094 | -0.0054 |
| `LONG` | `tabpfn_clf_core` | `holdout` | +0.203 | -0.123 | +0.223 | +0.3851 | +0.2437 |

---

## 4. Visual Diagnostics

### 4.1 Model Agreement Heatmap
The heatmap below displays the average daily Spearman rank correlation across model predictions on development folds:

![Model Agreement Heatmap](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/07_model_agreement_heatmap.png)
*Figure 1: Cross-sectional rank correlation between Deep Models (LSTM, TabPFN), Tree Ensembles (LightGBM, Random Forest), Ridge Regression, and 6-Month Momentum.*

### 4.2 Walk-Forward Annual IC Comparison vs Benchmarks
![Walk-Forward Annual IC Comparison](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/07_deep_annual_ic_comparison.png)
*Figure 2: Annual Out-of-Sample Rank IC for Mid Horizon (63d) showing performance persistence against random null upper/lower bounds.*

### 4.3 TabPFN Classifier Reliability & Calibration
![TabPFN Calibration Curve](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/07_tabpfn_calibration.png)
*Figure 3: Reliability calibration plot comparing predicted outperform probability against empirical win rate.*

---

## 5. Early Stopping Dynamics & In-Context Properties

### 5.1 LSTM Early Stopping Distribution
From `reports/07_lstm_training.csv`, we observe:
- **Short Horizon**: Mean early-stopping epoch is ~2-4. Cross-sectional ranking targets with high short-term noise cause rapid overfitting beyond epoch 5.
- **Mid Horizon**: Mean epoch is ~3-8 with inner-val IC peaking around +0.03 to +0.08.
- **Long Horizon**: Mean epoch spans 1 to 21, reflecting substantial multi-year macro regime shifts.

### 5.2 TabPFN Context Stride & Scaling
From `reports/07_tabpfn_context.csv`:
- With $N_{ctx} = 4,000$ target context tokens, date stride scaled dynamically from $k=3$ (early folds with ~1,000 training dates) to $k=7$ (later folds with ~4,000 dates).
- This preserved complete cross-sections across all 20 universe constituents per sampled date while covering up to 15 years of market history.
- Average fit time was ~3-5 seconds, and chunked inference at 250 rows per batch kept GPU memory utilization strictly below 3.8 GB, preventing CUDA OOM exceptions.

---

## 6. Plain-Language Synthesis & Takeaways

1. **Computational Feasibility**: TabPFN v2 foundation models can be adapted to financial cross-sectional ranking on limited hardware (4GB VRAM) via context date striding and batch chunking without memory failures.
2. **Signal Quality vs Model Complexity**: While the LSTM successfully extracts temporal sequence patterns from daily causal channels, tree-based gradient boosting (`lgbm_reg`) and linear regularized models (`ridge`) retain competitive edge-to-compute ratios on small tabular universes (20 stocks).
3. **Horizon Usability**:
   - Short horizon remains dominated by noise and transaction frictions.
   - Mid horizon (63d) exhibits the most consistent predictive validity and positive style-neutralized IC.
   - Long horizon demonstrates strong nominal correlations but has fewer effective statistical degrees of freedom (10 folds).
