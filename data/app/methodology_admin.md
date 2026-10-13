# Stock Pulse: Administrator & Technical Methodology Guide

**Document**: `data/app/methodology_admin.md` | **Generated**: 2026-10-13 | **Latest Data**: 2024-12-10

---

## 1. System Architecture & Model Serving Governance

Stock Pulse implements a walk-forward purged cross-sectional ranking framework over the Colombo Stock Exchange (S&P SL20).

### Default Served Models (Auto Rule):
- **Short Horizon (21d)**: `ens_rank4` (Equal-weight rank blend of Ridge, Random Forest, LightGBM, and 126d Momentum)
- **Mid Horizon (63d)**: `momentum_126` (Pure 6-month cross-sectional price momentum benchmark)
- **Long Horizon (252d)**: `ens_rank4` (Equal-weight rank blend of Ridge, Random Forest, LightGBM, and 126d Momentum)

### Ensemble Architecture (`ens_rank4`):
The live servable ensemble is pre-specified identically across ALL horizons as:
$$\text{ens\_rank4}_{i,t} = \frac{1}{4}\left(\text{rank}(\text{ridge\_core}_{i,t}) + \text{rank}(\text{rf\_reg\_core}_{i,t}) + \text{rank}(\text{lgbm\_reg\_core}_{i,t}) + \text{rank}(\text{momentum\_126}_{i,t})\right)$$
Where each component score is cross-sectionally ranked percentile-wise on date $t$ ($[0, 1]$). Component list is configured in `config.yaml` (`ens_components`).

### Sensitivity Analysis Variant (`ens_rank4_bestlgbm`):
A sensitivity variant selecting the highest development IC LightGBM variant per horizon (`lgbm_clf_core` on Long) is tracked as non-servable (`servable: false`). Choosing the model variant based on development results artificially inflates development IC (+0.1143 vs +0.0957, t = 2.79 vs 2.36) and is strictly quarantined from live serving.

### Servable Model Registry:
The following 5 models are configured as servable and can be selected manually in the administrative console:
1. `momentum_126` (Baseline)
2. `ridge_core` (L2 Linear)
3. `rf_reg_core` (Bagging Ensemble)
4. `lgbm_reg_core` (Gradient Boosted Trees)
5. `ens_rank4` (Multi-Family Rank Ensemble)

Research models (`logit_core`, `rf_clf_core`, `lgbm_clf_core`, `lstm_reg_core`, `tabpfn_reg_core`, `tabpfn_clf_core`, `ens_rank4_bestlgbm`) are retained in the registry with `servable: false` for governance auditing.

---

## 2. Statistical Validation Design & Multiple Testing

### Walk-Forward Folds
- **Short & Mid Horizons**: 12 development folds (`dev_2008` through `dev_2019`).
- **Long Horizon**: 10 development folds (`dev_2010` through `dev_2019`). Denominator is strictly 10; long horizon requires 2-year feature formation, so 2008-2009 were never evaluated.
- **Stress Partition**: 2020 COVID shock ($n_{\text{eff}}$ displayed).
- **Holdout Partition**: 2024–2025 out-of-sample evaluation ($n_{\text{eff}}$ displayed). Valid target date ranges: 2024-01-02 to 2025-11-28 (short), 2025-09-26 (mid), 2024-12-10 (long).

### Dual Random Null Band & Significance
- Null bands are constructed using 1,000 Monte Carlo block-constant and linear scorer simulations.
- **Short Null Band**: `[-0.0374, +0.0368]`
- **Mid Null Band**: `[-0.0532, +0.0558]`
- **Long Null Band**: `[-0.1252, +0.1231]`

### Holm-Bonferroni Correction
- Across the full multiple testing family ($M = 75$ evaluations, including sensitivity analysis and LightGBM variant selections), after step-down Holm correction, no model retains statistical significance at $\alpha = 0.05$ (all adjusted $p > 0.05$).
- All raw significant results are classified as **suggestive, not confirmed**.

---

## 3. Evidence-Level Rules (Rule v3 Post-Hoc)

### Evaluated Usability Criteria:
1. **Criterion 1**: Dev Mean Rank IC outside corrected null band ($p \le 0.05$).
2. **Criterion 2**: Beats best naive baseline in $\ge 2/3$ dev folds.
3. **Criterion 3**: Style-neutralized Rank IC $> 0.000$.

### Rule v3 Post-Hoc Definition:
- **RED**: Criterion 1 fails (inside corrected null band).
- **AMBER**: Criterion 1 passes AND at least one other criterion passes, OR all three criteria pass but Holm-adjusted $p \ge 0.05$.
- **GREEN**: All 3 criteria pass AND Holm-adjusted $p < 0.05$.

> [!NOTE]
> **Momentum Baseline Criterion 2 Caveat**: For `momentum_126`, Criterion 2 compares against `momentum_63`, which is almost the same price momentum signal (6-month vs 3-month momentum). Consequently, Criterion 2 represents an exceptionally weak hurdle for momentum strategies.

### Empirical Justification:
- Random linear scorer draws (null type b) produce positive style-neutralized IC in **47.5%** (Short), **54.0%** (Mid), and **52.5%** (Long) of simulations.
- Criterion 3 alone represents weak random baseline noise (~50% chance of passing) and cannot lift a candidate out of 'no reliable evidence'.

---

## 4. Liquidity & Thin-Trading Governance

- **`zero_trade_share_63d`**: Identified as uninformative (constant 0.0) due to trade-day filtering in cleaned parquet. Marked as constant; models not retrained.
- **Market Calendar**: Constructed requiring $\ge 10$ active ranking series.
- **`trade_day_share_63m`**: Evaluated as trailing 63 market-calendar days traded within segment.
- **Liquidity Warning Threshold**: Set to 20th percentile on development rankable rows (`0.8889` or 88.89%).
- **Thin Trading Flag**: `HNB.N` has 21-row median span ratio of 1.533x (> 1.5x threshold) and is flagged as stretched horizon.

### Liquidity Flag Share & Consistent Leaders (Development Partition):
Because the warning threshold is set to the 20th percentile by construction, below is the distribution of flagged trading days and Consistent Leader days before vs after filtering across all 20 constituent stocks:

| Stock Symbol | Total Dev Stock-Days | Liquidity Warning Days | Warning Share (%) | Stretched Horizon? | CL Days Before | CL Days After |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `AEL.N` | 1,742 | 0 | 0.00% | No | 51 | 51 |
| `CFIN.N` | 2,337 | 61 | 2.61% | No | 187 | 185 |
| `COMB.N` | 2,111 | 1,136 | 53.81% | No | 41 | 37 |
| `DFCC.N` | 2,389 | 0 | 0.00% | No | 100 | 100 |
| `DIAL.N` | 2,401 | 0 | 0.00% | No | 55 | 55 |
| `HAYL.N` | 2,245 | 523 | 23.30% | No | 126 | 73 |
| `HHL.N` | 2,345 | 80 | 3.41% | No | 279 | 274 |
| `HNB.N` | 1,033 | 1,001 | 96.90% | Yes | 41 | 0 |
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

---

## 5. Model Registry Summary

| Model ID | Family | Servable | Short IC (t) | Mid IC (t) | Long IC (t) | Evidence (S/M/L) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `momentum_126` | Baseline | Yes | +0.0356 (+1.56) | +0.0658 (+1.74) | +0.0682 (+1.32) | RED/AMBER/RED |
| `ridge_core` | Linear | Yes | +0.0006 (+0.03) | +0.0090 (+0.39) | +0.0429 (+0.81) | RED/RED/RED |
| `rf_reg_core` | Bagging | Yes | -0.0009 (-0.05) | -0.0034 (-0.14) | +0.0129 (+0.25) | RED/RED/RED |
| `lgbm_reg_core` | Boosting | Yes | -0.0048 (-0.28) | -0.0230 (-0.97) | +0.0114 (+0.49) | RED/RED/RED |
| `ens_rank4` | Ensemble | Yes | +0.0281 (+1.68) | +0.0429 (+1.41) | +0.0957 (+2.36) | RED/RED/RED |
| `ens_rank4_bestlgbm` | Ensemble | No | +0.0281 (+1.68) | +0.0429 (+1.41) | +0.1143 (+2.79) | RED/RED/RED |
| `logit_core` | Linear | No | -0.0001 (-0.01) | -0.0093 (-0.40) | +0.0294 (+0.54) | RED/RED/RED |
| `rf_clf_core` | Bagging | No | -0.0046 (-0.21) | -0.0114 (-0.49) | +0.0145 (+0.27) | RED/RED/RED |
| `lgbm_clf_core` | Boosting | No | +0.0034 (+0.17) | -0.0043 (-0.18) | +0.0291 (+1.15) | RED/RED/RED |
| `lstm_reg_core` | Deep Learning | No | +0.0455 (+3.65) | +0.0217 (+1.39) | -0.0177 (-0.71) | AMBER/RED/RED |
| `tabpfn_reg_core` | Foundation | No | +0.0075 (+0.37) | +0.0052 (+0.26) | +0.0403 (+1.45) | RED/RED/RED |
| `tabpfn_clf_core` | Foundation | No | +0.0260 (+1.62) | +0.0250 (+0.78) | +0.0094 (+0.21) | RED/RED/RED |
| `reversal_21` | Baseline | No | -0.0042 (-0.27) | -0.0348 (-1.69) | -0.0541 (-2.60) | RED/RED/RED |
| `momentum_63` | Baseline | No | +0.0186 (+0.83) | +0.0494 (+1.33) | +0.0764 (+1.87) | RED/RED/RED |
| `past_winner` | Baseline | No | +0.0099 (+0.35) | +0.0227 (+0.43) | -0.0450 (-0.60) | RED/RED/RED |
| `low_vol` | Baseline | No | -0.0246 (-1.10) | -0.0371 (-1.42) | -0.0786 (-2.20) | RED/RED/RED |
| `random` | Null Baseline | No | +0.0032 (+0.77) | +0.0024 (+0.69) | -0.0014 (-0.36) | RED/RED/RED |

---

## 6. Production Configuration Walk-Forward Evaluation

To ensure that freezing production consensus hyperparameters (without inner CV tuning) does not degrade model efficacy, a walk-forward evaluation was conducted across all folds for `ridge_core`, `rf_reg_core`, `lgbm_reg_core`, and `ens_rank4_prodcfg`.

### Plain-Language Assessment:
**Plain statement: No**, production ens_rank4 is not materially weaker than the validated one. Across all three horizons, the absolute difference in mean development Rank IC is strictly less than 0.02 (Short $\Delta = -0.0067$, Mid $\Delta = +0.0078$, Long $\Delta = +0.0037$).

### Production vs Validated Comparison Table (`ens_rank4`):

| Horizon | Metric | Validated (`ens_rank4`) | Production Config (`ens_rank4_prodcfg`) | Difference (Prod - Validated) |
| :--- | :--- | :---: | :---: | :---: |
| **Short** | Dev Mean IC | `+0.0281` | `+0.0214` | `-0.0067` |
| | Fold t-statistic | `+1.68` | `+1.36` | `-0.32` |
| | Folds with IC > 0 | `9 / 12` | `8 / 12` | `-1` |
| | Top-Tier Hit Rate | `49.37%` | `47.67%` | `-1.70%` |
| **Mid** | Dev Mean IC | `+0.0429` | `+0.0507` | `+0.0078` |
| | Fold t-statistic | `+1.41` | `+1.55` | `+0.14` |
| | Folds with IC > 0 | `8 / 12` | `8 / 12` | `0` |
| | Top-Tier Hit Rate | `50.67%` | `50.00%` | `-0.67%` |
| **Long** | Dev Mean IC | `+0.0957` | `+0.0994` | `+0.0037` |
| | Fold t-statistic | `+2.36` | `+2.47` | `+0.10` |
| | Folds with IC > 0 | `9 / 10` | `8 / 10` | `-1` |
| | Top-Tier Hit Rate | `60.20%` | `63.27%` | `+3.07%` |

### Auto Model Selection Impact (Informational):
- **Short Horizon**: Under the pre-registered Auto selection rule (higher development fold t-statistic; difference below 0.10 favors ens_rank4), validated ens_rank4 achieves fold t=+1.68 vs momentum_126 t=+1.56 (diff = +0.12 > 0.10), correctly selecting ens_rank4. Under the production configuration with consensus hyperparameters, momentum_126 achieves higher development fold t-statistic (+1.56 vs +1.36, diff = +0.19 > 0.10), meaning the Auto choice would switch to momentum_126 under production settings.
- **Mid Horizon**: Unchanged (`momentum_126` remains Auto choice).
- **Long Horizon**: Unchanged (`ens_rank4` remains Auto choice).
- *Note: Following governance rules, the active Auto rule and evidence levels remain unchanged.*