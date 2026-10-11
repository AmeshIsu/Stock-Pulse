# Stock Pulse: Administrator & Technical Methodology Guide

**Document**: `data/app/methodology_admin.md` | **Generated**: 2026-10-11 | **Latest Data**: 2024-12-10

---

## 1. System Architecture & Model Serving Governance

Stock Pulse implements a walk-forward purged cross-sectional ranking framework over the Colombo Stock Exchange (S&P SL20).

### Default Served Models (Auto Rule):
- **Short Horizon (21d)**: `ens_rank4` (Equal-weight rank blend of Ridge, Random Forest, LightGBM, and 126d Momentum)
- **Mid Horizon (63d)**: `momentum_126` (Pure 6-month cross-sectional price momentum benchmark)
- **Long Horizon (252d)**: `ens_rank4` (Equal-weight rank blend)

### Servable Model Registry:
The following 5 models are configured as servable and can be selected manually in the administrative console:
1. `momentum_126` (Baseline)
2. `ridge_core` (L2 Linear)
3. `rf_reg_core` (Bagging Ensemble)
4. `lgbm_reg_core` (Gradient Boosted Trees)
5. `ens_rank4` (Multi-Family Rank Ensemble)

Research models (`logit_core`, `rf_clf_core`, `lgbm_clf_core`, `lstm_reg_core`, `tabpfn_reg_core`, `tabpfn_clf_core`) are retained in the registry with `servable: false` for governance auditing.

---

## 2. Statistical Validation Design & Multiple Testing

### Walk-Forward Folds
- **Short & Mid Horizons**: 12 development folds (`dev_2008` through `dev_2019`).
- **Long Horizon**: 10 development folds (`dev_2010` through `dev_2019`). Denominator is strictly 10; long horizon requires 2-year feature formation, so 2008-2009 were never evaluated.
- **Stress Partition**: 2020 COVID shock ($n_{\text{eff}}$ displayed).
- **Holdout Partition**: 2024–2025 out-of-sample evaluation ($n_{\text{eff}}$ displayed).

### Dual Random Null Band & Significance
- Null bands are constructed using 1,000 Monte Carlo block-constant and linear scorer simulations.
- **Short Null Band**: `[-0.0374, +0.0368]`
- **Mid Null Band**: `[-0.0532, +0.0558]`
- **Long Null Band**: `[-0.1252, +0.1231]`

### Holm-Bonferroni Correction
- Across all 69 family evaluations, after step-down Holm correction, no model retains statistical significance at $\alpha = 0.05$.
- All raw significant results are classified as **suggestive, not confirmed**.

---

## 3. Evidence-Level Rules (Post-Hoc Tightening)

### Evaluated Usability Criteria:
1. **Criterion 1**: Dev Mean Rank IC outside corrected null band ($p \le 0.05$).
2. **Criterion 2**: Beats best naive baseline in $\ge 2/3$ dev folds.
3. **Criterion 3**: Style-neutralized Rank IC $> 0.000$.

### Tightened Conservative Rule (`post_hoc: true`):
- **RED**: Criterion 1 fails (inside null band).
- **AMBER**: Criterion 1 passes and $\ge 1$ other criterion passes.
- **GREEN**: All 3 criteria pass.

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

---

## 5. Model Registry Summary

| Model ID | Family | Servable | Short IC (t) | Mid IC (t) | Long IC (t) | Evidence (S/M/L) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `momentum_126` | Baseline | Yes | +0.0356 (+1.56) | +0.0658 (+1.74) | +0.0682 (+1.32) | RED/RED/RED |
| `ridge_core` | Linear | Yes | +0.0006 (+0.03) | +0.0090 (+0.39) | +0.0429 (+0.81) | RED/RED/RED |
| `rf_reg_core` | Bagging | Yes | -0.0009 (-0.05) | -0.0034 (-0.14) | +0.0129 (+0.25) | RED/RED/RED |
| `lgbm_reg_core` | Boosting | Yes | -0.0048 (-0.28) | -0.0230 (-0.97) | +0.0114 (+0.49) | RED/RED/RED |
| `ens_rank4` | Ensemble | Yes | +0.0315 (+1.72) | +0.0429 (+1.41) | +0.1143 (+2.79) | RED/RED/RED |
| `logit_core` | Linear | No | -0.0001 (-0.01) | -0.0093 (-0.40) | +0.0294 (+0.54) | RED/RED/RED |
| `rf_clf_core` | Bagging | No | -0.0046 (-0.21) | -0.0114 (-0.49) | +0.0145 (+0.27) | RED/RED/RED |
| `lgbm_clf_core` | Boosting | No | +0.0034 (+0.17) | -0.0043 (-0.18) | +0.0291 (+1.15) | RED/RED/RED |
| `lstm_reg_core` | Deep Learning | No | +0.0455 (+3.65) | +0.0217 (+1.39) | -0.0177 (-0.71) | AMBER/RED/RED |
| `tabpfn_reg_core` | Foundation | No | +0.0075 (+0.37) | +0.0052 (+0.26) | +0.0403 (+1.45) | RED/RED/RED |
| `tabpfn_clf_core` | Foundation | No | +0.0260 (+1.62) | +0.0250 (+0.78) | +0.0094 (+0.21) | RED/RED/RED |
| `reversal_21` | Baseline | No | -0.0042 (-0.27) | -0.0348 (-1.69) | -0.0541 (-2.60) | RED/RED/RED |
| `momentum_63` | Baseline | No | +0.0186 (+0.83) | +0.0494 (+1.33) | +0.0764 (+1.87) | RED/RED/RED |
| `low_vol` | Baseline | No | -0.0246 (-1.10) | -0.0371 (-1.42) | -0.0786 (-2.20) | RED/RED/RED |
| `past_winner` | Baseline | No | +0.0099 (+0.35) | +0.0227 (+0.43) | -0.0450 (-0.60) | RED/RED/RED |
| `random` | Baseline | No | +0.0032 (+0.77) | +0.0024 (+0.69) | -0.0014 (-0.36) | RED/RED/RED |
| `equal_weight_universe` | Baseline | No | N/A | N/A | N/A | RED/RED/RED |