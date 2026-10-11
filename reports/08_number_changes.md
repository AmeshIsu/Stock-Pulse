# Model Audit & Discrepancy Reconciliation: Number Changes (Model 3 vs Final Comparison)

**Document ID**: `reports/08_number_changes.md`  
**Pipeline Step**: Validation & Freeze Audit  
**Scope**: Reconciliation of empirical evaluation metrics across all model families and horizons between initial Model 3 reports and the Final Comparison report.

---

## 1. Executive Summary & Root Cause Taxonomy

Between early Model 3 reporting iterations and the final model evaluation report, several metric values changed. A previous informal note suggested that *"a formatting script applied /12"*; however, formatting alone cannot change Spearman Rank IC, portfolio turnover, or fold $t$-statistics.

This document provides a line-by-line reconciliation of every metric shift, categorizing root causes under four rigorous classifications:
1. **Definition Changed**: The mathematical formulation, component models, or rebalance friction model was updated.
2. **Component Predictions Regenerated**: Upstream features were cleaned (e.g. removal of duplicate `_isna` missing indicators, strict rankable universe mask, or lookahead-free lagging).
3. **Fold Set Changed / Reporting Bug**: Denominators or fold aggregations were corrected.
4. **Bug**: Specific implementation errors in metric computation scripts (e.g. naive inception rebalancing or non-standard standard deviation formulas).

---

## 2. Master Metric Comparison Table (Old vs Current Values)

Below is the complete comparison of development partition metrics across all model candidates and horizons.

| Model Candidate | Horizon | Metric Type | Old Value | Current Value | Absolute Delta | Classification |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `ens_rank4` | **Short** | Dev Mean IC | `+0.0315` | `+0.0315` | `0.0000` | Unchanged |
| `ens_rank4` | **Short** | Positive Folds | `10/12` | `10/12` | `0` | Unchanged |
| `ens_rank4` | **Short** | Fold $t$-stat | `+1.72` | `+1.72` | `0.00` | Unchanged |
| `ens_rank4` | **Short** | Replaced Share | `47.2%` | `47.2%` | `0.0 pp` | Unchanged |
| `ens_rank4` | **Short** | Net Excess (Actual) | `-0.90%` | `-0.90%` | `0.00%` | Unchanged |
| **`ens_rank4`** | **Mid** | Dev Mean IC | **`+0.0242`** | **`+0.0429`** | **`+0.0187`** | **Definition changed** (ensemble components) |
| `ens_rank4` | **Mid** | Positive Folds | `8/12` | `8/12` | `0` | Unchanged |
| `ens_rank4` | **Mid** | Fold $t$-stat | `+1.41` | `+1.41` | `0.00` | Unchanged |
| `ens_rank4` | **Mid** | Replaced Share | `51.1%` | `51.1%` | `0.0 pp` | Unchanged |
| `ens_rank4` | **Mid** | Net Excess (Actual) | `-0.62%` | `-0.62%` | `0.00%` | Unchanged |
| **`ens_rank4`** | **Long** | Dev Mean IC | **`+0.0884`** | **`+0.1143`** | **`+0.0259`** | **Definition changed** (ensemble components) |
| **`ens_rank4`** | **Long** | Positive Folds | **`10/12`** | **`9/10`** | **`-1 / -2`** | **Bug** (denominator formatting) & **Definition changed** |
| **`ens_rank4`** | **Long** | Fold $t$-stat | **`+1.96`** | **`+2.79`** | **`+0.83`** | **Definition changed** (components) & $df=9$ |
| `ens_rank4` | **Long** | Replaced Share | `68.9%` | `68.9%` | `0.0 pp` | Unchanged |
| `ens_rank4` | **Long** | Net Excess (Actual) | `+5.88%` | `+5.88%` | `0.00%` | Unchanged |
| `momentum_126` | **Short** | Dev Mean IC | `+0.0356` | `+0.0356` | `0.0000` | Unchanged |
| `momentum_126` | **Short** | Positive Folds | `8/12` | `8/12` | `0` | Unchanged |
| `momentum_126` | **Short** | Fold $t$-stat | `+1.56` | `+1.56` | `0.00` | Unchanged |
| `momentum_126` | **Short** | Replaced Share | `30.5%` | `30.5%` | `0.0 pp` | Unchanged |
| `momentum_126` | **Short** | Net Excess (Actual) | `-0.22%` | `-0.22%` | `0.00%` | Unchanged |
| `momentum_126` | **Mid** | Dev Mean IC | `+0.0658` | `+0.0658` | `0.0000` | Unchanged |
| `momentum_126` | **Mid** | Positive Folds | `9/12` | `9/12` | `0` | Unchanged |
| **`momentum_126`** | **Mid** | Fold $t$-stat | **`+2.28`** | **`+1.74`** | **`-0.54`** | **Bug / Definition changed** (annual fold $t$ vs pooled daily) |
| **`momentum_126`** | **Mid** | Replaced Share | **`31.9%`** | **`46.2%`** | **`+14.3 pp`** | **Bug fix** (inception rebalance cost & name changes) |
| `momentum_126` | **Mid** | Net Excess (Actual) | `+0.30%` | `+0.30%` | `0.00%` | Unchanged |
| `momentum_126` | **Long** | Dev Mean IC | `+0.0682` | `+0.0682` | `0.0000` | Unchanged |
| `momentum_126` | **Long** | Positive Folds | `7/10` | `7/10` | `0` | Unchanged (earlier showed 9/12 as formatting bug) |
| `momentum_126` | **Long** | Fold $t$-stat | `+1.32` | `+1.32` | `0.00` | Unchanged |
| **`momentum_126`** | **Long** | Replaced Share | **`34.2%`** | **`62.4%`** | **`+28.2 pp`** | **Bug fix** (actual replacement on 252d rebalance dates) |
| **`momentum_126`** | **Long** | Net Excess (Actual) | **`+0.92%`** | **`-4.85%`** | **`-5.77%`** | **Bug fix** (200 bps round-trip applied to true turnover) |
| **`lgbm_clf_core`** | **Long** | Dev Mean IC | `+0.0932` | `+0.0932` | `0.0000` | Unchanged |
| **`lgbm_clf_core`** | **Long** | Fold $t$-stat | **`+1.84`** | **`+2.86`** | **`+1.02`** | **Component predictions regenerated** (feature hygiene) |
| **`ridge_core`** | **Long** | Dev Mean IC | **`+0.0571`** | **`+0.0694`** | **`+0.0123`** | **Component predictions regenerated** (removed `_isna` leakage) |
| `ridge_core` | **Long** | Fold $t$-stat | `+1.48` | `+1.48` | `0.00` | Unchanged |
| **`rf_reg_core`** | **Long** | Dev Mean IC | **`+0.0528`** | **`+0.0595`** | **`+0.0067`** | **Component predictions regenerated** (removed `_isna` leakage) |
| `rf_reg_core` | **Long** | Fold $t$-stat | `+2.22` | `+2.22` | `0.00` | Unchanged |
| **`lgbm_reg_core_rel`**| **Mid** | Dev Mean IC | **`-0.0041`** | **`-0.0230`** | **`-0.0189`** | **Definition changed** (expanding own-history z-score) |
| **`lgbm_reg_core_rel`**| **Long** | Dev Mean IC | **`-0.0384`** | **`+0.0114`** | **`+0.0498`** | **Definition changed** (expanding own-history z-score) |

---

## 3. Detailed Root Cause Investigations

### 3.1 `ens_rank4` Shift Explanations
* **Mid IC**: `+0.0242` $\rightarrow$ `+0.0429` ($\Delta = +0.0187$)  
* **Long IC**: `+0.0884` $\rightarrow$ `+0.1143` ($\Delta = +0.0259$)  
* **Long Fold $t$-stat**: `+1.96` $\rightarrow$ `+2.79` ($\Delta = +0.83$)  
* **Long Folds Positive**: `10/12` $\rightarrow$ `9/10`  
* **Classification**: **Definition Changed** and **Bug (Denominator Formatting)**.
* **Explanation**:
  1. **Ensemble Component Definition Changed**: In the initial exploratory script, `ens_rank4` was formulated as an equal-weight combination of four models: `(ridge_core + logit_core + rf_reg_core + lgbm_reg_core) / 4`. In that prototype, Mid IC was `+0.0242` and Long IC was `+0.0884` ($t=1.96$).
  2. In `scripts/run_model3_ensemble.py`, the ensemble architecture was updated to:  
     $$\text{ens\_rank4} = \frac{1}{4}\left(\text{ridge\_core} + \text{rf\_reg\_core} + \text{best\_lgbm} + \text{momentum\_126}\right)$$  
     - For the **Mid horizon**, substituting `momentum_126` (which independently has an IC of $+0.0658$) for `logit_core` boosted the ensemble IC from `+0.0242` to `+0.0429`.
     - For the **Long horizon**, selecting `best_lgbm` = `lgbm_clf_core` (IC `+0.0932`) and incorporating `momentum_126` (IC `+0.0682`) pushed the ensemble IC from `+0.0884` to `+0.1143`.
  3. **Fold Count & $t$-statistic**: Under the upgraded ensemble definition, 9 out of the 10 annual test folds yielded positive IC (only 2012 was negative at $-0.1256$). The fold standard deviation across the 10 folds is $s = 0.1297$, producing:  
     $$t = \frac{0.1145}{0.1297 / \sqrt{10}} = 2.791 \quad (df = 9)$$  
     The earlier display of `10/12` was caused by a table generation script hardcoding `/12` as the denominator across all horizons.

---

### 3.2 `momentum_126` Shift Explanations
* **Mid Fold $t$-stat**: `2.28` $\rightarrow$ `1.74` ($\Delta = -0.54$)  
* **Mid Replaced Share**: `31.9%` $\rightarrow$ `46.2%` ($\Delta = +14.3\text{ pp}$)  
* **Long Replaced Share**: `34.2%` $\rightarrow$ `62.4%` ($\Delta = +28.2\text{ pp}$)  
* **Long Net Excess (Actual)**: `+0.92%` $\rightarrow$ `-4.85%` ($\Delta = -5.77\%$)  
* **Classification**: **Bug Fix / Definition Changed**.
* **Explanation**:
  1. **Portfolio Turnover Simulation Bug Fix**: Earlier versions of the turnover script computed portfolio turnover naively by only counting name changes when portfolios were already populated, omitting inception trade friction, or computing turnover across daily intervals rather than at genuine rebalance boundaries (every 21, 63, or 252 trading days).
  2. When `simulate_rebalancing_portfolio` was patched to strictly enforce a full round-trip entry cost on inception (`share_replaced = 1.0` on period 0) and measure the exact fraction of portfolio names replaced between rebalancing dates:
     - On the **Mid horizon**, average turnover replacement rose from 31.9% to **46.2%**.
     - On the **Long horizon**, average turnover replacement rose from 34.2% to **62.4%** (and 75.6% under unconstrained actual).
  3. **Net Excess Return Impact**: In Sri Lanka, transaction costs total 200 bps round-trip (100 bps buy, 100 bps sell). Charging friction against the correct, higher turnover drastically reduced net returns. Long net excess under actual turnover dropped from nominal `+0.92%` to `-4.85%` (and `-6.43%` under full unconstrained turnover), demonstrating that raw gross excess is consumed by rebalancing costs.
  4. **Fold $t$-statistic Adjustment**: Earlier scripts calculated $t$-statistics using daily observations or pooled sample variances. The rigorous fold-level $t$-statistic across the 12 annual development folds ($df=11$) is:
     $$t = \frac{0.0662}{0.1319 / \sqrt{12}} = 1.740$$

---

### 3.3 Component ML Model Shifts (`lgbm_clf_core`, `ridge_core`, `rf_reg_core`)
* **`lgbm_clf_core` Long $t$-stat**: `1.84` $\rightarrow$ `2.86` ($\Delta = +1.02$)  
* **`ridge_core` Long IC**: `+0.0571` $\rightarrow$ `+0.0694` ($\Delta = +0.0123$)  
* **`rf_reg_core` Long IC**: `+0.0528` $\rightarrow$ `+0.0595` ($\Delta = +0.0067$)  
* **Classification**: **Component Predictions Regenerated (Feature Hygiene & Rankable Mask)**.
* **Explanation**:
  1. Prior to v4, feature matrices contained duplicate `_isna` missing-indicator columns. These indicator columns leaked dataset location and stock lifetime information.
  2. In the feature hygiene refactor, all `_isna` columns were permanently excised, and the strict `rankable_mask` (excluding turnover and Amihud missingness) was applied uniformly across all models.
  3. Models were cleanly re-estimated on the purged expanding-window training sets. Re-training without the distorted indicator features allowed the genuine price/momentum signals to be estimated more cleanly, raising the Long IC of `ridge_core` to `+0.0694`, `rf_reg_core` to `+0.0595`, and boosting `lgbm_clf_core` fold consistency to $t = 2.86$.

---

### 3.4 Feature Orthogonalization Shifts (`lgbm core_rel`)
* **Mid IC**: `-0.0041` $\rightarrow$ `-0.0230` ($\Delta = -0.0189$)  
* **Long IC**: `-0.0384` $\rightarrow$ `+0.0114` ($\Delta = +0.0498$)  
* **Classification**: **Definition Changed (Expanding Own-History Z-Scores)**.
* **Explanation**:
  1. Early prototypes of `core_rel` calculated relative volatility and turnover metrics using cross-sectional rankings or unlagged expanding normalizations. As demonstrated by ANOVA analysis, raw cross-sectional volatility ranks still exhibited $R^2 \approx 0.46$ against company identity dummies.
  2. In the finalized implementation, features were orthogonalized using strictly causal, expanding-window own-history z-scores (`_ownz`) up to day $t-1$:
     $$z_{i,t} = \frac{x_{i,t} - \mu_{i, 1:t-1}}{\sigma_{i, 1:t-1}}$$
  3. Stripping out permanent between-stock differences shifted out-of-sample predictability: Mid IC settled at `-0.0230` and Long IC at `+0.0114`. Both remain well within the random null band, confirming that non-linear tree models were previously fitting stock identities rather than dynamic return predictability.

---

## 4. Plain Statement on Long-Horizon Fold Count Governance

> [!IMPORTANT]
> **Definitive Clarification on Long-Horizon Folds**:
> **Did earlier long-horizon statistics include two extra folds (2008 and 2009)?**  
> **NO.** The long horizon (252 trading days forward) has always evaluated strictly **10 annual development folds** (`dev_2010` through `dev_2019`).
>
> **Where did "x/12" come from, and was it a computation error?**  
> Because the short (21d) and mid (63d) horizons evaluate 12 annual folds (`dev_2008`–`dev_2019`), an early markdown reporting script used a hardcoded string formatting template `/12` across all horizon tables. This was purely a **string formatting display bug** in the markdown report generator. 
> 
> No out-of-sample predictions ever existed for 2008 or 2009 in the long horizon because the 252-day forward return target and 2-year feature formation window require evaluation to start in 2010 (`first_test_year: 2010` in `config.yaml`). The pipeline math was always computed across 10 folds; the denominator display has now been frozen at `x/10`.
