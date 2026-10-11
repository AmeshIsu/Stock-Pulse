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
