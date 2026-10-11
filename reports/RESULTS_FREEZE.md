# Stock Pulse: Results Freeze Protocol

**Document ID**: `reports/RESULTS_FREEZE.md`  
**Status**: ACTIVE FREEZE  
**Freeze Date**: 2026-10-11  
**Manifest Reference**: [`results_manifest.json`](file:///c:/Users/USER/Documents/Group%20Project/Group%20Project_1/reports/results_manifest.json)

---

## 1. Freeze Scope & Cryptographic Integrity

To ensure empirical reproducibility and prevent silent drift in validation statistics, all core research outputs and configuration files are cryptographically frozen.

The freeze covers:
1. **Master Results Table**: `reports/results_long.csv` (contains all fold-level and summary evaluations across Short, Mid, and Long horizons).
2. **Configuration Settings**: `config.yaml` (contains feature sets, horizon mappings, cost structures, and model parameters).
3. **Out-of-Sample Predictions**: All 72 parquet files in `reports/oos_predictions/` representing walk-forward test fold predictions for all baseline, linear, tree, ensemble, and foundation models.

The SHA-256 checksum of every frozen file is indexed in [`reports/results_manifest.json`](file:///c:/Users/USER/Documents/Group%20Project/Group%20Project_1/reports/results_manifest.json).

---

## 2. Enforcement & Automated Test Gate

An automated test suite (`tests/test_reconciliation_and_split.py`) continuously verifies the SHA-256 hashes of all tracked files against `results_manifest.json`.

If any tracked file is modified without an accompanying entry in [`reports/CHANGELOG.md`](file:///c:/Users/USER/Documents/Group%20Project/Group%20Project_1/reports/CHANGELOG.md) specifying:
- The date of modification,
- The authorized reason, and
- Updated manifest entries,

the test suite will fail immediately, halting pipeline builds.
