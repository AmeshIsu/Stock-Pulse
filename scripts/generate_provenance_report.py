"""Generate reports/08_provenance.csv for all out-of-sample prediction files."""
import os
import glob
import hashlib
import datetime
from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parents[1]
m3_report_path = base_dir / "reports" / "archive_v4" / "06_model3_lgbm.md"
m3_mtime = m3_report_path.stat().st_mtime
m3_dt = datetime.datetime.fromtimestamp(m3_mtime)

git_commit = "49bfaa4e098025d3fda4c41b2d6b50b07e7196a1"

def get_generator_info(filename: str):
    parts = filename.replace(".parquet", "").split("_")
    horizon = None
    for h in ["short", "mid", "long"]:
        if h in parts:
            horizon = h
            break
            
    idx_h = parts.index(horizon)
    model_family = "_".join(parts[:idx_h])
    feature_set = "_".join(parts[idx_h+1:])
    
    if model_family == "ens_rank4_bestlgbm":
        script = "scripts/update_ensemble_and_eval.py"
        func = "compute_ensemble"
    elif model_family == "ens_rank4":
        script = "scripts/update_ensemble_and_eval.py"
        func = "compute_ensemble"
    elif model_family in ["equal_weight_universe", "past_winner", "reversal_21", "momentum_63", "momentum_126", "low_vol", "random"]:
        script = "src/models/baselines.py"
        func = "run_all_baselines"
    elif model_family in ["ridge", "logit"]:
        script = "src/models/linear_models.py"
        func = "run_all_linear_experiments"
    elif model_family in ["rf_reg", "rf_clf"]:
        script = "src/models/forest_models.py"
        func = "run_all_forest_experiments"
    elif model_family in ["lgbm_reg", "lgbm_clf"]:
        script = "src/models/boosting_models.py"
        func = "run_boosting_experiment"
    elif model_family in ["lstm_reg"]:
        script = "src/models/deep_models.py"
        func = "run_lstm_pipeline"
    elif model_family in ["tabpfn_reg", "tabpfn_clf"]:
        script = "src/models/tabpfn_models.py"
        func = "run_tabpfn_pipeline"
    else:
        script = "unknown"
        func = "unknown"
        
    return model_family, horizon, feature_set, script, func

rows = []
for p in sorted(glob.glob(str(base_dir / "reports" / "oos_predictions" / "*.parquet"))):
    path_obj = Path(p)
    stat = path_obj.stat()
    created_dt = datetime.datetime.fromtimestamp(stat.st_ctime).isoformat()
    modified_dt = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
    
    sha256 = hashlib.sha256(path_obj.read_bytes()).hexdigest()
    model_family, horizon, feature_set, script, func = get_generator_info(path_obj.name)
    
    rel_path = f"reports/oos_predictions/{path_obj.name}"
    modified_after_m3 = (stat.st_mtime > m3_mtime)
    
    rows.append({
        "path": rel_path,
        "created_timestamp": created_dt,
        "modified_timestamp": modified_dt,
        "sha256": sha256,
        "generating_script": script,
        "generating_function": func,
        "git_commit": git_commit,
        "seed": 42,
        "feature_set": feature_set,
        "horizon": horizon,
        "modified_after_model3_report": modified_after_m3
    })

df = pd.DataFrame(rows)
out_csv = base_dir / "reports" / "08_provenance.csv"
df.to_csv(out_csv, index=False)
print(f"Created {out_csv} with {len(df)} rows.")
print("Summary of modified_after_model3_report:")
print(df["modified_after_model3_report"].value_counts())
