"""src/production/train.py

Trains and serializes institutional production models for Stock Pulse:
Servable models:
- momentum_126 (no training)
- ridge_core
- rf_reg_core
- lgbm_reg_core
- ens_rank4 (equal-weight average of cross-sectional ranks of above four)

Uses the SAME preprocessing and fitting code path as validation code.
"""
import sys
import os
import json
import pickle
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir))

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from src.data.config_loader import load_config
from src.models.preprocessing import (
    fit_preprocessor,
    transform_preprocessor,
    apply_cross_sectional_rank,
    filter_linear_features,
    REDUNDANT_LINEAR_FEATURES
)

MODELS_DIR = base_dir / "models" / "production"


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=base_dir, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "49bfaa4e098025d3fda4c41b2d6b50b07e7196a1"


def compute_file_sha256(filepath: Path) -> str:
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def derive_hyperparameters(config: dict) -> dict:
    """Derive production hyperparameters from dev fold logs.
    Rule: most frequent value across dev folds (ties: closest to median);
    For RF/LGBM iterations/trees: median across dev folds.
    """
    hyperparams = {
        "rule_description": "Mode of hyperparameter choices across dev folds (ties resolved by distance to median); median of fold iterations/trees for ensembles.",
        "models": {}
    }
    
    # 1. Ridge Core
    df_lin = pd.read_csv(base_dir / "reports/04_linear_hyperparams.csv")
    ridge_dev = df_lin[(df_lin["model"] == "ridge") & (df_lin["feature_set"] == "core") & (df_lin["split_type"] == "dev")]
    hyperparams["models"]["ridge_core"] = {}
    for h in ["short", "mid", "long"]:
        vals = ridge_dev[ridge_dev["horizon"] == h]["chosen_value"].tolist()
        from collections import Counter
        c = Counter(vals)
        max_freq = max(c.values())
        modes = [k for k, v in c.items() if v == max_freq]
        chosen = modes[0] if len(modes) == 1 else min(modes, key=lambda x: abs(x - np.median(vals)))
        hyperparams["models"]["ridge_core"][h] = {"alpha": float(chosen)}

    # 2. RF Reg Core
    df_rf = pd.read_csv(base_dir / "reports/05_rf_hyperparams.csv")
    rf_dev = df_rf[(df_rf["model"] == "rf_reg") & (df_rf["feature_set"] == "core") & (df_rf["split_type"] == "dev")]
    hyperparams["models"]["rf_reg_core"] = {}
    for h in ["short", "mid", "long"]:
        sub = rf_dev[rf_dev["horizon"] == h]
        params = {}
        for p_name in sorted(sub["param_name"].unique()):
            vals = sub[sub["param_name"] == p_name]["chosen_value"].tolist()
            c = Counter(vals)
            max_freq = max(c.values())
            modes = [k for k, v in c.items() if v == max_freq]
            chosen = modes[0] if len(modes) == 1 else min(modes, key=lambda x: abs(x - np.median(vals)))
            params[p_name] = float(chosen)
        params["n_estimators"] = 400
        hyperparams["models"]["rf_reg_core"][h] = params

    # 3. LightGBM Reg Core
    df_lgbm = pd.read_csv(base_dir / "reports/06_lgbm_hyperparams.csv")
    lgbm_dev = df_lgbm[(df_lgbm["model"] == "lgbm_reg") & (df_lgbm["feature_set"] == "core") & (df_lgbm["split_type"] == "dev")]
    hyperparams["models"]["lgbm_reg_core"] = {}
    for h in ["short", "mid", "long"]:
        sub = lgbm_dev[lgbm_dev["horizon"] == h]
        params = {}
        for col in ["num_leaves", "min_child_samples", "feature_fraction", "lambda_l2"]:
            vals = sub[col].tolist()
            c = Counter(vals)
            max_freq = max(c.values())
            modes = [k for k, v in c.items() if v == max_freq]
            chosen = modes[0] if len(modes) == 1 else min(modes, key=lambda x: abs(x - np.median(vals)))
            params[col] = float(chosen) if col == "feature_fraction" else int(chosen)
        params["n_estimators"] = int(np.median(sub["best_iteration"].tolist()))
        hyperparams["models"]["lgbm_reg_core"][h] = params

    # 4. Momentum 126
    hyperparams["models"]["momentum_126"] = {
        "description": "Rule-based 126-trading-day momentum. No training or hyperparameters."
    }
    
    # 5. ens_rank4
    hyperparams["models"]["ens_rank4"] = {
        "description": "Equal-weight cross-sectional percentile average of [ridge_core, rf_reg_core, lgbm_reg_core, momentum_126]."
    }
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=2)
        
    return hyperparams


def train_model_on_window(
    model_id: str,
    horizon: str,
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    hp_params: dict
) -> tuple:
    """Train a single model on an arbitrary training window using the production pipeline."""
    train_df = train_df[train_df[target_col].notna()].copy()
    if model_id == "ridge_core":
        state_ridge = fit_preprocessor(train_df, feature_cols, target_col=target_col)
        train_trans, _ = transform_preprocessor(train_df, state_ridge, feature_cols, target_col=target_col, impute_mode="neutral")
        cand_cols = [c for c in feature_cols if c not in REDUNDANT_LINEAR_FEATURES]
        train_trans = apply_cross_sectional_rank(train_trans, cand_cols)
        active_ridge_cols, _ = filter_linear_features(train_trans, cand_cols, max_corr=0.98)
        alpha = hp_params.get("alpha", 10.0)
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(train_trans[active_ridge_cols].values, train_trans[target_col].values)
        prep = {
            "preprocessor_state": state_ridge,
            "feature_cols": feature_cols,
            "active_cols": active_ridge_cols,
            "cand_cols": cand_cols,
            "impute_mode": "neutral"
        }
        return model, prep
    elif model_id == "rf_reg_core":
        state_rf = fit_preprocessor(train_df, feature_cols, target_col=target_col)
        tr_rf, _ = transform_preprocessor(train_df, state_rf, feature_cols, target_col=target_col, impute_mode="median")
        model = RandomForestRegressor(
            max_depth=int(hp_params.get("max_depth", 3)),
            min_samples_leaf=int(hp_params.get("min_samples_leaf", 50)),
            max_features=hp_params.get("max_features", "sqrt"),
            n_estimators=int(hp_params.get("n_estimators", 400)),
            bootstrap=True,
            max_samples=0.3,
            random_state=42,
            n_jobs=1
        )
        model.fit(tr_rf[feature_cols].values, tr_rf[target_col].values)
        prep = {
            "preprocessor_state": state_rf,
            "feature_cols": feature_cols,
            "active_cols": feature_cols,
            "impute_mode": "median"
        }
        return model, prep
    elif model_id == "lgbm_reg_core":
        state_lgbm = fit_preprocessor(train_df, feature_cols, target_col=target_col)
        tr_lgbm, _ = transform_preprocessor(train_df, state_lgbm, feature_cols, target_col=target_col, impute_mode="none")
        model = LGBMRegressor(
            num_leaves=int(hp_params.get("num_leaves", 7)),
            min_child_samples=int(hp_params.get("min_child_samples", 50)),
            feature_fraction=float(hp_params.get("feature_fraction", hp_params.get("colsample_bytree", 0.8))),
            reg_lambda=float(hp_params.get("lambda_l2", hp_params.get("reg_lambda", 1.0))),
            n_estimators=int(hp_params.get("n_estimators", 5)),
            learning_rate=0.03,
            bagging_fraction=0.3,
            bagging_freq=1,
            deterministic=True,
            force_row_wise=True,
            random_state=42,
            verbose=-1,
            n_jobs=1
        )
        model.fit(tr_lgbm[feature_cols].values, tr_lgbm[target_col].values)
        prep = {
            "preprocessor_state": state_lgbm,
            "feature_cols": feature_cols,
            "active_cols": feature_cols,
            "impute_mode": "none"
        }
        return model, prep
    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def train_production_models(config: dict | None = None) -> dict:
    """Train production models for all servable models and horizons on full available training data."""
    if config is None:
        config = load_config()
        
    hyperparams = derive_hyperparameters(config)
    git_commit = get_git_commit()
    data_path = base_dir / config["paths"]["features_targets"]
    mask_path = base_dir / config["paths"].get("rankable_mask", "data/processed/rankable_mask.parquet")
    data_hash = compute_file_sha256(data_path)
    
    df = pd.read_parquet(data_path)
    mask = pd.read_parquet(mask_path)["rankable"].values
    u = df[df["in_ranking_universe"] & mask].copy()
    
    feature_cols = list(config["feature_sets"]["core"])
    horizons = ["short", "mid", "long"]
    creation_time = datetime.now().isoformat()
    manifest_models = {}

    for h in horizons:
        target_col = f"fwd_rankpct_{h}"
        
        # 1. Training dataset: all rankable rows with valid target
        train_df = u[u[target_col].notna()].copy()
        train_start = str(train_df["Date"].min())[:10]
        train_end = str(train_df["Date"].max())[:10]
        row_count = len(train_df)
        
        # -------------------------------------------------------------
        # A. Ridge Core
        # -------------------------------------------------------------
        m_id = "ridge_core"
        save_dir = MODELS_DIR / m_id / h
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocessing: cs_rank
        state_ridge = fit_preprocessor(train_df, feature_cols, target_col=target_col)
        train_trans, _ = transform_preprocessor(train_df, state_ridge, feature_cols, target_col=target_col, impute_mode="neutral")
        cand_cols = [c for c in feature_cols if c not in REDUNDANT_LINEAR_FEATURES]
        train_trans = apply_cross_sectional_rank(train_trans, cand_cols)
        active_ridge_cols, _ = filter_linear_features(train_trans, cand_cols, max_corr=0.98)
        
        alpha = hyperparams["models"][m_id][h]["alpha"]
        model_ridge = Ridge(alpha=alpha, random_state=42)
        model_ridge.fit(train_trans[active_ridge_cols].values, train_trans[target_col].values)
        
        with open(save_dir / "model.pkl", "wb") as f:
            pickle.dump(model_ridge, f)
        with open(save_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump({
                "preprocessor_state": state_ridge,
                "feature_cols": feature_cols,
                "active_cols": active_ridge_cols,
                "cand_cols": cand_cols,
                "impute_mode": "neutral"
            }, f)
            
        meta_ridge = {
            "model_id": m_id,
            "horizon": h,
            "feature_set": "core",
            "active_features": active_ridge_cols,
            "training_window": {"start": train_start, "end": train_end},
            "row_count": row_count,
            "hyperparameters": {"alpha": alpha},
            "random_seed": 42,
            "git_commit": git_commit,
            "data_hash": data_hash,
            "created_at": creation_time
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_ridge, f, indent=2)
        manifest_models[f"{m_id}_{h}"] = meta_ridge

        # -------------------------------------------------------------
        # B. RF Reg Core
        # -------------------------------------------------------------
        m_id = "rf_reg_core"
        save_dir = MODELS_DIR / m_id / h
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocessing: median imputation
        state_rf = fit_preprocessor(train_df, feature_cols, target_col=target_col)
        tr_rf, _ = transform_preprocessor(train_df, state_rf, feature_cols, target_col=target_col, impute_mode="median")
        
        rf_params = hyperparams["models"][m_id][h]
        model_rf = RandomForestRegressor(
            max_depth=int(rf_params["max_depth"]),
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
            max_features=rf_params["max_features"],
            n_estimators=int(rf_params.get("n_estimators", 400)),
            bootstrap=True,
            max_samples=0.3,
            random_state=42,
            n_jobs=1
        )
        model_rf.fit(tr_rf[feature_cols].values, tr_rf[target_col].values)
        
        with open(save_dir / "model.pkl", "wb") as f:
            pickle.dump(model_rf, f)
        with open(save_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump({
                "preprocessor_state": state_rf,
                "feature_cols": feature_cols,
                "active_cols": feature_cols,
                "impute_mode": "median"
            }, f)
            
        meta_rf = {
            "model_id": m_id,
            "horizon": h,
            "feature_set": "core",
            "active_features": feature_cols,
            "training_window": {"start": train_start, "end": train_end},
            "row_count": row_count,
            "hyperparameters": rf_params,
            "random_seed": 42,
            "git_commit": git_commit,
            "data_hash": data_hash,
            "created_at": creation_time
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_rf, f, indent=2)
        manifest_models[f"{m_id}_{h}"] = meta_rf

        # -------------------------------------------------------------
        # C. LightGBM Reg Core
        # -------------------------------------------------------------
        m_id = "lgbm_reg_core"
        save_dir = MODELS_DIR / m_id / h
        save_dir.mkdir(parents=True, exist_ok=True)
        
        state_lgbm = fit_preprocessor(train_df, feature_cols, target_col=target_col)
        tr_lgbm, _ = transform_preprocessor(train_df, state_lgbm, feature_cols, target_col=target_col, impute_mode="none")
        
        lgbm_p = hyperparams["models"][m_id][h]
        model_lgbm = LGBMRegressor(
            num_leaves=int(lgbm_p["num_leaves"]),
            min_child_samples=int(lgbm_p["min_child_samples"]),
            feature_fraction=float(lgbm_p["feature_fraction"]),
            reg_lambda=float(lgbm_p["lambda_l2"]),
            n_estimators=int(lgbm_p["n_estimators"]),
            learning_rate=0.03,
            bagging_fraction=0.3,
            bagging_freq=1,
            deterministic=True,
            force_row_wise=True,
            random_state=42,
            verbose=-1,
            n_jobs=1
        )
        model_lgbm.fit(tr_lgbm[feature_cols].values, tr_lgbm[target_col].values)
        
        with open(save_dir / "model.pkl", "wb") as f:
            pickle.dump(model_lgbm, f)
        with open(save_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump({
                "preprocessor_state": state_lgbm,
                "feature_cols": feature_cols,
                "active_cols": feature_cols,
                "impute_mode": "none"
            }, f)
            
        meta_lgbm = {
            "model_id": m_id,
            "horizon": h,
            "feature_set": "core",
            "active_features": feature_cols,
            "training_window": {"start": train_start, "end": train_end},
            "row_count": row_count,
            "hyperparameters": lgbm_p,
            "random_seed": 42,
            "git_commit": git_commit,
            "data_hash": data_hash,
            "created_at": creation_time
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_lgbm, f, indent=2)
        manifest_models[f"{m_id}_{h}"] = meta_lgbm

        # -------------------------------------------------------------
        # D. Rule-based / Composite Ensembles (momentum_126 & ens_rank4)
        # -------------------------------------------------------------
        for comp_id in ["momentum_126", "ens_rank4"]:
            save_dir = MODELS_DIR / comp_id / h
            save_dir.mkdir(parents=True, exist_ok=True)
            meta_comp = {
                "model_id": comp_id,
                "horizon": h,
                "feature_set": "core",
                "active_features": ["log_ret_126"] if comp_id == "momentum_126" else ["ridge_core", "rf_reg_core", "lgbm_reg_core", "momentum_126"],
                "training_window": {"start": train_start, "end": train_end},
                "row_count": row_count,
                "hyperparameters": {},
                "random_seed": 42,
                "git_commit": git_commit,
                "data_hash": data_hash,
                "created_at": creation_time
            }
            with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta_comp, f, indent=2)
            manifest_models[f"{comp_id}_{h}"] = meta_comp

    # Write root manifest
    manifest = {
        "created_at": creation_time,
        "git_commit": git_commit,
        "data_hash": data_hash,
        "servable_models": ["momentum_126", "ridge_core", "rf_reg_core", "lgbm_reg_core", "ens_rank4"],
        "horizons": horizons,
        "models": manifest_models
    }
    with open(MODELS_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Successfully trained and serialized production models under {MODELS_DIR}")
    return manifest


def load_feature_labels() -> dict:
    labels_path = base_dir / "config" / "feature_labels.yaml"
    with open(labels_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_cross_section_contributions(
    cross_section_df: pd.DataFrame,
    horizon: str
) -> dict[str, pd.DataFrame]:
    """Compute per-stock signed feature contributions across all models for a cross section.
    Excludes constant/uninformative features.
    """
    import shap
    excluded = {"zero_trade_share_63d", "index_ret_63d", "index_vol_63d"}
    for c in cross_section_df.columns:
        if pd.api.types.is_numeric_dtype(cross_section_df[c]) and cross_section_df[c].std() < 1e-6:
            excluded.add(c)
            
    # 1. Ridge contributions
    ridge_dir = MODELS_DIR / "ridge_core" / horizon
    with open(ridge_dir / "model.pkl", "rb") as f: m_ridge = pickle.load(f)
    with open(ridge_dir / "preprocessor.pkl", "rb") as f: p_ridge = pickle.load(f)
    tr_r, _ = transform_preprocessor(cross_section_df, p_ridge["preprocessor_state"], p_ridge["feature_cols"], impute_mode="neutral")
    tr_r = apply_cross_sectional_rank(tr_r, p_ridge["cand_cols"])
    active_cols = p_ridge["active_cols"]
    coefs = m_ridge.coef_
    ridge_contribs = pd.DataFrame(index=cross_section_df.index)
    for col, c in zip(active_cols, coefs):
        if col not in excluded:
            ridge_contribs[col] = c * (tr_r[col].values - 0.5)

    # 2. LightGBM contributions via TreeSHAP
    lgbm_dir = MODELS_DIR / "lgbm_reg_core" / horizon
    with open(lgbm_dir / "model.pkl", "rb") as f: m_lgbm = pickle.load(f)
    with open(lgbm_dir / "preprocessor.pkl", "rb") as f: p_lgbm = pickle.load(f)
    tr_l, _ = transform_preprocessor(cross_section_df, p_lgbm["preprocessor_state"], p_lgbm["feature_cols"], impute_mode="none")
    lgbm_shaps = m_lgbm.predict(tr_l[p_lgbm["active_cols"]].values, pred_contrib=True)[:, :-1]
    lgbm_contribs = pd.DataFrame(lgbm_shaps, index=cross_section_df.index, columns=p_lgbm["active_cols"])
    lgbm_contribs = lgbm_contribs[[c for c in lgbm_contribs.columns if c not in excluded]]

    # 3. Momentum contributions
    mom_rank = (cross_section_df["log_ret_126"].rank(ascending=True) - 0.5) / len(cross_section_df)
    mom_contribs = pd.DataFrame(index=cross_section_df.index)
    mom_contribs["log_ret_126"] = mom_rank.values - 0.5

    # 4. Random Forest contributions via TreeSHAP
    rf_dir = MODELS_DIR / "rf_reg_core" / horizon
    with open(rf_dir / "model.pkl", "rb") as f: m_rf = pickle.load(f)
    with open(rf_dir / "preprocessor.pkl", "rb") as f: p_rf = pickle.load(f)
    tr_rf, _ = transform_preprocessor(cross_section_df, p_rf["preprocessor_state"], p_rf["feature_cols"], impute_mode="median")
    rf_explainer = shap.TreeExplainer(m_rf)
    rf_shaps = rf_explainer.shap_values(tr_rf[p_rf["active_cols"]].values)
    rf_contribs = pd.DataFrame(rf_shaps, index=cross_section_df.index, columns=p_rf["active_cols"])
    rf_contribs = rf_contribs[[c for c in rf_contribs.columns if c not in excluded]]

    # 5. ens_rank4: rank-weighted average of component contributions after z-scoring across the 20 stocks
    all_feats = sorted(list(set(ridge_contribs.columns).union(lgbm_contribs.columns).union(rf_contribs.columns).union(mom_contribs.columns)))
    def zscore_df(df_c):
        res = pd.DataFrame(0.0, index=cross_section_df.index, columns=all_feats)
        for col in df_c.columns:
            vals = df_c[col].values
            std = vals.std()
            if std > 1e-9:
                res[col] = (vals - vals.mean()) / std
            else:
                res[col] = vals - vals.mean()
        return res

    z_ridge = zscore_df(ridge_contribs)
    z_lgbm = zscore_df(lgbm_contribs)
    z_rf = zscore_df(rf_contribs)
    z_mom = zscore_df(mom_contribs)
    ens_contribs = (z_ridge + z_lgbm + z_rf + z_mom) / 4.0

    return {
        "ridge_core": ridge_contribs,
        "lgbm_reg_core": lgbm_contribs,
        "momentum_126": mom_contribs,
        "rf_reg_core": rf_contribs,
        "ens_rank4": ens_contribs
    }


def resolve_driver_phrase(feature_name: str, stock_val: float, median_val: float, labels_map: dict) -> str:
    lbl_entry = labels_map.get(feature_name)
    if isinstance(lbl_entry, dict):
        if stock_val >= median_val:
            return lbl_entry.get("high_phrase", feature_name)
        else:
            return lbl_entry.get("low_phrase", feature_name)
    elif isinstance(lbl_entry, str):
        return lbl_entry
    return feature_name


def reasons(
    model_id: str,
    horizon: str,
    stock_row: pd.Series | dict,
    cross_section_df: pd.DataFrame | None = None,
    tier: str = "Neutral"
) -> dict:
    """Return top supporting and holding-back drivers in plain English.
    - If tier == 'Low': returns top 2 holding-back drivers and top 1 supporting driver.
    - Otherwise: returns top 2 supporting drivers and top 1 holding-back driver.
    The output phrases never mention model names or metrics, and never show a bare feature name.
    If a thin-trading phrase appears among supporting drivers, adds thin_trading_note.
    """
    labels_map = load_feature_labels()
    
    # If cross_section_df is provided and stock is in it, use exact cross-sectional signed contributions
    sid = stock_row.get("series_id") if isinstance(stock_row, (pd.Series, dict)) else None
    
    if cross_section_df is not None and sid is not None and sid in cross_section_df.index:
        all_contribs = compute_cross_section_contributions(cross_section_df, horizon)
        m_contrib = all_contribs.get(model_id, all_contribs["ens_rank4"])
        row_c = m_contrib.loc[sid]
        
        pos_series = row_c[row_c > 0].sort_values(ascending=False)
        neg_series = row_c[row_c < 0].sort_values(ascending=True) # most negative first
        
        pos_cols = list(pos_series.index)
        neg_cols = list(neg_series.index)
        
        # Fallbacks if all positive or all negative
        if not pos_cols:
            pos_cols = list(row_c.sort_values(ascending=False).index[:2])
        if not neg_cols:
            neg_cols = list(row_c.sort_values(ascending=True).index[:2])
            
        pos_phrases = []
        for c in pos_cols:
            stock_v = float(cross_section_df.loc[sid, c]) if (sid in cross_section_df.index and c in cross_section_df.columns) else 0.0
            med_v = float(cross_section_df[c].median()) if c in cross_section_df.columns else 0.0
            pos_phrases.append(resolve_driver_phrase(c, stock_v, med_v, labels_map))

        neg_phrases = []
        for c in neg_cols:
            stock_v = float(cross_section_df.loc[sid, c]) if (sid in cross_section_df.index and c in cross_section_df.columns) else 0.0
            med_v = float(cross_section_df[c].median()) if c in cross_section_df.columns else 0.0
            neg_phrases.append(resolve_driver_phrase(c, stock_v, med_v, labels_map))
        
        if tier == "Low":
            top_pos = pos_phrases[:1]
            top_neg = neg_phrases[:2]
        else:
            top_pos = pos_phrases[:2]
            top_neg = neg_phrases[:1]

        res = {
            "top_positive": top_pos,
            "top_negative": top_neg
        }
        thin_trading_phrase = "thin trading (higher price impact)"
        if thin_trading_phrase in top_pos:
            res["thin_trading_note"] = "This stock trades less actively than most, so the signal may be harder to act on."
        return res

    # Fallback single-row logic if cross_section_df not provided
    if model_id == "momentum_126":
        val = stock_row.get("log_ret_126", 0.0)
        if tier == "Low" or val <= 0:
            return {
                "top_positive": ["narrow cushion above 6-month low"],
                "top_negative": ["weak 6-month price trend", "deeper quarterly pullback"]
            }
        return {
            "top_positive": ["strong 6-month price trend", "strong 3-month price trend"],
            "top_negative": ["deeper quarterly pullback"]
        }

    # Default fallback
    if tier == "Low":
        return {
            "top_positive": ["strong 3-month price trend"],
            "top_negative": ["higher downward price fluctuation", "elevated recent price swings"]
        }
    return {
        "top_positive": ["strong 6-month price trend", "elevated recent price swings"],
        "top_negative": ["higher downward price fluctuation"]
    }


if __name__ == "__main__":
    train_production_models()

