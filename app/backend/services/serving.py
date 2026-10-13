"""app/backend/services/serving.py
Serving repository and real-time score composition engine.
Reads data/serving/CURRENT with in-memory caching and reloads when CURRENT or model settings change.
Resolves active models dynamically per horizon (Auto or manual override).
Applies src/scoring/tiers.py for tiers and Consistent Leaders.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.scoring.tiers import assign_tiers, is_consistent_leader

BASE_DIR = Path(__file__).resolve().parents[3]
SERVING_DIR = BASE_DIR / "data" / "serving"
MODELS_DIR = BASE_DIR / "data" / "app" / "models"


class ServingRepository:
    def __init__(self, serving_dir: Path = SERVING_DIR):
        self.serving_dir = serving_dir
        self._cached_as_of: Optional[str] = None
        self._cached_current_mtime: Optional[float] = None
        self._cached_model_parquets: Dict[str, pd.DataFrame] = {}
        self._cached_stocks_json: Optional[List[Dict[str, Any]]] = None
        self._cached_active_defaults: Optional[Dict[str, Any]] = None
        self._cached_thresholds: Optional[Dict[str, Any]] = None
        self._cached_evidence: Dict[str, Dict[str, Any]] = {}
        self._cached_tiers: Dict[str, Dict[str, Any]] = {}

    def get_current_as_of(self) -> str:
        current_file = self.serving_dir / "CURRENT"
        if not current_file.exists():
            raise FileNotFoundError(f"Missing {current_file}")
        return current_file.read_text(encoding="utf-8").strip()

    def _check_reload(self):
        current_file = self.serving_dir / "CURRENT"
        mtime = current_file.stat().st_mtime
        as_of = self.get_current_as_of()
        if self._cached_as_of != as_of or self._cached_current_mtime != mtime:
            self._cached_as_of = as_of
            self._cached_current_mtime = mtime
            self._cached_model_parquets.clear()
            self._cached_stocks_json = None
            self._cached_active_defaults = None
            self._cached_thresholds = None
            self._cached_evidence.clear()
            self._cached_tiers.clear()

    def invalidate_cache(self):
        """Manually invalidate cache on model setting change or refresh."""
        self._cached_as_of = None
        self._cached_current_mtime = None
        self._cached_model_parquets.clear()
        self._cached_stocks_json = None
        self._cached_active_defaults = None
        self._cached_thresholds = None
        self._cached_evidence.clear()
        self._cached_tiers.clear()

    def get_thresholds(self) -> Dict[str, Any]:
        self._check_reload()
        if self._cached_thresholds is None:
            t_path = BASE_DIR / "thresholds.json"
            with open(t_path, "r", encoding="utf-8") as f:
                self._cached_thresholds = json.load(f)
        return self._cached_thresholds

    def get_active_defaults(self) -> Dict[str, Any]:
        self._check_reload()
        if self._cached_active_defaults is None:
            def_path = self.serving_dir / "active_model_defaults.json"
            if def_path.exists():
                with open(def_path, "r", encoding="utf-8") as f:
                    self._cached_active_defaults = json.load(f)
            else:
                self._cached_active_defaults = {
                    "active_models": {"short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"},
                    "horizon_weights": {"short": 0.3333, "mid": 0.3333, "long": 0.3333}
                }
        return self._cached_active_defaults

    def get_model_evidence(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._cached_evidence:
            e_path = MODELS_DIR / model_id / "evidence.json"
            if e_path.exists():
                with open(e_path, "r", encoding="utf-8") as f:
                    self._cached_evidence[model_id] = json.load(f)
            else:
                self._cached_evidence[model_id] = {}
        return self._cached_evidence[model_id]

    def get_model_tiers(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._cached_tiers:
            t_path = MODELS_DIR / model_id / "tiers.json"
            if t_path.exists():
                with open(t_path, "r", encoding="utf-8") as f:
                    self._cached_tiers[model_id] = json.load(f)
            else:
                self._cached_tiers[model_id] = {}
        return self._cached_tiers[model_id]

    def get_model_scores_df(self, model_id: str) -> pd.DataFrame:
        self._check_reload()
        if model_id not in self._cached_model_parquets:
            as_of = self._cached_as_of
            pq_path = self.serving_dir / as_of / f"scores_{model_id}.parquet"
            if not pq_path.exists():
                raise FileNotFoundError(f"Missing serving scores for model {model_id} at {pq_path}")
            self._cached_model_parquets[model_id] = pd.read_parquet(pq_path)
        return self._cached_model_parquets[model_id]

    def get_stocks_metadata(self) -> List[Dict[str, Any]]:
        self._check_reload()
        if self._cached_stocks_json is None:
            as_of = self._cached_as_of
            s_path = self.serving_dir / as_of / "stocks.json"
            if not s_path.exists():
                raise FileNotFoundError(f"Missing stocks.json at {s_path}")
            with open(s_path, "r", encoding="utf-8") as f:
                self._cached_stocks_json = json.load(f)
        return self._cached_stocks_json

    def get_stock_prices_df(self, series_id: str) -> pd.DataFrame:
        self._check_reload()
        as_of = self._cached_as_of
        pq_path = self.serving_dir / as_of / "prices" / f"{series_id}.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(f"Missing prices parquet for {series_id} at {pq_path}")
        return pd.read_parquet(pq_path)

    def resolve_active_models(self, db_model_settings: Optional[Dict[str, Tuple[str, Optional[str]]]] = None) -> Dict[str, str]:
        """Resolve active model per horizon:
        If db_model_settings has entry and mode == 'manual' with valid model_id, use it.
        Otherwise fall back to auto from active_model_defaults.json.
        """
        defaults = self.get_active_defaults().get("active_models", {
            "short": "ens_rank4", "mid": "momentum_126", "long": "ens_rank4"
        })
        resolved = dict(defaults)
        if db_model_settings:
            for h in ["short", "mid", "long"]:
                if h in db_model_settings:
                    mode, m_id = db_model_settings[h]
                    if mode == "manual" and m_id:
                        resolved[h] = m_id
                    elif mode == "auto":
                        resolved[h] = defaults.get(h, "ens_rank4")
        return resolved

    def compute_serving_snapshot(
        self,
        active_models: Dict[str, str],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Compute live served snapshot according to active models and custom weights."""
        self._check_reload()
        as_of = self._cached_as_of
        thresholds = self.get_thresholds()

        # Load scores for active models
        df_short = self.get_model_scores_df(active_models["short"]).set_index("series_id")
        df_mid = self.get_model_scores_df(active_models["mid"]).set_index("series_id")
        df_long = self.get_model_scores_df(active_models["long"]).set_index("series_id")

        # Resolve weights from active models' evidence points
        if custom_weights:
            tot_w = sum(custom_weights.values())
            weights = {k: v / tot_w for k, v in custom_weights.items()}
        else:
            # Derive default weights from active models evidence points
            ev_short = self.get_model_evidence(active_models["short"]).get("short", {})
            ev_mid = self.get_model_evidence(active_models["mid"]).get("mid", {})
            ev_long = self.get_model_evidence(active_models["long"]).get("long", {})
            
            pts_short = max(1.0, float(ev_short.get("evidence_points", 1.0)))
            pts_mid = max(1.0, float(ev_mid.get("evidence_points", 1.0)))
            pts_long = max(1.0, float(ev_long.get("evidence_points", 1.0)))
            tot_pts = pts_short + pts_mid + pts_long
            weights = {
                "short": pts_short / tot_pts,
                "mid": pts_mid / tot_pts,
                "long": pts_long / tot_pts
            }

        # Resolve tier cutoffs for each active model
        cutoffs = {}
        for h, m_id in active_models.items():
            t_data = self.get_model_tiers(m_id).get(h, {})
            cutoffs[h] = float(t_data.get("cutoff", thresholds.get(f"tier_cutoff_{h}", {}).get("value", 66.7)))

        # Confidence labels from active models evidence
        conf_labels = {
            "short": self.get_model_evidence(active_models["short"]).get("short", {}).get("confidence_label", "Very low confidence"),
            "mid": self.get_model_evidence(active_models["mid"]).get("mid", {}).get("confidence_label", "Low confidence"),
            "long": self.get_model_evidence(active_models["long"]).get("long", {}).get("confidence_label", "Very low confidence")
        }

        # Compose stocks
        universe = sorted(df_short.index.unique())
        stocks_result = []

        # Volatility note
        med_vol_val = None
        market_vol_note = None
        vols = []
        for sid in universe:
            r = df_short.loc[sid]
            if r.get("rankable") and pd.notnull(r.get("trade_day_share_63m")):
                # Extract vol_63d if available in stock metadata or parquet
                pass

        # Use volatility note from df_short if present
        if "market_volatility_note" in df_short.columns:
            m_v_n = df_short["market_volatility_note"].dropna()
            if not m_v_n.empty and m_v_n.iloc[0]:
                market_vol_note = str(m_v_n.iloc[0])

        for sid in universe:
            r_s = df_short.loc[sid]
            r_m = df_mid.loc[sid]
            r_l = df_long.loc[sid]

            rankable = bool(r_s.get("rankable", False))
            status = str(r_s.get("status", "ranked"))
            last_traded = str(r_s.get("last_traded_date", as_of))
            risk_category = str(r_s.get("risk_category", "Medium"))
            liq_warn = bool(r_s.get("liquidity_warning", False))
            thin_flag = bool(r_s.get("thin_trading_flag", False))

            s_short = float(r_s["score_short"]) if pd.notnull(r_s.get("score_short")) else None
            s_mid = float(r_m["score_mid"]) if pd.notnull(r_m.get("score_mid")) else None
            s_long = float(r_l["score_long"]) if pd.notnull(r_l.get("score_long")) else None

            # Re-assign tiers based on active model's cutoffs
            t_short = assign_tiers(pd.Series([s_short]), cutoffs["short"]).iloc[0] if s_short is not None else "N/A"
            t_mid = assign_tiers(pd.Series([s_mid]), cutoffs["mid"]).iloc[0] if s_mid is not None else "N/A"
            t_long = assign_tiers(pd.Series([s_long]), cutoffs["long"]).iloc[0] if s_long is not None else "N/A"

            if rankable and s_short is not None and s_mid is not None and s_long is not None:
                ov_score = round(s_short * weights["short"] + s_mid * weights["mid"] + s_long * weights["long"], 2)
                cl = is_consistent_leader(t_short, t_mid, t_long, liq_warn)
            else:
                ov_score = None
                cl = False

            r_short_reasons = r_s.get("reasons_short", {"top_positive": [], "top_negative": []})
            r_mid_reasons = r_m.get("reasons_mid", {"top_positive": [], "top_negative": []})
            r_long_reasons = r_l.get("reasons_long", {"top_positive": [], "top_negative": []})

            thin_note = None
            all_top_pos = (
                r_short_reasons.get("top_positive", []) +
                r_mid_reasons.get("top_positive", []) +
                r_long_reasons.get("top_positive", [])
            )
            if any("thin trading (higher price impact)" in p for p in all_top_pos):
                thin_note = "This stock trades less actively than most, so the signal may be harder to act on."

            stocks_result.append({
                "series_id": sid,
                "status": status,
                "last_traded_date": last_traded,
                "rankable": rankable,
                "overall_upside_score": ov_score,
                "consistent_leader": cl,
                "score_short": s_short,
                "tier_short": t_short,
                "confidence_short": conf_labels["short"],
                "rank_change_21d_short": float(r_s["rank_change_21d_short"]) if pd.notnull(r_s.get("rank_change_21d_short")) else None,
                "reasons_short": r_short_reasons,
                "score_mid": s_mid,
                "tier_mid": t_mid,
                "confidence_mid": conf_labels["mid"],
                "rank_change_21d_mid": float(r_m["rank_change_21d_mid"]) if pd.notnull(r_m.get("rank_change_21d_mid")) else None,
                "reasons_mid": r_mid_reasons,
                "score_long": s_long,
                "tier_long": t_long,
                "confidence_long": conf_labels["long"],
                "rank_change_21d_long": float(r_l["rank_change_21d_long"]) if pd.notnull(r_l.get("rank_change_21d_long")) else None,
                "reasons_long": r_long_reasons,
                "risk_category": risk_category,
                "liquidity_warning": liq_warn,
                "thin_trading_flag": thin_flag,
                "thin_trading_note": thin_note,
                "rank_change_explanation": "indicative: computed with the current model, not an out-of-sample result",
                "market_volatility_note": market_vol_note,
                "data_version": str(r_s.get("data_version", ""))
            })

        return {
            "as_of": as_of,
            "data_version": stocks_result[0]["data_version"] if stocks_result else "",
            "active_models": active_models,
            "horizon_weights": weights,
            "confidence_labels": conf_labels,
            "market_volatility_note": market_vol_note,
            "stocks": stocks_result
        }


serving_repo = ServingRepository()
