"""Generate stock_pulse_full_evaluation.ipynb programmatically."""
import json
from pathlib import Path

out_path = Path(__file__).parent / "stock_pulse_full_evaluation.ipynb"

def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

cells = []

# ─── Title ───────────────────────────────────────────────────────────────────
cells.append(md(
    "# 📊 Stock Pulse — Comprehensive Model Evaluation\n\n"
    "Run **every cell top-to-bottom**. All numbers derive from on-disk artefacts.\n\n"
    "| # | Topic |\n"
    "|---|-------|\n"
    "| 0 | Setup & Data Loading |\n"
    "| 1 | Model Comparison |\n"
    "| 2 | Learning Curves |\n"
    "| 3 | Classification Metrics (Accuracy, Precision, Recall, F1, Specificity) |\n"
    "| 4 | ROC-AUC & PR-AUC |\n"
    "| 5 | Regression Metrics (MAE, MSE) & Predicted vs. Actual Analysis |\n"
    "| 6 | Overfitting / Underfitting / Generalisation Gap |\n"
    "| 7 | Cross-Validation |\n"
    "| 8 | Computational Efficiency |\n"
    "| 9 | Class Imbalance |\n"
    "| 10 | Interpretability — Ridge Coefficients |\n"
    "| 11 | Explainability — SHAP & Permutation Importance |\n"
    "| 12 | Summary Dashboard |"
))

# ─── Section 0 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 0. Setup & Data Loading"))

cells.append(code(
    '"""0.1 Imports"""\n'
    'import warnings, time, itertools, json\n'
    'from pathlib import Path\n'
    'import numpy as np\n'
    'import pandas as pd\n'
    'import matplotlib.pyplot as plt\n'
    'import matplotlib.ticker as mtick\n'
    'import matplotlib.patches as mpatches\n'
    'import seaborn as sns\n'
    'from sklearn.metrics import (\n'
    '    roc_auc_score, average_precision_score,\n'
    '    precision_recall_curve, roc_curve, confusion_matrix,\n'
    '    mean_absolute_error, mean_squared_error,\n'
    '    accuracy_score, f1_score, precision_score, recall_score\n'
    ')\n'
    'from scipy.stats import spearmanr\n'
    'warnings.filterwarnings("ignore")\n'
    'try:\n'
    '    from IPython.display import display\n'
    'except ImportError:\n'
    '    display = print\n'
    '\n'
    'def show_styled(df, caption="", fmt="{:.4f}", cmap="RdYlGn", axis=0):\n'
    '    try:\n'
    '        import jinja2\n'
    '        return df.style.background_gradient(cmap=cmap, axis=axis).format(fmt).set_caption(caption)\n'
    '    except Exception:\n'
    '        if caption:\n'
    '            print(f"=== {caption} ===")\n'
    '        return df.round(4)\n'
    '\n'
    'plt.rcParams.update({\n'
    '    "figure.dpi": 130, "axes.spines.top": False,\n'
    '    "axes.spines.right": False, "font.family": "DejaVu Sans",\n'
    '    "axes.titlesize": 13, "axes.labelsize": 11,\n'
    '})\n'
    'PALETTE = ["#2196F3","#FF5722","#4CAF50","#9C27B0",\n'
    '           "#FF9800","#00BCD4","#E91E63","#795548","#607D8B","#FFC107"]\n'
    'print("Imports OK")'
))

cells.append(code(
    '"""0.2 Paths — robustly find project root"""\n'
    'ROOT = Path(".").resolve()\n'
    'if not (ROOT / "reports").exists():\n'
    '    ROOT = ROOT.parent\n'
    'if not (ROOT / "reports").exists():\n'
    '    ROOT = Path("..").resolve()\n'
    'OOS  = ROOT / "reports" / "oos_predictions"\n'
    'REP  = ROOT / "reports"\n'
    'print(f"Project root : {ROOT}")\n'
    'n_p = len(list(OOS.glob("*.parquet")))\n'
    'print(f"Parquet files: {n_p}")'
))

cells.append(code(
    '"""0.3 Load results_long.csv"""\n'
    'RESULTS = pd.read_csv(REP / "results_long.csv")\n'
    'print(f"Shape   : {RESULTS.shape}")\n'
    'print(f"Models  : {sorted(RESULTS[\'model\'].unique())}")\n'
    'print(f"Horizons: {sorted(RESULTS[\'horizon\'].unique())}")\n'
    'print(f"Metrics : {RESULTS[\'metric\'].nunique()} unique")'
))

cells.append(code(
    '"""0.4 Helpers"""\n'
    'def mean_metric(metric, horizon="mid", split_type="dev", models=None):\n'
    '    sub = RESULTS[\n'
    '        (RESULTS["metric"] == metric) &\n'
    '        (RESULTS["horizon"] == horizon) &\n'
    '        (RESULTS["split_type"] == split_type)\n'
    '    ]\n'
    '    if models is not None:\n'
    '        sub = sub[sub["model"].isin(models)]\n'
    '    return sub.groupby("model")["value"].mean().sort_values(ascending=False)\n'
    '\n'
    'PRIMARY_MODELS = ["ens_rank4","lgbm_reg","rf_reg","ridge",\n'
    '                  "lgbm_clf","rf_clf","momentum_126","lstm_reg"]\n'
    'ALL_MODELS  = sorted(RESULTS["model"].unique().tolist())\n'
    'HORIZONS    = ["short","mid","long"]\n'
    'FEATURE_LABELS = {\n'
    '    "log_ret_5":"5d Ret","log_ret_10":"10d Ret","log_ret_21":"21d Ret",\n'
    '    "log_ret_63":"63d Ret","log_ret_126":"126d Ret","log_ret_252":"252d Ret",\n'
    '    "vol_21d":"21d Vol","vol_63d":"63d Vol","downside_vol_63d":"Downside Vol",\n'
    '    "rsi_14":"RSI-14","macd_hist_norm":"MACD Hist","bollinger_pct_b_20":"Bollinger%B",\n'
    '    "amihud_illiq_63d":"Amihud","zero_trade_share_63d":"Zero-Trade%",\n'
    '    "rel_ret_21d":"RelRet 21d","rel_ret_63d":"RelRet 63d",\n'
    '    "cs_rank_mom_63d":"CS Mom Rank","cs_rank_vol_63d":"CS Vol63",\n'
    '    "cs_rank_vol_21d":"CS Vol21","index_ret_63d":"Idx Ret","index_vol_63d":"Idx Vol",\n'
    '}\n'
    'print("Helpers ready")'
))

cells.append(code(
    '"""0.5 Load OOS parquet files"""\n'
    't0 = time.perf_counter()\n'
    '\n'
    'def load_oos(stem, hz="mid", fs="core"):\n'
    '    p = OOS / f"{stem}_{hz}_{fs}.parquet"\n'
    '    return pd.read_parquet(p) if p.exists() else pd.DataFrame()\n'
    '\n'
    'OOS_DICT = {}\n'
    'LOAD_TARGETS = [\n'
    '    ("ens_rank4","mid","core"), ("lgbm_clf","mid","core"),\n'
    '    ("lgbm_reg","mid","core"), ("rf_clf","mid","core"),\n'
    '    ("rf_reg","mid","core"),   ("ridge","mid","core"),\n'
    '    ("momentum_126","mid","core"), ("logit","mid","core"),\n'
    '    ("lstm_reg","mid","core"),\n'
    '    ("ens_rank4","short","core"), ("lgbm_reg","short","core"),\n'
    '    ("ridge","short","core"),\n'
    '    ("ens_rank4","long","core"), ("lgbm_reg","long","core"),\n'
    '    ("ridge","long","core"),\n'
    ']\n'
    'for stem, hz, fs in LOAD_TARGETS:\n'
    '    key = f"{stem}_{hz}"\n'
    '    df  = load_oos(stem, hz, fs)\n'
    '    if not df.empty:\n'
    '        df["Date"] = pd.to_datetime(df["Date"])\n'
    '        OOS_DICT[key] = df\n'
    '\n'
    'print(f"Loaded {len(OOS_DICT)} OOS frames in {time.perf_counter()-t0:.1f}s")\n'
    'for k,v in OOS_DICT.items():\n'
    '    print(f"  {k:30s} rows={v.shape[0]}")'
))

# ─── Section 1 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 1. Model Comparison"))

cells.append(code(
    '"""1.1 Summary table – key metrics across horizons (dev folds)"""\n'
    'KEY_METRICS = [\n'
    '    "rank_ic_mean","rank_ic_ir","top_k_precision",\n'
    '    "top_k_fwd_excess_mean","portfolio_net_excess_mean",\n'
    '    "portfolio_win_rate","mae_rank","rmse_rank"\n'
    ']\n'
    'rows = []\n'
    'for hz in HORIZONS:\n'
    '    for model in ALL_MODELS:\n'
    '        sub = RESULTS[\n'
    '            (RESULTS["model"]==model) & (RESULTS["horizon"]==hz) &\n'
    '            (RESULTS["split_type"]=="dev") & (RESULTS["metric"].isin(KEY_METRICS))\n'
    '        ]\n'
    '        row = {"model":model,"horizon":hz}\n'
    '        row.update(sub.groupby("metric")["value"].mean().to_dict())\n'
    '        rows.append(row)\n'
    'COMP = pd.DataFrame(rows).set_index(["model","horizon"])\n'
    'COMP = COMP[[c for c in KEY_METRICS if c in COMP.columns]]\n'
    'mid_comp = COMP.xs("mid",level="horizon").sort_values("rank_ic_mean",ascending=False)\n'
    'display(show_styled(mid_comp, caption="Dev-set mean (mid horizon). Green=better."))'
))

cells.append(code(
    '"""1.2 Rank IC bar chart — all three horizons"""\n'
    'fig, axes = plt.subplots(1,3,figsize=(17,5))\n'
    'for ax, hz in zip(axes, HORIZONS):\n'
    '    ic = mean_metric("rank_ic_mean", hz, "dev").dropna()\n'
    '    clrs = ["#1565C0" if "ens" in m else "#2196F3" if "lgbm" in m\n'
    '            else "#4CAF50" if "rf" in m else "#FF9800" if m=="ridge"\n'
    '            else "#9E9E9E" for m in ic.index]\n'
    '    ax.barh(ic.index[::-1], ic.values[::-1], color=clrs[::-1])\n'
    '    ax.axvline(0, color="black", lw=0.8, ls="--")\n'
    '    ax.set_title(f"Rank IC — {hz.capitalize()}")\n'
    '    ax.set_xlabel("Mean Rank IC (dev folds)")\n'
    '    for p, v in zip(ax.patches, ic.values[::-1]):\n'
    '        ax.text(max(v,0)+0.001, p.get_y()+p.get_height()/2,\n'
    '                f"{v:.4f}", va="center", fontsize=7.5)\n'
    'plt.suptitle("Model Comparison — Rank IC",fontsize=14,fontweight="bold",y=1.01)\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""1.3 Top-5 Precision heatmap"""\n'
    'prec = {}\n'
    'for hz in HORIZONS:\n'
    '    for m in ALL_MODELS:\n'
    '        v = RESULTS[\n'
    '            (RESULTS["model"]==m)&(RESULTS["horizon"]==hz)&\n'
    '            (RESULTS["split_type"]=="dev")&(RESULTS["metric"]=="top_k_precision")\n'
    '        ]["value"].mean()\n'
    '        prec.setdefault(m,{})[hz] = v\n'
    'prec_df = pd.DataFrame(prec).T.sort_values("mid",ascending=False)\n'
    'prec_df.columns=["Short","Mid","Long"]\n'
    'fig, ax = plt.subplots(figsize=(7,7))\n'
    'sns.heatmap(prec_df.astype(float),annot=True,fmt=".3f",\n'
    '            cmap="RdYlGn",center=0.5,vmin=0.4,vmax=0.65,ax=ax,\n'
    '            linewidths=0.5,cbar_kws={"label":"Top-5 Precision"})\n'
    'ax.set_title("Top-5 Precision (0.5 = random baseline)")\n'
    'plt.tight_layout(); plt.show()'
))

# ─── Section 2 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 2. Learning Curves (Walk-Forward Fold Progression)"))

cells.append(code(
    '"""2.1 IC per fold — all horizons"""\n'
    'SHOW = ["ens_rank4","lgbm_reg","rf_reg","ridge","momentum_126"]\n'
    'fig, axes = plt.subplots(1,3,figsize=(17,5))\n'
    'for ax, hz in zip(axes, HORIZONS):\n'
    '    for i, m in enumerate(SHOW):\n'
    '        sub = RESULTS[\n'
    '            (RESULTS["model"]==m)&(RESULTS["horizon"]==hz)&\n'
    '            (RESULTS["split_type"]=="dev")&(RESULTS["metric"]=="rank_ic_mean")\n'
    '        ].copy()\n'
    '        sub = sub[sub["fold"].str.startswith("dev_")].copy()\n'
    '        if sub.empty: continue\n'
    '        sub["year"] = sub["fold"].str.extract(r"(\\d{4})")[0].astype(int)\n'
    '        sub = sub.sort_values("year")\n'
    '        ax.plot(sub["year"],sub["value"],marker="o",ms=4,\n'
    '                label=m,color=PALETTE[i],lw=1.8)\n'
    '    ax.axhline(0,color="black",lw=0.7,ls="--")\n'
    '    ax.set_title(f"Learning Curve — {hz.capitalize()}")\n'
    '    ax.set_xlabel("Test Year"); ax.set_ylabel("Rank IC")\n'
    '    ax.legend(fontsize=8)\n'
    'plt.suptitle("Learning Curves: IC per Walk-Forward Fold",\n'
    '             fontsize=14,fontweight="bold",y=1.02)\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""2.2 Cumulative mean IC (expanding window benefit)"""\n'
    'hz = "mid"\n'
    'fig, ax = plt.subplots(figsize=(11,5))\n'
    'for i, m in enumerate(SHOW):\n'
    '    sub = RESULTS[\n'
    '        (RESULTS["model"]==m)&(RESULTS["horizon"]==hz)&\n'
    '        (RESULTS["split_type"]=="dev")&(RESULTS["metric"]=="rank_ic_mean")\n'
    '    ].copy()\n'
    '    sub = sub[sub["fold"].str.startswith("dev_")].copy()\n'
    '    if sub.empty: continue\n'
    '    sub["year"] = sub["fold"].str.extract(r"(\\d{4})")[0].astype(int)\n'
    '    sub = sub.sort_values("year")\n'
    '    sub["cum_ic"] = sub["value"].expanding().mean()\n'
    '    ax.plot(sub["year"],sub["cum_ic"],marker="s",ms=4,\n'
    '            label=m,color=PALETTE[i],lw=2)\n'
    'ax.axhline(0,color="black",lw=0.7,ls="--")\n'
    'ax.set_title(f"Cumulative Mean IC ({hz} horizon)")\n'
    'ax.set_xlabel("Test Year"); ax.set_ylabel("Expanding-Window Mean IC")\n'
    'ax.legend(); plt.tight_layout(); plt.show()'
))

# ─── Section 3 ───────────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 3. Classification Metrics — Accuracy, Precision, Recall, F1, Specificity"
))

cells.append(code(
    '"""3.1 Compute from OOS parquets"""\n'
    'clf_records = []\n'
    '\n'
    'def _clf(df, hz, label, use_prob):\n'
    '    tc = f"outperform_{hz}"\n'
    '    valid = df.dropna(subset=[tc])\n'
    '    if use_prob and "pred_prob" in valid.columns:\n'
    '        valid = valid.dropna(subset=["pred_prob"])\n'
    '        scores = valid["pred_prob"].values\n'
    '    else:\n'
    '        valid = valid.copy()\n'
    '        valid["cs_med"] = valid.groupby("Date")["pred_score"].transform("median")\n'
    '        scores = valid["pred_score"].rank(pct=True).values\n'
    '    y_pred = (scores >= 0.5).astype(int)\n'
    '    y_true = valid[tc].astype(int).values\n'
    '    if y_true.sum() == 0 or y_true.sum() == len(y_true): return\n'
    '    tn,fp,fn,tp = confusion_matrix(y_true,y_pred).ravel()\n'
    '    clf_records.append({\n'
    '        "Model": label, "Horizon": hz,\n'
    '        "Accuracy":    accuracy_score(y_true,y_pred),\n'
    '        "Precision":   precision_score(y_true,y_pred,zero_division=0),\n'
    '        "Recall":      recall_score(y_true,y_pred,zero_division=0),\n'
    '        "F1":          f1_score(y_true,y_pred,zero_division=0),\n'
    '        "Specificity": tn/(tn+fp) if (tn+fp)>0 else float("nan"),\n'
    '        "ROC-AUC":     roc_auc_score(y_true,scores),\n'
    '        "PR-AUC":      average_precision_score(y_true,scores),\n'
    '        "TP":int(tp),"TN":int(tn),"FP":int(fp),"FN":int(fn),\n'
    '    })\n'
    '\n'
    'for mk,hz,up in [\n'
    '    ("lgbm_clf","mid",True),("rf_clf","mid",True),("logit","mid",True),\n'
    '    ("ens_rank4","mid",False),("lgbm_reg","mid",False),\n'
    '    ("rf_reg","mid",False),("ridge","mid",False),\n'
    ']:\n'
    '    if f"{mk}_{hz}" in OOS_DICT:\n'
    '        _clf(OOS_DICT[f"{mk}_{hz}"],hz,mk,up)\n'
    '\n'
    'CLF_DF = pd.DataFrame(clf_records).set_index("Model")\n'
    'print("Computed for:", CLF_DF.index.tolist())\n'
    'CLF_DF[["Accuracy","Precision","Recall","F1","Specificity","ROC-AUC","PR-AUC"]].round(4)'
))

cells.append(code(
    '"""3.2 Radar chart"""\n'
    'RM = ["Accuracy","Precision","Recall","F1","Specificity"]\n'
    'angles = np.linspace(0,2*np.pi,len(RM),endpoint=False).tolist()+[0]\n'
    'fig, ax = plt.subplots(figsize=(8,7),subplot_kw=dict(polar=True))\n'
    'for i,(m,row) in enumerate(CLF_DF[RM].iterrows()):\n'
    '    v = row.values.tolist()+[row.values[0]]\n'
    '    ax.plot(angles,v,"o-",lw=2,label=m,color=PALETTE[i%len(PALETTE)])\n'
    '    ax.fill(angles,v,alpha=0.07,color=PALETTE[i%len(PALETTE)])\n'
    'ax.set_thetagrids(np.degrees(angles[:-1]),RM,fontsize=11)\n'
    'ax.set_ylim(0,1)\n'
    'ax.set_title("Classification Metrics Radar\\n(mid horizon, full OOS)",pad=20)\n'
    'ax.legend(loc="upper right",bbox_to_anchor=(1.3,1.15),fontsize=9)\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""3.3 Confusion matrices"""\n'
    'n = len(clf_records)\n'
    'fig, axes = plt.subplots(1,n,figsize=(4*n,4))\n'
    'if n == 1: axes = [axes]\n'
    'for ax, rec in zip(axes, clf_records):\n'
    '    cm = np.array([[rec["TN"],rec["FP"]],[rec["FN"],rec["TP"]]])\n'
    '    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",\n'
    '                xticklabels=["Pred 0","Pred 1"],\n'
    '                yticklabels=["True 0","True 1"],ax=ax,cbar=False)\n'
    '    ax.set_title(f\'{rec["Model"]}\\nF1={rec["F1"]:.3f}\')\n'
    'plt.suptitle("Confusion Matrices — mid horizon",fontsize=13,y=1.02)\n'
    'plt.tight_layout(); plt.show()'
))

# ─── Section 4 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 4. ROC-AUC & PR-AUC"))

cells.append(code(
    '"""4.1 ROC and PR curves"""\n'
    'hz = "mid"; tc = f"outperform_{hz}"\n'
    'PMOD = [\n'
    '    ("lgbm_clf",True,"LGBM Clf"),("rf_clf",True,"RF Clf"),\n'
    '    ("logit",True,"Logit"),("ens_rank4",False,"Ensemble"),\n'
    '    ("lgbm_reg",False,"LGBM Reg"),("ridge",False,"Ridge"),\n'
    ']\n'
    'fig, (ax_roc,ax_pr) = plt.subplots(1,2,figsize=(14,6))\n'
    'for i,(mk,up,lab) in enumerate(PMOD):\n'
    '    key = f"{mk}_{hz}"\n'
    '    if key not in OOS_DICT: continue\n'
    '    df = OOS_DICT[key].dropna(subset=[tc])\n'
    '    y  = df[tc].astype(int).values\n'
    '    s  = (df["pred_prob"].fillna(0.5).values if up and "pred_prob" in df\n'
    '          else df["pred_score"].rank(pct=True).values)\n'
    '    if y.sum()==0: continue\n'
    '    fpr,tpr,_ = roc_curve(y,s)\n'
    '    pre,rec,_ = precision_recall_curve(y,s)\n'
    '    auc_v = roc_auc_score(y,s)\n'
    '    pr_v  = average_precision_score(y,s)\n'
    '    ax_roc.plot(fpr,tpr,color=PALETTE[i],lw=2,label=f"{lab} ({auc_v:.3f})")\n'
    '    ax_pr.plot(rec,pre,color=PALETTE[i],lw=2,label=f"{lab} ({pr_v:.3f})")\n'
    'ax_roc.plot([0,1],[0,1],"k--",lw=1)\n'
    'ax_roc.set(xlabel="FPR",ylabel="TPR",title="ROC Curve (mid, full OOS)")\n'
    'ax_roc.legend(fontsize=8)\n'
    'pos_r = float(OOS_DICT["ens_rank4_mid"][tc].mean())\n'
    'ax_pr.axhline(pos_r,color="grey",ls="--",lw=1,label=f"Baseline={pos_r:.3f}")\n'
    'ax_pr.set(xlabel="Recall",ylabel="Precision",title="PR Curve (mid, full OOS)")\n'
    'ax_pr.legend(fontsize=8)\n'
    'plt.suptitle("ROC-AUC & PR-AUC",fontsize=14,fontweight="bold")\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""4.2 AUC / PR-AUC bar comparison"""\n'
    'a_s = CLF_DF["ROC-AUC"].sort_values(ascending=False)\n'
    'p_s = CLF_DF["PR-AUC"].reindex(a_s.index)\n'
    'x = np.arange(len(a_s)); w = 0.38\n'
    'fig, ax = plt.subplots(figsize=(10,5))\n'
    'b1 = ax.bar(x-w/2,a_s.values,w,label="ROC-AUC",color="#1976D2")\n'
    'b2 = ax.bar(x+w/2,p_s.values,w,label="PR-AUC", color="#E64A19")\n'
    'ax.axhline(0.5,color="grey",ls="--",lw=1)\n'
    'ax.set_xticks(x); ax.set_xticklabels(a_s.index,rotation=30,ha="right")\n'
    'ax.set(ylabel="Score",ylim=(0,1.05),title="ROC-AUC vs PR-AUC")\n'
    'ax.legend()\n'
    'for b in list(b1)+list(b2):\n'
    '    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.008,\n'
    '            f"{b.get_height():.3f}",ha="center",fontsize=8)\n'
    'plt.tight_layout(); plt.show()'
))

# ─── Section 5 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 5. Regression Metrics (MAE, MSE) & Predicted vs. Actual Analysis"))

cells.append(code(
    '"""5.1 Compute MAE/MSE from OOS frames"""\n'
    'reg_records = []\n'
    'for mk in ("ens_rank4","lgbm_reg","rf_reg","ridge","lgbm_clf","rf_clf","momentum_126","lstm_reg"):\n'
    '    for hz in ("short","mid","long"):\n'
    '        key = f"{mk}_{hz}"\n'
    '        if key not in OOS_DICT: continue\n'
    '        df = OOS_DICT[key].dropna(subset=[f"fwd_ret_{hz}","pred_score"])\n'
    '        if df.empty: continue\n'
    '        ps = df.groupby("Date")["pred_score"].rank(pct=True)\n'
    '        trk = f"fwd_rankpct_{hz}"\n'
    '        tr  = df[trk] if trk in df.columns else ps*0+0.5\n'
    '        reg_records.append({\n'
    '            "Model":mk,"Horizon":hz,\n'
    '            "MAE (rank)":  mean_absolute_error(tr,ps),\n'
    '            "MSE (rank)":  mean_squared_error(tr,ps),\n'
    '            "RMSE (rank)": mean_squared_error(tr,ps)**0.5,\n'
    '            "MAE (return)": mean_absolute_error(df[f"fwd_ret_{hz}"],df["pred_score"]),\n'
    '            "MSE (return)": mean_squared_error(df[f"fwd_ret_{hz}"],df["pred_score"]),\n'
    '        })\n'
    'REG_DF = pd.DataFrame(reg_records)\n'
    'display(show_styled(REG_DF.pivot_table(index="Model",columns="Horizon",values="MAE (rank)").round(4),\\\n'
    '                    caption="MAE on rank-percentile (lower = better)",cmap="RdYlGn_r",axis=None))'
))

cells.append(code(
    '"""5.2 MAE/MSE bar chart (mid horizon)"""\n'
    'hz = "mid"\n'
    'sub_r = REG_DF[REG_DF["Horizon"]==hz].set_index("Model")\n'
    'fig, axes = plt.subplots(1,2,figsize=(14,5))\n'
    'for ax,(col,ttl) in zip(axes,[\n'
    '    ("MAE (rank)","MAE on Rank-Percentile (lower=better)"),\n'
    '    ("MSE (rank)","MSE on Rank-Percentile (lower=better)"),\n'
    ']):\n'
    '    vals = sub_r[col].sort_values()\n'
    '    clrs = ["#D32F2F" if v>vals.median() else "#388E3C" for v in vals]\n'
    '    ax.barh(vals.index[::-1],vals.values[::-1],color=clrs[::-1])\n'
    '    for p,v in zip(ax.patches,vals.values[::-1]):\n'
    '        ax.text(v+0.001,p.get_y()+p.get_height()/2,f"{v:.4f}",va="center",fontsize=8)\n'
    '    ax.set_title(ttl); ax.set_xlabel(col)\n'
    'plt.suptitle(f"Regression Metrics — {hz} horizon",fontsize=13,fontweight="bold")\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""5.3 Predicted vs Actual & Residual Analysis (Ensemble, mid horizon)"""\n'
    'if "ens_rank4_mid" in OOS_DICT:\n'
    '    df = OOS_DICT["ens_rank4_mid"].dropna(subset=["fwd_rankpct_mid","pred_score"]).copy()\n'
    '    df["pred_rankpct"] = df.groupby("Date")["pred_score"].rank(pct=True)\n'
    '    df["error"] = df["fwd_rankpct_mid"] - df["pred_rankpct"]\n'
    '    fig, axes = plt.subplots(2,2,figsize=(15,11))\n'
    '    # 1. Hexbin density: Predicted vs Actual\n'
    '    ax1 = axes[0,0]\n'
    '    hb = ax1.hexbin(df["pred_rankpct"],df["fwd_rankpct_mid"],gridsize=35,cmap="Blues",mincnt=1)\n'
    '    ax1.plot([0,1],[0,1],"r--",lw=2,label="Perfect Calibration (y = x)")\n'
    '    ax1.set(xlabel="Predicted Rank-Percentile",ylabel="Actual Realized Rank-Percentile",\n'
    '            title="Predicted vs Actual Rank (Hexbin Density)")\n'
    '    ax1.legend(loc="upper left")\n'
    '    plt.colorbar(hb,ax=ax1,label="Count")\n'
    '    # 2. Residual Error Distribution\n'
    '    ax2 = axes[0,1]\n'
    '    ax2.hist(df["error"],bins=50,color="#1976D2",alpha=0.7,density=True,edgecolor="white")\n'
    '    ax2.axvline(0,color="red",ls="--",lw=2,label="Zero Error")\n'
    '    mean_err = df["error"].mean(); std_err = df["error"].std()\n'
    '    ax2.axvline(mean_err,color="orange",ls="-",lw=2,label=f"Mean Error = {mean_err:.3f}")\n'
    '    ax2.set(xlabel="Residual Error (Actual − Predicted)",ylabel="Density",\n'
    '            title=f"Prediction Residual Distribution (Std = {std_err:.3f})")\n'
    '    ax2.legend()\n'
    '    # 3. Decile Calibration: Predicted Decile vs Realized Forward Return\n'
    '    ax3 = axes[1,0]\n'
    '    df["decile"] = pd.qcut(df["pred_rankpct"],10,labels=False)+1\n'
    '    dec_ret = df.groupby("decile")["fwd_ret_mid"].mean()*100\n'
    '    clrs = ["#D32F2F" if r<dec_ret.median() else "#388E3C" for r in dec_ret]\n'
    '    ax3.bar(dec_ret.index,dec_ret.values,color=clrs,alpha=0.85)\n'
    '    ax3.set_xticks(range(1,11))\n'
    '    ax3.set(xlabel="Predicted Rank Decile (1=Lowest, 10=Highest)",\n'
    '            ylabel="Actual Realized Mean Forward Return (%)",\n'
    '            title="Decile Calibration: Predicted Decile vs Realized Return")\n'
    '    for i,v in enumerate(dec_ret):\n'
    '        ax3.text(i+1,v+0.1,f"{v:.2f}%",ha="center",fontsize=8.5)\n'
    '    # 4. Residual Tracking Error Over Time\n'
    '    ax4 = axes[1,1]\n'
    '    daily_err = df.groupby("Date")["error"].mean()\n'
    '    ax4.plot(daily_err.index,daily_err.rolling(63,min_periods=20).mean(),color="#7B1FA2",lw=2,label="63-day Rolling Mean Error")\n'
    '    ax4.axhline(0,color="black",ls="--",lw=1)\n'
    '    ax4.set(xlabel="Date",ylabel="Mean Error (Actual − Predicted)",title="Residual Tracking Error Over Time")\n'
    '    ax4.legend()\n'
    '    plt.suptitle("Ensemble (ens_rank4) — Predicted vs Actual & Residual Analysis (mid horizon)",fontsize=14,fontweight="bold",y=1.01)\n'
    '    plt.tight_layout(); plt.show()'
))

# ─── Section 6 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 6. Overfitting / Underfitting / Generalisation Gap"))

cells.append(code(
    '"""6.1 Dev vs Holdout IC gap table"""\n'
    'gap_rows = []\n'
    'for hz in HORIZONS:\n'
    '    for m in ALL_MODELS:\n'
    '        dev = RESULTS[\n'
    '            (RESULTS["model"]==m)&(RESULTS["horizon"]==hz)&\n'
    '            (RESULTS["split_type"]=="dev")&(RESULTS["metric"]=="rank_ic_mean")\n'
    '        ]["value"].mean()\n'
    '        hold = RESULTS[\n'
    '            (RESULTS["model"]==m)&(RESULTS["horizon"]==hz)&\n'
    '            (RESULTS["split_type"]=="holdout")&(RESULTS["metric"]=="rank_ic_mean")\n'
    '        ]["value"].mean()\n'
    '        if not (dev!=dev): # not nan\n'
    '            gap_rows.append({"Model":m,"Horizon":hz,\n'
    '                             "Dev IC":dev,"Holdout IC":hold,\n'
    '                             "Gap":dev-hold,"Gap%":(dev-hold)/(abs(dev)+1e-9)*100})\n'
    'GAP_DF = pd.DataFrame(gap_rows)\n'
    'mid_gap = GAP_DF[GAP_DF["Horizon"]=="mid"].set_index("Model").sort_values("Gap")\n'
    'mid_gap[["Dev IC","Holdout IC","Gap","Gap%"]].round(4)'
))

cells.append(code(
    '"""6.2 Scatter: Dev IC vs Holdout IC (colour = gap magnitude)"""\n'
    'hz = "mid"\n'
    'sub = GAP_DF[(GAP_DF["Horizon"]==hz)].dropna(subset=["Holdout IC"])\n'
    'fig, ax = plt.subplots(figsize=(9,7))\n'
    'sc = ax.scatter(sub["Dev IC"],sub["Holdout IC"],c=sub["Gap"],\n'
    '                cmap="RdYlGn_r",s=100,zorder=3)\n'
    'plt.colorbar(sc,ax=ax,label="Gap (Dev−Holdout IC)")\n'
    'for _,row in sub.iterrows():\n'
    '    ax.annotate(row["Model"],(row["Dev IC"],row["Holdout IC"]),\n'
    '                fontsize=7,textcoords="offset points",xytext=(4,4))\n'
    'lim = [min(sub["Dev IC"].min(),sub["Holdout IC"].min())-0.01,\n'
    '       max(sub["Dev IC"].max(),sub["Holdout IC"].max())+0.01]\n'
    'ax.plot(lim,lim,"k--",lw=1,label="Perfect generalisation")\n'
    'ax.set(xlabel="Dev IC",ylabel="Holdout IC",\n'
    '       title=f"Dev vs Holdout IC ({hz})\\nPoints below diagonal = overfit")\n'
    'ax.legend(); plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""6.3 IC variance over folds (instability/overfit proxy)"""\n'
    'fig, axes = plt.subplots(1,3,figsize=(17,5))\n'
    'for ax, hz in zip(axes,HORIZONS):\n'
    '    for i,m in enumerate(["ens_rank4","lgbm_reg","ridge","momentum_126"]):\n'
    '        sub = RESULTS[\n'
    '            (RESULTS["model"]==m)&(RESULTS["horizon"]==hz)&\n'
    '            (RESULTS["split_type"]=="dev")&(RESULTS["metric"]=="rank_ic_mean")\n'
    '        ].copy()\n'
    '        sub = sub[sub["fold"].str.startswith("dev_")].copy()\n'
    '        if sub.empty: continue\n'
    '        sub["year"] = sub["fold"].str.extract(r"(\\d{4})")[0].astype(int)\n'
    '        sub = sub.sort_values("year")\n'
    '        ax.plot(sub["year"],sub["value"].rolling(3,min_periods=1).std(),\n'
    '                marker="o",ms=4,label=m,color=PALETTE[i],lw=1.8)\n'
    '    ax.set_title(f"IC Variance — {hz.capitalize()}")\n'
    '    ax.set_xlabel("Test Year"); ax.set_ylabel("IC Rolling Std")\n'
    '    ax.legend(fontsize=8)\n'
    'plt.suptitle("Overfitting Indicator: IC Variance (high = unstable)",\n'
    '             fontsize=13,fontweight="bold",y=1.02)\n'
    'plt.tight_layout(); plt.show()'
))

# ─── Section 7 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 7. Cross-Validation — Walk-Forward IC Summary"))

cells.append(code(
    '"""7.1 Fold summary CSV"""\n'
    'try:\n'
    '    folds_df = pd.read_csv(REP / "folds.csv")\n'
    '    print(f"Fold summary: {folds_df.shape}")\n'
    '    display(folds_df.head(10))\n'
    'except FileNotFoundError:\n'
    '    print("folds.csv not found"); folds_df = pd.DataFrame()'
))

cells.append(code(
    '"""7.2 IC distribution per fold (boxplot)"""\n'
    'ic_pivot = RESULTS[\n'
    '    (RESULTS["horizon"]=="mid")&(RESULTS["split_type"]=="dev")&\n'
    '    (RESULTS["metric"]=="rank_ic_mean")&(RESULTS["model"].isin(PRIMARY_MODELS))\n'
    '].pivot_table(index="fold",columns="model",values="value")\n'
    'ic_pivot = ic_pivot[ic_pivot.index.str.startswith("dev_")].copy()\n'
    'if not ic_pivot.empty:\n'
    '    ic_pivot["yr"] = ic_pivot.index.str.extract(r"(\\d{4})")[0].astype(int)\n'
    '    ic_pivot = ic_pivot.sort_values("yr").drop(columns="yr")\n'
    '    fig, ax = plt.subplots(figsize=(13,6))\n'
    '    ic_pivot.boxplot(ax=ax,vert=True,patch_artist=True,\n'
    '                     boxprops=dict(facecolor="#BBDEFB",alpha=0.7),\n'
    '                     medianprops=dict(color="#1565C0",lw=2))\n'
    '    ax.axhline(0,color="red",ls="--",lw=1)\n'
    '    ax.set_title("IC Distribution across Walk-Forward Folds (mid)")\n'
    '    ax.set_ylabel("Rank IC per Fold")\n'
    '    ax.tick_params(axis="x",rotation=30)\n'
    '    plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""7.3 Model × Fold IC heatmap"""\n'
    'ic_mat = RESULTS[\n'
    '    (RESULTS["horizon"]=="mid")&(RESULTS["split_type"]=="dev")&\n'
    '    (RESULTS["metric"]=="rank_ic_mean")&(RESULTS["model"].isin(PRIMARY_MODELS))\n'
    '].pivot_table(index="model",columns="fold",values="value")\n'
    'col_order = sorted(ic_mat.columns,\n'
    '    key=lambda c: int(c.split("_")[1]) if "_" in c else 0)\n'
    'ic_mat = ic_mat[col_order]\n'
    'fig, ax = plt.subplots(figsize=(14,6))\n'
    'sns.heatmap(ic_mat,annot=True,fmt=".3f",cmap="RdYlGn",center=0,\n'
    '            linewidths=0.4,ax=ax,cbar_kws={"label":"Rank IC"})\n'
    'ax.set_title("Cross-Validation Heatmap: IC per Model × Fold (mid)")\n'
    'ax.tick_params(axis="x",rotation=45)\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""7.4 Fold t-statistic per model"""\n'
    'ts = mean_metric("folds_t_stat","mid","dev")\n'
    'fig, ax = plt.subplots(figsize=(9,5))\n'
    'clrs = ["#1565C0" if v>1.65 else "#FF8F00" if v>1.28 else "#B71C1C" for v in ts.values]\n'
    'ax.barh(ts.index[::-1],ts.values[::-1],color=clrs[::-1])\n'
    'ax.axvline(1.65,color="orange",ls="--",lw=1.5,label="t=1.65 (90%)")\n'
    'ax.axvline(1.96,color="green", ls="--",lw=1.5,label="t=1.96 (95%)")\n'
    'ax.set_title("Fold t-Statistic (mid horizon, dev folds)")\n'
    'ax.set_xlabel("t-stat"); ax.legend(fontsize=9)\n'
    'plt.tight_layout(); plt.show()'
))

# ─── Section 8 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 8. Computational Efficiency"))

cells.append(code(
    '"""8.1 Parquet file sizes as model-complexity proxy"""\n'
    'sz = pd.DataFrame([{"file":p.stem,"size_kb":p.stat().st_size/1024}\n'
    '                   for p in sorted(OOS.glob("*.parquet"))])\\\n'
    '       .sort_values("size_kb",ascending=False)\n'
    'display(sz.head(12))'
))

cells.append(code(
    '"""8.2 Scoring latency benchmark (Spearman IC per date)"""\n'
    'tm_records = []\n'
    'BENCH = [\n'
    '    ("Ensemble","ens_rank4_mid"),("LGBM Reg","lgbm_reg_mid"),\n'
    '    ("RF Reg","rf_reg_mid"),("Ridge","ridge_mid"),\n'
    '    ("LGBM Clf","lgbm_clf_mid"),("Momentum","momentum_126_mid"),\n'
    ']\n'
    'for lab, key in BENCH:\n'
    '    if key not in OOS_DICT: continue\n'
    '    df = OOS_DICT[key]; hz = key.split("_")[-1]\n'
    '    t0 = time.perf_counter(); n = 0\n'
    '    for dt,grp in df.groupby("Date"):\n'
    '        v = grp.dropna(subset=["pred_score",f"fwd_ret_{hz}"])\n'
    '        if len(v)>=4: spearmanr(v["pred_score"],v[f"fwd_ret_{hz}"])\n'
    '        n += 1\n'
    '    ms = (time.perf_counter()-t0)/max(n,1)*1000\n'
    '    tm_records.append({"Model":lab,"N rows":len(df),"N dates":df["Date"].nunique(),"ms/date":ms})\n'
    '\n'
    'TM = pd.DataFrame(tm_records).set_index("Model")\n'
    'display(TM.round(3))\n'
    'fig, ax = plt.subplots(figsize=(9,5))\n'
    'ax.barh(TM.index[::-1],TM["ms/date"].values[::-1],color="#1976D2")\n'
    'ax.set_title("IC Computation Latency (ms per date)")\n'
    'ax.set_xlabel("Milliseconds per date")\n'
    'plt.tight_layout(); plt.show()'
))

# ─── Section 9 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 9. Class Imbalance Analysis"))

cells.append(code(
    '"""9.1 Positive-rate (outperformance) per fold"""\n'
    'imb = []\n'
    'for hz in ("short","mid","long"):\n'
    '    key = f"ens_rank4_{hz}"\n'
    '    if key not in OOS_DICT: continue\n'
    '    df = OOS_DICT[key]; tc = f"outperform_{hz}"\n'
    '    if tc not in df.columns: continue\n'
    '    for fold,grp in df.groupby("fold"):\n'
    '        v = grp[tc].dropna()\n'
    '        if len(v)<10: continue\n'
    '        imb.append({"Horizon":hz,"Fold":fold,"Pos Rate":v.mean(),"N":len(v)})\n'
    '\n'
    'IMB = pd.DataFrame(imb)\n'
    'fig, axes = plt.subplots(1,3,figsize=(15,5),sharey=True)\n'
    'for ax, hz in zip(axes,HORIZONS):\n'
    '    sub = IMB[IMB["Horizon"]==hz].sort_values("Fold")\n'
    '    if sub.empty: continue\n'
    '    ax.bar(range(len(sub)),sub["Pos Rate"],color="#42A5F5",alpha=0.85)\n'
    '    ax.axhline(0.5,color="red",ls="--",lw=1,label="50%")\n'
    '    ax.set_xticks(range(len(sub)))\n'
    '    ax.set_xticklabels(sub["Fold"].str.replace("dev_",""),rotation=45,ha="right",fontsize=7)\n'
    '    ax.set_title(f"Positive Rate — {hz.capitalize()}")\n'
    '    ax.set_ylim(0,1)\n'
    '    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))\n'
    '    ax.legend(fontsize=8)\n'
    'plt.suptitle("Class Imbalance: Outperformance Positive Rate by Fold",fontsize=13,y=1.02)\n'
    'plt.tight_layout(); plt.show()\n'
    'print("Overall positive rate:", IMB.groupby("Horizon")["Pos Rate"].mean().round(4))'
))

cells.append(code(
    '"""9.2 Universe size over time"""\n'
    'if "ens_rank4_mid" in OOS_DICT:\n'
    '    cs = OOS_DICT["ens_rank4_mid"].groupby("Date").size()\n'
    '    fig, ax = plt.subplots(figsize=(12,4))\n'
    '    ax.fill_between(cs.index,cs.values,alpha=0.6,color="#1976D2")\n'
    '    ax.axhline(cs.mean(),color="orange",ls="--",lw=1.5,label=f"Mean={cs.mean():.1f}")\n'
    '    ax.set_title("Universe Size (Stocks per Date)")\n'
    '    ax.set_ylabel("N stocks"); ax.legend()\n'
    '    plt.tight_layout(); plt.show()'
))

# ─── Section 10 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 10. Interpretability — Ridge Coefficients"))

cells.append(code(
    '"""10.1 Load Ridge coefficients"""\n'
    'COEF = pd.read_csv(REP / "04_linear_coefficients.csv")\n'
    'print(f"Shape: {COEF.shape}")\n'
    'display(COEF.head(3))\n'
    'FEATURE_LABELS = {\n'
    '    "log_ret_5":"5d Ret","log_ret_10":"10d Ret","log_ret_21":"21d Ret",\n'
    '    "log_ret_63":"63d Ret","log_ret_126":"126d Ret","log_ret_252":"252d Ret",\n'
    '    "vol_21d":"21d Vol","vol_63d":"63d Vol","downside_vol_63d":"Downside Vol",\n'
    '    "rsi_14":"RSI-14","macd_hist_norm":"MACD Hist","bollinger_pct_b_20":"Bollinger%B",\n'
    '    "amihud_illiq_63d":"Amihud","zero_trade_share_63d":"Zero-Trade%",\n'
    '    "rel_ret_21d":"RelRet 21d","rel_ret_63d":"RelRet 63d",\n'
    '    "cs_rank_mom_63d":"CS Mom Rank","cs_rank_vol_63d":"CS Vol63",\n'
    '    "cs_rank_vol_21d":"CS Vol21","index_ret_63d":"Idx Ret","index_vol_63d":"Idx Vol",\n'
    '}'
))

cells.append(code(
    '"""10.2 Mean |coefficient| per feature"""\n'
    'fig, axes = plt.subplots(1,3,figsize=(18,6))\n'
    'for ax, hz in zip(axes,HORIZONS):\n'
    '    sub = COEF[(COEF["horizon"]==hz)&(COEF["split_type"]=="dev")]\\\n'
    '        .groupby("feature")["coefficient"].mean().abs()\\\n'
    '        .sort_values(ascending=False).head(15)\n'
    '    labs = [FEATURE_LABELS.get(f,f) for f in sub.index]\n'
    '    clrs = ["#1565C0" if "ret" in f else "#E65100" if "vol" in f\n'
    '            else "#2E7D32" if "rank" in f else "#6A1B9A" for f in sub.index]\n'
    '    ax.barh(labs[::-1],sub.values[::-1],color=clrs[::-1])\n'
    '    ax.set_title(f"Ridge |Coef| — {hz.capitalize()}")\n'
    '    ax.set_xlabel("Mean |Coefficient|")\n'
    'plt.suptitle("Ridge Interpretability",fontsize=14,fontweight="bold",y=1.01)\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""10.3 Coefficient stability over folds (mid)"""\n'
    'hz = "mid"\n'
    'sub = COEF[(COEF["horizon"]==hz)&(COEF["split_type"]=="dev")]\n'
    'top_f = sub.groupby("feature")["coefficient"].mean().abs().nlargest(8).index\n'
    'fig, ax = plt.subplots(figsize=(12,6))\n'
    'for i,feat in enumerate(top_f):\n'
    '    fs = sub[sub["feature"]==feat].copy()\n'
    '    fs = fs[fs["fold"].str.contains(r"\\d{4}")].copy()\n'
    '    if fs.empty: continue\n'
    '    fs["year"] = fs["fold"].str.extract(r"(\\d{4})")[0].astype(int)\n'
    '    fs = fs.sort_values("year")\n'
    '    ax.plot(fs["year"],fs["coefficient"],marker="o",ms=4,\n'
    '            label=FEATURE_LABELS.get(feat,feat),color=PALETTE[i],lw=1.8)\n'
    'ax.axhline(0,color="black",lw=0.8,ls="--")\n'
    'ax.set_title(f"Ridge Coefficient Stability — {hz}")\n'
    'ax.set(xlabel="Fold Year",ylabel="Coefficient")\n'
    'ax.legend(fontsize=8,ncol=2)\n'
    'plt.tight_layout(); plt.show()'
))

# ─── Section 11 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 11. Explainability — SHAP Values & Permutation Importance"))

cells.append(code(
    '"""11.1 Load SHAP & permutation data"""\n'
    'SHAP_DF = pd.read_csv(REP / "06_lgbm_shap_values.csv")\n'
    'PERM_DF = pd.read_csv(REP / "06_lgbm_permutation_importance.csv")\n'
    'print("SHAP:",SHAP_DF.shape,"| PERM:",PERM_DF.shape)\n'
    'display(SHAP_DF.head(3))'
))

cells.append(code(
    '"""11.2 Top SHAP features per horizon"""\n'
    'FAM_COLORS = {\n'
    '    "Momentum & Reversal":"#1565C0","Volatility & Risk":"#E65100",\n'
    '    "Liquidity & Turnover":"#2E7D32","Trend & Technical":"#6A1B9A",\n'
    '    "Market & Benchmark":"#00838F",\n'
    '}\n'
    'fig, axes = plt.subplots(1,3,figsize=(18,6))\n'
    'for ax, hz in zip(axes,HORIZONS):\n'
    '    sub = SHAP_DF[SHAP_DF["horizon"]==hz].groupby("feature")["mean_abs_shap"].mean()\n'
    '    if sub.empty: ax.set_title(f"No SHAP for {hz}"); continue\n'
    '    top = sub.nlargest(12).sort_values()\n'
    '    fam_lu = SHAP_DF[SHAP_DF["horizon"]==hz].groupby("feature")["family"].first()\n'
    '    clrs = [FAM_COLORS.get(fam_lu.get(f,""),"#78909C") for f in top.index]\n'
    '    ax.barh([FEATURE_LABELS.get(f,f) for f in top.index],top.values,color=clrs)\n'
    '    ax.set_title(f"Mean |SHAP| — {hz.capitalize()}")\n'
    '    ax.set_xlabel("Mean |SHAP|")\n'
    'pats = [mpatches.Patch(color=c,label=k) for k,c in FAM_COLORS.items()]\n'
    'fig.legend(handles=pats,loc="lower center",ncol=3,fontsize=9,frameon=False)\n'
    'plt.suptitle("SHAP Feature Importance (LGBM)",fontsize=14,fontweight="bold",y=1.01)\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""11.3 SHAP vs Permutation Importance"""\n'
    'hz = "mid"\n'
    'sh = SHAP_DF[SHAP_DF["horizon"]==hz].groupby("feature")["mean_abs_shap"].mean()\n'
    'pm = PERM_DF[PERM_DF["horizon"]==hz].groupby("feature")["importance"].mean()\n'
    'comb = pd.DataFrame({"SHAP":sh,"Perm":pm}).dropna()\n'
    'rho,_ = spearmanr(comb["SHAP"].rank(),comb["Perm"].rank())\n'
    'print(f"Spearman ρ (SHAP vs Permutation): {rho:.3f}")\n'
    'comb["SHAP_n"] = comb["SHAP"]/comb["SHAP"].max()\n'
    'comb["Perm_n"] = comb["Perm"]/comb["Perm"].max()\n'
    'top = comb.nlargest(12,"SHAP")\n'
    'x=np.arange(len(top)); w=0.4\n'
    'fig, ax = plt.subplots(figsize=(14,5))\n'
    'ax.bar(x-w/2,top["SHAP_n"],w,label="SHAP (norm)",color="#1976D2")\n'
    'ax.bar(x+w/2,top["Perm_n"],w,label="Permutation (norm)",color="#E64A19")\n'
    'ax.set_xticks(x)\n'
    'ax.set_xticklabels([FEATURE_LABELS.get(f,f) for f in top.index],rotation=40,ha="right")\n'
    'ax.set_title(f"SHAP vs Permutation Importance (mid) — ρ={rho:.3f}")\n'
    'ax.set_ylabel("Normalised Importance"); ax.legend()\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""11.4 Feature family contribution (SHAP)"""\n'
    'fig, axes = plt.subplots(1,3,figsize=(16,5))\n'
    'for ax, hz in zip(axes,HORIZONS):\n'
    '    sub = SHAP_DF[SHAP_DF["horizon"]==hz]\\\n'
    '        .groupby(["family","feature"])["mean_abs_shap"].mean().reset_index()\n'
    '    if sub.empty: continue\n'
    '    fa = sub.groupby("family")["mean_abs_shap"].sum().sort_values()\n'
    '    clrs = [FAM_COLORS.get(f,"#78909C") for f in fa.index]\n'
    '    ax.barh(fa.index,fa.values,color=clrs)\n'
    '    ax.set_title(f"Family SHAP — {hz.capitalize()}")\n'
    '    ax.set_xlabel("Sum Mean |SHAP|")\n'
    'plt.suptitle("Feature Family Contribution (SHAP)",fontsize=13,y=1.01)\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""11.5 SHAP stability over folds (mid)"""\n'
    'hz = "mid"\n'
    'top_f = SHAP_DF[SHAP_DF["horizon"]==hz]\\\n'
    '    .groupby("feature")["mean_abs_shap"].mean().nlargest(6).index\n'
    'fig, ax = plt.subplots(figsize=(12,6))\n'
    'for i, feat in enumerate(top_f):\n'
    '    fs = SHAP_DF[(SHAP_DF["horizon"]==hz)&(SHAP_DF["feature"]==feat)].copy()\n'
    '    fs = fs[fs["fold"].str.contains(r"\\d{4}", na=False)].copy()\n'
    '    if fs.empty: continue\n'
    '    fs["year"] = fs["fold"].str.extract(r"(\\d{4})")[0].astype(int)\n'
    '    fs = fs.sort_values("year")\n'
    '    ax.plot(fs["year"],fs["mean_abs_shap"],marker="o",ms=4,\n'
    '            label=FEATURE_LABELS.get(feat,feat),color=PALETTE[i],lw=1.8)\n'
    'ax.set_title(f"SHAP Stability over Folds — {hz}")\n'
    'ax.set(xlabel="Fold Year",ylabel="Mean |SHAP|"); ax.legend(fontsize=9)\n'
    'plt.tight_layout(); plt.show()'
))

# ─── Section 12 ───────────────────────────────────────────────────────────────
cells.append(md("---\n## 12. Summary Dashboard"))

cells.append(code(
    '"""12.1 Master comparison table"""\n'
    'sum_rows = []\n'
    'for m in PRIMARY_MODELS:\n'
    '    row = {"Model":m}\n'
    '    for hz in HORIZONS:\n'
    '        for mt,sh in [("rank_ic_mean","IC"),("top_k_precision","Prec"),\n'
    '                      ("portfolio_win_rate","WinR")]:\n'
    '            v = RESULTS[\n'
    '                (RESULTS["model"]==m)&(RESULTS["horizon"]==hz)&\n'
    '                (RESULTS["split_type"]=="dev")&(RESULTS["metric"]==mt)\n'
    '            ]["value"].mean()\n'
    '            row[f"{hz[0].upper()} {sh}"] = v\n'
    '    if m in CLF_DF.index:\n'
    '        row["ROC-AUC"] = CLF_DF.loc[m,"ROC-AUC"]\n'
    '        row["F1"]      = CLF_DF.loc[m,"F1"]\n'
    '    rr = REG_DF[(REG_DF["Model"]==m)&(REG_DF["Horizon"]=="mid")]\n'
    '    if not rr.empty: row["MAE (rank)"] = rr["MAE (rank)"].values[0]\n'
    '    sum_rows.append(row)\n'
    'SUMMARY = pd.DataFrame(sum_rows).set_index("Model")\n'
    'display(show_styled(SUMMARY, caption="Master Summary — S=Short M=Mid L=Long, dev folds"))'
))

cells.append(code(
    '"""12.2 Summary radar"""\n'
    'DM = ["M IC","M Prec","M WinR","S IC","L IC"]\n'
    'dash = SUMMARY[[c for c in DM if c in SUMMARY.columns]].dropna(how="all")\n'
    'def _norm(s):\n'
    '    r=s.max()-s.min()\n'
    '    return (s-s.min())/r if r>0 else s*0+0.5\n'
    'dash_n = dash.apply(_norm)\n'
    'angles = np.linspace(0,2*np.pi,len(dash_n.columns),endpoint=False).tolist()+[0]\n'
    'fig, ax = plt.subplots(figsize=(9,8),subplot_kw=dict(polar=True))\n'
    'for i,(m,row) in enumerate(dash_n.iterrows()):\n'
    '    v = row.values.tolist()+[row.values[0]]\n'
    '    ax.plot(angles,v,"o-",lw=2,label=m,color=PALETTE[i%len(PALETTE)])\n'
    '    ax.fill(angles,v,alpha=0.05,color=PALETTE[i%len(PALETTE)])\n'
    'ax.set_thetagrids(np.degrees(angles[:-1]),dash_n.columns,fontsize=11)\n'
    'ax.set_ylim(0,1)\n'
    'ax.set_title("Model Summary Radar (min-max normalised)",pad=25,fontsize=13)\n'
    'ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.15),fontsize=9)\n'
    'plt.tight_layout(); plt.show()'
))

cells.append(code(
    '"""12.3 Final text summary"""\n'
    'print("="*68)\n'
    'print(" STOCK PULSE — MODEL EVALUATION SUMMARY")\n'
    'print("="*68)\n'
    'if "M IC" in SUMMARY.columns:\n'
    '    best_ic = SUMMARY["M IC"].idxmax()\n'
    '    print(f"\\nBest Rank IC (mid dev) : {best_ic}  ({SUMMARY.loc[best_ic,\'M IC\']:.4f})")\n'
    'if "F1" in SUMMARY.columns and SUMMARY["F1"].notna().any():\n'
    '    bf1 = SUMMARY["F1"].idxmax()\n'
    '    print(f"Best F1 Score          : {bf1}  ({SUMMARY.loc[bf1,\'F1\']:.4f})")\n'
    'if "ROC-AUC" in SUMMARY.columns and SUMMARY["ROC-AUC"].notna().any():\n'
    '    bauc = SUMMARY["ROC-AUC"].idxmax()\n'
    '    print(f"Best ROC-AUC           : {bauc}  ({SUMMARY.loc[bauc,\'ROC-AUC\']:.4f})")\n'
    'print("\\nGeneralisation Gap (Dev IC - Holdout IC, mid):")\n'
    'mg = GAP_DF[GAP_DF["Horizon"]=="mid"].set_index("Model")\n'
    'for m in PRIMARY_MODELS:\n'
    '    if m not in mg.index: continue\n'
    '    g = mg.loc[m,"Gap"]; flag = "⚠️  OVERFIT?" if g>0.05 else "✓"\n'
    '    print(f"  {m:20s}  gap={g:+.4f}  {flag}")\n'
    'ts = SHAP_DF[SHAP_DF["horizon"]=="mid"].groupby("feature")["mean_abs_shap"].mean().idxmax()\n'
    'print(f"\\nTop SHAP Feature (mid) : {FEATURE_LABELS.get(ts,ts)}")\n'
    'print("\\n"+"="*68)'
))

# ── Write ─────────────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)

print(f"Written: {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
