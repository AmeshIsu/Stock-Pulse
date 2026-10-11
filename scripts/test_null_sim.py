import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.data.config_loader import load_config
from src.models.validation import build_folds
from src.models.preprocessing import REDUNDANT_LINEAR_FEATURES, apply_cross_sectional_rank

config = load_config()
df = pd.read_parquet('data/processed/features_targets.parquet')
mask = pd.read_parquet('data/processed/rankable_mask.parquet')['rankable'].values
df_u = df[df['in_ranking_universe'] & mask].copy()

all_folds = build_folds(df_u, config)

def run_dual_null_test(horizon='short', split_type='dev', n_draws=1000, seed=42):
    t0 = time.time()
    h_days = config['horizons'][horizon]
    target_col = f'fwd_excess_{horizon}'
    ret_col = f'fwd_ret_{horizon}'
    
    h_folds = [f for f in all_folds if f.horizon == horizon and f.split_type == split_type]
    test_dfs = []
    for f in h_folds:
        tdf = df_u.loc[f.test_indices].copy()
        tdf['fold_name'] = f.fold_name
        test_dfs.append(tdf)
    part_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    if part_df.empty:
        return None
        
    core_feats = [c for c in config['feature_sets']['core'] if c not in REDUNDANT_LINEAR_FEATURES and c not in ['index_ret_63d', 'index_vol_63d']]
    part_df_ranked = apply_cross_sectional_rank(part_df, core_feats)
    
    # Pre-extract fold and daily structures
    fold_groups = []
    for fold_name, f_df in part_df_ranked.groupby('fold_name', sort=False):
        dates = sorted(f_df['Date'].unique())
        # Break into h-day blocks (segment breaks reset blocks)
        date_to_block = {}
        for b_idx, start_idx in enumerate(range(0, len(dates), h_days)):
            b_dates = dates[start_idx : start_idx + h_days]
            for d in b_dates:
                date_to_block[d] = b_idx
        n_blocks = len(range(0, len(dates), h_days))
        
        unique_series = sorted(f_df['series_id'].unique())
        series_to_idx = {s: i for i, s in enumerate(unique_series)}
        
        daily_slices = []
        for dt, d_grp in f_df.groupby('Date', sort=False):
            valid = d_grp.dropna(subset=[target_col, ret_col])
            if len(valid) < 8:
                continue
            s_indices = np.array([series_to_idx[s] for s in valid['series_id']], dtype=int)
            tgt_ranks = valid[target_col].rank().values
            tgt_c = tgt_ranks - tgt_ranks.mean()
            norm_c = np.linalg.norm(tgt_c)
            tgt_norm = tgt_c / norm_c if norm_c > 1e-9 else tgt_c
            X_mat = valid[core_feats].values # shape (n_stocks, p)
            daily_slices.append({
                'Date': dt,
                'block_idx': date_to_block[dt],
                's_indices': s_indices,
                'tgt_norm': tgt_norm,
                'X_mat': X_mat,
                'n_stocks': len(valid)
            })
            
        fold_groups.append({
            'fold_name': fold_name,
            'n_series': len(unique_series),
            'n_blocks': n_blocks,
            'daily_slices': daily_slices
        })
        
    p_feats = len(core_feats)
    rng_a = np.random.default_rng(seed + 101)
    rng_b = np.random.default_rng(seed + 202)
    rng_old = np.random.default_rng(seed + 303)
    
    daily_ics_a_list = []
    daily_ics_b_list = []
    daily_ics_old_list = []
    
    for f_idx, fg in enumerate(fold_groups):
        if not fg['daily_slices']:
            continue
        # (a) Block-constant random scores: shape (n_draws, n_blocks, n_series)
        block_scores = rng_a.uniform(0.0, 1.0, size=(n_draws, fg['n_blocks'], fg['n_series']))
        # (b) Random linear weights for this fold: shape (n_draws, p_feats)
        w_fold = rng_b.standard_normal(size=(n_draws, p_feats))
        
        for d_slice in fg['daily_slices']:
            tgt_norm = d_slice['tgt_norm']
            n_stk = d_slice['n_stocks']
            b_idx = d_slice['block_idx']
            s_idx = d_slice['s_indices']
            
            # (a) Block constant
            scores_a = block_scores[:, b_idx, :][:, s_idx] # (n_draws, n_stk)
            perms_a = scores_a.argsort(axis=1)
            ranks_a = perms_a.argsort(axis=1).astype(float) + 1.0
            ra_c = ranks_a - ranks_a.mean(axis=1, keepdims=True)
            norm_a = np.linalg.norm(ra_c, axis=1, keepdims=True)
            ra_norm = ra_c / np.where(norm_a > 1e-9, norm_a, 1.0)
            corrs_a = np.dot(ra_norm, tgt_norm)
            daily_ics_a_list.append(corrs_a)
            
            # (b) Random linear
            scores_b = np.dot(w_fold, d_slice['X_mat'].T) # (n_draws, n_stk)
            perms_b = scores_b.argsort(axis=1)
            ranks_b = perms_b.argsort(axis=1).astype(float) + 1.0
            rb_c = ranks_b - ranks_b.mean(axis=1, keepdims=True)
            norm_b = np.linalg.norm(rb_c, axis=1, keepdims=True)
            rb_norm = rb_c / np.where(norm_b > 1e-9, norm_b, 1.0)
            corrs_b = np.dot(rb_norm, tgt_norm)
            daily_ics_b_list.append(corrs_b)
            
            # (c) Old null: fresh i.i.d.
            scores_old = rng_old.uniform(0.0, 1.0, size=(n_draws, n_stk))
            perms_old = scores_old.argsort(axis=1)
            ranks_old = perms_old.argsort(axis=1).astype(float) + 1.0
            ro_c = ranks_old - ranks_old.mean(axis=1, keepdims=True)
            norm_old = np.linalg.norm(ro_c, axis=1, keepdims=True)
            ro_norm = ro_c / np.where(norm_old > 1e-9, norm_old, 1.0)
            corrs_old = np.dot(ro_norm, tgt_norm)
            daily_ics_old_list.append(corrs_old)
            
    draw_ics_a = np.mean(daily_ics_a_list, axis=0) # (n_draws,)
    draw_ics_b = np.mean(daily_ics_b_list, axis=0)
    draw_ics_old = np.mean(daily_ics_old_list, axis=0)
    
    pct25_a, pct975_a = np.percentile(draw_ics_a, [2.5, 97.5])
    pct25_b, pct975_b = np.percentile(draw_ics_b, [2.5, 97.5])
    pct25_old, pct975_old = np.percentile(draw_ics_old, [2.5, 97.5])
    
    width_a = pct975_a - pct25_a
    width_b = pct975_b - pct25_b
    width_old = pct975_old - pct25_old
    
    if width_b >= width_a:
        wider_type = 'random_linear'
        corr_low, corr_high = pct25_b, pct975_b
        wider_draws = draw_ics_b
    else:
        wider_type = 'block_constant'
        corr_low, corr_high = pct25_a, pct975_a
        wider_draws = draw_ics_a
        
    corr_width = corr_high - corr_low
    
    elapsed = time.time() - t0
    print(f"[{horizon.upper()} {split_type.upper()}] Elapsed: {elapsed:.2f}s")
    print(f"  Old Band:         [{pct25_old:+.4f}, {pct975_old:+.4f}] (width: {width_old:.4f})")
    print(f"  (a) Block Band:   [{pct25_a:+.4f}, {pct975_a:+.4f}] (width: {width_a:.4f}, {width_a/width_old:.2f}x old)")
    print(f"  (b) Linear Band:  [{pct25_b:+.4f}, {pct975_b:+.4f}] (width: {width_b:.4f}, {width_b/width_old:.2f}x old)")
    print(f"  => Corrected ({wider_type}): [{corr_low:+.4f}, {corr_high:+.4f}] (width: {corr_width:.4f}, {corr_width/width_old:.2f}x old)")
    
    return {
        'horizon': horizon,
        'split_type': split_type,
        'old_lower': pct25_old, 'old_upper': pct975_old, 'old_width': width_old,
        'block_lower': pct25_a, 'block_upper': pct975_a, 'block_width': width_a,
        'linear_lower': pct25_b, 'linear_upper': pct975_b, 'linear_width': width_b,
        'corrected_lower': corr_low, 'corrected_upper': corr_high, 'corrected_width': corr_width,
        'wider_type': wider_type,
        'draw_ics': wider_draws
    }

if __name__ == '__main__':
    for h in ['short', 'mid', 'long']:
        for s in ['dev', 'holdout']:
            run_dual_null_test(horizon=h, split_type=s)
