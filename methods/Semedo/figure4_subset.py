from __future__ import annotations
import numpy as np 
from core.runtime import runtime
from datetime import datetime
import time
from ..rrr import RRRAnalyzer
from ..visualization import SemedoFigures, d95_from_curves
from .utils import build_trial_matrix
from .matching import match_subset_from_prebuilt

def build_figure_4_subset(
    source_region : int = 1,
    target_region : int = 3,
    *,
    analysis_type : str  = "residual",
    d_max         : int  = 35,
    alpha         : float | None = None,
    outer_splits  : int  = 3,
    inner_splits  : int  = 3,
    random_state  : int  = 0,
    n_runs     : int = 5,
    n_src      : int = 113,
    n_tgt      : int = 28,
    k_subsets  : int | None = 10,
    force_recompute: bool = False,
):
    """
    Generate Subset Semedo figure (Figure 4 SUBSET - Multi-run analysis).
    Conducts random electrode subset sampling per run and evaluates performance.
    Supports caching via .npz file.
    """
    def _log(msg): print(f"[SubsetSemedo][{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    # 1. Setup paths and config
    cfg = runtime.get_cfg()
    tgt_nm = runtime.get_consts().REGION_ID_TO_NAME[target_region]
    out_dir = cfg.get_data_path() / "Semedo_plots" / "Figure_4_subset" / analysis_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine effective counts first for filename
    rois = cfg.get_rois()
    all_src_idx = np.where(rois == source_region)[0]
    all_tgt_idx = np.where(rois == target_region)[0]
    n_src_eff = min(n_src, all_src_idx.size)
    n_tgt_eff = min(n_tgt, all_tgt_idx.size)

    base_name  = (
        f"{cfg.get_monkey_name().replace(' ', '')}_Figure_4_subset_"
        f"src{n_src_eff}_tgt{n_tgt_eff}_runs{n_runs}_"
        f"{analysis_type}_V1_to_{tgt_nm}"
    )
    npz_path = out_dir / f"{base_name}.npz"

    base_colors = ["#9C1C1C", "#1565C0", "#2E7D32", "#7B1FA2", "#F57C00", "#00796B", "#5D4037"]
    colors = [base_colors[i % len(base_colors)] for i in range(n_runs)]

    # --- Caching Check ---
    if not force_recompute and npz_path.exists():
        print(f"[Cache] Found existing data: {npz_path}")
        print(f"[Cache] Loading data and plotting...")
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                runs_full = data["runs_full"].tolist()
                runs_match = data["runs_match"].tolist()
                d95_full_runs = data["d95_full_runs"].tolist()
                d95_match_runs = data["d95_match_runs"].tolist()
                d95_full_sub_all = data["d95_full_sub_all"].tolist()
                d95_match_sub_all = data["d95_match_sub_all"].tolist()
            
            png_path = npz_path.with_suffix(".png")
            SemedoFigures.plot_figure_4_subset(
                runs_full, runs_match, d95_full_runs, d95_match_runs,
                d95_full_sub_all, d95_match_sub_all, target_region, analysis_type,
                n_src_eff, n_tgt_eff, n_runs, k_subsets, outer_splits, inner_splits, d_max, random_state, colors,
                save_path=str(png_path)
            )
            print(f"[✓] Figure regenerated from cache.")
            return
        except Exception as e:
            print(f"[Cache] Error loading cache: {e}. Recomputing...")

    # --- Computation ---
    if n_src_eff < 2 or n_tgt_eff < 1:
        raise ValueError("Not enough electrodes for requested subsets.")

    trials   = runtime.get_data_manager()._load_trials()
    all_idxs = np.arange(len(trials))
    
    runs_full, runs_match = [], []
    d95_full_runs, d95_match_runs = [], []
    d95_full_sub_all, d95_match_sub_all = [], []

    used_src, used_tgt = set(), set()

    # 2. Run Loop
    for run_idx in range(n_runs):
        rng = np.random.default_rng(random_state + run_idx)
        col = colors[run_idx]

        # A. Sample Electrodes
        rem_src = [e for e in all_src_idx if e not in used_src]
        rem_tgt = [e for e in all_tgt_idx if e not in used_tgt]
        
        def pick_subset(pool, remaining, n):
            if len(remaining) >= n:
                chosen = rng.choice(remaining, n, replace=False)
            else:
                others = [x for x in pool if x not in remaining]
                fill = rng.choice(others, n - len(remaining), replace=False)
                chosen = np.concatenate([remaining, fill])
            return chosen

        src_subset = pick_subset(all_src_idx, rem_src, n_src_eff)
        tgt_subset = pick_subset(all_tgt_idx, rem_tgt, n_tgt_eff)
        
        used_src.update(src_subset)
        used_tgt.update(tgt_subset)
        _log(f"Run {run_idx+1}: V1-sub={len(src_subset)}, {tgt_nm}-sub={len(tgt_subset)}")

        # B. Build Matrices (All Trials)
        X_src_all = build_trial_matrix(region_id=source_region, analysis_type=analysis_type, trials=None, electrode_indices=src_subset)
        Y_tgt_all = build_trial_matrix(region_id=target_region, analysis_type=analysis_type, trials=None, electrode_indices=tgt_subset)

        # C. Match Splitting
        match_loc, remain_loc, _, _ = match_subset_from_prebuilt(
            X_src_all, Y_tgt_all, src_subset, trial_idx=None, n_bins=20, seed=random_state+run_idx
        )

        Y_full, X_full = Y_tgt_all, X_src_all
        Y_match, X_match = X_src_all[:, match_loc], X_src_all[:, remain_loc]

        # D. Evaluate Global Performance (Panels A/B)
        res_full = RRRAnalyzer._performance_from_mats(
            Y_full, X_full, d_max=d_max, alpha=alpha, 
            outer_splits=outer_splits, inner_splits=inner_splits, random_state=random_state+run_idx
        )
        res_match = RRRAnalyzer._performance_from_mats(
            Y_match, X_match, d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits, random_state=random_state+run_idx
        )
        
        d95_f = int(d95_from_curves(res_full["rrr_R2_mean"], res_full["ridge_R2_mean"], d_max))
        d95_m = int(d95_from_curves(res_match["rrr_R2_mean"], res_match["ridge_R2_mean"], d_max))
        d95_full_runs.append(d95_f)
        d95_match_runs.append(d95_m)
        
        runs_full.append({**res_full, "rrr": res_full["rrr_R2_mean"], "sem": res_full["rrr_R2_sem"], "ridge": res_full["ridge_R2_mean"], "d95": d95_f, "color": col})
        runs_match.append({**res_match, "rrr": res_match["rrr_R2_mean"], "sem": res_match["rrr_R2_sem"], "ridge": res_match["ridge_R2_mean"], "d95": d95_m, "color": col})

        # E. Subset Analysis (Panel D)
        if k_subsets is None:
            # Repetitions
            rep_arr = np.array([int(tr["allmat_row"][3]) - 1 for tr in trials])
            groups = [np.flatnonzero(rep_arr == g) for g in np.unique(rep_arr)]
        else:
            # Random subsets
            rng_sub = np.random.default_rng(random_state + run_idx * 100)
            perm = rng_sub.permutation(len(trials))
            sz = len(trials) // k_subsets
            groups = [perm[i*sz : (i+1)*sz] for i in range(k_subsets)]

        for g_idx in groups:
            Y_sub, X_sub = Y_match[g_idx, :], X_match[g_idx, :]
            
            d_f = int(d95_from_curves(
                RRRAnalyzer._performance_from_mats(
                    Y_full[g_idx,:], X_full[g_idx,:], d_max=d_max, alpha=alpha, 
                    outer_splits=outer_splits, inner_splits=inner_splits, random_state=random_state+run_idx
                )["rrr_R2_mean"], 
                RRRAnalyzer._performance_from_mats(
                    Y_full[g_idx,:], X_full[g_idx,:], d_max=d_max, alpha=alpha, 
                    outer_splits=outer_splits, inner_splits=inner_splits, random_state=random_state+run_idx
                )["ridge_R2_mean"], 
                d_max
            ))
            
            d_m = int(d95_from_curves(
                RRRAnalyzer._performance_from_mats(
                    Y_sub, X_sub, d_max=d_max, alpha=alpha, 
                    outer_splits=outer_splits, inner_splits=inner_splits, random_state=random_state+run_idx
                )["rrr_R2_mean"], 
                RRRAnalyzer._performance_from_mats(
                    Y_sub, X_sub, d_max=d_max, alpha=alpha, 
                    outer_splits=outer_splits, inner_splits=inner_splits, random_state=random_state+run_idx
                )["ridge_R2_mean"], 
                d_max
            ))

            d95_full_sub_all.append((run_idx, d_f))
            d95_match_sub_all.append((run_idx, d_m))

    # --- Save to Cache ---
    np.savez(
        npz_path,
        runs_full=runs_full,
        runs_match=runs_match,
        d95_full_runs=d95_full_runs,
        d95_match_runs=d95_match_runs,
        d95_full_sub_all=d95_full_sub_all,
        d95_match_sub_all=d95_match_sub_all,
        n_src_eff=n_src_eff,
        n_tgt_eff=n_tgt_eff
    )
    print(f"[✓] Data cached → {npz_path}")

    # 3. Plot
    png_path = npz_path.with_suffix(".png")
    SemedoFigures.plot_figure_4_subset(
        runs_full, runs_match, d95_full_runs, d95_match_runs,
        d95_full_sub_all, d95_match_sub_all, target_region, analysis_type,
        n_src_eff, n_tgt_eff, n_runs, k_subsets, outer_splits, inner_splits, d_max, random_state, colors,
        save_path=str(png_path)
    )
