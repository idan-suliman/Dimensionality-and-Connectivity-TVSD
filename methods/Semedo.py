# Semedo.py
# ----------------
# Main entry point for generating Semedo 2019 related figures (Standard 5A, Subset, and 5B).
# Orchestrates data loading, analysis (RRR/PCA), and calls visualization functions.
# ----------------

from .rrr import RRRAnalyzer
from core.runtime import runtime
from .matchingSubset import MATCHINGSUBSET
import numpy as np
import matplotlib.pyplot as plt    
from matplotlib import gridspec
from itertools import cycle
import matplotlib.patheffects as pe
import time
from datetime import datetime
from pathlib import Path
import csv 

# Visualization imports
from .visualization import (
    d95_from_curves, 
    jitter, 
    square_limits, 
    labeled_dot, 
    SemedoFigures,
    smart_label
)
from .pca import RegionPCA

# Alias for RRR performance function
perf_wrapper = RRRAnalyzer.performance


# =============================================================================
# Helper Functions & Wrappers
# =============================================================================

def build_trial_matrix(*args, **kwargs):
    """
    Local wrapper for runtime.get_data_manager().build_trial_matrix.
    Ensures legacy calls in this file work without verbose imports.
    """
    return runtime.get_data_manager().build_trial_matrix(*args, **kwargs)


def get_match_subset_indices(source_region: int, target_region: int, *, analysis_type: str) -> np.ndarray:
    """
    Load (or compute if missing) the physical electrode indices for V1 that match
    the target region's firing rate distribution.
    File: TARGET_RRR/<ANALYSIS>/V1_to_<TARGET>_<ANALYSIS>.npz
    """
    consts = runtime.get_consts()
    cfg = runtime.get_cfg()

    target_name = consts.REGION_ID_TO_NAME[target_region]
    match_path = (
        cfg.get_data_path() / "TARGET_RRR" / analysis_type.upper() /
        f"V1_to_{target_name}_{analysis_type}.npz"
    )
    if not match_path.exists():
        MATCHINGSUBSET.match_and_save(
            "V1", target_name,
            stat_mode=analysis_type, show_plot=False, verbose=False
        )
    data = np.load(match_path)
    return data["phys_idx"]


def build_groups_by_rep_or_subsets(trials: list[dict], *, k_subsets: int | None, random_state: int):
    """
    Partition trials into groups:
    - If k_subsets is None: Group by repetition index (0..29).
    - If k_subsets is Int:  Partition randomly into K subsets.
    
    Returns:
        groups (list[np.ndarray]): List of trial indices per group.
        label_D (str): Description string for the grouping.
        id_print (func): Helper to format group ID for printing.
        rep_arr (np.ndarray): Array of repetition IDs for all trials.
    """
    rep_arr = np.array([int(tr["allmat_row"][3]) - 1 for tr in trials], dtype=int)

    if k_subsets is None:
        rep_ids = np.unique(rep_arr)
        groups = [np.flatnonzero(rep_arr == g) for g in rep_ids]
        label_D = "Repetitions"
        id_print = (lambda g: f"rep {g:2}")
    else:
        rng = np.random.default_rng(random_state)
        idxs = np.arange(len(trials))
        rng.shuffle(idxs)
        splits = np.array_split(idxs, k_subsets)
        groups = [np.asarray(part, dtype=int) for part in splits]
        label_D = f"{k_subsets} random subsets"
        id_print = (lambda g: f"set {g:2}")

    return groups, label_D, id_print, rep_arr


def match_subset_from_prebuilt(
    X_src_all: np.ndarray,
    Y_tgt_all: np.ndarray,
    src_subset_phys: np.ndarray,
    *,
    trial_idx: np.ndarray | None,
    n_bins: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute V1-MATCH sub-selection efficiently using pre-built data matrices.
    Matches the firing rate distribution of Y_tgt to X_src subsets.
    
    Returns:
        match_loc (np.ndarray): Local column indices for Match electrodes.
        remain_loc (np.ndarray): Local column indices for Remaining electrodes.
        match_phys (np.ndarray): Physical IDs of Match electrodes.
        remain_phys (np.ndarray): Physical IDs of Remaining electrodes.
    """
    rng = np.random.default_rng(seed)

    # Slice rows (trials)
    if trial_idx is None:
        Xs, Xt = X_src_all, Y_tgt_all
    else:
        Xs, Xt = X_src_all[trial_idx, :], Y_tgt_all[trial_idx, :]

    # Calculate means
    ms = Xs.mean(axis=0)
    mt = Xt.mean(axis=0)
    n_src_sub, n_tgt_sub = Xs.shape[1], Xt.shape[1]

    # Shared bin edges
    vmin = float(min(ms.min(), mt.min())) if (ms.size and mt.size) else 0.0
    vmax = float(max(ms.max(), mt.max())) if (ms.size and mt.size) else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(vmin), float(vmin + 1e-6)

    edges = np.linspace(vmin, vmax, int(n_bins) + 1)
    src_bin_of = np.clip(np.searchsorted(edges, ms, side="right") - 1, 0, n_bins - 1)
    tgt_counts = [int(((mt >= edges[b]) & (mt < edges[b + 1])).sum()) for b in range(n_bins)]

    picked_local = []
    surplus_bins = [[] for _ in range(n_bins)]
    deficits = np.zeros(n_bins, dtype=int)

    # Pass 1: Local satisfaction
    for b in range(n_bins):
        s_idxs = np.where(src_bin_of == b)[0]
        need = tgt_counts[b]
        if need == 0:
            surplus_bins[b] = list(s_idxs)
            continue
        if s_idxs.size <= need:
            picked_local.extend(s_idxs.tolist())
            deficits[b] = need - s_idxs.size
        else:
            chosen = rng.choice(s_idxs, size=need, replace=False)
            picked_local.extend(chosen.tolist())
            surplus_bins[b] = [ix for ix in s_idxs if ix not in set(chosen.tolist())]

    # Pass 2: Fill deficits from neighbors
    for b in np.where(deficits > 0)[0]:
        deficit = int(deficits[b])
        dist = 1
        while deficit > 0 and dist < n_bins:
            for nb in (b - dist, b + dist):
                if 0 <= nb < n_bins and surplus_bins[nb]:
                    take = min(len(surplus_bins[nb]), deficit)
                    chosen = rng.choice(surplus_bins[nb], size=take, replace=False)
                    picked_local.extend(chosen.tolist())
                    surplus_bins[nb] = [ix for ix in surplus_bins[nb] if ix not in set(chosen.tolist())]
                    deficit -= take
                    if deficit == 0: break
            dist += 1
        
        # Fallback: global fill
        if deficit > 0:
            remaining = [ix for pool in surplus_bins for ix in pool]
            if remaining:
                take = min(deficit, len(remaining))
                chosen = rng.choice(remaining, size=take, replace=False)
                picked_local.extend(chosen.tolist())
                # Cleanup surplus lists (slow but robust)
                cset = set(chosen.tolist())
                for nb in range(n_bins):
                    if surplus_bins[nb]:
                        surplus_bins[nb] = [ix for ix in surplus_bins[nb] if ix not in cset]

    # Unique and Cap
    picked_local = sorted(set(picked_local))[:n_tgt_sub]
    picked_local = np.asarray(picked_local, dtype=int)

    match_loc  = picked_local
    remain_loc = np.asarray([i for i in range(n_src_sub) if i not in set(match_loc.tolist())], dtype=int)
    
    match_phys  = np.asarray(src_subset_phys, dtype=int)[match_loc]
    remain_phys = np.asarray(src_subset_phys, dtype=int)[remain_loc]

    if remain_loc.size == 0:
        raise ValueError("All V1 electrodes were matched; nothing left for X in MATCH model.")

    return match_loc, remain_loc, match_phys, remain_phys


def match_subset_for_trials_and_electrodes(
    source_region: int,
    target_region: int,
    *,
    analysis_type: str,
    trial_idx: np.ndarray | slice | None,
    src_subset_phys: np.ndarray,
    tgt_subset_phys: np.ndarray,
    n_bins: int = 20,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match V1 subset to Target subset firing rates *dynamically* on specific trials.
    (Similar to prebuilt matcher but builds matrices on demand).
    """
    rng = np.random.default_rng(seed)

    Xs = build_trial_matrix(source_region, analysis_type, trial_idx=trial_idx, electrode_indices=src_subset_phys)
    Xt = build_trial_matrix(target_region, analysis_type, trial_idx=trial_idx, electrode_indices=tgt_subset_phys)
    
    # Delegate logic could be unified, but keeping implementation inline for now to ensure strict equivalence
    # with original logic (binning & matching).
    # ... (Simplified here by calling the prebuilt logic if possible? Yes.)
    
    # We can reuse the prebuilt logic since we have Xs and Xt
    match_loc, remain_loc, _, _ = match_subset_from_prebuilt(
        Xs, Xt, src_subset_phys, trial_idx=None, n_bins=n_bins, seed=seed
    )
    
    # Map back to physical
    match_src_phys = src_subset_phys[match_loc]
    src_remain_phys = src_subset_phys[remain_loc]

    if verbose:
        print(f"[match] V1-subset={len(src_subset_phys)} → match={len(match_src_phys)}")

    return match_src_phys, src_remain_phys


# =============================================================================
# Main Analysis Functions
# =============================================================================

def build_figure_4(
    source_region: int = 1,          # V1
    target_region: int = 3,          # IT (or V4 = 2)
    *,
    analysis_type: str = "residual", 
    d_max: int = 35,
    alpha: float | None = None,
    outer_splits: int = 3,
    inner_splits: int = 3,
    random_state: int = 0,
    k_subsets: int = 10,
    force_recompute: bool = False,
):
    """
    Generate Standard Semedo-style multi-panel figure (Figure 4).
    Calculates RRR and dimensionalities for Full and Match models.
    Supports caching via .npz file (same base name as image).
    """
    cfg = runtime.get_cfg()
    tgt_name = runtime.get_consts().REGION_ID_TO_NAME[target_region]
    
    # Construct paths
    out_dir = runtime.get_cfg().get_data_path() / "Semedo_plots" / "Figure_4"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"sub{k_subsets}" if k_subsets else "rep30"
    base_name  = f"{runtime.get_cfg().get_monkey_name().replace(' ', '')}_Figure_4_{suffix}_{analysis_type}_V1_to_{tgt_name}"
    npz_path = out_dir / f"{base_name}.npz"
    # Note: plot_semedo_figure constructs its own path, but we can verify consistency or ignore it.
    # Actually, plot_semedo_figure uses `fname = ...` internally. 
    # To ensure consistency, we should ideally pass the path or handle saving inside plot_semedo_figure.
    # However, `plot_semedo_figure` is imported from visualization. 
    # Let's trust that identical params produce identical filenames in both places (which they seem to).
    
    # --- Caching Check ---
    if not force_recompute and npz_path.exists():
        print(f"[Cache] Found existing data: {npz_path}")
        print(f"[Cache] Loading data and plotting...")
        
        with np.load(npz_path, allow_pickle=True) as data:
            # Reconstruct dictionaries from keys starting with prefix
            # This requires careful saving/loading or storing dicts as objects (allow_pickle=True)
            # Storing as object arrays is easiest for dictionaries.
            perf_full = data["perf_full"].item()
            perf_match = data["perf_match"].item()
            d95_full_g = int(data["d95_full_g"])
            d95_match_g = int(data["d95_match_g"])
            d95_full_rep = data["d95_full_rep"].tolist()
            d95_match_rep = data["d95_match_rep"].tolist()
            label_D = str(data["label_D"])
            # verify consistency
            if int(data["d_max"]) != d_max:
                 print(f"[Warning] Cached d_max ({data['d_max']}) != requested ({d_max})")

        # Use derived PNG path for checking or re-plotting?
        # User wants the path passed.
        png_path = npz_path.with_suffix(".png")
        SemedoFigures.plot_figure_4(
            perf_full, perf_match, d95_full_g, d95_match_g,
            d95_full_rep, d95_match_rep, d_max, target_region, analysis_type,
            k_subsets, outer_splits, inner_splits, random_state, label_D,
            save_path=str(png_path)
        )
        print(f"[✓] Figure regenerated from cache.")
        return

    # --- 1. Load Data ---
    trials = runtime.get_data_manager()._load_trials()
    groups, label_D, id_print, rep_arr = build_groups_by_rep_or_subsets(
        trials, k_subsets=k_subsets, random_state=random_state
    )

    # --- 2. Global Performance (Panels A & B) ---
    perf_full  = perf_wrapper(
        source_region, target_region, analysis_type=analysis_type, match_to_target=False,
        d_max=d_max, alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=random_state
    )
    perf_match = perf_wrapper(
        source_region, target_region, analysis_type=analysis_type, match_to_target=True,
        d_max=d_max, alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=random_state
    )

    d95_full_g  = d95_from_curves(perf_full ["rrr_R2_mean"], perf_full ["ridge_R2_mean"], d_max)
    d95_match_g = d95_from_curves(perf_match["rrr_R2_mean"], perf_match["ridge_R2_mean"], d_max)

    # --- 3. Per-Group Performance (Panel D) ---
    d95_full_rep, d95_match_rep = [], []

    for g_idx, grp_idx in enumerate(groups):
        res_f = perf_wrapper(
            source_region, target_region, analysis_type=analysis_type, match_to_target=False,
            d_max=d_max, trial_subset=grp_idx, outer_splits=outer_splits, inner_splits=inner_splits, random_state=random_state
        )
        res_m = perf_wrapper(
            source_region, target_region, analysis_type=analysis_type, match_to_target=True,
            d_max=d_max, trial_subset=grp_idx, outer_splits=outer_splits, inner_splits=inner_splits, random_state=random_state
        )
        
        df = d95_from_curves(res_f["rrr_R2_mean"], res_f["ridge_R2_mean"], d_max)
        dm = d95_from_curves(res_m["rrr_R2_mean"], res_m["ridge_R2_mean"], d_max)
        
        d95_full_rep.append(df)
        d95_match_rep.append(dm)
        print(f"[{id_print(g_idx)}] d95_full={df}  d95_match={dm}")

    # --- Save to Cache ---
    np.savez(
        npz_path,
        perf_full=perf_full,
        perf_match=perf_match,
        d95_full_g=d95_full_g,
        d95_match_g=d95_match_g,
        d95_full_rep=d95_full_rep,
        d95_match_rep=d95_match_rep,
        d_max=d_max,
        label_D=label_D,
        analysis_type=analysis_type
    )
    print(f"[✓] Data cached → {npz_path}")

    # --- 4. Plot ---
    png_path = npz_path.with_suffix(".png")
    SemedoFigures.plot_figure_4(
        perf_full, perf_match, d95_full_g, d95_match_g,
        d95_full_rep, d95_match_rep, d_max, target_region, analysis_type,
        k_subsets, outer_splits, inner_splits, random_state, label_D,
        save_path=str(png_path)
    )


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
                # n_src_eff/tgt stored in cache should match, but we use calculated ones for consistency/filename
                # n_src_eff = int(data["n_src_eff"]) 
                # n_tgt_eff = int(data["n_tgt_eff"])
            
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
        t_start = time.perf_counter()
        rng = np.random.default_rng(random_state + run_idx)
        col = colors[run_idx]

        # A. Sample Electrodes
        rem_src = [e for e in all_src_idx if e not in used_src]
        rem_tgt = [e for e in all_tgt_idx if e not in used_tgt]
        
        # Helper to pick subset
        def pick_subset(pool, remaining, n):
            if len(remaining) >= n:
                chosen = rng.choice(remaining, n, replace=False)
            else:
                fill = rng.choice(pool, n - len(remaining), replace=False)
                # Ensure fill doesn't duplicate existing `remaining`
                # (Simple way: just sample N from total pool, preferring remaining? 
                #  Logic in original was strict non-replacement from fill. Adapting slightly for clarity)
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
            full_indices = np.arange(len(trials))
            # Fixed seed for subset selection to be consistent across runs if desired, 
            # OR use logic from Semedo.py to shuffle. 
            # Re-using built-in logic via build_groups... or manual splitting?
            # Existing code:
            rng_sub = np.random.default_rng(random_state + run_idx * 100)
            perm = rng_sub.permutation(len(trials))
            n = len(trials)
            sz = n // k_subsets
            groups = [perm[i*sz : (i+1)*sz] for i in range(k_subsets)]

        for g_idx in groups:
            Y_sub, X_sub = Y_match[g_idx, :], X_match[g_idx, :]
            
            # Recalculate d95 for this subset
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



def build_semedo_figure_5_b(
    *,
    analysis_type: str = "residual",
    num_sets: int = 10,
    seed: int = 0,
    outer_splits: int = 3,
    inner_splits: int = 3,
    alpha: float | None = None,
    threshold: float = 0.95,
    verbose: bool = True,
    filename: str | None = None,
    csv_name: str | None = None,
    force_recompute: bool = False,
) -> str:
    """
    Generate Figure 5B (Comparison of RRR vs Ridge performance and dimensionality).
    If a CSV with the same name exists and force_recompute is False, plots from CSV directly.
    """
    # runtime.set_cfg(monkey_name, z_score_index) -> Config assumed set by driver
    consts = runtime.get_consts()
    cfg    = runtime.get_cfg()
    
    if filename is None:
        filename = f"figure_5_B_{analysis_type.upper()}_{num_sets}_SETS.png"
    
    if csv_name is None:
        csv_name = str(Path(filename).with_suffix(".csv"))
        
    data_root: Path = cfg.get_data_path()
    out_dir: Path = data_root / "Semedo_plots" / "figure_5_B"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / csv_name
    fig_path = out_dir / filename

    # --- caching check ---
    if not force_recompute and csv_path.exists():
        if verbose:
            print(f"[Cache] Found existing data: {csv_path}")
            print(f"[Cache] Skipping analysis and re-plotting from CSV...")
        
        SemedoFigures.plot_figure_5_b(str(csv_path), str(fig_path))
        if verbose:
             print(f"[✓] Fig 5B saved (from cache) → {fig_path}")
        return str(fig_path)

    rid_v1 = int(consts.REGION_NAME_TO_ID["V1"])
    rid_v4 = int(consts.REGION_NAME_TO_ID["V4"])
    rid_it = int(consts.REGION_NAME_TO_ID["IT"])
    stat_key  = analysis_type.strip().lower()

    # subsets
    subset_dir = data_root / "TARGET_RRR" / stat_key.upper()
    v1_subset_v4_phys = np.asarray(np.load(subset_dir / f"V1_to_V4_{stat_key}.npz")["phys_idx"], dtype=int)
    v1_subset_it_phys = np.asarray(np.load(subset_dir / f"V1_to_IT_{stat_key}.npz")["phys_idx"], dtype=int)



    # load full matrices
    V1_full = build_trial_matrix(region_id=rid_v1, analysis_type=stat_key)
    V4_full = build_trial_matrix(region_id=rid_v4, analysis_type=stat_key)
    IT_full = build_trial_matrix(region_id=rid_it, analysis_type=stat_key)
    T = int(V1_full.shape[0])

    # complements
    all_v1_phys = np.flatnonzero(cfg.get_rois() == rid_v1)
    comp_v4_phys = np.array(sorted(set(all_v1_phys) - set(v1_subset_v4_phys)), dtype=int)
    comp_it_phys = np.array(sorted(set(all_v1_phys) - set(v1_subset_it_phys)), dtype=int)

    # Helpers adapted to project utils
    def _pca_d95_util(M):
        return RegionPCA().fit(M).get_n_components(threshold)

    def _d95_rrr_util(res, dmax_val):
        return d95_from_curves(res["rrr_R2_mean"], res["ridge_R2_mean"], dmax_val)

    # ---------- FULL DATA POINTS ----------
    full_x_v4 = _pca_d95_util(V4_full)
    full_x_it = _pca_d95_util(IT_full)

    full_x_v1v4 = _pca_d95_util(
        build_trial_matrix(
            region_id=rid_v1,
            analysis_type=stat_key,
            trials=None,
            electrode_indices=v1_subset_v4_phys
        )
    )
    full_x_v1it = _pca_d95_util(
        build_trial_matrix(
            region_id=rid_v1,
            analysis_type=stat_key,
            trials=None,
            electrode_indices=v1_subset_it_phys
        )
    )

    X_v1_minus_v4_full = build_trial_matrix(
        region_id=rid_v1, analysis_type=stat_key, trials=None, electrode_indices=comp_v4_phys
    )
    X_v1_minus_it_full = build_trial_matrix(
        region_id=rid_v1, analysis_type=stat_key, trials=None, electrode_indices=comp_it_phys
    )

    dmax_calc = lambda X, Y: int(min(X.shape[1], Y.shape[1], 512))

    full_res_v4 = RRRAnalyzer._performance_from_mats(
        V4_full, X_v1_minus_v4_full,
        d_max=dmax_calc(X_v1_minus_v4_full, V4_full),
        alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+111
    )
    full_res_v1v4 = RRRAnalyzer._performance_from_mats(
        build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=None, electrode_indices=v1_subset_v4_phys),
        X_v1_minus_v4_full,
        d_max=dmax_calc(X_v1_minus_v4_full, V1_full), # Note: Using V1_full dimension reference from original context or just X?
        # Original snippet used V1_full in dmax calculation for subsets. Let's stick to X/Y shapes.
        # Actually in user snippet: d_max=dmax(X_v1_minus_v4_full, V1_full). V1_full is ~100 dims.
        alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+222
    )
    full_res_it = RRRAnalyzer._performance_from_mats(
        IT_full, X_v1_minus_it_full,
        d_max=dmax_calc(X_v1_minus_it_full, IT_full),
        alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+333
    )
    full_res_v1it = RRRAnalyzer._performance_from_mats(
        build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=None, electrode_indices=v1_subset_it_phys),
        X_v1_minus_it_full,
        d_max=dmax_calc(X_v1_minus_it_full, V1_full),
        alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+444
    )

    # Recalculate d_max for d95 extraction (should match d_max passed to RRR)
    # The snippet used dmax(...) inside the call.
    # Note: d95_from_curves takes a scalar d_max. 
    # We should use the same d_max as used in RRR.
    
    full_y_v4 = _d95_rrr_util(full_res_v4, dmax_calc(X_v1_minus_v4_full, V4_full))
    full_y_v1v4 = _d95_rrr_util(full_res_v1v4, dmax_calc(X_v1_minus_v4_full, V1_full))
    full_y_it = _d95_rrr_util(full_res_it, dmax_calc(X_v1_minus_it_full, IT_full))
    full_y_v1it = _d95_rrr_util(full_res_v1it, dmax_calc(X_v1_minus_it_full, V1_full))

    full_points = {
        "V4":             (full_x_v4, full_y_v4),
        "Target V4":  (full_x_v1v4, full_y_v1v4),
        "IT":             (full_x_it, full_y_it),
        "Target IT":  (full_x_v1it, full_y_v1it),
    }

    # ---------- RANDOM SETS ----------
    rng = np.random.default_rng(seed)
    trials_per_set = T // num_sets
    perm = rng.permutation(T)
    # Ensure strict integer division doesn't lose trials if we want coverage, but original split logic:
    parts = [perm[i*trials_per_set:(i+1)*trials_per_set] for i in range(num_sets)]

    groups = {
        "V4":             {"color": "#ca1b1b", "xs": [], "ys": []},
        "Target V4":  {"color": "#cf2359", "xs": [], "ys": []},
        "IT":             {"color": "#0529ae", "xs": [], "ys": []},
        "Target IT":  {"color": "#3391e3", "xs": [], "ys": []},
    }

    for i, idx in enumerate(parts):
        X_v1_minus_v4 = build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=idx, electrode_indices=comp_v4_phys)
        X_v1_minus_it = build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=idx, electrode_indices=comp_it_phys)
        Y_v4 = V4_full[idx, :]
        Y_it = IT_full[idx, :]
        Y_v1_sub_v4 = build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=idx, electrode_indices=v1_subset_v4_phys)
        Y_v1_sub_it = build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=idx, electrode_indices=v1_subset_it_phys)

        # PCA
        x_v4  = _pca_d95_util(Y_v4)
        x_it  = _pca_d95_util(Y_it)
        x_v1v4 = _pca_d95_util(Y_v1_sub_v4)
        x_v1it = _pca_d95_util(Y_v1_sub_it)

        # RRR
        dm_v4 = dmax_calc(X_v1_minus_v4, Y_v4)
        y_v4 = _d95_rrr_util(
            RRRAnalyzer._performance_from_mats(
                Y_v4, X_v1_minus_v4, d_max=dm_v4, alpha=alpha, 
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i
            ), dm_v4
        )

        dm_v1v4 = dmax_calc(X_v1_minus_v4, Y_v1_sub_v4)
        y_v1v4 = _d95_rrr_util(
            RRRAnalyzer._performance_from_mats(
                Y_v1_sub_v4, X_v1_minus_v4, d_max=dm_v1v4, alpha=alpha,
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i+1
            ), dm_v1v4
        )

        dm_it = dmax_calc(X_v1_minus_it, Y_it)
        y_it = _d95_rrr_util(
            RRRAnalyzer._performance_from_mats(
                Y_it, X_v1_minus_it, d_max=dm_it, alpha=alpha,
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i+2
            ), dm_it
        )

        dm_v1it = dmax_calc(X_v1_minus_it, Y_v1_sub_it)
        y_v1it = _d95_rrr_util(
            RRRAnalyzer._performance_from_mats(
                Y_v1_sub_it, X_v1_minus_it, d_max=dm_v1it, alpha=alpha,
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i+3
            ), dm_v1it
        )

        groups["V4"]["xs"].append(x_v4);              groups["V4"]["ys"].append(y_v4)
        groups["Target V4"]["xs"].append(x_v1v4);     groups["Target V4"]["ys"].append(y_v1v4)
        groups["IT"]["xs"].append(x_it);              groups["IT"]["ys"].append(y_it)
        groups["Target IT"]["xs"].append(x_v1it);     groups["Target IT"]["ys"].append(y_v1it)

        if verbose:
            print(f"[{i+1}/{num_sets}] V4({x_v4},{y_v4}) | V1-V4({x_v1v4},{y_v1v4}) | IT({x_it},{y_it}) | V1-IT({x_v1it},{y_v1it})")

    # Collect data for CSV
    # Each item: [group, x, y, point_type]
    csv_rows = []

    # 1. Random Subsets
    for name, g in groups.items():
        for x, y in zip(g["xs"], g["ys"]):
            csv_rows.append([name, int(x), int(y), "subset"])

    # 2. Full Points
    for name, (fx, fy) in full_points.items():
        csv_rows.append([name, int(fx), int(fy), "full"])

    # Save CSV
    csv_path = out_dir / csv_name
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "x_dimensionality_d95", "y_predictive_d95", "point_type"])
        w.writerows(csv_rows)
    print(f"[✓] CSV data saved → {csv_path}")

    # Call Visualization to plot from CSV
    
    # We use the filename arg to determine where the plot goes.
    fig_path = out_dir / filename
    SemedoFigures.plot_figure_5_b(str(csv_path), str(fig_path))
    
    if verbose:
        print(f"[✓] Fig 5B saved → {fig_path}")

    return str(fig_path)
