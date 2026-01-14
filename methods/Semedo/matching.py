from __future__ import annotations
import numpy as np
import time
from datetime import datetime
from core.runtime import runtime
from ..matchingSubset import MATCHINGSUBSET
from ..rrr import RRRAnalyzer

# Local wrapper for convenience if needed, but we can import direct from source
def build_trial_matrix(*args, **kwargs):
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
