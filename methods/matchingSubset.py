# matchingSubset.py
from core.runtime import runtime
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path


# ------------------------ helper: build matrix ------------------------



def _compute_matrix(region: str, stat: str,
                    day_filter: list[int] | None = None):
    consts = runtime.consts
    cfg    = runtime.cfg

    rid  = consts.REGION_NAME_TO_ID[region]
    idx  = np.flatnonzero(cfg.get_rois() == rid)

    trials = None
    if day_filter:
        day_filter_set = set(day_filter)
        trials = [
            tr for tr in runtime.data_manager._load_trials()
            if tr.get("day_id") in day_filter_set
        ]

    X = runtime.data_manager.build_trial_matrix(
        region_id=rid,
        analysis_type=stat,
        trials=trials,
    )
    return X, idx

def match_and_save(
                source_region : str,
                target_region : str,
                *,
                stat_mode  : str = "window",
                n_bins     : int = 20,
                day_filter : list[int] | None = None,
                seed       : int = 0,
                show_plot  : bool = True,
                verbose    : bool = True,
                debug_bins : bool = False):
    """
    Histogram-matching (V1 ➝ V4/IT) – creates a V1 subset of size |V4| and saves:
    • subset.npz
    • BEFORE / AFTER histograms (with ⟨FR⟩ mean firing rate shown in titles)
    """

    rng = np.random.default_rng(seed)
    if stat_mode not in {"window", "baseline100", "residual"}:
        raise ValueError("stat_mode must be 'window' / 'baseline100' / 'residual'")
    # 1) matrices + mean firing
    Xs, idx_s = _compute_matrix(source_region, stat_mode, day_filter)
    Xt, _     = _compute_matrix(target_region, stat_mode, day_filter)
    ms, mt    = Xs.mean(0), Xt.mean(0)
    n_tgt     = Xt.shape[1]

    # 2) unified bins
    vmin, vmax = float(min(ms.min(), mt.min())), float(max(ms.max(), mt.max()))
    edges      = np.linspace(vmin, vmax, n_bins + 1)
    src_bin_of = np.searchsorted(edges, ms, side="right") - 1
    src_bin_of = np.clip(src_bin_of, 0, n_bins - 1)

    picked, surplus_bins, deficits = [], [[] for _ in range(n_bins)], np.zeros(n_bins, int)
    tgt_counts = [((mt >= edges[b]) & (mt < edges[b + 1])).sum() for b in range(n_bins)]

    for b in range(n_bins):
        s_idxs   = np.where(src_bin_of == b)[0]
        need     = tgt_counts[b]
        if need == 0:          
            surplus_bins[b] = list(s_idxs)   
            continue

        if s_idxs.size <= need:            
            picked.extend(s_idxs)
            deficits[b] = need - s_idxs.size
        else:                             
            chosen = rng.choice(s_idxs, need, replace=False)
            picked.extend(chosen)
            surplus_bins[b] = [idx for idx in s_idxs if idx not in chosen]

    for b in np.where(deficits > 0)[0]:
        deficit = int(deficits[b])
        dist    = 1
        while deficit > 0 and dist < n_bins:
            for nb in (b - dist, b + dist):            
                if 0 <= nb < n_bins and surplus_bins[nb]:
                    take   = min(len(surplus_bins[nb]), deficit)
                    chosen = rng.choice(surplus_bins[nb], take, replace=False)
                    picked.extend(chosen)
                    surplus_bins[nb] = [idx for idx in surplus_bins[nb] if idx not in chosen]
                    deficit -= take
                    if deficit == 0:
                        break
            dist += 1
        if deficit:   # fallback
            remaining = [idx for sl in surplus_bins for idx in sl]
            take      = min(deficit, len(remaining))
            if take:
                chosen = rng.choice(remaining, take, replace=False)
                picked.extend(chosen)
                for idx in chosen:
                    surplus_bins[src_bin_of[idx]].remove(idx)

    picked = np.asarray(sorted(set(picked))[:n_tgt], dtype=int)  

    # 5) subset & phys-idx
    data_sub = Xs[:, picked]
    phys_sub = idx_s[picked]

    # 6) save files
    npz_path = runtime.paths.get_matching_path(stat_mode, source_region, target_region, ".npz")
    np.savez_compressed(npz_path,
                        data=data_sub.astype(np.float32),
                        phys_idx=phys_sub.astype(np.int32))
    if verbose:
        print(f"[✓] subset ({phys_sub.size} ch.) saved → {root / (base + '.npz')}")

    # 7) histograms + ⟨FR⟩ annotation
    if show_plot:
        fr_V4     = mt.mean()
        fr_fullV1 = ms.mean()
        fr_subV1  = ms[picked].mean()
        edges_auto = edges

        # BEFORE
        plt.figure(figsize=(8, 4))
        plt.hist(ms, bins=edges_auto, alpha=.5, color="crimson", label="V1-full")
        plt.hist(mt, bins=edges_auto, alpha=.5, color="royalblue", label=target_region)
        plt.title(f"{source_region}→{target_region} | {stat_mode} (BEFORE)\n"
                f"V4 = {fr_V4:.2f}  |  V1-full = {fr_fullV1:.2f}")
        plt.xlabel("Mean firing", fontsize=20, fontweight="bold")
        plt.ylabel("Electrode count", fontsize=20, fontweight="bold")
        plt.tick_params(axis='both', which='major', labelsize=18, width=1.5)
        plt.tight_layout()
        plt.legend(fontsize=15)
        plt.savefig(runtime.paths.get_matching_path(stat_mode, source_region, target_region, "_before_hist.png"), dpi=300)
        plt.close()

        # AFTER
        plt.figure(figsize=(8, 4))
        plt.hist(ms[picked], bins=edges_auto, alpha=.5,
                color="crimson", label=f"{source_region}-subset")
        plt.hist(mt, bins=edges_auto, alpha=.5,
                color="royalblue", label=target_region)
        if debug_bins:
            src_cnt, _ = np.histogram(ms[picked], bins=edges_auto)
            tgt_cnt, _ = np.histogram(mt,         bins=edges_auto)
            plt.step(edges_auto[:-1], src_cnt, where="mid", color="crimson")
            plt.step(edges_auto[:-1], tgt_cnt, where="mid", color="royalblue")
        plt.title(f"{source_region}→{target_region} | {stat_mode} (AFTER)\n"
                f"V4 = {fr_V4:.2f}  |  V1-subset = {fr_subV1:.2f}")
        plt.xlabel("Mean firing", fontsize=20, fontweight="bold")
        plt.ylabel("Electrode count", fontsize=20, fontweight="bold")
        plt.tick_params(axis='both', which='major', labelsize=18, width=1.5)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig(runtime.paths.get_matching_path(stat_mode, source_region, target_region, "_hist.png"), dpi=300)
        plt.close()

        if verbose:
            print(f"[✓] Histograms saved → {base}_before_hist.png / {base}_hist.png")

    return data_sub, phys_sub
