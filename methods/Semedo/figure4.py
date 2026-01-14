from __future__ import annotations
import numpy as np 
from core.runtime import runtime
from ..rrr import RRRAnalyzer
from visualization import SemedoFigures, d95_from_curves
from .utils import build_groups_by_rep_or_subsets, build_trial_matrix

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
    
    # --- Caching Check ---
    if not force_recompute and npz_path.exists():
        print(f"[Cache] Found existing data: {npz_path}")
        print(f"[Cache] Loading data and plotting...")
        
        with np.load(npz_path, allow_pickle=True) as data:
            perf_full = data["perf_full"].item()
            perf_match = data["perf_match"].item()
            d95_full_g = int(data["d95_full_g"])
            d95_match_g = int(data["d95_match_g"])
            d95_full_rep = data["d95_full_rep"].tolist()
            d95_match_rep = data["d95_match_rep"].tolist()
            label_D = str(data["label_D"])
            if int(data["d_max"]) != d_max:
                 print(f"[Warning] Cached d_max ({data['d_max']}) != requested ({d_max})")

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
    perf_wrapper = RRRAnalyzer.performance
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
