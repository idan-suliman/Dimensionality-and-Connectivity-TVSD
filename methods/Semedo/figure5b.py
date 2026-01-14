from __future__ import annotations
import numpy as np 
from pathlib import Path
import csv
from core.runtime import runtime
from ..rrr import RRRAnalyzer
from ..visualization import SemedoFigures, d95_from_curves
from ..pca import RegionPCA
from .utils import build_trial_matrix

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

    # subsets (precomputed matching required)
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
        d_max=dmax_calc(X_v1_minus_v4_full, V1_full),
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
        d_max=dmax_calc(X_v1_minus_it_full, V1_full), # Used V1_full in snippet
        alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+444
    )

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
    csv_rows = []

    # 1. Random Subsets
    for name, g in groups.items():
        for x, y in zip(g["xs"], g["ys"]):
            csv_rows.append([name, int(x), int(y), "subset"])

    # 2. Full Points
    for name, (fx, fy) in full_points.items():
        csv_rows.append([name, int(fx), int(fy), "full"])

    # Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "x_dimensionality_d95", "y_predictive_d95", "point_type"])
        w.writerows(csv_rows)
    print(f"[✓] CSV data saved → {csv_path}")

    # Call Visualization
    SemedoFigures.plot_figure_5_b(str(csv_path), str(fig_path))
    
    if verbose:
        print(f"[✓] Fig 5B saved → {fig_path}")

    return str(fig_path)
