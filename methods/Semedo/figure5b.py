from __future__ import annotations
import numpy as np 
from pathlib import Path
import csv
from core.runtime import runtime
from ..rrr import RRRAnalyzer
from visualization import SemedoFigures
from ..pca import RegionPCA



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
    # runtime.update(monkey_name, z_score_index) -> Config assumed set by driver
    consts = runtime.consts
    cfg    = runtime.cfg
    dm     = runtime.data_manager
    
    # Get default path from runtime
    fig_path = runtime.paths.get_semedo_figure_path("Figure_5_B", analysis_type, num_sets=num_sets)
    
    # If a custom filename is provided, override the name but keep the directory
    if filename is not None:
        fig_path = fig_path.parent / filename
        
    # Ensure directory exists (redundant if get_semedo_figure_path does it, but safe for custom filename case)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
        
    if csv_name is None:
        csv_path = fig_path.with_suffix(".csv")
    else:
        csv_path = fig_path.parent / csv_name

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
    # Using runtime.paths to get matching files
    path_v1_v4 = runtime.paths.get_matching_path(stat_key, "V1", "V4")
    path_v1_it = runtime.paths.get_matching_path(stat_key, "V1", "IT")
    
    v1_subset_v4_phys = np.asarray(np.load(path_v1_v4)["phys_idx"], dtype=int)
    v1_subset_it_phys = np.asarray(np.load(path_v1_it)["phys_idx"], dtype=int)

    # load full matrices
    V1_full = dm.build_trial_matrix(region_id=rid_v1, analysis_type=stat_key)
    V4_full = dm.build_trial_matrix(region_id=rid_v4, analysis_type=stat_key)
    IT_full = dm.build_trial_matrix(region_id=rid_it, analysis_type=stat_key)
    T = int(V1_full.shape[0])

    # complements
    all_v1_phys = np.flatnonzero(cfg.get_rois() == rid_v1)
    comp_v4_phys = np.array(sorted(set(all_v1_phys) - set(v1_subset_v4_phys)), dtype=int)
    comp_it_phys = np.array(sorted(set(all_v1_phys) - set(v1_subset_it_phys)), dtype=int)

    # Helpers adapted to project utils




    # ---------- FULL DATA POINTS ----------
    full_x_v4 = RegionPCA().fit(V4_full).get_n_components(threshold)
    full_x_it = RegionPCA().fit(IT_full).get_n_components(threshold)

    full_x_v1v4 = RegionPCA().fit(
        dm.build_trial_matrix(
            region_id=rid_v1,
            analysis_type=stat_key,
            trials=None,
            electrode_indices=v1_subset_v4_phys
        )
    ).get_n_components(threshold)

    full_x_v1it = RegionPCA().fit(
        dm.build_trial_matrix(
            region_id=rid_v1,
            analysis_type=stat_key,
            trials=None,
            electrode_indices=v1_subset_it_phys
        )
    ).get_n_components(threshold)

    X_v1_minus_v4_full = dm.build_trial_matrix(
        region_id=rid_v1, analysis_type=stat_key, trials=None, electrode_indices=comp_v4_phys
    )
    X_v1_minus_it_full = dm.build_trial_matrix(
        region_id=rid_v1, analysis_type=stat_key, trials=None, electrode_indices=comp_it_phys
    )



    full_res_v4 = RRRAnalyzer._performance_from_mats(
        V4_full, X_v1_minus_v4_full,
        d_max=int(min(X_v1_minus_v4_full.shape[1], V4_full.shape[1], 512)),
        alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+111
    )
    full_res_v1v4 = RRRAnalyzer._performance_from_mats(
        dm.build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=None, electrode_indices=v1_subset_v4_phys),
        X_v1_minus_v4_full,
        d_max=int(min(X_v1_minus_v4_full.shape[1], V1_full.shape[1], 512)),
        alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+222
    )
    full_res_it = RRRAnalyzer._performance_from_mats(
        IT_full, X_v1_minus_it_full,
        d_max=int(min(X_v1_minus_it_full.shape[1], IT_full.shape[1], 512)),
        alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+333
    )
    full_res_v1it = RRRAnalyzer._performance_from_mats(
        dm.build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=None, electrode_indices=v1_subset_it_phys),
        X_v1_minus_it_full,
        d_max=int(min(X_v1_minus_it_full.shape[1], V1_full.shape[1], 512)), # Used V1_full in snippet
        alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+444
    )

    full_y_v4 = RRRAnalyzer.calc_d95(full_res_v4["rrr_R2_mean"], full_res_v4["ridge_R2_mean"], int(min(X_v1_minus_v4_full.shape[1], V4_full.shape[1], 512)))
    full_y_v1v4 = RRRAnalyzer.calc_d95(full_res_v1v4["rrr_R2_mean"], full_res_v1v4["ridge_R2_mean"], int(min(X_v1_minus_v4_full.shape[1], V1_full.shape[1], 512)))
    full_y_it = RRRAnalyzer.calc_d95(full_res_it["rrr_R2_mean"], full_res_it["ridge_R2_mean"], int(min(X_v1_minus_it_full.shape[1], IT_full.shape[1], 512)))
    full_y_v1it = RRRAnalyzer.calc_d95(full_res_v1it["rrr_R2_mean"], full_res_v1it["ridge_R2_mean"], int(min(X_v1_minus_it_full.shape[1], V1_full.shape[1], 512)))

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
        X_v1_minus_v4 = dm.build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=idx, electrode_indices=comp_v4_phys)
        X_v1_minus_it = dm.build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=idx, electrode_indices=comp_it_phys)
        Y_v4 = V4_full[idx, :]
        Y_it = IT_full[idx, :]
        Y_v1_sub_v4 = dm.build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=idx, electrode_indices=v1_subset_v4_phys)
        Y_v1_sub_it = dm.build_trial_matrix(region_id=rid_v1, analysis_type=stat_key, trials=idx, electrode_indices=v1_subset_it_phys)

        # PCA
        x_v4  = RegionPCA().fit(Y_v4).get_n_components(threshold)
        x_it  = RegionPCA().fit(Y_it).get_n_components(threshold)
        x_v1v4 = RegionPCA().fit(Y_v1_sub_v4).get_n_components(threshold)
        x_v1it = RegionPCA().fit(Y_v1_sub_it).get_n_components(threshold)

        # RRR
        dm_v4 = int(min(X_v1_minus_v4.shape[1], Y_v4.shape[1], 512))
        y_v4 = RRRAnalyzer.calc_d95(
            RRRAnalyzer._performance_from_mats(
                Y_v4, X_v1_minus_v4, d_max=dm_v4, alpha=alpha, 
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i
            )["rrr_R2_mean"], 
            RRRAnalyzer._performance_from_mats(
                Y_v4, X_v1_minus_v4, d_max=dm_v4, alpha=alpha, 
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i
            )["ridge_R2_mean"], 
            dm_v4
        )

        dm_v1v4 = int(min(X_v1_minus_v4.shape[1], Y_v1_sub_v4.shape[1], 512))
        y_v1v4 = RRRAnalyzer.calc_d95(
            RRRAnalyzer._performance_from_mats(
                Y_v1_sub_v4, X_v1_minus_v4, d_max=dm_v1v4, alpha=alpha,
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i+1
            )["rrr_R2_mean"],
            RRRAnalyzer._performance_from_mats(
                Y_v1_sub_v4, X_v1_minus_v4, d_max=dm_v1v4, alpha=alpha,
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i+1
            )["ridge_R2_mean"], 
            dm_v1v4
        )

        dm_it = int(min(X_v1_minus_it.shape[1], Y_it.shape[1], 512))
        y_it = RRRAnalyzer.calc_d95(
            RRRAnalyzer._performance_from_mats(
                Y_it, X_v1_minus_it, d_max=dm_it, alpha=alpha,
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i+2
            )["rrr_R2_mean"],
            RRRAnalyzer._performance_from_mats(
                Y_it, X_v1_minus_it, d_max=dm_it, alpha=alpha,
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i+2
            )["ridge_R2_mean"],
            dm_it
        )

        dm_v1it = int(min(X_v1_minus_it.shape[1], Y_v1_sub_it.shape[1], 512))
        y_v1it = RRRAnalyzer.calc_d95(
            RRRAnalyzer._performance_from_mats(
                Y_v1_sub_it, X_v1_minus_it, d_max=dm_v1it, alpha=alpha,
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i+3
            )["rrr_R2_mean"],
            RRRAnalyzer._performance_from_mats(
                Y_v1_sub_it, X_v1_minus_it, d_max=dm_v1it, alpha=alpha,
                outer_splits=outer_splits, inner_splits=inner_splits, random_state=seed+10*i+3
            )["ridge_R2_mean"],
            dm_v1it
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
