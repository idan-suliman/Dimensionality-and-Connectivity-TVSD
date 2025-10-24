from rrr import RRR_Centered_matching
from runtime import runtime
from matchingSubset import MATCHINGSUBSET
import numpy as np, matplotlib.pyplot as plt    
from matplotlib import gridspec
from itertools import cycle
import matplotlib.patheffects as pe

def make_semedo_figure(
    source_region: int = 1,          # V1
    target_region: int = 3,          # IT (or V4 = 2)
    *,
    analysis_type: str = "residual", # "window" | "baseline100" | "residual"
    d_max: int = 35,
    alpha: float | None = None,
    outer_splits: int = 3,
    inner_splits: int = 3,
    random_state: int = 0,
    k_subsets: int | None = None,    # None → repetitions; int → K random subsets (Panel D only)
):
    """Generate Semedo-style multi-panel figure (A–D) summarizing RRR performance and predictive subspace dimensionality."""

    # ====================== Load and organize trials ======================
    trials = runtime.get_cfg()._load_trials()

    # repetition index (0..29) without mutating trials
    rep_arr = np.array([int(tr["allmat_row"][3]) - 1 for tr in trials], dtype=int)

    # Group trials by repetitions or random subsets -> groups of TRIAL INDICES (ints)
    if k_subsets is None:
        rep_ids = np.unique(rep_arr)
        groups  = [np.flatnonzero(rep_arr == g) for g in rep_ids]  # list[np.ndarray[int]]
        label_D  = "Repetitions"
        id_print = lambda g: f"rep {g:2}"
    else:
        rng    = np.random.default_rng(random_state)
        idxs   = np.arange(len(trials))
        rng.shuffle(idxs)
        splits = np.array_split(idxs, k_subsets)
        groups = [np.asarray(part, dtype=int) for part in splits]   # list[np.ndarray[int]]
        label_D = f"{k_subsets} random subsets"
        id_print = lambda g: f"set {g:2}"

    # ============================ Helper functions ============================
    def _d95(rrr_mean: np.ndarray, ridge_mean: float) -> int:
        thr = 0.95 * float(ridge_mean)
        idx = np.where(rrr_mean >= thr)[0]
        return int(idx[0] + 1) if idx.size else int(d_max)

    def _perf(match: bool):
        return RRR_Centered_matching.performance(
            source_region, target_region,
            analysis_type=analysis_type,
            match_to_target=match,
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state
        )

    def _labeled_dot(ax, x, y, label, *, face, edge="k",
                     size=240, text_size=12, text_color="white"):
        ax.scatter([x], [y], s=size, facecolors=face, edgecolors=edge,
                   linewidths=1.6, zorder=7)
        ax.text(x, y, str(label), ha="center", va="center",
                color=text_color, fontsize=text_size, weight="bold",
                zorder=9, path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])

    def _square_limits(x_vals, y_vals, *, base_min=1, scale=1.5):
        vmax = float(np.max([np.max(np.atleast_1d(x_vals)),
                             np.max(np.atleast_1d(y_vals))]))
        lim_max = int(np.ceil(scale * vmax))
        lim_max = max(lim_max, base_min + 1)
        return (base_min, lim_max)

    def _jitter(values, rng: np.random.Generator, *, scale: float = 0.15):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr
        return arr + rng.uniform(-scale, scale, size=arr.shape)

    # ====================== Compute global RRR results (Panels A + B) ======================
    perf_full  = _perf(False)  # Standard V1→Target mapping
    perf_match = _perf(True)   # Matched V1 subset → (remaining V1)

    d95_full_g  = _d95(perf_full ["rrr_R2_mean"],  perf_full ["ridge_R2_mean"])
    d95_match_g = _d95(perf_match["rrr_R2_mean"], perf_match["ridge_R2_mean"])

    # ============================ Prepare ROI indices ============================
    rois         = runtime.get_cfg().get_rois()
    src_idx_full = np.where(rois == source_region)[0]
    tgt_idx_full = np.where(rois == target_region)[0]

    # Load or create matching subset indices
    match_path = (
        runtime.get_cfg().get_data_path() / "TARGET_RRR" / analysis_type.upper() /
        f"V1_to_{runtime.get_consts().REGION_ID_TO_NAME[target_region]}_{analysis_type}.npz"
    )
    if not match_path.exists():
        MATCHINGSUBSET.match_and_save(
            "V1", runtime.get_consts().REGION_ID_TO_NAME[target_region],
            stat_mode=analysis_type, show_plot=False, verbose=False
        )
    match_idx = np.load(match_path)["phys_idx"]

    # Pre-compute remaining V1 (X) indices once
    remain_mask    = ~np.isin(src_idx_full, match_idx)
    src_idx_remain = src_idx_full[remain_mask]

    # ====================== Compute per-group dimensionalities (Panel D) ======================
    d95_full_rep, d95_match_rep = [], []

    def _mat(rid: int, elec_idx: np.ndarray | None, trial_idx: np.ndarray | slice | None) -> np.ndarray:
        """Build matrix for a region with an optional electrode subset on a given TRIAL-INDEX subset."""
        return runtime.get_cfg().build_trial_matrix(
            region_id=rid,
            analysis_type=analysis_type,
            trials=trial_idx,                                   # <-- TRIAL INDICES
            electrode_indices=(None if elec_idx is None else np.asarray(elec_idx, dtype=int)),
            return_stimulus_ids=False
        )

    for g_idx, grp_idx in enumerate(groups):  # grp_idx is np.ndarray[int] of trial indices
        # Unmatched (V1 → target)
        res_f = RRR_Centered_matching._performance_from_mats(
            _mat(target_region, tgt_idx_full, grp_idx),
            _mat(1,             src_idx_full, grp_idx),
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state
        )
        d95_full_rep.append(_d95(res_f["rrr_R2_mean"], res_f["ridge_R2_mean"]))

        # Matched (V1-match → remaining V1)
        res_m = RRR_Centered_matching._performance_from_mats(
            _mat(1, match_idx,      grp_idx),
            _mat(1, src_idx_remain, grp_idx),
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state
        )
        d95_match_rep.append(_d95(res_m["rrr_R2_mean"], res_m["ridge_R2_mean"]))

        print(f"[{id_print(g_idx)}] "
              f"ridge_full={res_f['ridge_R2_mean']:.3f}  "
              f"ridge_match={res_m['ridge_R2_mean']:.3f}  "
              f"d95_full={d95_full_rep[-1]}  d95_match={d95_match_rep[-1]}")

    # =============================== Create Figure Layout ===============================
    fig = plt.figure(figsize=(14, 13), dpi=400)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12,
                        wspace=0.15, hspace=0.30)

    gs  = gridspec.GridSpec(2, 2)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[1, 1])

    dims   = np.arange(1, d_max + 1)
    tgt    = runtime.get_consts().REGION_ID_TO_NAME[target_region]
    colA, colB = "#9C1C1C", "#1565C0"
    label_fs = 17

    # =============================== Panel A (Full RRR curve) ===============================
    axA.errorbar(dims, perf_full["rrr_R2_mean"], yerr=perf_full["rrr_R2_sem"],
                 fmt="o-", ms=3.8, lw=1.35, capsize=3, color=colA, zorder=2)
    axA.scatter([1], [perf_full["ridge_R2_mean"]], marker="^", s=90,
                color=colA, edgecolors="k", zorder=3)

    if np.isfinite(d95_full_g):
        r2d = perf_full["rrr_R2_mean"][int(d95_full_g) - 1]
        _labeled_dot(axA, int(d95_full_g), float(r2d), int(d95_full_g),
                     face=colA, edge="k", size=240, text_size=12, text_color="white")

    axA.set_title(f"Predicting {tgt}", color=colA, pad=10, fontsize=15)
    axA.grid(alpha=.25)
    axA.text(-0.07, 1.05, "A", transform=axA.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")
    axA.set_box_aspect(1)

    # =============================== Panel B (Matched subset RRR curve) ===============================
    axB.errorbar(dims, perf_match["rrr_R2_mean"], yerr=perf_match["rrr_R2_sem"],
                 fmt="o-", ms=3.8, lw=1.35, capsize=3, color=colB, zorder=2)
    axB.scatter([1], [perf_match["ridge_R2_mean"]], marker="^", s=90,
                color=colB, edgecolors="k", zorder=3)

    if np.isfinite(d95_match_g):
        r2d = perf_match["rrr_R2_mean"][int(d95_match_g) - 1]
        _labeled_dot(axB, int(d95_match_g), float(r2d), int(d95_match_g),
                     face=colB, edge="k", size=240, text_size=12, text_color="white")

    axB.set_title(f"Predicting V1-match {tgt}", color=colB, pad=10, fontsize=15)
    axB.grid(alpha=.25)
    axB.text(-0.07, 1.05, "B", transform=axB.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")
    axB.set_box_aspect(1)

    # =============================== Panel C (Global comparison: A vs B) ===============================
    if np.isfinite(d95_match_g) and np.isfinite(d95_full_g):
        xg, yg = int(d95_match_g), int(d95_full_g)
        xmin, xmax = _square_limits([xg], [yg], base_min=1, scale=1.5)
    else:
        xmin, xmax = _square_limits([1], [1], base_min=1, scale=1.5)

    rng_jitter = np.random.default_rng(random_state)

    axC.plot([xmin, xmax], [xmin, xmax], ls="--", lw=0.9, color="k")  # y=x
    if np.isfinite(d95_match_g) and np.isfinite(d95_full_g):
        jitter_x = _jitter([xg], rng_jitter)[0]
        jitter_y = _jitter([yg], rng_jitter)[0]
        axC.scatter([jitter_x], [jitter_y], s=175, facecolors="white", edgecolors="black", zorder=4)
    axC.set_xlim(xmin, xmax)
    axC.set_ylim(xmin, xmax)
    axC.set_aspect('equal', adjustable='box')
    ticks_c = np.arange(xmin, xmax + 1, max(1, int(np.ceil((xmax - xmin) / 6))))
    axC.set_xticks(ticks_c)
    axC.set_yticks(ticks_c)
    axC.grid(False)
    axC.text(-0.07, 1.05, "C", transform=axC.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")

    # =============================== Panel D (Per-group subsets) ===============================
    if len(d95_match_rep) and len(d95_full_rep):
        xmin_d, xmax_d = _square_limits(d95_match_rep, d95_full_rep, base_min=1, scale=1.5)
    else:
        xmin_d, xmax_d = (1, max(2, int(np.ceil(1.5 * d_max))))

    jitter_match = _jitter(d95_match_rep, rng_jitter)
    jitter_full  = _jitter(d95_full_rep,  rng_jitter)
    axD.scatter(jitter_match, jitter_full, s=60, facecolors="white", edgecolors="black")
    axD.plot([xmin_d, xmax_d], [xmin_d, xmax_d], ls="--", lw=0.9, color="k")
    axD.set_xlim(xmin_d, xmax_d)
    axD.set_ylim(xmin_d, xmax_d)
    axD.set_aspect('equal', adjustable='box')
    ticks_d = np.arange(xmin_d, xmax_d + 1, max(1, int(np.ceil((xmax_d - xmin_d) / 6))))
    axD.set_xticks(ticks_d)
    axD.set_yticks(ticks_d)
    axD.text(0.98, 0.02, label_D, transform=axD.transAxes, ha="right", va="bottom", fontsize=11)
    axD.grid(False)
    axD.text(-0.07, 1.05, "D", transform=axD.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")

    # =============================== Shared axis labels ===============================
    for ax in (axA, axB, axC, axD):
        ax.set_ylabel(None)

    boxA, boxB = axA.get_position(), axB.get_position()
    boxC, boxD = axC.get_position(), axD.get_position()

    left_col_ycenter  = 0.5 * (((boxA.y0 + boxA.y1) / 2) + ((boxB.y0 + boxB.y1) / 2))
    right_block_ycent = 0.5 * (((boxC.y0 + boxC.y1) / 2) + ((boxD.y0 + boxD.y1) / 2))
    left_ylabel_x     = boxA.x0 - 0.066
    right_ylabel_x    = boxC.x0 - 0.043
    right_xlabel_y    = min(boxC.y0, boxD.y0) - 0.058
    left_xlabel_y     = min(boxA.y0, boxB.y0) - 0.058
    left_block_xcent  = 0.5 * (((boxA.x0 + boxA.x1) / 2) + ((boxB.x0 + boxB.x1) / 2))
    right_block_xcent = 0.5 * (((boxC.x0 + boxC.x1) / 2) + ((boxD.x0 + boxD.y1) / 2))  # typo fix if needed

    fig.text(left_ylabel_x, left_col_ycenter,
             rf"Mean $R^2$  (outer {outer_splits}, inner {inner_splits})",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+1, color="black")

    fig.text(right_ylabel_x, right_block_ycent,
             f"{runtime.get_consts().REGION_ID_TO_NAME[target_region]} Predictive dimensions",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+1, color=colA)

    fig.text(left_block_xcent, left_xlabel_y,
             "Predictive dimensions (d)",
             va="top", ha="center", fontsize=label_fs+1, color="black")

    fig.text(right_block_xcent, right_xlabel_y,
             "Target V1 Predictive dimensions",
             va="top", ha="center", fontsize=label_fs+1, color=colB)

    # =============================== Main title & saving ===============================
    top_row_ymax = max(boxA.y1, boxC.y1)
    fig.suptitle(
        f"{runtime.get_cfg().get_monkey_name()}  |  {runtime.get_cfg().get_zscore_title()}  |  {analysis_type.upper()}",
        fontsize=18, y=min(0.998, top_row_ymax + 0.080), fontweight="bold"
    )

    out_dir = runtime.get_cfg().get_data_path() / "Semedo_plots" / f"Semedo_full_electrodes_{k_subsets}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"sub{k_subsets}" if k_subsets else "rep30"
    fname  = f"{runtime.get_cfg().get_monkey_name().replace(' ', '')}_semendo_{suffix}_{analysis_type}_V1_to_{tgt}.png"

    fig.savefig(out_dir / fname, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[✓] Semedo figure saved → {out_dir/fname}")





def make_subset_semedo_figure1(
    source_region : int = 1,      # V1
    target_region : int = 3,      # IT (or 2 = V4)
    *,
    analysis_type : str  = "residual",   # "window" | "baseline100" | "residual"
    d_max         : int  = 35,
    alpha         : float | None = None,
    outer_splits  : int  = 2,
    inner_splits  : int  = 3,
    random_state  : int  = 0,
    # ---- subset parameters ----
    n_run   : int = 5,
    n_src   : int = 113,
    n_tgt   : int = 28,
    k_subsets : int = 10
):
    """
    Subset-Semedo figure (A–D)
    Panels:
        A,B – RRR curves for multiple runs (one color per run)
        C   – single dot per run (d95_match vs. d95_full)
        D   – multiple dots per run (subset-level)
    """

    cfg     = runtime.get_cfg()
    consts  = runtime.get_consts()
    rng     = np.random.default_rng(random_state)
    trials  = cfg._load_trials()  # only for length; matrices computed with indices (not dicts)

    # ---------------- helpers -------------------
    def _matrix(rid: int, elec_idx: np.ndarray | list[int] | None, trial_idx: np.ndarray | slice | None):
        """Build matrix using new build_trial_matrix API: trials are indices/boolean/slice; electrodes by indices."""
        return cfg.build_trial_matrix(
            region_id=rid,
            analysis_type=analysis_type,
            trials=trial_idx,  # indices/slice/None
            electrode_indices=(None if elec_idx is None else np.asarray(elec_idx, dtype=int)),
            return_stimulus_ids=False
        )

    def _d95(r2_vec, ridge_val):
        idx = np.where(r2_vec >= 0.95 * float(ridge_val))[0]
        return int(idx[0] + 1) if idx.size else np.nan

    def _labeled_dot(ax, x, y, label, *, face, edge="k",
                     size=240, text_size=11, text_color="white"):
        ax.scatter([x], [y], s=size, facecolors=face, edgecolors=edge,
                   linewidths=1.5, zorder=6)
        ax.text(x, y, str(label), ha="center", va="center",
                color=text_color, fontsize=text_size, weight="bold",
                zorder=8, path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])

    def _square_limits(x_vals, y_vals, *, base_min=1, scale=1.5):
        xv = np.atleast_1d(x_vals).astype(float)
        yv = np.atleast_1d(y_vals).astype(float)
        xv = xv[np.isfinite(xv)]  # ignore NaNs
        yv = yv[np.isfinite(yv)]
        vmax = float(np.max([np.max(xv) if xv.size else base_min,
                             np.max(yv) if yv.size else base_min]))
        lim_max = int(np.ceil(scale * vmax))
        lim_max = max(lim_max, base_min + 1)
        return (base_min, lim_max)

    # ---------------- electrode indices -------------------
    rois = cfg.get_rois()
    all_src_idx = np.where(rois == source_region)[0]
    all_tgt_idx = np.where(rois == target_region)[0]
    n_src_eff = min(n_src, all_src_idx.size)
    n_tgt_eff = min(n_tgt, all_tgt_idx.size)

    base_cols = ["#9C1C1C", "#1565C0", "#2E7D32", "#7B1FA2", "#F57C00", "#00796B"]
    col_cycle = cycle(base_cols)
    tgt_nm = consts.REGION_ID_TO_NAME[target_region]

    # ---------------- figure layout -------------------
    fig = plt.figure(figsize=(14, 13), dpi=400)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.15, hspace=0.30)
    gs  = gridspec.GridSpec(2, 2)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[1, 1])

    label_fs, tick_fs = 17, 12
    for ax in (axA, axB, axC, axD):
        ax.tick_params(labelsize=tick_fs)
        ax.xaxis.labelpad = 16
        ax.yaxis.labelpad = 20

    # ---------------- per-run computation -------------------
    d95_full_runs, d95_match_runs = [], []
    d95_full_subsets, d95_match_subsets = [], []
    dims = np.arange(1, d_max + 1)

    for run_idx in range(n_run):
        col = next(col_cycle)

        # random electrode subsets
        src_subset = rng.choice(all_src_idx, n_src_eff, replace=False)
        tgt_subset = rng.choice(all_tgt_idx, n_tgt_eff, replace=False)

        # pick "matched" V1 electrodes by smallest mean activity across ALL trials
        mean_src  = _matrix(source_region, src_subset, trial_idx=None).mean(0)
        order_src = np.argsort(mean_src)
        match_src  = src_subset[order_src[:n_tgt_eff]]
        src_remain = src_subset[~np.isin(src_subset, match_src)]

        # FULL runs (all trials → trial_idx=None)
        X_full = _matrix(source_region, src_subset, trial_idx=None)
        Y_full = _matrix(target_region, tgt_subset, trial_idx=None)
        res_full = RRR_Centered_matching._performance_from_mats(
            Y_full, X_full, d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state + run_idx
        )

        # MATCH runs (all trials)
        X_match = _matrix(source_region, src_remain, trial_idx=None)
        Y_match = _matrix(source_region, match_src,  trial_idx=None)
        res_match = RRR_Centered_matching._performance_from_mats(
            Y_match, X_match, d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state + run_idx
        )

        # ---- Plot Panels A + B ----
        axA.errorbar(dims, res_full["rrr_R2_mean"], yerr=res_full["rrr_R2_sem"],
                     fmt="o-", ms=3.8, lw=1.35, capsize=3, color=col)
        axA.scatter([1], [res_full["ridge_R2_mean"]],
                    marker="^", s=90, color=col, edgecolors="k")
        axA.set_box_aspect(1)

        axB.errorbar(dims, res_match["rrr_R2_mean"], yerr=res_match["rrr_R2_sem"],
                     fmt="o-", ms=3.8, lw=1.35, capsize=3, color=col)
        axB.scatter([1], [res_match["ridge_R2_mean"]],
                    marker="^", s=90, color=col, edgecolors="k")
        axB.set_box_aspect(1)

        # ---- Global d95 values ----
        d95_f = _d95(res_full["rrr_R2_mean"],  res_full["ridge_R2_mean"])
        d95_m = _d95(res_match["rrr_R2_mean"], res_match["ridge_R2_mean"])
        d95_full_runs.append(d95_f)
        d95_match_runs.append(d95_m)

        if np.isfinite(d95_f):
            r2d = res_full["rrr_R2_mean"][d95_f - 1]
            _labeled_dot(axA, d95_f, r2d, int(d95_f), face=col)
        if np.isfinite(d95_m):
            r2d = res_match["rrr_R2_mean"][d95_m - 1]
            _labeled_dot(axB, d95_m, r2d, int(d95_m), face=col)

        # ---- build k_subsets points: split TRIAL INDICES (not dicts) ----
        idx_trials = rng.permutation(len(trials))
        chunks = np.array_split(idx_trials, k_subsets)

        for ch_idx in chunks:
            # subset by trial indices
            res_f_sub = RRR_Centered_matching._performance_from_mats(
                _matrix(target_region,  tgt_subset, ch_idx),
                _matrix(source_region, src_subset, ch_idx),
                d_max=d_max, alpha=alpha,
                outer_splits=max(2, outer_splits),
                inner_splits=max(2, inner_splits),
                random_state=random_state + run_idx
            )
            res_m_sub = RRR_Centered_matching._performance_from_mats(
                _matrix(source_region, match_src,  ch_idx),
                _matrix(source_region, src_remain, ch_idx),
                d_max=d_max, alpha=alpha,
                outer_splits=max(2, outer_splits),
                inner_splits=max(2, inner_splits),
                random_state=random_state + run_idx
            )
            d95_f_sub = _d95(res_f_sub["rrr_R2_mean"],  res_f_sub["ridge_R2_mean"])
            d95_m_sub = _d95(res_m_sub["rrr_R2_mean"], res_m_sub["ridge_R2_mean"])
            d95_full_subsets.append((run_idx, d95_f_sub))
            d95_match_subsets.append((run_idx, d95_m_sub))

    # ---------------- Panels C + D -------------------
    valid_x = [int(x) for x in d95_match_runs if np.isfinite(x)]
    valid_y = [int(y) for y in d95_full_runs  if np.isfinite(y)]
    xmin, xmax = _square_limits(valid_x + [1], valid_y + [1])

    rng_jitter   = np.random.default_rng(random_state)
    jitter_scale = 0.15

    # Panel C – נקודה אחת לכל ריצה
    for run_idx, (xf, yf) in enumerate(zip(d95_match_runs, d95_full_runs)):
        if not (np.isfinite(xf) and np.isfinite(yf)):
            continue
        col = base_cols[run_idx % len(base_cols)]
        xj  = xf + rng_jitter.uniform(-jitter_scale, jitter_scale)
        yj  = yf + rng_jitter.uniform(-jitter_scale, jitter_scale)
        axC.scatter(xj, yj, s=100, facecolors="white", edgecolors=col,
                    linewidth=1.5, zorder=5)

    axC.plot([xmin, xmax], [xmin, xmax], ls="--", lw=0.9, color="k")
    axC.set_xlim(xmin, xmax)
    axC.set_ylim(xmin, xmax)
    axC.grid(False)
    axC.set_box_aspect(1)

    # Panel D – מספר נקודות (subsets) לכל ריצה
    for (run_idx, yf), (_, xf) in zip(d95_full_subsets, d95_match_subsets):
        if not (np.isfinite(xf) and np.isfinite(yf)):
            continue
        col = base_cols[run_idx % len(base_cols)]
        xj  = xf + rng_jitter.uniform(-jitter_scale, jitter_scale)
        yj  = yf + rng_jitter.uniform(-jitter_scale, jitter_scale)
        axD.scatter(xj, yj, s=55, facecolors="white", edgecolors=col,
                    linewidth=1.2)

    axD.plot([xmin, xmax], [xmin, xmax], ls="--", lw=0.9, color="k")
    axD.set_xlim(xmin, xmax)
    axD.set_ylim(xmin, xmax)
    axD.grid(False)
    axD.set_box_aspect(1)

    # ---- Legend (only in Panel D) ----
    handles = [plt.Line2D([0], [0], marker='o', color='none',
                          markerfacecolor='white',
                          markeredgecolor=c, markersize=8, lw=0)
               for c in base_cols[:n_run]]
    labels  = [f"Run {i+1}" for i in range(n_run)]
    axD.legend(handles, labels,
               loc="lower right",
               fontsize=9,
               frameon=False,
               title="Runs",
               title_fontsize=10)

    # ---------------- Cosmetics -------------------
    axA.set_title(f"Predicting {tgt_nm}", color="#9C1C1C", pad=10, fontsize=15)
    axB.set_title(f"Predicting V1-match {tgt_nm}", color="#1565C0", pad=10, fontsize=15)
    for ax in (axA, axB):
        ax.grid(alpha=.25)

    for label, ax in zip("ABCD", (axA, axB, axC, axD)):
        ax.text(-0.07, 1.05, label, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")

    # shared axis labels
    for ax in (axA, axB, axC, axD):
        ax.set_ylabel(None)

    boxA, boxB, boxC, boxD = axA.get_position(), axB.get_position(), axC.get_position(), axD.get_position()
    left_col_ycenter  = 0.5 * (((boxA.y0 + boxA.y1) / 2) + ((boxB.y0 + boxB.y1) / 2))
    right_block_ycent = 0.5 * (((boxC.y0 + boxC.y1) / 2) + ((boxD.y0 + boxD.y1) / 2))
    left_ylabel_x     = boxA.x0 - 0.066
    right_ylabel_x    = boxC.x0 - 0.043
    right_xlabel_y    = min(boxC.y0, boxD.y0) - 0.058
    left_xlabel_y     = min(boxA.y0, boxB.y0) - 0.058
    left_block_xcent  = 0.5 * (((boxA.x0 + boxA.x1) / 2) + ((boxB.x0 + boxB.y1) / 2))
    right_block_xcent = 0.5 * (((boxC.x0 + boxC.x1) / 2) + ((boxD.x0 + boxD.x1) / 2))

    fig.text(left_ylabel_x, left_col_ycenter,
             rf"Mean $R^2$  (outer {outer_splits}, inner {inner_splits})",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+1, color="black")
    fig.text(right_ylabel_x, right_block_ycent,
             f"{tgt_nm} Predictive dimensions",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+1, color="#9C1C1C")

    fig.text(left_block_xcent, left_xlabel_y,
             "Predictive dimensions (d)",
             va="top", ha="center", fontsize=label_fs+1, color="black")

    fig.text(right_block_xcent, right_xlabel_y,
             "Target V1 Predictive dimensions",
             va="top", ha="center", fontsize=label_fs+1, color="#1565C0")

    # global title
    fig.suptitle(
        f"{cfg.get_monkey_name()}  |  {cfg.get_zscore_title()}  |  "
        f"{analysis_type.upper()}  (n_src={n_src_eff}, n_tgt={n_tgt_eff}, runs={n_run})",
        fontsize=16, y=.995, fontweight="bold"
    )

    # ---------------- save -------------------
    out_dir = cfg.get_data_path() / "Semedo_plots" / f"{n_run}_multisubset" / analysis_type
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_tag  = consts.REGION_ID_TO_NAME[target_region]
    file_name = (
        f"{cfg.get_monkey_name()} "
        f"({n_src_eff},{n_tgt_eff}) "
        f"{analysis_type.upper()} "
        f"({tgt_tag}).png"
    ).replace(" ", "_")

    fig.savefig(out_dir / file_name, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[✓] Subset-Semedo figure saved → {out_dir / file_name}")




def make_subset_semedo_figure(
    source_region : int = 1,      # V1
    target_region : int = 3,      # IT (or 2 = V4)
    *,
    analysis_type : str  = "residual",   # "window" | "baseline100" | "residual"
    d_max         : int  = 35,
    alpha         : float | None = None,
    outer_splits  : int  = 2,
    inner_splits  : int  = 3,
    random_state  : int  = 0,
    # ---- subset parameters ----
    n_run   : int = 5,
    n_src   : int = 113,
    n_tgt   : int = 28,
    k_subsets : int = 10
):
    """
    Subset-Semedo figure (A–D)
    Panels:
        A,B – RRR curves for multiple runs (one color per run)
        C   – single dot per run (d95_match vs. d95_full)
        D   – multiple dots per run (subset-level)
    """

    cfg     = runtime.get_cfg()
    consts  = runtime.get_consts()
    rng     = np.random.default_rng(random_state)
    trials  = cfg._load_trials()  # only for length; matrices computed with indices (not dicts)

    # ---------------- helpers -------------------
    def _matrix(rid: int, elec_idx: np.ndarray | list[int] | None, trial_idx: np.ndarray | slice | None):
        """Build matrix using new build_trial_matrix API: trials are indices/boolean/slice; electrodes by indices."""
        return cfg.build_trial_matrix(
            region_id=rid,
            analysis_type=analysis_type,
            trials=trial_idx,  # indices/slice/None
            electrode_indices=(None if elec_idx is None else np.asarray(elec_idx, dtype=int)),
            return_stimulus_ids=False
        )

    def _d95(r2_vec, ridge_val):
        idx = np.where(r2_vec >= 0.95 * float(ridge_val))[0]
        return int(idx[0] + 1) if idx.size else np.nan

    def _labeled_dot(ax, x, y, label, *, face, edge="k",
                     size=240, text_size=11, text_color="white"):
        ax.scatter([x], [y], s=size, facecolors=face, edgecolors=edge,
                   linewidths=1.5, zorder=6)
        ax.text(x, y, str(label), ha="center", va="center",
                color=text_color, fontsize=text_size, weight="bold",
                zorder=8, path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])

    def _square_limits(x_vals, y_vals, *, base_min=1, scale=1.5):
        xv = np.atleast_1d(x_vals).astype(float)
        yv = np.atleast_1d(y_vals).astype(float)
        xv = xv[np.isfinite(xv)]  # ignore NaNs
        yv = yv[np.isfinite(yv)]
        vmax = float(np.max([np.max(xv) if xv.size else base_min,
                             np.max(yv) if yv.size else base_min]))
        lim_max = int(np.ceil(scale * vmax))
        lim_max = max(lim_max, base_min + 1)
        return (base_min, lim_max)

    # ---------------- electrode indices -------------------
    rois = cfg.get_rois()
    all_src_idx = np.where(rois == source_region)[0]
    all_tgt_idx = np.where(rois == target_region)[0]
    n_src_eff = min(n_src, all_src_idx.size)
    n_tgt_eff = min(n_tgt, all_tgt_idx.size)

    base_cols = ["#9C1C1C", "#1565C0", "#2E7D32", "#7B1FA2", "#F57C00", "#00796B"]
    col_cycle = cycle(base_cols)
    tgt_nm = consts.REGION_ID_TO_NAME[target_region]

    # ---------------- figure layout -------------------
    fig = plt.figure(figsize=(14, 13), dpi=400)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.15, hspace=0.30)
    gs  = gridspec.GridSpec(2, 2)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[1, 1])

    label_fs, tick_fs = 17, 12
    for ax in (axA, axB, axC, axD):
        ax.tick_params(labelsize=tick_fs)
        ax.xaxis.labelpad = 16
        ax.yaxis.labelpad = 20

    # ---------------- per-run computation -------------------
    d95_full_runs, d95_match_runs = [], []
    d95_full_subsets, d95_match_subsets = [], []
    dims = np.arange(1, d_max + 1)

    for run_idx in range(n_run):
        col = next(col_cycle)

        # random electrode subsets
        src_subset = rng.choice(all_src_idx, n_src_eff, replace=False)
        tgt_subset = rng.choice(all_tgt_idx, n_tgt_eff, replace=False)

        # pick "matched" V1 electrodes by smallest mean activity across ALL trials
        mean_src  = _matrix(source_region, src_subset, trial_idx=None).mean(0)
        order_src = np.argsort(mean_src)
        match_src  = src_subset[order_src[:n_tgt_eff]]
        src_remain = src_subset[~np.isin(src_subset, match_src)]

        # FULL runs (all trials → trial_idx=None)
        X_full = _matrix(source_region, src_subset, trial_idx=None)
        Y_full = _matrix(target_region, tgt_subset, trial_idx=None)
        res_full = RRR_Centered_matching._performance_from_mats(
            Y_full, X_full, d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state + run_idx
        )

        # MATCH runs (all trials)
        X_match = _matrix(source_region, src_remain, trial_idx=None)
        Y_match = _matrix(source_region, match_src,  trial_idx=None)
        res_match = RRR_Centered_matching._performance_from_mats(
            Y_match, X_match, d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state + run_idx
        )

        # ---- Plot Panels A + B ----
        axA.errorbar(dims, res_full["rrr_R2_mean"], yerr=res_full["rrr_R2_sem"],
                     fmt="o-", ms=3.8, lw=1.35, capsize=3, color=col)
        axA.scatter([1], [res_full["ridge_R2_mean"]],
                    marker="^", s=90, color=col, edgecolors="k")
        axA.set_box_aspect(1)

        axB.errorbar(dims, res_match["rrr_R2_mean"], yerr=res_match["rrr_R2_sem"],
                     fmt="o-", ms=3.8, lw=1.35, capsize=3, color=col)
        axB.scatter([1], [res_match["ridge_R2_mean"]],
                    marker="^", s=90, color=col, edgecolors="k")
        axB.set_box_aspect(1)

        # ---- Global d95 values ----
        d95_f = _d95(res_full["rrr_R2_mean"],  res_full["ridge_R2_mean"])
        d95_m = _d95(res_match["rrr_R2_mean"], res_match["ridge_R2_mean"])
        d95_full_runs.append(d95_f)
        d95_match_runs.append(d95_m)

        if np.isfinite(d95_f):
            r2d = res_full["rrr_R2_mean"][d95_f - 1]
            _labeled_dot(axA, d95_f, r2d, int(d95_f), face=col)
        if np.isfinite(d95_m):
            r2d = res_match["rrr_R2_mean"][d95_m - 1]
            _labeled_dot(axB, d95_m, r2d, int(d95_m), face=col)

        # ---- build k_subsets points: split TRIAL INDICES (not dicts) ----
        idx_trials = rng.permutation(len(trials))
        chunks = np.array_split(idx_trials, k_subsets)

        for ch_idx in chunks:
            # subset by trial indices
            res_f_sub = RRR_Centered_matching._performance_from_mats(
                _matrix(target_region,  tgt_subset, ch_idx),
                _matrix(source_region, src_subset, ch_idx),
                d_max=d_max, alpha=alpha,
                outer_splits=max(2, outer_splits),
                inner_splits=max(2, inner_splits),
                random_state=random_state + run_idx
            )
            res_m_sub = RRR_Centered_matching._performance_from_mats(
                _matrix(source_region, match_src,  ch_idx),
                _matrix(source_region, src_remain, ch_idx),
                d_max=d_max, alpha=alpha,
                outer_splits=max(2, outer_splits),
                inner_splits=max(2, inner_splits),
                random_state=random_state + run_idx
            )
            d95_f_sub = _d95(res_f_sub["rrr_R2_mean"],  res_f_sub["ridge_R2_mean"])
            d95_m_sub = _d95(res_m_sub["rrr_R2_mean"], res_m_sub["ridge_R2_mean"])
            d95_full_subsets.append((run_idx, d95_f_sub))
            d95_match_subsets.append((run_idx, d95_m_sub))

    # ---------------- Panels C + D -------------------
    valid_x = [int(x) for x in d95_match_runs if np.isfinite(x)]
    valid_y = [int(y) for y in d95_full_runs  if np.isfinite(y)]
    xmin, xmax = _square_limits(valid_x + [1], valid_y + [1])

    rng_jitter   = np.random.default_rng(random_state)
    jitter_scale = 0.15

    # Panel C – נקודה אחת לכל ריצה
    for run_idx, (xf, yf) in enumerate(zip(d95_match_runs, d95_full_runs)):
        if not (np.isfinite(xf) and np.isfinite(yf)):
            continue
        col = base_cols[run_idx % len(base_cols)]
        xj  = xf + rng_jitter.uniform(-jitter_scale, jitter_scale)
        yj  = yf + rng_jitter.uniform(-jitter_scale, jitter_scale)
        axC.scatter(xj, yj, s=100, facecolors="white", edgecolors=col,
                    linewidth=1.5, zorder=5)

    axC.plot([xmin, xmax], [xmin, xmax], ls="--", lw=0.9, color="k")
    axC.set_xlim(xmin, xmax)
    axC.set_ylim(xmin, xmax)
    axC.grid(False)
    axC.set_box_aspect(1)

    # Panel D – מספר נקודות (subsets) לכל ריצה
    for (run_idx, yf), (_, xf) in zip(d95_full_subsets, d95_match_subsets):
        if not (np.isfinite(xf) and np.isfinite(yf)):
            continue
        col = base_cols[run_idx % len(base_cols)]
        xj  = xf + rng_jitter.uniform(-jitter_scale, jitter_scale)
        yj  = yf + rng_jitter.uniform(-jitter_scale, jitter_scale)
        axD.scatter(xj, yj, s=55, facecolors="white", edgecolors=col,
                    linewidth=1.2)

    axD.plot([xmin, xmax], [xmin, xmax], ls="--", lw=0.9, color="k")
    axD.set_xlim(xmin, xmax)
    axD.set_ylim(xmin, xmax)
    axD.grid(False)
    axD.set_box_aspect(1)

    # ---- Legend (only in Panel D) ----
    handles = [plt.Line2D([0], [0], marker='o', color='none',
                          markerfacecolor='white',
                          markeredgecolor=c, markersize=8, lw=0)
               for c in base_cols[:n_run]]
    labels  = [f"Run {i+1}" for i in range(n_run)]
    axD.legend(handles, labels,
               loc="lower right",
               fontsize=9,
               frameon=False,
               title="Runs",
               title_fontsize=10)

    # ---------------- Cosmetics -------------------
    axA.set_title(f"Predicting {tgt_nm}", color="#9C1C1C", pad=10, fontsize=15)
    axB.set_title(f"Predicting V1-match {tgt_nm}", color="#1565C0", pad=10, fontsize=15)
    for ax in (axA, axB):
        ax.grid(alpha=.25)

    for label, ax in zip("ABCD", (axA, axB, axC, axD)):
        ax.text(-0.07, 1.05, label, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")

    # shared axis labels
    for ax in (axA, axB, axC, axD):
        ax.set_ylabel(None)

    boxA, boxB, boxC, boxD = axA.get_position(), axB.get_position(), axC.get_position(), axD.get_position()
    left_col_ycenter  = 0.5 * (((boxA.y0 + boxA.y1) / 2) + ((boxB.y0 + boxB.y1) / 2))
    right_block_ycent = 0.5 * (((boxC.y0 + boxC.y1) / 2) + ((boxD.y0 + boxD.y1) / 2))
    left_ylabel_x     = boxA.x0 - 0.066
    right_ylabel_x    = boxC.x0 - 0.043
    right_xlabel_y    = min(boxC.y0, boxD.y0) - 0.058
    left_xlabel_y     = min(boxA.y0, boxB.y0) - 0.058
    left_block_xcent  = 0.5 * (((boxA.x0 + boxA.x1) / 2) + ((boxB.x0 + boxB.y1) / 2))
    right_block_xcent = 0.5 * (((boxC.x0 + boxC.x1) / 2) + ((boxD.x0 + boxD.x1) / 2))

    fig.text(left_ylabel_x, left_col_ycenter,
             rf"Mean $R^2$  (outer {outer_splits}, inner {inner_splits})",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+1, color="black")
    fig.text(right_ylabel_x, right_block_ycent,
             f"{tgt_nm} Predictive dimensions",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+1, color="#9C1C1C")

    fig.text(left_block_xcent, left_xlabel_y,
             "Predictive dimensions (d)",
             va="top", ha="center", fontsize=label_fs+1, color="black")

    fig.text(right_block_xcent, right_xlabel_y,
             "Target V1 Predictive dimensions",
             va="top", ha="center", fontsize=label_fs+1, color="#1565C0")

    # global title
    fig.suptitle(
        f"{cfg.get_monkey_name()}  |  {cfg.get_zscore_title()}  |  "
        f"{analysis_type.upper()}  (n_src={n_src_eff}, n_tgt={n_tgt_eff}, runs={n_run})",
        fontsize=16, y=.995, fontweight="bold"
    )

    # ---------------- save -------------------
    out_dir = cfg.get_data_path() / "Semedo_plots" / f"{n_run}_multisubset" / analysis_type
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_tag  = consts.REGION_ID_TO_NAME[target_region]
    file_name = (
        f"{cfg.get_monkey_name()} "
        f"({n_src_eff},{n_tgt_eff}) "
        f"{analysis_type.upper()} "
        f"({tgt_tag}).png"
    ).replace(" ", "_")

    fig.savefig(out_dir / file_name, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[✓] Subset-Semedo figure saved → {out_dir / file_name}")





"""
runtime.set_cfg("Monkey N", 1)
make_semedo_figure(
                            1,
                            3,
                            analysis_type="residual",
                            k_subsets=10,
                        )
"""

monkeys = ["Monkey N", "Monkey F"]
zscore_codes = [1, 2, 3, 4]
methods = ["window", "baseline100", "residual"]
targets = [2, 3]
for monkey in monkeys:
        for zc in zscore_codes:
            runtime.set_cfg(monkey, zc)
            consts = runtime.get_consts()
            region_name = consts.REGION_ID_TO_NAME

            print(f"\n[CFG] {monkey} | Z={zc} | Z-Title='{runtime.get_cfg().get_zscore_title()}'")

            for method in methods:
                for tgt in targets:
                        print(f"  → Running: {monkey}, Z={zc}, method='{method}', {region_name[1]}→{region_name[tgt]}, k_subsets={10}")
                        make_subset_semedo_figure(
                            1,
                            tgt,
                            analysis_type=method,
                            k_subsets=10,
                        )





