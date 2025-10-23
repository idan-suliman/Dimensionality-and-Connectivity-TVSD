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
    outer_splits: int = 2,
    inner_splits: int = 3,
    random_state: int = 0,
    k_subsets: int | None = None,    # None → repetitions; int → K random subsets (used for panel D only)
):
    """Generate Semedo-style multi-panel figure (A–D) summarizing RRR performance and predictive subspace dimensionality."""

    # ====================== Load and organize trials ======================
    trials = runtime.get_cfg()._load_trials()

    # Create repetition index if not available (needed for subset division)
    if k_subsets is None and "rep_idx" not in trials[0]:
        if "allmat_row" not in trials[0]:
            raise RuntimeError("allmat_row missing – cannot derive repetitions.")
        for tr in trials:
            tr["rep_idx"] = int(tr["allmat_row"][3]) - 1  # Convert 1–30 → 0–29

    # Group trials by repetitions or random subsets
    if k_subsets is None:
        rep_ids  = np.unique([tr["rep_idx"] for tr in trials])
        groups   = [[tr for tr in trials if tr["rep_idx"] == g] for g in rep_ids]
        label_D  = "Repetitions"
        id_print = lambda g: f"rep {g:2}"
    else:
        rng     = np.random.default_rng(random_state)
        idxs    = np.arange(len(trials)); rng.shuffle(idxs)
        splits  = np.array_split(idxs, k_subsets)
        groups  = [[trials[i] for i in part] for part in splits]
        label_D = f"{k_subsets} random subsets"
        id_print = lambda g: f"set {g:2}"

    # ============================ Helper functions ============================
    def _d95(rrr_mean: np.ndarray, ridge_mean: float) -> int:
        """Return dimensionality at which RRR reaches 95% of ridge performance."""
        thr = 0.95 * float(ridge_mean)
        idx = np.where(rrr_mean >= thr)[0]
        return int(idx[0] + 1) if idx.size else int(d_max)

    def _perf(match: bool):
        """Run full RRR performance computation (either matched or unmatched)."""
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
        """Draw a labeled circle marker with the number centered and outlined for visibility."""
        ax.scatter([x], [y], s=size, facecolors=face, edgecolors=edge,
                   linewidths=1.6, zorder=7)
        ax.text(x, y, str(label), ha="center", va="center",
                color=text_color, fontsize=text_size, weight="bold",
                zorder=9, path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])

    def _square_limits(x_vals, y_vals, *, base_min=1, scale=1.5):
        """Compute symmetric axis limits enlarged by a given scale factor."""
        vmax = float(np.max([np.max(np.atleast_1d(x_vals)),
                             np.max(np.atleast_1d(y_vals))]))
        lim_max = int(np.ceil(scale * vmax))
        lim_max = max(lim_max, base_min + 1)
        return (base_min, lim_max)

    def _jitter(values, rng: np.random.Generator, *, scale: float = 0.15):
        """Add small uniform noise to integer-valued coordinates for visual separation."""
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

    # ====================== Compute per-group dimensionalities (Panel D) ======================
    d95_full_rep, d95_match_rep = [], []

    def _mat(rid: int, idx: np.ndarray, sub_tr: list[dict]) -> np.ndarray:
        """Assemble mean MUA matrix using the config helper for consistent preprocessing."""
        if len(sub_tr) == 0:
            raise ValueError("Trial subset cannot be empty when building matrices.")

        mat = runtime.get_cfg().build_trial_matrix(
            region_id=rid,
            analysis_type=analysis_type,
            trials=sub_tr,
            rois=np.asarray(idx, dtype=int),
        )

        # Ensure downstream routines receive float32 arrays (build_trial_matrix already returns
        # float32, but we guard against older caches).
        if mat.dtype != np.float32:
            mat = mat.astype(np.float32, copy=False)

        return mat

    for g_idx, grp in enumerate(groups):
        # Unmatched (V1 → target)
        res_f = RRR_Centered_matching._performance_from_mats(
            _mat(target_region, tgt_idx_full, grp),
            _mat(1, src_idx_full, grp),
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state
        )
        d95_full_rep.append(_d95(res_f["rrr_R2_mean"], res_f["ridge_R2_mean"]))

        # Matched (V1-match → remaining V1)
        res_m = RRR_Centered_matching._performance_from_mats(
            _mat(1, match_idx, grp),
            _mat(1, src_idx_full[~np.isin(src_idx_full, match_idx)], grp),
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
    fig = plt.figure(figsize=(14, 9.4), dpi=400)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.12,
                        wspace=0.26, hspace=0.30)

    gs  = gridspec.GridSpec(2, 2, width_ratios=[3.2, 2.6])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[1, 1])

    dims   = np.arange(1, d_max + 1)
    tgt    = runtime.get_consts().REGION_ID_TO_NAME[target_region]
    colA, colB = "#9C1C1C", "#1565C0"
    label_fs, tick_fs = 15, 12

    for ax in (axA, axB, axC, axD):
        ax.tick_params(labelsize=tick_fs)
        ax.xaxis.labelpad = 16
        ax.yaxis.labelpad = 20
        ax.set_box_aspect(1)
        
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
    axA.text(-0.12, 1.05, "A", transform=axA.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")

    # =============================== Panel B (Matched subset RRR curve) ===============================
    axB.errorbar(dims, perf_match["rrr_R2_mean"], yerr=perf_match["rrr_R2_sem"],
                 fmt="o-", ms=3.8, lw=1.35, capsize=3, color=colB, zorder=2)
    axB.scatter([1], [perf_match["ridge_R2_mean"]], marker="^", s=90,
                color=colB, edgecolors="k", zorder=3)

    if np.isfinite(d95_match_g):
        r2d = perf_match["rrr_R2_mean"][int(d95_match_g) - 1]
        _labeled_dot(axB, int(d95_match_g), float(r2d), int(d95_match_g),
                     face=colB, edge="k", size=240, text_size=12, text_color="white")

    axB.set_xlabel("Predictive dimensions (d)", fontsize=label_fs, labelpad=16)
    axB.set_title(f"Predicting V1-match {tgt}", color=colB, pad=10, fontsize=15)
    axB.grid(alpha=.25)
    axB.text(-0.12, 1.05, "B", transform=axB.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")

    # =============================== Panel C (Global comparison: A vs B) ===============================
    if np.isfinite(d95_match_g) and np.isfinite(d95_full_g):
        xg, yg = int(d95_match_g), int(d95_full_g)
        xmin, xmax = _square_limits([xg], [yg], base_min=1, scale=1.5)
    else:
        xmin, xmax = _square_limits([1], [1], base_min=1, scale=1.5)

    rng_jitter = np.random.default_rng(random_state)

    axC.plot([xmin, xmax], [xmin, xmax], ls="--", lw=0.9, color="k")  # y=x reference
    if np.isfinite(d95_match_g) and np.isfinite(d95_full_g):
        jitter_x = _jitter([xg], rng_jitter)[0]
        jitter_y = _jitter([yg], rng_jitter)[0]
        axC.scatter([jitter_x], [jitter_y], s=175, facecolors="white", edgecolors="black", zorder=4)
    axC.set_xlim(xmin, xmax)
    axC.set_ylim(xmin, xmax)
    ticks_c = np.arange(xmin, xmax + 1, max(1, int(np.ceil((xmax - xmin) / 6))))
    axC.set_xticks(ticks_c)
    axC.set_yticks(ticks_c)
    axC.grid(False)
    axC.text(-0.12, 1.05, "C", transform=axC.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")

    # =============================== Panel D (Per-group subsets) ===============================
    if len(d95_match_rep) and len(d95_full_rep):
        xmin_d, xmax_d = _square_limits(d95_match_rep, d95_full_rep, base_min=1, scale=1.5)
    else:
        xmin_d, xmax_d = (1, max(2, int(np.ceil(1.5 * d_max))))

    jitter_match = _jitter(d95_match_rep, rng_jitter)
    jitter_full = _jitter(d95_full_rep, rng_jitter)
    axD.scatter(jitter_match, jitter_full, s=60, facecolors="white", edgecolors="black")
    axD.plot([xmin_d, xmax_d], [xmin_d, xmax_d], ls="--", lw=0.9, color="k")
    axD.set_xlim(xmin_d, xmax_d)
    axD.set_ylim(xmin_d, xmax_d)
    ticks_d = np.arange(xmin_d, xmax_d + 1, max(1, int(np.ceil((xmax_d - xmin_d) / 6))))
    axD.set_xticks(ticks_d)
    axD.set_yticks(ticks_d)
    axD.text(0.98, 0.02, label_D, transform=axD.transAxes, ha="right", va="bottom", fontsize=11)
    axD.grid(False)
    axD.text(-0.12, 1.05, "D", transform=axD.transAxes,
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
    right_block_xcent = 0.5 * (((boxC.x0 + boxC.x1) / 2) + ((boxD.x0 + boxD.x1) / 2))

    fig.text(left_ylabel_x, left_col_ycenter,
             rf"Mean $R^2$  (outer {outer_splits}, inner {inner_splits})",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+1, color="black")

    fig.text(right_ylabel_x, right_block_ycent,
             f"{runtime.get_consts().REGION_ID_TO_NAME[target_region]} Predictive dimensions",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+1, color=colA)

    fig.text(right_block_xcent, right_xlabel_y,
             "Target V1 Predictive dimensions",
             va="top", ha="center", fontsize=label_fs+1, color=colB)

    # =============================== Main title & saving ===============================
    top_row_ymax = max(boxA.y1, boxC.y1)
    fig.suptitle(
        f"{runtime.get_cfg().get_monkey_name()}  |  {runtime.get_cfg().get_zscore_title()}  |  {analysis_type.upper()}",
        fontsize=16, y=min(0.998, top_row_ymax + 0.080), fontweight="bold"
    )

    # Save into folder reflecting current subset configuration
    out_dir = runtime.get_cfg().get_data_path() / "Semedo_plots" / f"Semedo_full_electrodes_{k_subsets}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"sub{k_subsets}" if k_subsets else "rep30"
    tgt    = runtime.get_consts().REGION_ID_TO_NAME[target_region]
    fname  = f"{runtime.get_cfg().get_monkey_name().replace(' ', '')}_semendo_{suffix}_{analysis_type}_V1_to_{tgt}.png"

    fig.savefig(out_dir / fname, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[✓] Semedo figure saved → {out_dir/fname}")


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
    trials  = cfg._load_trials()

    # ---------------- build matrix -------------------
    def _matrix(rid: int, idx: np.ndarray, sub_trials: list[dict]) -> np.ndarray:
        return cfg.build_trial_matrix(
            region_id=rid,
            analysis_type=analysis_type,
            trials=sub_trials,
            rois=idx,
        )


    def _d95(r2_vec, ridge_val):
        idx = np.where(r2_vec >= 0.95*ridge_val)[0]
        return idx[0] + 1 if idx.size else np.nan

    def _labeled_dot(ax, x, y, label, *, face, edge="k",
                     size=240, text_size=11, text_color="white"):
        ax.scatter([x], [y], s=size, facecolors=face, edgecolors=edge,
                   linewidths=1.5, zorder=6)
        ax.text(x, y, str(label), ha="center", va="center",
                color=text_color, fontsize=text_size, weight="bold",
                zorder=8, path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])

    def _square_limits(x_vals, y_vals, *, base_min=1, scale=1.5):
        vmax = float(np.max([np.max(np.atleast_1d(x_vals)),
                             np.max(np.atleast_1d(y_vals))]))
        lim_max = int(np.ceil(scale * vmax))
        lim_max = max(lim_max, base_min + 1)
        return (base_min, lim_max)

    # ---------------- electrode indices -------------------
    rois = cfg.get_rois()
    all_src_idx = np.where(rois == source_region)[0]
    all_tgt_idx = np.where(rois == target_region)[0]

    base_cols = ["#9C1C1C", "#1565C0", "#2E7D32", "#7B1FA2", "#F57C00", "#00796B"]
    col_cycle = cycle(base_cols)
    tgt_nm = consts.REGION_ID_TO_NAME[target_region]

    # ---------------- figure layout -------------------
    fig = plt.figure(figsize=(12.6, 8.2), dpi=400)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.12,
                        wspace=0.38, hspace=0.26)
    gs  = gridspec.GridSpec(2, 3, width_ratios=[3.2, 2.5, 2.5])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[:, 1])
    axD = fig.add_subplot(gs[:, 2])

    label_fs, tick_fs = 15, 12
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
        src_subset = rng.choice(all_src_idx, n_src, replace=False)
        tgt_subset = rng.choice(all_tgt_idx, n_tgt, replace=False)

        mean_src = _matrix(source_region, src_subset, trials).mean(0)
        order_src = np.argsort(mean_src)
        match_src  = src_subset[order_src[:n_tgt]]
        src_remain = src_subset[~np.isin(src_subset, match_src)]

        X_full = _matrix(source_region, src_subset, trials)
        Y_full = _matrix(target_region, tgt_subset, trials)
        res_full = RRR_Centered_matching._performance_from_mats(
            Y_full, X_full, d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state + run_idx
        )

        X_match = _matrix(source_region, src_remain, trials)
        Y_match = _matrix(source_region, match_src, trials)
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

        axB.errorbar(dims, res_match["rrr_R2_mean"], yerr=res_match["rrr_R2_sem"],
                     fmt="o-", ms=3.8, lw=1.35, capsize=3, color=col)
        axB.scatter([1], [res_match["ridge_R2_mean"]],
                    marker="^", s=90, color=col, edgecolors="k")

        # ---- Global d95 values ----
        d95_f = _d95(res_full["rrr_R2_mean"], res_full["ridge_R2_mean"])
        d95_m = _d95(res_match["rrr_R2_mean"], res_match["ridge_R2_mean"])
        d95_full_runs.append(d95_f)
        d95_match_runs.append(d95_m)

        if np.isfinite(d95_f):
            r2d = res_full["rrr_R2_mean"][d95_f - 1]
            _labeled_dot(axA, d95_f, r2d, int(d95_f), face=col)
        if np.isfinite(d95_m):
            r2d = res_match["rrr_R2_mean"][d95_m - 1]
            _labeled_dot(axB, d95_m, r2d, int(d95_m), face=col)

        # ---- build k_subsets points ----
        idx_trials = rng.permutation(len(trials))
        chunks = np.array_split(idx_trials, k_subsets)

        for ch in chunks:
            sub_tr = [trials[i] for i in ch]
            res_f_sub = RRR_Centered_matching._performance_from_mats(
                _matrix(target_region, tgt_subset, sub_tr),
                _matrix(source_region, src_subset, sub_tr),
                d_max=d_max, alpha=alpha,
                outer_splits=max(2, outer_splits),
                inner_splits=max(2, inner_splits),
                random_state=random_state + run_idx
            )
            res_m_sub = RRR_Centered_matching._performance_from_mats(
                _matrix(source_region, match_src, sub_tr),
                _matrix(source_region, src_remain, sub_tr),
                d_max=d_max, alpha=alpha,
                outer_splits=max(2, outer_splits),
                inner_splits=max(2, inner_splits),
                random_state=random_state + run_idx
            )
            d95_f_sub = _d95(res_f_sub["rrr_R2_mean"], res_f_sub["ridge_R2_mean"])
            d95_m_sub = _d95(res_m_sub["rrr_R2_mean"], res_m_sub["ridge_R2_mean"])
            d95_full_subsets.append((run_idx, d95_f_sub))
            d95_match_subsets.append((run_idx, d95_m_sub))

    # ---------------- Panels C + D -------------------
    xmin, xmax = _square_limits(d95_match_runs + [1], d95_full_runs + [1])

    # Panel C – נקודה אחת לכל ריצה
    for run_idx, (xf, yf) in enumerate(zip(d95_match_runs, d95_full_runs)):
        col = base_cols[run_idx % len(base_cols)]
        axC.scatter(xf, yf, s=100, facecolors="white", edgecolors=col,
                    linewidth=1.5, zorder=5)

    axC.plot([xmin, xmax], [xmin, xmax], ls="--", lw=0.9, color="k")
    axC.set_xlim(xmin, xmax)
    axC.set_ylim(xmin, xmax)
    axC.grid(False)

    # Panel D – מספר נקודות (subsets) לכל ריצה
    for (run_idx, yf), (_, xf) in zip(d95_full_subsets, d95_match_subsets):
        col = base_cols[run_idx % len(base_cols)]
        axD.scatter(xf, yf, s=55, facecolors="white", edgecolors=col,
                    linewidth=1.2)

    axD.plot([xmin, xmax], [xmin, xmax], ls="--", lw=0.9, color="k")
    axD.set_xlim(xmin, xmax)
    axD.set_ylim(xmin, xmax)
    axD.grid(False)

    # ---- Legend (only in Panel D) ----
    handles = [plt.Line2D([0], [0],
                        marker='o', color='none',
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
    tgt_label = f"{tgt_nm} Predictive dimensions"
    axA.set_title(f"Predicting {tgt_nm}", color="#9C1C1C", pad=10, fontsize=15)
    axB.set_title(f"Predicting V1-match {tgt_nm}", color="#1565C0", pad=10, fontsize=15)
    axB.set_xlabel("Predictive dimensions (d)", fontsize=label_fs)
    for ax in (axA, axB): ax.grid(alpha=.25)

    for label, ax in zip("ABCD", (axA, axB, axC, axD)):
        ax.text(-0.07, 1.01, label, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=20, fontweight="bold")

    # shared axis labels
    for ax in (axA, axB, axC, axD): ax.set_ylabel(None)
    boxA, boxB, boxC, boxD = axA.get_position(), axB.get_position(), axC.get_position(), axD.get_position()
    left_col_ycenter = 0.5 * (((boxA.y0 + boxA.y1) / 2) + ((boxB.y0 + boxB.y1) / 2))
    right_block_ycent = 0.5 * (((boxC.y0 + boxC.y1) / 2) + ((boxD.y0 + boxD.y1) / 2))
    left_ylabel_x = boxA.x0 - 0.066
    right_ylabel_x = boxC.x0 - 0.043
    right_xlabel_y = min(boxC.y0, boxD.y0) - 0.058
    right_block_xcent = 0.5 * (((boxC.x0 + boxC.x1) / 2) + ((boxD.x0 + boxD.x1) / 2))

    fig.text(left_ylabel_x, left_col_ycenter,
             rf"Mean $R^2$  (outer {outer_splits}, inner {inner_splits})",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+1, color="black")
    fig.text(right_ylabel_x, right_block_ycent,
             f"{tgt_nm} Predictive dimensions",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+1, color="#9C1C1C")
    fig.text(right_block_xcent, right_xlabel_y,
             "Target V1 Predictive dimensions",
             va="top", ha="center", fontsize=label_fs+1, color="#1565C0")

    # global title
    fig.suptitle(
        f"{runtime.get_cfg().get_monkey_name()}  |  {runtime.get_cfg().get_zscore_title()}  |  "
        f"{analysis_type.upper()}  (n_src={n_src}, n_tgt={n_tgt}, runs={n_run})",
        fontsize=16, y=.995, fontweight="bold"
    )

    # ---------------- save -------------------
    scale_factor = 1  # 512 electrodes in baseline V1
    folder_tag = f"{n_run}_multisubset×{scale_factor}"
    out_dir = runtime.get_cfg().get_data_path() / "Semedo_plots" / folder_tag / analysis_type
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_tag  = runtime.get_consts().REGION_ID_TO_NAME[target_region]
    file_name = (
        f"{runtime.get_cfg().get_monkey_name()} "
        f"({n_src},{n_tgt}) "
        f"{analysis_type.upper()} "
        f"({tgt_tag}).png"
    ).replace(" ", "_")

    fig.savefig(out_dir / file_name, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[✓] Subset-Semedo figure saved → {out_dir / file_name}")




runtime.set_cfg("Monkey F",1)
make_semedo_figure(1,2,analysis_type="baseline100",k_subsets=10)
