# Semedo.py
from rrr import RRR_Centered_matching
from runtime import runtime
from matchingSubset import MATCHINGSUBSET
import numpy as np, matplotlib.pyplot as plt    
from matplotlib import gridspec
from itertools import cycle
import matplotlib.patheffects as pe
import time
from datetime import datetime

# ============================ Helper functions (module-level) ============================
# Returns the minimal dimension d where the RRR curve reaches ≥95% of the Ridge R² (or d_max if never reached).
def d95_from_curves(rrr_mean: np.ndarray, ridge_mean: float, d_max: int) -> int:
    """
    Compute the smallest dimension d such that RRR(d) ≥ 0.95 * Ridge_R².
    Falls back to `d_max` if the threshold is not reached.

    Parameters
    ----------
    rrr_mean : np.ndarray
        Mean R² values across dimensions for RRR (length = d_max).
    ridge_mean : float
        Mean R² of the Ridge baseline.
    d_max : int
        Maximum dimension considered.

    Returns
    -------
    int
        Minimal dimension d (1-based) meeting the 95% criterion, or d_max.
    """
    thr = 0.95 * float(ridge_mean)
    idx = np.where(rrr_mean >= thr)[0]
    return int(idx[0] + 1) if idx.size else int(d_max)


# Thin wrapper that calls `RRR_Centered_matching.performance` with explicit, named arguments.
def perf_wrapper(
    source_region: int,
    target_region: int,
    *,
    analysis_type: str,
    match_to_target: bool,
    d_max: int,
    alpha: float | None,
    outer_splits: int,
    inner_splits: int,
    random_state: int,
    trial_subset: np.ndarray | None = None,
):
    """
    Unified interface for both global and subset-level RRR evaluations.
    If `trial_subset` is provided, builds data matrices only for those trials
    and runs `_performance_from_mats`; otherwise, uses the full built-in
    `RRR_Centered_matching.performance` logic.

    Returns
    -------
    dict
        A dictionary with RRR and Ridge performance metrics.
    """

    # --- subset version ---
    if trial_subset is not None:
        rois = runtime.get_cfg().get_rois()
        src_idx_full = np.where(rois == source_region)[0]
        tgt_idx_full = np.where(rois == target_region)[0]

        if match_to_target:
            # load / create match subset indices
            match_idx = get_match_subset_indices(source_region, target_region, analysis_type=analysis_type)
            remain_mask    = ~np.isin(src_idx_full, match_idx)
            src_idx_remain = src_idx_full[remain_mask]

            X = build_trial_matrix(source_region, analysis_type, trial_subset, src_idx_remain)  # V1 minus match
            Y = build_trial_matrix(source_region, analysis_type, trial_subset, match_idx)       # match subset
        else:
            X = build_trial_matrix(source_region, analysis_type, trial_subset, src_idx_full)    # V1
            Y = build_trial_matrix(target_region, analysis_type, trial_subset, tgt_idx_full)    # Target region

        return RRR_Centered_matching._performance_from_mats(
            Y, X,
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state
        )

    # --- full version (all trials) ---
    return RRR_Centered_matching.performance(
        source_region, target_region,
        analysis_type=analysis_type,
        match_to_target=match_to_target,
        d_max=d_max, alpha=alpha,
        outer_splits=outer_splits, inner_splits=inner_splits,
        random_state=random_state
    )


# Plots a point and draws a bold, stroked text label centered on that point.
def labeled_dot(ax, x, y, label, *, face, edge: str = "k",
                size: float = 240, text_size: float = 12, text_color: str = "white"):
    """
    Draw a filled marker with a legible label (white text with black stroke)
    directly on top of the point.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x, y : float
        Coordinates of the point.
    label : Any
        Text label to place over the point.
    face : Any
        Face color for the marker.
    edge : str
        Edge color for the marker.
    size : float
        Marker size (points^2).
    text_size : float
        Font size of the overlaid label.
    text_color : str
        Color of the overlaid label.
    """
    ax.scatter([x], [y], s=size, facecolors=face, edgecolors=edge,
               linewidths=1.6, zorder=7)
    ax.text(x, y, str(label), ha="center", va="center",
            color=text_color, fontsize=text_size, weight="bold",
            zorder=9, path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])


# Computes symmetric square axis limits that cover both x and y value ranges.
def square_limits(x_vals, y_vals, *, base_min: int = 1, scale: float = 1.5) -> tuple[int, int]:
    """
    Compute square plot limits that jointly cover x and y values and leave some headroom.

    Parameters
    ----------
    x_vals, y_vals : array-like
        Collections of x/y values to be covered by the limits.
    base_min : int
        The minimal axis lower bound (typically 1 for dimensionality axes).
    scale : float
        Multiplicative factor enlarging the upper bound for readability.

    Returns
    -------
    (int, int)
        (xmin, xmax) to be applied to both axes for a square aspect.
    """
    vmax = float(np.max([np.max(np.atleast_1d(x_vals)),
                         np.max(np.atleast_1d(y_vals))]))
    lim_max = int(np.ceil(scale * vmax))
    lim_max = max(lim_max, base_min + 1)
    return (base_min, lim_max)


# Adds small uniform noise to values to reduce overplotting of identical integers.
def jitter(values, rng: np.random.Generator, *, scale: float = 0.15) -> np.ndarray:
    """
    Apply small uniform jitter to (typically integer) values to visually separate overlapping points.

    Parameters
    ----------
    values : array-like
        Values to jitter (often integer d95 values).
    rng : np.random.Generator
        Random number generator (for reproducibility).
    scale : float
        Half-width of the uniform jitter range [-scale, +scale].

    Returns
    -------
    np.ndarray
        Jittered values (float).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    return arr + rng.uniform(-scale, scale, size=arr.shape)


# Builds trial-index groups by repetition (30 reps) or into K random subsets.
def build_groups_by_rep_or_subsets(
    trials: list[dict],
    *,
    k_subsets: int | None,
    random_state: int
):
    """
    Group trial indices by repetition (if k_subsets is None) or into K random subsets.
    The repetition index is read from ALLMAT row #4 (1-based) and converted to 0..29.

    Parameters
    ----------
    trials : list[dict]
        Trial records as returned by `runtime.get_cfg()._load_trials()`.
        Each trial must contain 'allmat_row'.
    k_subsets : int | None
        If None, group by repetition; otherwise split all trial indices into `k_subsets` parts.
    random_state : int
        Seed for deterministic shuffling when creating random subsets.

    Returns
    -------
    groups : list[np.ndarray]
        A list of index arrays (trial indices) for each group.
    label_D : str
        Human-readable label for panel D ("Repetitions" or "K random subsets").
    id_print : Callable[[int], str]
        Function that formats group indices (e.g., "rep  3", "set  7").
    rep_arr : np.ndarray
        The repetition index (0..29) for each trial, aligned with `trials`.
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


# Constructs a trials × electrodes matrix for a region, optionally restricted by trial/electrode indices.
def build_trial_matrix(
    region_id: int,
    analysis_type: str,
    trial_idx: np.ndarray | slice | None,
    electrode_indices: np.ndarray | None,
) -> np.ndarray:
    """
    Convenience wrapper over cfg.build_trial_matrix to build a data matrix for a given region,
    limited to specific trial indices and an optional electrode subset.

    Returns
    -------
    np.ndarray
        Typically shape (n_trials, n_electrodes) after the chosen analysis/averaging pipeline.
    """
    return runtime.get_cfg().build_trial_matrix(
        region_id=region_id,
        analysis_type=analysis_type,
        trials=trial_idx,  # IMPORTANT: TRIAL INDICES (ints), not stimulus IDs
        electrode_indices=(None if electrode_indices is None else np.asarray(electrode_indices, dtype=int)),
        return_stimulus_ids=False
    )


# Loads (or creates) the V1-match physical electrode indices from TARGET_RRR/<ANALYSIS>/...
def get_match_subset_indices(
    source_region: int,
    target_region: int,
    *,
    analysis_type: str
) -> np.ndarray:
    """
    Load (or compute and save) the matched V1 subset indices (phys_idx) for the requested target region
    and analysis type. The on-disk layout is:
        TARGET_RRR/<ANALYSIS>/V1_to_<TARGETNAME>_<analysis_type>.npz

    Returns
    -------
    np.ndarray
        Array of physical V1 electrode indices that define the "V1-match" subset.
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

    # ====================== Grouping (reps or random subsets) ======================
    groups, label_D, id_print, rep_arr = build_groups_by_rep_or_subsets(
        trials, k_subsets=k_subsets, random_state=random_state
    )

    # ====================== Compute global RRR results (Panels A + B) ======================
    perf_full  = perf_wrapper(
        source_region, target_region,
        analysis_type=analysis_type, match_to_target=False,
        d_max=d_max, alpha=alpha,
        outer_splits=outer_splits, inner_splits=inner_splits,
        random_state=random_state
    )
    perf_match = perf_wrapper(
        source_region, target_region,
        analysis_type=analysis_type, match_to_target=True,
        d_max=d_max, alpha=alpha,
        outer_splits=outer_splits, inner_splits=inner_splits,
        random_state=random_state
    )

    d95_full_g  = d95_from_curves(perf_full ["rrr_R2_mean"],  perf_full ["ridge_R2_mean"],  d_max)
    d95_match_g = d95_from_curves(perf_match["rrr_R2_mean"], perf_match["ridge_R2_mean"], d_max)

    # ====================== Compute per-group dimensionalities (Panel D) ======================
    d95_full_rep, d95_match_rep = [], []

    for g_idx, grp_idx in enumerate(groups):
        # ---- FULL model: V1 → Target ----
        res_f = perf_wrapper(
            source_region, target_region,
            analysis_type=analysis_type,
            match_to_target=False,
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state,
            trial_subset=grp_idx,         # <── subset mode
        )
        d95_full_rep.append(d95_from_curves(res_f["rrr_R2_mean"], res_f["ridge_R2_mean"], d_max))

        # ---- MATCH model: (V1−Match) → Match ----
        res_m = perf_wrapper(
            source_region, target_region,
            analysis_type=analysis_type,
            match_to_target=True,
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state,
            trial_subset=grp_idx,         # <── subset mode
        )
        d95_match_rep.append(d95_from_curves(res_m["rrr_R2_mean"], res_m["ridge_R2_mean"], d_max))

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
        labeled_dot(axA, int(d95_full_g), float(r2d), int(d95_full_g),
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
        labeled_dot(axB, int(d95_match_g), float(r2d), int(d95_match_g),
                    face=colB, edge="k", size=240, text_size=12, text_color="white")

    axB.set_title(f"Predicting V1-match {tgt}", color=colB, pad=10, fontsize=15)
    axB.grid(alpha=.25)
    axB.text(-0.07, 1.05, "B", transform=axB.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")
    axB.set_box_aspect(1)

    # =============================== Panel C (Global comparison: A vs B) ===============================
    if np.isfinite(d95_match_g) and np.isfinite(d95_full_g):
        xg, yg = int(d95_match_g), int(d95_full_g)
        xmin, xmax = square_limits([xg], [yg], base_min=1, scale=1.5)
    else:
        xmin, xmax = square_limits([1], [1], base_min=1, scale=1.5)

    rng_jitter = np.random.default_rng(random_state)

    axC.plot([xmin, xmax], [xmin, xmax], ls="--", lw=0.9, color="k")  # y=x
    if np.isfinite(d95_match_g) and np.isfinite(d95_full_g):
        jitter_x = jitter([xg], rng_jitter)[0]
        jitter_y = jitter([yg], rng_jitter)[0]
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
        xmin_d, xmax_d = square_limits(d95_match_rep, d95_full_rep, base_min=1, scale=1.5)
    else:
        xmin_d, xmax_d = (1, max(2, int(np.ceil(1.5 * d_max))))

    jitter_match = jitter(d95_match_rep, rng_jitter)
    jitter_full  = jitter(d95_full_rep,  rng_jitter)
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
    left_block_xcent  = (boxB.x0 + boxB.x1) / 2
    right_block_xcent = (boxD.x0 + boxD.x1) / 2 

    fig.text(left_ylabel_x, left_col_ycenter,
             rf"Mean $R^2$  (outer {outer_splits}, inner {inner_splits})",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+7, color="black")

    fig.text(right_ylabel_x, right_block_ycent,
             f"{runtime.get_consts().REGION_ID_TO_NAME[target_region]} Predictive dimensions",
             va="center", ha="right", rotation="vertical", fontsize=label_fs+7, color="#9C1C1C")

    fig.text(left_block_xcent, left_xlabel_y,
             "Predictive dimensions (d)",
             va="top", ha="center", fontsize=label_fs+3, color="black")

    fig.text(right_block_xcent, right_xlabel_y,
             "Target V1 Predictive dimensions",
             va="top", ha="center", fontsize=label_fs+3, color="#1565C0")
   
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


# ============================ Helper: per-trials matching on electrode subsets ============================
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
    Select a V1-MATCH subset *within a given V1 electrode subset* to match the
    mean-firing distribution of a *given Target electrode subset*, computed on
    the specified trial subset.

    This replicates the logic of MATCHINGSUBSET.match_and_save (binning + neighbor
    deficit fill), but works in-memory and respects both trial and electrode subsets.

    Returns
    -------
    match_src_phys : np.ndarray
        Physical indices of the chosen V1-MATCH electrodes (len ≈ len(tgt_subset_phys)).
    src_remain_phys : np.ndarray
        Physical indices of the remaining V1 electrodes in the provided src_subset_phys.
    """
    rng = np.random.default_rng(seed)

    # --- Build data matrices restricted by trials & the provided electrode subsets ---
    Xs = build_trial_matrix(source_region, analysis_type, trial_idx=trial_idx, electrode_indices=src_subset_phys)  # (n_trials, n_src_sub)
    Xt = build_trial_matrix(target_region, analysis_type, trial_idx=trial_idx, electrode_indices=tgt_subset_phys)  # (n_trials, n_tgt_sub)

    # --- Mean firing per electrode over the selected trials ---
    ms = Xs.mean(axis=0)  # aligned with src_subset_phys
    mt = Xt.mean(axis=0)  # aligned with tgt_subset_phys
    n_tgt = Xt.shape[1]

    # --- Define common bin edges over the union range of source & target means ---
    vmin = float(min(ms.min(), mt.min())) if (ms.size and mt.size) else 0.0
    vmax = float(max(ms.max(), mt.max())) if (ms.size and mt.size) else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # Robustify degenerate ranges (all-equal or NaNs): make a tiny span
        vmin, vmax = float(vmin), float(vmin + 1e-6)

    edges = np.linspace(vmin, vmax, int(n_bins) + 1)

    # src bins
    src_bin_of = np.searchsorted(edges, ms, side="right") - 1
    src_bin_of = np.clip(src_bin_of, 0, n_bins - 1)

    # target bin counts (how many we want in each bin)
    tgt_counts = [int(((mt >= edges[b]) & (mt < edges[b + 1])).sum()) for b in range(n_bins)]

    picked_local_idx: list[int] = []
    surplus_bins: list[list[int]] = [[] for _ in range(n_bins)]
    deficits = np.zeros(n_bins, dtype=int)

    # --- First pass: satisfy each bin locally if possible; record deficits/surpluses ---
    for b in range(n_bins):
        s_idxs = np.where(src_bin_of == b)[0]  # local indices (0..len(src_subset_phys)-1)
        need   = tgt_counts[b]
        if need == 0:
            surplus_bins[b] = list(s_idxs)
            continue
        if s_idxs.size <= need:
            picked_local_idx.extend(s_idxs.tolist())
            deficits[b] = need - s_idxs.size
        else:
            chosen = rng.choice(s_idxs, size=need, replace=False)
            picked_local_idx.extend(chosen.tolist())
            # the remainder are surplus
            surplus_bins[b] = [i for i in s_idxs if i not in set(chosen.tolist())]

    # --- Second pass: fill deficits from neighboring bins outward ---
    for b in np.where(deficits > 0)[0]:
        deficit = int(deficits[b])
        dist = 1
        while deficit > 0 and dist < n_bins:
            for nb in (b - dist, b + dist):
                if 0 <= nb < n_bins and surplus_bins[nb]:
                    take = min(len(surplus_bins[nb]), deficit)
                    chosen = rng.choice(surplus_bins[nb], size=take, replace=False)
                    picked_local_idx.extend(chosen.tolist())
                    # remove chosen from surplus
                    chosen_set = set(chosen.tolist())
                    surplus_bins[nb] = [ix for ix in surplus_bins[nb] if ix not in chosen_set]
                    deficit -= take
                    if deficit == 0:
                        break
            dist += 1

        # fallback: if still deficit, grab from any remaining surplus globally
        if deficit > 0:
            remaining = [ix for pool in surplus_bins for ix in pool]
            if remaining:
                take = min(deficit, len(remaining))
                chosen = rng.choice(remaining, size=take, replace=False)
                picked_local_idx.extend(chosen.tolist())
                chosen_set = set(chosen.tolist())
                # remove globally
                for nb in range(n_bins):
                    if surplus_bins[nb]:
                        surplus_bins[nb] = [ix for ix in surplus_bins[nb] if ix not in chosen_set]
            # if *still* deficit, we just proceed with fewer than n_tgt (rare/degenerate)

    # --- Enforce unique, cap to n_tgt (target subset size) ---
    picked_local_idx = sorted(set(picked_local_idx))
    if len(picked_local_idx) > n_tgt:
        picked_local_idx = picked_local_idx[:n_tgt]
    elif len(picked_local_idx) < n_tgt:
        # fill randomly from remaining local indices
        remaining_all = [i for i in range(len(src_subset_phys)) if i not in set(picked_local_idx)]
        need = n_tgt - len(picked_local_idx)
        if need > 0 and remaining_all:
            extra = rng.choice(remaining_all, size=min(need, len(remaining_all)), replace=False)
            picked_local_idx.extend(extra.tolist())

    picked_local_idx = np.array(sorted(set(picked_local_idx)), dtype=int)
    match_src_phys = np.asarray(src_subset_phys, dtype=int)[picked_local_idx]
    src_remain_phys = np.asarray(src_subset_phys, dtype=int)[~np.isin(src_subset_phys, match_src_phys)]

    if verbose:
        print(f"[match] trials={('all' if trial_idx is None else len(np.atleast_1d(trial_idx)))} | "
              f"V1-subset={len(src_subset_phys)} → match={len(match_src_phys)} | "
              f"target-subset={len(tgt_subset_phys)}")

    if src_remain_phys.size == 0:
        raise ValueError("All V1 electrodes were matched; nothing left in X for MATCH model.")

    return match_src_phys, src_remain_phys

# ============================ FAST matching on prebuilt matrices ============================
def match_subset_from_prebuilt(
    X_src_all: np.ndarray,          # (n_trials, n_src_sub)  — V1 subset matrix for ALL trials
    Y_tgt_all: np.ndarray,          # (n_trials, n_tgt_sub)  — Target subset matrix for ALL trials
    src_subset_phys: np.ndarray,    # physical V1 channel ids aligned to X_src_all columns
    *,
    trial_idx: np.ndarray | None,   # subset of rows (trials) or None for ALL
    n_bins: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute V1-MATCH *within* a given V1 electrode subset using the same binning logic,
    but operating directly on *prebuilt* matrices. Returns both local-column indices
    (for slicing X_src_all) and physical indices (for bookkeeping).

    Returns
    -------
    match_loc : np.ndarray[int]   — local column indices in X_src_all for "V1-MATCH"
    remain_loc: np.ndarray[int]   — local column indices in X_src_all for "V1-REM"
    match_phys: np.ndarray[int]   — physical ids of match electrodes
    remain_phys: np.ndarray[int]  — physical ids of remain electrodes
    """
    rng = np.random.default_rng(seed)

    # Restrict to the requested trials (rows)
    if trial_idx is None:
        Xs = X_src_all
        Xt = Y_tgt_all
    else:
        Xs = X_src_all[trial_idx, :]
        Xt = Y_tgt_all[trial_idx, :]

    # Mean firing per electrode over the selected trials
    ms = Xs.mean(axis=0)  # shape (n_src_sub)
    mt = Xt.mean(axis=0)  # shape (n_tgt_sub)
    n_src_sub = Xs.shape[1]
    n_tgt_sub = Xt.shape[1]

    # Build common bins over joint range
    vmin = float(min(ms.min(), mt.min())) if (ms.size and mt.size) else 0.0
    vmax = float(max(ms.max(), mt.max())) if (ms.size and mt.size) else 1.0
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
        vmin, vmax = float(vmin), float(vmin + 1e-6)

    edges = np.linspace(vmin, vmax, int(n_bins) + 1)

    src_bin_of = np.searchsorted(edges, ms, side="right") - 1
    src_bin_of = np.clip(src_bin_of, 0, n_bins - 1)

    tgt_counts = [int(((mt >= edges[b]) & (mt < edges[b + 1])).sum()) for b in range(n_bins)]

    picked_local: list[int] = []
    surplus_bins: list[list[int]] = [[] for _ in range(n_bins)]
    deficits = np.zeros(n_bins, dtype=int)

    # First pass: local satisfaction
    for b in range(n_bins):
        s_idxs = np.where(src_bin_of == b)[0]
        need   = tgt_counts[b]
        if need == 0:
            surplus_bins[b] = list(s_idxs)
            continue
        if s_idxs.size <= need:
            picked_local.extend(s_idxs.tolist())
            deficits[b] = need - s_idxs.size
        else:
            chosen = rng.choice(s_idxs, size=need, replace=False)
            picked_local.extend(chosen.tolist())
            chosen_set = set(chosen.tolist())
            surplus_bins[b] = [ix for ix in s_idxs if ix not in chosen_set]

    # Second pass: fill deficits from neighboring bins
    for b in np.where(deficits > 0)[0]:
        deficit = int(deficits[b])
        dist = 1
        while deficit > 0 and dist < n_bins:
            for nb in (b - dist, b + dist):
                if 0 <= nb < n_bins and surplus_bins[nb]:
                    take = min(len(surplus_bins[nb]), deficit)
                    chosen = rng.choice(surplus_bins[nb], size=take, replace=False)
                    picked_local.extend(chosen.tolist())
                    chosen_set = set(chosen.tolist())
                    surplus_bins[nb] = [ix for ix in surplus_bins[nb] if ix not in chosen_set]
                    deficit -= take
                    if deficit == 0:
                        break
            dist += 1
        if deficit > 0:
            remaining = [ix for pool in surplus_bins for ix in pool]
            if remaining:
                take = min(deficit, len(remaining))
                chosen = rng.choice(remaining, size=take, replace=False)
                picked_local.extend(chosen.tolist())
                chosen_set = set(chosen.tolist())
                # remove globally
                for nb in range(n_bins):
                    if surplus_bins[nb]:
                        surplus_bins[nb] = [ix for ix in surplus_bins[nb] if ix not in chosen_set]

    # Enforce unique and cap to n_tgt_sub
    picked_local = sorted(set(picked_local))[:n_tgt_sub]
    picked_local = np.asarray(picked_local, dtype=int)

    # Local -> physical
    match_loc  = picked_local
    remain_loc = np.asarray([i for i in range(n_src_sub) if i not in set(match_loc.tolist())], dtype=int)

    match_phys  = np.asarray(src_subset_phys, dtype=int)[match_loc]
    remain_phys = np.asarray(src_subset_phys, dtype=int)[remain_loc]

    if remain_loc.size == 0:
        raise ValueError("All V1 electrodes were matched; nothing left for X in MATCH model.")

    return match_loc, remain_loc, match_phys, remain_phys

def make_subset_semedo_figure(
    source_region : int = 1,      # V1
    target_region : int = 3,      # IT=3, V4=2
    *,
    analysis_type : str  = "residual",   # "window" | "baseline100" | "residual"
    d_max         : int  = 35,
    alpha         : float | None = None,
    outer_splits  : int  = 3,
    inner_splits  : int  = 3,
    random_state  : int  = 0,
    # ---- subset parameters ----
    n_runs     : int = 5,          # number of electrode-subset runs (color per run)
    n_src      : int = 113,        # number of V1 electrodes per run
    n_tgt      : int = 28,         # number of target (V4/IT) electrodes per run
    k_subsets  : int | None = 10   # None→ repetitions; int→ K random trial subsets (per run)
):
    """
    Subset-Semedo figure (A–D), compute-first then layout:

    Per run:
      • Randomly choose electrode subsets for V1 (src) and Target (tgt).
      • Panel A: FULL model on ALL trials, restricted to the chosen electrodes (Y=Target, X=V1).
      • Panel B: MATCH model on ALL trials, restricted to the chosen V1 subset:
                 split V1-subset into (V1-MATCH) and (V1-REM) using global phys_idx order,
                 falling back to activity-based fill if needed.
      • Panel C: One dot per run → (d95_MATCH, d95_FULL).
      • Panel D: Within each run, split trials (repetitions or K random subsets) and compute d95 per subset
                 for BOTH FULL and MATCH using the SAME electrode subsets of that run.

    Notes:
      - Matching here is decided per run using the global MATCH phys_idx ranking, restricted to
        the current V1 subset; if not enough overlap, we fill by lowest-mean-activity electrodes.
      - All trial/electrode slicing is done by `build_trial_matrix`, then evaluated with
        `RRR_Centered_matching._performance_from_mats`.
    """

    # ====================== PHASE 1 — DATA & METRICS COMPUTATION ======================
    def _log(msg: str) -> None:
        print(f"[SubsetSemedo][{datetime.now().strftime('%H:%M:%S')}] {msg}")

    t_phase = time.perf_counter()
    _log("Phase 1 started: loading trials and preparing pools…")

    cfg    = runtime.get_cfg()
    consts = runtime.get_consts()
    tgt_nm = consts.REGION_ID_TO_NAME[target_region]

    # 1) Load trials once (we only need indices/length; matrices are built via helpers)
    trials    = cfg._load_trials()
    n_trials  = len(trials)
    all_idxs  = np.arange(n_trials)
    _log(f"Trials loaded: n_trials={n_trials}")

    # 2) ROI-wise electrode pools and effective sizes
    rois        = cfg.get_rois()
    all_src_idx = np.where(rois == source_region)[0]
    all_tgt_idx = np.where(rois == target_region)[0]
    n_src_eff   = min(n_src, all_src_idx.size)
    n_tgt_eff   = min(n_tgt, all_tgt_idx.size)
    _log(f"Electrode pools → V1:{all_src_idx.size} | {tgt_nm}:{all_tgt_idx.size} "
        f"| requested per run → src:{n_src}→{n_src_eff}, tgt:{n_tgt}→{n_tgt_eff}")
    if n_src_eff < 2 or n_tgt_eff < 1:
        raise ValueError("Not enough electrodes for the requested subsets (n_src or n_tgt too small).")

    # 3) Containers to accumulate results for later plotting
    runs_full   = []   # list of dicts: {'rrr','sem','ridge','d95','color'}
    runs_match  = []   # list of dicts: {'rrr','sem','ridge','d95','color'}
    d95_full_runs, d95_match_runs = [], []
    d95_full_sub_all, d95_match_sub_all = [], []  # list of (run_idx, d95_value)

    # 4) Visual palette (one color per run)
    base_colors = ["#9C1C1C", "#1565C0", "#2E7D32", "#7B1FA2", "#F57C00", "#00796B", "#5D4037"]
    colors      = [base_colors[i % len(base_colors)] for i in range(n_runs)]
    _log(f"Color palette prepared for {n_runs} runs.")

    # 5) To minimize overlap between runs, track used electrodes
    used_src: set[int] = set()
    used_tgt: set[int] = set()

    # 6) Loop over runs
    dims = np.arange(1, d_max + 1)

    for run_idx in range(n_runs):
        t_run = time.perf_counter()
        run_seed = int(random_state + run_idx)
        rng      = np.random.default_rng(run_seed)
        col      = colors[run_idx]
        _log(f"[Run {run_idx+1}/{n_runs}] Selecting electrode subsets with minimal overlap…")

        # 6.a) Electrode subsets with minimal overlap across runs
        #     Prefer sampling only from electrodes not used yet; top-up if needed.
        rem_src = np.array([e for e in all_src_idx if e not in used_src])
        if rem_src.size >= n_src_eff:
            src_subset = rng.choice(rem_src, n_src_eff, replace=False)
        else:
            need = n_src_eff - rem_src.size
            fill = rng.choice(all_src_idx, need, replace=False)
            fill = np.array([e for e in fill if e not in rem_src])
            src_subset = np.concatenate([rem_src, fill])[:n_src_eff]
        used_src.update(src_subset.tolist())

        rem_tgt = np.array([e for e in all_tgt_idx if e not in used_tgt])
        if rem_tgt.size >= n_tgt_eff:
            tgt_subset = rng.choice(rem_tgt, n_tgt_eff, replace=False)
        else:
            need = n_tgt_eff - rem_tgt.size
            fill = rng.choice(all_tgt_idx, need, replace=False)
            fill = np.array([e for e in fill if e not in rem_tgt])
            tgt_subset = np.concatenate([rem_tgt, fill])[:n_tgt_eff]
        used_tgt.update(tgt_subset.tolist())
        _log(f"[Run {run_idx+1}] V1 subset={src_subset.size}, {tgt_nm} subset={tgt_subset.size}")

        # 6.b) Build matrices ONCE per run (ALL trials)
        #     Later for subsets we only slice rows, which is cheap.
        _log(f"[Run {run_idx+1}] Building trial matrices (ALL trials)…")
        X_src_all = build_trial_matrix(
            source_region, analysis_type, trial_idx=None, electrode_indices=src_subset
        )  # shape (n_trials, n_src_eff)
        Y_tgt_all = build_trial_matrix(
            target_region, analysis_type, trial_idx=None, electrode_indices=tgt_subset
        )  # shape (n_trials, n_tgt_eff)
        _log(f"[Run {run_idx+1}] Built matrices shapes → X_src_all={X_src_all.shape}, Y_tgt_all={Y_tgt_all.shape}")

        # 6.c) MATCH split on ALL trials using the fast helper (local indices returned)
        _log(f"[Run {run_idx+1}] Computing MATCH split on ALL trials…")
        match_loc_all, remain_loc_all, match_src_all, src_remain_all = match_subset_from_prebuilt(
            X_src_all, Y_tgt_all, src_subset_phys=src_subset,
            trial_idx=None, n_bins=20, seed=run_seed
        )
        _log(f"[Run {run_idx+1}] MATCH sizes → match={match_loc_all.size}, remain={remain_loc_all.size}")

        # 6.d) Build FULL/MATCH matrices for ALL trials from the prebuilt arrays
        #     FULL: Y = Target, X = V1
        Y_full_all = Y_tgt_all
        X_full_all = X_src_all
        #     MATCH: Y = V1-match, X = V1-remain (both from X_src_all columns)
        Y_match_all = X_src_all[:, match_loc_all]
        X_match_all = X_src_all[:, remain_loc_all]

        # 6.e) Evaluate FULL & MATCH (ALL trials)
        _log(f"[Run {run_idx+1}] Evaluating FULL & MATCH (ALL trials)…")
        res_full = RRR_Centered_matching._performance_from_mats(
            Y_full_all, X_full_all,
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=run_seed
        )
        res_match = RRR_Centered_matching._performance_from_mats(
            Y_match_all, X_match_all,
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=run_seed
        )

        d95_full_g  = int(d95_from_curves(res_full ["rrr_R2_mean"],  res_full ["ridge_R2_mean"],  d_max))
        d95_match_g = int(d95_from_curves(res_match["rrr_R2_mean"], res_match["ridge_R2_mean"], d_max))
        _log(f"[Run {run_idx+1}] FULL ridge={res_full['ridge_R2_mean']:.3f}, d95={d95_full_g} | "
            f"MATCH ridge={res_match['ridge_R2_mean']:.3f}, d95={d95_match_g}")

        d95_full_runs.append(d95_full_g)
        d95_match_runs.append(d95_match_g)

        runs_full.append({
            "rrr":   res_full["rrr_R2_mean"],
            "sem":   res_full["rrr_R2_sem"],
            "ridge": res_full["ridge_R2_mean"],
            "d95":   d95_full_g,
            "color": col
        })
        runs_match.append({
            "rrr":   res_match["rrr_R2_mean"],
            "sem":   res_match["rrr_R2_sem"],
            "ridge": res_match["ridge_R2_mean"],
            "d95":   d95_match_g,
            "color": col
        })

        # 6.f) Build trial groups for Panel D (per run)
        if k_subsets is None:
            # Repetitions based on ALLMAT repetition index (0..29)
            rep_arr = np.array([int(tr["allmat_row"][3]) - 1 for tr in trials], dtype=int)
            rep_ids = np.unique(rep_arr)
            groups  = [np.flatnonzero(rep_arr == g) for g in rep_ids]
            label_D = "Repetitions"
        else:
            # K random trial subsets for this run (different per run)
            idxs   = all_idxs.copy()
            rng.shuffle(idxs)
            splits = np.array_split(idxs, k_subsets)
            groups = [np.asarray(part, dtype=int) for part in splits]
            label_D = f"{k_subsets} random subsets × {n_runs} runs"
        _log(f"[Run {run_idx+1}] Trial grouping for Panel D → mode: {label_D}, groups={len(groups)}")

        # 6.g) For each trial subset: slice rows + recompute matching FAST on the slice
        for gi, grp_idx in enumerate(groups, start=1):
            _log(f"[Run {run_idx+1}] Subset {gi}/{len(groups)}: n_trials={len(grp_idx)} → FULL eval…")
            # Slice rows (trials) only — no rebuilds
            Y_full_sub = Y_tgt_all[grp_idx, :]
            X_full_sub = X_src_all[grp_idx, :]

            # FULL on subset
            res_f_sub  = RRR_Centered_matching._performance_from_mats(
                Y_full_sub, X_full_sub,
                d_max=d_max, alpha=alpha,
                outer_splits=max(2, outer_splits),
                inner_splits=max(2, inner_splits),
                random_state=run_seed
            )
            d95_full_sub = int(d95_from_curves(res_f_sub["rrr_R2_mean"], res_f_sub["ridge_R2_mean"], d_max))
            d95_full_sub_all.append((run_idx, d95_full_sub))

            # MATCH on subset: recompute matching using only these rows, still within SAME electrodes of the run
            _log(f"[Run {run_idx+1}] Subset {gi}/{len(groups)}: matching + MATCH eval…")
            match_loc_sub, remain_loc_sub, match_src_sub, src_remain_sub = match_subset_from_prebuilt(
                X_src_all, Y_tgt_all, src_subset_phys=src_subset,
                trial_idx=grp_idx, n_bins=20, seed=run_seed
            )

            Y_match_sub = X_src_all[np.ix_(grp_idx, match_loc_sub)]
            X_match_sub = X_src_all[np.ix_(grp_idx, remain_loc_sub)]
            res_m_sub   = RRR_Centered_matching._performance_from_mats(
                Y_match_sub, X_match_sub,
                d_max=d_max, alpha=alpha,
                outer_splits=max(2, outer_splits),
                inner_splits=max(2, inner_splits),
                random_state=run_seed
            )
            d95_match_sub = int(d95_from_curves(res_m_sub["rrr_R2_mean"], res_m_sub["ridge_R2_mean"], d_max))
            d95_match_sub_all.append((run_idx, d95_match_sub))
            _log(f"[Run {run_idx+1}] Subset {gi}/{len(groups)} → d95 FULL={d95_full_sub}, MATCH={d95_match_sub}")

        _log(f"[Run {run_idx+1}] Completed in {time.perf_counter() - t_run:.2f}s.")

    _log(f"Phase 1 finished in {time.perf_counter() - t_phase:.2f}s. Proceeding to Phase 2 (figure layout)…")

    # ====================== PHASE 2 — FIGURE LAYOUT & DRAWING ======================

    # 7) Prepare figure/axes
    fig = plt.figure(figsize=(14, 13), dpi=400)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.15, hspace=0.30)
    gs  = gridspec.GridSpec(2, 2)
    axA = fig.add_subplot(gs[0, 0])  # FULL curves per run
    axB = fig.add_subplot(gs[1, 0])  # MATCH curves per run
    axC = fig.add_subplot(gs[0, 1])  # one dot per run (d95_MATCH, d95_FULL)
    axD = fig.add_subplot(gs[1, 1])  # many dots across subsets (d95_MATCH, d95_FULL)

    label_fs = 17
    colA, colB = "#9C1C1C", "#1565C0"
    rng_jitter = np.random.default_rng(random_state)

    # 8) Panel A — FULL curves (one color per run)
    for i, cur in enumerate(runs_full):
        axA.errorbar(dims, cur["rrr"], yerr=cur["sem"],
                     fmt="-o", ms=3.2, lw=1.1, capsize=3,
                     color=cur["color"], zorder=2, alpha=0.95)
        axA.scatter([1], [cur["ridge"]], marker="^", s=70,
                    color=cur["color"], edgecolors="k", zorder=3)
        # Label the d95 point with the run index for clarity
        if np.isfinite(cur["d95"]):
            d = int(cur["d95"])
            y = float(cur["rrr"][d - 1])   # y-value at d95
            labeled_dot(axA, d, y, d,      # <<< label with d95 (not run index)
                        face=cur["color"], edge="k", size=200, text_size=10, text_color="white")
    axA.set_title(f"Predicting {tgt_nm}", color=colA, pad=10, fontsize=15)
    axA.grid(alpha=.25)
    axA.text(-0.07, 1.05, "A", transform=axA.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")
    axA.set_box_aspect(1)

    # 9) Panel B — MATCH curves (one color per run)
    for i, cur in enumerate(runs_match):
        axB.errorbar(dims, cur["rrr"], yerr=cur["sem"],
                     fmt="-o", ms=3.2, lw=1.1, capsize=3,
                     color=cur["color"], zorder=2, alpha=0.95)
        axB.scatter([1], [cur["ridge"]], marker="^", s=70,
                    color=cur["color"], edgecolors="k", zorder=3)
        if np.isfinite(cur["d95"]):
            d = int(cur["d95"])
            y = float(cur["rrr"][d - 1])
            labeled_dot(axB, d, y, d,      # <<< label with d95
                        face=cur["color"], edge="k", size=200, text_size=10, text_color="white")
    axB.set_title(f"Predicting V1-match {tgt_nm}", color=colB, pad=10, fontsize=15)
    axB.grid(alpha=.25)
    axB.text(-0.07, 1.05, "B", transform=axB.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")
    axB.set_box_aspect(1)

    # 10) Panel C — one dot per run: (d95_MATCH, d95_FULL)
    if len(d95_full_runs) and len(d95_match_runs):
        xmin_c, xmax_c = square_limits(d95_match_runs, d95_full_runs, base_min=1, scale=1.5)
    else:
        xmin_c, xmax_c = (1, max(2, int(np.ceil(1.5 * d_max))))
    axC.plot([xmin_c, xmax_c], [xmin_c, xmax_c], ls="--", lw=0.9, color="k")

    for i, (xm, ym) in enumerate(zip(d95_match_runs, d95_full_runs, strict=False)):
        # Light jitter for visibility + number label = run index
        jx = xm + rng_jitter.uniform(-0.15, 0.15)
        jy = ym + rng_jitter.uniform(-0.15, 0.15)
        axC.scatter([jx], [jy], s=155, facecolors="white", edgecolors=colors[i], linewidths=1.5, zorder=4)
        axC.text(jx, jy, str(i+1), ha="center", va="center", fontsize=10, weight="bold",
                 color="black", zorder=5, path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    axC.set_xlim(xmin_c, xmax_c)
    axC.set_ylim(xmin_c, xmax_c)
    axC.set_aspect('equal', adjustable='box')
    ticks_c = np.arange(xmin_c, xmax_c + 1, max(1, int(np.ceil((xmax_c - xmin_c) / 6))))
    axC.set_xticks(ticks_c)
    axC.set_yticks(ticks_c)
    axC.grid(False)
    axC.text(-0.07, 1.05, "C", transform=axC.transAxes,
             ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")

    # 11) Panel D — all subset dots across runs (drop-in replacement)
    # Compute square limits from all subset d95 values (MATCH on x, FULL on y)
    d95_match_vals = [v for (_, v) in d95_match_sub_all]
    d95_full_vals  = [v for (_, v) in d95_full_sub_all]
    if len(d95_full_vals) and len(d95_match_vals):
        xmin_d, xmax_d = square_limits(d95_match_vals, d95_full_vals, base_min=1, scale=1.5)
    else:
        xmin_d, xmax_d = (1, max(2, int(np.ceil(1.5 * d_max))))

    # Reference line y = x
    axD.plot([xmin_d, xmax_d], [xmin_d, xmax_d], ls="--", lw=0.9, color="k")

    # Scatter all subset points, color encodes run index; light jitter for visibility
    for (run_idx, yf), (_, xf) in zip(d95_full_sub_all, d95_match_sub_all, strict=False):
        jx = xf + rng_jitter.uniform(-0.15, 0.15)
        jy = yf + rng_jitter.uniform(-0.15, 0.15)
        axD.scatter([jx], [jy], s=46, facecolors="white",
                    edgecolors=colors[run_idx], linewidths=0.9, alpha=0.95)

    # Axes cosmetics
    axD.set_xlim(xmin_d, xmax_d)
    axD.set_ylim(xmin_d, xmax_d)
    axD.set_aspect('equal', adjustable='box')
    ticks_d = np.arange(xmin_d, xmax_d + 1, max(1, int(np.ceil((xmax_d - xmin_d) / 6))))
    axD.set_xticks(ticks_d)
    axD.set_yticks(ticks_d)
    axD.grid(False)

    # Move the subtitle up a bit and add a subtle white background for readability
    subtitle = ("Repetitions" if k_subsets is None else f"{k_subsets} random subsets × {n_runs} runs")
    axD.text(
        0.98, 0.98,                      # was 0.02 → lifted to 0.12 to avoid legend overlap
        subtitle,
        transform=axD.transAxes,
        ha="right", va="top",
        fontsize=11,
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    )

    # Panel label
    axD.text(-0.07, 1.05, "D", transform=axD.transAxes,
            ha="left", va="bottom", fontsize=20, fontweight="bold", color="black")

    # 12) Shared labels and overall title
    for ax in (axA, axB):
        ax.set_xlabel("Predictive dimensions (d)", fontsize=label_fs)
        ax.set_ylabel(rf"Mean $R^2$  (outer {outer_splits}, inner {inner_splits})", fontsize=label_fs)
    for ax in (axC, axD):
        ax.set_xlabel("Target V1 Predictive dimensions", fontsize=label_fs, color=colB)
        ax.set_ylabel(f"{tgt_nm} Predictive dimensions",  fontsize=label_fs, color=colA)

    boxA, boxC = axA.get_position(), axC.get_position()
    top_row_ymax = max(boxA.y1, boxC.y1)
    fig.suptitle(
        f"{cfg.get_monkey_name()}  |  {cfg.get_zscore_title()}  |  "
        f"{analysis_type.upper()}  (n_src={n_src_eff}, n_tgt={n_tgt_eff}, runs={n_runs})",
        fontsize=18, y=min(0.998, top_row_ymax + 0.080), fontweight="bold"
    )

    # 13) Legend (Panel D) — color ↔ run index
    handles = [plt.Line2D([0], [0], marker='o', color='none',
                          markerfacecolor='white',
                          markeredgecolor=colors[i],
                          markersize=7, lw=0)
               for i in range(n_runs)]
    labels  = [f"Run {i+1}" for i in range(n_runs)]
    axD.legend(handles, labels, loc="lower right", fontsize=9, frameon=False, title="Runs", title_fontsize=10)

    # 14) Save
    out_dir = cfg.get_data_path() / "Semedo_plots" / f"{n_runs}_multisubset" / analysis_type
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_tag  = consts.REGION_ID_TO_NAME[target_region]
    file_name = (
        f"{cfg.get_monkey_name()} "
        f"({n_src_eff},{n_tgt_eff}) "
        f"{analysis_type.upper()} "
        f"({tgt_tag}).png"
    ).replace(" ", "_")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname  = (
        f"{cfg.get_monkey_name().replace(' ', '')}_semendo_subset_"
        f"src{n_src_eff}_tgt{n_tgt_eff}_runs{n_runs}_"
        f"{analysis_type}_V1_to_{tgt_nm}.png"
    )
    fig.savefig(out_dir / fname, dpi=400, facecolor="white")
    plt.close(fig)
    print(f"[✓] Subset-Semedo figure saved → {out_dir/fname}")



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
                        make_semedo_figure(
                            1,
                            tgt,
                            analysis_type=method,
                            k_subsets=10,
                        )
"""