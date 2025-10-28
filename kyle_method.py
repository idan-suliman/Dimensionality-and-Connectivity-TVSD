# kyle_method.py
from __future__ import annotations
from typing import List, Tuple, Dict
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from runtime import runtime

"""
Fast and clean utilities to quantify and visualize the stability of
*variational subspaces* across stimulus repetitions.

Design goals
------------
• **One aggregation per (monkey, zscore, region, method)**:
  We call `cfg.build_trial_matrix(...)` **once** and then slice rows per repetition.
  This avoids the huge overhead of re-aggregating over *all* trials 30×.

• Participation-Ratio dimensionality:
  PCA via SVD on the centered (100 × n_electrodes) repetition matrix,
  with effective dimensionality D = ceil(PR).

• Subspace overlap:
  For two repetitions, compute the cross-projection matrix between their
  D-dimensional PCA bases, take SVD, and average squared singular values (∈[0,1]).

Public API
----------
load_repetition_matrices(method, region)
compute_variational_space(X)
compute_repetition_overlap(method, region)
plot_overlap_matrix(overlap_matrix, D_common, method)
plot_all_overlaps_grid(region=1, monkeys=(...), zscore_codes=(...), methods=(...))
"""

# Internal cache: avoid recomputing the (N_trials × n_electrodes) aggregated matrix
# Keyed by (monkey_name, zscore_title, region_id, method)
_FULL_AGG_CACHE: Dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
# value = (full_mat, rep_idx_array, stim_id_array)


def _get_full_region_aggregation(method: str, region: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the full aggregated matrix once, plus repetition and stimulus indices.

    Returns
    -------
    full_mat : (N_trials, n_electrodes)
        Time-averaged MUA per trial×electrode (float32) for the given region.
    rep_idx : (N_trials,)
        Repetition index per trial (0..29).
    stim_id : (N_trials,)
        Stimulus id per trial (0..99).

    Notes
    -----
    • `cfg.build_trial_matrix` aggregates over *all trials* by design (including
      residual computation if requested), so it’s optimal to call it once and
      slice rows afterwards.
    """
    cfg = runtime.get_cfg()
    key = (cfg.get_monkey_name(), cfg.get_zscore_title(), int(region), str(method).lower().strip())
    if key in _FULL_AGG_CACHE:
        return _FULL_AGG_CACHE[key]

    trials = cfg._load_trials()

    # Build the FULL matrix once (no trial subselect)
    full_mat = cfg.build_trial_matrix(
        region_id=region,
        analysis_type=method,
        trials=None,                 # <-- ALL trials (the heavy part)
        return_stimulus_ids=False
    )  # (N_trials, n_electrodes), float32

    # Meta arrays computed cheaply from the Python trial dicts
    rep_idx = np.array(
        [ (tr.get("rep_idx") if tr.get("rep_idx") is not None else int(tr["allmat_row"][3]) - 1)
          for tr in trials ],
        dtype=int
    )
    stim_id = np.array([tr["stimulus_id"] for tr in trials], dtype=int)

    _FULL_AGG_CACHE[key] = (full_mat, rep_idx, stim_id)
    return full_mat, rep_idx, stim_id


# =============================================================================
# Data loading (FAST)
# =============================================================================
def load_repetition_matrices(method: str, region: int) -> List[np.ndarray]:
    """
    Build per-repetition matrices (100 × n_electrodes) **fast**.

    Parameters
    ----------
    method : str
        Analysis type for `build_trial_matrix` (e.g., "residual", "baseline100", "window").
    region : int
        Region ID (1=V1, 2=V4, 3=IT).

    Returns
    -------
    list of np.ndarray
        Length = consts.NUM_REPETITIONS (typically 30).
        Each item is a (100 × n_electrodes) float32 matrix for one repetition.

    Implementation details
    ----------------------
    • Calls the heavy aggregator ONCE to get `full_mat` for all trials.
    • Slices rows per repetition using the cached `rep_idx` array.
    • Rows within each repetition are sorted by `stimulus_id` (0..99) to enforce
      consistent image ordering across repetitions.
    """
    cfg = runtime.get_cfg()
    consts = runtime.get_consts()

    full_mat, rep_idx, stim_id = _get_full_region_aggregation(method, region)

    print(f"[INFO] {cfg.get_monkey_name()} | Z={cfg.get_zscore_title()} | Method={method}")
    print(f"[INFO] Full aggregated matrix: {full_mat.shape}  |  #reps={consts.NUM_REPETITIONS}")

    mats: List[np.ndarray] = []
    for r in range(consts.NUM_REPETITIONS):
        ridx = np.flatnonzero(rep_idx == r)
        if ridx.size == 0:
            raise ValueError(f"No trials found for repetition r={r}.")
        # enforce a stable 0..99 image ordering within the repetition:
        order = np.argsort(stim_id[ridx])
        mats.append(full_mat[ridx[order], :])

    print(f"[INFO] Built {len(mats)} repetition matrices (FAST path, 1× aggregation)")
    return mats


# =============================================================================
# Variational space (PCA + Participation Ratio)
# =============================================================================
def compute_variational_space(X: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute the dominant PCA subspace and its effective dimensionality via PR.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_features)
        Data matrix per repetition (100 × n_electrodes).

    Returns
    -------
    pcs_dominant : np.ndarray, shape (D, n_features)
        First D right-singular vectors (principal axes).
    D : int
        Effective dimensionality: D = ceil(PR) with PR = (∑v)^2 / ∑(v^2),
        where v are the per-component variances normalized to sum to 1.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    variances = (S ** 2) / max(1, (X.shape[0] - 1))
    if variances.sum() > 0:
        v = variances / variances.sum()
        denom = np.sum(v ** 2)
        pr = (np.sum(v) ** 2) / denom if denom > 0 else 0.0
    else:
        pr = 0.0

    D = int(np.ceil(pr)) if pr > 0 else 1
    pcs_dominant = Vt[:D, :]
    return pcs_dominant, D


# =============================================================================
# Overlap computation
# =============================================================================
def compute_repetition_overlap(method: str, region: int) -> Tuple[np.ndarray, int]:
    """
    Compute pairwise subspace overlaps across repetitions for (method, region).

    Parameters
    ----------
    method : str
        Analysis type passed to `build_trial_matrix`.
    region : int
        Region ID (1=V1, 2=V4, 3=IT).

    Returns
    -------
    overlap_matrix : np.ndarray, shape (R, R)
        Symmetric matrix; entry (i,j) = mean of squared singular values of
        the cross-projection between the two D-dimensional PCA bases.
    D_common : int
        Shared dimensionality = floor(mean(D_rep)) across repetitions.
    """
    reps_data = load_repetition_matrices(method, region)
    n_reps = len(reps_data)
    print(f"[INFO] Loaded {n_reps} repetition matrices")

    spaces: List[np.ndarray] = []
    Ds: List[int] = []

    for i, X in enumerate(reps_data):
        pcs, D = compute_variational_space(X)
        spaces.append(pcs)
        Ds.append(D)
        print(f"[INFO]  - Rep {i+1:02d}: PR≈{D:.2f}, D={D}")

    D_common = max(1, int(np.floor(np.mean(Ds)))) if Ds else 1
    print(f"[INFO] Common dominant dimensionality: D={D_common}")

    # Truncate all spaces to D_common
    spaces = [pcs[:D_common, :] for pcs in spaces]

    # Pairwise overlaps
    overlap_matrix = np.eye(n_reps, dtype=float)
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            C = spaces[i] @ spaces[j].T
            svals = np.linalg.svd(C, compute_uv=False)
            overlap = float(np.mean(svals ** 2))  # ∈ [0,1]
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap

    print(f"[INFO] Overlap matrix computed — shape: {overlap_matrix.shape}")
    return overlap_matrix, D_common


# =============================================================================
# Single overlap matrix plot  — with REP 1..30 tick labels on both axes
# =============================================================================
def plot_overlap_matrix(overlap_matrix: np.ndarray, D_common: int, method: str) -> None:
    """
    Plot a single overlap matrix with a clear title and a fixed [0,1] color scale.
    X/Y tick labels are 'REP 1' .. 'REP N' on both axes.
    """
    cfg = runtime.get_cfg()
    n_reps = overlap_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(overlap_matrix, cmap="viridis", vmin=0, vmax=1)

    ax.set_title(
        f"Overlap Between Repetition Subspaces\n"
        f"{cfg.get_monkey_name()}, Z={cfg.get_zscore_title()}, Method={method}, D={D_common}",
        fontsize=13, pad=12, weight="bold"
    )
    ax.set_xlabel("Repetition", fontsize=12)
    ax.set_ylabel("Repetition", fontsize=12)

    # --- REP 1..N on both axes ---
    ticks = np.arange(n_reps)
    labels = [f"REP {i+1}" for i in range(n_reps)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Subspace overlap", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.grid(False)
    ax.tick_params(axis="both", which="major", length=0)  # cleaner with text ticks
    ax.set_aspect("auto")

    plt.tight_layout()
    plt.show()


# =============================================================================
# Grid of overlap matrices — square heatmaps, tight vertical spacing,
# Z headers + method headers with **fixed, small gap** above the heatmaps.
#=============================================================================
def plot_all_overlaps_grid(
    region: int = 1,
    monkeys: tuple[str, ...] = ("Monkey N", "Monkey F"),
    zscore_codes: tuple[int, ...] = (1, 2, 3, 4),
    methods: tuple[str, ...] = ("residual", "baseline100"),
    figsize_scale: float = 1.0,
    cell_size: float = 2.7,        # width (inches) per heatmap cell
    gap_suptitle: float = 0.035,   # gap from Z-headers to the main title (figure coords)
    *,
    save: bool = False,
    save_dir: str | Path | None = None,
    save_dpi: int = 400,
    show: bool = True,
) -> str | None:
    """
    Compact grid with:
      • Square heatmaps.
      • Method headers very close to the heatmaps (uniform small gap).
      • Z headers just above methods (tight but clean).
      • Slightly larger gap to the main title (controlled by `gap_suptitle`).

    If `save=True`, the figure is written to:
        <TVSD>/PLOTS_HEAT_MAP/<Monkeys>__<Zs>__<Methods>__<Region>.png
    and the saved path is returned. If `show=False`, the figure window is not shown.
    """
    consts = runtime.get_consts()
    region_name = consts.REGION_ID_TO_NAME.get(region, f"Region{region}")

    # ---------- pre-compute overlaps ----------
    results: list[tuple[str, int, str, np.ndarray, int]] = []
    for m, z, met in product(monkeys, zscore_codes, methods):
        runtime.set_cfg(m, z)
        O, D = compute_repetition_overlap(met, region)
        results.append((m, z, met, O, D))

    R, Z, M = len(monkeys), len(zscore_codes), len(methods)
    C = Z * M

    # ---------- figure geometry ----------
    cs = float(cell_size) * float(figsize_scale)
    fig_w = cs * C
    head_h_z   = 0.20   # Z headers row (tight)
    head_h_met = 0.10   # method headers row (VERY tight → close to heatmaps)
    cbar_h     = 0.10
    fig_h = cs * (R + head_h_z + head_h_met + cbar_h)
    fig = plt.figure(figsize=(fig_w, fig_h))

    height_ratios = [head_h_z, head_h_met] + [1] * R + [cbar_h]
    gs = GridSpec(
        nrows=len(height_ratios), ncols=C, figure=fig,
        height_ratios=height_ratios,
        hspace=0.15, wspace=0.1
    )

    axes = np.empty((R, C), dtype=object)
    last_im = None

    # ---------- draw heatmaps (rows start at index 2) ----------
    for m, z, met, O, D in results:
        r = monkeys.index(m)
        c = zscore_codes.index(z) * M + methods.index(met)
        ax = fig.add_subplot(gs[2 + r, c])
        axes[r, c] = ax

        last_im = ax.imshow(O, vmin=0, vmax=1, cmap="viridis")
        ax.text(0.985, 0.05, f"D={D}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="w",
                bbox=dict(facecolor="k", alpha=0.25, pad=2, edgecolor="none"))
        
        ax.set_xticks([0, 29], [1, 30])          
        ax.set_yticks([0, 29], [1, 30])

        ax.tick_params(axis='both', length=0, width=0, labelsize=6)
        ax.tick_params(axis='x', pad=1, bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=6)
        ax.tick_params(axis='y', pad=1, labelsize=6)
        
        ax.set_box_aspect(1)        # square heatmap
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.canvas.draw()  # ensure positions

    # ---------- headers in dedicated rows (no overlap) ----------
    header_axes_z = []
    for zi, z in enumerate(zscore_codes):
        c0, c1 = zi * M, zi * M + (M - 1)

        # Z header
        ax_z = fig.add_subplot(gs[0, c0:c1+1])
        ax_z.axis("off")
        ax_z.text(0.5, 0.25, f"{consts.ZSCORE_INFO[z][0]}", transform=ax_z.transAxes,
                  ha="center", va="center", fontsize=11, fontweight="bold")
        header_axes_z.append(ax_z)

        # Method headers (tight to heatmaps): y≈0.15 keeps a **small, fixed** gap
        for mi, met in enumerate(methods):
            ax_m = fig.add_subplot(gs[1, zi * M + mi])
            ax_m.axis("off")
            ax_m.text(0.5, 0.15, met, transform=ax_m.transAxes,
                      ha="center", va="bottom", fontsize=11)

    # ---------- row labels very close to the first column ----------
    first_col_left = axes[0, 0].get_position().x0
    label_x = first_col_left - 0.010
    for r, m in enumerate(monkeys):
        y0, y1 = axes[r, 0].get_position().y0, axes[r, 0].get_position().y1
        cy = (y0 + y1) / 2.0
        fig.text(label_x, cy, m, ha="right", va="center",
                 rotation=90, fontsize=13, fontweight="bold")

    # ---------- main title with proportional gap above Z headers ----------
    top_of_z_headers = max(ax.get_position().y1 for ax in header_axes_z)
    suptitle_y = min(0.99, top_of_z_headers + gap_suptitle)
    fig.suptitle(
        f"Repetitions Subspace Overlap (D-mean) — {region_name}\n",
        fontsize=22, y=suptitle_y, fontweight="bold"
    )

    # ---------- colorbar ----------
    cax = fig.add_subplot(gs[-1, 2:6])
    cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cbar.set_label("Subspace Overlap", fontsize=12)

    # ---------- save/show ----------
    saved_path: str | None = None
    if save:
        base_dir = Path(save_dir) if save_dir is not None else (runtime.get_consts().BASE_DIR / "PLOTS_HEAT_MAP")
        base_dir.mkdir(parents=True, exist_ok=True)
        # Build an informative filename
        mtag = "M-" + "-".join(m.replace(" ", "") for m in monkeys)
        ztag = "Z-" + "-".join(str(z) for z in zscore_codes)
        atag = "A-" + "-".join(methods)
        fname = f"{mtag}__{ztag}__{atag}__{region_name}.png"
        out_path = base_dir / fname
        fig.savefig(out_path, dpi=save_dpi, facecolor="white", bbox_inches="tight")
        saved_path = str(out_path)
        print(f"[✓] Saved heatmap grid → {saved_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path


# -----------------------------------------------------------------------------
# Convenience: run for V1, V4, IT and save under TVSD/PLOTS_HEAT_MAP
# -----------------------------------------------------------------------------
def save_all_regions_heatmaps(
    regions: tuple[int, ...] = (1, 2, 3),                  # V1, V4, IT
    monkeys: tuple[str, ...] = ("Monkey N", "Monkey F"),
    zscore_codes: tuple[int, ...] = (1, 2, 3, 4),
    methods: tuple[str, ...] = ("residual", "baseline100"),
    figsize_scale: float = 1.0,
    cell_size: float = 2.7,
    save_dir: str | Path | None = None,                   # default = <TVSD>/PLOTS_HEAT_MAP
    save_dpi: int = 400,
) -> list[str]:
    """
    Loop over the requested regions (default: V1, V4, IT), render each grid,
    and save PNG files under TVSD/PLOTS_HEAT_MAP.
    Returns a list of saved file paths.
    """
    saved: list[str] = []
    for rid in regions:
        path = plot_all_overlaps_grid(
            region=rid,
            monkeys=monkeys,
            zscore_codes=zscore_codes,
            methods=methods,
            figsize_scale=figsize_scale,
            cell_size=cell_size,
            save=True,
            save_dir=save_dir,         # None → TVSD/PLOTS_HEAT_MAP
            save_dpi=save_dpi,
            show=False,
        )
        if path:
            saved.append(path)
    if saved:
        print("[✓] All regions saved:")
        for p in saved:
            print("   •", p)
    return saved



if __name__ == "__main__":
    
     save_all_regions_heatmaps(
         regions=(1,2,3 ),                       # V1, V4, IT
         monkeys=("Monkey N", "Monkey F"),
         zscore_codes=(1, 2, 3, 4),
         methods=("residual", "baseline100"),
         figsize_scale=1.0,
         cell_size=2.7,
         save_dir=None,                           # default: <TVSD>/PLOTS_HEAT_MAP
         save_dpi=450
     )
    