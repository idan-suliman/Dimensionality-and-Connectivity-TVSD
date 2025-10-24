from runtime import runtime
import numpy as np
import matplotlib.pyplot as plt
from __future__ import annotations
from collections import defaultdict
from itertools import product
from matplotlib.gridspec import GridSpec

# לטעון ולסדר את החזרות במטריצות נפרדות
def load_repetition_matrices(method: str):
    """
    Build 30 repetition matrices (each: 100 × n_electrodes).
    Uses project constants via runtime.get_consts().
    
    Flow:
      1) Load trials + constants.
      2) Sanity checks: stimulus_id exists, 100 unique stimuli, each appears 30 times,
         each repetition has 100 trials.
      3) Optional residual: per-stimulus mean subtraction (pre-averaging, per timepoint & electrode).
      4) Average over time window:
            - baseline100 → [0, 100)
            - residual    → REGION_WINDOWS[region] from constants
            - otherwise   → [0, TRIAL_LENGTH_MS)
      5) Return list length 30 of matrices (100 × n_electrodes).
    """
    # ---------- Load config & constants ----------
    cfg     = runtime.get_cfg()
    consts  = runtime.get_consts()
    trials  = cfg._load_trials()

    print(f"[🐒] {cfg.get_monkey_name()} | Z={cfg.get_zscore_title()} | Method={method}")
    print(f"[📦] Loaded {len(trials)} trials")

    rep_groups: dict[int, list[dict]] = defaultdict(list)
    for tr in trials:
        rep = tr.get("rep_idx")
        if rep is None:
            rep = int(tr["allmat_row"][3]) - 1
        rep_groups[int(rep)].append(tr)

    first = trials[0]
    region_id = first.get("region_id")
    if region_id is None:
        region_name = str(first.get("region", "V1")).upper()
        region_id = consts.REGION_NAME_TO_ID.get(region_name, 1)
    region_id = int(region_id)

    return [
        cfg.build_trial_matrix(
            region_id=region_id,
            analysis_type=method,
            trials=rep_groups[r],
            stimulus_key="stimulus_id",
        )
        for r in range(consts.NUM_REPETITIONS)
    ]


# לכל מטריצת חזרה לעלות PCA ומימד דומיננטי
def compute_variational_space(X: np.ndarray):
    # Center the data
    X_centered = X - X.mean(axis=0, keepdims=True)

    # PCA using SVD (no need to compute covariance matrix explicitly)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Vt.shape = (n_components, n_features)
    eigvals = (S ** 2) / (X.shape[0] - 1)  # variance explained per component

    # Normalize variances
    v = eigvals / eigvals.sum()

    # Participation Ratio (effective dimensionality)
    pr_value = (v.sum() ** 2) / np.sum(v ** 2)
    D = int(np.ceil(pr_value))  # dominant dimensionality

    # Return both the full PCA space and the D-dimensional span
    pcs_dominant = Vt[:D, :]  # span of first D components
    all_pcs = Vt               # all components up to min(rows, cols)

    return pcs_dominant, D

# חישוב המ"פ של כל זוג מרחבים של חזרות
def compute_repetition_overlap(method: str):
    # Load data
    reps_data = load_repetition_matrices(method)
    n_reps = len(reps_data)
    print(f"[📊] Loaded {n_reps} repetition matrices")

    # Compute variational spaces
    spaces, Ds = [], []
    for i, X in enumerate(reps_data):
        pcs, D = compute_variational_space(X)
        spaces.append(pcs)
        Ds.append(D)
        print(f"  • Rep {i+1:02d}: PR={D:.2f}, D={D}")

    # Find shared dimensionality
    D_common = int(np.floor(np.mean(Ds)))
    # D_common = int(np.min(Ds))
    print(f"[🧩] Common dominant dimensionality: D={D_common}")

    # Cut all spaces to the shared dimensionality
    spaces = [pcs[:D_common, :] for pcs in spaces]

    # Compute pairwise overlaps
    overlap_matrix = np.eye(n_reps)
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            # Cross-projection matrix
            C = spaces[i] @ spaces[j].T  # shape (D, D)
            # Singular values of overlap
            svals = np.linalg.svd(C, compute_uv=False)
            overlap = np.mean(svals ** 2)  # average squared singular values
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap

    print(f"[✅] Overlap matrix computed — shape: {overlap_matrix.shape}")
    return overlap_matrix, D_common

# מפת חום לכל קוף שיטה מיצוע
def plot_overlap_matrix(overlap_matrix: np.ndarray, D_common: int, method: str):
    
    cfg = runtime.get_cfg()
    n_reps = overlap_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(overlap_matrix, cmap='viridis', vmin=0, vmax=1)

    # Titles and labels
    ax.set_title(f"Overlap Between Repetition Subspaces\n"
                 f"{cfg.get_monkey_name()}, Z={cfg.get_zscore_title()}, Method={method}, D={D_common}",
                 fontsize=13, pad=12, weight='bold')

    ax.set_xlabel("Repetition index", fontsize=12)
    ax.set_ylabel("Repetition index", fontsize=12)

    # Ticks every 5 reps for readability
    ticks = np.arange(0, n_reps, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticklabels(ticks + 1)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Subspace overlap", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Aesthetics
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

# מדפיס את כל המפות חום לשני הקופים וכל השיטות כלומר כל הקומבינציות ביחד
def plot_all_overlaps_grid(
    monkeys=("Monkey N", "Monkey F"),
    zscore_codes=(1,2,3,4),
    methods=("residual","baseline100"),
    figsize_scale=1.0
):
    # --- compute all overlaps (no plotting) ---
    results=[]
    for m,z,met in product(monkeys,zscore_codes,methods):
        runtime.set_cfg(m, z)
        O,D = compute_repetition_overlap(met)
        results.append((m,z,met,O,D))

    R,Z,M = len(monkeys), len(zscore_codes), len(methods)
    C = Z*M

    # === גריד מדויק עם גבהים מותאמים מאוד ===
    fig = plt.figure(figsize=(4.8*C*figsize_scale, 3.3*R*figsize_scale))
    # גובה כל שורה מוקטן מאוד, ורווח hspace כמעט 0
    gs = GridSpec(
        R+1, C, figure=fig,
        height_ratios=[1]*R + [0.1],
        hspace=0.02, wspace=0.05
    )

    axes = np.empty((R, C), dtype=object)
    last_im = None

    # === ציור כל סאבפלוט ===
    for m,z,met,O,D in results:
        r = monkeys.index(m)
        c = zscore_codes.index(z)*M + methods.index(met)
        ax = fig.add_subplot(gs[r, c])
        axes[r,c] = ax
        last_im = ax.imshow(O, vmin=0, vmax=1, cmap="viridis")
        ax.text(0.98, 0.05, f"D={D}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color="w",
                bbox=dict(facecolor="k", alpha=0.25, pad=2, edgecolor="none"))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    # === כותרות עליונות ===
    fig.canvas.draw()
    cx = lambda a: (a.get_position().x0 + a.get_position().x1)/2
    cy = lambda a: (a.get_position().y0 + a.get_position().y1)/2
    z_y, m_y = 0.96, 0.94  # ↓ הנמכה קלה כדי ליצור הפרדה יפה
    for zi,z in enumerate(zscore_codes):
        c0, c1 = zi*M, zi*M+(M-1)
        fig.text((cx(axes[0,c0])+cx(axes[0,c1]))/2, z_y, f"Z={z}",
                 ha="center", va="top", fontsize=14, fontweight="bold")
        for mi,met in enumerate(methods):
            fig.text(cx(axes[0, zi*M+mi]), m_y, met,
                     ha="center", va="top", fontsize=11)

    # === תוויות שורה ===
    for r,m in enumerate(monkeys):
        fig.text(0.006, cy(axes[r,0]), m, ha="left", va="center",
                 rotation=90, fontsize=13, fontweight="bold")

    # === כותרת כללית ===
    fig.suptitle("repetitions Subspace Overlap, D-min",
                 fontsize=16, y=0.983, fontweight="bold")

    # === מקרא (Colorbar) נמוך ומופרד ===
    cax = fig.add_subplot(gs[-1, :])
    cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cbar.set_label("Subspace Overlap", fontsize=12)

    plt.show()





plot_all_overlaps_grid()
