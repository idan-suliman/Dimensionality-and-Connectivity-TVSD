from __future__ import annotations
import numpy as np
import matplotlib.patheffects as pe

def d95_from_curves(rrr_mean: np.ndarray, ridge_mean: float, d_max: int) -> int:
    """
    Compute minimal dimension d such that RRR(d) >= 0.95 * Ridge_R^2.
    Returns d_max if 95% threshold is not reached.
    """
    thr = 0.95 * float(ridge_mean)
    idx = np.where(rrr_mean >= thr)[0]
    return int(idx[0] + 1) if idx.size else int(d_max)

def jitter(values, rng: np.random.Generator, *, scale: float = 0.15) -> np.ndarray:
    """
    Add small uniform noise to values for scatter plot visibility.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    return arr + rng.uniform(-scale, scale, size=arr.shape)

def square_limits(x_vals, y_vals, *, base_min: int = 1, scale: float = 1.5) -> tuple[int, int]:
    """
    Compute symmetric square axis limits covering x and y values.
    """
    vmax_list = [np.max(np.atleast_1d(x_vals)), np.max(np.atleast_1d(y_vals))]
    vmax = float(np.max(vmax_list)) if vmax_list else 1.0
    lim_max = int(np.ceil(scale * vmax))
    lim_max = max(lim_max, base_min + 1)
    return (base_min, lim_max)

def labeled_dot(ax, x, y, label, *, face, edge: str = "k",
                size: float = 240, text_size: float = 12, text_color: str = "white"):
    """
    Draw a filled scatter marker with a centered, legible label.
    """
    ax.scatter([x], [y], s=size, facecolors=face, edgecolors=edge,
               linewidths=1.6, zorder=7)
    ax.text(x, y, str(label), ha="center", va="center",
            color=text_color, fontsize=text_size, weight="bold",
            zorder=9, path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])

def smart_label(ax, name, xs, ys, color, groups):
    """
    Place a text label near the cluster centroid, avoiding overlap with other clusters.
    Uses a greedy search for the best direction (8 surrounding points).
    """
    if len(xs) == 0:
        return

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    # 1. Calculate cluster stats
    cx, cy = xs.mean(), ys.mean()
    rx = max(xs.std() * 2.2, 1.3)
    ry = max(ys.std() * 2.2, 1.3)

    # 2. Candidate offsets (8 directions)
    dirs = np.array([
        [ 1,  0], [ -1,  0],
        [ 0,  1], [  0, -1],
        [ 1,  1], [ -1,  1],
        [ 1, -1], [ -1, -1],
    ], float)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    best_dir = np.array([0, 1])
    best_score = -1e18

    # 3. Find optimal position maximizing distance to other clusters
    for d in dirs:
        tx = cx + d[0] * (rx + 1.8)
        ty = cy + d[1] * (ry + 1.8)

        # Check boundaries
        if not (xmin + 1 < tx < xmax - 1): continue
        if not (ymin + 1 < ty < ymax - 1): continue

        # Check distance to others
        min_dist = 1e9
        for gname, g in groups.items():
            if gname == name or len(g["xs"]) == 0:
                continue
            gx = np.asarray(g["xs"], float)
            gy = np.asarray(g["ys"], float)
            dist = np.min((gx - tx)**2 + (gy - ty)**2)
            min_dist = min(min_dist, dist)

        if min_dist > best_score:
            best_score = min_dist
            best_dir = d

    # 4. Place label
    tx = cx + best_dir[0] * (rx + 1.8)
    ty = cy + best_dir[1] * (ry + 1.8)

    ax.text(
        tx, ty, name, color=color, fontsize=15, fontweight="bold",
        ha="center", va="center", bbox=None, zorder=50
    )
