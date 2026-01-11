"""
visualization.py
----------------
Consolidated plotting utilities for TVSD codebase.
Extracts logic from Semedo.py, kyle_method.py, etc.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Any
import matplotlib.patheffects as pe
from pathlib import Path
from itertools import product
from core.runtime import runtime

# =============================================================================
# Helper Functions
# =============================================================================
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
import pandas as pd
from collections import Counter, defaultdict

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


# =============================================================================
# Main Plotting Functions
# =============================================================================


class SemedoFigures:
    """
    Container for Semedo 2019 replication figures.
    """

    @staticmethod
    def plot_figure_4(
        perf_full: dict,
        perf_match: dict,
        d95_full_g: int, 
        d95_match_g: int,
        d95_full_rep: list[int],
        d95_match_rep: list[int],
        d_max: int,
        target_region: int,
        analysis_type: str,
        k_subsets: int | None,
        outer_splits: int,
        inner_splits: int,
        random_state: int,
        label_D: str,
        save_path: str | None = None,
    ):
        """
        Reconstruct the Standard 4-panel Figure 4 (Previously Semedo Figure 5A style).
        - Panel A: Full model performance curve.
        - Panel B: Match model performance curve.
        - Panel C: Single run summary (d95 Full vs d95 Match).
        - Panel D: Distribution of d95 over subsets/repetitions.
        """
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
        label_fs = 20

        # --- Panel A: Full Model ---
        axA.errorbar(dims, perf_full["rrr_R2_mean"], yerr=perf_full["rrr_R2_sem"],
                     fmt="o-", ms=3.8, lw=1.35, capsize=3, color=colA, zorder=2)
        axA.scatter([1], [perf_full["ridge_R2_mean"]], marker="^", s=90,
                    color=colA, edgecolors="k", zorder=3)
        if np.isfinite(d95_full_g):
            r2d = perf_full["rrr_R2_mean"][int(d95_full_g) - 1]
            labeled_dot(axA, int(d95_full_g), float(r2d), int(d95_full_g),
                        face=colA, edge="k", size=240, text_size=12, text_color="white")
        axA.set_title(f"Predicting {tgt}", color=colA, pad=10, fontsize=22)
        axA.grid(alpha=.25)
        axA.text(-0.07, 1.05, "A", transform=axA.transAxes, ha="left", va="bottom", fontsize=20, fontweight="bold")
        axA.set_box_aspect(1)
        axA.tick_params(axis='both', which='major', labelsize=20, width=1.5)

        # --- Panel B: Match Model ---
        axB.errorbar(dims, perf_match["rrr_R2_mean"], yerr=perf_match["rrr_R2_sem"],
                     fmt="o-", ms=3.8, lw=1.35, capsize=3, color=colB, zorder=2)
        axB.scatter([1], [perf_match["ridge_R2_mean"]], marker="^", s=90,
                    color=colB, edgecolors="k", zorder=3)
        if np.isfinite(d95_match_g):
            r2d = perf_match["rrr_R2_mean"][int(d95_match_g) - 1]
            labeled_dot(axB, int(d95_match_g), float(r2d), int(d95_match_g),
                        face=colB, edge="k", size=240, text_size=12, text_color="white")
        axB.set_title(f"Predicting V1-match {tgt}", color=colB, pad=10, fontsize=22)
        axB.grid(alpha=.25)
        axB.text(-0.07, 1.05, "B", transform=axB.transAxes, ha="left", va="bottom", fontsize=20, fontweight="bold")
        axB.set_box_aspect(1)
        axB.tick_params(axis='both', which='major', labelsize=20, width=1.5)

        # --- Panel C: Summary Dot ---
        if np.isfinite(d95_match_g) and np.isfinite(d95_full_g):
            xg, yg = int(d95_match_g), int(d95_full_g)
            xmin, xmax = square_limits([xg], [yg], base_min=1, scale=1.5)
        else:
            xmin, xmax = square_limits([1], [1], base_min=1, scale=1.5)
        
        rng_jitter = np.random.default_rng(random_state)
        axC.plot([xmin, xmax], [xmin, xmax], ls="--", lw=0.9, color="k")
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
        axC.text(-0.07, 1.05, "C", transform=axC.transAxes, ha="left", va="bottom", fontsize=20, fontweight="bold")
        axC.tick_params(axis='both', which='major', labelsize=20, width=1.5)

        # --- Panel D: Subsets/Repetitions ---
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
        axD.text(0.02, 0.98, label_D, transform=axD.transAxes, ha="left", va="top",
                 fontsize=20, fontweight="bold", color="black")
        axD.grid(False)
        axD.text(-0.07, 1.05, "D", transform=axD.transAxes, ha="left", va="bottom", fontsize=20, fontweight="bold")
        axD.tick_params(axis='both', which='major', labelsize=20, width=1.5)

        # --- Clean up and shared labels ---
        for ax in (axA, axB, axC, axD):
            ax.set_ylabel(None)
        
        # Calculate label positions (Figure coordinates)
        fig.canvas.draw()
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
                 va="center", ha="right", rotation="vertical", fontsize=label_fs+7, color="black",fontweight="bold")
        fig.text(right_ylabel_x, right_block_ycent,
                 f"{tgt} Predictive dimensions",
                 va="center", ha="right", rotation="vertical", fontsize=label_fs+7, color="#9C1C1C",fontweight="bold")
        fig.text(left_block_xcent, left_xlabel_y, "Predictive dimensions (d)",
                 va="top", ha="center", fontsize=label_fs+3, color="black",fontweight="bold")
        fig.text(right_block_xcent, right_xlabel_y, "Target V1 Predictive dimensions",
                 va="top", ha="center", fontsize=label_fs+3, color="#1565C0",fontweight="bold")
        
        top_row_ymax = max(boxA.y1, boxC.y1)
        fig.suptitle(
            f"{runtime.get_cfg().get_monkey_name()}  |  {runtime.get_cfg().get_zscore_title()}  |  {analysis_type.upper()}",
            fontsize=20, y=min(0.998, top_row_ymax + 0.080), fontweight="bold"
        )

        # --- Save ---
        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=400, facecolor="white")
            print(f"[✓] Figure 4 saved → {out_path}")
        
        plt.close(fig)


    @staticmethod
    def plot_figure_4_subset(
        runs_full: list[dict],
        runs_match: list[dict],
        d95_full_runs: list[int],
        d95_match_runs: list[int],
        d95_full_sub_all: list[tuple[int, int]],
        d95_match_sub_all: list[tuple[int, int]],
        target_region: int,
        analysis_type: str,
        n_src_eff: int,
        n_tgt_eff: int,
        n_runs: int,
        k_subsets: int | None,
        outer_splits: int,
        inner_splits: int,
        d_max: int,
        random_state: int,
        colors: list[str],
        save_path: str | None = None,
    ) -> None:
        """
        Plot the Multi-Run Subset Figure 4.
        - Panel A: Full curves (all runs).
        - Panel B: Match curves (all runs).
        - Panel C: Per-run d95 summary.
        - Panel D: Per-subset d95 summary (color-coded by run).
        """
        cfg = runtime.get_cfg()
        tgt_nm = runtime.get_consts().REGION_ID_TO_NAME[target_region]

        # --- Figure Layout ---
        fig = plt.figure(figsize=(14, 13), dpi=400)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12, wspace=0.15, hspace=0.30)
        gs  = gridspec.GridSpec(2, 2)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[1, 0])
        axC = fig.add_subplot(gs[0, 1])
        axD = fig.add_subplot(gs[1, 1])

        label_fs = 20
        colA, colB = "#9C1C1C", "#1565C0"
        rng_jitter = np.random.default_rng(random_state)
        dims = np.arange(1, d_max + 1)

        # --- Panel A: FULL curves (per run) ---
        for i, cur in enumerate(runs_full):
            axA.errorbar(dims, cur["rrr"], yerr=cur["sem"],
                         fmt="-o", ms=3.2, lw=1.1, capsize=3,
                         color=cur["color"], zorder=2, alpha=0.95)
            axA.scatter([1], [cur["ridge"]], marker="^", s=70,
                        color=cur["color"], edgecolors="k", zorder=3)
            if np.isfinite(cur["d95"]):
                d = int(cur["d95"])
                y = float(cur["rrr"][d - 1])
                labeled_dot(axA, d, y, d, face=cur["color"], edge="k", size=200, text_size=10, text_color="white")
        axA.set_title(f"Predicting {tgt_nm}", color=colA, pad=10, fontsize=21)
        axA.grid(alpha=.25)
        axA.text(-0.07, 1.05, "A", transform=axA.transAxes, ha="left", va="bottom", fontsize=20, fontweight="bold")
        axA.set_box_aspect(1)
        axA.tick_params(axis='both', which='major', labelsize=23, width=1.5)

        # --- Panel B: MATCH curves (per run) ---
        for i, cur in enumerate(runs_match):
            axB.errorbar(dims, cur["rrr"], yerr=cur["sem"],
                         fmt="-o", ms=3.2, lw=1.1, capsize=3,
                         color=cur["color"], zorder=2, alpha=0.95)
            axB.scatter([1], [cur["ridge"]], marker="^", s=70,
                        color=cur["color"], edgecolors="k", zorder=3)
            if np.isfinite(cur["d95"]):
                d = int(cur["d95"])
                y = float(cur["rrr"][d - 1])
                labeled_dot(axB, d, y, d, face=cur["color"], edge="k", size=200, text_size=10, text_color="white")
        axB.set_title(f"Predicting V1-match {tgt_nm}", color=colB, pad=10, fontsize=21)
        axB.grid(alpha=.25)
        axB.text(-0.07, 1.05, "B", transform=axB.transAxes, ha="left", va="bottom", fontsize=20, fontweight="bold")
        axB.set_box_aspect(1)
        axB.tick_params(axis='both', which='major', labelsize=23, width=1.5)

        # --- Panel C: Run Summary ---
        if len(d95_full_runs) and len(d95_match_runs):
            xmin_c, xmax_c = square_limits(d95_match_runs, d95_full_runs, base_min=1, scale=1.5)
        else:
            xmin_c, xmax_c = (1, max(2, int(np.ceil(1.5 * d_max))))
        axC.plot([xmin_c, xmax_c], [xmin_c, xmax_c], ls="--", lw=0.9, color="k")
        for i, (xm, ym) in enumerate(zip(d95_match_runs, d95_full_runs)):
            jx = xm + rng_jitter.uniform(-0.15, 0.15)
            jy = ym + rng_jitter.uniform(-0.15, 0.15)
            axC.scatter([jx], [jy], s=155, facecolors="white",
                        edgecolors=colors[i], linewidths=1.5, zorder=4)
            axC.text(jx, jy, str(i+1), ha="center", va="center", fontsize=10, weight="bold",
                     color="black", zorder=5, path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        axC.set_xlim(xmin_c, xmax_c)
        axC.set_ylim(xmin_c, xmax_c)
        axC.set_aspect('equal', adjustable='box')
        ticks_c = np.arange(xmin_c, xmax_c + 1, max(1, int(np.ceil((xmax_c - xmin_c) / 6))))
        axC.set_xticks(ticks_c)
        axC.set_yticks(ticks_c)
        axC.grid(False)
        axC.text(-0.07, 1.05, "C", transform=axC.transAxes, ha="left", va="bottom", fontsize=20, fontweight="bold")
        axC.tick_params(axis='both', which='major', labelsize=23, width=1.5)

        # --- Panel D: Subset Scatter ---
        d95_match_vals = [v for (_, v) in d95_match_sub_all]
        d95_full_vals  = [v for (_, v) in d95_full_sub_all]
        if len(d95_full_vals) and len(d95_match_vals):
            xmin_d, xmax_d = square_limits(d95_match_vals, d95_full_vals, base_min=1, scale=1.5)
        else:
            xmin_d, xmax_d = (1, max(2, int(np.ceil(1.5 * d_max))))
        
        axD.plot([xmin_d, xmax_d], [xmin_d, xmax_d], ls="--", lw=0.9, color="k") # y=x ref

        for (run_idx, yf), (_, xf) in zip(d95_full_sub_all, d95_match_sub_all):
            jx = xf + rng_jitter.uniform(-0.15, 0.15)
            jy = yf + rng_jitter.uniform(-0.15, 0.15)
            axD.scatter([jx], [jy], s=46, facecolors="white",
                        edgecolors=colors[run_idx], linewidths=0.9, alpha=0.95)

        axD.set_xlim(xmin_d, xmax_d)
        axD.set_ylim(xmin_d, xmax_d)
        axD.set_aspect('equal', adjustable='box')
        ticks_d = np.arange(xmin_d, xmax_d + 1, max(1, int(np.ceil((xmax_d - xmin_d) / 6))))
        axD.set_xticks(ticks_d)
        axD.set_yticks(ticks_d)
        axD.grid(False)
        
        # Subtitle inside panel
        axD.text(0.98, 0.98,
                 ("Repetitions" if k_subsets is None else f"{k_subsets} random subsets × {n_runs} runs"),
                 transform=axD.transAxes, ha="right", va="top", fontsize=19, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        axD.text(-0.07, 1.05, "D", transform=axD.transAxes, ha="left", va="bottom", fontsize=20, fontweight="bold")
        axD.tick_params(axis='both', which='major', labelsize=23, width=1.5)

        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
                              markeredgecolor=colors[i], markersize=7, lw=0) for i in range(n_runs)]
        labels = [f"Run {i+1}" for i in range(n_runs)]
        axD.legend(handles, labels, loc="lower right", fontsize=12, frameon=False, title="Runs", title_fontsize=14)

        # --- Suptitle ---
        boxA, boxC = axA.get_position(), axC.get_position()
        top_row_ymax = max(boxA.y1, boxC.y1)
        fig.suptitle(
            f"{cfg.get_monkey_name()}  |  {cfg.get_zscore_title()}  |  "
            f"{analysis_type.upper()}  (n_src={n_src_eff}, n_tgt={n_tgt_eff}, runs={n_runs})",
            fontsize=20, y=min(0.998, top_row_ymax + 0.080), fontweight="bold"
        )

        # --- Grouped Labels ---
        fig.canvas.draw()
        boxA, boxB = axA.get_position(), axB.get_position()
        boxC, boxD = axC.get_position(), axD.get_position()

        left_x0, left_x1 = min(boxA.x0, boxB.x0), max(boxA.x1, boxB.x1)
        left_y0, left_y1 = min(boxA.y0, boxB.y0), max(boxA.y1, boxB.y1)
        left_xctr = 0.5 * (left_x0 + left_x1)
        left_yctr = 0.5 * (left_y0 + left_y1)

        right_x0, right_x1 = min(boxC.x0, boxD.x0), max(boxC.x1, boxD.x1)
        right_y0, right_y1 = min(boxC.y0, boxD.y0), max(boxC.y1, boxD.y1)
        right_xctr = 0.5 * (right_x0 + right_x1)
        right_yctr = 0.5 * (right_y0 + right_y1)

        fig.text(left_x0 - 0.065, left_yctr,
                 rf"Mean $R^2$  (outer {outer_splits}, inner {inner_splits})",
                 va="center", ha="right", rotation="vertical", fontsize=label_fs + 6, color="black", fontweight="bold")
        fig.text(right_x0 - 0.045, right_yctr,
                 f"{tgt_nm} Predictive dimensions",
                 va="center", ha="right", rotation="vertical", fontsize=label_fs + 6, color=colA, fontweight="bold")
        fig.text(left_xctr, left_y0 - 0.035, "Predictive dimensions (d)",
                 va="top", ha="center", fontsize=label_fs + 2, color="black", fontweight="bold")
        fig.text(right_xctr, right_y0 - 0.035, "Target V1 Predictive dimensions",
                 va="top", ha="center", fontsize=label_fs + 2, color=colB, fontweight="bold")

        # --- Save ---
        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=400, facecolor="white")
            print(f"[✓] Figure 4 Subset saved → {out_path}")
        
        plt.close(fig)


    @staticmethod
    def plot_figure_5_b(csv_path: str, out_path: str | None = None, title: str | None = None):
        """
        Generate Semedo Figure 5B (Dimensionality Matching) by reading data from a CSV file.
        
        Args:
            csv_path: Path to the CSV file containing [group, x, y, point_type]
            out_path: If provided, saves the figure here.
        """
        import csv

        # Configuration mapped by group name
        # We expect standard names: "V4", "Target V4", "IT", "Target IT"
        styles = {
            "V4":             {"color": "#ca1b1b"},
            "Target V4":      {"color": "#cf2359"},
            "IT":             {"color": "#0529ae"},
            "Target IT":      {"color": "#3391e3"},
        }
        
        # Containers
        subsets = {k: {"xs": [], "ys": []} for k in styles}
        full_points = {k: {"xs": [], "ys": []} for k in styles} # Usually 1 per group
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                grp = row["group"]
                x = float(row["x_dimensionality_d95"])
                y = float(row["y_predictive_d95"])
                pt_type = row.get("point_type", "subset") # Default to subset for backward compat
                
                if grp not in styles:
                    # Fallback for unknown groups?
                    continue
                    
                if pt_type == "full":
                    full_points[grp]["xs"].append(x)
                    full_points[grp]["ys"].append(y)
                else:
                    subsets[grp]["xs"].append(x)
                    subsets[grp]["ys"].append(y)

        # Plotting
        fig, ax = plt.subplots(figsize=(7, 6.2), dpi=140)
        rng = np.random.default_rng(0) # Fixed seed for jitter consistency

        for name, style in styles.items():
            # Scatter Subsets (Small dots, jittered)
            xs = np.array(subsets[name]["xs"])
            ys = np.array(subsets[name]["ys"])
            color = style["color"]
            
            if len(xs) > 0:
                xs_j = jitter(xs, rng, scale=0.2)
                ys_j = jitter(ys, rng, scale=0.2)
                ax.scatter(xs_j, ys_j, s=50, alpha=0.9, color=color, label=name if "Target" not in name else "")
                
                # Mean dot
                ax.scatter([xs.mean()], [ys.mean()], s=40, edgecolor='k', linewidths=0.8, color=color)

            # Scatter Full Points (Large Diamonds)
            fxs = np.array(full_points[name]["xs"])
            fys = np.array(full_points[name]["ys"])
            if len(fxs) > 0:
                 ax.scatter(
                    fxs, fys,
                    s=90,
                    color=color,
                    marker="D",
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=1.0,
                    zorder=20,
                )

        # y = x line
        # Calculate limits based on all data
        all_vals = []
        for d in (subsets, full_points):
            for g in d.values():
                all_vals.extend(g["xs"])
                all_vals.extend(g["ys"])
                
        lim_max = max(all_vals) * 1.05 if all_vals else 50
        ax.plot([0, lim_max], [0, lim_max], 'k--', lw=1)
        
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel("Target Population dimensionality", fontsize=20, fontweight="bold")
        ax.set_ylabel("Number of Predictive dimensions", fontsize=19, fontweight="bold")
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.8, length=6)

        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        # Smart Labels for subsets
        # We reconstruct the 'groups' structure expected by smart_label
        # smart_label expects groups[name]["xs"] to be the coordinates
        dummy_groups_for_label = {k: v for k, v in subsets.items()}
        
        for name, style in styles.items():
            if len(subsets[name]["xs"]) > 0:
                smart_label(ax, name, subsets[name]["xs"], subsets[name]["ys"], style["color"], dummy_groups_for_label)

        fig.tight_layout()
        
        if out_path:
            # Ensure directory exists? visualization usually just saves.
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight")
            print(f"[✓] Re-plotted Figure 5B from CSV → {out_path}")
        else:
            plt.show() # Interactive fallback

        plt.close(fig)


def plot_rrr_ridge_comparison(
    source_region: int = 1,
    target_region: int = 2,
    *,
    d_max: int = 35,
    alpha: float | None = None,
    outer_splits: int = 3,
    inner_splits: int = 3,
    random_state: int = 0,
    analysis_types: tuple[str, ...] | None = None,
    match_to_target: bool = False,
    _external_X: np.ndarray | None = None,
    _external_Ys: dict[str, np.ndarray] | None = None,
    custom_title: str | None = None,
):
    """
    Plot RRR vs Ridge performance curves (Figure 5B style).
    Delegates math to methods.rrr.RRRAnalyzer to avoid circular imports.
    """
    from .rrr import RRRAnalyzer  # Local import

    if analysis_types is None:
        analysis_types = runtime.get_consts().ANALYSIS_TYPES

    using_external = (_external_X is not None) and (_external_Ys is not None)
    cmap = {"window": "#C21807", "baseline100": "#1565C0", "residual": "#2E7D32"}

    # --- Figure Layout ---
    n_info = len(analysis_types)
    fig_h  = 4.2 + 0.35 * n_info
    fig    = plt.figure(figsize=(7, fig_h))

    gs = gridspec.GridSpec(2, 1, height_ratios=[4.0, 0.35 * n_info], hspace=0.30)
    ax_main = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    dims, yvals = np.arange(1, d_max + 1), []
    sing_vals, lam_lines = [], []

    # --- Main Calculation Loop ---
    for at in analysis_types:
        col = cmap.get(at, "#000000")

        if using_external:
            X, Y = _external_X, _external_Ys[at]
        else:
            Y, X = RRRAnalyzer.build_mats(source_region, target_region, at, match_to_target=match_to_target)

        res = RRRAnalyzer.compute_performance(
            Y, X,
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state
        )

        ax_main.errorbar(dims, res["rrr_R2_mean"], yerr=res["rrr_R2_sem"],
                        fmt="o-", ms=3, lw=1.2, capsize=3,
                        color=col, ecolor=col)
        ax_main.scatter([1], [res["ridge_R2_mean"]], marker="^",
                        s=80, color=col, edgecolors="k")
        yvals.extend(res["rrr_R2_mean"]); yvals.append(res["ridge_R2_mean"])

        # Mark d95
        thr = 0.95 * res["ridge_R2_mean"]
        idx = np.where(res["rrr_R2_mean"] >= thr)[0]
        if idx.size:
            d95, r2d = int(idx[0]+1), float(res["rrr_R2_mean"][idx[0]])
            ax_main.scatter(d95, r2d, s=140, facecolor=col,
                            edgecolors="k", lw=1)
            ax_main.text(d95, r2d, f"{d95}", color="white", weight="bold",
                        ha="center", va="center",
                        path_effects=[pe.Stroke(linewidth=1.8, foreground="black"), pe.Normal()])
            ax_main.axvline(d95, color=col, ls="--", lw=0.7, alpha=0.4)

        if alpha is None:
            # SV of centered X
            Xc = X - X.mean(0, keepdims=True)
            sing_vals.append(int(round(np.linalg.svd(Xc, compute_uv=False)[0])))
            
            lam_vec, _ = RRRAnalyzer._lambda_grid(X)
            chosen = res["lambdas"]
            idx_pos = [int(np.abs(lam_vec - l).argmin()+1) for l in chosen]
            pairs = [f"{int(round(l))} ({p})" for l, p in zip(chosen, idx_pos)]
            if len(pairs) > 3:
                pairs = pairs[:3] + ["…"]
            lam_lines.append(", ".join(pairs))

    # --- Cosmetics ---
    pad = 0.05 * (max(yvals) - min(yvals) if (yvals and max(yvals) > min(yvals)) else 0.05)
    ymin, ymax = (0.0, 1.0)
    if yvals:
         ymin, ymax = min(yvals)-pad, max(yvals)+pad
    ax_main.set_ylim(max(0.0, ymin), min(1.0, ymax))
    ax_main.set_xlabel("Predictive dimensions (d)", labelpad=6)        
    ax_main.set_ylabel(rf"Mean $R^2$  (CV: outer {outer_splits}, inner {inner_splits})")
    ax_main.grid(alpha=0.3)

    tgt_lbl = runtime.get_consts().REGION_ID_TO_NAME.get(target_region, str(target_region))
    ax_main.set_title(custom_title or
                    f"V1 → {'V1-match '+tgt_lbl if match_to_target else tgt_lbl}",
                    fontsize=12, pad=10)

    # --- Info Block ---
    ax_info.axis("off")
    if alpha is None:
        step = 1.0 / (n_info + 1)
        for i, (at, s1, lam) in enumerate(zip(analysis_types, sing_vals, lam_lines)):
            ax_info.text(0.5, 1-step*(i+1),
                        f"{at}:  σ₁ = {s1}   |   λ* per fold:  {lam}",
                        ha="center", va="center",
                        fontsize=9, color=cmap.get(at, "k"))

    # --- Save ---
    fig.tight_layout()
    tag = "nestedLam" if alpha is None else f"lam{RRRAnalyzer._lambda_for_fname(alpha)}"
    plot_dir = RRRAnalyzer._plot_dir(match_to_target)
    
    base = (f"{runtime.get_cfg().get_monkey_name().replace(' ', '')}_rrr_"
            f"{'target' if match_to_target else 'regular'}_"
            f"V1_to_{tgt_lbl}_{tag}")
            
    fname = plot_dir / f"{base}.png"
    fig.savefig(fname, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[✓] RRR figure saved → {fname}")


def plot_lag_histogram(
    panels: list[tuple[np.ndarray, np.ndarray, float, str]],
    results: dict,
    regions: tuple[int, ...],
    monkey: str,
    method: str,
    zscore_title: str,
    out: str | None = None,
    show: bool = True
):
    """
    Plot histograms of permutation betas for Lag analysis.
    """
    nR = len(regions)
    fig, axes = plt.subplots(1, nR, figsize=(5.6 * nR, 4.6), dpi=150, constrained_layout=True)
    if nR == 1:
        axes = [axes]
    
    if not isinstance(axes, (list, np.ndarray)):
         axes = [axes]

    for ax, (hist, bin_edges, beta_obs, name), rid in zip(axes, panels, regions):
        pval = results[rid]["p_perm"]
        D    = results[rid]["D_common"]
        n_perm = results[rid]["n_perm"]
        
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.bar(centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.9)
        ax.axvline(beta_obs, color="k", lw=2.2, label=f"β_obs={beta_obs:.4g}")
        ax.set_xlabel("Permutation slope β (Monte-Carlo)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{name} (D={D})\n p_perm={pval:.3g}  |  perms={n_perm:,}", fontsize=12)
        ax.legend(frameon=False, fontsize=10)

    sup = f"{monkey} | Z={zscore_title} | {method} | WLS slope β (Δr>0)"
    fig.suptitle(sup, fontsize=14)

    if out:
        plt.savefig(out, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_overlap_matrix(overlap_matrix: np.ndarray, D_common: int, method: str) -> None:
    """
    Plot heatmap of subspace overlaps (Kyle's method).
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
    ax.tick_params(axis="both", which="major", length=0)
    ax.set_aspect("auto")

    plt.tight_layout()
    plt.show()

# =============================================================================
# Dimensionality vs Correlation Visualization
# =============================================================================

class DimCorrVisualizer:
    
    @staticmethod
    def plot_curves(results_list: list[dict[str, Any]], 
                   title: str, 
                   subtitle: str,
                   output_path: str):
        """
        Plots multiple curves on a single figure.
        results_list: List of result dicts (from DimCorrAnalyzer).
        """
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(results_list))))
        
        for idx, res in enumerate(results_list):
            dims = np.array(res["dims"])
            rhos = np.array(res["rhos"])
            pvals = np.array(res["p_vals"])
            
            # Label generation
            if res["type"] == "region":
                if "src" in res:
                    lbl = res["src"]
                else:
                    try:
                        name = runtime.get_consts().REGION_ID_TO_NAME[res['id']]
                        lbl = name
                    except:
                        lbl = f"Region {res['id']}"
            else:
                lbl = f"{res['src']}->{res['tgt']}"
                
            color = colors[idx]
            
            # Plot Curve
            ax.plot(dims, rhos, label=lbl, color=color, linewidth=2, marker='o', markersize=4, alpha=0.7)
            
            # Highlight Significant Points
            sig_mask = pvals < 0.05
            if np.any(sig_mask):
                ax.scatter(dims[sig_mask], rhos[sig_mask], 
                           color=color, s=50, edgecolors='k', zorder=5, 
                           label=None) # Don't duplicate label
                           
        ax.set_xlabel("Number of Dimensions (Cumulative)", fontsize=16, fontweight='bold')
        ax.set_ylabel("Spearman Correlation", fontsize=16, fontweight='bold')
        
        # Titles
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.96)
        ax.set_title(subtitle, fontsize=14, color='gray', pad=10)
        
        # Significance Annotation
        ax.text(0.95, 0.95, "Bold points: Permutation p < 0.05", 
                transform=ax.transAxes, ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=14, width=2)
        
        # Legend: Outside, Centered, Large
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                 ncol=3, fontsize=14, frameon=False)
        
        # Save
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Adjust layout to accommodate external elements
        fig.subplots_adjust(top=0.88, bottom=0.20, left=0.15, right=0.95)
        
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        print(f"[Plot] Saved to {output_path}")



def plot_all_overlaps_grid(
    region: int,
    monkeys: tuple[str, ...],
    zscore_codes: tuple[int, ...],
    methods: tuple[str, ...],
    results: list[tuple[str, int, str, np.ndarray, int]], 
    figsize_scale: float = 1.0,
    cell_size: float = 2.7,
    gap_suptitle: float = 0.035,
    save: bool = False,
    save_dir: Path | None = None,
    save_dpi: int = 400,
    show: bool = True
) -> str | None:
    """
    Render a grid of overlap matrices covering multiple monkeys, z-scores, and methods.
    """
    consts = runtime.get_consts()
    region_name = consts.REGION_ID_TO_NAME.get(region, f"Region{region}")
    
    R, Z, M = len(monkeys), len(zscore_codes), len(methods)
    C = Z * M

    cs = float(cell_size) * float(figsize_scale)
    fig_w = cs * C
    head_h_z   = 0.20
    head_h_met = 0.10
    cbar_h     = 0.10
    fig_h = cs * (R + head_h_z + head_h_met + cbar_h)
    fig = plt.figure(figsize=(fig_w, fig_h))

    height_ratios = [head_h_z, head_h_met] + [1] * R + [cbar_h]
    gs = gridspec.GridSpec(
        nrows=len(height_ratios), ncols=C, figure=fig,
        height_ratios=height_ratios,
        hspace=0.15, wspace=0.1
    )

    axes = np.empty((R, C), dtype=object)
    last_im = None
    
    # --- Matrix Plotting Loop ---
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
        ax.set_box_aspect(1)
        for spine in ax.spines.values(): spine.set_visible(False)

    fig.canvas.draw()

    # --- Headers ---
    header_axes_z = []
    for zi, z in enumerate(zscore_codes):
        c0, c1 = zi * M, zi * M + (M - 1)
        ax_z = fig.add_subplot(gs[0, c0:c1+1])
        ax_z.axis("off")
        ax_z.text(0.5, 0.25, f"{consts.ZSCORE_INFO[z][0]}", transform=ax_z.transAxes,
                  ha="center", va="center", fontsize=11, fontweight="bold")
        header_axes_z.append(ax_z)
        for mi, met in enumerate(methods):
            ax_m = fig.add_subplot(gs[1, zi * M + mi])
            ax_m.axis("off")
            ax_m.text(0.5, 0.15, met, transform=ax_m.transAxes,
                      ha="center", va="bottom", fontsize=11)

    # --- Row Labels ---
    first_col_left = axes[0, 0].get_position().x0
    label_x = first_col_left - 0.010
    for r, m in enumerate(monkeys):
        y0, y1 = axes[r, 0].get_position().y0, axes[r, 0].get_position().y1
        cy = (y0 + y1) / 2.0
        fig.text(label_x, cy, m, ha="right", va="center", rotation=90, fontsize=13, fontweight="bold")

    # --- Main Title ---
    top_of_z_headers = max(ax.get_position().y1 for ax in header_axes_z)
    suptitle_y = min(0.99, top_of_z_headers + gap_suptitle)
    fig.suptitle(f"Repetitions Subspace Overlap (D-mean) — {region_name}\n",
                 fontsize=22, y=suptitle_y, fontweight="bold")

    # --- Colorbar ---
    # --- Colorbar ---
    if C >= 6:
        cax = fig.add_subplot(gs[-1, 2:C-2]) # Center it reasonably
    else:
        # Fallback for small grids
        cax = fig.add_subplot(gs[-1, :]) 
        
    cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cbar.set_label("Subspace Overlap", fontsize=12)

    # --- Save ---
    saved_path = None
    if save:
        base_dir = Path(save_dir) if save_dir is not None else (runtime.get_consts().BASE_DIR / "PLOTS_HEAT_MAP")
        base_dir.mkdir(parents=True, exist_ok=True)
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


def plot_repetition_stability(
    results: list[dict],
    out_dir: str | None = None,
    show_errorbars: bool = True,
    title_suffix: str = ""
):
    """
    Plots the Repetition Stability curve (Mean Overlap vs Lag).
    Supports both Single Region (Overlap) and Connection (Overlap + R2) results.
    
    Args:
        results: List of result dicts (from RepetitionStabilityAnalyzer).
        out_dir: Directory to save plots.
        show_errorbars: Whether to show SEM error bars.
        title_suffix: Extra text for title.
    """
    from matplotlib.ticker import MaxNLocator
    from .repetition_stability import RepetitionStabilityAnalyzer # Local import to access extractor
    from scipy.stats import spearmanr

    # Create figure
    fig = plt.figure(figsize=(14, 12), dpi=150)
    
    # Check if we have any connection results (to decide on secondary axis)
    has_connection = any(r["type"] == "connection" for r in results)
    
    ax = fig.add_axes([0.12, 0.32, 0.75, 0.58]) 
    ax2 = ax.twinx() if has_connection else None
    
    # Styles - Distinct colors for Region vs Connection
    colors = {
        # Regions: Blue, Orange, Green (Standard categorical)
        1: "tab:blue", 2: "tab:orange", 3: "tab:green",
        # Connections (src, tgt): Red, Purple, Brown (Distinct from above)
        (1, 2): "tab:red", (1, 3): "tab:purple", (2, 3): "tab:brown"
    }
    region_map = runtime.get_consts().REGION_ID_TO_NAME
    
    all_handles = []
    
    for res in results:
        O = res["matrix"]
        
        # Extract Lag Data using the helper from Analyzer
        lags_raw, overlaps_raw, u_lags, overlap_means, overlap_sems = RepetitionStabilityAnalyzer.extract_lag_data(O)
        
        # Stats Overlap
        rho = res["spearman_rho"]
        p = res["p_value"]
        
        # Stats R2 (Compute on the fly if connection)
        r2_stats_str = ""
        lag_r2s = [] 
        
        if res["type"] == "connection":
             r2s = np.array(res["block_r2s"])
             # Create pairwise R2 matrix: M[i,j] = (R2[i] + R2[j])/2
             # This represents the average performance of the two blocks being compared.
             # If performance degrades over time, "Lag" correlations will pick it up (because late blocks are involved in high lags with early blocks? No).
             # Wait, Lag=1 involves (1,2), (2,3)... (9,10).
             # Lag=9 involves (1,10).
             # If linear decay: avg((1,2)) > avg((1,10))?
             # R2(1)=0.9, R2(10)=0.1.
             # Lag 1: (0.9+0.8)/2 ... -> ~0.5.
             # Lag 9: (0.9+0.1)/2 = 0.5.
             # Actually, if R2 decays linearly, the average over pairs might be flat relative to Lag?
             # Let's think: Lag k pairs are (t, t+k).
             # Avg R2 = Mean_t [ (R2(t) + R2(t+k))/2 ]
             # If R2(t) = 1 - alpha*t.
             # Then R2(t) + R2(t+k) = 2 - alpha(2t+k).
             # Sum over t=1 to N-k: (N-k)*2 - alpha * (Sum of 2t + k*(N-k)).
             # This seems dependent on k.
             # However, simpler: The user just wants to know "Correlation of R2 with Time" or "Correlation of R2 with Lag"?
             # "stats ... in the case of R^2 ... since we see its average in the graph ... what is the correlation".
             # The graph shows R2 vs Lag. So they want Correlation(Lag, R2_at_Lag).
             # Just Spearson(u_lags, lag_r2s).
             
             for l in u_lags:
                vals = []
                for i in range(len(r2s) - l):
                    vals.append((r2s[i] + r2s[i+l]) / 2.0)
                lag_r2s.append(np.mean(vals) if vals else np.nan)
             
             # Compute correlation between (Lag) and (R2 at Lag)
             # Using the raw points would be better, but we only plotted means?
             # Let's match the plot: Correlation of the plotted points?
             # Or better: construct the long vector of (lag, r2_pair) and correlate.
             r2_lags_long = []
             r2_vals_long = []
             for l in u_lags:
                  for i in range(len(r2s) - l):
                      r2_lags_long.append(l)
                      r2_vals_long.append((r2s[i] + r2s[i+l]) / 2.0)
             
             if len(r2_lags_long) > 2:
                 rho_r2, p_r2 = spearmanr(r2_lags_long, r2_vals_long)
                 
                 # P-value Formatting
                 if p_r2 < 0.001:
                    p_str_r2 = f"$p < 10^{{{int(np.log10(p_r2))}}}$" if p_r2 > 0 else "$p < 0.001$"
                 else:
                    p_str_r2 = f"$p={p_r2:.3f}$"
                 r2_stats_str = f" | $R^2$: ρ={rho_r2:.2f}, {p_str_r2}"


        # Label & Color
        if res["type"] == "region":
            rid = res["region_id"]
            nm = region_map[rid]
            color = colors.get(rid, "black")
            label_base = f"{nm}"
        else:
            sid, tid = res["src_id"], res["tgt_id"]
            nm = f"{region_map[sid]} → {region_map[tid]}"
            color = colors.get((sid, tid), "black")
            label_base = nm
            
        # P-value Formatting
        if p < 0.001:
            p_str = f"$p < 10^{{{int(np.log10(p))}}}$" if p > 0 else "$p < 0.001$"
        else:
            p_str = f"$p={p:.3f}$"
            
        legend_label = f"{label_base} (Ovlp: ρ={rho:.2f}, {p_str}{r2_stats_str})"
        
        # Plot Overlap (Primary Axis)
        if show_errorbars:
            h = ax.errorbar(u_lags, overlap_means, yerr=overlap_sems, fmt='-o', color=color, 
                            lw=4, ms=12, capsize=8, elinewidth=3, zorder=5, label=legend_label)
        else:
            h, = ax.plot(u_lags, overlap_means, '-o', color=color, lw=4, ms=12, zorder=5, label=legend_label)
        
        all_handles.append(h)
        
        # Plot Performance (Secondary Axis) - Only for Connections
        if res["type"] == "connection" and ax2 is not None:
            ax2.plot(u_lags, lag_r2s, '--^', color=color, alpha=0.35, lw=2.5, ms=9, zorder=3)

    # Aesthetics - DRASTICALLY INCREASED FONTS
    ax.set_xlabel("Repetition lag (Δblock)", fontsize=28, fontweight="bold", labelpad=20)
    ax.set_ylabel("Subspace Overlap", fontsize=28, fontweight="bold", labelpad=20)
    
    if ax2:
        ax2.set_ylabel("Predictive Performance (Mean $R^2$)", fontsize=28, fontweight="bold", labelpad=25, color='dimgray')
        ax2.tick_params(axis='y', labelsize=24, colors='dimgray', width=3, length=10)
        ax2.spines['right'].set_linewidth(4)
        ax2.spines['right'].set_color('dimgray')
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', labelsize=24, width=3, length=10)
    
    # Spines
    ax.spines[['top']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(4)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Title & Metadata
    if results:
        r0 = results[0]
        monkey = r0["monkey"]
        z_code = r0["z_code"]
        at = r0["method"].upper() if hasattr(r0["method"], "upper") else str(r0["method"]).upper()
        bs = r0["block_size"]
        
        info_text = f"{monkey} | {at} | Z-{z_code} | BlockSize={bs}"
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=20, fontweight="normal",
                va='top', ha='right', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        
        fig.suptitle(f"Repetition Stability Analysis {title_suffix}", fontsize=30, fontweight="bold", y=0.96)

    # Legend - HUGE SIZE, 2 Columns
    ax.legend(handles=all_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=2, frameon=False, fontsize=20, columnspacing=1.5, handletextpad=0.5) 

    # Save
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        # Construct filename from first result metadata
        r0 = results[0]
        ftype = "Analysis" # Generic since we often plot both
        # Using string conversion just in case
        mk = str(r0['monkey']).replace(' ','')
        md = str(r0['method'])
        bs = str(r0['block_size'])
        fname = f"{mk}_RepetitionStability_{md}_blk{bs}.png"
        
        fig.savefig(p / fname, dpi=300, bbox_inches="tight")
        print(f"[✓] Stability Plot Saved → {p / fname}")
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# General Plots (Migrated from Generalplots.py)
# =============================================================================

class GeneralPlots:
    """
    Namespace for general exploratory plots.
    Migrated from Generalplots.py.
    Focuses on Repetition-wise timecourses, amplitude distributions, and global dataset statistics.
    """
    
    @staticmethod
    def get_title(region: str) -> str:
        """Standardized plot title with monkey and z-score info."""
        return f"{region} • {runtime.get_cfg().get_monkey_name()}  •  {runtime.get_cfg().get_zscore_title()}"

    @staticmethod
    def plot_mean_amplitude_by_repetition():
        """
        Plot 1: Average amplitude across all stimuli per repetition.
        Plot 2: Same, but broken down by Region (V1/V4/IT).
        Color-coded by 'Day ID' to show temporal drift.
        """
        data = runtime.get_cfg()._load_trials()
        rois = runtime.get_cfg().get_rois()
        region_names = runtime.get_consts().REGION_ID_TO_NAME

        region_channels = {
            region_names[1]: np.where(rois == 1)[0],
            region_names[2]: np.where(rois == 2)[0],
            region_names[3]: np.where(rois == 3)[0],
        }

        # --- Aggregation ---
        reps_table = defaultdict(list)
        rep_to_days = defaultdict(list)
        rep_to_amps = {reg: defaultdict(list) for reg in region_channels}

        for trial in data:
            rep = int(trial["allmat_row"][3]) - 1
            mua = trial["mua"]
            mean_a = mua.mean()
            
            reps_table[rep].append(mean_a)
            rep_to_days[rep].append(trial["day_id"])
            
            for reg, ch_idx in region_channels.items():
                rep_to_amps[reg][rep].append(mua[:, ch_idx].mean())

        rep_indices = sorted(reps_table.keys())
        avg_amplitudes = [np.mean(reps_table[rep]) for rep in rep_indices]

        # Determine majority day per repetition for coloring
        rep_main_day = {
            rep: int(Counter(days).most_common(1)[0][0])
            for rep, days in rep_to_days.items()
        }

        day_labels = [rep_main_day[r] for r in rep_indices]
        all_days = sorted(set(day_labels))
        day_to_idx = {d: i for i, d in enumerate(all_days)}
        color_indices = [day_to_idx[d] for d in day_labels]
        
        cmap = plt.get_cmap("tab10", len(all_days))
        norm = mcolors.BoundaryNorm(boundaries=np.arange(0, len(all_days)+1), ncolors=len(all_days))

        mean_by_region = {
            reg: [np.mean(rep_to_amps[reg][r]) for r in rep_indices]
            for reg in region_channels
        }
        region_colors = {reg: color for reg, color in zip(region_channels.keys(), ["tab:red", "tab:blue", "tab:green"])}

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
        ylabel = "Mean Z-scored MUA" if runtime.get_cfg().get_zscore_code() != 1 else "Mean Raw MUA"

        # Graph 1: Global Mean
        ax1.plot(rep_indices, avg_amplitudes, color="gray", linewidth=1.5, zorder=1)
        for x, y, ci in zip(rep_indices, avg_amplitudes, color_indices):
            ax1.scatter(x, y, color=cmap(ci), edgecolors='k', linewidths=0.4, s=60, zorder=2)
        ax1.set_ylabel(ylabel, fontsize=10, labelpad=10)
        ax1.set_title("Mean Response Amplitude per Repetition (All Regions)")
        ax1.grid(True, alpha=0.3)

        # Graph 2: Per Region
        for reg, y_vals in mean_by_region.items():
            ax2.plot(rep_indices, y_vals, label=reg, color=region_colors[reg], linewidth=2, zorder=1)
            for x, y, ci in zip(rep_indices, y_vals, color_indices):
                ax2.scatter(x, y, color=cmap(ci), edgecolors='k', linewidths=0.4, s=60, zorder=2)

        ax2.set_xlabel("Repetition Index (0–29)")
        ax2.set_ylabel(ylabel, fontsize=10, labelpad=10)
        ax2.set_title("Mean Response per Repetition (By Region)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(title="Region")

        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(len(all_days)) + 0.5)
        cbar.set_ticklabels([f"Day {d}" for d in all_days])
        cbar.set_label("Majority Recording Day")

        fig.suptitle(GeneralPlots.get_title("Mean All Regions"), fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.9, 0.94])

        out_path = runtime.get_cfg().get_plot_dir() / "mean_response_by_repetition.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[📈] Saved plot to: {out_path.name}")


    @staticmethod
    def plot_repeatwise_timecourses(region: str = "V1"):
        """Plot Mean and STD timecourses for each repetition (0..29), colored Red->Blue."""
        data = runtime.get_cfg()._load_trials()
        rois = runtime.get_cfg().get_rois()

        region_ids = {"V1": 1, "V4": 2, "IT": 3}
        region_id = region_ids[region.upper()]
        channels = np.where(rois == region_id)[0]
        NUM_REPS = 30

        # Group trials
        rep_to_trials = defaultdict(list)
        for trial in data:
            rep = int(trial["allmat_row"][3]) - 1
            rep_to_trials[rep].append(trial)

        # Compute curves
        rep_mean_tc, rep_std_tc = [], []
        for r in range(NUM_REPS):
            trials = rep_to_trials[r]
            if not trials:
                continue
            mua_stack = np.stack([t["mua"][:, channels] for t in trials], axis=0) # (Trials, Time, Chans)
            rep_mean_tc.append(mua_stack.mean(axis=(0, 2))) # Mean over Trials & Channels
            rep_std_tc.append(mua_stack.std(axis=(0, 2)))

        # Colors
        cmap = LinearSegmentedColormap.from_list("RedToBlue", ["red", "blue"])
        norm = Normalize(vmin=0, vmax=NUM_REPS - 1)
        colors = [cmap(norm(r)) for r in range(NUM_REPS)]

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for r in range(len(rep_mean_tc)):
            ax1.plot(rep_mean_tc[r], color=colors[r], linewidth=1)
            ax2.plot(rep_std_tc[r], color=colors[r], linewidth=1)

        ax1.set_title(f"{region} — Mean Z-Score Over Time per Repeat", fontsize=20)
        ax1.set_ylabel("Mean Z-Score", fontsize=20, labelpad=10)
        ax1.grid(True)
        ax1.tick_params(labelsize=14)

        ax2.set_title(f"{region} — STD of Z-Score Over Time per Repeat", fontsize=18)
        ax2.set_xlabel("Time (ms)", fontsize=18)
        ax2.set_ylabel("STD", fontsize=18, labelpad=10)
        ax2.grid(True)
        ax2.tick_params(labelsize=14)

        # Colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0, NUM_REPS - 1])
        cbar.set_label("Repetition Index (Red=Early, Blue=Late)", fontweight="bold", fontsize=20)
        cbar.ax.tick_params(labelsize=14)

        fig.suptitle(GeneralPlots.get_title(region), fontsize=22, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.9, 0.94])

        out_path = runtime.get_cfg().get_plot_dir() / f"{region}_repeatwise_timecourses.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[📈] Saved plot to: {out_path.name}")


    @staticmethod
    def plot_repeatwise_timecourses_all_regions():
        """Run plot_repeatwise_timecourses for V1, V4, and IT."""
        for region in ["V1", "V4", "IT"]:
            print(f"\n[🧠] Plotting repeatwise timecourses for region: {region}")
            GeneralPlots.plot_repeatwise_timecourses(region)


    @staticmethod
    def plot_mean_std_amplitude_by_repetition():
        """Plot Mean ± STD amplitude per repetition using Error Bars."""
        data = runtime.get_cfg()._load_trials()
        rois_logical = runtime.get_cfg().get_rois()
        region_names = runtime.get_consts().REGION_ID_TO_NAME
        
        region_channels = {
            region_names[1]: np.where(rois_logical == 1)[0],
            region_names[2]: np.where(rois_logical == 2)[0],
            region_names[3]: np.where(rois_logical == 3)[0],
        }

        # Build Stats
        records = []
        rep_to_trials = defaultdict(list)
        for trial in data:
            rep = int(trial["allmat_row"][3]) - 1
            rep_to_trials[rep].append(trial)

        for rep in sorted(rep_to_trials.keys()):
            trials = rep_to_trials[rep]
            for region, ch_idx in region_channels.items():
                if len(ch_idx) == 0: continue
                # Stack: (Trials, Time, Channels)
                stack = np.stack([t["mua"][:, ch_idx] for t in trials])
                # Global mean per trial-channel (or mean over time first)
                # Original logic: mua[:, ch_idx].mean(axis=0) -> mean per channel over time
                # Then we collect those.
                # Let's replicate original logic:
                all_amps = []
                for t in trials:
                    # Mean over time for each channel
                    chan_means = t["mua"][:, ch_idx].mean(axis=0)
                    all_amps.extend(chan_means)
                
                records.append({
                    "Repetition": rep,
                    "Region": region,
                    "MeanAmplitude": np.mean(all_amps),
                    "StdAmplitude": np.std(all_amps)
                })

        df = pd.DataFrame.from_records(records)

        # Plot
        plt.figure(figsize=(12, 6))
        for region in df["Region"].unique():
            sub = df[df["Region"] == region].sort_values("Repetition")
            plt.errorbar(
                sub["Repetition"], sub["MeanAmplitude"], yerr=sub["StdAmplitude"],
                label=region, marker='o', capsize=4
            )

        plt.title("Mean ± Std of Amplitude per Repetition")
        plt.xlabel("Repetition Index")
        plt.ylabel("Mean Amplitude (Z-scored)")
        plt.grid(True)
        plt.legend(title="Region")
        plt.suptitle(GeneralPlots.get_title("Mean All Regions"), fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        out_path = runtime.get_cfg().get_plot_dir() / "mean_std_amplitude_by_repetition.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[📈] Saved errorbar plot to: {out_path.name}")


    @staticmethod
    def plot_global_electrode_and_trial_distribution():
        """
        Global overview: Electrode count per region per monkey + Trial count per day.
        """
        print("[🌍] Building global electrode & trial distribution plot...")
        consts = runtime.get_consts()
        base = consts.BASE_DIR
        region_map = consts.REGION_ID_TO_NAME
        monkeys = consts.MONKEYS

        electrode_recs = []
        trial_by_monkey = defaultdict(dict)
        summary = {}

        for monkey in monkeys:
            # Note: switching configs modifies global state; intentional for this global plot
            runtime.set_cfg(monkey, 1) 
            cfg = runtime.get_cfg()

            # Electrodes
            cnts = Counter(cfg.get_rois())
            summary[monkey] = {region_map[rid]: cnts.get(rid, 0) for rid in region_map}
            for rid, rname in region_map.items():
                electrode_recs.append((monkey, rname, cnts.get(rid, 0)))

            # Trials
            trials = cfg._load_trials()
            day_key = "day_id" if "day_id" in trials[0] else "dayID"
            for d, c in Counter(t[day_key] for t in trials).items():
                trial_by_monkey[monkey][d] = c
            summary[monkey]["trials"] = sum(trial_by_monkey[monkey].values())

        print("\n[📊] Summary:")
        for m, info in summary.items():
            print(f"  {m}: {info}")

        # Plotting
        max_day = max((max(d.keys()) for d in trial_by_monkey.values() if d), default=0)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
        fig.suptitle("Global Electrode & Trial Distribution", fontsize=16, fontweight="bold")

        # 1. Electrodes
        x1 = np.arange(len(electrode_recs))
        vals1 = [x[2] for x in electrode_recs]
        cols1 = ["steelblue" if x[0] == "Monkey F" else "orange" for x in electrode_recs]
        
        ax1.bar(x1, vals1, color=cols1, edgecolor="black")
        ax1.set_xticks(x1)
        ax1.set_xticklabels([f"{x[0][7]}-{x[1]}" for x in electrode_recs]) # 'Monkey F' -> 'F'
        ax1.set_ylabel("Electrode Count")
        ax1.set_title("Electrodes per Region per Monkey")
        ax1.legend(handles=[Patch(color="steelblue", label="Monkey F"), Patch(color="orange", label="Monkey N")])
        for i, v in zip(x1, vals1): ax1.text(i, v+5, str(v), ha="center", va="bottom", fontsize=9)

        # 2. Trials per Day
        days = np.arange(1, max_day + 1)
        width = 0.35
        f_vals = [trial_by_monkey[consts.MONKEY_F].get(d, 0) for d in days]
        n_vals = [trial_by_monkey[consts.MONKEY_N].get(d, 0) for d in days]

        ax2.bar(days - width/2, f_vals, width, color="steelblue", label=consts.MONKEY_F)
        ax2.bar(days + width/2, n_vals, width, color="orange", label=consts.MONKEY_N)
        ax2.set_xlabel("Day ID")
        ax2.set_ylabel("Trial Count")
        ax2.legend()
        for d, v in zip(days, f_vals): ax2.text(d-width/2, v+5, str(v) if v>0 else "", ha="center", va="bottom", fontsize=8)
        for d, v in zip(days, n_vals): ax2.text(d+width/2, v+5, str(v) if v>0 else "", ha="center", va="bottom", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = base / "PLOTS" / "global_electrode_and_trial_distribution.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=300)
        plt.close()
        print(f"[✅] Saved global plot -> {out}")


    @staticmethod
    def plot_repeatwise_mean_timecourses_all_regions():
        """
        Plot mean Z-score timecourse per repetition for all regions (stacked).
        """
        data = runtime.get_cfg()._load_trials()
        rois = runtime.get_cfg().get_rois()
        region_ids = {"V1": 1, "V4": 2, "IT": 3}
        NUM_REPS = 30

        cmap = LinearSegmentedColormap.from_list("RedToBlue", ["red", "blue"])
        norm = Normalize(vmin=0, vmax=NUM_REPS - 1)
        colors = [cmap(norm(r)) for r in range(NUM_REPS)]
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        rep_to_trials = defaultdict(list)
        for trial in data:
            rep = int(trial["allmat_row"][3]) - 1
            rep_to_trials[rep].append(trial)

        for idx, region in enumerate(["V1", "V4", "IT"]):
            rid = region_ids[region]
            chans = np.where(rois == rid)[0]
            ax = axes[idx]
            
            for r in range(NUM_REPS):
                trials = rep_to_trials[r]
                if not trials: continue
                # Mean over trials and channels -> (Time,)
                tc = np.stack([t["mua"][:, chans] for t in trials]).mean(axis=(0, 2))
                ax.plot(tc, color=colors[r], linewidth=1)
            
            ax.set_title(region, fontsize=20, fontweight="bold")
            ax.grid(True)

        cbar_ax = fig.add_axes([0.92, 0.10, 0.02, 0.76])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0, NUM_REPS - 1])
        cbar.set_label("Repetition Index (Red=Early, Blue=Late)", fontsize=18)

        fig.suptitle("Mean Z-Score Timecourses per Region", fontsize=22, fontweight="bold")
        fig.supylabel("Mean Z-Score", fontsize=22)
        fig.supxlabel("Time (ms)", fontsize=22)
        fig.tight_layout(rect=[0.05, 0.05, 0.90, 0.92])

        out_path = runtime.get_cfg().get_plot_dir() / "AllRegions_repeatwise_mean_timecourses.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[📈] Saved plot to: {out_path.name}")


    @staticmethod
    def generate_all_plots_for_all_combinations():
        """Run standard plotting suite for all monkeys and z-scores."""
        monkeys = runtime.get_consts().MONKEYS
        zscore_codes = [1, 2, 3, 4]

        for monkey in monkeys:
            for z in zscore_codes:
                print(f"\n--- {monkey} | Z={z} ---")
                runtime.set_cfg(monkey, z)
                GeneralPlots.plot_mean_amplitude_by_repetition()
                GeneralPlots.plot_repeatwise_timecourses_all_regions()
                GeneralPlots.plot_mean_std_amplitude_by_repetition()

