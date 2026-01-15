from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as pe
from pathlib import Path
from core.runtime import runtime
from .utils import jitter, square_limits, labeled_dot, smart_label

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
        """Standard 4-panel Figure 4."""
        fig = plt.figure(figsize=(14, 13), dpi=400)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12,
                            wspace=0.15, hspace=0.30)

        gs  = gridspec.GridSpec(2, 2)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[1, 0])
        axC = fig.add_subplot(gs[0, 1])
        axD = fig.add_subplot(gs[1, 1])

        dims   = np.arange(1, d_max + 1)
        tgt    = runtime.consts.REGION_ID_TO_NAME[target_region]
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
            f"{runtime.cfg.get_monkey_name()}  |  {runtime.cfg.get_zscore_title()}  |  {analysis_type.upper()}",
            fontsize=20, y=min(0.998, top_row_ymax + 0.080), fontweight="bold"
        )

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
        """Plot the Multi-Run Subset Figure 4."""
        cfg = runtime.cfg
        tgt_nm = runtime.consts.REGION_ID_TO_NAME[target_region]

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

        # A: FULL
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

        # B: MATCH
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

        # C: Summary
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

        # D: Subsets
        d95_match_vals = [v for (_, v) in d95_match_sub_all]
        d95_full_vals  = [v for (_, v) in d95_full_sub_all]
        if len(d95_full_vals) and len(d95_match_vals):
            xmin_d, xmax_d = square_limits(d95_match_vals, d95_full_vals, base_min=1, scale=1.5)
        else:
            xmin_d, xmax_d = (1, max(2, int(np.ceil(1.5 * d_max))))
        
        axD.plot([xmin_d, xmax_d], [xmin_d, xmax_d], ls="--", lw=0.9, color="k")

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

        # Suptitle
        boxA, boxC = axA.get_position(), axC.get_position()
        top_row_ymax = max(boxA.y1, boxC.y1)
        fig.suptitle(
            f"{cfg.get_monkey_name()}  |  {cfg.get_zscore_title()}  |  "
            f"{analysis_type.upper()}  (n_src={n_src_eff}, n_tgt={n_tgt_eff}, runs={n_runs})",
            fontsize=20, y=min(0.998, top_row_ymax + 0.080), fontweight="bold"
        )

        # Group Labels
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

        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=400, facecolor="white")
            print(f"[✓] Figure 4 Subset saved → {out_path}")
        
        plt.close(fig)


    @staticmethod
    def plot_figure_5_b(csv_path: str, out_path: str | None = None, title: str | None = None):
        """Generate Semedo Figure 5B (Dimensionality Matching) by reading data from a CSV file."""
        import csv

        styles = {
            "V4":             {"color": "#ca1b1b"},
            "Target V4":      {"color": "#cf2359"},
            "IT":             {"color": "#0529ae"},
            "Target IT":      {"color": "#3391e3"},
        }
        
        subsets = {k: {"xs": [], "ys": []} for k in styles}
        full_points = {k: {"xs": [], "ys": []} for k in styles} # Usually 1 per group
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                grp = row["group"]
                x = float(row["x_dimensionality_d95"])
                y = float(row["y_predictive_d95"])
                pt_type = row.get("point_type", "subset")
                
                if grp not in styles: continue
                    
                if pt_type == "full":
                    full_points[grp]["xs"].append(x)
                    full_points[grp]["ys"].append(y)
                else:
                    subsets[grp]["xs"].append(x)
                    subsets[grp]["ys"].append(y)

        # Plotting
        fig, ax = plt.subplots(figsize=(7, 6.2), dpi=140)
        rng = np.random.default_rng(0)

        for name, style in styles.items():
            # Scatters
            xs = np.array(subsets[name]["xs"])
            ys = np.array(subsets[name]["ys"])
            color = style["color"]
            
            if len(xs) > 0:
                xs_j = jitter(xs, rng, scale=0.2)
                ys_j = jitter(ys, rng, scale=0.2)
                ax.scatter(xs_j, ys_j, s=50, alpha=0.9, color=color, label=name if "Target" not in name else "")
                ax.scatter([xs.mean()], [ys.mean()], s=40, edgecolor='k', linewidths=0.8, color=color)

            # Full Points
            fxs = np.array(full_points[name]["xs"])
            fys = np.array(full_points[name]["ys"])
            if len(fxs) > 0:
                 ax.scatter(fxs, fys, s=90, color=color, marker="D", edgecolor="black", linewidth=0.8, alpha=1.0, zorder=20)

        # y = x line
        all_vals = []
        for d in (subsets, full_points):
            for g in d.values():
                all_vals.extend(g["xs"]); all_vals.extend(g["ys"])
        lim_max = max(all_vals) * 1.05 if all_vals else 50
        ax.plot([0, lim_max], [0, lim_max], 'k--', lw=1)
        
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel("Target Population dimensionality", fontsize=20, fontweight="bold")
        ax.set_ylabel("Number of Predictive dimensions", fontsize=19, fontweight="bold")
        ax.tick_params(axis='both', which='major', labelsize=18, width=1.8, length=6)

        for tick in ax.get_xticklabels(): tick.set_fontweight('bold')
        for tick in ax.get_yticklabels(): tick.set_fontweight('bold')
        
        if title: ax.set_title(title, fontsize=14, fontweight="bold")

        dummy_groups_for_label = {k: v for k, v in subsets.items()}
        for name, style in styles.items():
            if len(subsets[name]["xs"]) > 0:
                smart_label(ax, name, subsets[name]["xs"], subsets[name]["ys"], style["color"], dummy_groups_for_label)

        fig.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight")
            print(f"[✓] Re-plotted Figure 5B from CSV → {out_path}")
        else:
            plt.show()
        plt.close(fig)
