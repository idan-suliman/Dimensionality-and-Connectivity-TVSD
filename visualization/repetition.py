from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr
from pathlib import Path
from core.runtime import runtime
from methods.repetition_stability.utils import extract_lag_data

def plot_repetition_stability(
    results: list[dict],
    out_dir: str | None = None,
    show_errorbars: bool = True,
    title_suffix: str = ""
):
    """
    Plots the Repetition Stability curve (Mean Overlap vs Lag).
    """
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
    region_map = runtime.consts.REGION_ID_TO_NAME
    
    all_handles = []
    
    for res in results:
        O = res["matrix"]
        
        # Extract Lag Data 
        lags_raw, overlaps_raw, u_lags, overlap_means, overlap_sems = extract_lag_data(O)
        
        # Stats Overlap
        rho = res["spearman_rho"]
        p = res["p_value"]
        
        # Stats R2 (Compute on the fly if connection)
        r2_stats_str = ""
        lag_r2s = [] 
        
        if res["type"] == "connection":
             r2s = np.array(res["block_r2s"])
             # Pairwise R2 matrix averaging
             for l in u_lags:
                vals = []
                for i in range(len(r2s) - l):
                    vals.append((r2s[i] + r2s[i+l]) / 2.0)
                lag_r2s.append(np.mean(vals) if vals else np.nan)
             
             # Compute correlation between (Lag) and (R2 at Lag)
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
            rid = int(res["region_id"])
            nm = region_map[rid]
            color = colors.get(rid, "black")
            label_base = f"{nm}"
        else:
            sid, tid = int(res["src_id"]), int(res["tgt_id"])
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
    ax.set_xlabel("Repetition lag (Δblock)", fontsize=38, labelpad=10)
    ax.set_ylabel("Subspace Overlap", fontsize=40, labelpad=20)
    ax.set_ylim(bottom=0, top=0.8)
    
    if ax2:
        ax2.set_ylabel("Predictive Performance (Mean $R^2$)", fontsize=30, labelpad=25, color='dimgray')
        ax2.tick_params(axis='y', labelsize=28, colors='dimgray', width=3, length=10)
        ax2.spines['right'].set_linewidth(4)
        ax2.spines['right'].set_color('dimgray')
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', labelsize=28, width=3, length=10)
    
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
        info_text = f"Block Size={bs}"
        # info_text = f"{monkey} | {at} | Z-{z_code} | BlockSize={bs}"
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=28, fontweight="normal",
                va='top', ha='right', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        
        # fig.suptitle(f"Repetition Stability Analysis {title_suffix}", fontsize=30, fontweight="bold", y=0.96)

    # Legend - HUGE SIZE, 2 Columns
    ax.legend(handles=all_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=2, frameon=False, fontsize=20, columnspacing=1.5, handletextpad=0.5) 

    # Save
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        # Construct filename from first result metadata
        r0 = results[0]
        # Construct filename using helper
        fname = runtime.paths.get_rep_stability_path(
            output_dir="",
            monkey_name=r0["monkey"],
            analysis_type=r0["method"], 
            group_size=r0["block_size"],
            region_id=int(r0["region_id"]) if r0["type"] == "region" else None,
            src_tgt=(int(r0["src_id"]), int(r0["tgt_id"])) if r0["type"] == "connection" else None,
            extension=".png"
        ).name 

        full_path = runtime.paths.get_rep_stability_path(
            output_dir=out_dir,
            monkey_name=r0["monkey"],
            analysis_type=r0["method"],
            group_size=r0["block_size"],
            region_id=int(r0["region_id"]) if r0["type"] == "region" else None,
            src_tgt=(int(r0["src_id"]), int(r0["tgt_id"])) if r0["type"] == "connection" else None,
            extension=".png"
        )
        
        fig.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"[✓] Stability Plot Saved → {full_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_overlap_matrix(overlap_matrix: np.ndarray, D_common: int, method: str) -> None:
    """
    Plot heatmap of subspace overlaps (Kyle's method).
    """
    cfg = runtime.cfg
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
    consts = runtime.consts
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
    if C >= 6:
        cax = fig.add_subplot(gs[-1, 2:C-2]) # Center it reasonably
    else:
        cax = fig.add_subplot(gs[-1, :]) 
        
    cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cbar.set_label("Subspace Overlap", fontsize=12)

    # --- Save ---
    saved_path = None
    if save:
        base_dir = Path(save_dir) if save_dir is not None else (runtime.consts.BASE_DIR / "PLOTS_HEAT_MAP")
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
