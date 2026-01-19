from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr
from pathlib import Path
from core.runtime import runtime
from methods.repetition_stability.utils import extract_lag_data
from visualization.utils import jitter

def plot_repetition_stability_original(
    results: list[dict],
    out_dir: str | None = None,
    show_errorbars: bool = True,
    title_suffix: str = "",
    show_permutation: bool = False
):
    """
    Plots the Repetition Stability curve (Mean Overlap vs Lag).
    ORIGINAL VERSION: Includes all lags.
    """
    # Create figure
    fig = plt.figure(figsize=(14, 12), dpi=150)
    
    # Check if we have any connection results (to decide on secondary axis)
    has_connection = any(r["type"] == "connection" for r in results)
    
    ax = fig.add_axes([0.12, 0.32, 0.75, 0.58]) 
    ax2 = ax.twinx() if has_connection else None
    
    # Styles - Distinct colors for Region vs Connection
    colors = {
        # Regions: Blue, Orange, RED (Changed to avoid conflict)
        1: "tab:blue", 2: "tab:orange", 3: "tab:red",
        # Connections (src, tgt): GREEN (V1->V4), Purple, Brown
        (1, 2): "tab:green", (1, 3): "tab:purple", (2, 3): "tab:brown"
    }
    region_map = runtime.consts.REGION_ID_TO_NAME
    
    all_handles = []
    
    # RNG for jitter
    rng = np.random.default_rng(42)
    
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
        # Scatter Raw Overlaps directly (color with alpha -> "gray-color")
        # Add Jitter to x-values
        lags_jittered = jitter(lags_raw, rng, scale=0.15)
        ax.scatter(lags_jittered, overlaps_raw, color=color, alpha=0.4, s=55, zorder=2, edgecolors='none')

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
        
        # Determine unique regions and connected present
        r_ids = sorted(list(set(int(r["region_id"]) for r in results if r["type"] == "region")))
        c_ids = sorted(list(set((int(r["src_id"]), int(r["tgt_id"])) for r in results if r["type"] == "connection")))
        
        # Logic: If single item, use standard naming. If multiple, create composite name via suffix.
        if len(results) == 1:
            region_arg = int(r0["region_id"]) if r0["type"] == "region" else None
            conn_arg = (int(r0["src_id"]), int(r0["tgt_id"])) if r0["type"] == "connection" else None
            suffix_arg = ""
        else:
            region_arg = None
            conn_arg = None
            
            parts = []
            if r_ids:
                parts.append("Reg" + "".join(str(i) for i in r_ids))
            if c_ids:
                parts.append("Conn" + "".join(f"{s}{t}" for s, t in c_ids))
            
            suffix_arg = "_" + "_".join(parts)
            
        full_path = runtime.paths.get_rep_stability_path(
            output_dir=out_dir,
            monkey_name=r0["monkey"],
            analysis_type=r0["method"],
            group_size=r0["block_size"],
            region_id=region_arg,
            src_tgt=conn_arg,
            suffix=suffix_arg,
            extension=".png"
        )
        
        fig.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"[✓] Stability Plot Saved → {full_path}")
        plt.close(fig)
        
        # Optional: Plot Permutation Test
        if show_permutation:
             plot_permutation_test(results, out_dir, suffix_arg)
             
    else:
        plt.show()

def plot_repetition_stability_temp(
    results: list[dict],
    out_dir: str | None = None,
    show_errorbars: bool = True,
    title_suffix: str = "",
    show_permutation: bool = False
):
    """
    Plots the Repetition Stability curve (Mean Overlap vs Lag).
    FILTERED VERSION: Ignores lags 7, 8, 9.
    """
    # Create figure
    fig = plt.figure(figsize=(14, 12), dpi=150)
    
    # Check if we have any connection results (to decide on secondary axis)
    has_connection = any(r["type"] == "connection" for r in results)
    
    ax = fig.add_axes([0.12, 0.32, 0.75, 0.58]) 
    ax2 = ax.twinx() if has_connection else None
    
    # Styles - Distinct colors for Region vs Connection
    colors = {
        # Regions: Blue, Orange, RED (Changed to avoid conflict)
        1: "tab:blue", 2: "tab:orange", 3: "tab:red",
        # Connections (src, tgt): GREEN (V1->V4), Purple, Brown
        (1, 2): "tab:green", (1, 3): "tab:purple", (2, 3): "tab:brown"
    }
    region_map = runtime.consts.REGION_ID_TO_NAME
    
    all_handles = []
    
    # RNG for jitter
    rng = np.random.default_rng(42)
    
    for res in results:
        O = res["matrix"]
        
        # Extract Lag Data 
        lags_raw, overlaps_raw, u_lags, overlap_means, overlap_sems = extract_lag_data(O)
        
        # --- TEMPORARY: IGNORE LAGS 7, 8, 9 ---
        ignored_lags = [7, 8, 9]
        
        # Filter raw data
        mask_raw = ~np.isin(lags_raw, ignored_lags)
        lags_raw = lags_raw[mask_raw]
        overlaps_raw = overlaps_raw[mask_raw]
        
        # Filter aggregated data
        mask_u = ~np.isin(u_lags, ignored_lags)
        u_lags = u_lags[mask_u]
        overlap_means = overlap_means[mask_u]
        overlap_sems = overlap_sems[mask_u]
        
        # Recalculate Stats (Spearman) on filtered data
        # Note: This invalidates the pre-computed permutation test results if they included these lags.
        if len(lags_raw) > 2:
            rho, p = spearmanr(lags_raw, overlaps_raw)
        else:
            rho, p = np.nan, np.nan
            
        # UPDATE RESULTS DICT for Permutation Plot (so it shows the filtered observed val)
        res["spearman_rho"] = rho
        res["p_value"] = p
        # -------------------------------------
        
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
                 
                 # UPDATE RESULTS DICT for Permutation Plot
                 res["r2_rho"] = rho_r2
                 res["r2_p_val"] = p_r2


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
        # Scatter Raw Overlaps directly (color with alpha -> "gray-color")
        # Add Jitter to x-values
        lags_jittered = jitter(lags_raw, rng, scale=0.15)
        ax.scatter(lags_jittered, overlaps_raw, color=color, alpha=0.4, s=55, zorder=2, edgecolors='none')

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
        
        # Determine unique regions and connected present
        r_ids = sorted(list(set(int(r["region_id"]) for r in results if r["type"] == "region")))
        c_ids = sorted(list(set((int(r["src_id"]), int(r["tgt_id"])) for r in results if r["type"] == "connection")))
        
        # Logic: If single item, use standard naming. If multiple, create composite name via suffix.
        if len(results) == 1:
            region_arg = int(r0["region_id"]) if r0["type"] == "region" else None
            conn_arg = (int(r0["src_id"]), int(r0["tgt_id"])) if r0["type"] == "connection" else None
            suffix_arg = ""
        else:
            region_arg = None
            conn_arg = None
            
            parts = []
            if r_ids:
                parts.append("Reg" + "".join(str(i) for i in r_ids))
            if c_ids:
                parts.append("Conn" + "".join(f"{s}{t}" for s, t in c_ids))
            
            suffix_arg = "_" + "_".join(parts)
            
        full_path = runtime.paths.get_rep_stability_path(
            output_dir=out_dir,
            monkey_name=r0["monkey"],
            analysis_type=r0["method"],
            group_size=r0["block_size"],
            region_id=region_arg,
            src_tgt=conn_arg,
            suffix=suffix_arg,
            extension=".png"
        )
        
        fig.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"[✓] Stability Plot Saved → {full_path}")
        plt.close(fig)
        
        # Optional: Plot Permutation Test
        if show_permutation:
             plot_permutation_test(results, out_dir, suffix_arg)
             
    else:
        plt.show()

# --- ROUTING LOGIC ---
# Use the FILTERED version for now (Temp)
plot_repetition_stability = plot_repetition_stability_original

def plot_permutation_test(results: list[dict], out_dir: str, suffix: str = ""):
    """
    Plots histogram of permutation test results for each analyzed item.
    """
    import seaborn as sns
    from matplotlib import gridspec
    
    # Collect all things to plot
    # Each item: (title, color, obs, pval, dist, label_prefix)
    plot_items = []
    
    for res in results:
        # 1. Overlap (All types)
        if "perm_rhos" in res:
            if res["type"] == "region":
                nm = runtime.consts.REGION_ID_TO_NAME[int(res["region_id"])]
                title = f"{nm} (Region) - Overlap"
                color = "tab:blue"
            else:
                s = runtime.consts.REGION_ID_TO_NAME[int(res["src_id"])]
                t = runtime.consts.REGION_ID_TO_NAME[int(res["tgt_id"])]
                title = f"{s} → {t} - Overlap"
                color = "tab:purple"
            
            plot_items.append({
                "title": title,
                "color": color,
                "dist": res["perm_rhos"],
                "obs": res["spearman_rho"],
                "pval": res["p_value"],
                "xlabel": "Spearman Correlation (ρ)"
            })
            
        # 2. R2 (Connection only)
        if res["type"] == "connection" and "perm_r2s" in res and len(res["perm_r2s"]) > 0:
             s = runtime.consts.REGION_ID_TO_NAME[int(res["src_id"])]
             t = runtime.consts.REGION_ID_TO_NAME[int(res["tgt_id"])]
             title = f"{s} → {t} - $R^2$ Stability"
             color = "tab:brown" # distinct color
             
             plot_items.append({
                "title": title,
                "color": color,
                "dist": res["perm_r2s"],
                "obs": res["r2_rho"],
                "pval": res["r2_p_val"],
                "xlabel": "Spearman Correlation (Lag vs $R^2$)"
            })

    n_plots = len(plot_items)
    if n_plots == 0:
        return

    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    fig = plt.figure(figsize=(5 * cols, 4 * rows), dpi=150)
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)
    
    for i, item in enumerate(plot_items):
        ax = fig.add_subplot(gs[i])
        
        perm_rhos = item["dist"]
        observed = item["obs"]
        p_val = item["pval"]
        
        # Histogram
        sns.histplot(perm_rhos, kde=True, ax=ax, color='gray', stat='count', element="step", alpha=0.4)
        ax.axvline(observed, color=item["color"], linestyle='--', linewidth=3, label=f"Observed ρ={observed:.2f}")
        
        ax.set_title(f"{item['title']}\np={p_val:.4f}", fontsize=12, fontweight='bold')
        ax.set_xlabel(item["xlabel"])
        ax.legend(loc='upper left', fontsize=10)
        
    r0 = results[0]
    monkey = r0["monkey"]
    
    if out_dir:
        # Use the same logic as the main plot to get into the correct subdirectory
        full_path = runtime.paths.get_rep_stability_path(
            output_dir=out_dir,
            monkey_name=monkey,
            analysis_type=r0["method"],
            group_size=r0["block_size"],
            region_id=None, # Generic
            src_tgt=None,   # Generic
            suffix=f"_PermutationTest{suffix}",
            extension=".png"
        )
        
        fig.savefig(full_path, dpi=200, bbox_inches="tight")
        print(f"[✓] Permutation Plot Saved → {full_path}")
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
