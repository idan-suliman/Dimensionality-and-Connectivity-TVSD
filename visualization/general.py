from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
import pandas as pd
import matplotlib.colors as mcolors
from collections import Counter, defaultdict
from matplotlib import gridspec
import matplotlib.patheffects as pe
from pathlib import Path
from core.runtime import runtime
from .utils import d95_from_curves, jitter, square_limits, labeled_dot, smart_label

class GeneralPlots:
    """
    Namespace for general exploratory plots.
    """
    
    @staticmethod
    def get_title(region: str) -> str:
        """Standardized plot title."""
        return f"{region} • {runtime.get_cfg().get_monkey_name()}  •  {runtime.get_cfg().get_zscore_title()}"

    @staticmethod
    def plot_mean_amplitude_by_repetition():
        """
        Plot 1: Average amplitude across all stimuli per repetition.
        Plot 2: Same, but broken down by Region (V1/V4/IT).
        """
        data = runtime.get_data_manager()._load_trials()
        rois = runtime.get_cfg().get_rois()
        region_names = runtime.get_consts().REGION_ID_TO_NAME

        region_channels = {
            region_names[1]: np.where(rois == 1)[0],
            region_names[2]: np.where(rois == 2)[0],
            region_names[3]: np.where(rois == 3)[0],
        }

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

        rep_main_day = {
            rep: int(Counter(days).most_common(1)[0][0])
            for rep, days in rep_to_days.items()
        }

        day_labels = [rep_main_day[r] for r in rep_indices]
        all_days = sorted(set(day_labels))
        day_to_idx = {d: i for i, d in enumerate(all_days)}
        color_indices = [day_to_idx[d] for d in day_labels]
        
        cmap = plt.get_cmap("tab10", len(all_days))
        norm = BoundaryNorm(boundaries=np.arange(0, len(all_days)+1), ncolors=len(all_days))

        mean_by_region = {
            reg: [np.mean(rep_to_amps[reg][r]) for r in rep_indices]
            for reg in region_channels
        }
        region_colors = {reg: color for reg, color in zip(region_channels.keys(), ["tab:red", "tab:blue", "tab:green"])}

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
        ylabel = "Mean Z-scored MUA" if runtime.get_cfg().get_zscore_code() != 1 else "Mean Raw MUA"

        ax1.plot(rep_indices, avg_amplitudes, color="gray", linewidth=1.5, zorder=1)
        for x, y, ci in zip(rep_indices, avg_amplitudes, color_indices):
            ax1.scatter(x, y, color=cmap(ci), edgecolors='k', linewidths=0.4, s=60, zorder=2)
        ax1.set_ylabel(ylabel, fontsize=10, labelpad=10)
        ax1.set_title("Mean Response Amplitude per Repetition (All Regions)")
        ax1.grid(True, alpha=0.3)

        for reg, y_vals in mean_by_region.items():
            ax2.plot(rep_indices, y_vals, label=reg, color=region_colors[reg], linewidth=2, zorder=1)
            for x, y, ci in zip(rep_indices, y_vals, color_indices):
                ax2.scatter(x, y, color=cmap(ci), edgecolors='k', linewidths=0.4, s=60, zorder=2)

        ax2.set_xlabel("Repetition Index (0–29)")
        ax2.set_ylabel(ylabel, fontsize=10, labelpad=10)
        ax2.set_title("Mean Response per Repetition (By Region)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(title="Region")

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
        data = runtime.get_data_manager()._load_trials()
        rois = runtime.get_cfg().get_rois()

        region_ids = {"V1": 1, "V4": 2, "IT": 3}
        region_id = region_ids[region.upper()]
        channels = np.where(rois == region_id)[0]
        NUM_REPS = 30

        rep_to_trials = defaultdict(list)
        for trial in data:
            rep = int(trial["allmat_row"][3]) - 1
            rep_to_trials[rep].append(trial)

        rep_mean_tc, rep_std_tc = [], []
        for r in range(NUM_REPS):
            trials = rep_to_trials[r]
            if not trials: continue
            mua_stack = np.stack([t["mua"][:, channels] for t in trials], axis=0) 
            rep_mean_tc.append(mua_stack.mean(axis=(0, 2))) 
            rep_std_tc.append(mua_stack.std(axis=(0, 2)))

        cmap = LinearSegmentedColormap.from_list("RedToBlue", ["red", "blue"])
        norm = Normalize(vmin=0, vmax=NUM_REPS - 1)
        colors = [cmap(norm(r)) for r in range(NUM_REPS)]

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
        for region in ["V1", "V4", "IT"]:
            print(f"\n[🧠] Plotting repeatwise timecourses for region: {region}")
            GeneralPlots.plot_repeatwise_timecourses(region)

    @staticmethod
    def plot_mean_std_amplitude_by_repetition():
        data = runtime.get_data_manager()._load_trials()
        rois_logical = runtime.get_cfg().get_rois()
        region_names = runtime.get_consts().REGION_ID_TO_NAME
        
        region_channels = {
            region_names[1]: np.where(rois_logical == 1)[0],
            region_names[2]: np.where(rois_logical == 2)[0],
            region_names[3]: np.where(rois_logical == 3)[0],
        }

        records = []
        rep_to_trials = defaultdict(list)
        for trial in data:
            rep = int(trial["allmat_row"][3]) - 1
            rep_to_trials[rep].append(trial)

        for rep in sorted(rep_to_trials.keys()):
            trials = rep_to_trials[rep]
            for region, ch_idx in region_channels.items():
                if len(ch_idx) == 0: continue
                all_amps = []
                for t in trials:
                    chan_means = t["mua"][:, ch_idx].mean(axis=0)
                    all_amps.extend(chan_means)
                
                records.append({
                    "Repetition": rep,
                    "Region": region,
                    "MeanAmplitude": np.mean(all_amps),
                    "StdAmplitude": np.std(all_amps)
                })

        df = pd.DataFrame.from_records(records)

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
        print("[🌍] Building global electrode & trial distribution plot...")
        consts = runtime.get_consts()
        base = consts.BASE_DIR
        region_map = consts.REGION_ID_TO_NAME
        monkeys = consts.MONKEYS

        electrode_recs = []
        trial_by_monkey = defaultdict(dict)
        summary = {}

        for monkey in monkeys:
            runtime.set_cfg(monkey, 1) 
            cfg = runtime.get_cfg()

            cnts = Counter(cfg.get_rois())
            summary[monkey] = {region_map[rid]: cnts.get(rid, 0) for rid in region_map}
            for rid, rname in region_map.items():
                electrode_recs.append((monkey, rname, cnts.get(rid, 0)))

            trials = runtime.get_data_manager()._load_trials()
            day_key = "day_id" if "day_id" in trials[0] else "dayID"
            for d, c in Counter(t[day_key] for t in trials).items():
                trial_by_monkey[monkey][d] = c
            summary[monkey]["trials"] = sum(trial_by_monkey[monkey].values())

        print("\n[📊] Summary:")
        for m, info in summary.items():
            print(f"  {m}: {info}")

        max_day = max((max(d.keys()) for d in trial_by_monkey.values() if d), default=0)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
        fig.suptitle("Global Electrode & Trial Distribution", fontsize=16, fontweight="bold")

        x1 = np.arange(len(electrode_recs))
        vals1 = [x[2] for x in electrode_recs]
        cols1 = ["steelblue" if x[0] == "Monkey F" else "orange" for x in electrode_recs]
        
        ax1.bar(x1, vals1, color=cols1, edgecolor="black")
        ax1.set_xticks(x1)
        ax1.set_xticklabels([f"{x[0][7]}-{x[1]}" for x in electrode_recs]) 
        ax1.set_ylabel("Electrode Count")
        ax1.set_title("Electrodes per Region per Monkey")
        ax1.legend(handles=[Patch(color="steelblue", label="Monkey F"), Patch(color="orange", label="Monkey N")])
        for i, v in zip(x1, vals1): ax1.text(i, v+5, str(v), ha="center", va="bottom", fontsize=9)

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
        data = runtime.get_data_manager()._load_trials()
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

        out_path = runtime.get_cfg().get_plot_dir() / "repeatwise_mean_timecourses_all_regions.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[📈] Saved plot to: {out_path.name}")


    # plot 1
    @staticmethod
    def plot_zscore_comparison_v1_v4(monkey: str = "Monkey F"):
        """
        Creates a 4x2 subplot grid (Rows: Z-scores 1-4, Cols: V1, V4).
        Plots repeatwise timecourses for V1 and V4 for each Z-score configuration.
        """
        print(f"\n[🧠] Plotting Z-Score Comparison for {monkey} (V1 & V4)...")
        
        region_ids = {"V1": 1, "V4": 2}
        NUM_REPS = 30
        
        # Setup Figure - Using (24, 16) for wider aspect ratio as requested ("rectangular")
        fig, axes = plt.subplots(4, 2, figsize=(27, 16), sharex=True)
        
        # Color Map
        cmap = LinearSegmentedColormap.from_list("RedToBlue", ["red", "blue"])
        norm = Normalize(vmin=0, vmax=NUM_REPS - 1)
        colors = [cmap(norm(r)) for r in range(NUM_REPS)]
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Specific user-defined Row Titles
        # Z=1: RAW
        # Z=2: PER-DAY
        # Z=3: GLOBAL
        # Z=4: REPETITION-WISE
        z_score_row_labels = {
             1: "RAW",
             2: "DAY",
             3: "GLOBAL",
             4: "REPETITION"
        }

        # Iterate 1..4 (Rows)
        for z_idx, z_code in enumerate([1, 2, 3, 4]):
            # Re-initialize runtime for this Z-score
            runtime.set_cfg(monkey, z_code)
            data = runtime.get_data_manager()._load_trials()
            rois = runtime.get_cfg().get_rois()
            
            row_label_text = z_score_row_labels[z_code]

            rep_to_trials = defaultdict(list)
            for trial in data:
                rep = int(trial["allmat_row"][3]) - 1
                rep_to_trials[rep].append(trial)

            # Pre-calculate timecourses to find common Y-limits for this ROW
            # Store (concatenated_timecourses, mean_timecourses_per_rep)
            row_data = {}
            all_min = float('inf')
            all_max = float('-inf')

            for region in ["V1", "V4"]:
                rid = region_ids[region]
                chans = np.where(rois == rid)[0]
                
                # Compute all repetition means
                means_list = []
                for r in range(NUM_REPS):
                    trials = rep_to_trials[r]
                    if not trials: 
                        means_list.append(None)
                        continue
                    # Shape: (n_trials, n_chans, n_time) -> mean(axis=0,2) ?? 
                    # WAIT: standard function does: mua_stack.mean(axis=(0, 2)) -> SINGLE VALUE per channel? 
                    # NO, standard function logic:
                    # mua_stack = np.stack([t["mua"][:, chans] for t in trials], axis=0) -> (n_trials, n_time, n_chans)
                    # Note: "mua" in trials is usually (Time, Channel) or (Channel, Time)? 
                    # Let's check `general.py` line 327: 
                    # tc = np.stack([t["mua"][:, chans] for t in trials]).mean(axis=(0, 2))
                    # t["mua"] shape is (Time, Channels). 
                    # Stack -> (Trials, Time, Channels).
                    # mean(axis=(0, 2)) -> Average over Trials(0) and Channels(2) -> Result is (Time,) Vector.
                    # Correct.
                    
                    tc = np.stack([t["mua"][:, chans] for t in trials]).mean(axis=(0, 2))
                    means_list.append(tc)
                    all_min = min(all_min, tc.min())
                    all_max = max(all_max, tc.max())
                
                row_data[region] = means_list

            # Plotting for this Row
            for col_idx, region in enumerate(["V1", "V4"]):
                ax = axes[z_idx, col_idx]
                means_list = row_data[region]
                
                # Add Zero Line
                ax.axhline(0, color='gray', alpha=0.4, linewidth=2, linestyle='-')

                for r, tc in enumerate(means_list):
                    if tc is not None:
                        ax.plot(tc, color=colors[r], linewidth=2) # Slightly thicker

                # Unified Y-Limits for this Row
                # Add a small margin
                margin = (all_max - all_min) * 0.1
                y_low, y_high = all_min - margin, all_max + margin
                ax.set_ylim(y_low, y_high)
                
                # Strict X-Limits to start at 0
                ax.set_xlim(0, 300) 
                
                # 3 Ticks (Min, Mid, Max)
                mid = (y_low + y_high) / 2
                ticks = [y_low, mid, y_high]
                ax.set_yticks(ticks)
                
                # Format to 1 decimal place
                from matplotlib.ticker import FormatStrFormatter
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                
                # Styling
                
                # Column Titles (Top Row Only) - HUGE, moved consistently
                if z_idx == 0:
                    ax.set_title(f"{region}", fontsize=45, pad=20)
                
                # Row Labels (Left Column Only)
                if col_idx == 0:
                   # Left-most label: Smaller (28), Bold, Moved slightly left (-0.18)
                   ax.text(-0.18, 0.5, row_label_text, transform=ax.transAxes, 
                           rotation='vertical', va='center', ha='center', fontsize=30, fontweight='bold')
                else:
                    # Hide Y-axis labels for V4
                    ax.tick_params(labelleft=False)

                # Huge Ticks - Unified Size (32 for both)
                ax.tick_params(axis='y', which='major', labelsize=32, width=3, length=10)
                ax.tick_params(axis='x', which='major', labelsize=32, width=3, length=10)
                
                # Explicit X-Ticks
                ax.set_xticks([0, 150, 300])
                
                # Grid
                ax.grid(True, alpha=0.3)

        # X Axis Label - HUGE - Moved UP (y=0.06)
        fig.supxlabel("TIME (ms)", fontsize=50, y=0.06)
        
        # Adjust layout FIRST to finalize axes positions
        # rect reduced on right to 0.88 to give room for cbar
        # Adjusted left to 0.05 to reduce left margin gap
        fig.tight_layout(rect=[0.05, 0.09, 0.88, 0.98], h_pad=0.5, w_pad=2)
        
        # Now add Colorbar aligned to the axes
        # Get position of top-right and bottom-right axes
        pos_top = axes[0, 1].get_position()
        pos_bot = axes[3, 1].get_position()
        
        cbar_bottom = pos_bot.y0
        cbar_top = pos_top.y1
        cbar_height = cbar_top - cbar_bottom
        # Place cbar slightly to the right of the plots
        cbar_left = pos_top.x1 + 0.01 
        cbar_width = 0.02
        
        cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height]) 
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0, NUM_REPS - 1])
        
        # Label: "move its text... a bit left" -> Reduce labelpad
        cbar.set_label("Repetition Index (Red=Early, Blue=Late)", fontsize=36, fontweight="bold", labelpad=20)
        
        # Change ticks to 1 and 30, BOLD and HUGE
        cbar.set_ticklabels(["1", "30"], fontweight="bold", fontsize=40)
        cbar.ax.tick_params(size=0) 

        # Add Global Y-Axis Label
        # Calculate vertical center based on axes, similar to colorbar logic
        pos_top_left = axes[0, 0].get_position()
        pos_bot_left = axes[3, 0].get_position()
        y_center_left = (pos_top_left.y1 + pos_bot_left.y0) / 2
        
        # Moved to x=0.015 to sit tight in the reduced margin, centered vertically on the plots
        fig.text(0.015, y_center_left, "Average electrode activity", va='center', rotation='vertical', fontsize=50) 


        
        out_path = runtime.get_cfg().get_plot_dir() / f"PREVIEW_zscore_comparison_{monkey.replace(' ', '_')}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[🖼️] Generated preview plot: {out_path.name}")


    # plot 2
    @staticmethod
    def plot_electrode_matching_v1_v4(monkey_name: str = "Monkey F", analysis_type: str = "residual"):
        """
        Generates a high-quality 'Before vs After' histogram of electrode matching 
        for V1 -> V4 (Residual mode).
        """
        print(f"\n[🧠] Plotting Electrode Matching V1->V4 ({analysis_type}) for {monkey_name}...")
        
        # Ensure correct config
        runtime.set_cfg(monkey_name, 3) # Z=3 (Global) as per user request (Method 3)
        
        from methods.matchingSubset import MATCHINGSUBSET
        
        region_source = "V1"
        region_target = "V4"
        stat_mode = analysis_type
        
        # 1. Get Data
        Xs, idx_s = MATCHINGSUBSET._compute_matrix(region_source, stat_mode)
        Xt, _     = MATCHINGSUBSET._compute_matrix(region_target, stat_mode)
        
        # 2. Compute Subset (using the matching logic)
        # We can re-call match_and_save to get the indices, or just copy the logic.
        # Calling it is safer to ensure consistency.
        _, phys_sub_idx = MATCHINGSUBSET.match_and_save(
            region_source, region_target, stat_mode=stat_mode, show_plot=False, verbose=False
        )
        
        # Map physical indices back to logical indices in Xs
        # idx_s contains the physical indices of the rows in Xs.
        # phys_sub_idx contains the subset of physical indices.
        # We need the boolean mask or indices into Xs that correspond to phys_sub_idx
        subset_mask = np.isin(idx_s, phys_sub_idx)
        
        # 3. Prepare Means
        ms = Xs.mean(0) # V1 Full Means
        mt = Xt.mean(0) # V4 Means
        ms_sub = ms[subset_mask] # V1 Subset Means
        
        # 4. Plot
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
        
        # Shared bins
        vmin = min(ms.min(), mt.min())
        vmax = max(ms.max(), mt.max())
        edges = np.linspace(vmin, vmax, 21)
        
        # Colors
        col_v1_full = "#E91E63" # Pink/Red
        col_v1_sub  = "#9C27B0" # Purpleish
        col_v4      = "#2196F3" # Blue
        alpha = 0.6
        
        # --- Panel A: BEFORE ---
        ax1 = axes[0]
        ax1.hist(ms, bins=edges, alpha=alpha, color=col_v1_full, label="V1")
        ax1.hist(mt, bins=edges, alpha=alpha, color=col_v4,      label="V4")
        
        ax1.set_title("BEFORE", fontsize=40, pad=20)
        ax1.set_xlabel("Mean Firing Rate", fontsize=45, labelpad=15)
        ax1.set_ylabel("Electrode Count", fontsize=40, labelpad=15)
        ax1.tick_params(axis='both', labelsize=40, width=3, length=10)
        ax1.legend(fontsize=25)
        ax1.grid(alpha=0.2)
        
        # --- Panel B: AFTER ---
        ax2 = axes[1]
        ax2.hist(ms_sub, bins=edges, alpha=alpha, color=col_v1_full, label="Target V1") # Keep color consistent? Or use purple to show mixed?
        # User image showed V1-subset as Reddish/Purple overlap. Let's use same Red but maybe different label color?
        # Actually, let's stick to the user reference image style:
        # Before: Red (Full) vs Blue (V4)
        # After:  Red (Subset) vs Blue (V4) -> Overlap looks purple.
        # So we use the SAME Red color for V1-subset.
        
        ax2.hist(mt, bins=edges, alpha=alpha, color=col_v4,      label="V4")
        
        ax2.set_title("AFTER", fontsize=40, pad=20)
        ax2.set_xlabel("Mean Firing Rate", fontsize=45, labelpad=15)
        # Shared Y label is fine, but repeated is okay too.
        ax2.tick_params(axis='both', labelsize=40, width=3, length=10)
        ax2.legend(fontsize=25)
        ax2.grid(alpha=0.2)
        
        # Add Arrow between them? Hard to do with subplots beautifully, skipping for now unless layout permits.
        # We can add a big arrow in the middle.
        
        fig.tight_layout()
        
        # Save
        out_path = runtime.get_cfg().get_plot_dir() / f"MATCHING_V1_V4_{stat_mode.upper()}_HighRes.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[✓] Matching Histogram saved -> {out_path.name}")


    # semedo figure 4 side by side plot 3
    @staticmethod
    def plot_semedo_figure_4_side_by_side(
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
        Modified Figure 4: Panels A and B are side-by-side. 
        Panels C and D are REMOVED. Main Title REMOVED.
        """
        # Reduced height since we only have 1 row
        fig = plt.figure(figsize=(25, 12), dpi=400) # Slightly wider/taller to help
        # Adjust specific margins - significantly increased LEFT and WSPACE
        fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.18, wspace=0.35)

        gs  = gridspec.GridSpec(1, 2)
        axA = fig.add_subplot(gs[0, 0]) # Left
        axB = fig.add_subplot(gs[0, 1]) # Right

        dims   = np.arange(1, d_max + 1)
        tgt    = runtime.get_consts().REGION_ID_TO_NAME[target_region]
        colA, colB = "#9C1C1C", "#1565C0"
        label_fs = 20

        # Calculate Unified Y-Limits
        # Get min/max from means + sems to be safe
        y_vals_all = []
        for p in [perf_full, perf_match]:
            # R2 mean +/- sem
            arr_mean = p["rrr_R2_mean"][:30] # Limit to 30 dims for calculation if we are clipping x
            arr_sem  = p["rrr_R2_sem"][:30]
            y_vals_all.append(arr_mean + arr_sem)
            y_vals_all.append(arr_mean - arr_sem)
            # Also ridge point
            y_vals_all.append(p["ridge_R2_mean"])

        flat_y = np.concatenate([np.atleast_1d(x) for x in y_vals_all])
        y_min_val, y_max_val = np.min(flat_y), np.max(flat_y)
        # Add padding
        y_range = y_max_val - y_min_val
        y_lims = (y_min_val - 0.05 * y_range, y_max_val + 0.05 * y_range)

        # Styling Helper
        def style_panel(ax, title, color, letter):
            ax.set_title(title, color=color, pad=20, fontsize=40, fontweight='bold')
            ax.grid(alpha=.25)
            # Letter moved down slightly (1.01) to stay near plot
            ax.text(-0.07, 1.01, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=48, fontweight="bold")
            # Increased labelsize 36 -> 45
            ax.tick_params(axis='both', which='major', labelsize=45, width=3, length=10)
            ax.set_xlabel("Predictive dimensions (d)", fontsize=45, labelpad=15)
            ax.set_box_aspect(1) # FORCE SQUARE
            
            # Limits
            ax.set_xlim(0, 30)
            ax.set_ylim(y_lims)
            
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        # --- Panel A: Full Model ---
        # Limit plotting to d_max=30 or user req
        plot_dims = dims[dims <= 30]
        p_full_mean = perf_full["rrr_R2_mean"][:len(plot_dims)]
        p_full_sem  = perf_full["rrr_R2_sem"][:len(plot_dims)]

        axA.errorbar(plot_dims, p_full_mean, yerr=p_full_sem,
                     fmt="o-", ms=10, lw=3, capsize=6, color=colA, zorder=2)
        # Larger Triangle (s=400 -> 1000)
        axA.scatter([1], [perf_full["ridge_R2_mean"]], marker="^", s=1000,
                    color=colA, edgecolors="k", zorder=3)
        
        if np.isfinite(d95_full_g):
            d95 = int(d95_full_g)
            if d95 <= 30:
                r2d = perf_full["rrr_R2_mean"][d95 - 1]
                # HUGE circle and text (increased per request)
                labeled_dot(axA, d95, float(r2d), d95,
                            face=colA, edge="k", size=2200, text_size=42, text_color="white")
        
        # Simplified Title per specific request/image
        style_panel(axA, f"Predicting {tgt}", colA, "A")
        axA.set_ylabel(rf"Mean $R^2$", fontsize=45, labelpad=15)

        # --- Panel B: Match Model ---
        p_match_mean = perf_match["rrr_R2_mean"][:len(plot_dims)]
        p_match_sem  = perf_match["rrr_R2_sem"][:len(plot_dims)]

        axB.errorbar(plot_dims, p_match_mean, yerr=p_match_sem,
                     fmt="o-", ms=10, lw=3, capsize=6, color=colB, zorder=2)
        # Larger Triangle
        axB.scatter([1], [perf_match["ridge_R2_mean"]], marker="^", s=1000,
                    color=colB, edgecolors="k", zorder=3)

        if np.isfinite(d95_match_g):
            d95 = int(d95_match_g)
            if d95 <= 30:
                r2d = perf_match["rrr_R2_mean"][d95 - 1]
                # HUGE circle and text (increased per request)
                labeled_dot(axB, d95, float(r2d), d95,
                            face=colB, edge="k", size=2200, text_size=42, text_color="white")

        style_panel(axB, f"Predicting Target V1", colB, "B")
        axB.set_ylabel(rf"Mean $R^2$", fontsize=45, labelpad=15)

        # NO SUPLTITLE

        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Alter filename to indicate side-by-side
            new_path = out_path.with_name(out_path.stem + "_SIDE_BY_SIDE" + out_path.suffix)
            fig.savefig(new_path, dpi=400, facecolor="white")
            print(f"[✓] Figure 4 (Side-by-Side) saved → {new_path}")
        
        plt.close(fig)

    @staticmethod
    def plot_semedo_figure_4_side_by_side_wrapper(monkey_name: str, force_Z_code: int = None):
        """
        Wrapper to find the most relevant cached .npz data for Figure 4 and plot it side-by-side.
        """
        # 1. Setup Configuration
        if force_Z_code:
            runtime.set_cfg(monkey_name, force_Z_code)
        elif runtime.get_cfg().get_monkey_name() != monkey_name:
             runtime.set_cfg(monkey_name, 2) # Default to 2 if not set

        cfg = runtime.get_cfg()
        
        # 2. Locate Data
        out_dir = cfg.get_data_path() / "Semedo_plots" / "Figure_4"
        if not out_dir.exists():
            print(f"[!] No Semedo Figure 4 data directory found: {out_dir}")
            return
            
        # Search for .npz files matching the monkey
        pattern = f"{monkey_name.replace(' ', '')}_Figure_4_*.npz"
        files = list(out_dir.glob(pattern))
        
        if not files:
            print(f"[!] No .npz cache files found for {monkey_name} in {out_dir}")
            return
            
        # Sort by modification time to get the most recent
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        target_file = files[0]
        print(f"[Cache] Using most recent cache file: {target_file.name}")
        
        # 3. Load Data
        with np.load(target_file, allow_pickle=True) as data:
            perf_full = data["perf_full"].item()
            perf_match = data["perf_match"].item()
            d95_full_g = int(data["d95_full_g"])
            d95_match_g = int(data["d95_match_g"])
            d95_full_rep = data["d95_full_rep"].tolist()
            d95_match_rep = data["d95_match_rep"].tolist()
            d_max = int(data["d_max"])
            label_D = str(data["label_D"])
            # Some older caches might not have analysis_type, default it
            analysis_type = str(data["analysis_type"]) if "analysis_type" in data else "residual"
            
            # Reconstruct other params if missing or standard
            k_subsets = 10 if "sub10" in target_file.name else None
            # Target region usually V4 or IT.
            target_region = 3 if "IT" in target_file.name else 2 # Default to IT (3) if found, else V4 (2).
            # Default split params (formatting only)
            outer_splits = 3
            inner_splits = 3
            random_state = 0 # Not used for logic, just jitter

        # 4. Plot
        save_path = target_file.with_suffix(".png") # Will be modified by function to _SIDE_BY_SIDE
        
        GeneralPlots.plot_semedo_figure_4_side_by_side(
            perf_full, perf_match, d95_full_g, d95_match_g,
            d95_full_rep, d95_match_rep, d_max, target_region, analysis_type,
            k_subsets, outer_splits, inner_splits, random_state, label_D,
            save_path=str(save_path)
        )

   
    # plot 4 subset
    @staticmethod
    def plot_semedo_figure_4_subset_side_by_side(
        results_full: list,
        results_match: list,
        d_max: int,
        target_region: int,
        analysis_type: str,
        save_path: str | None = None,
    ):
        """
        Modified Figure 4 SUBSET: Panels A and B are side-by-side. 
        Plots 5 runs (or N runs) for both Full and Match conditions.
        """
        # Reduced height since we only have 1 row
        fig = plt.figure(figsize=(25, 12), dpi=400) # Slightly wider/taller to help
        # Adjust specific margins - significantly increased LEFT and WSPACE
        fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.18, wspace=0.35)

        gs  = gridspec.GridSpec(1, 2)
        axA = fig.add_subplot(gs[0, 0]) # Left
        axB = fig.add_subplot(gs[0, 1]) # Right

        dims   = np.arange(1, d_max + 1)
        tgt    = runtime.get_consts().REGION_ID_TO_NAME[target_region]
        
        # Colors for runs - using a colormap or fixed list
        # User image uses distinct colors: Orange, Blue, Green, Purple, Red?
        # Let's use a quality palette
        cmap = plt.get_cmap("tab10")
        
        # Label Helper
        label_fs = 20

        # Calculate Unified Y-Limits
        # Get min/max from means + sems to be safe
        y_vals_all = []
        
        def collect_y_vals(run_list):
            for res in run_list:
                # Handle both dict and object (if loaded differently)
                p = res if isinstance(res, dict) else vars(res)
                arr_mean = p["rrr_R2_mean"][:30]
                arr_sem  = p["rrr_R2_sem"][:30]
                y_vals_all.append(arr_mean + arr_sem)
                y_vals_all.append(arr_mean - arr_sem)
                y_vals_all.append(p["ridge_R2_mean"])

        collect_y_vals(results_full)
        collect_y_vals(results_match)

        flat_y = np.concatenate([np.atleast_1d(x) for x in y_vals_all])
        y_min_val, y_max_val = np.min(flat_y), np.max(flat_y)
        # Add padding
        y_range = y_max_val - y_min_val
        y_lims = (y_min_val - 0.05 * y_range, y_max_val + 0.05 * y_range)

        # Styling Helper
        def style_panel(ax, title, color, letter):
            # Same aesthetics as Plot 2
            ax.set_title(title, color=color, pad=20, fontsize=40, fontweight='bold') # Added bold
            ax.grid(alpha=.25)
            # Letter moved down slightly (1.01) to stay near plot
            ax.text(-0.07, 1.01, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=48, fontweight="bold")
            # Increased labelsize 36 -> 45
            ax.tick_params(axis='both', which='major', labelsize=45, width=3, length=10)
            ax.set_xlabel("Predictive dimensions (d)", fontsize=45, labelpad=15)
            ax.set_box_aspect(1) # FORCE SQUARE
            
            # Limits
            ax.set_xlim(0, 30)
            ax.set_ylim(y_lims)
            
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        # --- Plotting Helper ---
        def plot_runs(ax, run_list, base_color_idx=0, is_panel_a=True):
            plot_dims = dims[dims <= 30]
            
            for i, res in enumerate(run_list):
                p = res if isinstance(res, dict) else vars(res)
                p_mean = p["rrr_R2_mean"][:len(plot_dims)]
                p_sem  = p["rrr_R2_sem"][:len(plot_dims)]
                
                # Color cycler
                color = cmap(i) 
                
                # Line
                ax.errorbar(plot_dims, p_mean, yerr=p_sem,
                            fmt="o-", ms=8, lw=2, capsize=4, color=color, zorder=2, alpha=0.9, label=f"Run {i+1}")
                
                # Ridge Triangle
                ax.scatter([1], [p["ridge_R2_mean"]], marker="^", s=1000,
                           color=color, edgecolors="k", zorder=3)
                           
                # d95 Dot
                # In subset runs, key is often just 'd95' or 'd95_model' depending on calc
                d95_key = 'd95' if 'd95' in p else 'd95_model'
                if d95_key in p:
                     d95_val = p[d95_key]
                     # Ensure it's a scalar
                     if isinstance(d95_val, (list, np.ndarray)):
                         if len(d95_val) > 0: d95 = int(d95_val[0])
                         else: d95 = 0
                     else:
                         d95 = int(d95_val)
                         
                     if d95 > 0 and d95 <= 30:
                         r2d = p["rrr_R2_mean"][d95 - 1]
                         labeled_dot(ax, d95, float(r2d), d95,
                                    face=color, edge="k", size=2200, text_size=42, text_color="white")

        # --- Panel A: Full Model ---
        plot_runs(axA, results_full)
        # Use Red Title #9C1C1C
        style_panel(axA, f"Predicting {tgt}", "#9C1C1C", "A")
        axA.set_ylabel(rf"Mean $R^2$", fontsize=45, labelpad=15)

        # --- Panel B: Match Model ---
        plot_runs(axB, results_match)
        # Use Blue Title #1565C0
        style_panel(axB, f"Predicting Target V1", "#1565C0", "B")
        axB.set_ylabel(rf"Mean $R^2$", fontsize=45, labelpad=15)

        # Legend?
        # axB.legend(fontsize=20, loc='lower right')

        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            new_path = out_path.with_name(out_path.stem + "_SIDE_BY_SIDE" + out_path.suffix)
            fig.savefig(new_path, dpi=400, facecolor="white")
            print(f"[✓] Figure 4 Subset (Side-by-Side) saved → {new_path}")
        
        plt.close(fig)

    @staticmethod
    def plot_semedo_figure_4_subset_wrapper(monkey_name: str, analysis_type: str = "residual"):
        """
        Wrapper to find the most relevant cached .npz data for Figure 4 SUBSET and plot.
        """
        # Ensure runtime is configured
        if runtime.get_cfg() is None or runtime.get_cfg().get_monkey_name() != monkey_name:
            runtime.set_cfg(monkey_name, 3) # Defaulting to Global Z-score (3)

        cfg = runtime.get_cfg()
        # Ensure we look in the right Z-score/analysis folder?
        # The user provided path had 'subset/residual' inside.
        
        # We try to use the configured data path
        # Pattern: <DataPath>/Semedo_plots/Figure_4_subset/<analysis_type>/*.npz
        
        base_dir = cfg.get_data_path() / "Semedo_plots" / "Figure_4_subset" / analysis_type
        
        if not base_dir.exists():
             # Fallback to just Figure_4_subset root?
             base_dir = cfg.get_data_path() / "Semedo_plots" / "Figure_4_subset"
        
        if not base_dir.exists():
            print(f"[!] No Subset data directory found: {base_dir}")
            return

        pattern = f"{monkey_name.replace(' ', '')}_Figure_4_subset_*.npz"
        files = list(base_dir.rglob(pattern)) # Recursive in case structure varies
        
        if not files:
            print(f"[!] No .npz subset cache files found for {monkey_name} in {base_dir}")
            return
            
        # Sort
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        target_file = files[0]
        print(f"[Cache] Using most recent subset cache: {target_file.name}")
        
        with np.load(target_file, allow_pickle=True) as data:
            # Keys found: ['runs_full', 'runs_match', ...]
            
            if "runs_full" in data:
                # likely object array of dicts
                results_full = data["runs_full"].tolist() 
            elif "results_full" in data:
                results_full = data["results_full"].tolist()
            else:
                print(f"[!] 'runs_full' not found in {target_file.name}")
                return

            if "runs_match" in data:
                 results_match = data["runs_match"].tolist()
            elif "results_match" in data:
                 results_match = data["results_match"].tolist()
            else:
                 results_match = []

            # Meta params
            d_max = int(data["d_max"]) if "d_max" in data else 35
            target_region = 3 if "IT" in target_file.name else 2
             
        # Plot
        save_path = target_file.with_suffix(".png")
        GeneralPlots.plot_semedo_figure_4_subset_side_by_side(
            results_full, results_match, d_max, target_region, analysis_type,
            save_path=str(save_path)
        )


    # plot 5
    # Figure 5B (Standalone from CSV)
    # -------------------------------------------------------------------------
    @staticmethod
    def plot_figure_5_b_csv_wrapper(monkey_name: str, analysis_type: str = "residual"):
        """
        Plots Figure 5B using the standard CSV output found in Semedo_plots/figure_5_B.
        Filters for V4 and Target V4 (labeled as Target V1).
        """
        # 1. Config
        if runtime.get_cfg() is None or runtime.get_cfg().get_monkey_name() != monkey_name:
             runtime.set_cfg(monkey_name, 3)
        cfg = runtime.get_cfg()

        # 2. Find CSV
        # Expected: .../Semedo_plots/figure_5_B/figure_5_B_{ANALYSIS}_*_SETS.csv
        base_dir = cfg.get_data_path() / "Semedo_plots" / "figure_5_B"
        
        if not base_dir.exists():
            print(f"[!] No figure_5_B directory found: {base_dir}")
            return

        pattern = f"figure_5_B_{analysis_type.upper()}_*_SETS.csv"
        files = list(base_dir.glob(pattern))
        
        if not files:
            print(f"[!] No CSV found matching {pattern} in {base_dir}")
            return
            
        # Pick most recent or first
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        target_file = files[0]
        print(f"[Data] Using CSV: {target_file.name}")
        
        # 3. Read Data
        data_v4 = {"x": [], "y": [], "type": []}
        data_tv1 = {"x": [], "y": [], "type": []} # Target V4 in CSV -> Target V1 in plot
        
        import csv
        with open(target_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                grp = row["group"]
                x = float(row["x_dimensionality_d95"])
                y = float(row["y_predictive_d95"])
                pt_type = row["point_type"]
                
                if grp == "V4":
                    data_v4["x"].append(x)
                    data_v4["y"].append(y)
                    data_v4["type"].append(pt_type)
                elif grp == "Target V4":
                    data_tv1["x"].append(x)
                    data_tv1["y"].append(y)
                    data_tv1["type"].append(pt_type)
        
        # 4. Plot
        out_name = target_file.stem + "_V4_ONLY_HIGHER_RES.png"
        save_path = target_file.parent / out_name
        
        GeneralPlots.plot_figure_5_b_v4_only(data_v4, data_tv1, str(save_path))

    @staticmethod
    def plot_figure_5_b_v4_only(data_v4: dict, data_tv1: dict, save_path: str):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        
        # Style Settings
        col_v4 = "#9C1C1C" # Red (Consistent with Fig 4)
        col_tv1 = "#1565C0" # Blue (Consistent with Fig 4)
        
        # Helper to plot group
        def plot_group(d, color, label):
            xs = np.array(d["x"])
            ys = np.array(d["y"])
            types = np.array(d["type"])
            
            # Subsets ONLY (Full points removed per request)
            mask_sub = (types == "subset")
            if np.any(mask_sub):
                # Apply Jitter
                from .utils import jitter
                rng = np.random.default_rng(42)
                xs_j = jitter(xs[mask_sub], rng, scale=2) # Increased scale
                ys_j = jitter(ys[mask_sub], rng, scale=2) # Increased scale
                
                # Main scatter points with jitter
                ax.scatter(xs_j, ys_j, s=250, color=color, alpha=0.6, edgecolors='none', label=label)
                # Mean dot (Hollow ring with black edge) - Plot MEAN on original non-jittered data
                ax.scatter([xs[mask_sub].mean()], [ys[mask_sub].mean()], s=250, 
                           facecolors='none', edgecolors='k', linewidth=3, zorder=5)

        plot_group(data_v4, col_v4, "V4")
        plot_group(data_tv1, col_tv1, "Target V1")
        
        # Identity Line - Fixed to 100 per request
        mx = 100
        ax.plot([0, mx], [0, mx], 'k--', alpha=0.3, zorder=0)
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
            
        ax.set_aspect('equal')
        
        # Labels
        ax.set_xlabel("Target Population dimensionality", fontsize=35, labelpad=20)
        ax.set_ylabel("Predictive dimensions", fontsize=40, labelpad=20)
        # ax.set_title("V4 & Target V1 Dimensionality", fontsize=32, fontweight="bold", pad=20) # REMOVED per request
        
        ax.tick_params(axis='both', which='major', labelsize=40, width=3, length=12)
        # Custom Legend Elements
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=r'V1 $\rightarrow$ V4',
                   markerfacecolor=col_v4, markersize=25, alpha=0.6), 
            Line2D([0], [0], marker='o', color='w', label=r'V1 $\rightarrow$ Target V1',
                   markerfacecolor=col_tv1, markersize=25, alpha=0.6)
        ]
        ax.legend(handles=legend_elements, fontsize=30, loc='upper left', framealpha=0.9)
        ax.grid(alpha=0.2)
        
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, facecolor="white")
            print(f"[✓] Figure 5B (V4 Only) saved -> {Path(save_path).name}")
        plt.close(fig)



