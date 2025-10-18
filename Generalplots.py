from runtime import runtime
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import numpy as np
from matplotlib.cm import ScalarMappable
from pathlib import Path
import pandas as pd
import pickle
from scipy.io import loadmat
from collections import Counter, defaultdict
from matplotlib.patches import Patch


class Plots:

    def get_title(Region: str):
        return f"{Region} • {runtime.get_cfg().get_monkey_name()}  •  {runtime.get_cfg().get_zscore_title()}"


    @staticmethod
    def plot_mean_amplitude_by_repetition():
        data = runtime.get_cfg()._load_trials()
        rois = runtime.get_cfg().get_rois()
        region_names = runtime.get_consts().REGION_ID_TO_NAME

        region_channels = {
            region_names[1]: np.where(rois == 1)[0],
            region_names[2]: np.where(rois == 2)[0],
            region_names[3]: np.where(rois == 3)[0],
        }

        # ---------- Graph 1: overall mean ----------
        reps_table = defaultdict(list)
        for trial in data:
            rep = int(trial["allmat_row"][3]) - 1
            reps_table[rep].append(trial["mua"].mean())

        rep_indices = sorted(reps_table.keys())
        avg_amplitudes = [np.mean(reps_table[rep]) for rep in rep_indices]

        # ---------- Graph 2: per region + color by day ----------
        rep_to_days = defaultdict(list)
        rep_to_amps = {reg: defaultdict(list) for reg in region_channels}

        for trial in data:
            rep = int(trial["allmat_row"][3]) - 1
            day = trial["day_id"]
            mua = trial["mua"]
            rep_to_days[rep].append(day)
            for reg, ch_idx in region_channels.items():
                rep_to_amps[reg][rep].append(mua[:, ch_idx].mean())

        rep_main_day = {
            rep: int(np.unique(days, return_counts=True)[0][np.argmax(np.unique(days, return_counts=True)[1])])
            for rep, days in rep_to_days.items()
        }

        rep_idx = rep_indices
        day_labels = [rep_main_day[r] for r in rep_idx]
        all_days = sorted(set(day_labels))
        day_to_idx = {d: i for i, d in enumerate(all_days)}
        color_indices = [day_to_idx[d] for d in day_labels]
        cmap = plt.get_cmap("tab10", len(all_days))
        norm = mcolors.BoundaryNorm(boundaries=np.arange(0, len(all_days)+1), ncolors=len(all_days))

        mean_by_region = {
            reg: [np.mean(rep_to_amps[reg][r]) for r in rep_idx]
            for reg in region_channels
        }

        region_colors = {
            reg: color for reg, color in zip(region_channels.keys(), ["tab:red", "tab:blue", "tab:green"])
        }

        # ---------- Plot ----------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

        ylabel = "Mean Z-scored Multiunit Activity" if runtime.get_cfg().get_zscore_code() != 1 else "Mean Raw Multiunit Activity"

        # ---------- Graph 1 — line + colored scatter ----------
        ax1.plot(rep_idx, avg_amplitudes, color="gray", linewidth=1.5, zorder=1)
        for x, y, ci in zip(rep_idx, avg_amplitudes, color_indices):
            ax1.scatter(x, y, color=cmap(ci), edgecolors='k', linewidths=0.4, s=60, zorder=2)

        ax1.set_ylabel(ylabel, fontsize=10, labelpad=10)
        ax1.set_title("Mean Response Amplitude per Repetition (across all stimuli)")
        ax1.grid(True, alpha=0.3)

        # ---------- Graph 2 — by region: each with unique line color + colored scatter ----------
        for reg, y_vals in mean_by_region.items():
            line_color = region_colors[reg]
            ax2.plot(rep_idx, y_vals, label=reg, color=line_color, linewidth=2, zorder=1)
            for x, y, ci in zip(rep_idx, y_vals, color_indices):
                ax2.scatter(x, y, color=cmap(ci), edgecolors='k', linewidths=0.4, s=60, zorder=2)

        ax2.set_xlabel("Repetition Index (0–29)")
        ax2.set_ylabel(ylabel, fontsize=10, labelpad=10)
        ax2.set_title("Mean Activity per Repetition, by Brain Region")
        ax2.grid(True, alpha=0.3)
        ax2.legend(title="Region")

        # ---------- Colorbar ----------
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(len(all_days)))
        cbar.set_ticklabels([f"Day {d}" for d in all_days])
        cbar.set_label("Majority day per repetition")

        # ---------- Suptitle ----------
        fig.suptitle(Plots.get_title(Region= "mean all regions"), fontsize=16, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 0.9, 0.94])

        out_path = runtime.get_cfg().get_plot_dir() / "mean_response_by_repetition.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[📈] Saved plot to: {out_path.name}")

    @staticmethod
    def plot_repeatwise_timecourses(region: str = "V1"):
        data = runtime.get_cfg()._load_trials()
        rois = runtime.get_cfg().get_rois()

        region_ids = {"V1": 1, "V4": 2, "IT": 3}
        region_id = region_ids[region.upper()]
        channels = np.where(rois == region_id)[0]
        NUM_REPS = 30

        # Group trials by repetition
        rep_to_trials = {r: [] for r in range(NUM_REPS)}
        for trial in data:
            rep = int(trial["allmat_row"][3]) - 1
            rep_to_trials[rep].append(trial)

        # Compute mean + std timecourses per repetition
        rep_mean_tc = []
        rep_std_tc = []

        for r in range(NUM_REPS):
            trials = rep_to_trials[r]
            mua_stack = np.stack([t["mua"][:, channels] for t in trials], axis=0)
            mean_tc = mua_stack.mean(axis=(0, 2))
            std_tc  = mua_stack.std(axis=(0, 2))
            rep_mean_tc.append(mean_tc)
            rep_std_tc.append(std_tc)

        rep_mean_tc = np.array(rep_mean_tc)
        rep_std_tc  = np.array(rep_std_tc)

        # Color gradient: Red → Blue
        cmap = LinearSegmentedColormap.from_list("RedToBlue", ["red", "blue"])
        norm = Normalize(vmin=0, vmax=NUM_REPS - 1)
        colors = [cmap(norm(r)) for r in range(NUM_REPS)]
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # ---------- Plot ----------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        for r in range(NUM_REPS):
            ax1.plot(rep_mean_tc[r], color=colors[r], linewidth=1)
            ax2.plot(rep_std_tc[r], color=colors[r], linewidth=1)

        # Ax1 – Mean
        ax1.set_title(f"{region} — Mean Z-Score Over Time per Repeat")
        ax1.set_ylabel("Mean Z-Score", fontsize=10, labelpad=10)
        ax1.grid(True)

        # Ax2 – STD
        ax2.set_title(f"{region} — STD of Z-Score Over Time per Repeat")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("STD of Z-Score", fontsize=10, labelpad=10)
        ax2.grid(True)

        # Colorbar
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0, NUM_REPS - 1])
        cbar.set_label("Repetition Index (red = early, blue = late)")

        # Suptitle
        fig.suptitle(Plots.get_title(Region= region), fontsize=16, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 0.9, 0.94])

        out_path = runtime.get_cfg().get_plot_dir() / f"{region}_repeatwise_timecourses.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[📈] Saved plot to: {out_path.name}")
    
    ##this function is the same but print the 3 plots for each of the rois
    @staticmethod
    def plot_repeatwise_timecourses_all_regions():
        """Generate repeatwise timecourse plots for all regions: V1, V4, IT"""
        for region in ["V1", "V4", "IT"]:
            print(f"\n[🧠] Plotting repeatwise timecourses for region: {region}")
            Plots.plot_repeatwise_timecourses(region)

    @staticmethod
    def plot_mean_std_amplitude_by_repetition():
        data = runtime.get_cfg()._load_trials()
        rois_logical = runtime.get_cfg().get_rois()
        region_names = runtime.get_consts().REGION_ID_TO_NAME
        region_channels = {
            region_names[1]: np.where(rois_logical == 1)[0],
            region_names[2]: np.where(rois_logical == 2)[0],
            region_names[3]: np.where(rois_logical == 3)[0],
        }

        # Build region × repetition amplitude table
        region_rep_table = defaultdict(list)

        for trial in data:
            rep = int(trial["allmat_row"][3]) - 1
            mua = trial["mua"]

            for region, ch_idx in region_channels.items():
                amps = mua[:, ch_idx].mean(axis=0)  # mean over time per electrode
                for amp in amps:
                    region_rep_table[(region, rep)].append(amp)

        # Build DataFrame
        records = []
        for (region, rep), amps in region_rep_table.items():
            records.append({
                "Repetition": rep,
                "Region": region,
                "MeanAmplitude": np.mean(amps),
                "StdAmplitude": np.std(amps)
            })

        df = pd.DataFrame.from_records(records)

        # Plot
        plt.figure(figsize=(12, 6))
        for region in df["Region"].unique():
            sub = df[df["Region"] == region].sort_values("Repetition")
            plt.errorbar(
                sub["Repetition"],
                sub["MeanAmplitude"],
                yerr=sub["StdAmplitude"],
                label=region,
                marker='o',
                capsize=4
            )

        plt.title("Mean ± Std of Z-scored Amplitude per Repetition by Brain Region")
        plt.xlabel("Repetition Index (0–29)")
        plt.ylabel("Mean Amplitude ± Std (Z-scored)", fontsize=10, labelpad=10)
        plt.grid(True)
        plt.legend(title="Region")

        # Suptitle
        plt.suptitle(Plots.get_title(Region= "mean all regions"), fontsize=16, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = runtime.get_cfg().get_plot_dir() / "mean_std_amplitude_by_repetition.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[📈] Saved errorbar plot to: {out_path.name}")

    @staticmethod
    def plot_global_electrode_and_trial_distribution():
        """
        Global overview of the TVSD dataset:
            1. Electrode count per brain region (ROI) for each monkey
            2. Trial count per recording day, grouped by monkey

        Output → <base_dir>/PLOTS/global_electrode_and_trial_distribution.png
        """
        print("[🌍] Building global electrode & trial distribution plot...")

        consts = runtime.get_consts()
        base = consts.BASE_DIR
        region_map = consts.REGION_ID_TO_NAME
        monkeys = consts.MONKEYS

        # ---------------------------------------------------------------
        # Collect data: electrode counts per region, and trial counts per day
        # ---------------------------------------------------------------
        electrode_recs = []                 # (monkey, ROI, count)
        trial_by_monkey = defaultdict(dict) # monkey → {day: count}
        summary = {}                        # store counts for console log

        for monkey in monkeys:
            cfg = runtime.get_cfg()

            # --- ROI assignment for all electrodes (after mapping) ---
            rois_logical = cfg.get_rois()
            cnts = Counter(rois_logical)

            # Collect electrode counts
            summary[monkey] = {region_map[rid]: cnts.get(rid, 0) for rid in region_map}
            for roi_id, roi_name in region_map.items():
                electrode_recs.append((monkey, roi_name, cnts.get(roi_id, 0)))

            # --- Count trials per recording day ---
            trials = cfg._load_trials()
            day_key = "day_id" if "day_id" in trials[0] else "dayID"
            for d, c in Counter(t[day_key] for t in trials).items():
                trial_by_monkey[monkey][d] = c

            # Save total trial count for log
            summary[monkey]["trials"] = sum(trial_by_monkey[monkey].values())

        # Print summary log
        print("\n[📊] Electrode & Trial Summary:")
        for monkey, info in summary.items():
            v1 = info.get("V1", 0)
            v4 = info.get("V4", 0)
            it = info.get("IT", 0)
            tcount = info.get("trials", 0)
            print(f"  [{monkey[7]}]  V1={v1:<4} | V4={v4:<4} | IT={it:<4} | total trials={tcount}")
        print()

        max_day = max(max(d for d in trial_by_monkey[m]) for m in monkeys)

        # ---------------------------------------------------------------
        # Build figure layout
        # ---------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
        fig.suptitle("Global Electrode & Trial Distribution", fontsize=16, fontweight="bold")

        # ---------------------------------------------------------------
        # (1) Electrode counts per region
        # ---------------------------------------------------------------
        x1 = np.arange(len(electrode_recs))
        vals1 = [c for _, _, c in electrode_recs]
        cols1 = ["steelblue" if m == "Monkey F" else "orange" for m, _, _ in electrode_recs]

        ax1.bar(x1, vals1, color=cols1, edgecolor="black")
        ax1.set_ylabel("Electrode Count")
        ax1.set_xticks(x1)
        ax1.set_xticklabels([f"{m[7]}-{r}" for m, r, _ in electrode_recs])
        ax1.set_ylim(0, max(500, int(max(vals1) * 1.1)))

        # Label each bar with count
        for xi, v in zip(x1, vals1):
            ax1.text(xi, v + 5, str(v), ha="center", va="bottom", fontsize=9)

        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        ax1.set_title("Electrodes per Region per Monkey")
        ax1.legend(handles=[
            Patch(color="steelblue", label=consts.MONKEY_F),
            Patch(color="orange", label=consts.MONKEY_N)
        ], loc="upper right")

        # ---------------------------------------------------------------
        # (2) Trial counts per recording day
        # ---------------------------------------------------------------
        days = np.arange(1, max_day + 1)
        width = 0.35
        f_vals = [trial_by_monkey[consts.MONKEY_F].get(d, 0) for d in days]
        n_vals = [trial_by_monkey[consts.MONKEY_N].get(d, 0) for d in days]

        ax2.bar(days - width / 2, f_vals, width,
                color="steelblue", edgecolor="black", label=consts.MONKEY_F)
        ax2.bar(days + width / 2, n_vals, width,
                color="orange", edgecolor="black", label=consts.MONKEY_N)

        ax2.set_xlabel("Day ID")
        ax2.set_ylabel("Trial Count")
        ax2.set_xticks(days)
        ax2.set_xticklabels([f"Day {d}" for d in days])
        ax2.set_ylim(0, max(500, int(max(f_vals + n_vals) * 1.1)))

        # Label bars with counts
        for d, v in zip(days, f_vals):
            ax2.text(d - width / 2, v + 5, str(v), ha="center", va="bottom", fontsize=8)
        for d, v in zip(days, n_vals):
            ax2.text(d + width / 2, v + 5, str(v), ha="center", va="bottom", fontsize=8)

        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        ax2.set_title("Trials per Day (Grouped by Monkey)")
        ax2.legend(loc="upper right")

        # ---------------------------------------------------------------
        # Save output
        # ---------------------------------------------------------------
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_dir = base / "PLOTS"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "global_electrode_and_trial_distribution.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[✅] Saved global distribution plot → {out_path}\n")
    

    ##this function is the same but print everything monkey and z-score
    @staticmethod
    def generate_all_plots_for_all_combinations():
        monkeys = runtime.get_consts().MONKEYS
        zscore_codes = [1, 2, 3, 4]

        for monkey in monkeys:
            for z in zscore_codes:
                print(f"\n============================")
                print(f"[🔁] Running: {monkey}, Z-Score {z}")
                print(f"============================")

                #try:
                runtime.set_cfg(monkey, z)

                # Plot 1
                Plots.plot_mean_amplitude_by_repetition()

                # Plot 2
                Plots.plot_repeatwise_timecourses_all_regions()

                # Plot 3 
                Plots.plot_mean_std_amplitude_by_repetition()

                # Plot 4 – Dimensionality by region-specific window
                # Plots.plot_dimensionality_svd_windows_by_roi([0.90, 0.95])

                # Plot 5 – Dimensionality by uniform first 100ms
                # Plots.plot_dimensionality_svd_windows_first_100ms([0.90, 0.95])

                #except Exception as e:
                #    print(f"[⚠️] Error while processing {monkey} / Z{z}: {e}")