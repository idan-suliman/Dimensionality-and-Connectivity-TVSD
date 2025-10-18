from runtime import runtime
from Generalplots import Plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import svd

def plot_dimensionality_svd_windows_by_roi(thresholds: list = [0.90, 0.95]):
    # Load from class
    mapping = runtime.get_cfg().get_mapping()
    rois_logical = runtime.get_cfg().get_rois()
    trial_list = runtime.get_cfg()._load_trials()
    region_names = runtime.get_consts().REGION_ID_TO_NAME
    region_windows = runtime.get_consts().REGION_WINDOWS
    monkey = runtime.get_cfg().get_monkey_name()

    # Reverse map: 'V1' → 1
    name_to_id = {v: k for k, v in region_names.items()}

    # Channels by region
    channels_by_region = {
        region: np.where(rois_logical == name_to_id[region])[0]
        for region in region_names.values()
    }

    # Helper
    def compute_dim(X: np.ndarray, threshold: float) -> int:
        s = svd(X, compute_uv=False)
        var_exp = np.cumsum(s**2) / np.sum(s**2)
        return int(np.searchsorted(var_exp, threshold) + 1)

    # Group by day
    days = sorted(set(t["day_id"] for t in trial_list))
    mua_by_day = {
        day: np.stack([t["mua"] for t in trial_list if t["day_id"] == day], axis=1)
        for day in days
    }

    all_dfs = []
    for threshold in thresholds:
        records = []
        for day in days:
            mua_tensor = mua_by_day[day]  # (time, trial, channel)
            for region, ch_ids in channels_by_region.items():
                region_id = name_to_id[region]
                t0, t1 = region_windows[region_id]
                data = mua_tensor[t0:t1, :, ch_ids].mean(axis=0).T  # (neurons, trials)
                data -= data.mean(axis=1, keepdims=True)
                dim_k = compute_dim(data, threshold)
                records.append(dict(Day=day, Region=region, Components=dim_k))
        df = pd.DataFrame(records)
        df["Threshold"] = f"{int(threshold * 100)}%"
        all_dfs.append(df)

    full_df = pd.concat(all_dfs)

    # === PLOT ===
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(len(thresholds), 1, figsize=(10, 5 * len(thresholds)), sharex=True)

    if len(thresholds) == 1:
        axes = [axes]

    for ax, df in zip(axes, all_dfs):
        sns.barplot(data=df, x="Day", y="Components", hue="Region", errorbar=None, ax=ax)
        ax.set_title(f"Dimensionality (SVD) – {df['Threshold'].iloc[0]} Variance Explained in the windows for each region", fontsize=14)
        ax.set_ylabel("Components")
        ax.legend(title="Region", loc="upper right")

    fig.suptitle(f"{monkey}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = runtime.get_cfg().get_plot_dir() / "dimensionality_svd_windows_by_roi.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[📊] Saved SVD dimensionality plot to: {out_path}")

def plot_dimensionality_svd_windows_first_100ms(thresholds: list = [0.90, 0.95]):
    # Load from class
    mapping = runtime.get_cfg().get_mapping()
    rois_logical = runtime.get_cfg().get_rois()
    trial_list = runtime.get_cfg()._load_trials()
    region_names = runtime.get_consts().REGION_ID_TO_NAME
    region_windows = runtime.get_consts().REGION_WINDOWS
    monkey = runtime.get_cfg().get_monkey_name()

    # Reverse map: 'V1' → 1
    name_to_id = {v: k for k, v in region_names.items()}

    # Channels by region
    channels_by_region = {
        region: np.where(rois_logical == name_to_id[region])[0]
        for region in region_names.values()
    }

    # Helper
    def compute_dim(X: np.ndarray, threshold: float) -> int:
        s = svd(X, compute_uv=False)
        var_exp = np.cumsum(s**2) / np.sum(s**2)
        return int(np.searchsorted(var_exp, threshold) + 1)

    # Group by day
    days = sorted(set(t["day_id"] for t in trial_list))
    mua_by_day = {
        day: np.stack([t["mua"] for t in trial_list if t["day_id"] == day], axis=1)
        for day in days
    }

    all_dfs = []
    for threshold in thresholds:
        records = []
        for day in days:
            mua_tensor = mua_by_day[day]  # (time, trial, channel)
            for region, ch_ids in channels_by_region.items():
                t0, t1 = 0, 100  # <-- Fixed window: first 100 ms
                data = mua_tensor[t0:t1, :, ch_ids].mean(axis=0).T  # (neurons, trials)
                data -= data.mean(axis=1, keepdims=True)
                dim_k = compute_dim(data, threshold)
                records.append(dict(Day=day, Region=region, Components=dim_k))
        df = pd.DataFrame(records)
        df["Threshold"] = f"{int(threshold * 100)}%"
        all_dfs.append(df)

    full_df = pd.concat(all_dfs)

    # === PLOT ===
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(len(thresholds), 1, figsize=(10, 5 * len(thresholds)), sharex=True)

    if len(thresholds) == 1:
        axes = [axes]

    for ax, df in zip(axes, all_dfs):
        sns.barplot(data=df, x="Day", y="Components", hue="Region", errorbar=None, ax=ax)
        ax.set_title(f"Dimensionality (SVD) – {df['Threshold'].iloc[0]} Variance Explained in first 100ms", fontsize=14)
        ax.set_ylabel("Components")
        ax.legend(title="Region", loc="upper right")

    fig.suptitle(f"{monkey}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = runtime.get_cfg().get_plot_dir() / "dimensionality_svd_windows_first_100ms.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[📊] Saved SVD dimensionality plot to: {out_path}")
