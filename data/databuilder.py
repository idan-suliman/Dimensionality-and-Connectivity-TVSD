# databuilder.py
from core.runtime import runtime
import numpy as np
import pickle
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

class DataBuilder:
    @staticmethod
    def build_if_missing():
        cfg = runtime.get_cfg()
        file_path = runtime.get_cfg().get_main_data_file_path()
        if file_path.exists():
            print(f"[✓] Data file already exists: {file_path.name}")
            return

        print(f"[!] Data file not found: {file_path.name}")
        print(f"[∴] Building data for: {cfg.get_zscore_title()}")
        zcode = cfg.get_zscore_code()
        if zcode == 2:
            raw_data = DataBuilder.load_raw_data()
            DataBuilder.build_original_zscore(raw_data)
        elif zcode == 3:
            raw_data = DataBuilder.load_raw_data()
            DataBuilder.build_global_zscore(raw_data)
        elif zcode == 4:
            raw_data = DataBuilder.load_raw_data()
            DataBuilder.build_repetition_zscore(raw_data)
        else:
            raise ValueError("Only Z-score types 2–4 are buildable dynamically.")

    @staticmethod
    def load_raw_data():
        """Loads the raw test trials from Z=1"""
        raw_path = runtime.get_consts().BASE_DIR / runtime.get_cfg().get_monkey_name() / "1_without_z_score" / "test_trials_full_data.pkl"

        if not raw_path.exists():
            raise FileNotFoundError(f"Expected raw file not found: {raw_path}")

        print(f"[📂] Loading raw data from: {raw_path.name}")
        with open(raw_path, "rb") as f:
            return pickle.load(f)


    @staticmethod
    def build_original_zscore(raw_data):
        """Build Z-score per electrode using region-specific time windows, fully via CONFIG."""
        print("[🔧] Running original Z-score normalization...")

        cfg = runtime.get_cfg()
        consts = runtime.get_consts()

        region_map     = consts.REGION_ID_TO_NAME
        region_windows = consts.REGION_WINDOWS
        rois   = cfg.get_rois()

        output_pkl = cfg.get_main_data_file_path()
        output_csv = output_pkl.with_name("original_z_score_stats.csv")

        # --- group trials by day ---
        day_groups = defaultdict(list)
        for tr in raw_data:
            day_groups[tr["day_id"]].append(tr)

        stats = []
        from tqdm import tqdm
        for day, trials in tqdm(day_groups.items(), desc="Normalizing days"):
            data = np.stack([t["mua"] for t in trials], axis=1)  # (300, n_trials, 1024)
            zdata = np.empty_like(data, dtype=np.float32)

            # --- normalize each electrode using its region-specific window ---
            for ch in range(data.shape[2]):
                region_id = rois[ch]
                start, end = region_windows[region_id]
                vals = data[start:end, :, ch].reshape(-1)
                mu, std = vals.mean(), vals.std()
                std = 1.0 if std == 0 else std

                zdata[:, :, ch] = (data[:, :, ch] - mu) / std
                stats.append({
                    "day": day,
                    "electrode": ch,
                    "region": region_map[region_id],
                    "mean": mu,
                    "std": std
                })

            # --- update trials ---
            for i, tr in enumerate(trials):
                tr["mua"] = zdata[:, i, :]

        # --- save normalized data and stats ---
        import pickle, pandas as pd
        with open(output_pkl, "wb") as f:
            pickle.dump(raw_data, f)
        pd.DataFrame(stats).to_csv(output_csv, index=False)

        print(f"[💾] Saved normalized data → {output_pkl.name}")
        print(f"[📊] Saved Z-score stats → {output_csv.name}")


    @staticmethod
    def build_global_zscore(raw_data):
        """Global Z-score normalization across all days using CONFIG (no manual paths or mapping)."""

        print("[🔧] Running global Z-score normalization across all days...")

        cfg     = runtime.get_cfg()
        consts  = runtime.get_consts()

        region_map     = consts.REGION_ID_TO_NAME
        region_windows = consts.REGION_WINDOWS
        rois   = cfg.get_rois()

        output_pkl = cfg.get_main_data_file_path()
        output_csv = output_pkl.with_name("zscore_stats_by_electrode_over_all_days.csv")

        # --- stack all trials into one tensor: (time=300, n_trials, n_channels=1024)
        data = np.stack([tr["mua"] for tr in raw_data], axis=1).astype(np.float32)
        zdata = np.empty_like(data, dtype=np.float32)

        stats = []
        print("🔁 Normalizing each electrode using its region-specific response window:")
        from tqdm import tqdm
        for ch in tqdm(range(data.shape[2])):
            region_id = rois[ch]
            start, end = region_windows[region_id]
            vals = data[start:end, :, ch].reshape(-1)
            mu, std = vals.mean(), vals.std()
            std = 1.0 if std == 0 else std

            zdata[:, :, ch] = (data[:, :, ch] - mu) / std
            stats.append({
                "electrode": ch,
                "region": region_map[region_id],
                "mean": mu,
                "std": std
            })

        # --- update trials with normalized data
        for i, tr in enumerate(raw_data):
            tr["mua"] = zdata[:, i, :]

        # --- save outputs
        import pickle, pandas as pd
        with open(output_pkl, "wb") as f:
            pickle.dump(raw_data, f)
        pd.DataFrame(stats).to_csv(output_csv, index=False)

        print(f"[💾] Saved normalized data → {output_pkl.name}")
        print(f"[📊] Saved electrode-level stats → {output_csv.name}")


    @staticmethod
    def build_repetition_zscore(raw_data):
        """Repetition-wise Z-score normalization using CONFIG (per electrode × repetition)."""
        print("[🔧] Running repetition-wise Z-score normalization...")

        cfg     = runtime.get_cfg()
        consts  = runtime.get_consts()

        region_map     = consts.REGION_ID_TO_NAME
        region_windows = consts.REGION_WINDOWS
        rois   = cfg.get_rois()

        output_pkl = cfg.get_main_data_file_path()
        output_csv = output_pkl.with_name("zscore_stats_by_electrode_repeat.csv")

        # --- group trials by repetition (0-based)
        NUM_REPS = consts.NUM_REPETITIONS
        rep_groups = {r: [] for r in range(NUM_REPS)}
        for tr in raw_data:
            rep = int(tr["allmat_row"][3]) - 1
            rep_groups[rep].append(tr)

        stats = []
        print("🔁 Normalizing by electrode × repetition using region-specific window:")
        from tqdm import tqdm
        for rep, trials in tqdm(rep_groups.items()):
            if not trials:
                continue

            data = np.stack([t["mua"] for t in trials], axis=1).astype(np.float32)  # (300, n_trials, 1024)
            zdata = np.empty_like(data, dtype=np.float32)

            for ch in range(data.shape[2]):
                region_id = rois[ch]
                start, end = region_windows[region_id]
                vals = data[start:end, :, ch].reshape(-1)
                mu, std = vals.mean(), vals.std()
                std = 1.0 if std == 0 else std

                zdata[:, :, ch] = (data[:, :, ch] - mu) / std
                stats.append({
                    "repetition": rep,
                    "electrode": ch,
                    "region": region_map[region_id],
                    "mean": mu,
                    "std": std
                })

            # --- update normalized MUA per trial
            for i, tr in enumerate(trials):
                tr["mua"] = zdata[:, i, :]

        # --- save updated trial list + stats
        import pickle, pandas as pd
        with open(output_pkl, "wb") as f:
            pickle.dump(raw_data, f)
        pd.DataFrame(stats).to_csv(output_csv, index=False)

        print(f"[💾] Saved repetition-wise normalized data → {output_pkl.name}")
        print(f"[📊] Saved per-electrode repetition stats → {output_csv.name}")
