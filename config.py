# config.py
"""
Configuration helper for the TVSD project: centralizes path resolution, ROI vector selection, and array mapping load.
Provides a drop-in `configure(monkey_name, zscore_index)` that mirrors the legacy TVSDAnalysis.configure behavior.
It also exposes helpers to get/create common output directories (PLOTS, RRR variants).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence
import functools
import numpy as np
from scipy.io import loadmat
import constants

# ===== Internal utils (cached I/O) =====

@functools.lru_cache(maxsize=None)
def _load_mapping(monkey_name: str) -> np.ndarray:
    """Load the per-monkey 1024-channel mapping (MATLAB -> Python 0-based), cached by monkey name."""
    mat_path = constants.BASE_DIR / monkey_name / constants.LOGICAL_TO_PHYSICAL_MAPPING_FILENAME
    mat = loadmat(mat_path)
    mapping = mat["mapping"].flatten() - 1  # MATLAB (1-based) to Python (0-based)
    return mapping


# ===== Public API =====

class CONFIG:
    """Stateless configuration facade wrapping validation, path resolution, and directory helpers."""

    def __init__(self, monkey_name: str, zscore_index: int):
        """
        Validate inputs, resolve data_path, pick the ROI vector, load the mapping,
        and return a TVSDRuntime object. Exactly mirrors legacy side-effects, but
        without mutating external classes here.
        """
        # Validate monkey and Z-Score selection
        if monkey_name not in constants.ROIS_PHYSICAL_BY_MONKEY:
            raise ValueError(f"Monkey name '{monkey_name}' is not recognized.")
        if zscore_index not in constants.ZSCORE_INFO:
            raise ValueError(f"Z-score index '{zscore_index}' is not valid (choose 1–4).")

        # Resolve data path: <BASE_DIR>/<monkey>/<zscore_folder>
        z_folder = constants.ZSCORE_INFO[zscore_index][1]
        data_path = constants.BASE_DIR / monkey_name / z_folder
        data_path.mkdir(parents=True, exist_ok=True)

        # ROI vector and mapping
        mapping = _load_mapping(monkey_name)

        # Build runtime snapshot
        self.monkey_name=monkey_name
        self.zscore_code=zscore_index
        self.data_path=data_path
        self.need_mapping= True if monkey_name == constants.MONKEY_F else False
        self.mapping=mapping

        # tmp
        self.main_data_file_path= self.data_path / constants.MAIN_DATA_FILES[self.zscore_code]

        # User-friendly confirmation (kept to mimic legacy UX)
        print(f"[✓] TVSD configured: {monkey_name}, Z-Score = '{constants.ZSCORE_INFO[zscore_index][0]}'")

    # ---- getters ----

    def get_monkey_name(self):
        return self.monkey_name

    def get_mapping(self):
        return self.mapping
    
    def get_data_path(self):
        return self.data_path
    
    def get_zscore_title(self):
        return constants.ZSCORE_INFO[self.zscore_code][0]
    
    def get_zscore_code(self):
        return self.zscore_code
    
    def get_main_data_file_path(self):
        return self.main_data_file_path


    # ---- utilities ----
    def get_rois(self):
        rois = constants.ROIS_PHYSICAL_BY_MONKEY[self.monkey_name]
        if self.need_mapping:
            return rois[self.mapping]
        else:
            return rois

    # ------ Output directory helpers (create-on-demand) ------

    def get_plot_dir(self):
        """Return <data_path>/PLOTS; create if requested."""
        p = self.data_path / constants.DIR_PLOTS
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_target_rrr_dir(self, create: bool = True) -> Path:
        """Return <data_path>/TARGET_RRR; create if requested."""
        p = self.data_path / constants.DIR_TARGET_RRR
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def get_regular_rrr_dir(self, create: bool = True) -> Path:
        """Return <data_path>/REGULAR_RRR; create if requested."""
        p = self.data_path / constants.DIR_REGULAR_RRR
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def get_compare_rrr_dir(self, create: bool = True) -> Path:
        """Return <data_path>/COMPARE_RRR; create if requested."""
        p = self.data_path / constants.DIR_COMPARE_RRR
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def all_output_dirs(self, create: bool = True) -> Dict[str, Path]:
        """Return a dict of all common output directories under the active data path."""
        return {
            "plots": self.get_plot_dir(create=create),
            "target_rrr": self.get_target_rrr_dir(create=create),
            "regular_rrr": self.get_regular_rrr_dir(create=create),
            "compare_rrr": self.get_compare_rrr_dir(create=create),
        }
    
    def _load_trials(self):
        p = self.get_main_data_file_path()
        if not p.exists():
            raise FileNotFoundError(f"[!]: data file not found: {p}")
        import pickle
        with open(p, "rb") as f:
            trials = pickle.load(f)
        if trials is None:
            raise RuntimeError("Trials cache empty – run TVSDAnalysis.load_trials()")
        return trials

    def build_trial_matrix(
    self,
    *,
    region_id: int,
    analysis_type: str = "window",
    trials: list[int] | np.ndarray | slice | None = None,   # indices / boolean mask / slice only
    electrode_indices: np.ndarray | list[int] | None = None, # physical electrode indices to keep (optional)
    return_stimulus_ids: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Build (n_trials, n_electrodes) float32 by averaging a time window per trial×electrode.

        Design:
        • ROI membership comes from self.get_rois() (no external ROIs arg).
        • Aggregate first over ALL trials and ALL electrodes of the region.
        • If 'residual' – subtract per-stimulus means (computed on the full set) before any subselect.
        • Subselect trials/electrodes only after aggregation.
        • Stimulus IDs are ints in [0..99] under key 'stimulus_id'.
        • Trials subset is indices / boolean mask / slice ONLY (no list of dicts).
        """
        at = (analysis_type or "window").lower().strip()

        # Full trial universe (aggregation/residuals always use all trials).
        trial_universe = self._load_trials()
        N = len(trial_universe)

        # Region electrodes (physical indices) derived internally.
        region_rois = np.flatnonzero(self.get_rois() == int(region_id))

        # Optional output electrode subselect (performed after aggregation).
        # If none are provided or none match, keep all region electrodes.
        if electrode_indices is None:
            column_selector = None
        else:
            req = np.asarray(electrode_indices, dtype=int).ravel()
            pos = {int(e): i for i, e in enumerate(region_rois.tolist())}
            cols = [pos[e] for e in req if e in pos]
            column_selector = np.asarray(cols, dtype=int) if len(cols) else None

        # Time window: baseline100 → [0:100), else region-defined window (inclusive start, exclusive end).
        if at == "baseline100":
            win = slice(0, 100)
        else:
            start, end = constants.REGION_WINDOWS[int(region_id)]
            win = slice(start, end)

        # Aggregate (mean over time) for ALL region electrodes across ALL trials.
        def _time_mean_matrix(trial_seq: Sequence[dict]) -> np.ndarray:
            return np.stack(
                [np.asarray(tr["mua"], dtype=np.float64)[win][:, region_rois].mean(0) for tr in trial_seq],
                dtype=np.float32,
            )

        full_mat = _time_mean_matrix(trial_universe)

        # Stimulus IDs for the full universe (fixed key 'stimulus_id'; 100 stimuli total: 0..99).
        stim_ids_full = None
        if return_stimulus_ids or at == "residual":
            stim_ids_full = np.array([tr["stimulus_id"] for tr in trial_universe], dtype=int)

        # Residual subtraction on FULL region before any subselect.
        if at == "residual":
            n_elec = full_mat.shape[1]
            n_stim = 100  # stimuli are integers in [0..99]
            means = np.zeros((n_stim, n_elec), dtype=full_mat.dtype)
            for sid in range(n_stim):
                m = (stim_ids_full == sid)
                if np.any(m):
                    means[sid] = full_mat[m].mean(0)
            full_mat = full_mat - means[np.clip(stim_ids_full, 0, n_stim - 1)]

        # ---- Subselect trials (indices / boolean mask / slice); default = all ----
        if trials is None:
            out_idx = np.arange(N, dtype=int)
        elif isinstance(trials, slice):
            out_idx = np.arange(N, dtype=int)[trials]
        else:
            arr = np.asarray(trials)
            if arr.dtype == bool:
                out_idx = np.flatnonzero(arr[:N])
            else:
                idx = np.asarray(arr, dtype=int).ravel()
                out_idx = idx[(idx >= 0) & (idx < N)]

        # Apply trial/electrode subselect to the already-aggregated matrix.
        mat = full_mat[out_idx]
        if column_selector is not None:
            mat = mat[:, column_selector]

        if return_stimulus_ids:
            return mat, (stim_ids_full[out_idx] if stim_ids_full is not None else None)
        return mat
    