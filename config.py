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
        trials: list[dict] | None = None,
        rois: np.ndarray | None = None,
        stimulus_key: str = "stimulus_id",
        return_stimulus_ids: bool = False,
        residual_reference_trials: list[dict] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Compress the time dimension of each trial into a single value per electrode
        according to the requested analysis type.

        Parameters
        ----------
        region_id:
            Numerical region identifier (1=V1, 2=V4, 3=IT). Used to pick the
            region-specific window for "window" / "residual" aggregation.
        analysis_type:
            One of constants.ANALYSIS_TYPES – controls the time selection and
            whether stimulus-wise residual subtraction is applied.
        trials:
            Optional explicit list of trial dictionaries. Falls back to the
            cached trials loaded via `_load_trials()` when omitted.
        electrode_indices:
            Optional array of electrode indices in physical ordering. When not
            provided the electrodes belonging to `region_id` are selected from
            the ROI vector.
        stimulus_key:
            Trial dict key that identifies the stimulus. Required when
            `analysis_type == "residual"` or when `return_stimulus_ids` is True.
        return_stimulus_ids:
            If True, also return the 1D array of stimulus identifiers aligned
            with the rows of the output matrix.

        Returns
        -------
        matrix : np.ndarray
            Shape = (n_trials, n_electrodes) with dtype float32, containing the
            mean activity per electrode in the chosen time window.
        stimulus_ids : np.ndarray
            Only returned when `return_stimulus_ids` is True.
        """

        at = analysis_type.lower().strip()
        if at not in constants.ANALYSIS_TYPES:
            raise ValueError(
                f"analysis_type must be one of {constants.ANALYSIS_TYPES}, got '{analysis_type}'."
            )

        trials_list = self._load_trials() if trials is None else list(trials)
        if len(trials_list) == 0:
            raise ValueError("No trials available to build the matrix.")

        # Determine electrodes for the requested region when not explicitly provided.
        if rois is None:
            rois = self.get_rois()
            rois = np.flatnonzero(rois == int(region_id))
        else:
            rois = np.asarray(rois, dtype=int)

        if rois.size == 0:
            raise ValueError(f"No electrodes found for region id {region_id}.")

        # Choose time window: BASELINE100 → first 100 ms, otherwise region-specific window.
        if at == "baseline100":
            win = slice(0, 100)
        else:
            try:
                start, end = constants.REGION_WINDOWS[int(region_id)]
            except KeyError as exc:
                raise KeyError(f"Region id {region_id} is not defined in REGION_WINDOWS.") from exc
            win = slice(start, end)

        def _time_mean_matrix(trial_seq: list[dict]) -> np.ndarray:
            return np.stack(
                [
                    np.asarray(tr["mua"], dtype=np.float64)[win][:, rois].mean(0)
                    for tr in trial_seq
                ],
                dtype=np.float32,
            )

        # Mean activity within the chosen window for every trial/electrode.
        mat = _time_mean_matrix(trials_list)

        stim_ids = None
        if return_stimulus_ids or at == "residual":
            try:
                stim_ids = np.array([tr[stimulus_key] for tr in trials_list])
            except KeyError as exc:
                raise KeyError(
                    f"Trials must contain '{stimulus_key}' for analysis_type '{analysis_type}'."
                ) from exc

        if at == "residual":
            ref_trials = trials_list if residual_reference_trials is None else list(residual_reference_trials)
            if len(ref_trials) == 0:
                raise ValueError("Residual analysis requires non-empty reference trials.")

            ref_mat = mat if ref_trials is trials_list else _time_mean_matrix(ref_trials)

            try:
                ref_stim_ids = np.array([tr[stimulus_key] for tr in ref_trials])
            except KeyError as exc:
                raise KeyError(
                    f"Reference trials must contain '{stimulus_key}' for residual analysis."
                ) from exc

            stim_means = {}
            for sid in np.unique(ref_stim_ids):
                stim_means[sid] = ref_mat[ref_stim_ids == sid].mean(0)

            try:
                mat -= np.stack([stim_means[sid] for sid in stim_ids], axis=0)
            except KeyError as exc:
                raise KeyError(
                    "Residual reference trials do not cover all stimulus ids present in the target trials."
                ) from exc

        if return_stimulus_ids:
            return mat, stim_ids
        return mat

