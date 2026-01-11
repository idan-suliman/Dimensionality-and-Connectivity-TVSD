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
from . import constants

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
    

    