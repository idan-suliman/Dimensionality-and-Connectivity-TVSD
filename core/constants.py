# constants.py
"""
Centralized constants for the TVSD analysis project: 
paths, dataset structure, Z-score modes, region windows, ROI maps, plot directories, colors, and default RRR/CV settings.
"""
from __future__ import annotations
from pathlib import Path
from typing import Final, Dict, Tuple
import numpy as np

MONKEY_N = "Monkey N"
MONKEY_F = "Monkey F"
MONKEYS = [MONKEY_N, MONKEY_F]

# ========= Base paths and filenames =========
# You can override the base directory using the TVSD_BASE_DIR environment variable.
BASE_DIR: Final[Path] = Path("C:/Users/User/Desktop/TVSD")
LOGICAL_TO_PHYSICAL_MAPPING_FILENAME: Final[str] = "1024chns_mapping_20220105.mat"

# ========= Dataset meta =========
NUM_CHANNELS: Final[int] = 1024              # Total electrodes in the array
NUM_REPETITIONS: Final[int] = 30             # Repetitions per stimulus

# ========= Z-Score modes and folder names =========
# Map: code -> (human-readable name, folder name)
ZSCORE_INFO: Dict[int, Tuple[str, str]] = {
    1: ("No Z-Score",                         "1_without_z_score"),
    2: ("Original Z-Score",                   "2_original_z_score"),
    3: ("Z-Score per Electrode (global)",     "3_z_score_per_electrode_over_all_days"),
    4: ("Z-Score per Electrode + Repetition", "4_z_score_per_electrode_and_repetition"),
}

# Main data files per Z-Score mode (used by TVSDAnalysis.get_main_data_file)
MAIN_DATA_FILES: Dict[int, str] = {
    1: "test_trials_full_data.pkl",
    2: "test_trials_zscore_original.pkl",
    3: "test_trials_zscore_global.pkl",
    4: "test_trials_zscore_per_repetition.pkl",
}

# ========= Regions =========
# Region ID mapping: 1=V1, 2=V4, 3=IT
REGION_ID_TO_NAME: Dict[int, str] = {1: "V1", 2: "V4", 3: "IT"}
REGION_NAME_TO_ID: Dict[str, int] = {v: k for k, v in REGION_ID_TO_NAME.items()}

# Time windows (inclusive start, exclusive end)
REGION_WINDOWS: Dict[int, Tuple[int, int]] = {
    1: (125, 225),  # V1
    2: (150, 250),  # V4
    3: (175, 275),  # IT
}

# ========= Physical ROI maps (length 1024) by monkey =========
# Values: 1=V1, 2=V4, 3=IT
def build_rois_from_physical_by_monkey() -> Dict[str, np.ndarray]:
    def physical_rois_monkey_n() -> Dict[str, np.ndarray]:
        rois = np.zeros(NUM_CHANNELS, dtype=int)
        rois[:512]    = 1  # V1
        rois[512:768] = 2  # V4
        rois[768:]    = 3  # IT
        return rois[64:]   # drop 0-63, total 64

    def physical_rois_monkey_f() -> Dict[str, np.ndarray]:
        rois = np.zeros(NUM_CHANNELS, dtype=int)
        rois[:512]    = 1  # V1
        rois[512:832] = 3  # IT
        rois[832:]    = 2  # V4
        return rois

    return {
        MONKEY_F: physical_rois_monkey_f(),
        MONKEY_N: physical_rois_monkey_n(),
    }

ROIS_PHYSICAL_BY_MONKEY = build_rois_from_physical_by_monkey()


# ========= Common output directories =========
DIR_PLOTS: Final[str]       = "PLOTS"
DIR_TARGET_RRR: Final[str]  = "TARGET_RRR"
DIR_REGULAR_RRR: Final[str] = "REGULAR_RRR"
DIR_COMPARE_RRR: Final[str] = "COMPARE_RRR"

# ========= Analysis types =========
ANALYSIS_TYPES: Tuple[str, ...] = ("window", "baseline100", "residual")



# Default Analysis Parameters
DEFAULT_CV_OUTER_SPLITS = 3
DEFAULT_CV_INNER_SPLITS = 3
DEFAULT_N_PERMS = 2500
