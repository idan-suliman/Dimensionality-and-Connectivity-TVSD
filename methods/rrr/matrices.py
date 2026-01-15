from __future__ import annotations
import numpy as np
from typing import Optional, Sequence
from core.runtime import runtime
from ..matchingSubset import match_and_save

def build_mats(src_region: int, tgt_region: int, analysis_type: str, *, match_to_target: bool, trials: Optional[Sequence[int]] = None):
    """
    Construct X (source) and Y (target) matrices for analysis.
    If match_to_target is True, loads/computes the V1-subset that matches Target firing rates.
    """
    rois   = runtime.cfg.get_rois()
    src_name = runtime.consts.REGION_ID_TO_NAME[src_region]
    tgt_name = runtime.consts.REGION_ID_TO_NAME[tgt_region]

    def build(reg_id, idx):
        return runtime.data_manager.build_trial_matrix(
            region_id=reg_id,
            analysis_type=analysis_type,
            trials=trials,
            electrode_indices=np.asarray(idx, dtype=int),
            return_stimulus_ids=False
        )

    src_idx_full = np.where(rois == src_region)[0]
    tgt_idx_full = np.where(rois == tgt_region)[0]

    if match_to_target:
        # --- Load Matching Subset ---
        dir_match = (runtime.cfg.get_data_path() / "TARGET_RRR" / analysis_type.upper())
        dir_match.mkdir(parents=True, exist_ok=True)
        subset_f = dir_match / f"{src_name}_to_{tgt_name}_{analysis_type}.npz"
        
        if not subset_f.exists():
            match_and_save(
                src_name, tgt_name,
                stat_mode=analysis_type,
                show_plot=False, verbose=False)

        with np.load(subset_f) as z:
            match_idx = z["phys_idx"]          # V1-MATCH 

        remain_mask    = ~np.isin(src_idx_full, match_idx)
        src_idx_remain = src_idx_full[remain_mask]
        
        if src_idx_remain.size == 0:
            raise ValueError("All V1 electrodes were matched; nothing left in X!")

        X = build(src_region, src_idx_remain)  # V1 minus V1-MATCH
        Y = build(src_region, match_idx)       # V1-MATCH (acting as Target)
    else:
        X = build(src_region, src_idx_full)    # V1 full
        Y = build(tgt_region, tgt_idx_full)    # V4 / IT full

    return Y, X
