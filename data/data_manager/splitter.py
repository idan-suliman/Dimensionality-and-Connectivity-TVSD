from __future__ import annotations
import numpy as np


def get_repetition_matrices(
    manager,
    *,
    region_id: int,
    analysis_type: str = "window",
    trials: list[int] | np.ndarray | slice | None = None,
    electrode_indices: np.ndarray | list[int] | None = None,
) -> list[np.ndarray]:
    """
    Builds the trial matrix and splits it into a list of matrices, one per repetition.
    """
    # 1. Get the full matrix (all trials) plus metadata
    full_mat, stim_ids = manager.build_trial_matrix(
        region_id=region_id,
        analysis_type=analysis_type,
        trials=None,
        electrode_indices=electrode_indices,
        return_stimulus_ids=True
    )
    
    # 2. Get Trial Metadata (Repetition Index)
    trials_data = manager._load_trials()
    
    rep_indices = np.array(
        [ (tr.get("rep_idx") if tr.get("rep_idx") is not None else int(tr["allmat_row"][3]) - 1)
          for tr in trials_data ],
        dtype=int
    )
    
    # 3. Split by Repetition
    mats = []
    from core import constants
    n_reps = constants.NUM_REPETITIONS
    
    for r in range(n_reps):
        ridx = np.flatnonzero(rep_indices == r)
        if ridx.size == 0:
            continue
            
        # Enforce stable image order (0..99) using stim_ids
        current_stim_ids = stim_ids[ridx]
        order = np.argsort(current_stim_ids)
        
        # Slice and Order
        mats.append(full_mat[ridx[order], :])
        
    return mats
