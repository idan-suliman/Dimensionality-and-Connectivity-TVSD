from __future__ import annotations
import numpy as np
from typing import Sequence
from core.runtime import runtime

def build_trial_matrix(
    manager,
    *,
    region_id: int,
    analysis_type: str = "window",
    trials: list[int] | np.ndarray | slice | None = None,
    electrode_indices: np.ndarray | list[int] | None = None,
    return_stimulus_ids: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Build (n_trials, n_electrodes) float32 by averaging a time window per trial×electrode.
    """
    # --- Caching Key Gen ---
    use_cache = False
    cache_key = None
    
    if trials is None and electrode_indices is None:
            use_cache = True
            cache_key = (region_id, analysis_type, "all_trials", "all_electrodes", return_stimulus_ids)
            if cache_key in manager._matrix_cache:
                return manager._matrix_cache[cache_key]

    at = (analysis_type or "window").lower().strip()

    # Full trial universe
    trial_universe = manager._load_trials()
    N = len(trial_universe)

    # Region electrodes
    region_rois = np.flatnonzero(manager.config.get_rois() == int(region_id))

    # Optional output electrode subselect
    if electrode_indices is None:
        column_selector = None
    else:
        req = np.asarray(electrode_indices, dtype=int).ravel()
        pos = {int(e): i for i, e in enumerate(region_rois.tolist())}
        cols = [pos[e] for e in req if e in pos]
        column_selector = np.asarray(cols, dtype=int) if len(cols) else None

    # Time window
    if at == "baseline100":
        win = slice(0, 100)
    else:
        start, end = runtime.consts.REGION_WINDOWS[int(region_id)]
        win = slice(start, end)

    # Aggregate
    full_mat = _time_mean_matrix(trial_universe, win, region_rois)

    # Stimulus IDs
    stim_ids_full = None
    if return_stimulus_ids or at == "residual":
        stim_ids_full = np.array([tr["stimulus_id"] for tr in trial_universe], dtype=int)

    # Residual subtraction
    if at == "residual":
        n_elec = full_mat.shape[1]
        n_stim = 100
        means = np.zeros((n_stim, n_elec), dtype=full_mat.dtype)
        for sid in range(n_stim):
            m = (stim_ids_full == sid)
            if np.any(m):
                means[sid] = full_mat[m].mean(0)
        full_mat = full_mat - means[np.clip(stim_ids_full, 0, n_stim - 1)]

    # Subselect trials
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

    # Final selection
    mat = full_mat[out_idx]
    if column_selector is not None:
        mat = mat[:, column_selector]

    if return_stimulus_ids:
        res = (mat, (stim_ids_full[out_idx] if stim_ids_full is not None else None))
    else:
        res = mat
        
    # Store in cache
    if use_cache and cache_key is not None:
        manager._matrix_cache[cache_key] = res
        
    return res

def _time_mean_matrix(trial_seq: Sequence[dict], win: slice, region_rois: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.asarray(tr["mua"], dtype=np.float64)[win][:, region_rois].mean(0) for tr in trial_seq],
        dtype=np.float32,
    )
