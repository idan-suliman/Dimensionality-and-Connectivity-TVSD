"""
data_manager.py
----------------
Refactored Data Logic.
Separates 'Settings' (CONFIG) from 'Data Loading' (DataManager).
"""
from __future__ import annotations
import numpy as np
from core import constants
from typing import Sequence, Optional, Dict, Any, Tuple, List
import functools

class DataManager:
    """
    Handles data loading, caching, and matrix construction.
    Wraps a CONFIG object to access paths and ROI settings.
    """
    def __init__(self, config):
        self.config = config
        self._trials_cache = None
        # Cache for built matrices: Key -> (matrix, stim_ids or None)
        # Key: (region_id, analysis_type, trials_hash, electrode_hash, return_stimulus_ids)
        self._matrix_cache: Dict[Any, Any] = {}

    def _load_trials(self):
        """Load trials from pickle, cached in memory."""
        if self._trials_cache is not None:
            return self._trials_cache

        p = self.config.get_main_data_file_path()
        if not p.exists():
            print(f"[!] Data file not found: {p}")
            print("[*] Attempting to auto-build data via DataBuilder...")
            try:
                from .databuilder import DataBuilder
                DataBuilder.build_if_missing()
            except Exception as e:
                print(f"[!] Auto-build failed: {e}")

        if not p.exists():
            raise FileNotFoundError(f"[!]: data file not found: {p}")
        import pickle
        with open(p, "rb") as f:
            trials = pickle.load(f)
        if trials is None:
            raise RuntimeError("Trials cache empty – run TVSDAnalysis.load_trials()")
        
        self._trials_cache = trials
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
        
        DELEGATED from CONFIG: Uses self.config for ROIs and Paths.
        
        Design:
        • ROI membership comes from self.config.get_rois() (no external ROIs arg).
        • Aggregate first over ALL trials and ALL electrodes of the region.
        • If 'residual' – subtract per-stimulus means (computed on the full set) before any subselect.
        • Subselect trials/electrodes only after aggregation.
        • Stimulus IDs are ints in [0..99] under key 'stimulus_id'.
        • Trials subset is indices / boolean mask / slice ONLY (no list of dicts).
        """
        # --- Caching Key Gen ---
        # We try to create a hashable key for the cache.
        # Slices are not hashable, lists/arrays aren't. We allow modest caching for common cases (None).
        use_cache = False
        cache_key = None
        
        # Only cache if inputs are simple (None or hashable). 
        # For complex slicing logic, we might skip caching or implement complex hashing.
        # For now, we adopt a simple policy: Cache if 'trials' is None and 'electrode_indices' is None.
        if trials is None and electrode_indices is None:
             use_cache = True
             cache_key = (region_id, analysis_type, "all_trials", "all_electrodes", return_stimulus_ids)
             if cache_key in self._matrix_cache:
                 # print(f"[DEBUG] Cache Hit: {cache_key}")
                 return self._matrix_cache[cache_key]

        at = (analysis_type or "window").lower().strip()

        # Full trial universe (aggregation/residuals always use all trials).
        trial_universe = self._load_trials()
        N = len(trial_universe)

        # Region electrodes (physical indices) derived internally.
        region_rois = np.flatnonzero(self.config.get_rois() == int(region_id))

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
            res = (mat, (stim_ids_full[out_idx] if stim_ids_full is not None else None))
        else:
            res = mat
            
        # Store in cache
        if use_cache and cache_key is not None:
            self._matrix_cache[cache_key] = res
            
        return res

    def get_repetition_matrices(
        self,
        region_id: int,
        analysis_type: str = "window",
        trials: list[int] | np.ndarray | slice | None = None,
        electrode_indices: np.ndarray | list[int] | None = None,
    ) -> list[np.ndarray]:
        """
        Builds the trial matrix and splits it into a list of matrices, one per repetition.
        Assumes trials correspond to 30 repetitions of 100 stimuli each (Total=3000).
        Returns a list of 30 matrices, each (100, n_electrodes).
        
        Note: logic derived from previous 'kyle_method' implementation using 'rep_idx'.
        """
        # 1. Get the full matrix (all trials) plus metadata to identify repetitions
        # We force 'return_stimulus_ids=True' to help sorting if needed, and 'trials=None' for full
        full_mat, stim_ids = self.build_trial_matrix(
            region_id=region_id,
            analysis_type=analysis_type,
            trials=None,
            electrode_indices=electrode_indices,
            return_stimulus_ids=True
        )
        
        # 2. Get Trial Metadata (Repetition Index)
        # We need to access the raw trials to get 'rep_idx'
        trials_data = self._load_trials()
        
        rep_indices = np.array(
            [ (tr.get("rep_idx") if tr.get("rep_idx") is not None else int(tr["allmat_row"][3]) - 1)
              for tr in trials_data ],
            dtype=int
        )
        
        # 3. Split by Repetition
        # Expecting 30 repetitions (0..29)
        mats = []
        n_reps = constants.NUM_REPETITIONS # Should be 30
        
        for r in range(n_reps):
            ridx = np.flatnonzero(rep_indices == r)
            if ridx.size == 0:
                # If a repetition is missing, we might append None or raise
                # For now, consistent with previous behavior, we assume data exists.
                continue
                
            # Enforce stable image order (0..99) using stim_ids
            # stim_ids corresponds to full_mat rows
            current_stim_ids = stim_ids[ridx]
            order = np.argsort(current_stim_ids)
            
            # Slice and Order
            mats.append(full_mat[ridx[order], :])
            
        return mats
