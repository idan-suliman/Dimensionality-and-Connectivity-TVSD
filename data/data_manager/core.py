from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

# Import helper modules
from . import loader
from . import builder
from . import splitter

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
        return loader.load_trials(self)

    def build_trial_matrix(
        self,
        *,
        region_id: int,
        analysis_type: str = "window",
        trials: list[int] | np.ndarray | slice | None = None,
        electrode_indices: np.ndarray | list[int] | None = None,
        return_stimulus_ids: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Build (n_trials, n_electrodes) float32 by averaging a time window per trial×electrode.
        Delegates to builder module.
        """
        return builder.build_trial_matrix(
            self,
            region_id=region_id,
            analysis_type=analysis_type,
            trials=trials,
            electrode_indices=electrode_indices,
            return_stimulus_ids=return_stimulus_ids
        )

    def get_repetition_matrices(
        self,
        region_id: int,
        analysis_type: str = "window",
        trials: list[int] | np.ndarray | slice | None = None,
        electrode_indices: np.ndarray | list[int] | None = None,
    ) -> list[np.ndarray]:
        """
        Builds the trial matrix and splits it into a list of matrices, one per repetition.
        Delegates to splitter module.
        """
        return splitter.get_repetition_matrices(
            self,
            region_id=region_id,
            analysis_type=analysis_type,
            trials=trials,
            electrode_indices=electrode_indices
        )
