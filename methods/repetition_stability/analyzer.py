from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from core.runtime import runtime
from . import utils
from . import region
from . import connection
from . import pipeline
# Local imports for stats helper
from .stats import compute_lag_stats, compute_overlap_msc

class RepetitionStabilityAnalyzer:
    """
    Consolidated module for "Repetition Stability Analysis".
    Analyze the stability of neural subspaces (within regions) and predictive subspaces (between regions)
    across repeated presentations of the same stimuli.
    """
    
    def __init__(self, monkey: str, z_code: int, analysis_type: str, group_size: int = 3):
        """
        Args:
            monkey: e.g., "Monkey N"
            z_code: Z-score index (1-4)
            analysis_type: "residual" or "window" (or others)
            group_size: Number of repetitions to group into one block (default 3).
        """
        self.monkey = monkey
        self.z_code = z_code
        self.analysis_type = analysis_type
        self.group_size = group_size
        
        # Configure runtime
        runtime.set_cfg(self.monkey, self.z_code)
        self.dm = runtime.get_data_manager()
        self.cfg = runtime.get_cfg()

    def get_blocked_data(self, region_id: int) -> List[np.ndarray]:
        return utils.get_blocked_data(self, region_id)

    # Exposed for external use or subclass use
    def _compute_overlap_msc(self, A: np.ndarray, B: np.ndarray) -> float:
         return compute_overlap_msc(A, B)

    # Exposed for external use or subclass use
    def _compute_lag_stats(self, O: np.ndarray, n_perms: int = 2000):
         return compute_lag_stats(self, O, n_perms)
         
    # Exposed for external use
    def _get_subfolder_name(self):
         # Helper to get subfolder name like in DimCorr
         return f"Repetition_Stability"

    def analyze_region(self, region_id: int, fixed_d: int | None = None) -> Dict[str, Any]:
        return region.analyze_region(self, region_id, fixed_d)

    def analyze_connection(self, src_id: int, tgt_id: int, fixed_d: int | None = None) -> Dict[str, Any]:
        return connection.analyze_connection(self, src_id, tgt_id, fixed_d)
        
    def save_results(self, result: Dict[str, Any], output_dir: str, fpath=None):
        utils.save_results(self, result, output_dir, fpath)

    def get_file_path(self, output_dir: str, region_id: int | None = None, src_tgt: Any | None = None, suffix: str = ""):
        return utils.get_file_path(self, output_dir, region_id, src_tgt, suffix)

    def run_pipeline(self, force_recompute: bool = False):
        pipeline.run_pipeline(self, force_recompute)
