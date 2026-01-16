from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from core.runtime import runtime
from ..repetition_stability import RepetitionStabilityAnalyzer
from . import utils
from . import curves

class DimCorrAnalyzer:
    """
    Analyzer that computes the 'Dimensionality vs Stability Correlation' curve.
    Uses RepetitionStabilityAnalyzer for data loading and core stats metrics.
    """
    def __init__(self, monkey: str, z_code: int, analysis_type: str = "residual", group_size: int = 3):
        self.monkey = monkey
        self.z_code = z_code
        self.analysis_type = analysis_type
        self.group_size = group_size
        
        # Wraps RepStab analyzer to reuse logic
        self.rep_stab = RepetitionStabilityAnalyzer(
            monkey=monkey, 
            z_code=z_code, 
            analysis_type=analysis_type, 
            group_size=group_size
        )

    def get_file_path(self, output_dir: str, region_id: int | None = None, src_tgt: Tuple[int, int] | None = None, suffix: str = "_dim_corr") -> Any:
        # Delegate to runtime.paths
        return runtime.paths.get_dim_corr_path(
            self.monkey,
            self.analysis_type,
            self.group_size,
            region_id=region_id,
            src_tgt=src_tgt,
            suffix=suffix,
            output_dir=output_dir
        )

    def _compute_overlap_matrix(self, bases: List[np.ndarray]) -> np.ndarray:
        return utils.compute_overlap_matrix(self, bases)

    def analyze_region_curve(self, region_id: int, max_dims: int = 30, 
                             precomputed_blocks: Optional[List[np.ndarray]] = None,
                             force_recompute: bool = False) -> Dict[str, Any]:
        return curves.analyze_region_curve(self, region_id, max_dims, precomputed_blocks, force_recompute)

    def analyze_connection_curve(self, src_id: int, tgt_id: int, max_dims: int = 30,
                                 src_blocks: Optional[List[np.ndarray]] = None,
                                 tgt_blocks: Optional[List[np.ndarray]] = None,
                                 force_recompute: bool = False) -> Dict[str, Any]:
        return curves.analyze_connection_curve(self, src_id, tgt_id, max_dims, src_blocks, tgt_blocks, force_recompute)
