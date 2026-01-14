from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple, List, Any

def get_file_path(analyzer, output_dir: str, region_id: int | None = None, src_tgt: Tuple[int, int] | None = None, suffix: str = "_dim_corr") -> Any:
    """
    Generate consistent filename for saving/loading.
    Delegates to rep_stab but appends suffix.
    """
    # Create temp instance to borrow logic if needed, or just append suffix to what rep_stab would generate
    # But rep_stab.get_file_path() returns a Path.
    # We can ask rep_stab for the base path and modify it.
    
    base_path = analyzer.rep_stab.get_file_path(output_dir, region_id, src_tgt)
    # base_path is like .../Region_V1_rep_stab.npz
    # We want .../Region_V1_rep_stab_dim_corr.npz
    
    stem = base_path.stem
    parent = base_path.parent
    new_name = f"{stem}{suffix}.npz"
    return parent / new_name

def compute_overlap_matrix(analyzer, bases: List[np.ndarray]) -> np.ndarray:
    """
    Computes the overlap matrix between all pairs of bases.
    bases: List of (Features x D) arrays.
    """
    n = len(bases)
    O = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            # Use rep_stab's MSC overlap metric
            val = analyzer.rep_stab._compute_overlap_msc(bases[i], bases[j])
            O[i, j] = val
            O[j, i] = val
    return O
