from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple, List, Any

from core.runtime import runtime




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
