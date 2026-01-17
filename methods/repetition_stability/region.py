from __future__ import annotations
import numpy as np
from typing import Dict, Any
from ..pca import RegionPCA
from .stats import compute_lag_stats, compute_overlap_msc

def analyze_region(analyzer, region_id: int, fixed_d: int | None = None) -> Dict[str, Any]:
    """
    Computes pairwise overlap between blocks for a single region.
    Method: PCA on each block -> Overlap.
    """
    blocks = analyzer.get_blocked_data(region_id)
    n_blocks = len(blocks)
    
    # 1. Compute Subspaces & Dimensionality
    subspaces = []
    dims = []
    
    for X in blocks:
        # PCA (using RegionPCA)
        pca = RegionPCA(centered=True).fit(X)
        d = pca.dimensionality
        dims.append(d)
        # Store full components, truncate later
        subspaces.append(pca.get_components()) 
        
    # 2. Determine Common D
    if fixed_d is not None:
        D_common = fixed_d
    else:
        D_common = max(1, int(np.floor(np.mean(dims)))) if dims else 1
        
    print(f"[*] Region {region_id} Stability: Using D={D_common} (Mean was {np.mean(dims):.1f})")

    # 3. Truncate and Compute Overlap Matrix
    overlap_matrix = np.eye(n_blocks, dtype=float)
    
    # Pre-truncate bases to D_common (Basis must be D x Features, transposed to Features x D for overlap calc)
    bases = [s[:D_common, :].T for s in subspaces] # RegionPCA returns (Components x Features)
    
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            ov = compute_overlap_msc(bases[i], bases[j])
            overlap_matrix[i, j] = ov
            overlap_matrix[j, i] = ov
            
    # 4. Stats
    res_stats = compute_lag_stats(analyzer, overlap_matrix)
    rho, p_val = res_stats["rho"], res_stats["p_val"]
    
    return {
        "type": "region",
        "region_id": region_id,
        "matrix": overlap_matrix,
        "d_common": D_common,
        "spearman_rho": rho,
        "p_value": p_val,
        "perm_rhos": res_stats.get("perm_rhos", []),
        "monkey": analyzer.monkey,
        "z_code": analyzer.z_code,
        "method": analyzer.analysis_type,
        "block_size": analyzer.group_size,
        "subspaces": np.array(subspaces, dtype=object)
    }
