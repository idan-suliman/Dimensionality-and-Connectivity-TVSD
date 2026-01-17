from __future__ import annotations
import numpy as np
from typing import Dict, Any
from ..rrr import RRRAnalyzer
from .stats import compute_lag_stats, compute_overlap_msc, compute_r2_lag_stats

def analyze_connection(analyzer, src_id: int, tgt_id: int, fixed_d: int | None = None) -> Dict[str, Any]:
    """
    Computes predictive stability between source and target blocks.
    Method: CV-RRR on Block(i) -> Predict -> SVD -> Subspace.
    """
    X_blocks = analyzer.get_blocked_data(src_id)
    Y_blocks = analyzer.get_blocked_data(tgt_id)
    n_blocks = len(X_blocks)
    
    subspaces = [] # Predicted subspaces
    r2_per_block = [] # Performance tracking
    temp_d95s = []
    
    # RRR Configuration
    d_max = 50 # Use at least 50 components
    
    # 1. Process Blocks (Train RRR + Extract Subspace)
    for i in range(n_blocks):
        X, Y = X_blocks[i], Y_blocks[i]
        
        # CV-RRR to find lambda and performance
        perf = RRRAnalyzer.compute_performance(
            Y, X, d_max=d_max, outer_splits=None, inner_splits=None,
            alpha=None, random_state=42 + i
        )
        
        lam_opt = float(np.median(perf["lambdas"]))
        d95 = RRRAnalyzer.calc_d95(perf["rrr_R2_mean"], perf["ridge_R2_mean"], d_max)
        temp_d95s.append(d95)
        r2_per_block.append(perf["rrr_R2_mean"]) # Save full curve
        
        # Fit Ridge Model (B)
        # Center data
        Xc = X - X.mean(0)
        Yc = Y - Y.mean(0)
        
        # B = (X'X + lam*I)^-1 X'Y
        cov_xx = Xc.T @ Xc
        cov_xy = Xc.T @ Yc
        reg_eye = lam_opt * np.eye(Xc.shape[1])
        B_ridge = np.linalg.solve(cov_xx + reg_eye, cov_xy)
        
        # Extract Predictive Subspace
        # SVD of B gives directions in X
        U, _, _ = np.linalg.svd(B_ridge, full_matrices=False)
        subspaces.append(U) # Store full basis U (Feat_X x min(Fx, Fy))

    # 2. Determine Common D
    if fixed_d is not None:
        D_common = fixed_d
    else:
        D_common = max(1, int(np.floor(np.mean(temp_d95s))))
    
    print(f"[*] Connection {src_id}->{tgt_id}: Common D={D_common}")
    
    # 3. Extract final bases (Truncate U to D_common)
    final_bases = []
    block_r2_vals = []
    for i, U in enumerate(subspaces):
        # U is (Feat_X x K). We need (Feat_X x D_common).
        # Ensure we don't exceed ranks
        safe_D = min(D_common, U.shape[1])
        basis = U[:, :safe_D]
        final_bases.append(basis)
        
        # Store R2 at this D
        block_r2_vals.append(r2_per_block[i][safe_D-1] if safe_D>0 else 0)

    # 4. Compute Overlap
    overlap_matrix = np.eye(n_blocks, dtype=float)
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            ov = compute_overlap_msc(final_bases[i], final_bases[j])
            overlap_matrix[i, j] = ov
            overlap_matrix[j, i] = ov
            
    # 5. Stats
    res_stats = compute_lag_stats(analyzer, overlap_matrix)
    rho, p_val = res_stats["rho"], res_stats["p_val"]
    
    # 6. R2 Stats
    r2_stats = compute_r2_lag_stats(np.array(block_r2_vals))
    
    return {
        "type": "connection",
        "src_id": src_id,
        "tgt_id": tgt_id,
        "matrix": overlap_matrix,
        "d_common": D_common,
        "block_r2s": block_r2_vals, # Mean R2 performance
        "spearman_rho": rho,
        "p_value": p_val,
        "perm_rhos": res_stats.get("perm_rhos", []),
        "r2_rho": r2_stats["rho"],
        "r2_p_val": r2_stats["p_val"],
        "perm_r2s": r2_stats["perm_rhos"],
        "monkey": analyzer.monkey,
        "z_code": analyzer.z_code,
        "method": analyzer.analysis_type,
        "block_size": analyzer.group_size,
        "subspaces": np.array(subspaces, dtype=object)
    }
