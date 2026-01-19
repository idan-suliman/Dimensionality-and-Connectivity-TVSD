from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional
from core.runtime import runtime
from methods.pca import RegionPCA
from methods.rrr.analyzer import RRRAnalyzer
from methods.repetition_stability.stats import compute_overlap_msc

def compute_overlap_layers(src_id: int, tgt_id: int, max_dims: int = 40) -> Dict[str, Any]:
    """
    Computes the overlap (Mean Squared Cosine) between:
      1. PCA(Source) vs RRR Source Predictive Subspace (U of B)
      2. PCA(Target) vs RRR Target Predictive Subspace (V of B)
    
    Args:
        src_id: Source region ID (e.g., V1)
        tgt_id: Target region ID (e.g., V4)
        max_dims: Maximum number of cumulative dimensions to analyze.
        
    Returns:
        Dictionary containing:
            'dims': Array of dimensions (1..max_dims)
            'overlap_src': Overlap values for Source side
            'overlap_tgt': Overlap values for Target side
    """
    print(f"[Overlap] Computing overlap layers for {src_id} -> {tgt_id}...")
    
    # 1. Load Data (Full Matrix, No Repetitions)
    analysis_type = "window" 
    
    # helper to get names
    src_name = runtime.consts.REGION_ID_TO_NAME[src_id]
    tgt_name = runtime.consts.REGION_ID_TO_NAME[tgt_id]
    
    # Y is Target, X is Source
    Y, X = RRRAnalyzer.build_mats(src_id, tgt_id, analysis_type=analysis_type, match_to_target=False)
    
    # Center Data
    X_cent = X - X.mean(axis=0)
    Y_cent = Y - Y.mean(axis=0)
    
    # 2. PCA Subspaces
    print("[Overlap] Computing PCA...")
    pca_src = RegionPCA(centered=False).fit(X_cent)
    pca_tgt = RegionPCA(centered=False).fit(Y_cent)
    
    # Get all components (Features x D)
    V_pca_src = pca_src.eigenvectors_.T
    V_pca_tgt = pca_tgt.eigenvectors_.T
    
    # 3. RRR Predictive Subspaces
    print("[Overlap] Computing RRR...")
    
    # 3a. Find optimal alpha (lambda)
    d_search = max(50, max_dims)
    perf = RRRAnalyzer.compute_performance(
        Y, X, d_max=d_search,
        alpha=None, # Auto-detect
        random_state=42
    )
    
    # Use median lambda
    lam_opt = float(np.median(perf["lambdas"]))
    print(f"[Overlap] Optimal lambda: {lam_opt:.2f}")
    
    # 3b. Compute B_ridge
    cov_xx = X_cent.T @ X_cent
    cov_xy = X_cent.T @ Y_cent
    reg_eye = lam_opt * np.eye(X_cent.shape[1])
    
    B_ridge = np.linalg.solve(cov_xx + reg_eye, cov_xy)
    
    # 3c. Extract Predictive Subspaces via SVD of B
    U, S, Vt = np.linalg.svd(B_ridge, full_matrices=False)
    V = Vt.T 
    
    # 4. Compute Cumulative Overlaps
    dims = []
    overlap_src = []
    overlap_tgt = []
    
    actual_max = min(max_dims, V_pca_src.shape[1], V_pca_tgt.shape[1], U.shape[1])
    
    for d in range(1, actual_max + 1):
        dims.append(d)
        
        # Source Side: PCA(X) vs U
        sub_pca_src = V_pca_src[:, :d]
        sub_pred_src = U[:, :d]
        val_src = compute_overlap_msc(sub_pca_src, sub_pred_src)
        overlap_src.append(val_src)
        
        # Target Side: PCA(Y) vs V
        sub_pca_tgt = V_pca_tgt[:, :d]
        sub_pred_tgt = V[:, :d]
        val_tgt = compute_overlap_msc(sub_pca_tgt, sub_pred_tgt)
        overlap_tgt.append(val_tgt)
        
    return {
        "dims": np.array(dims),
        "overlap_src": np.array(overlap_src),
        "overlap_tgt": np.array(overlap_tgt),
        "meta": {
            "src": src_name,
            "tgt": tgt_name,
            "type": "overlap_layers"
        }
    }
