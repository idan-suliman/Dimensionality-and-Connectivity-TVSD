from __future__ import annotations
import numpy as np
import copy
from typing import Dict, Any, Optional, List
from sklearn.linear_model import Ridge
from core.runtime import runtime
from ..pca import RegionPCA
from ..rrr import RRRAnalyzer


def analyze_region_curve(analyzer, region_id: int, max_dims: int = 30, 
                         precomputed_blocks: Optional[List[np.ndarray]] = None,
                         force_recompute: bool = False) -> Dict[str, Any]:
    """
    Computes the curve for a single region (Intrinsic Stability).
    Returns dictionary with 'dims', 'rhos', 'p_vals'.
    """
    # Use get_data_path() implicit default
    fpath = runtime.paths.get_dim_corr_path(
        analyzer.monkey,
        analyzer.analysis_type,
        analyzer.group_size,
        region_id=region_id
    )
    
    if not force_recompute and fpath.exists():
        print(f"[DimCorr] Loading {fpath.name}")
        with np.load(fpath, allow_pickle=True) as data:
            if "result" in data:
                return data["result"].item()
            else:
                res = {k: data[k] for k in data.files}
                if "meta" in res and res["meta"].ndim == 0:
                     res["meta"] = res["meta"].item()
                return res

    print(f"[DimCorr] Computing Curve for Region {region_id}...")
    
    # 1. Get Blocked Data
    if precomputed_blocks is None:
        blocks = analyzer.rep_stab.get_blocked_data(region_id)
    else:
        blocks = precomputed_blocks

    n_blocks = len(blocks)
    if n_blocks < 2:
        return {"error": "Not enough blocks"}

    # 2. Get PCA Bases for *Max* dims (or large enough)
    # We compute PCA once per block up to max_dims, then slice.
    bases_dict = {} # block_idx -> bases
    
    # Pre-compute bases for all blocks up to max_dims
    for i, X in enumerate(blocks):
        # Center
        X_cent = X - X.mean(axis=0)
        # PCA
        pca = RegionPCA(centered=False).fit(X_cent)
        Vt = pca.eigenvectors_
        # V is (Features, D) -> Vt.T is (Features, D)
        # We need first max_dims components
        # Note: If features < max_dims, handled by min
        this_max = min(max_dims, X_cent.shape[1], X_cent.shape[0])
        V = Vt.T[:, :this_max] # (Features, d)
        bases_dict[i] = V

    # 3. Iterate Dimensions
    rhos = []
    p_vals = []
    dims = []

    # Prepare Lag Matrix (fixed)
    # Compute Overlap -> Compute Spearman
    
    for d in range(1, max_dims + 1):
        # Current bases: Slice first d columns
        current_bases = []
        possible = True
        for i in range(n_blocks):
            if bases_dict[i].shape[1] < d:
                possible = False
                break
            current_bases.append(bases_dict[i][:, :d])
        
        if not possible:
            break
            
        dims.append(d)
        
        # Compute Overlap Matrix (Mean Squared Cosine)
        O = analyzer._compute_overlap_matrix(current_bases)
        
        # Compute Stability Stats (Spearman rho of Lag)
        stats = analyzer.rep_stab._compute_lag_stats(O, n_perms=None)
        rhos.append(stats["rho"])
        p_vals.append(stats["p_val"])

    result = {
        "dims": np.array(dims),
        "rhos": np.array(rhos),
        "p_vals": np.array(p_vals),
        "meta": {"region": region_id, "type": "intrinsic"}
    }
    
    # Save
    analyzer.rep_stab.save_results(result, str(fpath.parent), fpath=fpath)
    return result

def analyze_connection_curve(analyzer, src_id: int, tgt_id: int, max_dims: int = 30,
                             src_blocks: Optional[List[np.ndarray]] = None,
                             tgt_blocks: Optional[List[np.ndarray]] = None,
                             force_recompute: bool = False) -> Dict[str, Any]:
    """
    Computes the curve for a connection (Predictive Stability).
    """
    # Use get_out_path() as base() implicit default
    fpath = runtime.paths.get_dim_corr_path(
        analyzer.monkey,
        analyzer.analysis_type,
        analyzer.group_size,
        src_tgt=(src_id, tgt_id)
    )
    
    if not force_recompute and fpath.exists():
        print(f"[DimCorr] Loading {fpath.name}")
        with np.load(fpath, allow_pickle=True) as data:
             if "result" in data:
                 return data["result"].item()
             else:
                 res = {k: data[k] for k in data.files}
                 if "meta" in res and res["meta"].ndim == 0:
                     res["meta"] = res["meta"].item()
                 return res

    print(f"[DimCorr] Computing Curve for {src_id}->{tgt_id}...")

    # 1. Get Data
    if src_blocks is None:
        src_blocks = analyzer.rep_stab.get_blocked_data(src_id)
    if tgt_blocks is None:
        tgt_blocks = analyzer.rep_stab.get_blocked_data(tgt_id)

    n_blocks = min(len(src_blocks), len(tgt_blocks))
    
    # 2. Iterate Dimensions
    rhos = []
    p_vals = []
    dims = []

    # Logic optimized for speed: calculate full Ridge prediction once, then slice top-d SVD components.
    # Optimized logic: calculate full Ridge prediction once, then slice top-d SVD components.

            
    
    bases_dict = {}

    for i in range(n_blocks):
        X = src_blocks[i]
        Y = tgt_blocks[i]
        
        # User Logic: Manual B-Ridge calculation using compute_performance for lambda
        # Center data
        X_c = X - X.mean(0)
        Y_c = Y - Y.mean(0)
        
        # CV-RRR to find lambda and performance
        # Using RRRAnalyzer.compute_performance to match 'old code' exact logic for lam_opt
        # Note: We need d_max for compute_performance, even if we just want lambda.
        # We'll use 50 as in the user snippet, or max_dims if larger.
        d_search = max(50, max_dims) 
        
        perf = RRRAnalyzer.compute_performance(
            Y, X, d_max=d_search, outer_splits=None, inner_splits=None,
            alpha=None, random_state=42 + i
        )
        
        # Use median lambda from folds
        lam_opt = float(np.median(perf["lambdas"]))
        
        # B = (X'X + lam*I)^-1 X'Y
        cov_xx = X_c.T @ X_c
        cov_xy = X_c.T @ Y_c
        reg_eye = lam_opt * np.eye(X_c.shape[1])
        B_ridge = np.linalg.solve(cov_xx + reg_eye, cov_xy)
        
        # Extract Predictive Subspace
        # SVD of B gives directions in X (Source Predictive Subspace)
        U, S, Vt = np.linalg.svd(B_ridge, full_matrices=False)
        
        # Bases are columns of U
        bases_dict[i] = U

    for d in range(1, max_dims + 1):
        if d > bases_dict[0].shape[1]: 
             # Should not happen if Y has enough neurons
             break
             
        dims.append(d)
        
        # Slice tops
        current_bases = []
        for i in range(n_blocks):
             current_bases.append(bases_dict[i][:, :d])
             
        # Overlap
        O = analyzer._compute_overlap_matrix(current_bases)
        
        # Stats
        stats = analyzer.rep_stab._compute_lag_stats(O, n_perms=None)
        rhos.append(stats["rho"])
        p_vals.append(stats["p_val"])

    result = {
        "dims": np.array(dims),
        "rhos": np.array(rhos),
        "p_vals": np.array(p_vals),
        "meta": {"src": src_id, "tgt": tgt_id, "type": "predictive"}
    }
    
    analyzer.rep_stab.save_results(result, str(fpath.parent), fpath=fpath)
    return result
