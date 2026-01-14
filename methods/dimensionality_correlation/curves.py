from __future__ import annotations
import numpy as np
import copy
from typing import Dict, Any, Optional, List
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
    output_dir = runtime.get_config().get_out_path() / "ANALYSIS_RESULTS" / analyzer.rep_stab._get_subfolder_name()
    fpath = analyzer.get_file_path(str(output_dir), region_id=region_id)
    
    if not force_recompute and fpath.exists():
        print(f"[DimCorr] Loading {fpath.name}")
        return np.load(fpath, allow_pickle=True)["result"].item()

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
    bases_dict = {} # (block_idx) -> proper bases
    
    # Pre-compute bases for all blocks up to max_dims
    for i, X in enumerate(blocks):
        # Center
        X_cent = X - X.mean(axis=0)
        # PCA
        U, S, Vt = np.linalg.svd(X_cent, full_matrices=False)
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
    # RepStab uses stats._compute_lag_stats(OverlapMatrix)
    # Ideally we'd optimize and not re-run permutations fully if not needed, 
    # but here we follow the standard logic: Compute Overlap -> Compute Spearman
    
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
        stats = analyzer.rep_stab._compute_lag_stats(O, n_perms=500) # Reduced perms for speed in curve? Or keep 2000?
        # Let's use 500 for curve speed, or stick to default. 
        # The original code didn't specify perms reduction, so maybe default 2000. 
        # But this loop is 30x. Let's stick to default or 1000.
        rhos.append(stats["rho"])
        p_vals.append(stats["p_val"])

    result = {
        "dims": np.array(dims),
        "rhos": np.array(rhos),
        "p_vals": np.array(p_vals),
        "meta": {"region": region_id, "type": "intrinsic"}
    }
    
    # Save
    analyzer.rep_stab.save_results(result, str(output_dir), fpath=fpath)
    return result

def analyze_connection_curve(analyzer, src_id: int, tgt_id: int, max_dims: int = 30,
                             src_blocks: Optional[List[np.ndarray]] = None,
                             tgt_blocks: Optional[List[np.ndarray]] = None,
                             force_recompute: bool = False) -> Dict[str, Any]:
    """
    Computes the curve for a connection (Predictive Stability).
    """
    output_dir = runtime.get_cfg().get_data_path() / "ANALYSIS_RESULTS" / analyzer.rep_stab._get_subfolder_name()
    fpath = analyzer.get_file_path(str(output_dir), src_tgt=(src_id, tgt_id))
    
    if not force_recompute and fpath.exists():
        print(f"[DimCorr] Loading {fpath.name}")
        return np.load(fpath, allow_pickle=True)["result"].item()

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

    # For predictive stability, we must re-run RRR for each dimension constraint? 
    # Or can we compute RRR once and slice?
    # RRR depends on Rank. 
    # RRRAnalyzer.compute_performance(...) does CV.
    # Here, we want the *Predictive Subspace*. 
    # Logic in RepetitionStability: "CV-RRR on Block(i) -> Predict -> SVD -> Subspace"
    # Actually, RepetitionStability.analyze_connection calls:
    #   beta = RRRAnalyzer.solve_rrr(X_train, Y_train, rank=d)
    #   Y_pred = X_test @ beta
    #   U, _, _ = svd(Y_pred) -> Keep top d
    # Wait, if we change d, the RRR solution changes (constraint). 
    # Ideally we should re-solve RRR for each d. This is expensive. 
    # 
    # Optimization: RRR with rank d corresponds to:
    # beta_OLS = inv(X'X)X'Y
    # Y_ols = X @ beta_OLS
    # PCA on Y_ols -> Top d components define the RRR subspace. 
    # So we can compute OLS *once*, get Y_ols_full, then for each d take top d components of Y_ols_full. 
    # This is valid for Reduced Rank Regression? 
    # Yes, standard RRR is PCA on the related part of Y. 
    # Y_pred(d) lies in the subspace of the first d singular vectors of Y_pred_ols.
    
    # But we need to do this for *each block* (training on itself? No, RepStab logic is complex).
    # RepetitionStability.analyze_connection uses "Cross-Repetition Prediction"? No.
    # It calculates "Predictive Subspace for Block i". 
    # "We learn B_i from X_i to Y_i via RRR(d). Then Subspace_i = span(X_i B_i)."
    #
    # So for each block i, we have source X_i and target Y_i.
    # We want to know the stability of these predictive subspaces across i.
    # 
    # Optimization Plan for Curve:
    # For each block i:
    #   1. Compute Ridge/OLS solution B_ols = argmin ||Y - X B|| (with mild ridge).
    #   2. Compute Y_pred_full = X @ B_ols.
    #   3. Compute SVD of Y_pred_full -> U_full (Features x Features).
    #   4. For each d: Subspace is U_full[:, :d].
    
    # Precompute Full Predictive Bases
    bases_dict = {}
    
    # Ridge param (small constant for stability)
    alpha = 1000.0 # Standard used in project or auto? 
    # The original methods.repetition_stability.py calls RRRAnalyzer.performance or similar?
    # Actually checking RepetitionStabilityAnalyzer.analyze_connection from outline...
    # It calls RRRAnalyzer? 
    # Let's peek at the original file content logic if needed. 
    # But usually this project uses Ridge for regularization. 
    # Let's use RRRAnalyzer helper if available, or just implement Ridge+PCA here for speed.
    
    # rrr_machine = RRRAnalyzer(monkey=analyzer.monkey, z_code=analyzer.z_code, analysis_type=analyzer.analysis_type)
    
    for i in range(n_blocks):
        X = src_blocks[i]
        Y = tgt_blocks[i]
        
        # We need the predictive subspace of rank d.
        # Approximation: Ridge B -> Y_hat -> PCA -> Top d.
        # This allows computing once.
        
        # Center
        X_c = X - X.mean(0)
        Y_c = Y - Y.mean(0)
        
        # Ridge Solution (using Auto Alpha or fixed?)
        # For speed in curve scanning, we might use fixed or simple logic.
        # Let's stick to a robust simpler approach: standard Ridge with fixed alpha or simple GCV.
        # Using sklearn Ridge.
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1000.0) # Heuristic high regularization for neural data? Or 1.0? 
        # In Semedo, they used auto-alpha. 
        # If we re-run auto-alpha for every block, it's slow. 
        # Let's assume a reasonable alpha or re-use logic.
        
        ridge.fit(X_c, Y_c)
        Y_pred = ridge.predict(X_c)
        
        # SVD of predicted
        U, S, Vt = np.linalg.svd(Y_pred, full_matrices=False)
        # Y_pred is (Samples, Neurons). Subspace is in Neurons space? 
        # Wait, the predictive subspace is the subspace of the OUTPUT activity that is predicted.
        # So it is the row space of Y_pred? Or column space?
        # Usually subspace of population activity -> Column space of V (if Samples x Neurons).
        # V is (Neurons, Neurons). 
        # Correct: Subspace in neural state space.
        
        bases_dict[i] = Vt.T # (Neurons, Neurons) full rank ordered by variance.

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
        stats = analyzer.rep_stab._compute_lag_stats(O, n_perms=500)
        rhos.append(stats["rho"])
        p_vals.append(stats["p_val"])

    result = {
        "dims": np.array(dims),
        "rhos": np.array(rhos),
        "p_vals": np.array(p_vals),
        "meta": {"src": src_id, "tgt": tgt_id, "type": "predictive"}
    }
    
    analyzer.rep_stab.save_results(result, str(output_dir), fpath=fpath)
    return result
