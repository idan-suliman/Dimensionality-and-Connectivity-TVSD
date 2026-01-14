from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from . import utils

def compute_overlap_msc(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute Mean Squared Cosine between two subspaces A and B.
    A, B: Orthonormal basis matrices (Features x D)
    """
    # Ensure orthonormal (though PCA output should be)
    # Using SVD on the "Interaction Matrix" (Ua.T @ Ub)
    C = A.T @ B
    svals = np.linalg.svd(C, compute_uv=False)
    return float(np.mean(svals ** 2))

def d95_from_perf(rrr_mean: np.ndarray, ridge_mean: float, d_max: int) -> int:
    """Helper for Connective Analysis (Semedo d95 logic)."""
    thr = 0.95 * float(ridge_mean)
    idx = np.where(rrr_mean >= thr)[0]
    return int(idx[0] + 1) if idx.size else int(d_max)

def compute_lag_stats(analyzer, O: np.ndarray, n_perms: int = 2000) -> dict:
    """
    Spearman Correlation (Lag vs Overlap) + Permutation Test.
    """
    xs, ys, _, _, _ = utils.extract_lag_data(O)
    
    if len(xs) < 2:
        return {"rho": 0.0, "p_val": 1.0}
        
    # Observed
    rho_obs, _ = spearmanr(xs, ys)
    
    # Permutation Test (shuffling Block indices)
    N = O.shape[0]
    rng = np.random.default_rng(42)
    
    perm_rhos = np.zeros(n_perms)
    for i in range(n_perms):
        p_idx = rng.permutation(N)
        O_perm = O[p_idx][:, p_idx]
        xs_p, ys_p, _, _, _ = utils.extract_lag_data(O_perm)
        r, _ = spearmanr(xs_p, ys_p)
        perm_rhos[i] = r
        
    # P-value: Fraction of permutations with rho <= rho_obs (one-tailed)
    # We expect rho to be negative (decay).
    p_val = (np.sum(perm_rhos <= rho_obs) + 1) / (n_perms + 1)
    
    return {"rho": rho_obs, "p_val": p_val}
