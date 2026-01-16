from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from typing import Dict, Any, Optional, Sequence
from . import optimization
from . import metrics
from . import matrices
from ..pca import RegionPCA
from core.runtime import runtime

def compute_performance(Y, X, *, d_max: int,
                         alpha: float | None,
                         outer_splits: int | None = None, inner_splits: int | None = None,
                         random_state: int,
                         lam_range: tuple[float, float, int] | None = None,
                         is_poisson_proxy: bool = False) -> Dict[str, Any]:
    """
    Run Nested Cross-Validation to evaluate RRR and Ridge performance.
    """
    if outer_splits is None:
        outer_splits = runtime.cfg.cv_outer_splits
    if inner_splits is None:
        inner_splits = runtime.cfg.cv_inner_splits

    kf_outer = ShuffleSplit(n_splits=outer_splits, train_size=0.9, test_size=0.1, random_state=random_state)
    
    ridge_scores  = np.zeros(outer_splits, np.float32)
    rrr_scores    = np.zeros((outer_splits, d_max), np.float32)
    lambdas_outer = np.zeros(outer_splits, np.float32)   

    for f, (tr, te) in enumerate(kf_outer.split(X)):
        Xtr_full, Xte  = X[tr], X[te]
        Ytr_full, Yte  = Y[tr], Y[te]

        lam_f = alpha
        if lam_f is None:
            lam_f = optimization.auto_alpha(Xtr_full, Ytr_full, inner_splits=inner_splits, 
                                           random_state=random_state + 31*f,
                                           lam_range=lam_range, is_poisson_proxy=is_poisson_proxy)
        lambdas_outer[f] = lam_f

        muX, muY  = Xtr_full.mean(0, keepdims=True), Ytr_full.mean(0, keepdims=True)
        Xtr_c, Xte_c  = Xtr_full - muX, Xte - muX
        Ytr_c, Yte_c  = Ytr_full - muY, Yte - muY

        # Ridge Performance
        ridge = Ridge(alpha=lam_f, fit_intercept=False).fit(Xtr_c, Ytr_c)
        ridge_scores[f] = metrics.mv_r2(Yte_c, ridge.predict(Xte_c), is_poisson_proxy)

        # RRR Performance via OLS + SVD projection
        # B_ols = (X'X + lam*I)^-1 X'Y
        B_ols = np.linalg.solve(Xtr_c.T @ Xtr_c + lam_f*np.eye(Xtr_c.shape[1]), Xtr_c.T @ Ytr_c)
        
        # Use RegionPCA instead of direct SVD
        # Note: Xtr_c @ B_ols is already "centered" in the sense that Xtr_c is centered.
        proj = Xtr_c @ B_ols
        pca = RegionPCA(centered=False).fit(proj)
        Vt = pca.eigenvectors_
        V = Vt.T
        for d in range(1, d_max + 1):
            Bd = B_ols @ V[:, :d] @ V[:, :d].T
            rrr_scores[f, d - 1] = metrics.mv_r2(Yte_c, Xte_c @ Bd, is_poisson_proxy)

    return dict(
        ridge_R2_mean = float(ridge_scores.mean()),
        ridge_R2_sem  = float(ridge_scores.std(ddof=1) / np.sqrt(outer_splits)),
        rrr_R2_mean   = rrr_scores.mean(0),
        rrr_R2_sem    = rrr_scores.std(0, ddof=1) / np.sqrt(outer_splits),
        lambdas       = lambdas_outer
    )

def performance_wrapper(source_region: int, target_region: int,
                *, d_max: int = 30, alpha: float | None = None,
                outer_splits: int | None = None, inner_splits: int | None = None,
                random_state: int = 0, analysis_type: str = "window",
                match_to_target: bool = False,
                trials: Optional[Sequence[int]] = None,
                **kwargs):
    """
    High-level wrapper: Builds matrices and runs compute_performance.
    """
    if trials is None and 'trial_subset' in kwargs:
        trials = kwargs['trial_subset']

    Y, X = matrices.build_mats(source_region, target_region,
                        analysis_type, match_to_target=match_to_target,
                        trials=trials)
                        
    return compute_performance(
        Y, X,
        d_max        = d_max,
        alpha        = alpha,
        outer_splits = outer_splits,
        inner_splits = inner_splits,
        random_state = random_state
    )
