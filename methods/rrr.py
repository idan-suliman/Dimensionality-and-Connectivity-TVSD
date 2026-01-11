"""
rrr.py
------
Stateless logic for Reduced Rank Regression (RRR) and Ridge with Nested CV.
Refactored from connectivity.py.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from typing import Dict, Any, Tuple, Optional, List, Sequence
from pathlib import Path
from core.runtime import runtime
from .matchingSubset import MATCHINGSUBSET

class RRRAnalyzer:
    """
    Stateless Logic for Reduced Rank Regression (RRR) and Ridge with Nested CV.
    Refactored from RRR_Centered_matching.
    Now includes data building and higher-level orchestration.
    """

    # =========================================================================
    # Helper Methods (Path, Math, Internals)
    # =========================================================================

    @staticmethod
    def _plot_dir(match_to_target: bool = True) -> Path:
        """
        Return <data_path>/TARGET_RRR (if match_to_target) or <data_path>/REGULAR_RRR.
        Creates the directory if it doesn't exist.
        """
        d = runtime.get_cfg().get_regular_rrr_dir()
        if match_to_target:
            d = runtime.get_cfg().get_target_rrr_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _lambda_for_fname(val: float) -> str:
        """Format lambda value for filenames (e.g. 1e3 -> 1p0e03)."""
        s = f"{val:.1e}".replace("+", "")
        return s.replace(".", "p")

    @staticmethod
    def mv_r2(y_true, y_pred, is_poisson_proxy: bool = False) -> float:
        """
        Compute multivariate R2.
        If is_poisson_proxy (legacy '1_without_z_score'), computes mean of per-electrode R2.
        Otherwise computes pooled R2 (1 - SSE_total / SST_total).
        """
        if is_poisson_proxy: 
            # Legacy "no_z" mode: mean of R2s
            sse = np.square(y_true - y_pred).sum(axis=0)
            sst = np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum(axis=0) + 1e-12
            r2_each = 1.0 - sse / sst
            return float(r2_each.mean())
        else:
             # Standard Pooled R2
             sse = np.square(y_true - y_pred).sum()
             sst = np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum() + 1e-12
             return float(1.0 - sse / sst)

    @staticmethod
    def _lambda_grid(X, *, lam_range: tuple[float, float, int] | None = None,
                     shrink=np.linspace(0.5, 0.99, 51), scale=True):
        """Generate a grid of lambda values for regularization."""
        if lam_range is not None:
            lam_min, lam_max, n_pts = lam_range
            lam_vec = np.linspace(lam_min, lam_max, int(n_pts), dtype=float)
            eig_part = None
            return lam_vec, eig_part
        
        # Heuristic based on eigenvalues
        s = X.std(0, ddof=1)
        keep = s > np.sqrt(np.finfo(s.dtype).eps)
        Z = (X[:, keep] - X[:, keep].mean(0)) / s[keep] if scale else \
            X[:, keep] - X[:, keep].mean(0)
        d = np.linalg.eigvalsh(Z.T @ Z)
        lam_vec = d.max() * (1 - shrink) / shrink
        eig_part = (d[:, None] / (d[:, None] + lam_vec[None, :])).sum(0)
        return lam_vec.astype(float), eig_part

    @staticmethod
    def auto_alpha(X, Y, inner_splits: int = 10, random_state: int = 0,
                    lam_range: tuple[float, float, int] | None = None,
                    is_poisson_proxy: bool = False):
        """Find optimal Alpha (Lambda) via inner CV."""
        lam_vec, _ = RRRAnalyzer._lambda_grid(X, lam_range=lam_range)

        ss_inner = ShuffleSplit(n_splits=inner_splits, train_size=0.9, test_size=0.1, random_state=random_state)
        cv_R2 = np.empty((len(lam_vec), inner_splits), np.float32)

        for j, lam in enumerate(lam_vec):
            for f, (tr, te) in enumerate(ss_inner.split(X)):
                mdl = Ridge(alpha=lam, fit_intercept=False).fit(X[tr], Y[tr])
                cv_R2[j, f] = RRRAnalyzer.mv_r2(Y[te], mdl.predict(X[te]), is_poisson_proxy)

        mean = cv_R2.mean(1)
        sem = cv_R2.std(1, ddof=1) / np.sqrt(inner_splits)
        # 1-SE Rule
        thr = mean.max() - sem[mean.argmax()]
        return float(lam_vec[np.where(mean >= thr)[0][0]])

    # =========================================================================
    # Data Processing
    # =========================================================================

    @staticmethod
    def build_mats(src_region: int, tgt_region: int, analysis_type: str, *, match_to_target: bool, trials: Optional[Sequence[int]] = None):
        """
        Construct X (source) and Y (target) matrices for analysis.
        If match_to_target is True, loads/computes the V1-subset that matches Target firing rates.
        """
        rois   = runtime.get_cfg().get_rois()
        src_name = runtime.get_consts().REGION_ID_TO_NAME[src_region]
        tgt_name = runtime.get_consts().REGION_ID_TO_NAME[tgt_region]

        def build(reg_id, idx):
            return runtime.get_data_manager().build_trial_matrix(
                region_id=reg_id,
                analysis_type=analysis_type,
                trials=trials,
                electrode_indices=np.asarray(idx, dtype=int),
                return_stimulus_ids=False
            )

        src_idx_full = np.where(rois == src_region)[0]
        tgt_idx_full = np.where(rois == tgt_region)[0]

        if match_to_target:
            # --- Load Matching Subset ---
            dir_match = (runtime.get_cfg().get_data_path() / "TARGET_RRR" / analysis_type.upper())
            dir_match.mkdir(parents=True, exist_ok=True)
            subset_f = dir_match / f"{src_name}_to_{tgt_name}_{analysis_type}.npz"
            
            if not subset_f.exists():
                MATCHINGSUBSET.match_and_save(
                    src_name, tgt_name,
                    stat_mode=analysis_type,
                    show_plot=False, verbose=False)

            with np.load(subset_f) as z:
                match_idx = z["phys_idx"]          # V1-MATCH 

            remain_mask    = ~np.isin(src_idx_full, match_idx)
            src_idx_remain = src_idx_full[remain_mask]
            
            if src_idx_remain.size == 0:
                raise ValueError("All V1 electrodes were matched; nothing left in X!")

            X = build(src_region, src_idx_remain)  # V1 minus V1-MATCH
            Y = build(src_region, match_idx)       # V1-MATCH (acting as Target)
        else:
            X = build(src_region, src_idx_full)    # V1 full
            Y = build(tgt_region, tgt_idx_full)    # V4 / IT full

        return Y, X

    # =========================================================================
    # Core Computation
    # =========================================================================

    @staticmethod
    def compute_performance(Y, X, *, d_max: int,
                             alpha: float | None,
                             outer_splits: int, inner_splits: int,
                             random_state: int,
                             lam_range: tuple[float, float, int] | None = None,
                             is_poisson_proxy: bool = False) -> Dict[str, Any]:
        """
        Run Nested Cross-Validation to evaluate RRR and Ridge performance.
        Returns dictionary with R2 scores (mean/sem) and chosen lambdas.
        """
        kf_outer = ShuffleSplit(n_splits=outer_splits, train_size=0.9, test_size=0.1, random_state=random_state)
        
        ridge_scores  = np.zeros(outer_splits, np.float32)
        rrr_scores    = np.zeros((outer_splits, d_max), np.float32)
        lambdas_outer = np.zeros(outer_splits, np.float32)   

        for f, (tr, te) in enumerate(kf_outer.split(X)):
            Xtr_full, Xte  = X[tr], X[te]
            Ytr_full, Yte  = Y[tr], Y[te]

            lam_f = alpha
            if lam_f is None:
                lam_f = RRRAnalyzer.auto_alpha(Xtr_full, Ytr_full, inner_splits=inner_splits, 
                                               random_state=random_state + 31*f,
                                               lam_range=lam_range, is_poisson_proxy=is_poisson_proxy)
            lambdas_outer[f] = lam_f

            muX, muY  = Xtr_full.mean(0, keepdims=True), Ytr_full.mean(0, keepdims=True)
            Xtr_c, Xte_c  = Xtr_full - muX, Xte - muX
            Ytr_c, Yte_c  = Ytr_full - muY, Yte - muY

            # Ridge Performance
            ridge = Ridge(alpha=lam_f, fit_intercept=False).fit(Xtr_c, Ytr_c)
            ridge_scores[f] = RRRAnalyzer.mv_r2(Yte_c, ridge.predict(Xte_c), is_poisson_proxy)

            # RRR Performance via OLS + SVD projection
            B_ols = np.linalg.solve(Xtr_c.T @ Xtr_c + lam_f*np.eye(Xtr_c.shape[1]), Xtr_c.T @ Ytr_c)
            _, _, Vt = np.linalg.svd(Xtr_c @ B_ols, full_matrices=False)
            V = Vt.T
            for d in range(1, d_max + 1):
                Bd = B_ols @ V[:, :d] @ V[:, :d].T
                rrr_scores[f, d - 1] = RRRAnalyzer.mv_r2(Yte_c, Xte_c @ Bd, is_poisson_proxy)

        return dict(
            ridge_R2_mean = float(ridge_scores.mean()),
            ridge_R2_sem  = float(ridge_scores.std(ddof=1) / np.sqrt(outer_splits)),
            rrr_R2_mean   = rrr_scores.mean(0),
            rrr_R2_sem    = rrr_scores.std(0, ddof=1) / np.sqrt(outer_splits),
            lambdas       = lambdas_outer
        )
    
    # Alias for backward compatibility (used in Semedo.py and legacy scripts)
    _performance_from_mats = compute_performance

    @staticmethod
    def performance(source_region: int, target_region: int,
                    *, d_max: int = 30, alpha: float | None = None,
                    outer_splits: int = 10, inner_splits: int = 10,
                    random_state: int = 0, analysis_type: str = "window",
                    match_to_target: bool = False,
                    trials: Optional[Sequence[int]] = None,
                    **kwargs):
        """
        High-level wrapper: Builds matrices and runs compute_performance.
        """
        if trials is None and 'trial_subset' in kwargs:
            trials = kwargs['trial_subset']

        Y, X = RRRAnalyzer.build_mats(source_region, target_region,
                            analysis_type, match_to_target=match_to_target,
                            trials=trials)
        return RRRAnalyzer.compute_performance(
            Y, X,
            d_max        = d_max,
            alpha        = alpha,
            outer_splits = outer_splits,
            inner_splits = inner_splits,
            random_state = random_state
        )
