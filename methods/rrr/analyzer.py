from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List, Sequence
from . import performance
from . import matrices
from . import optimization
from . import metrics

class RRRAnalyzer:
    """
    Stateless Logic for Reduced Rank Regression (RRR) and Ridge with Nested CV.
    Refactored from RRR_Centered_matching.
    Now includes data building and higher-level orchestration.
    """

    @staticmethod
    def _plot_dir(match_to_target: bool = True):
        return metrics._plot_dir(match_to_target)

    @staticmethod
    def _lambda_for_fname(val: float) -> str:
        return metrics._lambda_for_fname(val)

    @staticmethod
    def mv_r2(y_true, y_pred, is_poisson_proxy: bool = False) -> float:
        return metrics.mv_r2(y_true, y_pred, is_poisson_proxy)

    @staticmethod
    def calc_d95(rrr_mean: np.ndarray, ridge_mean: float, d_max: int) -> int:
        return metrics.calc_d95(rrr_mean, ridge_mean, d_max)

    @staticmethod
    def _lambda_grid(X, *, lam_range: tuple[float, float, int] | None = None,
                     shrink=None, scale=True):
        # Default shrink if not passed
        return optimization._lambda_grid(X, lam_range=lam_range, shrink=shrink, scale=scale)

    @staticmethod
    def auto_alpha(X, Y, inner_splits: int | None = None, random_state: int = 0,
                    lam_range: tuple[float, float, int] | None = None,
                    is_poisson_proxy: bool = False):
        return optimization.auto_alpha(X, Y, inner_splits, random_state, lam_range, is_poisson_proxy)

    @staticmethod
    def build_mats(src_region: int, tgt_region: int, analysis_type: str, *, match_to_target: bool, trials: Optional[Sequence[int]] = None):
        return matrices.build_mats(src_region, tgt_region, analysis_type, match_to_target=match_to_target, trials=trials)

    @staticmethod
    def compute_performance(Y, X, *, d_max: int,
                             alpha: float | None,
                             outer_splits: int | None = None, inner_splits: int | None = None,
                             random_state: int,
                             lam_range: tuple[float, float, int] | None = None,
                             is_poisson_proxy: bool = False) -> Dict[str, Any]:
        return performance.compute_performance(
            Y, X, d_max=d_max, alpha=alpha, outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state, lam_range=lam_range, is_poisson_proxy=is_poisson_proxy
        )
    
    # Alias for backward compatibility (used in Semedo.py and legacy scripts)
    _performance_from_mats = compute_performance

    @staticmethod
    def performance(source_region: int, target_region: int,
                    *, d_max: int = 30, alpha: float | None = None,
                    outer_splits: int | None = None, inner_splits: int | None = None,
                    random_state: int = 0, analysis_type: str = "window",
                    match_to_target: bool = False,
                    trials: Optional[Sequence[int]] = None,
                    **kwargs):
        return performance.performance_wrapper(
            source_region, target_region, d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state, analysis_type=analysis_type,
            match_to_target=match_to_target, trials=trials, **kwargs
        )
