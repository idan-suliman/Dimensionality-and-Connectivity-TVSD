from __future__ import annotations
import numpy as np
from pathlib import Path
from core.runtime import runtime

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

def _lambda_for_fname(val: float) -> str:
    """Format lambda value for filenames (e.g. 1e3 -> 1p0e03)."""
    s = f"{val:.1e}".replace("+", "")
    return s.replace(".", "p")

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
