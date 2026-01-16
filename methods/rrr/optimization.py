from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from . import metrics
from core.runtime import runtime

def _lambda_grid(X, *, lam_range: tuple[float, float, int] | None = None,
                 shrink=None, scale=True):
    """Generate a grid of lambda values for regularization."""
    if shrink is None:
        shrink = np.linspace(0.5, 0.99, 51)
        
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

def auto_alpha(X, Y, inner_splits: int | None = None, random_state: int = 0,
                lam_range: tuple[float, float, int] | None = None,
                is_poisson_proxy: bool = False):
    """Find optimal Alpha (Lambda) via inner CV."""
    if inner_splits is None:
        inner_splits = runtime.cfg.cv_inner_splits

    lam_vec, _ = _lambda_grid(X, lam_range=lam_range)

    ss_inner = ShuffleSplit(n_splits=inner_splits, train_size=0.9, test_size=0.1, random_state=random_state)
    cv_R2 = np.empty((len(lam_vec), inner_splits), np.float32)

    for j, lam in enumerate(lam_vec):
        for f, (tr, te) in enumerate(ss_inner.split(X)):
            mdl = Ridge(alpha=lam, fit_intercept=False).fit(X[tr], Y[tr])
            cv_R2[j, f] = metrics.mv_r2(Y[te], mdl.predict(X[te]), is_poisson_proxy)

    mean = cv_R2.mean(1)
    sem = cv_R2.std(1, ddof=1) / np.sqrt(inner_splits)
    # 1-SE Rule
    thr = mean.max() - sem[mean.argmax()]
    return float(lam_vec[np.where(mean >= thr)[0][0]])
