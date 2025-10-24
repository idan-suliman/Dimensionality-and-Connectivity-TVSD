from pathlib import Path
import numpy as np
from matchingSubset import MATCHINGSUBSET
from runtime import runtime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit


class RRR_Centered_matching:
    """
    Compute Ridge-full & RRR curves with *Nested  CV*
    (outer-CV for test-set, inner-CV for λ-selection) and plot them (±SEM).

    Parameters (key ones)
    ---------------------
    source_region     : int   – ROI id of V1     
    target_region     : int   – ROI id of target  (V4=2, IT=3 …)
    analysis_types    : tuple – "window" | "baseline100" | "residual"
    d_max             : int   – max rank (d) for RRR
    outer_splits      : int   – #folds in *outer* CV   (default 10)
    inner_splits      : int   – #folds in *inner* CV   (default 10)
    alpha             : float – fixed λ;  if None → inner-CV picks λ per fold
    """

    # ------------------------------ plot-dir -------------
    @staticmethod
    def _plot_dir(match_to_target: bool = True) -> Path:
        """
        Return <data_path>/TARGET_RRR (if match_to_target) or <data_path>/REGULAR_RRR (otherwise).
        Creates the directory if `create` is True.
        """
        d = runtime.get_cfg().get_regular_rrr_dir()
        if match_to_target:
            d = runtime.get_cfg().get_target_rrr_dir()

        d.mkdir(parents=True, exist_ok=True)
        
        return d


    def make_performance_plot(
        source_region: int = 1,
        target_region: int = 2,
        *,
        d_max: int = 35,
        alpha: float | None = None,
        outer_splits: int = 3,
        inner_splits: int = 3,
        random_state: int = 0,
        analysis_types: tuple[str, ...] | None = runtime.get_consts().ANALYSIS_TYPES,
        match_to_target: bool = False,
        _external_X: "np.ndarray | None" = None,
        _external_Ys: "dict[str, np.ndarray] | None" = None,
        custom_title: str | None = None,
    ):
        """
        Plot Ridge-full & RRR curves plus centered info block (σ₁ + λ*).
        """
        using_external = (_external_X is not None) and (_external_Ys is not None)
        cmap = {"window": "#C21807", "baseline100": "#1565C0", "residual": "#2E7D32"}

        # ---------------- figure layout ----------------
        n_info = len(analysis_types)
        fig_h  = 4.2 + 0.35 * n_info
        fig    = plt.figure(figsize=(7, fig_h))

        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=[4.0, 0.35 * n_info],
            hspace=0.30           
        )
        ax_main = fig.add_subplot(gs[0])
        ax_info = fig.add_subplot(gs[1])

        dims, yvals = np.arange(1, d_max + 1), []
        sing_vals, lam_lines = [], []

        # --------------- main loop ---------------------
        for at in analysis_types:
            col = cmap[at]
            X, Y = (_external_X, _external_Ys[at]) if using_external else \
                RRR_Centered_matching._build_mats(source_region, target_region,
                                at, match_to_target=match_to_target)

            res = RRR_Centered_matching._performance_from_mats(
                Y, X,
                d_max=d_max, alpha=alpha,
                outer_splits=outer_splits, inner_splits=inner_splits,
                random_state=random_state)

            ax_main.errorbar(dims, res["rrr_R2_mean"], yerr=res["rrr_R2_sem"],
                            fmt="o-", ms=3, lw=1.2, capsize=3,
                            color=col, ecolor=col)
            ax_main.scatter([1], [res["ridge_R2_mean"]], marker="^",
                            s=80, color=col, edgecolors="k")
            yvals.extend(res["rrr_R2_mean"]); yvals.append(res["ridge_R2_mean"])

            thr = 0.95 * res["ridge_R2_mean"]
            idx = np.where(res["rrr_R2_mean"] >= thr)[0]
            if idx.size:
                d95, r2d = int(idx[0]+1), float(res["rrr_R2_mean"][idx[0]])
                ax_main.scatter(d95, r2d, s=140, facecolor=col,
                                edgecolors="k", lw=1)
                ax_main.text(d95, r2d, f"{d95}", color="white", weight="bold",
                            ha="center", va="center",
                            path_effects=[pe.Stroke(linewidth=1.8,
                                                    foreground="black"), pe.Normal()])
                ax_main.axvline(d95, color=col, ls="--", lw=0.7, alpha=0.4)

            if alpha is None:
                Xc = X - X.mean(0, keepdims=True)
                sing_vals.append(int(round(np.linalg.svd(Xc, compute_uv=False)[0])))
                lam_vec, _ = RRR_Centered_matching._lambda_grid(X)
                chosen = res["lambdas"]
                idx_pos = [int(np.abs(lam_vec - l).argmin()+1) for l in chosen]
                pairs = [f"{int(round(l))} ({p})" for l, p in zip(chosen, idx_pos)]
                if len(pairs) > 3:
                    pairs = pairs[:3] + ["…"]
                lam_lines.append(", ".join(pairs))

        # --------------- cosmetics ---------------------
        pad = 0.05 * (max(yvals) - min(yvals) if max(yvals) > min(yvals) else 0.05)
        ax_main.set_ylim(max(0.0, min(yvals)-pad), min(1.0, max(yvals)+pad))
        ax_main.set_xlabel("Predictive dimensions (d)",
                        labelpad=6)        
        ax_main.set_ylabel(rf"Mean $R^2$  (CV: outer {outer_splits}, inner {inner_splits})")
        ax_main.grid(alpha=0.3)

        tgt_lbl = runtime.get_consts().REGION_ID_TO_NAME[target_region]
        ax_main.set_title(custom_title or
                        f"V1 → {'V1-match '+tgt_lbl if match_to_target else tgt_lbl}",
                        fontsize=12, pad=10)

        # --------------- info block --------------------
        ax_info.axis("off")
        if alpha is None:
            step = 1.0 / (n_info + 1)
            for i, (at, s1, lam) in enumerate(zip(analysis_types,
                                                sing_vals,
                                                lam_lines)):
                ax_info.text(0.5, 1-step*(i+1),
                            f"{at}:  σ₁ = {s1}   |   λ* per fold:  {lam}",
                            ha="center", va="center",
                            fontsize=9, color=cmap[at])

        # --------------- save --------------------------
        fig.tight_layout()
        tag = "nestedLam" if alpha is None else f"lam{RRR_Centered_matching._lambda_for_fname(alpha)}"
        plot_dir = RRR_Centered_matching._plot_dir(match_to_target)
        base = (f"{runtime.get_cfg().get_monkey_name().replace(' ', '')}_rrr_"
                f"{'target' if match_to_target else 'regular'}_"
                f"V1_to_{tgt_lbl}_{tag}")
        fname = plot_dir / f"{base}.png"
        fig.savefig(fname, dpi=300, facecolor="white")
        plt.close(fig)
        print(f"[✓] RRR figure saved → {fname}")

    # ----------------------- PUBLIC: numeric performance -
    def performance(source_region: int, target_region: int,
                    *, d_max: int = 30, alpha: float | None = None,
                    outer_splits: int = 10, inner_splits: int = 10,
                    random_state: int = 0, analysis_type: str = "window",
                    match_to_target: bool = False):
        """Return dict with mean±SEM for Ridge & RRR (Nested CV)."""
        Y, X = RRR_Centered_matching._build_mats(source_region, target_region,
                            analysis_type, match_to_target=match_to_target)
        return RRR_Centered_matching._performance_from_mats(
            Y, X,
            d_max        = d_max,
            alpha        = alpha,
            outer_splits = outer_splits,
            inner_splits = inner_splits,
            random_state = random_state
        )

    # -------------------- CORE: nested-CV implementation --------------------
    def _performance_from_mats(Y, X, *, d_max: int,
                            alpha: float | None,
                            outer_splits: int, inner_splits: int,
                            random_state: int,
                            lam_range: tuple[float, float, int] | None = None):
  
        # -------- outer CV: always 90 % train / 10 % test --------
        kf_outer = ShuffleSplit(
            n_splits   = outer_splits,
            train_size = 0.9,
            test_size  = 0.1,
            random_state = random_state
        )

        ridge_scores  = np.zeros(outer_splits, np.float32)
        rrr_scores    = np.zeros((outer_splits, d_max), np.float32)
        lambdas_outer = np.zeros(outer_splits, np.float32)   

        for f, (tr, te) in enumerate(kf_outer.split(X)):
            Xtr_full, Xte  = X[tr], X[te]
            Ytr_full, Yte  = Y[tr], Y[te]

            # -------- λ selection (inner CV) --------
            lam_f = alpha
            if lam_f is None:
                lam_f = RRR_Centered_matching._auto_alpha(
                    Xtr_full, Ytr_full,
                    inner_splits = inner_splits,
                    random_state = random_state + 31*f   # seed 
                )
            lambdas_outer[f] = lam_f

            # -------- centering (based on TRAIN only) --------
            muX, muY  = Xtr_full.mean(0, keepdims=True), Ytr_full.mean(0, keepdims=True)
            Xtr_c, Xte_c  = Xtr_full - muX, Xte - muX
            Ytr_c, Yte_c  = Ytr_full - muY, Yte - muY

            # -------- Ridge full --------
            ridge = Ridge(alpha=lam_f, fit_intercept=False).fit(Xtr_c, Ytr_c)
            ridge_scores[f] = RRR_Centered_matching.mv_r2(Yte_c, ridge.predict(Xte_c))

            # -------- RRR --------
            B_ols = np.linalg.solve(Xtr_c.T @ Xtr_c + lam_f*np.eye(Xtr_c.shape[1]),
                                    Xtr_c.T @ Ytr_c)
            _, _, Vt = np.linalg.svd(Xtr_c @ B_ols, full_matrices=False)
            V = Vt.T
            for d in range(1, d_max + 1):
                Bd = B_ols @ V[:, :d] @ V[:, :d].T
                rrr_scores[f, d - 1] = RRR_Centered_matching.mv_r2(Yte_c, Xte_c @ Bd)

        return dict(
            ridge_R2_mean = float(ridge_scores.mean()),
            ridge_R2_sem  = float(ridge_scores.std(ddof=1) / np.sqrt(outer_splits)),
            rrr_R2_mean   = rrr_scores.mean(0),
            rrr_R2_sem    = rrr_scores.std(0, ddof=1) / np.sqrt(outer_splits),
            lambdas       = lambdas_outer
        )

    # ------------------ λ-search (inner CV) -------------
    def _auto_alpha(
        X,
        Y,
        *,
        inner_splits: int = 10,
        random_state: int = 0,
        lam_range: tuple[float, float, int] | None = None,
    ):
        """
        Select λ by inner cross-validation.

        The inner CV always uses a 90 % train / 10 % validation split,
        regardless of the requested number of splits (inner_splits).
        """
        lam_vec, _ = RRR_Centered_matching._lambda_grid(X, lam_range=lam_range)

        ss_inner = ShuffleSplit(
            n_splits=inner_splits,
            train_size=0.9,
            test_size=0.1,
            random_state=random_state,
        )

        cv_R2 = np.empty((len(lam_vec), inner_splits), np.float32)

        for j, lam in enumerate(lam_vec):
            for f, (tr, te) in enumerate(ss_inner.split(X)):
                mdl = Ridge(alpha=lam, fit_intercept=False).fit(X[tr], Y[tr])
                cv_R2[j, f] = RRR_Centered_matching.mv_r2(Y[te], mdl.predict(X[te]))

        mean = cv_R2.mean(1)
        sem = cv_R2.std(1, ddof=1) / np.sqrt(inner_splits)
        thr = mean.max() - sem[mean.argmax()]  # one-SEM rule
        return float(lam_vec[np.where(mean >= thr)[0][0]])

    # ------------------- BUILD MATRICES------------------
    def _build_mats(src_region: int, tgt_region: int, analysis_type: str, *, match_to_target: bool):
        
        trials = runtime.get_cfg()._load_trials()
        rois   = runtime.get_cfg().get_rois()
        src_name = runtime._consts.REGION_ID_TO_NAME[src_region]
        tgt_name = runtime._consts.REGION_ID_TO_NAME[tgt_region]

        def build(reg_id, idx):
            return runtime.get_cfg().build_trial_matrix(
                region_id=reg_id,
                analysis_type=analysis_type,
                trials=None,                                   # <-- חשוב: לא להעביר רשימת dictים
                electrode_indices=np.asarray(idx, dtype=int), #     עובדים על כל הטריילים
                return_stimulus_ids=False
            )

        # ---------- full index vectors -------------------------------------
        src_idx_full = np.where(rois == src_region)[0]
        tgt_idx_full = np.where(rois == tgt_region)[0]

        # ---------- build X, Y ---------------------------------------------
        if match_to_target:
            # --- load / create matching subset -----------------------------
            dir_match = (runtime.get_cfg().get_data_path() / "TARGET_RRR" /
                        analysis_type.upper())
            dir_match.mkdir(parents=True, exist_ok=True)
            subset_f = dir_match / f"{src_name}_to_{tgt_name}_{analysis_type}.npz"
            if not subset_f.exists():
                MATCHINGSUBSET.match_and_save(
                    src_name, tgt_name,
                    stat_mode=analysis_type,
                    show_plot=False, verbose=False)

            with np.load(subset_f) as z:
                match_idx = z["phys_idx"]          # ← V1-MATCH 

            remain_mask    = ~np.isin(src_idx_full, match_idx)
            src_idx_remain = src_idx_full[remain_mask]
            if src_idx_remain.size == 0:
                raise ValueError("All V1 electrodes were matched; nothing left in X!")

            X = build(src_region, src_idx_remain)  # V1 minus V1-MATCH
            Y = build(src_region, match_idx)       # V1-MATCH (target)

        else:
            X = build(src_region, src_idx_full)    # V1 full
            Y = build(tgt_region, tgt_idx_full)    # V4 / IT full

        return Y, X


    # ---------------------- λ-grid & helpers-------------
    def _lambda_grid(X, *, lam_range: tuple[float, float, int] | None = None,
                    shrink=np.linspace(0.5, 0.99, 51), scale=True):
        if lam_range is not None:
            lam_min, lam_max, n_pts = lam_range
            lam_vec = np.linspace(lam_min, lam_max, int(n_pts), dtype=float)
            eig_part = None
            return lam_vec, eig_part
        s = X.std(0, ddof=1)
        keep = s > np.sqrt(np.finfo(s.dtype).eps)
        Z = (X[:, keep] - X[:, keep].mean(0)) / s[keep] if scale else \
            X[:, keep] - X[:, keep].mean(0)
        d = np.linalg.eigvalsh(Z.T @ Z)
        lam_vec = d.max() * (1 - shrink) / shrink
        eig_part = (d[:, None] / (d[:, None] + lam_vec[None, :])).sum(0)
        return lam_vec.astype(float), eig_part

    def mv_r2(y_true, y_pred):
        title = runtime.get_cfg().get_zscore_title().lower()
        no_z  = (title == "1_without_z_score")

        if no_z:               # ――― per-electrode R² and average ―――
            sse = np.square(y_true - y_pred).sum(axis=0)
            sst = np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum(axis=0) + 1e-12
            r2_each = 1.0 - sse / sst          # vector (n_electrodes,)
            return float(r2_each.mean())       # scalar

        else:                  # ――― standard pooled multivariate R² ―――
            sse = np.square(y_true - y_pred).sum()
            sst = np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum() + 1e-12
            return float(1.0 - sse / sst)

    def _lambda_for_fname(val: float) -> str:
        s = f"{val:.1e}".replace("+", "")
        return s.replace(".", "p")



