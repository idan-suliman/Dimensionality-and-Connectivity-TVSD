from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as pe
from core.runtime import runtime
from methods.rrr import RRRAnalyzer

def plot_rrr_ridge_comparison(
    source_region: int = 1,
    target_region: int = 2,
    *,
    d_max: int = 35,
    alpha: float | None = None,
    outer_splits: int = 3,
    inner_splits: int = 3,
    random_state: int = 0,
    analysis_types: tuple[str, ...] | None = None,
    match_to_target: bool = False,
    _external_X: np.ndarray | None = None,
    _external_Ys: dict[str, np.ndarray] | None = None,
    custom_title: str | None = None,
):
    """
    Plot RRR vs Ridge performance curves (Figure 5B style).
    """

    if analysis_types is None:
        analysis_types = runtime.consts.ANALYSIS_TYPES

    using_external = (_external_X is not None) and (_external_Ys is not None)
    cmap = {"window": "#C21807", "baseline100": "#1565C0", "residual": "#2E7D32"}

    n_info = len(analysis_types)
    fig_h  = 4.2 + 0.35 * n_info
    fig    = plt.figure(figsize=(7, fig_h))

    gs = gridspec.GridSpec(2, 1, height_ratios=[4.0, 0.35 * n_info], hspace=0.30)
    ax_main = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    dims, yvals = np.arange(1, d_max + 1), []
    sing_vals, lam_lines = [], []

    for at in analysis_types:
        col = cmap.get(at, "#000000")

        if using_external:
            X, Y = _external_X, _external_Ys[at]
        else:
            Y, X = RRRAnalyzer.build_mats(source_region, target_region, at, match_to_target=match_to_target)

        res = RRRAnalyzer.compute_performance(
            Y, X,
            d_max=d_max, alpha=alpha,
            outer_splits=outer_splits, inner_splits=inner_splits,
            random_state=random_state
        )

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
                        path_effects=[pe.Stroke(linewidth=1.8, foreground="black"), pe.Normal()])
            ax_main.axvline(d95, color=col, ls="--", lw=0.7, alpha=0.4)

        if alpha is None:
            Xc = X - X.mean(0, keepdims=True)
            sing_vals.append(int(round(np.linalg.svd(Xc, compute_uv=False)[0])))
            
            lam_vec, _ = RRRAnalyzer._lambda_grid(X)
            chosen = res["lambdas"]
            idx_pos = [int(np.abs(lam_vec - l).argmin()+1) for l in chosen]
            pairs = [f"{int(round(l))} ({p})" for l, p in zip(chosen, idx_pos)]
            if len(pairs) > 3:
                pairs = pairs[:3] + ["…"]
            lam_lines.append(", ".join(pairs))

    pad = 0.05 * (max(yvals) - min(yvals) if (yvals and max(yvals) > min(yvals)) else 0.05)
    ymin, ymax = (0.0, 1.0)
    if yvals:
         ymin, ymax = min(yvals)-pad, max(yvals)+pad
    ax_main.set_ylim(max(0.0, ymin), min(1.0, ymax))
    ax_main.set_xlabel("Predictive dimensions (d)", labelpad=6)        
    ax_main.set_ylabel(rf"Mean $R^2$  (CV: outer {outer_splits}, inner {inner_splits})")
    ax_main.grid(alpha=0.3)

    tgt_lbl = runtime.consts.REGION_ID_TO_NAME.get(target_region, str(target_region))
    ax_main.set_title(custom_title or
                    f"V1 → {'V1-match '+tgt_lbl if match_to_target else tgt_lbl}",
                    fontsize=12, pad=10)

    ax_info.axis("off")
    if alpha is None:
        step = 1.0 / (n_info + 1)
        for i, (at, s1, lam) in enumerate(zip(analysis_types, sing_vals, lam_lines)):
            ax_info.text(0.5, 1-step*(i+1),
                        f"{at}:  σ₁ = {s1}   |   λ* per fold:  {lam}",
                        ha="center", va="center",
                        fontsize=9, color=cmap.get(at, "k"))

    fig.tight_layout()
    tag = "nestedLam" if alpha is None else f"lam{RRRAnalyzer._lambda_for_fname(alpha)}"
    plot_dir = RRRAnalyzer._plot_dir(match_to_target)
    
    base = (f"{runtime.cfg.get_monkey_name().replace(' ', '')}_rrr_"
            f"{'target' if match_to_target else 'regular'}_"
            f"V1_to_{tgt_lbl}_{tag}")
            
    fname = plot_dir / f"{base}.png"
    fig.savefig(fname, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"[✓] RRR figure saved → {fname}")


def plot_lag_histogram(
    panels: list[tuple[np.ndarray, np.ndarray, float, str]],
    results: dict,
    regions: tuple[int, ...],
    monkey: str,
    method: str,
    zscore_title: str,
    out: str | None = None,
    show: bool = True
):
    """
    Plot histograms of permutation betas for Lag analysis.
    """
    nR = len(regions)
    fig, axes = plt.subplots(1, nR, figsize=(5.6 * nR, 4.6), dpi=150, constrained_layout=True)
    if nR == 1:
        axes = [axes]
    
    if not isinstance(axes, (list, np.ndarray)):
         axes = [axes]

    for ax, (hist, bin_edges, beta_obs, name), rid in zip(axes, panels, regions):
        pval = results[rid]["p_perm"]
        D    = results[rid]["D_common"]
        n_perm = results[rid]["n_perm"]
        
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.bar(centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.9)
        ax.axvline(beta_obs, color="k", lw=2.2, label=f"β_obs={beta_obs:.4g}")
        ax.set_xlabel("Permutation slope β (Monte-Carlo)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{name} (D={D})\n p_perm={pval:.3g}  |  perms={n_perm:,}", fontsize=12)
        ax.legend(frameon=False, fontsize=10)

    sup = f"{monkey} | Z={zscore_title} | {method} | WLS slope β (Δr>0)"
    fig.suptitle(sup, fontsize=14)

    if out:
        plt.savefig(out, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
