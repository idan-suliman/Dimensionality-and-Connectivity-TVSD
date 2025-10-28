# DimVsRepetition.py
from __future__ import annotations
from runtime import runtime
from rrr import RRR_Centered_matching
import numpy as np, pandas as pd, warnings, matplotlib.pyplot as plt
from pathlib import Path

class DimVsRepetition:
    """
    Plot #PCs (≥threshold) vs. repetition for three analysis types:
    • WINDOW      – region-specific response window
    • BASELINE100 – first 100 ms of trial
    • RESIDUAL    – stimulus-mean-subtracted window

    Uses TVSDAnalysis.configure(...) to get:
    - TVSDAnalysis.monkey_id
    - TVSDAnalysis.zscore_code, TVSDAnalysis.data_path
    Saves one PNG into:
    <TVSDAnalysis.data_path>/PLOTS/<threshold*100>_triple.png
    """
    
    @staticmethod
    def cv_rrr_summary_grouped(
        *,
        group_size   : int = 2,
        source_region: int = 1,
        target_region: int = 2,
        analysis_type: str = "window",
        outer_splits : int = 5,
        inner_splits : int = 5,
        alpha        : float | None = None,
        d_max_limit  : int | None = None,
        random_state : int = 0,
        n_runs       : int = 10,
        n_src        : int = 113,
        n_tgt        : int = 28,
        seed_subset  : int = 42,     # unused
        _max_resamples: int = 2000,  # unused
    ):
        """
        Same as before, with deterministic selection of subsets and minimal overlap (<=2 when possible),
        including printing/saving of overlap matrices.
        """
        if analysis_type not in {"window", "baseline100"}:
            raise ValueError("analysis_type must be 'window' or 'baseline100'.")

        # reps → blocks
        trials = runtime.get_cfg()._load_trials()
        if "rep_idx" not in trials[0]:
            for tr in trials:
                tr["rep_idx"] = int(tr["allmat_row"][3]) - 1
        rep_ids = np.unique([tr["rep_idx"] for tr in trials]); rep_ids.sort()
        blocks = [rep_ids[i:i + group_size] for i in range(0, len(rep_ids), group_size)]
        n_blocks = len(blocks)

        # pools
        rois  = runtime.get_cfg().get_rois()
        idx_X_all = np.where(rois == source_region)[0]
        idx_Y_all = np.where(rois == target_region)[0]
        if idx_X_all.size == 0 or idx_Y_all.size == 0:
            raise ValueError("No electrodes found for selected regions.")

        k_src = min(int(n_src), idx_X_all.size)
        k_tgt = min(int(n_tgt), idx_Y_all.size)
        if k_src <= 0 or k_tgt <= 0:
            raise ValueError("n_src/n_tgt must be positive.")
        if k_src > idx_X_all.size or k_tgt > idx_Y_all.size:
            raise ValueError("Requested n_src/n_tgt exceed available electrodes.")

        # window & matrix
        def _win(rid: int) -> slice:
            return slice(0, 100) if analysis_type == "baseline100" \
                else slice(*runtime.get_consts().REGION_WINDOWS[rid])

        def _mat(rid: int, idx: "np.ndarray", sub_tr: "list") -> "np.ndarray":
            w = _win(rid)
            return np.stack([tr["mua"][w][:, idx].mean(0, dtype=np.float64) for tr in sub_tr],
                            dtype=np.float64)

        # deterministic builder with overlap control
        def _build_subsets_overlap_constrained(pool_ids: "np.ndarray", k: int, R: int, target_ov: int = 2):
            pool_ids = np.asarray(pool_ids, dtype=int)
            N = pool_ids.size
            subsets: list[np.ndarray] = []
            use_count = np.zeros(N, dtype=int)
            id2pos = {int(e): i for i, e in enumerate(pool_ids)}

            if R * k <= N:
                for r in range(R):
                    start = r * k
                    sel_pos = np.arange(start, start + k, dtype=int)
                    sel = pool_ids[sel_pos]
                    subsets.append(sel)
                    use_count[sel_pos] += 1
                OM = np.zeros((R, R), dtype=int)
                return subsets, OM

            def _prev_membership_counts() -> np.ndarray:
                counts = np.zeros(N, dtype=int)
                for s in subsets:
                    counts[np.isin(pool_ids, s)] += 1
                return counts

            def _legal(e_id: int, overlaps: np.ndarray, limit: int) -> bool:
                if not subsets:
                    return True
                for j, prev in enumerate(subsets):
                    if e_id in prev and overlaps[j] + 1 > limit:
                        return False
                return True

            for r in range(R):
                chosen: list[int] = []
                overlaps = np.zeros(len(subsets), dtype=int)
                limit = target_ov
                while len(chosen) < k:
                    pmc = _prev_membership_counts()
                    order_pos = np.lexsort((pool_ids, use_count, pmc))
                    picked = False
                    for p in order_pos:
                        e = int(pool_ids[p])
                        if e in chosen:
                            continue
                        if _legal(e, overlaps, limit):
                            chosen.append(e)
                            for j, prev in enumerate(subsets):
                                if e in prev:
                                    overlaps[j] += 1
                            picked = True
                            if len(chosen) == k:
                                break
                    if not picked:
                        limit += 1
                        if limit > k:
                            break
                sel = np.asarray(chosen[:k], dtype=int)
                subsets.append(sel)
                for e in sel:
                    use_count[id2pos[int(e)]] += 1

            OM = np.zeros((R, R), dtype=int)
            for i in range(R):
                for j in range(i+1, R):
                    OM[i, j] = OM[j, i] = np.intersect1d(subsets[i], subsets[j], assume_unique=False).size
            return subsets, OM

        X_sets, OM_src = _build_subsets_overlap_constrained(idx_X_all, k_src, n_runs, target_ov=2)
        Y_sets, OM_tgt = _build_subsets_overlap_constrained(idx_Y_all, k_tgt, n_runs, target_ov=2)

        # overlap prints + memory
        def _print_viol(name, OM, k):
            viol = np.argwhere(np.triu(OM, 1) > 2)
            if viol.size:
                print(f"\n[OVERLAP] cv_rrr_summary_grouped | {name} | k={k} | n_runs={n_runs} | limit=2")
                for i, j in viol:
                    print(f"  run {i+1} ↔ run {j+1}: overlap={OM[i, j]}")
                return True
            return False

        printed = False
        printed |= _print_viol("SRC(V1)", OM_src, k_src)
        printed |= _print_viol(f"TGT({'V4' if target_region==2 else 'IT'})", OM_tgt, k_tgt)
        if printed:
            info = getattr(DimVsRepetition, "_last_overlap_info", {})
            info["cv_rrr_summary_grouped::SRC"] = OM_src
            info["cv_rrr_summary_grouped::TGT"] = OM_tgt
            DimVsRepetition._last_overlap_info = info  # type: ignore

        # -------- Runs and blocks ---------------------
        r2_runs  = np.zeros((n_runs, n_blocks), dtype=np.float64)
        d95_runs = np.zeros((n_runs, n_blocks), dtype=np.float64)
        lam_runs = np.zeros((n_runs, n_blocks), dtype=np.float64)
        sst_runs = np.zeros((n_runs, n_blocks), dtype=np.float64)
        sse_runs = np.zeros((n_runs, n_blocks), dtype=np.float64)
        rows = []

        for run in range(n_runs):
            idx_X = X_sets[run]
            idx_Y = Y_sets[run]
            for b_idx, block in enumerate(blocks):
                sub = [tr for tr in trials if tr["rep_idx"] in block]
                n_trials = len(sub)

                Y = _mat(target_region, idx_Y, sub)
                X = _mat(source_region, idx_X, sub)

                d_cap = min(X.shape[1], Y.shape[1])
                d_max = d_cap if (d_max_limit is None) else min(d_cap, int(d_max_limit))
                if d_max <= 0:
                    full_r2 = 0.0; d95 = 0; lam_used = np.nan; sst = 0.0; sse = 0.0
                else:
                    res = RRR_Centered_matching._performance_from_mats(
                        Y, X,
                        d_max        = d_max,
                        alpha        = alpha,
                        outer_splits = outer_splits,
                        inner_splits = inner_splits,
                        random_state = random_state + 1000*run + b_idx
                    )
                    r2_curve = np.asarray(res["rrr_R2_mean"], dtype=np.float64)
                    full_r2  = float(r2_curve[-1])
                    lam_used = float(np.nanmean(res["lambdas"]))
                    thr      = 0.95 * full_r2
                    idx95    = np.where(r2_curve >= thr)[0]
                    d95      = int(idx95[0] + 1) if idx95.size else d_max

                    Yc   = Y - Y.mean(0, keepdims=True)
                    Xc   = X - X.mean(0, keepdims=True)
                    B    = np.linalg.solve(Xc.T @ Xc + lam_used * np.eye(Xc.shape[1]), Xc.T @ Yc)
                    Yhat = Xc @ B
                    sst  = float((Yc ** 2).sum())
                    sse  = float(((Yc - Yhat) ** 2).sum())

                r2_runs[run, b_idx]  = full_r2
                d95_runs[run, b_idx] = d95
                lam_runs[run, b_idx] = lam_used
                sst_runs[run, b_idx] = sst
                sse_runs[run, b_idx] = sse

                label = f"{block[0]}-{block[-1]}" if len(block) > 1 else f"{block[0]}"
                rows.append(dict(
                    group     = label,
                    run       = run + 1,
                    reps_in   = len(block),
                    n_trials  = n_trials,
                    full_R2   = round(full_r2, 4),
                    d95       = int(d95),
                    lam       = lam_used,
                    SST       = round(sst, 1),
                    SSE       = round(sse, 1),
                ))

        df_runs = pd.DataFrame(rows)
        x_vals = sorted(df_runs["group"].unique(), key=lambda s: int(s.split('-')[0]))
        rep_numbers = np.array([int(s.split('-')[0]) + 1 for s in x_vals], dtype=int)
        d95_mean = np.array([df_runs[df_runs["group"] == b]["d95"].mean()     for b in x_vals], dtype=float)
        r2_mean  = np.array([df_runs[df_runs["group"] == b]["full_R2"].mean() for b in x_vals], dtype=float)

        return df_runs, rep_numbers, d95_mean, r2_mean
    
    @staticmethod
    def dims_per_repetition(
        *,
        region: int | str,
        analysis_type: str = "window",   # "window" | "baseline100"
        thr: float = 0.95,
        expected_reps: int = 30,
        n_stim_per_rep: int = 100,       # exactly 100 trials per repetition (per the dataset)
        subset_size: int | None = None,  # None → V4/IT=28, V1=all
        n_runs: int = 10,
        random_state: int | None = 0
    ) -> "np.ndarray":
        """
        Per-repetition dimensionality for a chosen region:
        number of PCs (via SVD) needed to explain ≥ `thr` of variance.

        This version is tailored to the verified structure:
        each trial dict has keys ['trial_idx','stimulus_id','day_id','allmat_row','mua'].
        Repetition index is taken as int(tr['allmat_row'][3]) - 1.
        There are exactly 30 repetitions, each with exactly 100 trials.
        """
        # ---------------- validation ----------------
        region_id = (int(region) if isinstance(region, (int, np.integer)) else {v:k for k,v in runtime.get_consts().REGION_ID_TO_NAME.items()}[str(region).upper()])
        at = analysis_type.lower().strip()
        if at not in {"window", "baseline100"}:
            raise ValueError("analysis_type must be 'window' or 'baseline100'.")
        if not (0.0 < thr <= 1.0):
            raise ValueError("thr must be in (0, 1].")
        if expected_reps <= 0 or n_runs <= 0 or n_stim_per_rep <= 0:
            raise ValueError("expected_reps, n_runs, and n_stim_per_rep must be positive.")

        trials = runtime.get_cfg()._load_trials()
        if not isinstance(trials, list) or len(trials) == 0:
            raise ValueError("Loaded trials are empty or malformed.")

        # ---------------- repetition index (dataset-specific) ----------------
        # Use ALLMAT row (position 3), minus 1 → 0-based repetition id
        try:
            rep_list = np.array([int(tr["allmat_row"][3]) - 1 for tr in trials], dtype=int)
        except Exception as e:
            raise KeyError("Could not extract repetition index from 'allmat_row'.") from e

        # Sort unique reps and keep exactly the first `expected_reps`
        rep_ids_all = np.unique(rep_list)
        rep_ids_all.sort()
        if rep_ids_all.size < expected_reps:
            raise ValueError(f"Found only {rep_ids_all.size} repetitions, expected {expected_reps}.")
        rep_ids = rep_ids_all[:expected_reps]

        # Group trials per repetition and validate exactly `n_stim_per_rep` per repetition
        rep_to_indices = {rid: np.flatnonzero(rep_list == rid) for rid in rep_ids}
        bad = {rid: idxs.size for rid, idxs in rep_to_indices.items() if idxs.size != n_stim_per_rep}
        if bad:
            details = ", ".join([f"rep {rid}: {c}" for rid, c in bad.items()])
            raise ValueError(
                f"Inconsistent #trials per repetition (expected {n_stim_per_rep} each): {details}"
            )

        # ---------------- time window ----------------
        # "baseline100" → first 100 ms; otherwise use region-specific window
        w = slice(0, 100) if at == "baseline100" else slice(*runtime.get_consts().REGION_WINDOWS[region_id])

        # ---------------- region electrodes mask ----------------
        # Map dataset order → physical order via TVSDAnalysis.mapping, then select the region id
        reg_mask = (runtime.get_cfg().get_rois() == region_id)
        n_elec_region = int(reg_mask.sum())
        if n_elec_region == 0:
            raise ValueError(f"No electrodes found for region id {region_id}.")
        col_idx = np.flatnonzero(reg_mask)

        # ---------------- vectorize all trials (time-mean per electrode) ----------------
        # Build a big matrix: rows = trials, cols = electrodes in the region
        n_total = len(trials)
        M_all = np.empty((n_total, n_elec_region), dtype=np.float64)
        for i, tr in enumerate(trials):
            # Average over the selected time window → one value per electrode
            mua = tr["mua"]
            V = mua[w][:, col_idx].mean(axis=0, dtype=np.float64)
            M_all[i, :] = V

        # ---------------- helper: minimal #PCs to reach `thr` ----------------
        def _dim_from_matrix(M2D: "np.ndarray", thr_val: float) -> int:
            # Drop electrodes that contain any NaN across the 100 trials
            ok = ~np.isnan(M2D).any(axis=0)
            if not np.any(ok):
                return 0
            X = M2D[:, ok]
            # Center columns (per electrode)
            X -= X.mean(axis=0, keepdims=True)
            # SVD is numerically robust even when n_samples < n_features
            _, s, _ = np.linalg.svd(X, full_matrices=False)
            if s.size == 0 or not np.any(s):
                return 0
            e = s**2
            cum = np.cumsum(e) / e.sum()
            # first index where cumulative variance ≥ thr → +1 = #PCs
            return int(np.searchsorted(cum, thr_val) + 1)

        # ---------------- deterministic subsets with minimal overlap across runs ----------------
        # Default subset size: V4/IT = 28, V1 = all electrodes in the region
        if subset_size is None:
            k = min(28, n_elec_region) if region_id in (2, 3) else n_elec_region
        else:
            if subset_size <= 0:
                raise ValueError("subset_size must be positive.")
            k = min(int(subset_size), n_elec_region)

        MAX_OV = 2  # desired max overlap between any two runs
        pool_ids = np.arange(n_elec_region, dtype=int)

        def _build_subsets_overlap_constrained(pool_ids: np.ndarray, k: int, R: int, max_ov: int = 2):
            """
            Deterministic greedy builder:
            - Creates R subsets of size k from pool_ids.
            - Tries to keep pairwise overlap ≤ max_ov when feasible.
            - If infeasible, gradually relaxes the tolerance (tol++).
            Returns: (list of index arrays, overlap_matrix [R x R]).
            """
            N = pool_ids.size
            subsets: list[np.ndarray] = []
            use_count = np.zeros(N, dtype=int)  # usage per position in pool_ids
            id2pos = {int(e): i for i, e in enumerate(pool_ids)}

            def _prev_membership_counts() -> np.ndarray:
                # how many previous subsets contain each electrode
                counts = np.zeros(N, dtype=int)
                for s in subsets:
                    counts[np.isin(pool_ids, s)] += 1
                return counts

            def _legal(e_id: int, overlaps: np.ndarray, tol: int) -> bool:
                # check if adding e_id would violate overlap limit with any previous subset
                if not subsets:
                    return True
                for j, prev in enumerate(subsets):
                    if e_id in prev and overlaps[j] + 1 > max_ov + tol:
                        return False
                return True

            for r in range(R):
                chosen: list[int] = []
                overlaps = np.zeros(len(subsets), dtype=int)  # current overlap vs. each previous subset
                tol = 0
                while len(chosen) < k:
                    pmc = _prev_membership_counts()
                    # prefer electrodes with fewer previous appearances, then lower use_count, then lower id
                    order = np.lexsort((pool_ids, use_count, pmc))
                    picked = False
                    for p in order:
                        e_id = int(pool_ids[p])
                        if e_id in chosen:
                            continue
                        if _legal(e_id, overlaps, tol):
                            chosen.append(e_id)
                            # update overlap counters
                            for j, prev in enumerate(subsets):
                                if e_id in prev:
                                    overlaps[j] += 1
                            picked = True
                            if len(chosen) == k:
                                break
                    if not picked:
                        # no legal candidate under current tolerance → relax tolerance
                        tol += 1
                        if tol > k:
                            warnings.warn("Overlap constraint infeasible; best-effort selection.", RuntimeWarning)
                            break
                sel = np.asarray(chosen[:k], dtype=int)
                subsets.append(sel)
                # update usage counts
                for e in sel:
                    use_count[id2pos[int(e)]] += 1

            Rn = len(subsets)
            OM = np.zeros((Rn, Rn), dtype=int)
            for i in range(Rn):
                for j in range(i+1, Rn):
                    OM[i, j] = OM[j, i] = np.intersect1d(subsets[i], subsets[j], assume_unique=False).size
            return subsets, OM

        subsets, OM = _build_subsets_overlap_constrained(pool_ids, k, n_runs, MAX_OV)
        if (OM > MAX_OV).any():
            region_name = {1:"V1", 2:"V4", 3:"IT"}.get(region_id, str(region_id))
            print(f"\n[OVERLAP] dims_per_repetition | region={region_name} | k={k} | n_runs={n_runs} | limit={MAX_OV}")
            viol = np.argwhere(np.triu(OM, 1) > MAX_OV)
            for i, j in viol:
                print(f"  run {i+1} ↔ run {j+1}: overlap={OM[i, j]}")
        info = getattr(DimVsRepetition, "_last_overlap_info", {})
        info[f"dims_per_repetition::{region_id}"] = OM
        DimVsRepetition._last_overlap_info = info  # type: ignore

        # ---------------- compute dimensionality per repetition and average across runs ----------------
        dims_runs = np.zeros((n_runs, expected_reps), dtype=np.float64)
        for run, sub_cols in enumerate(subsets):
            for r_idx, rid in enumerate(rep_ids):
                idxs = rep_to_indices[rid]            # 100 trial indices for this repetition
                M_rep = M_all[idxs][:, sub_cols]      # shape: (100 trials, k electrodes)
                dims_runs[run, r_idx] = _dim_from_matrix(M_rep, thr)

        dims_mean = dims_runs.mean(axis=0, dtype=np.float64).astype(np.float32)
        return dims_mean

    @staticmethod
    def d95_v4_it(
        *,
        group_size   : int              = 1,
        n_runs       : int              = 2,
        n_src        : int              = 112,
        n_tgt        : int              = 28,
        n_bins       : int              = 25,
        analysis_type: str              = "window",
        outer_splits : int              = 5,
        inner_splits : int              = 5,
        alpha        : float | None     = None,
        d_max_limit  : int | None       = None,
        seed_subset  : int              = 42,     # unused
        random_state : int              = 0,
        show_plot    : bool             = True,
        save_path    : Path | None    = None,
        max_overlap_src: int | None     = None,   # kept for signature; ignored (builder targets 2)
        max_overlap_tgt: int | None     = None,
        _max_resamples: int             = 2000,
    ) -> pd.DataFrame:
        """
        Deterministic subsets with minimal overlap (≈<=2 when feasible).
        Builds matched-vs-remaining V1 splits via histogram matching to V4/IT,
        then runs RRR: Y=matched(V1), X=remaining(V1).
        Returns df and per-block means of d95 and R² for targets V4/IT.
        """
        # ---------- load trials and split to blocks ----------
        trials = runtime.get_cfg()._load_trials()
        if "rep_idx" not in trials[0]:
            for tr in trials:
                tr["rep_idx"] = int(tr["allmat_row"][3]) - 1
        rep_ids = np.unique([tr["rep_idx"] for tr in trials]); rep_ids.sort()
        blocks = [rep_ids[i:i+group_size] for i in range(0, len(rep_ids), group_size)]

        if analysis_type == "residual" and group_size == 1:
            warnings.warn("Residual on a single repetition may collapse to zeros.", RuntimeWarning)

        # ---------- electrode pools ----------
        rois   = runtime.get_cfg().get_rois()
        idx_V1 = np.where(rois == 1)[0]
        idx_V4 = np.where(rois == 2)[0]
        idx_IT = np.where(rois == 3)[0]
        if (idx_V1.size < n_src) or (idx_V4.size < n_tgt) or (idx_IT.size < n_tgt):
            raise ValueError("n_src / n_tgt too large – not enough electrodes available.")

        # ---------- windows & matrices ----------
        def _slc(rid: int) -> slice:
            return slice(0, 100) if analysis_type == "baseline100" \
                else slice(*runtime.get_consts().REGION_WINDOWS[rid])

        def _mat(idx: "np.ndarray", trs: "list", rid: int) -> "np.ndarray":
            """
            Returns matrix shape [n_trials_in_block, len(idx)] with time-mean per trial.
            If analysis_type == 'residual', subtracts column mean.
            """
            w = _slc(rid)
            M = np.stack([tr["mua"][w][:, idx].mean(0).astype(np.float64) for tr in trs], dtype=np.float64)
            if analysis_type == "residual":
                M -= M.mean(0, keepdims=True)
            return M

        # ---------- histogram matching (fixed & deterministic) ----------
        def _hist_match(src_means: "np.ndarray", tgt_means: "np.ndarray", pool_idx: "np.ndarray") -> "np.ndarray":
            """
            Choose n_tgt indices from 'pool_idx' whose distribution (by bins of src_means)
            best matches the target distribution (bins of tgt_means). If a bin lacks
            candidates, borrow from nearest bins (±1, ±2, ...) deterministically.
            """
            # 1) binning with clipping
            mn = float(min(src_means.min(), tgt_means.min()))
            mx = float(max(src_means.max(), tgt_means.max()))
            edges = np.linspace(mn, mx, n_bins + 1)
            bin_src = np.clip(np.searchsorted(edges, src_means, side="right") - 1, 0, n_bins - 1)
            bin_tgt = np.clip(np.searchsorted(edges, tgt_means, side="right") - 1, 0, n_bins - 1)

            # 2) demand per bin from target; candidates per bin from source
            need  = np.bincount(bin_tgt, minlength=n_bins)           # target demand per bin
            cands = [np.where(bin_src == b)[0].tolist() for b in range(n_bins)]

            chosen_src_pos = []  # positions inside src_means/pool_idx

            # 3) primary fill from same bin
            for b in range(n_bins):
                take = min(need[b], len(cands[b]))
                if take > 0:
                    chosen_src_pos += cands[b][:take]
                    cands[b] = cands[b][take:]
                    need[b] -= take

            # 4) borrow from nearest bins until all needs are met or bins exhausted
            step = 1
            while np.any(need > 0) and step < n_bins:
                # iterate only bins that still need
                for b in np.where(need > 0)[0]:
                    if need[b] == 0:
                        continue
                    for sign in (-1, 1):
                        nb = b + sign * step
                        if 0 <= nb < n_bins and len(cands[nb]) > 0:
                            take = min(need[b], len(cands[nb]))
                            chosen_src_pos += cands[nb][:take]
                            cands[nb] = cands[nb][take:]
                            need[b] -= take
                            if need[b] == 0:
                                break
                step += 1

            # 5) still short? fill from whatever remains (deterministic order)
            if len(chosen_src_pos) < n_tgt:
                rest = []
                for lst in cands:
                    rest += lst
                if rest:
                    chosen_src_pos += rest[:(n_tgt - len(chosen_src_pos))]

            # 6) trim to n_tgt and map back to pool indices
            chosen_src_pos = chosen_src_pos[:n_tgt]
            chosen_src_pos = list(dict.fromkeys(chosen_src_pos))  # de-dup if any
            if len(chosen_src_pos) < n_tgt:
                # last fallback: top-up from any remaining not-yet-chosen indices
                all_pos = list(range(src_means.size))
                extra = [p for p in all_pos if p not in chosen_src_pos]
                chosen_src_pos += extra[:(n_tgt - len(chosen_src_pos))]

            return pool_idx[np.array(chosen_src_pos[:n_tgt], dtype=int)]

        # ---------- subset builder (deterministic, low-overlap) ----------
        def _build_subsets_overlap_constrained(pool_ids: "np.ndarray", k: int, R: int, target_ov: int = 2):
            pool_ids = np.asarray(pool_ids, dtype=int)
            N = pool_ids.size
            subsets: list[np.ndarray] = []
            use_count = np.zeros(N, dtype=int)
            id2pos = {int(e): i for i, e in enumerate(pool_ids)}

            # easy case: enough electrodes to avoid overlap entirely
            if R * k <= N:
                for r in range(R):
                    start = r * k
                    sel_pos = np.arange(start, start + k, dtype=int)
                    sel = pool_ids[sel_pos]
                    subsets.append(sel)
                    use_count[sel_pos] += 1
                OM = np.zeros((R, R), dtype=int)
                return subsets, OM

            def _prev_membership_counts() -> np.ndarray:
                counts = np.zeros(N, dtype=int)
                for s in subsets:
                    counts[np.isin(pool_ids, s)] += 1
                return counts

            def _legal(e_id: int, overlaps: np.ndarray, limit: int) -> bool:
                if not subsets:
                    return True
                for j, prev in enumerate(subsets):
                    if e_id in prev and overlaps[j] + 1 > limit:
                        return False
                return True

            for r in range(R):
                chosen: list[int] = []
                overlaps = np.zeros(len(subsets), dtype=int)
                limit = target_ov
                while len(chosen) < k:
                    pmc = _prev_membership_counts()
                    order_pos = np.lexsort((pool_ids, use_count, pmc))  # few prev-appearances → low use_count → low id
                    picked = False
                    for p in order_pos:
                        e = int(pool_ids[p])
                        if e in chosen:
                            continue
                        if _legal(e, overlaps, limit):
                            chosen.append(e)
                            for j, prev in enumerate(subsets):
                                if e in prev:
                                    overlaps[j] += 1
                            picked = True
                            if len(chosen) == k:
                                break
                    if not picked:
                        limit += 1
                        if limit > k:
                            break
                sel = np.asarray(chosen[:k], dtype=int)
                subsets.append(sel)
                for e in sel:
                    use_count[id2pos[int(e)]] += 1

            OM = np.zeros((R, R), dtype=int)
            for i in range(R):
                for j in range(i+1, R):
                    OM[i, j] = OM[j, i] = np.intersect1d(subsets[i], subsets[j], assume_unique=False).size
            return subsets, OM

        # build V1/V4/IT subsets
        src_sets, OM_src = _build_subsets_overlap_constrained(idx_V1, n_src, n_runs, target_ov=2)
        v4_sets,  OM_v4  = _build_subsets_overlap_constrained(idx_V4, n_tgt, n_runs, target_ov=2)
        it_sets,  OM_it  = _build_subsets_overlap_constrained(idx_IT, n_tgt, n_runs, target_ov=2)

        # print overlap violations (if any)
        def _print_viol(name, OM, k):
            viol = np.argwhere(np.triu(OM, 1) > 2)
            if viol.size:
                print(f"\n[OVERLAP] d95_v4_it | {name} | k={k} | n_runs={n_runs} | limit=2")
                for i, j in viol:
                    print(f"  run {i+1} ↔ run {j+1}: overlap={OM[i, j]}")
                return True
            return False

        printed = False
        printed |= _print_viol("SRC(V1)", OM_src, n_src)
        printed |= _print_viol("TGT(V4)", OM_v4,  n_tgt)
        printed |= _print_viol("TGT(IT)", OM_it,  n_tgt)
        if printed:
            info = getattr(DimVsRepetition, "_last_overlap_info", {})
            info["d95_v4_it::SRC"] = OM_src
            info["d95_v4_it::V4"]  = OM_v4
            info["d95_v4_it::IT"]  = OM_it
            DimVsRepetition._last_overlap_info = info  # type: ignore

        # ---------- main loops ----------
        rows = []
        for run in range(1, n_runs + 1):
            src_pool = src_sets[run - 1]
            tgt_V4   = v4_sets[run - 1]
            tgt_IT   = it_sets[run - 1]

            for blk in blocks:
                sub_tr = [tr for tr in trials if tr["rep_idx"] in blk]

                # compute per-electrode means (over trials in this block)
                m_src = _mat(src_pool, sub_tr, 1).mean(0)
                m_V4  = _mat(tgt_V4,  sub_tr, 2).mean(0)
                m_IT  = _mat(tgt_IT,  sub_tr, 3).mean(0)

                # histogram-match V1 to V4/IT to form Y; remaining V1 form X
                match_V4 = _hist_match(m_src, m_V4, src_pool)
                match_IT = _hist_match(m_src, m_IT, src_pool)
                remain_V4 = src_pool[~np.isin(src_pool, match_V4)]
                remain_IT = src_pool[~np.isin(src_pool, match_IT)]

                # quick sanity checks (can be commented out)
                assert len(np.intersect1d(match_V4, remain_V4)) == 0
                assert len(match_V4) == n_tgt and len(remain_V4) == n_src - n_tgt
                assert len(np.intersect1d(match_IT, remain_IT)) == 0
                assert len(match_IT) == n_tgt and len(remain_IT) == n_src - n_tgt

                for tgt_lbl, match_idx, remain_idx in (
                    ("V4", match_V4, remain_V4),
                    ("IT", match_IT, remain_IT)
                ):
                    Y = _mat(match_idx,  sub_tr, 1)  # Y = matched V1 electrodes
                    X = _mat(remain_idx, sub_tr, 1)  # X = remaining V1 electrodes

                    # ---- FIX 1: safe d_max (cannot exceed rank implied by Y and X) ----
                    d_max_auto = min(X.shape[1], Y.shape[1])
                    d_max = d_max_auto if d_max_limit is None else min(int(d_max_limit), d_max_auto)

                    if d_max <= 0:
                        full_r2 = 0.0
                        d95     = 0
                    else:
                        res = RRR_Centered_matching._performance_from_mats(
                            Y, X,
                            d_max        = d_max,
                            alpha        = alpha,
                            outer_splits = outer_splits,
                            inner_splits = inner_splits,
                            random_state = random_state + run,
                        )
                        r2_curve = np.asarray(res["rrr_R2_mean"], dtype=np.float64)
                        full_r2  = float(r2_curve[-1])
                        thr95    = 0.95 * full_r2
                        idx95    = np.where(r2_curve >= thr95)[0]
                        d95      = int(idx95[0] + 1) if idx95.size else d_max

                    rows.append(dict(
                        block   = f"{blk[0]}-{blk[-1]}" if len(blk) > 1 else f"{blk[0]}",
                        run     = run,
                        target  = tgt_lbl,
                        d95     = d95,
                        full_R2 = round(full_r2, 4),
                    ))

        df = pd.DataFrame(rows)

        # ---------- optional plot ----------
        if show_plot:
            import seaborn as sns, matplotlib.pyplot as plt
            sns.set_style("whitegrid")
            plt.figure(figsize=(8, 4))
            x_vals = sorted(df["block"].unique(), key=lambda s: int(s.split('-')[0]))
            for tgt, col in zip(("V4", "IT"), ("#1f77b4", "#ff7f0e")):
                y = [df[(df["block"] == b) & (df["target"] == tgt)]["d95"].mean()
                    for b in x_vals]
                plt.plot(x_vals, y, "-o", label=tgt, color=col)
            plt.xlabel("Repetition" if group_size == 1 else "Block")
            plt.ylabel("dominant dimension (d95)")
            max_labels = 15
            if len(x_vals) > max_labels:
                stride = int(np.ceil(len(x_vals) / max_labels))
                for i, label in enumerate(plt.gca().get_xticklabels()):
                    if i % stride: label.set_visible(False)
            plt.xticks(fontsize=6, rotation=45)
            plt.title(f"{runtime.get_cfg().get_monkey_name} | {runtime.get_cfg().get_zscore_title()} | {analysis_type.upper()}")
            plt.legend()
            plt.tight_layout()
            if save_path is not None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, facecolor="white")
            plt.show()

        # ---------- aggregate per block ----------
        x_vals = sorted(df["block"].unique(), key=lambda s: int(s.split('-')[0]))
        rep_numbers = np.array([int(s.split('-')[0]) + 1 for s in x_vals], dtype=int)
        d95_v4 = np.array([df[(df["block"] == b) & (df["target"] == "V4")]["d95"].mean()
                        for b in x_vals], dtype=float)
        d95_it = np.array([df[(df["block"] == b) & (df["target"] == "IT")]["d95"].mean()
                        for b in x_vals], dtype=float)
        r2_v4  = np.array([df[(df["block"] == b) & (df["target"] == "V4")]["full_R2"].mean()
                        for b in x_vals], dtype=float)
        r2_it  = np.array([df[(df["block"] == b) & (df["target"] == "IT")]["full_R2"].mean()
                        for b in x_vals], dtype=float)

        return df, rep_numbers, d95_v4, d95_it, r2_v4, r2_it

