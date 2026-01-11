"""
repetition_stability.py
-----------------------
Consolidated logic for "Repetition Stability Analysis".
Refactored from connectivity.py.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Sequence
from pathlib import Path
from core.runtime import runtime
from .pca import RegionPCA
from scipy.stats import spearmanr
from .rrr import RRRAnalyzer

# =============================================================================
# Repetition Stability Analyzer 
# =============================================================================

class RepetitionStabilityAnalyzer:
    """
    Consolidated module for "Repetition Stability Analysis".
    Analyze the stability of neural subspaces (within regions) and predictive subspaces (between regions)
    across repeated presentations of the same stimuli.
    """
    
    def __init__(self, monkey: str, z_code: int, analysis_type: str, group_size: int = 3):
        """
        Args:
            monkey: e.g., "Monkey N"
            z_code: Z-score index (1-4)
            analysis_type: "residual" or "window" (or others)
            group_size: Number of repetitions to group into one block (default 3).
        """
        self.monkey = monkey
        self.z_code = z_code
        self.analysis_type = analysis_type
        self.group_size = group_size
        
        # Configure runtime
        runtime.set_cfg(self.monkey, self.z_code)
        self.dm = runtime.get_data_manager()
        self.cfg = runtime.get_cfg()

    def get_blocked_data(self, region_id: int) -> List[np.ndarray]:
        """
        Loads repetition matrices and aggregates them into blocks.
        Result: List of (100*group_size, n_electrodes) matrices.
        """
        # 1. Load per-repetition matrices (Using DataManager new method)
        reps = self.dm.get_repetition_matrices(region_id, self.analysis_type)
        
        # 2. Group into blocks
        blocks = []
        n_blocks = len(reps) // self.group_size
        
        for b in range(n_blocks):
            # Stack 'group_size' repetitions vertically (more samples per block)
            idx_start = b * self.group_size
            idx_end = (b + 1) * self.group_size
            block_data = np.vstack(reps[idx_start:idx_end])
            blocks.append(block_data)
            
        print(f"[INFO] Region {region_id}: Loaded {len(reps)} reps -> {n_blocks} blocks (size {self.group_size}).")
        return blocks

    def _compute_overlap_msc(self, A: np.ndarray, B: np.ndarray) -> float:
        """
        Compute Mean Squared Cosine between two subspaces A and B.
        A, B: Orthonormal basis matrices (Features x D)
        """
        # Ensure orthonormal (though PCA output should be)
        # Using SVD on the "Interaction Matrix" (Ua.T @ Ub)
        C = A.T @ B
        svals = np.linalg.svd(C, compute_uv=False)
        return float(np.mean(svals ** 2))

    def _d95_from_perf(self, rrr_mean: np.ndarray, ridge_mean: float, d_max: int) -> int:
        """Helper for Connective Analysis (Semedo d95 logic)."""
        thr = 0.95 * float(ridge_mean)
        idx = np.where(rrr_mean >= thr)[0]
        return int(idx[0] + 1) if idx.size else int(d_max)

    # =========================================================================
    # 1. Single Region Stability (Intrinsic Subspace)
    # =========================================================================
    
    def analyze_region(self, region_id: int, fixed_d: int | None = None) -> Dict[str, Any]:
        """
        Computes pairwise overlap between blocks for a single region.
        Method: PCA on each block -> Overlap.
        """
        blocks = self.get_blocked_data(region_id)
        n_blocks = len(blocks)
        
        # 1. Compute Subspaces & Dimensionality
        subspaces = []
        dims = []
        
        for X in blocks:
            # PCA (using RegionPCA)
            pca = RegionPCA(centered=True).fit(X)
            d = pca.dimensionality
            dims.append(d)
            # Store full components for now, truncate later based on common D
            subspaces.append(pca.get_components()) 
            
        # 2. Determine Common D
        if fixed_d is not None:
            D_common = fixed_d
        else:
            D_common = max(1, int(np.floor(np.mean(dims)))) if dims else 1
            
        print(f"[*] Region {region_id} Stability: Using D={D_common} (Mean was {np.mean(dims):.1f})")

        # 3. Truncate and Compute Overlap Matrix
        overlap_matrix = np.eye(n_blocks, dtype=float)
        
        # Pre-truncate bases to D_common (Basis must be D x Features, transposed to Features x D for overlap calc)
        bases = [s[:D_common, :].T for s in subspaces] # RegionPCA returns (Components x Features)
        
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                ov = self._compute_overlap_msc(bases[i], bases[j])
                overlap_matrix[i, j] = ov
                overlap_matrix[j, i] = ov
                
        # 4. Stats
        rho, p_val = self._compute_lag_stats(overlap_matrix)
        
        return {
            "type": "region",
            "region_id": region_id,
            "matrix": overlap_matrix,
            "d_common": D_common,
            "spearman_rho": rho,
            "p_value": p_val,
            "monkey": self.monkey,
            "z_code": self.z_code,
            "method": self.analysis_type,
            "block_size": self.group_size,
            "subspaces": np.array(subspaces, dtype=object)
        }

    # =========================================================================
    # 2. Connection Stability (Predictive Subspace via RRR)
    # =========================================================================

    def analyze_connection(self, src_id: int, tgt_id: int, fixed_d: int | None = None) -> Dict[str, Any]:
        """
        Computes predictive stability between source and target blocks.
        Method: CV-RRR on Block(i) -> Predict -> SVD -> Subspace.
        """
        X_blocks = self.get_blocked_data(src_id)
        Y_blocks = self.get_blocked_data(tgt_id)
        n_blocks = len(X_blocks)
        
        subspaces = [] # Predicted subspaces
        r2_per_block = [] # Performance tracking
        temp_d95s = []
        
        # RRR Configuration
        d_max = 50 # User requested at least 50 components for reuse
        
        # 1. Process Blocks (Train RRR + Extract Subspace)
        for i in range(n_blocks):
            X, Y = X_blocks[i], Y_blocks[i]
            
            # CV-RRR to find lambda and performance
            # Utilizing existing project infrastructure
            perf = RRRAnalyzer.compute_performance(
                Y, X, d_max=d_max, outer_splits=3, inner_splits=3,
                alpha=None, random_state=42 + i
            )
            
            lam_opt = float(np.median(perf["lambdas"]))
            d95 = self._d95_from_perf(perf["rrr_R2_mean"], perf["ridge_R2_mean"], d_max)
            temp_d95s.append(d95)
            r2_per_block.append(perf["rrr_R2_mean"]) # Save full curve
            
            # Fit Ridge Model (B)
            # Center data
            Xc = X - X.mean(0)
            Yc = Y - Y.mean(0)
            
            # B = (X'X + lam*I)^-1 X'Y
            cov_xx = Xc.T @ Xc
            cov_xy = Xc.T @ Yc
            reg_eye = lam_opt * np.eye(Xc.shape[1])
            B_ridge = np.linalg.solve(cov_xx + reg_eye, cov_xy)
            
            # Extract Predictive Subspace
            # SVD of B gives directions in X
            U, _, _ = np.linalg.svd(B_ridge, full_matrices=False)
            subspaces.append(U) # Store full basis U (Feat_X x min(Fx, Fy))

        # 2. Determine Common D
        if fixed_d is not None:
            D_common = fixed_d
        else:
            D_common = max(1, int(np.floor(np.mean(temp_d95s))))
        
        print(f"[*] Connection {src_id}->{tgt_id}: Common D={D_common}")
        
        # 3. Extract final bases (Truncate U to D_common)
        final_bases = []
        block_r2_vals = []
        for i, U in enumerate(subspaces):
            # U is (Feat_X x K). We need (Feat_X x D_common).
            # Ensure we don't exceed ranks
            safe_D = min(D_common, U.shape[1])
            basis = U[:, :safe_D]
            final_bases.append(basis)
            
            # Store R2 at this D
            block_r2_vals.append(r2_per_block[i][safe_D-1] if safe_D>0 else 0)

        # 4. Compute Overlap
        overlap_matrix = np.eye(n_blocks, dtype=float)
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                ov = self._compute_overlap_msc(final_bases[i], final_bases[j])
                overlap_matrix[i, j] = ov
                overlap_matrix[j, i] = ov
                
        # 5. Stats
        rho, p_val = self._compute_lag_stats(overlap_matrix)
        
        return {
            "type": "connection",
            "src_id": src_id,
            "tgt_id": tgt_id,
            "matrix": overlap_matrix,
            "d_common": D_common,
            "block_r2s": block_r2_vals, # Mean R2 performance
            "spearman_rho": rho,
            "p_value": p_val,
            "monkey": self.monkey,
            "z_code": self.z_code,
            "method": self.analysis_type,
            "block_size": self.group_size,
            "subspaces": np.array(subspaces, dtype=object)
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _compute_lag_stats(self, O: np.ndarray, n_perms: int = 2000) -> Tuple[float, float]:
        """
        Spearman Correlation (Lag vs Overlap) + Permutation Test.
        """
        xs, ys, _, _, _ = self.extract_lag_data(O)
        
        if len(xs) < 2:
            return 0.0, 1.0
            
        # Observed
        rho_obs, _ = spearmanr(xs, ys)
        
        # Permutation Test (shuffling Block indices)
        N = O.shape[0]
        rng = np.random.default_rng(42)
        
        perm_rhos = np.zeros(n_perms)
        for i in range(n_perms):
            p_idx = rng.permutation(N)
            O_perm = O[p_idx][:, p_idx]
            xs_p, ys_p, _, _, _ = self.extract_lag_data(O_perm)
            r, _ = spearmanr(xs_p, ys_p)
            perm_rhos[i] = r
            
        # P-value: Fraction of permutations with rho <= rho_obs (one-tailed)
        # We expect rho to be negative (decay).
        p_val = (np.sum(perm_rhos <= rho_obs) + 1) / (n_perms + 1)
        
        return rho_obs, p_val

    @staticmethod
    def extract_lag_data(O: np.ndarray):
        """
        Extracts data for plotting: (Lags, Overlaps, UniqueLags, Means, SEMs)
        """
        N = O.shape[0]
        lags_all = []
        vals_all = []
        
        unique_lags = np.arange(1, N)
        means = []
        sems = []
        
        for l in unique_lags:
            diag = np.diagonal(O, offset=l)
            if diag.size > 0:
                lags_all.extend([l] * diag.size)
                vals_all.extend(diag)
                means.append(np.mean(diag))
                # SEM
                sems.append(np.std(diag, ddof=1) / np.sqrt(diag.size) if diag.size > 1 else 0.0)
            else:
                means.append(np.nan)
                sems.append(np.nan)
                
        return np.array(lags_all), np.array(vals_all), unique_lags, np.array(means), np.array(sems)

    def get_file_path(self, output_dir: str, region_id: int | None = None, src_tgt: Tuple[int, int] | None = None) -> Path:
        """
        Generate consistent filename for saving/loading.
        """
        from pathlib import Path
        p = Path(output_dir)
        mk = self.monkey.replace(" ", "")
        bt = f"blk{self.group_size}"
        
        if region_id is not None:
            nm = runtime.get_consts().REGION_ID_TO_NAME[region_id]
            fname = f"{mk}_{nm}_{self.analysis_type}_{bt}.npz"
        elif src_tgt is not None:
            s = runtime.get_consts().REGION_ID_TO_NAME[src_tgt[0]]
            t = runtime.get_consts().REGION_ID_TO_NAME[src_tgt[1]]
            fname = f"{mk}_{s}_to_{t}_{self.analysis_type}_{bt}.npz"
        else:
            raise ValueError("Must provide either region_id or src_tgt pair.")
            
        return p / fname

    def save_results(self, result: Dict[str, Any], output_dir: str):
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        
        if result["type"] == "region":
            save_path = self.get_file_path(output_dir, region_id=result["region_id"])
        else:
            save_path = self.get_file_path(output_dir, src_tgt=(result["src_id"], result["tgt_id"]))
            
        np.savez(save_path, **result)
        print(f"[Saved] {save_path}")

    def run_pipeline(self, force_recompute: bool = False):
        """
        Executes the full repetition stability analysis pipeline:
        1. Checks for existing cached data (unless force_recompute=True).
        2. Computes Region Stability (V1, V4, IT).
        3. Computes Connection Stability (V1->V4, V1->IT, V4->IT).
        4. Saves all results to disk.
        5. Generates and saves summary plots.
        """
        import numpy as np
        # Local import to avoid top-level circular dependency with visualization
        from .visualization import plot_repetition_stability

        # 1. Output Directories
        base_data_path = runtime.get_cfg().get_data_path()
        dir_region = base_data_path / "Repetition_Stability" / "Region"
        dir_conn = base_data_path / "Repetition_Stability" / "Connection"
        dir_plots = base_data_path / "Repetition_Stability" / "Repetition_Stability_plots"
        
        dir_region.mkdir(parents=True, exist_ok=True)
        dir_conn.mkdir(parents=True, exist_ok=True)
        dir_plots.mkdir(parents=True, exist_ok=True)

        all_results = []
        
        # 2. Region Analysis
        for rid in [1, 2, 3]: # V1, V4, IT
            fpath = self.get_file_path(str(dir_region), region_id=rid)
            
            if fpath.exists() and not force_recompute:
                print(f"    [SKIP] Region {rid} analysis already exists. Loading...")
                with np.load(fpath, allow_pickle=True) as data:
                    res = {k: data[k] for k in data.files}
                    # Scalar conversion for plotting
                    if 'region_id' in res: res['region_id'] = int(res['region_id'])
                    if 'p_value' in res: res['p_value'] = float(res['p_value'])
                    if 'spearman_rho' in res: res['spearman_rho'] = float(res['spearman_rho'])
                    # Convert string fields (loaded as 0-d arrays) to python strings
                    if 'method' in res: res['method'] = str(res['method'])
                    if 'monkey' in res: res['monkey'] = str(res['monkey'])
                    
                    all_results.append(res)
            else:
                print(f"    > Analyzing Region {rid}...")
                res = self.analyze_region(rid)
                self.save_results(res, str(dir_region))
                all_results.append(res)

        # 3. Connection Analysis
        pairs = [(1, 2), (1, 3), (2, 3)] # V1->V4, V1->IT, V4->IT
        for src, tgt in pairs:
            fpath = self.get_file_path(str(dir_conn), src_tgt=(src, tgt))
            
            if fpath.exists() and not force_recompute:
                    print(f"    [SKIP] Connection {src}->{tgt} analysis already exists. Loading...")
                    with np.load(fpath, allow_pickle=True) as data:
                        res = {k: data[k] for k in data.files}
                        if 'src_id' in res: res['src_id'] = int(res['src_id'])
                        if 'tgt_id' in res: res['tgt_id'] = int(res['tgt_id'])
                        if 'p_value' in res: res['p_value'] = float(res['p_value'])
                        if 'spearman_rho' in res: res['spearman_rho'] = float(res['spearman_rho'])
                        if 'method' in res: res['method'] = str(res['method'])
                        if 'monkey' in res: res['monkey'] = str(res['monkey'])
                        all_results.append(res)
            else:
                print(f"    > Analyzing Connection {src}->{tgt}...")
                res = self.analyze_connection(src, tgt)
                self.save_results(res, str(dir_conn))
                all_results.append(res)

        # 4. Visualization
        print(f"    > Generating Plots...")
        plot_repetition_stability(
            all_results,
            out_dir=str(dir_plots),
            show_errorbars=True
        )
