"""
dimensionality_correlation_logic.py
-----------------------------------
Logic for analyzing how Repetition Stability (Spearman Correlation of Lag Graph)
changes as a function of the number of dimensions considered.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
from core.runtime import runtime
from .pca import RegionPCA
from .rrr import RRRAnalyzer
from .repetition_stability import RepetitionStabilityAnalyzer

def run_standard_analysis(monkey: str, z_code: int, analysis_type: str, group_size: int = 3, force_recompute: bool = False, selection: List[int] | None = None):
    """
    Orchestrates the standard Dimensionality Correlation analysis:
    - Regions: V1, V4, IT
    - Connections: V1->V4, V1->IT, V4->IT
    - Generates and saves the plot.
    - selection: List of codes [1-6] to run. If None, runs all.
      1: V1->V4, 2: V1->IT, 3: V4->IT, 4: V1, 5: V4, 6: IT
    """
    # Setup
    runtime.set_cfg(monkey, z_code)
    analyzer = DimCorrAnalyzer(monkey, z_code, analysis_type, group_size=group_size)
    results = []
    
    if selection is None:
        selection = [1, 4, 5]
    
    name_to_id = runtime.get_consts().REGION_NAME_TO_ID
    
    print(f"=== Dimensionality vs Correlation Analysis: {monkey}, Z={z_code}, {analysis_type} ===")

    # Standard Regions mappings (Code -> Name)
    region_map = {4: "V1", 5: "V4", 6: "IT"}
    
    for code, r_name in region_map.items():
        if code in selection:
            if r_name not in name_to_id: continue
            rid = int(name_to_id[r_name])
            res = analyzer.analyze_region_curve(rid, max_dims=30, force_recompute=force_recompute)
            res['src'] = r_name
            res['type'] = 'region'
            results.append(res)
        
    # Standard Connections mappings (Code -> (Src, Tgt))
    conn_map = {1: ("V1", "V4"), 2: ("V1", "IT"), 3: ("V4", "IT")}
    
    for code, (src, tgt) in conn_map.items():
        if code in selection:
            if src not in name_to_id or tgt not in name_to_id: continue
            sid, tid = int(name_to_id[src]), int(name_to_id[tgt])
            res = analyzer.analyze_connection_curve(sid, tid, max_dims=30, force_recompute=force_recompute)
            res['src'] = src
            res['tgt'] = tgt
            results.append(res)
        
    # Plotting
    # Local import to avoid circular dependency
    # Plotting
    # Local import to avoid circular dependency
    from .visualization import DimCorrVisualizer
    out_dir = runtime.get_cfg().get_data_path() / "Dimensionality_vs_Correlation"
    
    # User Request: Append selection IDs to filename to avoid overwrites
    suffix_sel = "_" + "_".join(map(str, sorted(selection)))
    out_name = f"DimCorr_{monkey.replace(' ','')}_{analysis_type}_Standard{suffix_sel}.png"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    DimCorrVisualizer.plot_curves(
        results,
        title="Correlation vs. Dimensionality",
        subtitle=f"{monkey} | Z={z_code} | {analysis_type.upper()}",
        output_path=str(out_dir / out_name)
    )
    print(f"[Done] Plot saved to {out_dir / out_name}")


class DimCorrAnalyzer:
    """
    Analyzer that computes the 'Dimensionality vs Stability Correlation' curve.
    Uses RepetitionStabilityAnalyzer for data loading and core stats metrics.
    """
    
    def __init__(self, monkey: str, z_code: int, analysis_type: str = "residual", group_size: int = 3):
        self.monkey = monkey
        self.z_code = z_code
        self.analysis_type = analysis_type
        self.group_size = group_size
        
        # Configure runtime
        runtime.set_cfg(self.monkey, self.z_code)
        
        # Initialize RepetitionStabilityAnalyzer to reuse its logic
        self.rep_stab = RepetitionStabilityAnalyzer(monkey, z_code, analysis_type, group_size)
    
    def get_file_path(self, output_dir: str, region_id: int | None = None, src_tgt: Tuple[int, int] | None = None, suffix: str = "_dim_corr") -> Path:
        """
        Generate consistent filename for saving/loading.
        Delegates to rep_stab but appends suffix.
        """
        # We can implement this by calling rep_stab.get_file_path and modifying the name, 
        # or just keep local implementation if it's cleaner to handle suffix here.
        # Local implementation is clean enough and specific to this module's suffix requirement.
        p = Path(output_dir)
        mk = self.monkey.replace(" ", "")
        bt = f"blk{self.group_size}"
        
        if region_id is not None:
            nm = runtime.get_consts().REGION_ID_TO_NAME[region_id]
            fname = f"{mk}_{nm}_{self.analysis_type}_{bt}{suffix}.npz"
        elif src_tgt is not None:
            s = runtime.get_consts().REGION_ID_TO_NAME[src_tgt[0]]
            t = runtime.get_consts().REGION_ID_TO_NAME[src_tgt[1]]
            fname = f"{mk}_{s}_to_{t}_{self.analysis_type}_{bt}{suffix}.npz"
        else:
            raise ValueError("Must provide either region_id or src_tgt pair.")
            
        return p / fname

    def _compute_overlap_matrix(self, bases: List[np.ndarray]) -> np.ndarray:
        """
        Computes the overlap matrix between all pairs of bases.
        bases: List of (Features x D) arrays.
        """
        n_blocks = len(bases)
        overlap_matrix = np.eye(n_blocks, dtype=float)
        
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                # Reuse the overlap calculation from RepetitionStabilityAnalyzer if available
                # or just use standard calculation. RepStab has _compute_overlap_msc.
                # Since bases are already orthonormal (from PCA/SVD), we can directly use it.
                # bases[i] is Features x D. _compute_overlap_msc expects Features x D.
                # However, RepStab._compute_overlap_msc is nominally "private". We'll access it.
                ov = self.rep_stab._compute_overlap_msc(bases[i], bases[j])
                
                overlap_matrix[i, j] = ov
                overlap_matrix[j, i] = ov
                
        return overlap_matrix

    def analyze_region_curve(self, region_id: int, max_dims: int = 30, 
                             precomputed_blocks: Optional[List[np.ndarray]] = None,
                             force_recompute: bool = False) -> Dict[str, Any]:
        """
        Computes the curve for a single region (Intrinsic Stability).
        Returns dictionary with 'dims', 'rhos', 'p_vals'.
        """
        # 1. Paths
        source_dir = runtime.get_cfg().get_data_path() / "Repetition_Stability" / "Region"
        out_dir = source_dir / f"DimCorr_blk{self.group_size}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fpath = self.get_file_path(str(out_dir), region_id=region_id)
        
        # 2. Check Cache
        if not force_recompute and fpath.exists():
            print(f"    [SKIP] Region {region_id} analysis already exists. Loading from {fpath.name}...")
            with np.load(fpath, allow_pickle=True) as data:
                return {k: data[k] for k in data.files}

        print(f"[*] Analyzing Region {region_id} (Intrinsic)...")
        
        full_subspaces = None
        
        # 3. Try Reuse Subspaces from Repetition Stability (Source Directory)
        if not force_recompute:
            # Look for the file WITHOUT _dim_corr suffix in the SOURCE directory
            rep_stab_path = self.get_file_path(str(source_dir), region_id=region_id, suffix="")
            if rep_stab_path.exists():
                try:
                    with np.load(rep_stab_path, allow_pickle=True) as data:
                        if 'subspaces' in data:
                            full_subspaces = [x.astype(float) for x in data['subspaces']]
                            print(f"    [Reuse] Loaded subspaces from Repetition Stability: {rep_stab_path.name}")
                        else:
                            print(f"    [Cache] Found {rep_stab_path.name} but 'subspaces' key is missing. Recomputing...")
                except Exception as e:
                    print(f"    [Reuse] Failed to load subspaces: {e}")

        # 4. If no subspaces, compute them
        if full_subspaces is None:
            # Get Data
            if precomputed_blocks is not None:
                blocks = precomputed_blocks
                print(f"    - Used precomputed blocks.")
            else:
                 # REUSE: Use RepetitionStabilityAnalyzer to get blocks
                blocks = self.rep_stab.get_blocked_data(region_id)
            
            # Compute Full PCA
            full_subspaces = [] 
            for X in blocks:
                pca = RegionPCA(centered=True).fit(X)
                full_subspaces.append(pca.get_components())

        # 5. Iterate Dimensions
        results_dims = []
        results_rhos = []
        results_pvals = []
        
        limit_d = min([s.shape[0] for s in full_subspaces])
        actual_max = min(max_dims, limit_d)
        
        for d in range(1, actual_max + 1):
            # RegionPCA returns (Components x Features), we need (Features x D)
            sliced_bases = [s[:d, :].T for s in full_subspaces]
            
            O_mat = self._compute_overlap_matrix(sliced_bases)
            
            # REUSE: Use RepetitionStabilityAnalyzer for lag stats
            rho, p_val = self.rep_stab._compute_lag_stats(O_mat)
            
            results_dims.append(d)
            results_rhos.append(rho)
            results_pvals.append(p_val)
            
        result = {
            "type": "region",
            "id": region_id,
            "dims": results_dims,
            "rhos": results_rhos,
            "p_vals": results_pvals,
            "monkey": self.monkey,
            "method": self.analysis_type
        }
        
        # 6. Save
        np.savez(fpath, **result)
        print(f"    [Saved] {fpath}")
        
        return result

    def analyze_connection_curve(self, src_id: int, tgt_id: int, max_dims: int = 30,
                                 src_blocks: Optional[List[np.ndarray]] = None,
                                 tgt_blocks: Optional[List[np.ndarray]] = None,
                                 force_recompute: bool = False) -> Dict[str, Any]:
        """
        Computes the curve for a connection (Predictive Stability).
        """
        # 1. Paths
        source_dir = runtime.get_cfg().get_data_path() / "Repetition_Stability" / "Connection"
        out_dir = source_dir / f"DimCorr_blk{self.group_size}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fpath = self.get_file_path(str(out_dir), src_tgt=(src_id, tgt_id))
        
        # 2. Check Cache
        if not force_recompute and fpath.exists():
             print(f"    [SKIP] Connection {src_id}->{tgt_id} already exists. Loading from {fpath.name}...")
             with np.load(fpath, allow_pickle=True) as data:
                return {k: data[k] for k in data.files}

        print(f"[*] Analyzing Connection {src_id}->{tgt_id} (Predictive)...")
        
        full_pred_subspaces = None

        # 3. Try Reuse Subspaces (Source Directory)
        if not force_recompute:
            rep_stab_path = self.get_file_path(str(source_dir), src_tgt=(src_id, tgt_id), suffix="")
            if rep_stab_path.exists():
                try:
                    with np.load(rep_stab_path, allow_pickle=True) as data:
                        if 'subspaces' in data:
                            full_pred_subspaces = [x.astype(float) for x in data['subspaces']]
                            print(f"    [Reuse] Loaded subspaces from Repetition Stability: {rep_stab_path.name}")
                        else:
                            print(f"    [Cache] Found {rep_stab_path.name} but 'subspaces' key is missing. Recomputing...")
                except Exception as e:
                    print(f"    [Reuse] Failed to load subspaces: {e}")

        # 4. If no subspaces, compute them
        if full_pred_subspaces is None:
            # Get Data
            if src_blocks is not None:
                X_blocks = src_blocks
            else:
                X_blocks = self.rep_stab.get_blocked_data(src_id) # REUSE
                
            if tgt_blocks is not None:
                 Y_blocks = tgt_blocks
            else:
                 Y_blocks = self.rep_stab.get_blocked_data(tgt_id) # REUSE
                 
            n_blocks = len(X_blocks)
            
            # Compute RRR
            full_pred_subspaces = []
            
            for i in range(n_blocks):
                X, Y = X_blocks[i], Y_blocks[i]
                
                perf = RRRAnalyzer.compute_performance(
                    Y, X, d_max=max_dims, outer_splits=2, inner_splits=2,
                    alpha=None, random_state=42+i
                )
                lam_opt = float(np.median(perf["lambdas"]))
                
                Xc = X - X.mean(0)
                Yc = Y - Y.mean(0)
                reg_eye = lam_opt * np.eye(Xc.shape[1])
                cov_xx = Xc.T @ Xc
                cov_xy = Xc.T @ Yc
                B = np.linalg.solve(cov_xx + reg_eye, cov_xy)
                
                U, _, _ = np.linalg.svd(B, full_matrices=False)
                full_pred_subspaces.append(U) # (Features x Rank)
            
        # 5. Iterate Dimensions
        results_dims = []
        results_rhos = []
        results_pvals = []
        
        limit_d = min([u.shape[1] for u in full_pred_subspaces])
        actual_max = min(max_dims, limit_d)
        
        for d in range(1, actual_max + 1):
            sliced_bases = [u[:, :d] for u in full_pred_subspaces]
            
            O_mat = self._compute_overlap_matrix(sliced_bases)
            
            # REUSE: Use RepetitionStabilityAnalyzer for lag stats
            rho, p_val = self.rep_stab._compute_lag_stats(O_mat)
            
            results_dims.append(d)
            results_rhos.append(rho)
            results_pvals.append(p_val)
            
        result = {
            "type": "connection",
            "src": src_id,
            "tgt": tgt_id,
            "dims": results_dims,
            "rhos": results_rhos,
            "p_vals": results_pvals,
            "monkey": self.monkey,
            "method": self.analysis_type
        }
        
        # 6. Save
        np.savez(fpath, **result)
        print(f"    [Saved] {fpath}")
        
        return result
