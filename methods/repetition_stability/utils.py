from __future__ import annotations
import numpy as np
from typing import List, Any, Tuple, Dict
from pathlib import Path
from core.runtime import runtime

def get_blocked_data(analyzer, region_id: int) -> List[np.ndarray]:
    """
    Loads repetition matrices and aggregates them into blocks.
    Result: List of (100*group_size, n_electrodes) matrices.
    """
    # 1. Load per-repetition matrices (Using DataManager)
    reps = analyzer.dm.get_repetition_matrices(region_id, analyzer.analysis_type)
    
    # 2. Group into blocks
    blocks = []
    n_blocks = len(reps) // analyzer.group_size
    
    for b in range(n_blocks):
        # Stack 'group_size' repetitions vertically (more samples per block)
        idx_start = b * analyzer.group_size
        idx_end = (b + 1) * analyzer.group_size
        block_data = np.vstack(reps[idx_start:idx_end])
        blocks.append(block_data)
        
    print(f"[INFO] Region {region_id}: Loaded {len(reps)} reps -> {n_blocks} blocks (size {analyzer.group_size}).")
    return blocks

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

def save_results(analyzer, result: Dict[str, Any], output_dir: str, fpath=None):
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    
    if fpath is None:
        if result["type"] == "region":
            save_path = runtime.paths.get_rep_stability_path(
                analyzer.monkey,
                analyzer.analysis_type,
                analyzer.group_size,
                region_id=result["region_id"],
                output_dir=output_dir
            )
        else:
            save_path = runtime.paths.get_rep_stability_path(
                analyzer.monkey,
                analyzer.analysis_type,
                analyzer.group_size,
                src_tgt=(result["src_id"], result["tgt_id"]),
                output_dir=output_dir
            )
    else:
        save_path = fpath
        
    np.savez(save_path, **result)
    print(f"[Saved] {save_path}")
