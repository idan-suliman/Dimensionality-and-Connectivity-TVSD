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

def get_file_path(analyzer, output_dir: str, region_id: int | None = None, src_tgt: Tuple[int, int] | None = None, suffix: str = "") -> Path:
    """
    Generate consistent filename for saving/loading.
    """
    p = Path(output_dir)
    mk = analyzer.monkey.replace(" ", "")
    bt = f"blk{analyzer.group_size}"
    
    if region_id is not None:
        nm = runtime.get_consts().REGION_ID_TO_NAME.get(region_id, f"Reg{region_id}")
        fname = f"{mk}_{nm}_{analyzer.analysis_type}_{bt}{suffix}.npz"
    elif src_tgt is not None:
        s = runtime.get_consts().REGION_ID_TO_NAME.get(src_tgt[0], f"Reg{src_tgt[0]}")
        t = runtime.get_consts().REGION_ID_TO_NAME.get(src_tgt[1], f"Reg{src_tgt[1]}")
        fname = f"{mk}_{s}_to_{t}_{analyzer.analysis_type}_{bt}{suffix}.npz"
    else:
        raise ValueError("Must provide either region_id or src_tgt pair.")
        
    return p / fname

def save_results(analyzer, result: Dict[str, Any], output_dir: str, fpath=None):
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    
    if fpath is None:
        if result["type"] == "region":
            save_path = analyzer.get_file_path(output_dir, region_id=result["region_id"])
        else:
            save_path = analyzer.get_file_path(output_dir, src_tgt=(result["src_id"], result["tgt_id"]))
    else:
        save_path = fpath
        
    np.savez(save_path, **result)
    print(f"[Saved] {save_path}")
