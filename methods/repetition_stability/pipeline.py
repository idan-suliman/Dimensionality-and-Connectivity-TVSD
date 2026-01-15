from __future__ import annotations
import numpy as np
from core.runtime import runtime


def run_pipeline(analyzer, regions=None, connections=None, force_recompute: bool = False):
    """
    Executes the full repetition stability analysis pipeline:
    1. Checks for existing cached data (unless force_recompute=True).
    2. Computes Region Stability (V1, V4, IT).
    3. Computes Connection Stability (V1->V4, V1->IT, V4->IT).
    4. Saves all results to disk.
    5. Generates and saves summary plots.
    """
    # Local import to avoid top-level circular dependency with visualization
    # We will need to update this import if visualization is also refactored. 
    # For now, it might be methods.visualization.repetition_stability or similar. 
    # But since we haven't refactored visualization yet, it is methods.visualization.
    # Note: Refactoring Visualization is NEXT. So this import might break soon if we proceed linearly.
    # However, if we assume the old visualization.py is still there (it is), it's fine. 
    # But wait, we plan to refactor visualization too. 
    # Ideally, we should import from the new structure if possible, or keep it dynamic.
    from visualization import plot_repetition_stability

    # 1. Base Data Path
    base_data_path = runtime.cfg.get_data_path()


    all_results = []
    
    # Defaults
    if regions is None:
        regions = [1, 2, 3] # V1, V4, IT
    
    if connections is None:
        connections = [(1, 2), (1, 3), (2, 3)] # V1->V4, V1->IT, V4->IT

    # 2. Region Analysis
    for rid in regions:
        fpath = runtime.paths.get_rep_stability_path(
            analyzer.monkey, 
            analyzer.analysis_type, 
            analyzer.group_size, 
            region_id=rid,
            output_dir=base_data_path
        )
        
        if fpath.exists() and not force_recompute:
            print(f"    [SKIP] Region {rid} analysis already exists. Loading...")
            with np.load(fpath, allow_pickle=True) as data:
                res = {k: data[k] for k in data.files}
                # Scalar conversion for plotting
                if 'region_id' in res: res['region_id'] = int(res['region_id'])
                if 'p_value' in res: res['p_value'] = float(res['p_value'])
                if 'spearman_rho' in res: res['spearman_rho'] = float(res['spearman_rho'])
                if 'method' in res: res['method'] = str(res['method'])
                if 'monkey' in res: res['monkey'] = str(res['monkey'])
                
                all_results.append(res)
        else:
            print(f"    > Analyzing Region {rid}...")
            res = analyzer.analyze_region(rid)
            analyzer.save_results(res, base_data_path)
            all_results.append(res)

    # 3. Connection Analysis
    for src, tgt in connections:
        fpath = runtime.paths.get_rep_stability_path(
            analyzer.monkey, 
            analyzer.analysis_type, 
            analyzer.group_size, 
            src_tgt=(src, tgt),
            output_dir=base_data_path
        )
        
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
            res = analyzer.analyze_connection(src, tgt)
            analyzer.save_results(res, base_data_path)
            all_results.append(res)

    # 4. Visualization
    print(f"    > Generating Plots...")
    plot_repetition_stability(
        all_results,
        out_dir=str(base_data_path),
        show_errorbars=True
    )
