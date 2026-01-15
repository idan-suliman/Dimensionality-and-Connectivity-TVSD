from __future__ import annotations
from typing import List, Optional
import matplotlib.pyplot as plt
from core.runtime import runtime
from .analyzer import DimCorrAnalyzer
from visualization import DimCorrVisualizer


def run_standard_analysis(monkey: str, z_code: int, analysis_type: str, group_size: int = 3, force_recompute: bool = False, selection: List[int] | None = None):
    """
    Orchestrates the standard Dimensionality Correlation analysis:
    - Regions: V1, V4, IT
    - Connections: V1->V4, V1->IT, V4->IT
    - Generates and saves the plot.
    """
    analyzer = DimCorrAnalyzer(monkey, z_code, analysis_type, group_size)
    results = []

    # 1. Define Layout of what to run
    # Format: (label, func, kwargs)
    # We want specific order for plotting: 
    # V1->V4, V1->IT, V4->IT (Predictive)
    # V1, V4, IT (Intrinsic)
    
    to_run = [
        # Predictive
        {"id": 1, "title": "V1 -> V4", "type": "conn", "args": (1, 2)},
        {"id": 2, "title": "V1 -> IT", "type": "conn", "args": (1, 3)},
        {"id": 3, "title": "V4 -> IT", "type": "conn", "args": (2, 3)},
        # Intrinsic
        {"id": 4, "title": "V1 Intrinsic", "type": "reg", "args": (1,)},
        {"id": 5, "title": "V4 Intrinsic", "type": "reg", "args": (2,)},
        {"id": 6, "title": "IT Intrinsic", "type": "reg", "args": (3,)},
    ]
    
    if selection is not None:
        to_run = [x for x in to_run if x["id"] in selection]

    for item in to_run:
        print(f"--- Processing {item['title']} ---")
        if item["type"] == "conn":
            src, tgt = item["args"]
            res = analyzer.analyze_connection_curve(src, tgt, force_recompute=force_recompute)
        else:
            reg = item["args"][0]
            res = analyzer.analyze_region_curve(reg, force_recompute=force_recompute)
            
        res["plot_label"] = item["title"] # Add label for plotter
        results.append(res)
        
    # Plotting
    # We need a Visualizer
    # Collect IDs for filename
    run_ids = [str(x["id"]) for x in to_run]
    items_str = "-".join(run_ids)
    
    # Use get_dim_corr_path for summary plot path
    fp = runtime.paths.get_dim_corr_path(
        monkey,
        analysis_type,
        group_size,
        suffix=f"_items_{items_str}_DimCorr",
        extension=".png"
    )
    
    DimCorrVisualizer.plot_curves(
        results_list=results, 
        title=f"Dimensionality vs Stability Correlation ({monkey})",
        subtitle=f"Z={z_code}, {analysis_type}",
        output_path=str(fpath)
    )
    print(f"Saved figure to {fpath}")
    return results
