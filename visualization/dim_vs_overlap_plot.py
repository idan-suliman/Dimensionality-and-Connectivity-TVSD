from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

def plot_dim_vs_overlap(results: Dict[str, Any], output_path: str):
    """
    Plots the Overlap vs Cumulative Dimensions curves.
    
    Args:
        results: Dictionary output from methods/dimensionality_correlation/dim_vs_overlap.py
        output_path: Destination path for the image.
    """
    dims = results["dims"]
    overlap_src = results["overlap_src"]
    overlap_tgt = results["overlap_tgt"]
    meta = results.get("meta", {})
    
    src_name = meta.get("src", "Source")
    tgt_name = meta.get("tgt", "Target")
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Plot Source Overlap
    ax.plot(dims, overlap_src, label=f"Source Overlap ({src_name})", 
            color='#1f77b4', linewidth=3, marker='o', markersize=5, alpha=0.9)
            
    # Plot Target Overlap
    ax.plot(dims, overlap_tgt, label=f"Target Overlap ({tgt_name})", 
            color='#ff7f0e', linewidth=3, marker='s', markersize=5, alpha=0.9)
    
    # Styling
    ax.set_xlabel("Cumulative Dimensions", fontsize=24)
    ax.set_ylabel("Subspace Overlap", fontsize=24)
    
    ax.set_xlim(left=0, right=max(dims)+1)
    ax.set_ylim(0, 1)
    
    # Title
    title = f"Overlap with Predictive Subspace\n({src_name} $\\rightarrow$ {tgt_name})"
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.96)
    
    # Grid and Ticks
    ax.grid(True, linestyle='--', alpha=0.5)
    # ax.axhline(0, color='gray', linestyle='-', linewidth=0.8) # Removed per user request
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)
    
    # Legend
    ax.legend(loc='lower right', fontsize=18, frameon=True, framealpha=0.9)
    
    # Save
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.95)
    
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    print(f"[DimVsOverlapPlot] Saved to {output_path}")
