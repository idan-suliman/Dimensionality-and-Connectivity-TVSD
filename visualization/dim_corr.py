from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from core.runtime import runtime
from typing import Any

class DimCorrVisualizer:
    
    @staticmethod
    def plot_curves(results_list: list[dict[str, Any]], 
                   title: str, 
                   subtitle: str,
                   output_path: str):
        """
        Plots multiple curves on a single figure.
        results_list: List of result dicts (from DimCorrAnalyzer).
        """
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(results_list))))
        
        for idx, res in enumerate(results_list):
            dims = np.array(res["dims"])
            rhos = np.array(res["rhos"])
            pvals = np.array(res["p_vals"])
            
            # Label generation
            if res["type"] == "region":
                if "src" in res:
                    lbl = res["src"]
                else:
                    try:
                        name = runtime.consts.REGION_ID_TO_NAME[res['id']]
                        lbl = name
                    except:
                        lbl = f"Region {res['id']}"
            else:
                lbl = f"{res['src']}->{res['tgt']}"
                
            color = colors[idx]
            
            # Plot Curve
            ax.plot(dims, rhos, label=lbl, color=color, linewidth=2, marker='o', markersize=4, alpha=0.7)
            
            # Highlight Significant Points
            sig_mask = pvals < 0.01
            if np.any(sig_mask):
                ax.scatter(dims[sig_mask], rhos[sig_mask], 
                           color=color, s=50, edgecolors='k', zorder=5, 
                           label=None) # Don't duplicate label
                           
        ax.set_xlabel("Number of Dimensions (Cumulative)", fontsize=16, fontweight='bold')
        ax.set_ylabel("Spearman Correlation", fontsize=16, fontweight='bold')
        
        # Titles
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.96)
        ax.set_title(subtitle, fontsize=14, color='gray', pad=10)
        
        # Significance Annotation
        ax.text(0.95, 0.95, "Bold points: Permutation p < 0.05", 
                transform=ax.transAxes, ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=14, width=2)
        
        # Legend: Outside, Centered, Large
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                 ncol=3, fontsize=14, frameon=False)
        
        # Save
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Adjust layout to accommodate external elements
        fig.subplots_adjust(top=0.88, bottom=0.20, left=0.15, right=0.95)
        
        fig.savefig(str(p), dpi=150)
        plt.close(fig)
        print(f"[Plot] Saved to {output_path}")
