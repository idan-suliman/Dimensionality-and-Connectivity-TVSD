"""
run_dim_correlation.py
----------------------
Driver script for Dimensionality vs Correlation Analysis.
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from methods.dimensionality_correlation import run_standard_analysis

from core import constants
from methods.dimensionality_correlation import run_standard_analysis

# === Global Configuration ===
BLOCK_SIZE = 3          # Number of repetitions per block
FORCE = False           # Force recomputation (True) or use cache (False)
TYPE = "residual"       # 'residual' or 'window'

# Selection Codes: [1=V1->V4, 4=V1, 5=V4]
SELECTION = [3, 5, 6]

if __name__ == "__main__":
    monkeys = constants.MONKEYS
    z_scores = sorted(constants.ZSCORE_INFO.keys())
    
    print(f"--- Starting Global Dimensionality Correlation Analysis ---")
    print(f"Monkeys: {monkeys}")
    print(f"Z-Scores: {z_scores}")
    print(f"Selection: {SELECTION}")
    print("=" * 60)

    for monkey in monkeys:
        for z_score in z_scores:
            print(f"\n[Processing] {monkey} | Z={z_score} ...")
            try:
                run_standard_analysis(
                    monkey, z_score, TYPE, 
                    group_size=BLOCK_SIZE, 
                    force_recompute=FORCE, 
                    selection=SELECTION
                )
            except Exception as e:
                print(f"[ERROR] Failed for {monkey}, Z={z_score}: {e}")
                import traceback
                traceback.print_exc()

    print("\n[Done] All analyses completed.")
