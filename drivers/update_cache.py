import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core import constants
from methods.repetition_stability import RepetitionStabilityAnalyzer

# Configuration
monkeys = constants.MONKEYS 
z_scores = sorted(constants.ZSCORE_INFO.keys()) 
analysis_types = ['residual'] # Can add 'trial' if needed, but 'residual' is primary
block_size = 3

print(f"--- GLOBAL CACHE UPDATE STARTING ---")
print(f"Targeting: {monkeys}")
print(f"Z-Scores: {z_scores}")
print(f"Analysis Types: {analysis_types}")
print("-" * 50)

for monkey in monkeys:
    for z_score in z_scores:
        for analysis_type in analysis_types:
            print(f"\n[Processing] {monkey} | Z={z_score} | {analysis_type} ...")
            try:
                analyzer = RepetitionStabilityAnalyzer(monkey, z_score, analysis_type, block_size)
                # Force Recompute to ensure subspaces (50 dims) are saved
                analyzer.run_pipeline(force_recompute=True)
            except Exception as e:
                print(f"[ERROR] Failed for {monkey}, Z={z_score}: {e}")
                import traceback
                traceback.print_exc()

print(f"\n--- GLOBAL CACHE UPDATE COMPLETED ---")
