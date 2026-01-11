
from core.runtime import runtime
from core import constants
from methods.repetition_stability import RepetitionStabilityAnalyzer

# =============================================================================
# Repetition Stability Driver
# =============================================================================
# Runs the stability analysis for all monkeys, z-scores, and analysis types.
# Logic (paths, caching, plotting) is encapsulated in RepetitionStabilityAnalyzer.run_pipeline()

monkeys = constants.MONKEYS 
z_scores = sorted(constants.ZSCORE_INFO.keys()) 
analysis_types = ['residual'] 
block_size = 3
FORCE_RECOMPUTE = False

if __name__ == "__main__":
    print(f"Starting Repetition Stability Analysis...")

    for monkey in monkeys:
        for z_score in z_scores:
            print(f"\n\n{'='*80}")
            print(f"CONFIGURATION: {monkey} | Z-Score Mode: {z_score}")
            print(f"{'='*80}")
            
            try:
                for analysis_type in analysis_types:
                    print(f"\n  --- Analysis Type: {analysis_type.upper()} ---")
                    
                    # Instantiate Analyzer
                    analyzer = RepetitionStabilityAnalyzer(monkey, z_score, analysis_type, block_size)
                    
                    # Execution (Handles Caching, Computation, Saving, Plotting)
                    analyzer.run_pipeline(force_recompute=FORCE_RECOMPUTE)
                            
            except Exception as e:
                print(f"\n!!! CRITICAL ERROR in configuration {monkey} | Z-Score {z_score}:")
                print(f"{e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n\nAll Repetition Stability Analyses Completed.")
