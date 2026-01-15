
from core.runtime import runtime
from core import constants
from methods.repetition_stability import RepetitionStabilityAnalyzer

# =============================================================================
# Repetition Stability Driver
# =============================================================================
# Runs the stability analysis for all monkeys, z-scores, and analysis types.
# Logic (paths, caching, plotting) is encapsulated in RepetitionStabilityAnalyzer.run_pipeline()

# example usage:
monkeys = constants.MONKEYS 
z_scores = sorted(constants.ZSCORE_INFO.keys()) 
analysis_types = ['residual'] 
block_size = 3
FORCE_RECOMPUTE = False

if __name__ == "__main__":
    pass
