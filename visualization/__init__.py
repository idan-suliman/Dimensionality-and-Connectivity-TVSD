from .utils import jitter, square_limits, labeled_dot, smart_label
from .semedo import SemedoFigures
from .dim_corr import DimCorrVisualizer
from .repetition import plot_repetition_stability, plot_overlap_matrix, plot_all_overlaps_grid
from .general import GeneralPlots
from .rrr import plot_rrr_ridge_comparison, plot_lag_histogram

# For backwards compatibility with external scripts, expose classes/funcs directly
# if they were previously available under methods.visualization
