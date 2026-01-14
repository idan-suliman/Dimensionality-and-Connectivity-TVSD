# Visualization

This package centrally houses all plotting logic code for the project. It segregates plotting routines by the scientific method they visualize, ensuring that analysis code (in other `methods/` packages) remains pure and focused on computation.

## File Overview

| File | Description | Relations |
| :--- | :--- | :--- |
| **[`semedo.py`](./semedo.py)** | Contains plotting functions specifically for [Semedo Analysis](../Semedo/). Includes the complex multi-panel layouts for Figure 4 and Figure 5. | Used by `methods/Semedo/figure*.py`. |
| **[`dim_corr.py`](./dim_corr.py)** | Contains plotting functions for [Dimensionality Correlation](../dimensionality_correlation/). Plots stability curves (Spearman correlation vs Dimensions). | Used by `methods/dimensionality_correlation/standard_analysis.py`. |
| **[`repetition.py`](./repetition.py)** | Visualizers for [Repetition Stability](../repetition_stability/). Includes overlap vs. time lag plots and heatmap grids. | Used by `methods/repetition_stability/pipeline.py`. |
| **[`rrr.py`](./rrr.py)** | Plots specific to [RRR](../rrr/) performance, such as R2 comparison curves and lag histograms. | Used by `methods/rrr/analyzer.py`. |
| **[`general.py`](./general.py)** | General-purpose exploratory plots, such as raw amplitude traces, timecourses per region, and electrode distributions. | Used for data exploration drivers. |
| **[`utils.py`](./utils.py)** | Shared plotting utilities like `smart_label` (text placement), `jitter` (scatter plot noise), and axis formatting helpers. | Used by all other modules in this package. |
| **[`__init__.py`](./__init__.py)** | Exposes all plotting functions in a flat namespace. | Allows `from methods.visualization import ...` |
