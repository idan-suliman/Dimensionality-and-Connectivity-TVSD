# Dimensionality Correlation

This package implements the analysis of how the dimensionality of neural population activity correlates with other metrics (like time or performance). It focuses on "Intrinsic" (Region-based) and "Predictive" (Connection-based) stability curves.

## File Overview

| File | Description | Relations |
| :--- | :--- | :--- |
| **[`analyzer.py`](./analyzer.py)** | Defines the `DimCorrAnalyzer` class. This class acts as the stateful controller for the analysis, managing file paths, caching, and running the core computation loops. | Orchestrates `curves.py` logic; uses `utils.py`. |
| **[`curves.py`](./curves.py)** | Contains the mathematical logic for computing stability curves. It calculates rank-correlation between subspace overlaps and other variables over expanding dimensions. | Pure logic called by `analyzer.py`. |
| **[`standard_analysis.py`](./standard_analysis.py)** | Provides high-level functions to run the "Standard" analysis suite (e.g., comparing V1, V4, IT). It sets up the default parameters and sequences of region/connection checks. | The main entry point for drivers (e.g., `drivers/run_dim_correlation.py`). |
| **[`dim_vs_overlap.py`](./dim_vs_overlap.py)** | Computes the overlap (Mean Squared Cosine) between PCA-defined intrinsic subspaces and RRR-defined predictive subspaces (U and V) as a function of dimension. | Standalone logic used by `drivers/driver_dim_vs_overlap.py`. |
| **[`utils.py`](./utils.py)** | Helper functions for generating standardized filenames, loading specific cache files, and handling basic matrix operations required by the analyzer. | Used by `analyzer.py` and `standard_analysis.py`. |
| **[`__init__.py`](./__init__.py)** | Exposes the main analyzer and standard analysis functions. | Makes the package importable. |

## Implementation Details

### Predictive Stability Optimization
In `analyze_connection_curve`, we optimize the computation of predictive subspaces for varying dimensions `d`. 
Instead of re-solving RRR for each rank `d` (which changes the constraint), we utilize the property that standard RRR corresponds to PCA on the OLS predictions.
Optimization steps:
1.  Compute Ridge/OLS solution $B_{ols} = \arg\min ||Y - X B||$ once per block.
2.  Compute full prediction $\hat{Y}_{full} = X B_{ols}$.
3.  Compute SVD of $\hat{Y}_{full}$ to get basis $U_{full}$.
4.  For any rank `d`, the RRR predictive subspace is simply the first `d` columns of $U_{full}$.
This avoids re-fitting the regression model inside the dimension loop.

### Overlap Analysis
In `dim_vs_overlap.py`, we assess how similar the "Intrinsic" subspace (captured by PCA on the region's own activity) is to the "Predictive" subspace (captured by RRR components relevant for inter-region communication).
We compute the Mean Squared Cosine (MSC) between:
1.  **Source Side**: PCA(X) vs. Predictive Subspace $U$ (from $B = UV^T$).
2.  **Target Side**: PCA(Y) vs. Predictive Subspace $V$.
This reveals whether the dimensions that explain the most variance in a region are the same ones used for communicating with downstream/upstream regions.

