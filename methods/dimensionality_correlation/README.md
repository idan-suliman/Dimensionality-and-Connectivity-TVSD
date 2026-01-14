# Dimensionality Correlation

This package implements the analysis of how the dimensionality of neural population activity correlates with other metrics (like time or performance). It focuses on "Intrinsic" (Region-based) and "Predictive" (Connection-based) stability curves.

## File Overview

| File | Description | Relations |
| :--- | :--- | :--- |
| **[`analyzer.py`](./analyzer.py)** | Defines the `DimCorrAnalyzer` class. This class acts as the stateful controller for the analysis, managing file paths, caching, and running the core computation loops. | Orchestrates `curves.py` logic; uses `utils.py`. |
| **[`curves.py`](./curves.py)** | Contains the mathematical logic for computing stability curves. It calculates rank-correlation between subspace overlaps and other variables over expanding dimensions. | Pure logic called by `analyzer.py`. |
| **[`standard_analysis.py`](./standard_analysis.py)** | Provides high-level functions to run the "Standard" analysis suite (e.g., comparing V1, V4, IT). It sets up the default parameters and sequences of region/connection checks. | The main entry point for drivers (e.g., `drivers/run_dim_correlation.py`). |
| **[`utils.py`](./utils.py)** | Helper functions for generating standardized filenames, loading specific cache files, and handling basic matrix operations required by the analyzer. | Used by `analyzer.py` and `standard_analysis.py`. |
| **[`__init__.py`](./__init__.py)** | Exposes the main analyzer and standard analysis functions. | Makes the package importable. |
