# Reduced Rank Regression (RRR)

This package implements Reduced Rank Regression and Ridge Regression models. It provides the core mathematical machinery for analyzing predictive relationships between brain regions.

## File Overview

| File | Description | Relations |
| :--- | :--- | :--- |
| **[`analyzer.py`](./analyzer.py)** | Defines `RRRAnalyzer`. This is a static class (collection of methods) that serves as the main API for running RRR/Ridge stats. It delegates specific tasks to `matrices.py` and `performance.py`. | The entry point for other methods (e.g., Semedo, Repetition Stability). |
| **[`performance.py`](./performance.py)** | Computes model performance (Cross-Validated $R^2$). It handles the nested cross-validation loops for both RRR and Ridge regression. | The core computational engine called by `analyzer.py`. |
| **[`matrices.py`](./matrices.py)** | Responsible for building the Source ($X$) and Target ($Y$) matrices required for regression. It handles signal processing steps like time-delay embedding. | Prepares data for `performance.py`. Note: Currently relies on `methods/matchingSubset.py` for subset selection logic. |
| **[`optimization.py`](./optimization.py)** | Contains hyperparameter optimization logic, specifically for generating lambda grids and calculating regularization strengths (Alpha) for Ridge/RRR. | Used by `performance.py` during model fitting. |
| **[`metrics.py`](./metrics.py)** | Helper functions for calculating scoring metrics (like multivariate $R^2$) and formatting results. | Used by `performance.py` for evaluation. |
| **[`__init__.py`](./__init__.py)** | Exposes the `RRRAnalyzer`. | Makes the package importable. |
