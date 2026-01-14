# Repetition Stability

This package analyzes how stable neural representations are across repeated presentations (repetitions) of stimuli. It calculates overlaps of neural subspaces between different repetition blocks and correlates this with time (lag).

## File Overview

| File | Description | Relations |
| :--- | :--- | :--- |
| **[`analyzer.py`](./analyzer.py)** | Defines the `RepetitionStabilityAnalyzer` class. It manages the data flow, caching of intermediate results (like PCA/RRR models), and execution of stability checks. | The central controller; delegates to `region.py` and `connection.py`. |
| **[`region.py`](./region.py)** | Implements logic for "Region Stability" (Intrinsic). It computes how a single brain region's activity subspace overlaps with itself across different repetitions (using PCA). | Called by `analyzer.py` for single-region analysis. |
| **[`connection.py`](./connection.py)** | Implements logic for "Connection Stability" (Predictive). It computes how the predictive relationship (RRR) between two regions changes across repetitions. | Called by `analyzer.py` for cross-region analysis. |
| **[`stats.py`](./stats.py)** | Independent statistical functions for calculating overlap metrics (e.g., cosine similarity of subspaces) and correlating them with time lags. | Used by `region.py` and `connection.py`. |
| **[`pipeline.py`](./pipeline.py)** | Logic for the full analysis pipeline, including data preparation, running region/connection analyses sequentially, and triggering plots. | Higher-level orchestration used by drivers. |
| **[`utils.py`](./utils.py)** | Utilities for extraction of "Lag" data structure from matrix results, and handling consistent file naming/saving. | Used by `stats.py` and external visualization tools. |
| **[`__init__.py`](./__init__.py)** | Exposes the summary `RepetitionStabilityAnalyzer`. | Makes the package importable. |
