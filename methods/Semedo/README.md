# Semedo Analysis

This package specifically implements the logic to replicate and extend figures from Semedo et al. (2019). It focuses on comparing "Full" source populations against "Matched" subsets of neurons that share firing rate statistics with a target region.

## File Overview

| File | Description | Relations |
| :--- | :--- | :--- |
| **[`figure4.py`](./figure4.py)** | Implements the logic to generate **Figure 4**. This involves running RRR on the full source population vs. a matched subset. | Uses `RRRAnalyzer` to run the analysis. Does **not** use `matching.py` directly (subset logic is handled within the RRR pipeline). |
| **[`figure4_subset.py`](./figure4_subset.py)** | Implements **Figure 4 Subset**. This is an extended analysis that runs Figure 4 logic across many random subsets and multiple cross-validation runs. | Uses `matching.py` to explicitly calculate subsets each run. |
| **[`figure5b.py`](./figure5b.py)** | Implements **Figure 5B**. It analyzes the relationship between the dimensionality of the source activity and the dimensionality of the predictive subspace. | Loads pre-computed subset data from disk and does **not** use `matching.py` directly. |
| **[`matching.py`](./matching.py)** | Contains the algorithm to select a "Matched Subset" of neurons from a source region that statistically matches the firing rate distribution of a target region. | Explicitly used by `figure4_subset.py`. |
| **[`utils.py`](./utils.py)** | Utilities for building trial data specific to Semedo analyses, such as grouping trials by subset indices. | Helper for `figure4.py` and `figure5b.py`. |
| **[`__init__.py`](./__init__.py)** | Exposes the high-level build functions for the figures. | Makes the package importable. |
