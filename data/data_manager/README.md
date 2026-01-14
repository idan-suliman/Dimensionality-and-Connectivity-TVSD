# Data Manager

This package is responsible for loading, processing, and structuring neural recording data for analysis. It serves as the central data access point for all methods in the codebase.

## File Overview

| File | Description | relations |
| :--- | :--- | :--- |
| **[`core.py`](./core.py)** | Defines the main `DataManager` class. It orchestrates the entire data pipeline by delegating specific tasks (loading, building, splitting) to the other modules in this package. | The entry point used by external drivers and methods. |
| **[`loader.py`](./loader.py)** | Handles the low-level loading of trial data from disk/cache. It manages the retrieval of raw properties (spikes, behavior) before processing. | Called by `core.py` to fetch initial data. |
| **[`builder.py`](./builder.py)** | Contains logic to construct "Trial Matrices" (Numpy arrays) from raw loaded data. This includes time-binning, mean subtraction, and residual calculation. | Transforms data loaded by `loader.py`. |
| **[`splitter.py`](./splitter.py)** | Implements logic to split or group data, specifically separating trials by "Repetition" blocks for stability analyses. | Operates on matrices produced by `builder.py`. |
| **[`__init__.py`](./__init__.py)** | Exposes the `DataManager` class to the rest of the application. | Makes the package importable. |
