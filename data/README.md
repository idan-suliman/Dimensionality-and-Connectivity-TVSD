# Data & Data Management

This directory contains the code responsible for managing, loading, and processing the neural data used by the analysis methods.

## Components

| Component | Description |
| :--- | :--- |
| **[`data_manager`](./data_manager/)** | The core Python package that provides the API for loading trials, building matrices, and splitting data. **This is the primary interface for all analysis code.** |

## Data Flow
1.  **Raw Data**: Loaded from disk/cache (managed by `loader.py` in `data_manager`).
2.  **Processing**: Converted into numpy arrays (managed by `builder.py` in `data_manager`).
3.  **Consumption**: Consumed by methods in `../methods/` via the `DataManager` class.
