# TVSD Codebase Architecture Report

This document outlines the structure of the refactored TVSD codebase. The system is designed with a clear separation of concerns: **Core** infrastructure, **Data** management, **Methods** (analysis logic), **Drivers** (execution entry points), and **Utilities**.

## 1. `core/` Package (Infrastructure)
*Essential system configurations, constants, and global state management.*

### `core.runtime`
*   **Purpose**: Central singleton managing the global application state.
*   **Key Components**:
    *   `RUNTIME` class: Holds `CONFIG` and `DataManager` instances.
    *   `set_cfg(monkey, z_score)`: Initializes the environment for a run.

### `core.config`
*   **Purpose**: Handles file paths, monkey-specific parameters (ROIs, electrodes), and low-level configuration.
*   **Key Components**:
    *   `CONFIG` class: Resolves paths (`get_data_path`), loads electrode maps.

### `core.constants`
*   **Purpose**: Stores global constants, analysis parameters, and static definitions.
*   **Key Components**:
    *   `MONKEYS`, `ZSCORE_INFO`, `REGION_ID_TO_NAME`.
    *   Analysis parameters like `ANALYSIS_TYPES`.

---

## 2. `data/` Package (Data Management)
*Data loading, building, and caching logic.*

### `data.data_manager` (Package)
*   **Purpose**: Centralizes data loading. Prevents code duplication for building trial matrices.
*   **Key Components**:
    *   `DataManager` class:
        *   `_load_trials()`: Loads raw trial metadata.
        *   `build_trial_matrix(...)`: Loads data, applies z-scoring/residuals, filters trials.

### `data.databuilder`
*   **Purpose**: Responsible for the construction and normalization of data files from raw sources.
*   **Key Components**:
    *   `DataBuilder` class:
        *   `build_if_missing()`: Triggers data generation if files don't exist.
        *   Normalization logic (`build_original_zscore`, etc.).

---

## 3. `methods/` Package (Analysis & Logic)
*Statistical methods, algorithms, and visualization tools.*

### `methods.rrr`
*   **Purpose**: Stateless logic for Reduced Rank Regression (RRR) and Ridge with Nested CV.
*   **Key Components**:
    *   `RRRAnalyzer` class: Build matrices, run cross-validation, compute R2 performance.

### `methods.repetition_stability`
*   **Purpose**: Orchestration of connectivity stability analyses across repeated blocks.
*   **Key Components**:
    *   `RepetitionStabilityAnalyzer` class: Logic for analyzing subspace stability across repetitions.

### `methods.dimensionality_correlation`
*   **Purpose**: Analyzes the relationship between subspace stability and effective dimensionality.
*   **Key Components**:
    *   `DimensionalityCorrelationAnalyzer` class.

### `methods.Semedo`
*   **Purpose**: Implementation of analyses and figures from Semedo et al. (2019).
*   **Key Components**:
    *   `build_figure_4`: Generates the standard multi-panel performance figure.
    *   `build_figure_4_subset`: Generates the subset-based performance figure.
    *   `build_semedo_figure_5_b`: RRR vs Ridge dimensionality comparison.

### `methods.matchingSubset`
*   **Purpose**: Logic for finding "matched" electrode subsets (aligning V1 firing rates with Target variance).
*   **Key Components**:
    *   `MATCHINGSUBSET` class: `match_and_save`.

### `methods.pca`
*   **Purpose**: Dimensionality reduction utilities.
*   **Key Components**:
    *   `RegionPCA` class: `fit(X)`, `get_n_components(variance_threshold)`.

### `methods.visualization`
*   **Purpose**: Central repository for plotting functions.
*   **Key Components**:
    *   `plot_figure_4`, `plot_figure_5_b_from_csv`.
    *   General plotting helpers.

---

## 4. `drivers/` Package (Execution)
*Scripts to launch analyses.*

### `drivers.driver`
*   **Purpose**: Main entry point for **Repetition Stability Analysis**.
*   **Action**: Iterates over Monkeys and Z-Score modes to run the stability pipeline.

---

## 5. Summary of Execution Flow

1.  **Driver** (`drivers/driver.py`) is executed.
2.  It initializes **Runtime** (`core.runtime`) with specific Monkey/Z-Score settings.
3.  It invokes an **Analyzer** (e.g., from `methods.repetition_stability`).
4.  The Analyzer requests data from **DataManager** (`data.data_manager`).
5.  **DataManager** loads or builds data using **DataBuilder** (`data.databuilder`).
6.  The Analyzer performs computations (RRR, PCA) and results are passed to **Visualization** (`methods.visualization`) or saved to disk.
