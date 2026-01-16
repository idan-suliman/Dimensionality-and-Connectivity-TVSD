# Core Package (`core/`)

The `core` package is the foundation of the TVSD codebase. It manages global state, configuration, constants, and path resolution for the entire application.

## Key Components

### 1. `runtime.py` (`RUNTIME` Singleton)
*   **Role**: The central access point for the application. It aggregates configuration, constants, paths, and the data manager into a single global object.
*   **Usage**: `from core.runtime import runtime`
*   **Key Properties**:
    *   `runtime.cfg`: The current configuration object (see `config.py`).
    *   `runtime.consts`: Access to project constants (see `constants.py`).
    *   `runtime.paths`: Path generation utilities (see `paths.py`).
    *   `runtime.data_manager`: The active data manager instance.
    *   `runtime.update(...)`: Updates the global state (monkey, Z-score, parameters).

### 2. `config.py` (`CONFIG` Class)
*   **Role**: Manages dynamic configuration state based on user selection (e.g., which monkey, which normalization method).
*   **Responsibilities**:
    *   Stores `monkey_name`, `z_score_index`, `analysis_type`.
    *   Manages analysis parameters like `cv_outer_splits`, `cv_inner_splits`, `n_permutations`.
    *   Resolves the main data file path.
    *   Provides helper methods like `get_rois()` to retrieve electrode mappings.

### 3. `paths.py` (`Paths` Class)
*   **Role**: The single source of truth for file paths. It eliminates hardcoded paths scattered throughout the code.
*   **Responsibilities**:
    *   Generates consistent paths for output figures, cached `.npz` files, and intermediate data.
    *   **Key Methods**:
        *   `get_dim_corr_path(...)`: For dimensionality correlation results.
        *   `get_semedo_figure_path(...)`: For Semedo replication figures (Fig 4, 5B).
        *   `get_matching_path(...)`: For electrode matching subsets.
        *   `get_rep_stab_path(...)`: For repetition stability results.

### 4. `constants.py`
*   **Role**: Holds static, project-wide constants that do not change during runtime.
*   **Contents**:
    *   **Dataset Meta**: `NUM_CHANNELS`, `NUM_REPETITIONS`.
    *   **Mappings**: `REGION_ID_TO_NAME`, `REGION_WINDOWS`.
    *   **Defaults**: `DEFAULT_CV_OUTER_SPLITS`, `DEFAULT_N_PERMS`.
    *   **Directory Names**: `DIR_PLOTS`, `DIR_TARGET_RRR`, etc.
