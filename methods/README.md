# Scientific Methods

This directory contains the core scientific algorithms and analysis pipelines used in the project. Each method is encapsulated in its own sub-package.

## Packages

| Package | Purpose |
| :--- | :--- |
| **[`data_manager`](../data/data_manager/)** | (Located in `../data`) The source of data for all methods below. |
| **[`dimensionality_correlation`](./dimensionality_correlation/)** | Analyzes the stability of neural subspaces across dimensions. |
| **[`repetition_stability`](./repetition_stability/)** | Analyzes the stability of neural representations across repeated stimuli blocks. |
| **[`rrr`](./rrr/)** | Implements Reduced Rank Regression (RRR) and Ridge models for predictive modeling. |
| **[`Semedo`](./Semedo/)** | Replicates and extends methodology from Semedo et al. (2019). |
| **[`visualization`](./visualization/)** | Central repository for all plotting and figure generation code. |

## Principles
*   **One Idea Per Module**: Code is split to ensure maintenance and readability.
*   **Data Access**: All methods access data via the `DataManager` singleton.
*   **Visualization Separation**: Heavy plotting logic is offloaded to the `visualization` package to keep analytical code clean.
