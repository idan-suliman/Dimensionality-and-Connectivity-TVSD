# Scientific Methods

This directory contains the core scientific algorithms and analysis pipelines used in the project. Each method is encapsulated in its own sub-package.

## Packages

| Package | Purpose |
| :--- | :--- |

| **[`dimensionality_correlation`](./dimensionality_correlation/)** | Analyzes the stability of neural subspaces across dimensions, and computes the overlap between intrinsic and predictive subspaces. |
| **[`matchingSubset`](./matchingSubset.py)** | Logic for finding "matched" electrode subsets (aligning V1 firing rates with Target variance). |
| **[`pca`](./pca.py)** | Dimensionality reduction utilities (PCA helpers). |
| **[`repetition_stability`](./repetition_stability/)** | Analyzes the stability of neural representations across repeated stimuli blocks. |
| **[`rrr`](./rrr/)** | Implements Reduced Rank Regression (RRR) and Ridge models for predictive modeling. |
| **[`Semedo`](./Semedo/)** | Replicates and extends methodology from Semedo et al. (2019). |


## Principles
*   **One Idea Per Module**: Code is split to ensure maintenance and readability.
*   **Data Access**: All methods access data via the `DataManager` singleton.
*   **Visualization Separation**: Heavy plotting logic is offloaded to the `visualization` package to keep analytical code clean.
