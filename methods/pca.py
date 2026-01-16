"""
pca.py
-----------
Classes for subspace analysis (PCA, Dimensionality).
"""
import numpy as np
from typing import Tuple, Optional

class RegionPCA:
    """
    Perform PCA on a neural data matrix (Observations x Features).
    Calculates effective dimensionality using Participation Ratio (PR).
    """

    def __init__(self, centered: bool = True):
        self.centered = centered
        self.eigenvalues_ = None
        self.eigenvectors_ = None # components_ (D x n_features)
        self.dimensionality_ = None
        self.pr_ = None

    def fit(self, X: np.ndarray) -> 'RegionPCA':
        """
        Fit PCA on X.
        
        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
            e.g. (n_trials, n_electrodes)
        """
        if self.centered:
            X_fit = X - X.mean(axis=0, keepdims=True)
        else:
            X_fit = X

        # SVD approach is numerically stable
        # X = U S Vt
        # Cov = X.T X / (N-1)
        # S (singular values) related to eigenvalues of Cov by lambda = s^2 / (N-1)
        
        _, S, Vt = np.linalg.svd(X_fit, full_matrices=False)
        
        n_samples = X.shape[0]
        variances = (S ** 2) / max(1, (n_samples - 1))
        
        self.eigenvalues_ = variances
        
        # Participation Ratio
        if variances.sum() > 0:
            v_norm = variances / variances.sum()
            denom = np.sum(v_norm ** 2)
            self.pr_ = (np.sum(v_norm) ** 2) / denom if denom > 0 else 0.0
        else:
            self.pr_ = 0.0
            
        self.dimensionality_ = int(np.ceil(self.pr_)) if self.pr_ > 0 else 1
        
        # Store all components (sorted by variance, which SVD does automatically)
        self.eigenvectors_ = Vt
        
        return self

    def get_components(self, n_components: Optional[int] = None) -> np.ndarray:
        """
        Return the top n_components eigenvectors (D x n_features).
        If n_components is None, returns all.
        """
        if self.eigenvectors_ is None:
             raise RuntimeError("PCA not fitted.")
        
        if n_components is None:
            return self.eigenvectors_
        return self.eigenvectors_[:n_components, :]

    def transform(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """Project X onto the principal components."""
        if self.eigenvectors_ is None:
            raise RuntimeError("PCA not fitted.")
        
        comps = self.get_components(n_components)
        
        if self.centered:
             # Transform uses mean from fit typically, but here we assume simple centering
             X_centered = X - X.mean(axis=0, keepdims=True)
             return X_centered @ comps.T
        
        return X @ comps.T
    
    @property
    def dimensionality(self) -> int:
        return self.dimensionality_
        
    @property
    def participation_ratio(self) -> float:
        return self.pr_

    def get_n_components(self, variance_threshold: float = 0.95) -> int:
        """
        Compute number of components needed to explain variance_threshold fraction of variance.
        """
        if self.eigenvalues_ is None:
            raise RuntimeError("PCA not fitted.")
        
        # Calculate cumulative variance fraction
        cum_var = np.cumsum(self.eigenvalues_)
        total_var = np.sum(self.eigenvalues_)
        
        if total_var == 0:
            return 1
            
        frac = cum_var / total_var
        return int(np.searchsorted(frac, variance_threshold, side="left") + 1)
