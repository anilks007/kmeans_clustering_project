"""
K-Means clustering module.
Implements KMeans algorithm with optimal cluster selection methods.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def fit_kmeans(X, n_clusters=None, random_state=None, **kwargs):
    """
    Fit K-Means model to data.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    n_clusters : int, optional
        Number of clusters. If None, uses config default
    random_state : int, optional
        Random state for reproducibility. If None, uses config default
    **kwargs : dict
        Additional arguments to pass to KMeans
    
    Returns:
    --------
    tuple
        (kmeans, labels) - Fitted KMeans model and cluster labels
    """
    if n_clusters is None:
        n_clusters = config.DEFAULT_N_CLUSTERS
    
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    # Set default KMeans parameters from config
    kmeans_params = {
        'n_clusters': n_clusters,
        'init': config.KMEANS_INIT,
        'n_init': config.KMEANS_N_INIT,
        'max_iter': config.KMEANS_MAX_ITER,
        'random_state': random_state
    }
    
    # Override with any user-provided kwargs
    kmeans_params.update(kwargs)
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print(f"FITTING K-MEANS MODEL")
        print(f"{'='*60}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Initialization method: {kmeans_params['init']}")
        print(f"Random state: {random_state}")
    
    # Fit model
    kmeans = KMeans(**kmeans_params)
    labels = kmeans.fit_predict(X)
    
    if config.VERBOSE:
        print(f"✓ Model fitted successfully")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        print(f"  Iterations: {kmeans.n_iter_}")
        print(f"{'='*60}\n")
    
    return kmeans, labels


def calculate_inertias(X, max_k=None, min_k=None, random_state=None):
    """
    Calculate inertia for different numbers of clusters (for Elbow method).
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    max_k : int, optional
        Maximum number of clusters to test
    min_k : int, optional
        Minimum number of clusters to test
    random_state : int, optional
        Random state for reproducibility
    
    Returns:
    --------
    tuple
        (k_values, inertias) - List of K values and corresponding inertias
    """
    if max_k is None:
        max_k = config.MAX_CLUSTERS_TO_TEST
    if min_k is None:
        min_k = config.MIN_CLUSTERS_TO_TEST
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    k_values = range(min_k, max_k + 1)
    inertias = []
    
    if config.VERBOSE:
        print(f"\nCalculating inertias for K = {min_k} to {max_k}...")
    
    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            init=config.KMEANS_INIT,
            n_init=config.KMEANS_N_INIT,
            max_iter=config.KMEANS_MAX_ITER,
            random_state=random_state
        )
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
        if config.VERBOSE:
            print(f"  K={k}: Inertia={kmeans.inertia_:.2f}")
    
    return list(k_values), inertias


def calculate_silhouette_scores(X, max_k=None, min_k=None, random_state=None):
    """
    Calculate silhouette scores for different numbers of clusters.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    max_k : int, optional
        Maximum number of clusters to test
    min_k : int, optional
        Minimum number of clusters to test (must be >= 2)
    random_state : int, optional
        Random state for reproducibility
    
    Returns:
    --------
    tuple
        (k_values, silhouette_scores) - List of K values and corresponding scores
    """
    if max_k is None:
        max_k = config.MAX_CLUSTERS_TO_TEST
    if min_k is None:
        min_k = max(2, config.MIN_CLUSTERS_TO_TEST)  # Silhouette requires at least 2 clusters
    else:
        min_k = max(2, min_k)
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    k_values = range(min_k, max_k + 1)
    silhouette_scores = []
    
    if config.VERBOSE:
        print(f"\nCalculating silhouette scores for K = {min_k} to {max_k}...")
    
    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            init=config.KMEANS_INIT,
            n_init=config.KMEANS_N_INIT,
            max_iter=config.KMEANS_MAX_ITER,
            random_state=random_state
        )
        labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        sample_size = config.SILHOUETTE_SAMPLE_SIZE
        if sample_size is not None and len(X) > sample_size:
            score = silhouette_score(X, labels, sample_size=sample_size)
        else:
            score = silhouette_score(X, labels)
        
        silhouette_scores.append(score)
        
        if config.VERBOSE:
            print(f"  K={k}: Silhouette Score={score:.4f}")
    
    return list(k_values), silhouette_scores


def find_optimal_clusters(X, max_k=None, method='silhouette', random_state=None):
    """
    Find optimal number of clusters using specified method.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    max_k : int, optional
        Maximum number of clusters to test
    method : str
        Method to use: 'silhouette' or 'elbow'
    random_state : int, optional
        Random state for reproducibility
    
    Returns:
    --------
    int
        Optimal number of clusters
    """
    if method == 'silhouette':
        k_values, scores = calculate_silhouette_scores(X, max_k=max_k, random_state=random_state)
        optimal_k = k_values[np.argmax(scores)]
        
        if config.VERBOSE:
            print(f"\n✓ Optimal K (Silhouette method): {optimal_k}")
            print(f"  Best Silhouette Score: {max(scores):.4f}")
    
    elif method == 'elbow':
        k_values, inertias = calculate_inertias(X, max_k=max_k, random_state=random_state)
        
        # Simple elbow detection: find point of maximum curvature
        # Calculate second derivative
        if len(inertias) >= 3:
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                second_derivatives.append(second_deriv)
            
            # Find elbow (maximum second derivative)
            elbow_idx = np.argmax(second_derivatives) + 1
            optimal_k = k_values[elbow_idx]
        else:
            optimal_k = k_values[len(k_values) // 2]  # Default to middle value
        
        if config.VERBOSE:
            print(f"\n✓ Optimal K (Elbow method): {optimal_k}")
            print(f"  Inertia at K={optimal_k}: {inertias[elbow_idx]:.2f}")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'silhouette' or 'elbow'")
    
    return optimal_k


class KMeansAnalyzer:
    """
    Comprehensive K-Means clustering analyzer with built-in evaluation.
    """
    
    def __init__(self, X, random_state=None):
        """
        Initialize KMeansAnalyzer.
        
        Parameters:
        -----------
        X : array-like
            Scaled feature matrix
        random_state : int, optional
            Random state for reproducibility
        """
        self.X = X
        self.random_state = random_state if random_state is not None else config.RANDOM_STATE
        self.kmeans = None
        self.labels_ = None
        self.n_clusters_ = None
        self.inertias_ = None
        self.silhouette_scores_ = None
        self.k_values_ = None
        
        if config.VERBOSE:
            print(f"\n{'='*60}")
            print("K-MEANS ANALYZER INITIALIZED")
            print(f"{'='*60}")
            print(f"Data shape: {X.shape}")
            print(f"Random state: {self.random_state}")
            print(f"{'='*60}\n")
    
    def find_optimal_k(self, max_k=None, min_k=None):
        """
        Find optimal number of clusters using both methods.
        
        Parameters:
        -----------
        max_k : int, optional
            Maximum number of clusters to test
        min_k : int, optional
            Minimum number of clusters to test
        
        Returns:
        --------
        dict
            Dictionary with optimal K from different methods
        """
        # Calculate inertias
        k_values_inertia, self.inertias_ = calculate_inertias(
            self.X, max_k=max_k, min_k=min_k, random_state=self.random_state
        )
        
        # Calculate silhouette scores
        k_values_silhouette, self.silhouette_scores_ = calculate_silhouette_scores(
            self.X, max_k=max_k, min_k=min_k, random_state=self.random_state
        )
        
        self.k_values_ = k_values_silhouette
        
        # Find optimal K using silhouette
        optimal_k_silhouette = k_values_silhouette[np.argmax(self.silhouette_scores_)]
        
        results = {
            'silhouette': optimal_k_silhouette,
            'silhouette_score': max(self.silhouette_scores_),
            'k_values': self.k_values_,
            'inertias': self.inertias_,
            'silhouette_scores': self.silhouette_scores_
        }
        
        if config.VERBOSE:
            print(f"\n{'='*60}")
            print("OPTIMAL K ANALYSIS")
            print(f"{'='*60}")
            print(f"Recommended K (Silhouette): {optimal_k_silhouette}")
            print(f"Best Silhouette Score: {max(self.silhouette_scores_):.4f}")
            print(f"{'='*60}\n")
        
        return results
    
    def fit(self, n_clusters=None, **kwargs):
        """
        Fit K-Means model.
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters
        **kwargs : dict
            Additional arguments for KMeans
        
        Returns:
        --------
        self
        """
        self.kmeans, self.labels_ = fit_kmeans(
            self.X, n_clusters=n_clusters, random_state=self.random_state, **kwargs
        )
        self.n_clusters_ = self.kmeans.n_clusters
        
        return self
    
    def evaluate(self):
        """
        Evaluate the fitted model.
        
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        from .evaluation import evaluate_clustering
        
        return evaluate_clustering(self.X, self.labels_)
    
    def get_cluster_centers(self):
        """
        Get cluster centroids.
        
        Returns:
        --------
        np.ndarray
            Cluster centroids
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.kmeans.cluster_centers_
    
    def predict(self, X_new):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X_new : array-like
            New data to predict (must be scaled the same way as training data)
        
        Returns:
        --------
        np.ndarray
            Predicted cluster labels
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.kmeans.predict(X_new)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    
    print("Creating synthetic dataset...")
    X, y_true = make_blobs(n_samples=300, n_features=4, centers=3, random_state=42)
    
    print("\nTesting KMeansAnalyzer...")
    analyzer = KMeansAnalyzer(X)
    
    # Find optimal K
    results = analyzer.find_optimal_k(max_k=8)
    
    # Fit with optimal K
    analyzer.fit(n_clusters=results['silhouette'])
    
    # Evaluate
    metrics = analyzer.evaluate()
    print(f"\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
