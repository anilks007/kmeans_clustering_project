"""
K-Means clustering module.
Implements KMeans, DBSCAN, and Hierarchical clustering algorithms with optimal parameter selection methods.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
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


def fit_dbscan(X, eps=None, min_samples=None, **kwargs):
    """
    Fit DBSCAN clustering model to data.

    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    eps : float, optional
        Maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples : int, optional
        Minimum number of samples in a neighborhood for a point to be considered as a core point
    **kwargs : dict
        Additional arguments to pass to DBSCAN

    Returns:
    --------
    tuple
        (dbscan, labels) - Fitted DBSCAN model and cluster labels
    """
    if eps is None:
        # Auto-determine eps using k-distance method
        eps = find_optimal_eps(X)

    if min_samples is None:
        min_samples = max(2, int(0.03 * len(X)))  # 3% of dataset size, minimum 2

    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("FITTING DBSCAN MODEL")
        print(f"{'='*60}")
        print(f"Epsilon (eps): {eps:.3f}")
        print(f"Minimum samples: {min_samples}")

    # Fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = dbscan.fit_predict(X)

    if config.VERBOSE:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"✓ DBSCAN fitted successfully")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        print(f"{'='*60}\n")

    return dbscan, labels


def fit_hierarchical(X, n_clusters=None, linkage='ward', **kwargs):
    """
    Fit Hierarchical clustering model to data.

    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    n_clusters : int, optional
        Number of clusters to find
    linkage : str
        Linkage criterion: 'ward', 'complete', 'average', 'single'
    **kwargs : dict
        Additional arguments to pass to AgglomerativeClustering

    Returns:
    --------
    tuple
        (hierarchical, labels) - Fitted Hierarchical model and cluster labels
    """
    if n_clusters is None:
        n_clusters = config.DEFAULT_N_CLUSTERS

    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("FITTING HIERARCHICAL CLUSTERING MODEL")
        print(f"{'='*60}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Linkage method: {linkage}")

    # Fit Hierarchical clustering
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        **kwargs
    )
    labels = hierarchical.fit_predict(X)

    if config.VERBOSE:
        print(f"✓ Hierarchical clustering fitted successfully")
        print(f"  Number of clusters: {n_clusters}")
        print(f"{'='*60}\n")

    return hierarchical, labels


def find_optimal_eps(X, k=4):
    """
    Find optimal eps value for DBSCAN using k-distance method.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    k : int
        Number of nearest neighbors to consider

    Returns:
    --------
    float
        Optimal eps value
    """
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Sort distances to k-th nearest neighbor
    k_distances = np.sort(distances[:, k-1])

    # Find the "elbow" point in the k-distance plot
    # Use a simple method: find point where slope changes significantly
    if len(k_distances) > 10:
        # Calculate differences (slopes)
        diffs = np.diff(k_distances)
        # Find where the slope starts to increase significantly
        threshold = np.percentile(diffs, 75)  # 75th percentile
        elbow_idx = np.where(diffs > threshold)[0]

        if len(elbow_idx) > 0:
            optimal_idx = elbow_idx[0]
        else:
            optimal_idx = len(k_distances) // 3  # Fallback: 1/3 of the way
    else:
        optimal_idx = len(k_distances) // 3

    optimal_eps = k_distances[optimal_idx]

    if config.VERBOSE:
        print(f"✓ Optimal eps for DBSCAN: {optimal_eps:.3f} (k={k})")

    return optimal_eps


class ClusteringAnalyzer:
    """
    Comprehensive clustering analyzer supporting multiple algorithms (K-Means, DBSCAN, Hierarchical).
    """

    def __init__(self, X, algorithm='kmeans', random_state=None):
        """
        Initialize ClusteringAnalyzer.

        Parameters:
        -----------
        X : array-like
            Scaled feature matrix
        algorithm : str
            Clustering algorithm: 'kmeans', 'dbscan', 'hierarchical'
        random_state : int, optional
            Random state for reproducibility
        """
        self.X = X
        self.algorithm = algorithm.lower()
        self.random_state = random_state if random_state is not None else config.RANDOM_STATE

        # Algorithm-specific attributes
        self.model = None
        self.labels_ = None
        self.n_clusters_ = None

        # K-Means specific attributes
        self.inertias_ = None
        self.silhouette_scores_ = None
        self.k_values_ = None

        # DBSCAN specific attributes
        self.eps_ = None
        self.min_samples_ = None

        # Hierarchical specific attributes
        self.linkage_ = None

        if config.VERBOSE:
            print(f"\n{'='*60}")
            print(f"{algorithm.upper()} CLUSTERING ANALYZER INITIALIZED")
            print(f"{'='*60}")
            print(f"Data shape: {X.shape}")
            print(f"Algorithm: {algorithm}")
            print(f"Random state: {self.random_state}")
            print(f"{'='*60}\n")

    def find_optimal_k(self, max_k=None, min_k=None):
        """
        Find optimal number of clusters using both methods (K-Means only).
        """
        if self.algorithm != 'kmeans':
            if config.VERBOSE:
                print(f"✓ {self.algorithm.upper()} doesn't require optimal K selection")
            return {'optimal_k': None}

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

    def fit(self, n_clusters=None, eps=None, min_samples=None, linkage='ward', **kwargs):
        """
        Fit clustering model.

        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters (for K-Means and Hierarchical)
        eps : float, optional
            Epsilon for DBSCAN
        min_samples : int, optional
            Minimum samples for DBSCAN
        linkage : str, optional
            Linkage method for Hierarchical clustering
        **kwargs : dict
            Additional arguments for the clustering algorithm

        Returns:
        --------
        self
        """
        if self.algorithm == 'kmeans':
            self.model, self.labels_ = fit_kmeans(
                self.X, n_clusters=n_clusters, random_state=self.random_state, **kwargs
            )
            self.n_clusters_ = self.model.n_clusters

        elif self.algorithm == 'dbscan':
            self.model, self.labels_ = fit_dbscan(
                self.X, eps=eps, min_samples=min_samples, **kwargs
            )
            # Calculate number of clusters (excluding noise labeled as -1)
            self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
            self.eps_ = self.model.eps
            self.min_samples_ = self.model.min_samples

        elif self.algorithm == 'hierarchical':
            self.model, self.labels_ = fit_hierarchical(
                self.X, n_clusters=n_clusters, linkage=linkage, **kwargs
            )
            self.n_clusters_ = self.model.n_clusters
            self.linkage_ = linkage

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return self

    def evaluate(self):
        """
        Evaluate the fitted model.

        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        from .evaluation import evaluate_clustering

        # For DBSCAN, filter out noise points (-1 labels) for evaluation
        if self.algorithm == 'dbscan':
            # Only evaluate non-noise points
            mask = self.labels_ != -1
            if np.sum(mask) >= 2:  # Need at least 2 points for silhouette
                X_eval = self.X[mask]
                labels_eval = self.labels_[mask]
                return evaluate_clustering(X_eval, labels_eval)
            else:
                return {
                    'silhouette': None,
                    'davies_bouldin': None,
                    'calinski_harabasz': None,
                    'note': 'Insufficient non-noise points for evaluation'
                }
        else:
            return evaluate_clustering(self.X, self.labels_)

    def get_cluster_centers(self):
        """
        Get cluster centroids (K-Means only).

        Returns:
        --------
        np.ndarray or None
            Cluster centroids for K-Means, None for other algorithms
        """
        if self.algorithm == 'kmeans' and self.model is not None:
            return self.model.cluster_centers_
        return None

    def predict(self, X_new):
        """
        Predict cluster labels for new data (K-Means and Hierarchical only).

        Parameters:
        -----------
        X_new : array-like
            New data to predict (must be scaled the same way as training data)

        Returns:
        --------
        np.ndarray or None
            Predicted cluster labels, None for DBSCAN
        """
        if self.algorithm == 'dbscan':
            return None  # DBSCAN doesn't support prediction on new data
        elif self.model is not None:
            return self.model.predict(X_new)
        else:
            raise ValueError("Model not fitted yet. Call fit() first.")


# Backward compatibility
KMeansAnalyzer = ClusteringAnalyzer


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
