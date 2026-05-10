"""
Evaluation module for K-Means clustering.
Provides various metrics to assess cluster quality.
"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_samples
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def calculate_silhouette_score(X, labels, sample_size=None):
    """
    Calculate silhouette score for clustering.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    labels : array-like
        Cluster labels
    sample_size : int, optional
        Sample size for large datasets
    
    Returns:
    --------
    float
        Silhouette score (-1 to 1, higher is better)
    """
    if sample_size is None:
        sample_size = config.SILHOUETTE_SAMPLE_SIZE
    
    if sample_size is not None and len(X) > sample_size:
        score = silhouette_score(X, labels, sample_size=sample_size)
    else:
        score = silhouette_score(X, labels)
    
    return score


def calculate_davies_bouldin(X, labels):
    """
    Calculate Davies-Bouldin index for clustering.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    labels : array-like
        Cluster labels
    
    Returns:
    --------
    float
        Davies-Bouldin index (lower is better, 0-2 is good)
    """
    score = davies_bouldin_score(X, labels)
    return score


def calculate_calinski_harabasz(X, labels):
    """
    Calculate Calinski-Harabasz score (Variance Ratio Criterion).
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    labels : array-like
        Cluster labels
    
    Returns:
    --------
    float
        Calinski-Harabasz score (higher is better)
    """
    score = calinski_harabasz_score(X, labels)
    return score


def calculate_inertia(X, labels, centroids):
    """
    Calculate inertia (within-cluster sum of squares).
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    labels : array-like
        Cluster labels
    centroids : array-like
        Cluster centroids
    
    Returns:
    --------
    float
        Inertia value
    """
    inertia = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroids[i]) ** 2)
    
    return inertia


def interpret_silhouette_score(score):
    """
    Provide interpretation of silhouette score.
    
    Parameters:
    -----------
    score : float
        Silhouette score
    
    Returns:
    --------
    str
        Interpretation of the score
    """
    if score >= 0.71:
        return "Strong structure - Excellent clustering"
    elif score >= 0.51:
        return "Reasonable structure - Good clustering"
    elif score >= 0.26:
        return "Weak structure - Fair clustering"
    else:
        return "No substantial structure - Poor clustering"


def interpret_davies_bouldin(score):
    """
    Provide interpretation of Davies-Bouldin index.
    
    Parameters:
    -----------
    score : float
        Davies-Bouldin index
    
    Returns:
    --------
    str
        Interpretation of the score
    """
    if score <= 1.0:
        return "Excellent clustering - Very distinct clusters"
    elif score <= 2.0:
        return "Good clustering - Well-separated clusters"
    elif score <= 3.0:
        return "Fair clustering - Moderate separation"
    else:
        return "Poor clustering - Overlapping clusters"


def get_cluster_sizes(labels):
    """
    Get the size of each cluster.
    
    Parameters:
    -----------
    labels : array-like
        Cluster labels
    
    Returns:
    --------
    dict
        Dictionary mapping cluster ID to size
    """
    unique_labels = np.unique(labels)
    sizes = {}
    
    for label in unique_labels:
        sizes[int(label)] = int(np.sum(labels == label))
    
    return sizes


def get_silhouette_samples_by_cluster(X, labels):
    """
    Calculate silhouette scores for each sample, grouped by cluster.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    labels : array-like
        Cluster labels
    
    Returns:
    --------
    dict
        Dictionary mapping cluster ID to array of silhouette scores
    """
    sample_scores = silhouette_samples(X, labels)
    unique_labels = np.unique(labels)
    
    cluster_scores = {}
    for label in unique_labels:
        cluster_scores[int(label)] = sample_scores[labels == label]
    
    return cluster_scores


def evaluate_clustering(X, labels, centroids=None, verbose=None):
    """
    Comprehensive evaluation of clustering results.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    labels : array-like
        Cluster labels
    centroids : array-like, optional
        Cluster centroids (for inertia calculation)
    verbose : bool, optional
        Whether to print detailed results
    
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    if verbose is None:
        verbose = config.VERBOSE
    
    # Calculate metrics
    silhouette = calculate_silhouette_score(X, labels)
    davies_bouldin = calculate_davies_bouldin(X, labels)
    calinski_harabasz = calculate_calinski_harabasz(X, labels)
    cluster_sizes = get_cluster_sizes(labels)
    
    # Calculate inertia if centroids provided
    inertia = None
    if centroids is not None:
        inertia = calculate_inertia(X, labels, centroids)
    
    # Compile results
    results = {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'n_clusters': len(np.unique(labels)),
        'cluster_sizes': cluster_sizes
    }
    
    if inertia is not None:
        results['inertia'] = inertia
    
    # Print results if verbose
    if verbose:
        print(f"\n{'='*60}")
        print("CLUSTERING EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"\nNumber of Clusters: {results['n_clusters']}")
        print(f"\nCluster Sizes:")
        for cluster_id, size in cluster_sizes.items():
            print(f"  Cluster {cluster_id}: {size} samples")
        
        print(f"\n{'─'*60}")
        print("QUALITY METRICS")
        print(f"{'─'*60}")
        
        print(f"\nSilhouette Score: {silhouette:.4f}")
        print(f"  → {interpret_silhouette_score(silhouette)}")
        print(f"  Range: -1 to +1 (higher is better)")
        
        print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"  → {interpret_davies_bouldin(davies_bouldin)}")
        print(f"  Range: 0 to ∞ (lower is better)")
        
        print(f"\nCalinski-Harabasz Score: {calinski_harabasz:.2f}")
        print(f"  → Higher values indicate better-defined clusters")
        
        if inertia is not None:
            print(f"\nInertia (Within-cluster sum of squares): {inertia:.2f}")
            print(f"  → Lower values indicate tighter clusters")
        
        print(f"\n{'='*60}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*60}")
        
        # Overall assessment
        if silhouette >= 0.51 and davies_bouldin <= 2.0:
            print("✓ GOOD: Clustering quality is satisfactory")
        elif silhouette >= 0.26 and davies_bouldin <= 3.0:
            print("⚠ FAIR: Clustering quality is acceptable but could be improved")
        else:
            print("✗ POOR: Consider trying different number of clusters or preprocessing")
        
        print(f"{'='*60}\n")
    
    return results


def compare_clusterings(X, labels_list, names=None):
    """
    Compare multiple clustering results.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    labels_list : list of array-like
        List of cluster label arrays to compare
    names : list of str, optional
        Names for each clustering
    
    Returns:
    --------
    pd.DataFrame
        Comparison table of metrics
    """
    import pandas as pd
    
    if names is None:
        names = [f"Clustering_{i+1}" for i in range(len(labels_list))]
    
    results = []
    
    for labels, name in zip(labels_list, names):
        metrics = evaluate_clustering(X, labels, verbose=False)
        metrics['name'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[['name', 'n_clusters', 'silhouette', 'davies_bouldin', 'calinski_harabasz']]
    
    if config.VERBOSE:
        print("\n" + "="*80)
        print("CLUSTERING COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
    
    return df


def analyze_cluster_quality_per_cluster(X, labels):
    """
    Analyze quality metrics for each individual cluster.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    labels : array-like
        Cluster labels
    
    Returns:
    --------
    dict
        Dictionary with per-cluster analysis
    """
    cluster_scores = get_silhouette_samples_by_cluster(X, labels)
    cluster_sizes = get_cluster_sizes(labels)
    
    analysis = {}
    
    for cluster_id in cluster_scores.keys():
        scores = cluster_scores[cluster_id]
        analysis[cluster_id] = {
            'size': cluster_sizes[cluster_id],
            'mean_silhouette': float(np.mean(scores)),
            'min_silhouette': float(np.min(scores)),
            'max_silhouette': float(np.max(scores)),
            'std_silhouette': float(np.std(scores)),
            'negative_samples': int(np.sum(scores < 0)),
            'negative_percentage': float(100 * np.sum(scores < 0) / len(scores))
        }
    
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("PER-CLUSTER QUALITY ANALYSIS")
        print(f"{'='*60}")
        
        for cluster_id, metrics in analysis.items():
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {metrics['size']} samples")
            print(f"  Mean Silhouette: {metrics['mean_silhouette']:.4f}")
            print(f"  Silhouette Range: [{metrics['min_silhouette']:.4f}, {metrics['max_silhouette']:.4f}]")
            print(f"  Std Deviation: {metrics['std_silhouette']:.4f}")
            print(f"  Negative Samples: {metrics['negative_samples']} ({metrics['negative_percentage']:.1f}%)")
            
            if metrics['negative_percentage'] > 20:
                print(f"  ⚠ Warning: High percentage of poorly assigned samples")
        
        print(f"{'='*60}\n")
    
    return analysis


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    
    print("Creating synthetic dataset...")
    X, y_true = make_blobs(n_samples=300, n_features=4, centers=3, random_state=42)
    
    print("\nFitting K-Means...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    
    print("\nEvaluating clustering...")
    results = evaluate_clustering(X, labels, centroids=kmeans.cluster_centers_)
    
    print("\nPer-cluster analysis...")
    cluster_analysis = analyze_cluster_quality_per_cluster(X, labels)
