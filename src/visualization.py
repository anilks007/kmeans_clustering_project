"""
Visualization module for K-Means clustering.
Provides various plotting functions for cluster analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def setup_plot_style():
    """Set up matplotlib style from config."""
    try:
        plt.style.use(config.STYLE)
    except:
        plt.style.use('default')
    
    sns.set_palette(config.COLOR_PALETTE)


def plot_elbow_curve(k_values, inertias, optimal_k=None, save_path=None):
    """
    Plot elbow curve for K-Means clustering.
    
    Parameters:
    -----------
    k_values : list
        List of K values tested
    inertias : list
        Corresponding inertia values
    optimal_k : int, optional
        Optimal K to highlight on the plot
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    
    # Plot elbow curve
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8, label='Inertia')
    
    # Highlight optimal K if provided
    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        ax.plot(optimal_k, inertias[idx], 'r*', markersize=20, 
                label=f'Optimal K = {optimal_k}', zorder=5)
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12, fontweight='bold')
    ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add annotations
    ax.annotate('Look for the "elbow" point\nwhere improvement slows', 
                xy=(0.6, 0.95), xycoords='axes fraction',
                fontsize=9, style='italic', alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path or config.SAVE_FIGURES:
        if save_path is None:
            save_path = os.path.join(config.RESULTS_DIR, f'elbow_curve.{config.FIGURE_FORMAT}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        if config.VERBOSE:
            print(f"✓ Elbow curve saved to {save_path}")
    
    return fig


def plot_silhouette_analysis(X, labels, n_clusters=None, save_path=None):
    """
    Create silhouette plot for cluster analysis.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    labels : array-like
        Cluster labels
    n_clusters : int, optional
        Number of clusters (inferred from labels if not provided)
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    setup_plot_style()
    
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    # Calculate silhouette scores
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    
    y_lower = 10
    colors = plt.cm.get_cmap(config.COLOR_PALETTE)(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        # Get silhouette scores for cluster i
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_values,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}',
                fontsize=10, fontweight='bold')
        
        y_lower = y_upper + 10
    
    ax.set_xlabel('Silhouette Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_title(f'Silhouette Plot (K={n_clusters})\nAverage Score: {silhouette_avg:.4f}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
               label=f'Average: {silhouette_avg:.4f}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path or config.SAVE_FIGURES:
        if save_path is None:
            save_path = os.path.join(config.RESULTS_DIR, f'silhouette_plot_k{n_clusters}.{config.FIGURE_FORMAT}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        if config.VERBOSE:
            print(f"✓ Silhouette plot saved to {save_path}")
    
    return fig


def plot_clusters(X, labels, method='pca', centroids=None, save_path=None):
    """
    Visualize clusters in 2D using dimensionality reduction.
    
    Parameters:
    -----------
    X : array-like
        Scaled feature matrix
    labels : array-like
        Cluster labels
    method : str
        Dimensionality reduction method: 'pca' or 'first_two'
    centroids : array-like, optional
        Cluster centroids to plot
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    setup_plot_style()
    
    # Reduce to 2D
    if method == 'pca' and X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
        X_2d = pca.fit_transform(X)
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
        title_suffix = '(PCA projection)'
        
        if centroids is not None:
            centroids_2d = pca.transform(centroids)
    elif X.shape[1] >= 2:
        X_2d = X[:, :2]
        xlabel = 'Feature 1'
        ylabel = 'Feature 2'
        title_suffix = '(First 2 features)'
        centroids_2d = centroids[:, :2] if centroids is not None else None
    else:
        raise ValueError("Data must have at least 2 features")
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    
    # Plot clusters
    n_clusters = len(np.unique(labels))
    colors = plt.cm.get_cmap(config.COLOR_PALETTE)(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_points = X_2d[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                  c=[colors[i]], label=f'Cluster {i}',
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Plot centroids if provided
    if centroids is not None and centroids_2d is not None:
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                  c='red', marker='X', s=300, edgecolors='black',
                  linewidth=2, label='Centroids', zorder=10)
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'Cluster Visualization {title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path or config.SAVE_FIGURES:
        if save_path is None:
            save_path = os.path.join(config.RESULTS_DIR, f'cluster_visualization.{config.FIGURE_FORMAT}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        if config.VERBOSE:
            print(f"✓ Cluster visualization saved to {save_path}")
    
    return fig


def plot_silhouette_scores(k_values, silhouette_scores, optimal_k=None, save_path=None):
    """
    Plot silhouette scores for different K values.
    
    Parameters:
    -----------
    k_values : list
        List of K values tested
    silhouette_scores : list
        Corresponding silhouette scores
    optimal_k : int, optional
        Optimal K to highlight
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.DPI)
    
    # Plot silhouette scores
    ax.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8, label='Silhouette Score')
    
    # Highlight optimal K
    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        ax.plot(optimal_k, silhouette_scores[idx], 'r*', markersize=20,
                label=f'Optimal K = {optimal_k}', zorder=5)
    
    # Add quality zones
    ax.axhspan(0.71, 1.0, alpha=0.1, color='green', label='Strong structure')
    ax.axhspan(0.51, 0.71, alpha=0.1, color='yellow', label='Reasonable structure')
    ax.axhspan(0.26, 0.51, alpha=0.1, color='orange', label='Weak structure')
    ax.axhspan(0, 0.26, alpha=0.1, color='red', label='Poor structure')
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    
    if save_path or config.SAVE_FIGURES:
        if save_path is None:
            save_path = os.path.join(config.RESULTS_DIR, f'silhouette_scores.{config.FIGURE_FORMAT}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        if config.VERBOSE:
            print(f"✓ Silhouette scores plot saved to {save_path}")
    
    return fig


def plot_cluster_comparison(k_values, inertias, silhouette_scores, save_path=None):
    """
    Create a combined plot comparing inertia and silhouette scores.
    
    Parameters:
    -----------
    k_values : list
        List of K values tested
    inertias : list
        Corresponding inertia values
    silhouette_scores : list
        Corresponding silhouette scores
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=config.DPI)
    
    # Plot 1: Elbow curve
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia', fontsize=12, fontweight='bold')
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Silhouette scores
    ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # Add quality zones to silhouette plot
    ax2.axhspan(0.71, 1.0, alpha=0.1, color='green')
    ax2.axhspan(0.51, 0.71, alpha=0.1, color='yellow')
    ax2.axhspan(0.26, 0.51, alpha=0.1, color='orange')
    ax2.axhspan(0, 0.26, alpha=0.1, color='red')
    
    plt.suptitle('Optimal K Selection Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path or config.SAVE_FIGURES:
        if save_path is None:
            save_path = os.path.join(config.RESULTS_DIR, f'cluster_comparison.{config.FIGURE_FORMAT}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        if config.VERBOSE:
            print(f"✓ Comparison plot saved to {save_path}")
    
    return fig


def plot_all_visualizations(analyzer, X, save_dir=None):
    """
    Create all visualizations for a KMeansAnalyzer object.
    
    Parameters:
    -----------
    analyzer : KMeansAnalyzer
        Fitted KMeansAnalyzer object
    X : array-like
        Scaled feature matrix
    save_dir : str, optional
        Directory to save all figures
    
    Returns:
    --------
    dict
        Dictionary of figure objects
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    figures = {}
    
    # Elbow curve
    if analyzer.inertias_ is not None:
        figures['elbow'] = plot_elbow_curve(
            analyzer.k_values_, analyzer.inertias_,
            optimal_k=analyzer.n_clusters_,
            save_path=os.path.join(save_dir, f'elbow_curve.{config.FIGURE_FORMAT}')
        )
    
    # Silhouette scores
    if analyzer.silhouette_scores_ is not None:
        figures['silhouette_scores'] = plot_silhouette_scores(
            analyzer.k_values_, analyzer.silhouette_scores_,
            optimal_k=analyzer.n_clusters_,
            save_path=os.path.join(save_dir, f'silhouette_scores.{config.FIGURE_FORMAT}')
        )
    
    # Comparison plot
    if analyzer.inertias_ is not None and analyzer.silhouette_scores_ is not None:
        figures['comparison'] = plot_cluster_comparison(
            analyzer.k_values_, analyzer.inertias_, analyzer.silhouette_scores_,
            save_path=os.path.join(save_dir, f'cluster_comparison.{config.FIGURE_FORMAT}')
        )
    
    # Silhouette analysis
    if analyzer.labels_ is not None:
        figures['silhouette_analysis'] = plot_silhouette_analysis(
            X, analyzer.labels_, n_clusters=analyzer.n_clusters_,
            save_path=os.path.join(save_dir, f'silhouette_analysis.{config.FIGURE_FORMAT}')
        )
    
    # Cluster visualization
    if analyzer.labels_ is not None:
        centroids = analyzer.get_cluster_centers() if analyzer.kmeans is not None else None
        figures['clusters'] = plot_clusters(
            X, analyzer.labels_, centroids=centroids,
            save_path=os.path.join(save_dir, f'cluster_visualization.{config.FIGURE_FORMAT}')
        )
    
    if config.VERBOSE:
        print(f"\n✓ All visualizations created and saved to {save_dir}")
    
    return figures


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    
    print("Creating synthetic dataset...")
    X, y_true = make_blobs(n_samples=300, n_features=4, centers=3, random_state=42)
    
    print("\nFitting K-Means...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    
    print("\nCreating visualizations...")
    
    # Test individual plots
    k_values = list(range(2, 9))
    inertias = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_ for k in k_values]
    
    plot_elbow_curve(k_values, inertias, optimal_k=3)
    plot_silhouette_analysis(X, labels, n_clusters=3)
    plot_clusters(X, labels, centroids=kmeans.cluster_centers_)
    
    plt.show()
    
    print("\n✓ Visualizations created successfully!")
