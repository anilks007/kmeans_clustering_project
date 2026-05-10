"""
Streamlit web application for Clustering Analysis (K-Means, DBSCAN, Hierarchical)
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import os

# Import project modules
from src.data_preprocessing import load_and_preprocess_data
from src.clustering import ClusteringAnalyzer
from src.visualization import (
    plot_elbow_curve,
    plot_silhouette_analysis,
    plot_cluster_comparison,
    plot_dbscan_k_distance,
    plot_hierarchical_dendrogram,
    plot_clusters_3d,
    plot_cluster_profiles
)
import config

# Page configuration
st.set_page_config(
    page_title="Clustering Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# TITLE & INTRODUCTION
# ============================================
st.title("🎯 Advanced Clustering Analysis Dashboard")
st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    ### Welcome to the Interactive Clustering Tool
    Upload your data, choose an algorithm, adjust parameters, and explore clustering results in real-time!
    """)
with col2:
    st.info("💡 **Algorithms:**\nK-Means, DBSCAN, Hierarchical")

# ============================================
# SIDEBAR - FILE UPLOAD & PARAMETERS
# ============================================
st.sidebar.header("⚙️ Configuration")

# Algorithm selection
st.sidebar.subheader("🤖 Algorithm Selection")
algorithm = st.sidebar.selectbox(
    "Choose Clustering Algorithm:",
    ["K-Means", "DBSCAN", "Hierarchical"],
    help="Select the clustering algorithm to use"
)
algorithm_key = algorithm.replace('-', '').lower()

# File upload
st.sidebar.subheader("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload your data file with numeric features for clustering"
)

if uploaded_file is not None:
    # Read uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Select features
    st.sidebar.subheader("🎯 Feature Selection")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error("❌ No numeric columns found! Please upload data with numeric features.")
        st.stop()
    
    selected_features = st.sidebar.multiselect(
        "Select features for clustering:",
        numeric_cols,
        default=numeric_cols,
        help="Choose which columns to use for clustering"
    )
    
    if not selected_features:
        st.error("❌ Please select at least one feature!")
        st.stop()
    
    # Algorithm-specific parameters
    st.sidebar.subheader("🔢 Algorithm Parameters")

    if algorithm == "K-Means":
        max_samples = len(df)
        max_k_allowed = min(max_samples - 1, 10)

        k_value = st.sidebar.slider(
            "Number of Clusters (K)",
            min_value=2,
            max_value=max_k_allowed,
            value=3,
            help=f"Choose between 2 and {max_k_allowed} (limited by dataset size)"
        )

        # Find optimal K option
        find_optimal = st.sidebar.checkbox(
            "🔍 Find Optimal K First",
            value=True,
            help="Run elbow method and silhouette analysis to find optimal K"
        )

        # Store parameters
        params = {'n_clusters': k_value, 'find_optimal': find_optimal}

    elif algorithm == "DBSCAN":
        eps = st.sidebar.slider(
            "Epsilon (eps)",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Maximum distance between two samples for one to be considered as in the neighborhood of the other"
        )

        min_samples = st.sidebar.slider(
            "Minimum Samples",
            min_value=2,
            max_value=min(20, len(df)),
            value=5,
            help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point"
        )

        # Find optimal eps option
        find_optimal_eps = st.sidebar.checkbox(
            "🔍 Find Optimal Eps",
            value=True,
            help="Use k-distance plot to help find optimal eps value"
        )

        params = {'eps': eps, 'min_samples': min_samples, 'find_optimal_eps': find_optimal_eps}

    elif algorithm == "Hierarchical":
        max_samples = len(df)
        max_k_allowed = min(max_samples - 1, 10)

        k_value = st.sidebar.slider(
            "Number of Clusters (K)",
            min_value=2,
            max_value=max_k_allowed,
            value=3,
            help=f"Choose between 2 and {max_k_allowed} (limited by dataset size)"
        )

        linkage_methods = ["ward", "complete", "average", "single"]
        linkage = st.sidebar.selectbox(
            "Linkage Method",
            linkage_methods,
            index=0,
            help="Method to measure distance between clusters"
        )

        params = {'n_clusters': k_value, 'linkage': linkage}
    
    # ============================================
    # MAIN CONTENT AREA
    # ============================================
    
    st.markdown("---")
    st.header("📊 Data Preview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features Selected", len(selected_features))
    with col3:
        if algorithm == "K-Means":
            st.metric("Algorithm", f"{algorithm} (K={params['n_clusters']})")
        elif algorithm == "DBSCAN":
            st.metric("Algorithm", f"{algorithm} (eps={params['eps']})")
        elif algorithm == "Hierarchical":
            st.metric("Algorithm", f"{algorithm} (K={params['n_clusters']})")
    
    # Show data preview
    with st.expander("📋 View Raw Data", expanded=False):
        st.dataframe(df[selected_features], use_container_width=True)
    
    # ============================================
    # PREPROCESSING
    # ============================================
    st.markdown("---")
    st.header("🔧 Data Preprocessing")
    
    try:
        # Extract selected features
        X = df[selected_features].values
        
        # Check for missing values
        missing_count = pd.DataFrame(X).isnull().sum().sum()
        if missing_count > 0:
            st.warning(f"⚠️ Found {missing_count} missing values. Dropping rows with missing data...")
            df_clean = df[selected_features].dropna()
            X = df_clean.values
        else:
            df_clean = df[selected_features].copy()
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("✓ Missing values handled")
        with col2:
            st.success("✓ Features standardized (StandardScaler)")
        
        st.info(f"📊 Data shape: {X_scaled.shape} (samples × features)")
        
    except Exception as e:
        st.error(f"❌ Preprocessing error: {str(e)}")
        st.stop()
    
    # ============================================
    # FIND OPTIMAL PARAMETERS
    # ============================================
    if (algorithm == "K-Means" and params.get('find_optimal', False)) or (algorithm == "DBSCAN" and params.get('find_optimal_eps', False)):
        st.markdown("---")
        if algorithm == "K-Means":
            st.header("🔍 Finding Optimal K")
        elif algorithm == "DBSCAN":
            st.header("🔍 Finding Optimal Eps")

        with st.spinner("Analyzing... This may take a moment..."):
            analyzer = ClusteringAnalyzer(X_scaled, algorithm=algorithm_key, random_state=config.RANDOM_STATE)

            if algorithm == "K-Means":
                optimal_results = analyzer.find_optimal_k(max_k=min(max_k_allowed, 8), min_k=2)

                best_k_silhouette = optimal_results['silhouette']
                best_silhouette_score = optimal_results['silhouette_score']

                # Display optimal K results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Best K (Silhouette)",
                        best_k_silhouette,
                        f"{best_silhouette_score:.3f}"
                    )

                with col2:
                    st.metric(
                        "Best Silhouette Score",
                        f"{best_silhouette_score:.3f}",
                        "Higher is better"
                    )

                with col3:
                    st.metric(
                        "Tested K Range",
                        f"{optimal_results['k_values'][0]} - {optimal_results['k_values'][-1]}"
                    )

                # Elbow curve
                st.subheader("📈 Elbow Method")
                fig_elbow = plt.figure(figsize=(10, 4))
                plt.plot(optimal_results['k_values'], optimal_results['inertias'], 'bo-', linewidth=2, markersize=8)
                plt.axvline(best_k_silhouette, color='r', linestyle='--', label=f'Recommended K={best_k_silhouette}')
                plt.xlabel('Number of Clusters (K)', fontweight='bold')
                plt.ylabel('Inertia', fontweight='bold')
                plt.title('Elbow Curve - Look for the "elbow" point')
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(fig_elbow)
                st.caption("💡 The 'elbow point' shows diminishing returns - good balance point for K")

                # Silhouette scores
                st.subheader("📊 Silhouette Analysis")
                fig_silhouette = plt.figure(figsize=(10, 4))
                plt.plot(optimal_results['k_values'], optimal_results['silhouette_scores'], 'go-', linewidth=2, markersize=8)
                plt.axvline(best_k_silhouette, color='r', linestyle='--', label=f'Best K={best_k_silhouette}')
                plt.xlabel('Number of Clusters (K)', fontweight='bold')
                plt.ylabel('Silhouette Score', fontweight='bold')
                plt.title('Silhouette Score vs K - Higher is better')
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(fig_silhouette)
                st.caption("✨ Silhouette score (-1 to 1): Measures how well points fit their clusters")

            elif algorithm == "DBSCAN":
                # K-distance plot for eps selection
                st.subheader("📈 K-Distance Plot for Eps Selection")
                fig_kdist = plot_dbscan_k_distance(X_scaled, k=params['min_samples'])
                st.pyplot(fig_kdist)
                st.caption("💡 Look for the 'elbow' point in this plot to choose optimal eps value")
    
    # ============================================
    # CLUSTERING WITH SELECTED PARAMETERS
    # ============================================
    st.markdown("---")
    st.header(f"🎯 {algorithm} Clustering")

    with st.spinner(f"Clustering with {algorithm}..."):
        # Create and fit analyzer
        analyzer = ClusteringAnalyzer(X_scaled, algorithm=algorithm_key, random_state=config.RANDOM_STATE)

        # Fit with algorithm-specific parameters
        if algorithm == "K-Means":
            analyzer.fit(n_clusters=params['n_clusters'])
            iterations = analyzer.model.n_iter_
            inertia = analyzer.model.inertia_
            converged = iterations < config.KMEANS_MAX_ITER

            # Display clustering info
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Clusters", params['n_clusters'])

            with col2:
                st.metric("Iterations", iterations)

            with col3:
                st.metric("Inertia", f"{inertia:.2f}")

            with col4:
                st.metric("Converged", "✓ Yes" if converged else "✗ No")

        elif algorithm == "DBSCAN":
            analyzer.fit(eps=params['eps'], min_samples=params['min_samples'])

            # Display clustering info
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Clusters Found", analyzer.n_clusters_)

            with col2:
                st.metric("Core Samples", analyzer.model.core_sample_indices_.shape[0] if hasattr(analyzer.model, 'core_sample_indices_') else "N/A")

            with col3:
                st.metric("Noise Points", np.sum(analyzer.labels_ == -1))

            with col4:
                st.metric("Eps Used", f"{params['eps']:.1f}")

        elif algorithm == "Hierarchical":
            analyzer.fit(n_clusters=params['n_clusters'], linkage=params['linkage'])

            # Display clustering info
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Clusters", params['n_clusters'])

            with col2:
                st.metric("Linkage", params['linkage'].title())

            with col3:
                st.metric("Leaves", analyzer.model.n_leaves_)

            with col4:
                st.metric("Features", analyzer.model.n_features_in_)

        # Get results
        results = analyzer.evaluate()
        labels = analyzer.labels_
    
    # ============================================
    # EVALUATION METRICS
    # ============================================
    st.markdown("---")
    st.header("📈 Evaluation Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        silhouette = results.get('silhouette', None)
        if silhouette is None:
            silhouette_label = "N/A"
            silhouette_color = "⚪"
        else:
            silhouette_label = f"{silhouette:.3f}"
            silhouette_color = "🟢" if silhouette > 0.5 else "🟡" if silhouette > 0 else "🔴"

        st.metric(
            "Silhouette Score",
            silhouette_label,
            f"{silhouette_color} Range: -1 to +1"
        )
        st.caption("Measures cluster cohesion & separation\n(higher is better)")
    
    with col2:
        davies_bouldin = results.get('davies_bouldin', None)
        if davies_bouldin is None:
            davies_bouldin_label = "N/A"
            davies_bouldin_color = "⚪"
        else:
            davies_bouldin_label = f"{davies_bouldin:.3f}"
            davies_bouldin_color = "🟢" if davies_bouldin < 1 else "🟡" if davies_bouldin < 2 else "🔴"

        st.metric(
            "Davies-Bouldin Index",
            davies_bouldin_label,
            f"{davies_bouldin_color} Range: 0 to ∞"
        )
        st.caption("Cluster distinctiveness\n(lower is better)")
    
    with col3:
        calinski = results.get('calinski_harabasz', None)
        if calinski is None:
            calinski_label = "N/A"
        else:
            calinski_label = f"{calinski:.1f}"

        st.metric(
            "Calinski-Harabasz Score",
            calinski_label,
            "Higher is better"
        )
        st.caption("Ratio of between-cluster to within-cluster dispersion")
    
    # Cluster sizes
    st.subheader("👥 Cluster Distribution")
    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    # Handle DBSCAN noise points (label -1)
    if algorithm == "DBSCAN" and -1 in cluster_sizes.index:
        noise_count = cluster_sizes[-1]
        cluster_sizes = cluster_sizes.drop(-1)  # Remove noise from cluster sizes
        st.info(f"🔸 **{noise_count} points** classified as noise (outliers)")

    col1, col2 = st.columns([1, 2])

    with col1:
        if algorithm == "DBSCAN":
            cluster_df = pd.DataFrame({
                'Cluster': cluster_sizes.index,
                'Size': cluster_sizes.values,
                'Percentage': (cluster_sizes.values / len(labels) * 100).astype(int)
            })
            if -1 in pd.Series(labels).value_counts().index:
                noise_count = pd.Series(labels).value_counts()[-1]
                noise_df = pd.DataFrame({
                    'Cluster': ['Noise (-1)'],
                    'Size': [noise_count],
                    'Percentage': [int(noise_count / len(labels) * 100)]
                })
                cluster_df = pd.concat([cluster_df, noise_df], ignore_index=True)
        else:
            cluster_df = pd.DataFrame({
                'Cluster': cluster_sizes.index,
                'Size': cluster_sizes.values,
                'Percentage': (cluster_sizes.values / len(labels) * 100).astype(int)
            })
        st.dataframe(cluster_df, use_container_width=True)

    with col2:
        fig_dist = plt.figure(figsize=(8, 3))
        if algorithm == "DBSCAN" and -1 in pd.Series(labels).value_counts().index:
            # Plot clusters and noise separately
            cluster_labels = cluster_sizes.index
            cluster_counts = cluster_sizes.values
            plt.bar(cluster_labels, cluster_counts, color='steelblue', alpha=0.7, label='Clusters')
            plt.bar([-1], [noise_count], color='red', alpha=0.7, label='Noise')
            plt.xticks(list(cluster_labels) + [-1])
        else:
            plt.bar(cluster_sizes.index, cluster_sizes.values, color='steelblue', alpha=0.7)
        plt.xlabel('Cluster', fontweight='bold')
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.title(f'{algorithm} Cluster Distribution')
        plt.grid(True, alpha=0.3, axis='y')
        if algorithm == "DBSCAN":
            plt.legend()
        st.pyplot(fig_dist)
    
    # ============================================
    # CLUSTER VISUALIZATION
    # ============================================
    st.markdown("---")
    st.header("🎨 Cluster Visualization")

    from sklearn.decomposition import PCA

    # 2D Visualization
    st.subheader("2D Visualization (PCA)")

    if X_scaled.shape[1] >= 2:
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig_clusters = plt.figure(figsize=(10, 6))

        if algorithm == "DBSCAN":
            # Handle noise points (label -1) for DBSCAN
            core_mask = np.ones(len(labels), dtype=bool)
            if hasattr(analyzer.model, 'core_sample_indices_'):
                core_mask[:] = False
                core_mask[analyzer.model.core_sample_indices_] = True

            # Plot non-noise points
            non_noise_mask = labels != -1
            scatter = plt.scatter(X_pca[non_noise_mask, 0], X_pca[non_noise_mask, 1],
                                c=labels[non_noise_mask], cmap='viridis',
                                s=200, alpha=0.6, edgecolors='black', linewidth=1.5)

            # Plot noise points
            noise_mask = labels == -1
            if np.sum(noise_mask) > 0:
                plt.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1],
                           c='red', marker='x', s=100, alpha=0.8, label='Noise')

            plt.colorbar(scatter, label='Cluster')
            plt.legend()
        else:
            # For K-Means and Hierarchical
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis',
                                s=200, alpha=0.6, edgecolors='black', linewidth=1.5)

            # Plot cluster centers for K-Means
            if algorithm == "K-Means":
                centers = analyzer.get_cluster_centers()
                centers_pca = pca.transform(centers)
                plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X',
                           s=400, edgecolors='black', linewidth=2, label='Centroids')

            plt.colorbar(scatter, label='Cluster')
            if algorithm == "K-Means":
                plt.legend()

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
        plt.title(f'{algorithm} Clusters Visualization (PCA)', fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_clusters)

        st.caption(
            f"📊 PCA explained variance: {pca.explained_variance_ratio_[0]:.1%} + "
            f"{pca.explained_variance_ratio_[1]:.1%} = "
            f"{(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.1%}"
        )
    else:
        st.warning("⚠️ Need at least 2 features for visualization")

    # Algorithm-specific visualizations
    if algorithm == "Hierarchical":
        st.subheader("🔗 Hierarchical Clustering Dendrogram")
        with st.expander("View Dendrogram", expanded=False):
            fig_dendro = plot_hierarchical_dendrogram(
                X_scaled, n_clusters=analyzer.n_clusters_, linkage=analyzer.linkage_
            )
            st.pyplot(fig_dendro)

    # 3D Visualization (if enough features)
    if X_scaled.shape[1] >= 3:
        st.subheader("3D Visualization")
        with st.expander("View 3D Plot", expanded=False):
            centers = analyzer.get_cluster_centers() if algorithm == "K-Means" else None
            fig_3d = plot_clusters_3d(X_scaled, labels, centroids=centers)
            st.pyplot(fig_3d)

    # Cluster Profiles
    st.subheader("📊 Cluster Profiles")
    with st.expander("View Feature Profiles by Cluster", expanded=False):
        fig_profiles = plot_cluster_profiles(X_scaled, labels, feature_names=selected_features)
        st.pyplot(fig_profiles)
    
    # ============================================
    # RESULTS EXPORT
    # ============================================
    st.markdown("---")
    st.header("💾 Export Results")

    # Add cluster labels to original data
    df_results = df.copy()
    df_results['Cluster'] = labels

    # Create CSV download
    csv_buffer = StringIO()
    df_results.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Download Clustered Data (CSV)",
            data=csv_data,
            file_name="clustered_data.csv",
            mime="text/csv",
            help="Download your data with cluster assignments"
        )

    with col2:
        # Create summary report
        if algorithm == "K-Means":
            algorithm_params = f"- Number of Clusters (K): {params['n_clusters']}\n- Iterations: {analyzer.model.n_iter_}\n- Inertia: {analyzer.model.inertia_:.4f}"
        elif algorithm == "DBSCAN":
            noise_count = np.sum(labels == -1)
            algorithm_params = f"- Eps: {params['eps']}\n- Min Samples: {params['min_samples']}\n- Clusters Found: {analyzer.n_clusters_}\n- Noise Points: {noise_count}"
        elif algorithm == "Hierarchical":
            algorithm_params = f"- Number of Clusters (K): {params['n_clusters']}\n- Linkage Method: {params['linkage']}\n- Leaves: {analyzer.model.n_leaves_}"

        summary = f"""
        {algorithm.upper()} CLUSTERING REPORT
        =====================================
        Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

        ALGORITHM: {algorithm}
        {algorithm_params}

        DATASET:
        - Number of Samples: {len(df_results)}
        - Number of Features: {len(selected_features)}
        - Features Used: {', '.join(selected_features)}

        EVALUATION METRICS:
        - Silhouette Score: {results.get('silhouette', 'N/A')}
        - Davies-Bouldin Index: {results.get('davies_bouldin', 'N/A')}
        - Calinski-Harabasz Score: {results.get('calinski_harabasz', 'N/A')}

        CLUSTER DISTRIBUTION:
        {pd.Series(labels).value_counts().sort_index().to_string()}
        """

        st.download_button(
            label="📋 Download Report (TXT)",
            data=summary,
            file_name="clustering_report.txt",
            mime="text/plain",
            help="Download clustering summary report"
        )
    
    st.success("✅ Analysis complete!")

else:
    # No file uploaded yet
    st.info("👈 **Upload a CSV file** in the sidebar to get started!")
    
    st.markdown("---")
    st.header("📖 How to Use")
    st.markdown("""
    ### Step-by-Step Guide:

    1. **Choose Algorithm** 🤖
       - Select from K-Means, DBSCAN, or Hierarchical clustering
       - Each algorithm has different strengths and parameter requirements

    2. **Upload Data** 📁
       - Click "Choose a CSV file" in the sidebar
       - File should contain numeric columns for clustering

    3. **Select Features** 🎯
       - Choose which columns to use for clustering
       - Select numeric columns only

    4. **Set Parameters** 🔢
       - **K-Means**: Choose number of clusters (K)
       - **DBSCAN**: Set epsilon (eps) and minimum samples
       - **Hierarchical**: Choose K and linkage method

    5. **Find Optimal Parameters** 🔍
       - For K-Means: Use elbow method and silhouette analysis
       - For DBSCAN: Use k-distance plot to find optimal eps

    6. **Review Results** 📊
       - See clustering metrics and quality scores
       - View cluster visualizations (2D, 3D, profiles)
       - Check cluster distribution

    7. **Export Data** 💾
       - Download clustered data with assignments
       - Download analysis report

    ### Algorithm Guide:
    - **K-Means**: Good for spherical, evenly-sized clusters. Requires specifying K.
    - **DBSCAN**: Good for arbitrary-shaped clusters and noise detection. Automatically finds clusters.
    - **Hierarchical**: Good for visualizing cluster relationships. Creates a hierarchy of clusters.

    ### Metrics Explained:
    - **Silhouette Score**: -1 to +1 (higher is better) - measures how well points fit clusters
    - **Davies-Bouldin Index**: 0 to ∞ (lower is better) - measures cluster distinctiveness
    - **Calinski-Harabasz Score**: Higher is better - ratio of between-cluster to within-cluster dispersion
    """)
    
    st.markdown("---")
    st.header("📊 Example Data")
    st.markdown("""
    ### Need example data? Here's a sample CSV format:
    ```
    Feature1,Feature2,Feature3
    1.0,2.0,3.0
    1.5,1.8,2.9
    5.0,8.0,7.5
    ```
    
    Use **numeric values only** - remove any text columns before uploading.
    """)
