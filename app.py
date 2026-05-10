"""
Streamlit web application for K-Means Clustering Analysis
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
from src.clustering import KMeansAnalyzer
from src.visualization import (
    plot_elbow_curve, 
    plot_silhouette_analysis, 
    plot_cluster_comparison
)
import config

# Page configuration
st.set_page_config(
    page_title="K-Means Clustering Dashboard",
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
st.title("🎯 K-Means Clustering Analysis Dashboard")
st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    ### Welcome to the Interactive Clustering Tool
    Upload your data, adjust parameters, and explore clustering results in real-time!
    """)
with col2:
    st.info("💡 **How it works:**\n1. Upload CSV\n2. Set K value\n3. View results")

# ============================================
# SIDEBAR - FILE UPLOAD & PARAMETERS
# ============================================
st.sidebar.header("⚙️ Configuration")

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
    
    # K value selection
    st.sidebar.subheader("🔢 Clustering Parameters")
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
        st.metric("Clusters (K)", k_value)
    
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
    # FIND OPTIMAL K
    # ============================================
    if find_optimal:
        st.markdown("---")
        st.header("🔍 Finding Optimal K")
        
        with st.spinner("Analyzing... This may take a moment..."):
            analyzer = KMeansAnalyzer(X_scaled, random_state=config.RANDOM_STATE)
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
    
    # ============================================
    # CLUSTERING WITH SELECTED K
    # ============================================
    st.markdown("---")
    st.header("🎯 K-Means Clustering")
    
    with st.spinner(f"Clustering with K={k_value}..."):
        # Create and fit analyzer
        analyzer = KMeansAnalyzer(X_scaled, random_state=config.RANDOM_STATE)
        analyzer.fit(n_clusters=k_value)
        
        # Get results
        results = analyzer.evaluate()
        labels = analyzer.labels_
        centers = analyzer.get_cluster_centers()
        
        iterations = analyzer.kmeans.n_iter_
        inertia = analyzer.kmeans.inertia_
        converged = iterations < config.KMEANS_MAX_ITER
        
        # Display clustering info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Clusters", k_value)
        
        with col2:
            st.metric("Iterations", iterations)
        
        with col3:
            st.metric("Inertia", f"{inertia:.2f}")
        
        with col4:
            st.metric("Converged", "✓ Yes" if converged else "✗ No")
    
    # ============================================
    # EVALUATION METRICS
    # ============================================
    st.markdown("---")
    st.header("📈 Evaluation Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        silhouette = results.get('silhouette', 0)
        color = "🟢" if silhouette > 0.5 else "🟡" if silhouette > 0 else "🔴"
        st.metric(
            "Silhouette Score",
            f"{silhouette:.3f}",
            f"{color} Range: -1 to +1"
        )
        st.caption("Measures cluster cohesion & separation\n(higher is better)")
    
    with col2:
        davies_bouldin = results.get('davies_bouldin', 0)
        color = "🟢" if davies_bouldin < 1 else "🟡" if davies_bouldin < 2 else "🔴"
        st.metric(
            "Davies-Bouldin Index",
            f"{davies_bouldin:.3f}",
            f"{color} Range: 0 to ∞"
        )
        st.caption("Cluster distinctiveness\n(lower is better)")
    
    with col3:
        calinski = results.get('calinski_harabasz', 0)
        st.metric(
            "Calinski-Harabasz Score",
            f"{calinski:.1f}",
            "Higher is better"
        )
        st.caption("Ratio of between-cluster to within-cluster dispersion")
    
    # Cluster sizes
    st.subheader("👥 Cluster Distribution")
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(
            pd.DataFrame({
                'Cluster': cluster_sizes.index,
                'Size': cluster_sizes.values,
                'Percentage': (cluster_sizes.values / len(labels) * 100).astype(int)
            }),
            use_container_width=True
        )
    
    with col2:
        fig_dist = plt.figure(figsize=(8, 3))
        plt.bar(cluster_sizes.index, cluster_sizes.values, color='steelblue', alpha=0.7)
        plt.xlabel('Cluster', fontweight='bold')
        plt.ylabel('Number of Samples', fontweight='bold')
        plt.title('Samples per Cluster')
        plt.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig_dist)
    
    # ============================================
    # CLUSTER VISUALIZATION
    # ============================================
    st.markdown("---")
    st.header("🎨 Cluster Visualization")
    
    from sklearn.decomposition import PCA
    
    st.subheader("2D Visualization (PCA)")
    
    if X_scaled.shape[1] >= 2:
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig_clusters = plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                            s=200, alpha=0.6, edgecolors='black', linewidth=1.5)
        
        # Plot cluster centers (transformed)
        centers_pca = pca.transform(centers)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', 
                   s=400, edgecolors='black', linewidth=2, label='Centroids')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
        plt.title('K-Means Clusters Visualization (PCA)', fontweight='bold', fontsize=14)
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_clusters)
        
        st.caption(
            f"📊 PCA explained variance: {pca.explained_variance_ratio_[0]:.1%} + "
            f"{pca.explained_variance_ratio_[1]:.1%} = "
            f"{(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.1%}"
        )
    else:
        st.warning("⚠️ Need at least 2 features for visualization")
    
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
        summary = f"""
        K-MEANS CLUSTERING REPORT
        ==========================
        Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        PARAMETERS:
        - Number of Clusters (K): {k_value}
        - Number of Samples: {len(df_results)}
        - Number of Features: {len(selected_features)}
        - Features Used: {', '.join(selected_features)}
        
        METRICS:
        - Silhouette Score: {results.get('silhouette', 0):.4f}
        - Davies-Bouldin Index: {results.get('davies_bouldin', 0):.4f}
        - Calinski-Harabasz Score: {results.get('calinski_harabasz', 0):.2f}
        - Inertia: {analyzer.kmeans.inertia_:.4f}
        
        CLUSTER DISTRIBUTION:
        {cluster_sizes.to_string()}
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
    
    1. **Upload Data** 📁
       - Click "Choose a CSV file" in the sidebar
       - File should contain numeric columns for clustering
    
    2. **Select Features** 🎯
       - Choose which columns to use for clustering
       - Select numeric columns only
    
    3. **Find Optimal K** 🔍
       - Check the "Find Optimal K First" option
       - Dashboard will show Elbow Method and Silhouette Analysis
       - This helps identify the best number of clusters
    
    4. **Adjust K Value** 🔢
       - Use the slider to select number of clusters
       - Or follow the recommendation from optimal K analysis
    
    5. **Review Results** 📊
       - See clustering metrics and quality scores
       - View cluster visualization
       - Check cluster distribution
    
    6. **Export Data** 💾
       - Download clustered data with assignments
       - Download analysis report
    
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
