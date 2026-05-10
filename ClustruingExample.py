import pandas as pd
from src.data_preprocessing import load_and_preprocess_data
from src.clustering import KMeansAnalyzer
from src.visualization import plot_all_visualizations

# Load your data
df = pd.read_csv('data/feature_data.csv')

# Preprocess the data
X_scaled, scaler, df_clean = load_and_preprocess_data(
    'data/feature_data.csv',
    feature_columns=['Feature1', 'Feature2']
)

# Create KMeansAnalyzer instance
analyzer = KMeansAnalyzer(X_scaled)

# Find optimal K
analyzer.find_optimal_k(max_k=5)

# Fit with optimal K (assuming optimal K is found to be 3)
optimal_k = 3
analyzer.fit(n_clusters=optimal_k)

# Evaluate
results = analyzer.evaluate()
print(f"Silhouette Score: {results['silhouette']:.4f}")
print(f"Davies-Bouldin Index: {results['davies_bouldin']:.4f}")

# Visualize
plot_all_visualizations(analyzer, X_scaled)

# Get cluster assignments
df_clean['Cluster'] = analyzer.labels_
df_clean.to_csv('results/clustered_data.csv', index=False)
print("Clustering analysis completed and results saved to 'results/clustered_data.csv'.")
