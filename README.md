# K-Means Clustering Project

A comprehensive Python project for performing K-Means clustering analysis on datasets, with a focus on best practices and proper evaluation metrics.

## 📋 Project Overview

This project implements K-Means clustering with:
- Proper data preprocessing and standardization
- Optimal cluster number determination (Elbow & Silhouette methods)
- Multiple evaluation metrics (Silhouette Score, Davies-Bouldin Index)
- Comprehensive visualizations
- Modular, reusable code structure

## 🚀 Quick Start

### Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Jupyter Notebook (Recommended for beginners)
```bash
jupyter notebook notebooks/kmeans_analysis.ipynb
```

#### Option 2: Python Scripts
```python
from src.data_preprocessing import load_and_preprocess_data
from src.clustering import find_optimal_clusters, fit_kmeans
from src.evaluation import evaluate_clustering
from src.visualization import plot_elbow_curve, plot_clusters

# Load and preprocess data
X_scaled, scaler, df = load_and_preprocess_data('data/your_data.csv')

# Find optimal number of clusters
optimal_k = find_optimal_clusters(X_scaled, max_k=10)

# Fit model
kmeans, labels = fit_kmeans(X_scaled, n_clusters=optimal_k)

# Evaluate
metrics = evaluate_clustering(X_scaled, labels)

# Visualize
plot_clusters(X_scaled, labels, method='pca')
```

## 📁 Project Structure

```
kmeans_clustering_project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # Configuration settings
├── data/                              # Data directory
│   └── .gitkeep
├── notebooks/                         # Jupyter notebooks
│   └── kmeans_analysis.ipynb         # Main analysis notebook
├── src/                               # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py         # Data loading & cleaning
│   ├── clustering.py                 # KMeans implementation
│   ├── evaluation.py                 # Metrics & evaluation
│   └── visualization.py              # Plotting functions
└── results/                           # Output directory
    └── .gitkeep
```

## 🔑 Key Features

### 1. Data Preprocessing
- Automatic missing value handling
- Feature standardization using StandardScaler
- Support for CSV data files

### 2. Optimal Cluster Selection
- **Elbow Method**: Visual identification of optimal K
- **Silhouette Analysis**: Quantitative cluster quality assessment
- Automated K selection based on multiple metrics

### 3. Evaluation Metrics
- **Silhouette Score**: Measures cluster cohesion and separation
- **Davies-Bouldin Index**: Evaluates cluster distinctiveness
- **Inertia**: Within-cluster sum of squares

### 4. Visualizations
- Elbow curve plot
- Silhouette score plot
- 2D cluster visualization (using PCA)
- Cluster characteristics analysis

## 📊 Example Workflow

```python
import pandas as pd
from src.data_preprocessing import load_and_preprocess_data
from src.clustering import KMeansAnalyzer
from src.visualization import plot_all_visualizations

# 1. Load your data
df = pd.read_csv('data/states_data.csv')

# 2. Preprocess
X_scaled, scaler, df_clean = load_and_preprocess_data(
    'data/states_data.csv',
    feature_columns=['Health_indices1', 'Health_indices2', 'Per_capita_income', 'GDP']
)

# 3. Create analyzer
analyzer = KMeansAnalyzer(X_scaled)

# 4. Find optimal K
analyzer.find_optimal_k(max_k=10)

# 5. Fit with optimal K
analyzer.fit(n_clusters=3)

# 6. Evaluate
results = analyzer.evaluate()
print(f"Silhouette Score: {results['silhouette']:.4f}")
print(f"Davies-Bouldin Index: {results['davies_bouldin']:.4f}")

# 7. Visualize
plot_all_visualizations(analyzer, X_scaled)

# 8. Get cluster assignments
df_clean['Cluster'] = analyzer.labels_
df_clean.to_csv('results/clustered_data.csv', index=False)
```

## 📈 Interpreting Results

### Silhouette Score
- **0.71 - 1.00**: Strong, well-separated clusters
- **0.51 - 0.70**: Reasonable structure
- **0.26 - 0.50**: Weak structure
- **< 0.25**: No substantial structure

### Davies-Bouldin Index
- **Lower is better**
- **0 - 2**: Good clustering
- **> 2**: Poor clustering

### Elbow Method
- Look for the "elbow" point where inertia decrease slows
- This indicates diminishing returns from adding more clusters

## 🛠️ Configuration

Edit `config.py` to customize:
- Random seed for reproducibility
- Default number of clusters
- Visualization settings
- File paths

## 📚 Dependencies

- Python 3.7+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter

## 🎯 Best Practices Implemented

✅ Feature standardization before clustering  
✅ Multiple methods for determining optimal K  
✅ Comprehensive evaluation metrics  
✅ Reproducible results (random_state)  
✅ Proper handling of missing values  
✅ Modular, reusable code  
✅ Clear documentation and examples  

## 📝 Common Pitfalls Avoided

❌ Forgetting to standardize features  
❌ Not visualizing results  
❌ Picking K arbitrarily  
❌ Not handling missing values  
❌ Using categorical features without encoding  
❌ Not setting random state  

## 🤝 Contributing

Feel free to extend this project with:
- Additional clustering algorithms (DBSCAN, Hierarchical)
- More evaluation metrics
- Interactive visualizations (Plotly)
- CLI interface
- Automated reporting

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

Based on K-Means clustering best practices and scikit-learn documentation.

---

**Happy Clustering! 🎯**
