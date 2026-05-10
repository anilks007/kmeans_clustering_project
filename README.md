# Clustering Analysis Project

A Python clustering project with an interactive Streamlit dashboard and support for multiple clustering algorithms.

## 📋 Project Overview

This project now supports:
- **K-Means** clustering
- **DBSCAN** density-based clustering
- **Hierarchical** agglomerative clustering
- Interactive parameter tuning in a Streamlit app
- Cluster validation metrics and visual diagnostics
- CSV export of clustered results and summary report

## 🚀 Quick Start

### Installation

1. Clone or download this project.
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the Streamlit App

Start the web dashboard with:
```bash
python -m streamlit run app.py
```

Then open the URL shown in the terminal, typically:
```
http://localhost:8501
```

### What You Can Do in the App

- Upload your CSV dataset
- Select numeric features for clustering
- Choose between **K-Means**, **DBSCAN**, and **Hierarchical**
- Adjust algorithm parameters
- View optimal parameter guidance
- Inspect cluster metrics and visualizations
- Download clustered data and a report

## 📁 Project Structure

```
kmeans_clustering_project/
├── README.md
├── requirements.txt
├── config.py
├── app.py
├── ClustruingExample.py
├── data/
│   ├── feature_data.csv
│   └── SampleData.txt
└── src/
    ├── __init__.py
    ├── clustering.py
    ├── data_preprocessing.py
    ├── evaluation.py
    └── visualization.py
```

## 🧠 Supported Algorithms

- **K-Means**: good for well-separated, spherical clusters.
- **DBSCAN**: works with arbitrary cluster shapes and identifies noise.
- **Hierarchical**: builds cluster hierarchies and supports dendrogram visualization.

## 📌 Usage Example (Python)

```python
import pandas as pd
from src.data_preprocessing import load_data, preprocess_features
from src.clustering import ClusteringAnalyzer
from src.visualization import plot_clusters, plot_cluster_profiles

# Load data
csv_path = 'data/feature_data.csv'
df = load_data(csv_path)

# Preprocess features
X_scaled, scaler, df_clean = preprocess_features(
    df,
    feature_columns=['Feature1', 'Feature2']
)

# Run K-Means clustering
analyzer = ClusteringAnalyzer(X_scaled, algorithm='kmeans')
analyzer.fit(n_clusters=3)
results = analyzer.evaluate()
print(results)

# Visualize
fig = plot_clusters(X_scaled, analyzer.labels_)
fig_profiles = plot_cluster_profiles(X_scaled, analyzer.labels_, feature_names=['Feature1', 'Feature2'])
```

## 📈 Evaluation Metrics

The app and code support:
- **Silhouette Score**: cluster cohesion/separation
- **Davies-Bouldin Index**: cluster distinctiveness
- **Calinski-Harabasz Score**: between-cluster vs within-cluster dispersion
- **Inertia** (for K-Means)

## 🔧 Configuration

Edit `config.py` to customize:
- random seed
- default cluster settings
- visualization style
- save paths and file outputs

## 📦 Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- scipy

## 🌟 Notes

- The dashboard is the easiest way to explore the algorithms interactively.
- Use `DBSCAN` when you expect noise or non-spherical clusters.
- Use `Hierarchical` to inspect cluster structure with a dendrogram.
- Use `K-Means` for faster, centroid-based clustering.

## 🤝 Contributing

Contributions and improvements are welcome, including:
- new clustering algorithms
- additional visualizations
- CLI mode
- more dataset examples

## 📄 License

Add your license text here.

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

Based on K-Means clustering best practices and scikit-learn documentation.

---

**Happy Clustering! 🎯**
