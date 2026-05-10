"""
Configuration file for K-Means clustering project.
Modify these settings to customize the behavior of the clustering analysis.
"""

# Random seed for reproducibility
RANDOM_STATE = 42

# Clustering parameters
DEFAULT_N_CLUSTERS = 3
MAX_CLUSTERS_TO_TEST = 10
MIN_CLUSTERS_TO_TEST = 2

# KMeans algorithm parameters
KMEANS_INIT = 'k-means++'  # Initialization method: 'k-means++' or 'random'
KMEANS_N_INIT = 10  # Number of times KMeans will run with different centroid seeds
KMEANS_MAX_ITER = 300  # Maximum number of iterations

# Data preprocessing
HANDLE_MISSING = 'drop'  # Options: 'drop', 'mean', 'median', 'mode'
SCALING_METHOD = 'standard'  # Options: 'standard', 'minmax', 'robust'

# Visualization settings
FIGURE_SIZE = (10, 6)
DPI = 100
STYLE = 'seaborn-v0_8-darkgrid'  # Matplotlib style
COLOR_PALETTE = 'viridis'  # Color palette for clusters

# File paths
DATA_DIR = 'data/'
RESULTS_DIR = 'results/'
NOTEBOOKS_DIR = 'notebooks/'

# Output settings
SAVE_FIGURES = True
FIGURE_FORMAT = 'png'  # Options: 'png', 'jpg', 'svg', 'pdf'
SAVE_RESULTS = True

# Logging
VERBOSE = True  # Print progress messages
LOG_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# Feature engineering
AUTO_SELECT_FEATURES = False  # Automatically select numeric features
EXCLUDE_COLUMNS = ['id', 'ID', 'index']  # Columns to exclude from clustering

# Evaluation
CALCULATE_ALL_METRICS = True  # Calculate all available metrics
SILHOUETTE_SAMPLE_SIZE = None  # None for all samples, or specify a number for large datasets

# Advanced settings
USE_PCA_FOR_VISUALIZATION = True  # Use PCA to reduce to 2D for visualization
PCA_COMPONENTS = 2  # Number of PCA components for visualization
ENABLE_PARALLEL = False  # Use parallel processing (requires joblib)
N_JOBS = -1  # Number of parallel jobs (-1 uses all processors)
