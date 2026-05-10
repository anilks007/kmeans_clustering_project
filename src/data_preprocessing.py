"""
Data preprocessing module for K-Means clustering.
Handles data loading, cleaning, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(filepath, **kwargs):
    """
    Load data from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    **kwargs : dict
        Additional arguments to pass to pd.read_csv()
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        if config.VERBOSE:
            print(f"✓ Data loaded successfully from {filepath}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def handle_missing_values(df, method='drop', columns=None):
    """
    Handle missing values in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Method to handle missing values: 'drop', 'mean', 'median', 'mode'
    columns : list, optional
        Specific columns to handle. If None, handles all columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    missing_count = df_copy[columns].isnull().sum().sum()
    
    if missing_count == 0:
        if config.VERBOSE:
            print("✓ No missing values found")
        return df_copy
    
    if config.VERBOSE:
        print(f"⚠ Found {missing_count} missing values")
    
    if method == 'drop':
        df_copy = df_copy.dropna(subset=columns)
        if config.VERBOSE:
            print(f"  Dropped rows with missing values. New shape: {df_copy.shape}")
    
    elif method == 'mean':
        for col in columns:
            if df_copy[col].dtype in ['float64', 'int64']:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        if config.VERBOSE:
            print("  Filled missing values with column means")
    
    elif method == 'median':
        for col in columns:
            if df_copy[col].dtype in ['float64', 'int64']:
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
        if config.VERBOSE:
            print("  Filled missing values with column medians")
    
    elif method == 'mode':
        for col in columns:
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        if config.VERBOSE:
            print("  Filled missing values with column modes")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'drop', 'mean', 'median', or 'mode'")
    
    return df_copy


def select_numeric_features(df, exclude_columns=None):
    """
    Automatically select numeric features from dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    exclude_columns : list, optional
        Columns to exclude from selection
    
    Returns:
    --------
    list
        List of numeric column names
    """
    if exclude_columns is None:
        exclude_columns = config.EXCLUDE_COLUMNS
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
    
    if config.VERBOSE:
        print(f"✓ Selected {len(numeric_cols)} numeric features: {numeric_cols}")
    
    return numeric_cols


def scale_features(X, method='standard'):
    """
    Scale features using specified method.
    
    Parameters:
    -----------
    X : array-like or pd.DataFrame
        Features to scale
    method : str
        Scaling method: 'standard', 'minmax', or 'robust'
    
    Returns:
    --------
    tuple
        (X_scaled, scaler) - Scaled features and fitted scaler object
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_scaled = scaler.fit_transform(X)
    
    if config.VERBOSE:
        print(f"✓ Features scaled using {method} scaling")
        print(f"  Original shape: {X.shape}")
        print(f"  Scaled shape: {X_scaled.shape}")
    
    return X_scaled, scaler


def preprocess_features(df, feature_columns=None, handle_missing=None, scaling_method=None):
    """
    Complete preprocessing pipeline for features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_columns : list, optional
        Columns to use as features. If None, auto-selects numeric columns
    handle_missing : str, optional
        Method to handle missing values. If None, uses config default
    scaling_method : str, optional
        Scaling method. If None, uses config default
    
    Returns:
    --------
    tuple
        (X_scaled, scaler, df_clean) - Scaled features, scaler object, and cleaned dataframe
    """
    if config.VERBOSE:
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
    
    # Use config defaults if not specified
    if handle_missing is None:
        handle_missing = config.HANDLE_MISSING
    if scaling_method is None:
        scaling_method = config.SCALING_METHOD
    
    # Select features
    if feature_columns is None:
        if config.AUTO_SELECT_FEATURES:
            feature_columns = select_numeric_features(df)
        else:
            raise ValueError("feature_columns must be specified or AUTO_SELECT_FEATURES must be True in config")
    
    # Handle missing values
    df_clean = handle_missing_values(df, method=handle_missing, columns=feature_columns)
    
    # Extract features
    X = df_clean[feature_columns].values
    
    # Scale features
    X_scaled, scaler = scale_features(X, method=scaling_method)
    
    if config.VERBOSE:
        print("="*60)
        print("✓ Preprocessing complete!")
        print("="*60 + "\n")
    
    return X_scaled, scaler, df_clean


def load_and_preprocess_data(filepath, feature_columns=None, **kwargs):
    """
    Complete pipeline: load data and preprocess features.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    feature_columns : list, optional
        Columns to use as features
    **kwargs : dict
        Additional arguments for preprocessing
    
    Returns:
    --------
    tuple
        (X_scaled, scaler, df_clean) - Scaled features, scaler object, and cleaned dataframe
    """
    # Load data
    df = load_data(filepath)
    
    # Preprocess
    X_scaled, scaler, df_clean = preprocess_features(df, feature_columns, **kwargs)
    
    return X_scaled, scaler, df_clean


def inverse_transform_centroids(centroids, scaler, feature_names=None):
    """
    Transform scaled centroids back to original scale.
    
    Parameters:
    -----------
    centroids : array-like
        Scaled centroid coordinates
    scaler : sklearn scaler object
        Fitted scaler used for original transformation
    feature_names : list, optional
        Names of features for the resulting dataframe
    
    Returns:
    --------
    pd.DataFrame or np.ndarray
        Centroids in original scale
    """
    centroids_original = scaler.inverse_transform(centroids)
    
    if feature_names is not None:
        return pd.DataFrame(centroids_original, columns=feature_names)
    
    return centroids_original


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    print("Creating example dataset...")
    
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'State': [f'State_{i}' for i in range(50)],
        'Health_Index_1': np.random.uniform(50, 100, 50),
        'Health_Index_2': np.random.uniform(40, 95, 50),
        'Per_Capita_Income': np.random.uniform(20000, 80000, 50),
        'GDP': np.random.uniform(100000, 500000, 50)
    }
    
    # Add some missing values
    sample_data['Health_Index_1'][5] = np.nan
    sample_data['GDP'][10] = np.nan
    
    df = pd.DataFrame(sample_data)
    
    # Save sample data
    os.makedirs(config.DATA_DIR, exist_ok=True)
    sample_path = os.path.join(config.DATA_DIR, 'sample_data.csv')
    df.to_csv(sample_path, index=False)
    print(f"Sample data saved to {sample_path}")
    
    # Test preprocessing
    feature_cols = ['Health_Index_1', 'Health_Index_2', 'Per_Capita_Income', 'GDP']
    X_scaled, scaler, df_clean = preprocess_features(df, feature_columns=feature_cols)
    
    print(f"\nPreprocessed data shape: {X_scaled.shape}")
    print(f"Clean dataframe shape: {df_clean.shape}")
