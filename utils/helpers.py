"""
Common utility functions for ML projects
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data(filepath, sample_size=None):
    """
    Load and perform initial exploration of dataset
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    sample_size : int, optional
        Number of rows to display
    
    Returns:
    --------
    df : pd.DataFrame
        Loaded dataframe
    """
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head(sample_size or 5)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    return df

def check_missing_values(df):
    """Check and visualize missing values"""
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found!")
    else:
        print(f"Missing values:\n{missing[missing > 0]}")
        missing_percent = (missing / len(df)) * 100
        print(f"\nMissing percentage:\n{missing_percent[missing_percent > 0]}")
    return missing

def handle_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Handle outliers using IQR or Z-score method
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to check for outliers
    method : str
        'iqr' or 'zscore'
    threshold : float
        IQR threshold or zscore threshold
    
    Returns:
    --------
    df_clean : pd.DataFrame
        Dataframe with outliers handled
    """
    df_clean = df.copy()
    
    if method == 'iqr':
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    return df_clean

def scale_features(X_train, X_test, method='standard'):
    """
    Scale features using StandardScaler or MinMaxScaler
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    method : str
        'standard' or 'minmax'
    
    Returns:
    --------
    X_train_scaled, X_test_scaled : array-like
    scaler : sklearn scaler object
    """
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def plot_feature_distribution(df, columns, figsize=(15, 10)):
    """Plot distribution of features"""
    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols, 2, figsize=figsize)
    
    for idx, col in enumerate(columns):
        axes[idx, 0].hist(df[col], bins=30, edgecolor='black')
        axes[idx, 0].set_title(f'Histogram: {col}')
        axes[idx, 0].set_xlabel(col)
        axes[idx, 0].set_ylabel('Frequency')
        
        axes[idx, 1].boxplot(df[col])
        axes[idx, 1].set_title(f'Boxplot: {col}')
        axes[idx, 1].set_ylabel(col)
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df, figsize=(10, 8)):
    """Plot correlation matrix heatmap"""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    plt.title('Correlation Matrix')
    return fig

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate classification model
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

print("Helpers module loaded successfully!")