"""
Day 1: Exploratory Data Analysis & Data Preprocessing

This script demonstrates key EDA and preprocessing techniques:
- Loading and exploring a dataset
- Data cleaning and handling missing values
- Outlier detection and removal
- Feature scaling (StandardScaler and MinMaxScaler)
- Data visualization (histograms, box plots, scatter plots)
- Statistical analysis and correlation
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Allow importing from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import (
    check_missing_values,
    handle_outliers,
    scale_features,
    plot_feature_distribution,
    plot_correlation_matrix,
)


def load_dataset():
    """Load the Iris dataset into a pandas DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df


def explore_data(df):
    """Perform initial exploration of the dataset."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nBasic statistics:\n{df.describe()}")
    print(f"\nSpecies distribution:\n{df['species'].value_counts()}")


def clean_data(df):
    """
    Demonstrate data cleaning techniques.

    Introduces synthetic missing values and duplicates, then cleans them.
    """
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    df_dirty = df.copy()

    # Introduce missing values for demonstration
    rng = np.random.default_rng(42)
    for col in df_dirty.select_dtypes(include=[np.number]).columns:
        mask = rng.random(len(df_dirty)) < 0.05
        df_dirty.loc[mask, col] = np.nan

    # Introduce duplicate rows for demonstration
    duplicates = df_dirty.sample(n=5, random_state=42)
    df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)

    print(f"\nDirty dataset shape: {df_dirty.shape}")

    # --- Check missing values using utility ---
    print("\n--- Missing Values ---")
    check_missing_values(df_dirty)

    # --- Handle missing values: impute with median ---
    numeric_cols = df_dirty.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df_dirty[col].median()
        df_dirty[col] = df_dirty[col].fillna(median_val)

    print("\nAfter imputation:")
    check_missing_values(df_dirty)

    # --- Remove duplicates ---
    n_before = len(df_dirty)
    df_dirty = df_dirty.drop_duplicates()
    n_after = len(df_dirty)
    print(f"\nRemoved {n_before - n_after} duplicate rows")
    print(f"Clean dataset shape: {df_dirty.shape}")

    return df_dirty


def detect_outliers(df):
    """Detect and handle outliers using the IQR method."""
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Show outlier statistics before removal
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col}: {len(outliers)} outlier(s) detected")

    # Handle outliers using utility function
    df_clean = handle_outliers(df, numeric_cols, method="iqr", threshold=1.5)
    print(f"\nRows before outlier removal: {len(df)}")
    print(f"Rows after outlier removal:  {len(df_clean)}")

    return df_clean


def perform_feature_scaling(df):
    """Demonstrate Standard and Min-Max feature scaling."""
    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = df[numeric_cols].values

    # Standard scaling
    std_scaler = StandardScaler()
    features_standard = std_scaler.fit_transform(features)
    print("\nStandard Scaling (mean=0, std=1):")
    print(f"  Mean:  {features_standard.mean(axis=0).round(4)}")
    print(f"  Std:   {features_standard.std(axis=0).round(4)}")

    # Min-Max scaling
    mm_scaler = MinMaxScaler()
    features_minmax = mm_scaler.fit_transform(features)
    print("\nMin-Max Scaling (range 0-1):")
    print(f"  Min:   {features_minmax.min(axis=0).round(4)}")
    print(f"  Max:   {features_minmax.max(axis=0).round(4)}")


def visualize_data(df):
    """Create visualizations: histograms, box plots, scatter, and correlation."""
    print("\n" + "=" * 60)
    print("DATA VISUALIZATION")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Histograms and box plots via utility
    print("\nGenerating feature distribution plots...")
    fig = plot_feature_distribution(df, numeric_cols, figsize=(14, 12))
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "feature_distributions.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Correlation matrix via utility
    print("Generating correlation matrix...")
    fig = plot_correlation_matrix(df[numeric_cols], figsize=(8, 6))
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "correlation_matrix.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Scatter plot: petal length vs petal width coloured by species
    print("Generating scatter plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    for species in df["species"].unique():
        subset = df[df["species"] == species]
        ax.scatter(
            subset["petal length (cm)"],
            subset["petal width (cm)"],
            label=species,
            alpha=0.7,
        )
    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.set_title("Petal Length vs Petal Width by Species")
    ax.legend()
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "scatter_plot.png"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Visualizations saved to day_1_eda_and_preprocessing/")


def statistical_analysis(df):
    """Perform basic statistical analysis."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Summary statistics per species
    print("\nStatistics grouped by species:")
    print(df.groupby("species")[numeric_cols.tolist()].mean().round(3))

    # Correlation analysis
    corr = df[numeric_cols].corr()
    print("\nCorrelation matrix:")
    print(corr.round(3))

    # Highlight strong correlations
    print("\nStrong correlations (|r| > 0.7):")
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7:
                print(f"  {corr.columns[i]} <-> {corr.columns[j]}: {r:.3f}")


def main():
    """Run the full Day 1 EDA and Preprocessing pipeline."""
    # Step 1: Load dataset
    df = load_dataset()

    # Step 2: Explore data
    explore_data(df)

    # Step 3: Data cleaning (missing values & duplicates)
    df_clean = clean_data(df)

    # Step 4: Outlier detection
    df_clean = detect_outliers(df_clean)

    # Step 5: Feature scaling
    perform_feature_scaling(df_clean)

    # Step 6: Visualization
    visualize_data(df_clean)

    # Step 7: Statistical analysis
    statistical_analysis(df_clean)

    print("\n" + "=" * 60)
    print("Day 1 EDA & Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
