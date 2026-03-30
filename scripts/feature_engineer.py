# scripts/feature_engineer.py
"""Feature engineering module for crop yield prediction."""

import pandas as pd
import numpy as np
from .config import PROCESSED_DATA_PATH, FEATURES_DATA_PATH, ENGINEERED_FEATURES


def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate interaction features between temperature and rainfall."""
    # Temperature-Rainfall interaction
    df['temp_rainfall_interaction'] = df['avg_temp'] * df['average_rain_fall_mm_per_year']

    # Rainfall deviation from mean
    mean_rainfall = df['average_rain_fall_mm_per_year'].mean()
    df['rainfall_deviation'] = df['average_rain_fall_mm_per_year'] - mean_rainfall

    # Squared terms for non-linear relationships
    df['rainfall_squared'] = df['average_rain_fall_mm_per_year'] ** 2
    df['temp_squared'] = df['avg_temp'] ** 2

    # Ratio features
    df['pesticide_per_rainfall'] = df['pesticides_tonnes'] / (df['average_rain_fall_mm_per_year'] + 1)

    return df


def add_year_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features based on year (e.g., trend, season if applicable)."""
    # Year as numeric feature for trend capture
    df['year_normalized'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min())
    return df


def engineer_features(input_path: str = PROCESSED_DATA_PATH,
                    output_path: str = FEATURES_DATA_PATH) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Loads cleaned data, adds engineered features, and saves to output_path.
    """
    # Load cleaned data
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    # Apply feature engineering
    df = calculate_interaction_features(df)
    df = add_year_based_features(df)

    # Save to features path
    df.to_csv(output_path, index=False)

    return df


def get_feature_names() -> list:
    """Return list of all engineered feature names."""
    return ENGINEERED_FEATURES + ['year_normalized']


if __name__ == "__main__":
    print("Running feature engineering pipeline...")
    df = engineer_features()
    print(f"Engineered features saved. Shape: {df.shape}")
    print(f"New columns: {get_feature_names()}")
