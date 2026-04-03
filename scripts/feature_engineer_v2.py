"""
Advanced Feature Engineering Module (v2) - Unified Pipeline
Combines base, interaction, and professional agronomic features.
Includes fallback mechanisms for external API outages.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

# Default historical averages for India (fallback when APIs fail)
DEFAULT_SOIL_PH = 6.5
DEFAULT_SOIL_NITROGEN = 150  # mg/kg
DEFAULT_SOIL_ORGANIC_CARBON = 12.5  # g/kg
DEFAULT_NDVI = 0.65  # Healthy crop baseline


# ---------------------------------------------------------------------------
# Yield standardization
# ---------------------------------------------------------------------------

def standardize_yield(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all yield columns to kg/ha units and remove legacy columns."""
    if 'hg/ha_yield' in df.columns and 'yield_kg_ha' not in df.columns:
        df['yield_kg_ha'] = df['hg/ha_yield'] / 10.0

    if 'kg_per_ha_yield' in df.columns and 'yield_kg_ha' not in df.columns:
        df['yield_kg_ha'] = df['kg_per_ha_yield']

    # Remove legacy yield columns to avoid duplication
    for col in ['hg/ha_yield', 'kg_per_ha_yield']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Interaction and polynomial features (legacy from v1, retained)
# ---------------------------------------------------------------------------

def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate interaction features between temperature and rainfall."""
    df['temp_rainfall_interaction'] = df['avg_temp'] * df['average_rain_fall_mm_per_year']
    df['rainfall_deviation'] = (
        df['average_rain_fall_mm_per_year'] - df['average_rain_fall_mm_per_year'].mean()
    )
    df['rainfall_squared'] = df['average_rain_fall_mm_per_year'] ** 2
    df['temp_squared'] = df['avg_temp'] ** 2
    df['pesticide_per_rainfall'] = df['pesticides_tonnes'] / (df['average_rain_fall_mm_per_year'] + 1)
    return df


def add_year_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalization and trend-based year features."""
    year_min = df['Year'].min()
    year_max = df['Year'].max()
    df['year_normalized'] = (
        (df['Year'] - year_min) / (year_max - year_min) if year_max != year_min else 1.0
    )
    return df


# ---------------------------------------------------------------------------
# Climate stress indices (professional agronomic features)
# ---------------------------------------------------------------------------

def calculate_stress_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heat Stress and Drought Intensity metrics."""
    # Heat Stress Degree Days (simplified: count degrees > 35C threshold)
    df['heat_stress_degreedays'] = df['avg_temp'].apply(lambda x: max(0.0, x - 35.0))

    # Drought Intensity Index (1.0 if rainfall < 500mm, scaled 0-1 for higher)
    df['drought_intensity'] = df['average_rain_fall_mm_per_year'].apply(
        lambda x: max(0.0, 1.0 - (x / 500.0)) if x < 500 else 0.0
    )
    return df


# ---------------------------------------------------------------------------
# Satellite proxy and soil health
# ---------------------------------------------------------------------------

def add_satellite_soil_features(
    df: pd.DataFrame,
    soil_data: Optional[pd.DataFrame] = None,
    ndvi_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add NDVI proxy and soil NPK/pH columns.
    Falls back to defaults when external data is unavailable.
    """
    # 1. NDVI (from Sentinel or proxy)
    if ndvi_data is not None and not ndvi_data.empty:
        merged = df.merge(ndvi_data, on=['lat', 'lon'], how='left')
        df['ndvi'] = merged.get('ndvi', DEFAULT_NDVI)
    else:
        df['ndvi'] = DEFAULT_NDVI

    # NDVI adjusted for rainfall and heat stress correlation
    df['ndvi_adjusted'] = df['ndvi'] + (
        df['average_rain_fall_mm_per_year'] / 2000.0
    ) - (df.get('heat_stress_degreedays', pd.Series(0, index=df.index)) / 10.0)
    df['ndvi_adjusted'] = df['ndvi_adjusted'].clip(0, 1)

    # 2. Soil Health (NPK, pH, Organic Carbon)
    if soil_data is not None and not soil_data.empty and 'lat' in soil_data.columns:
        merged = df.merge(soil_data, on=['lat', 'lon'], how='left')
        df['soil_ph'] = merged.get('soil_ph', DEFAULT_SOIL_PH)
        df['soil_nitrogen'] = merged.get('soil_nitrogen', DEFAULT_SOIL_NITROGEN)
        df['soil_organic_carbon'] = merged.get('soil_organic_carbon', DEFAULT_SOIL_ORGANIC_CARBON)
    else:
        df['soil_ph'] = DEFAULT_SOIL_PH
        df['soil_nitrogen'] = DEFAULT_SOIL_NITROGEN
        df['soil_organic_carbon'] = DEFAULT_SOIL_ORGANIC_CARBON

    return df


# ---------------------------------------------------------------------------
# Economic features
# ---------------------------------------------------------------------------

def add_economic_features(df: pd.DataFrame, msp_growth_rate: float = 0.03) -> pd.DataFrame:
    """Add MSP trend and market viability proxy features."""
    base_year = 1990
    df['msp_trend'] = 1.0 + (df['Year'] - base_year) * msp_growth_rate
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

# Canonical order of features for model training
FEATURE_COLUMNS = [
    'Year', 'avg_temp', 'average_rain_fall_mm_per_year', 'pesticides_tonnes',
    'heat_stress_degreedays', 'drought_intensity',
    'ndvi', 'ndvi_adjusted',
    'soil_ph', 'soil_nitrogen', 'soil_organic_carbon',
    'msp_trend',
    'temp_rainfall_interaction', 'rainfall_deviation',
    'rainfall_squared', 'temp_squared', 'pesticide_per_rainfall',
    'year_normalized',
]


def engineer_features_v2(
    df: pd.DataFrame,
    soil_data: Optional[pd.DataFrame] = None,
    ndvi_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Full next-gen feature engineering pipeline.
    Returns DataFrame with standardized yield and all features.
    """
    # 1. Standardize yield units
    df = standardize_yield(df)

    # 2. Compute stress indices
    df = calculate_stress_indices(df)

    # 3. Add satellite and soil features
    df = add_satellite_soil_features(df, soil_data, ndvi_data)

    # 4. Add economic features
    df = add_economic_features(df)

    # 5. Apply interaction and year-based features
    df = calculate_interaction_features(df)
    df = add_year_based_features(df)

    return df


def get_feature_columns() -> List[str]:
    """Return the canonical list of feature column names."""
    return FEATURE_COLUMNS.copy()


if __name__ == "__main__":
    sample_df = pd.DataFrame({
        'Year': [2024, 2025, 2026],
        'avg_temp': [32.5, 36.1, 29.8],
        'average_rain_fall_mm_per_year': [1000, 450, 1200],
        'pesticides_tonnes': [5000, 5000, 5000],
        'hg/ha_yield': [25000, 22000, 27000],
    })

    print("Testing Unified Feature Engineering Pipeline (v2)...")
    result = engineer_features_v2(sample_df)
    print(f"\nShape: {result.shape}")
    print(f"Columns:\n{list(result.columns)}")
    print(f"\nSample output:\n{result.filter(['Year', 'yield_kg_ha', 'heat_stress_degreedays', 'ndvi_adjusted', 'msp_trend'])}")
