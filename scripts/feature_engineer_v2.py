"""
Advanced Feature Engineering Module (v2) for Project 2026.
Includes Fallback Mechanisms for External API Outages.
"""

import pandas as pd
import numpy as np

# Default historical averages for professional features (India)
DEFAULT_SOIL_PH = 6.5
DEFAULT_SOIL_NITROGEN = 150 # mg/kg
DEFAULT_NDVI = 0.65 # Healthy crop range

def calculate_professional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply advanced features even with dummy data if APIs fail.
    """
    # 1. Heat Stress (Simplified Index: Temp > 35C is stress)
    df['heat_stress_index'] = df['avg_temp'].apply(lambda x: max(0, x - 35))
    
    # 2. Drought Index (Rainfall < 500mm/year is high stress)
    df['drought_intensity'] = df['average_rain_fall_mm_per_year'].apply(lambda x: 1.0 if x < 500 else 0.0)
    
    # 3. Satellite Proxy (NDVI)
    # NDVI is typically seasonal. For now, we seed it based on rainfall & temp correlation.
    df['ndvi_proxy'] = DEFAULT_NDVI + (df['average_rain_fall_mm_per_year'] / 2000.0) - (df['heat_stress_index'] / 10.0)
    df['ndvi_proxy'] = df['ndvi_proxy'].clip(0, 1) # Must be 0-1
    
    # 4. Soil Health (NPK Proxy)
    # pH is relatively stable for locations. We assign a baseline.
    df['soil_ph'] = DEFAULT_SOIL_PH
    df['soil_nitrogen'] = DEFAULT_SOIL_NITROGEN
    
    # 5. Economic Viability (MSP Trend)
    # Assumes a 3% growth in price support annually
    base_year = 1990
    df['msp_trend'] = 1.0 + (df['Year'] - base_year) * 0.03
    
    return df

def engineer_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Full next-gen feature engineering pipeline."""
    # Start with base features
    from scripts.feature_engineer import calculate_interaction_features, add_year_based_features
    
    # 1. Apply existing engineered features (1.0)
    df = calculate_interaction_features(df)
    df = add_year_based_features(df)
    
    # 2. Apply advanced 2026 features (v2)
    df = calculate_professional_features(df)
    
    return df

if __name__ == "__main__":
    # Create sample data to test
    sample_df = pd.DataFrame({
        'Year': [2024, 2025, 2026],
        'avg_temp': [32.5, 36.1, 29.8],
        'average_rain_fall_mm_per_year': [1000, 450, 1200],
        'pesticides_tonnes': [5000, 5000, 5000]
    })
    
    print("Testing Next-Gen Feature Pipeline...")
    results = engineer_features_v2(sample_df)
    print(results[['Year', 'heat_stress_index', 'ndvi_proxy', 'msp_trend']])
