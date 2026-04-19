"""Tests for feature engineering module consolidation."""

import pandas as pd
import pytest
from scripts.feature_engineer_v2 import engineer_features_v2, FEATURE_COLUMNS


def test_v2_pipeline_produces_all_features():
    """Engineered output should contain all expected v2 features."""
    sample = pd.DataFrame({
        'Year': [2020, 2021],
        'avg_temp': [30.0, 31.5],
        'average_rain_fall_mm_per_year': [800, 950],
        'pesticides_tonnes': [5000, 5200],
        'hg/ha_yield': [20000, 21000]
    })
    result = engineer_features_v2(sample)

    # Check standardized yield
    assert 'yield_kg_ha' in result.columns
    # Check all v2 features present
    for col in FEATURE_COLUMNS:
        assert col in result.columns, f"Missing feature: {col}"
    # Check no legacy yield columns remain
    assert 'hg/ha_yield' not in result.columns
    assert 'kg_per_ha_yield' not in result.columns


def test_deprecated_wrapper_warns():
    """Deprecated feature_engineer module removed - test marked as skip."""
    # The old feature_engineer module was removed in v2 consolidation.
    # This test is kept as a placeholder for historical context.
    pytest.skip("Deprecated module feature_engineer removed - no longer applicable")


def test_feature_order_matches_config():
    """V2 feature columns should be in the expected order."""
    expected = [
        'Year', 'avg_temp', 'average_rain_fall_mm_per_year', 'pesticides_tonnes',
        'heat_stress_degreedays', 'drought_intensity',
        'ndvi', 'ndvi_adjusted',
        'soil_ph', 'soil_nitrogen', 'soil_organic_carbon',
        'msp_trend',
        'temp_rainfall_interaction', 'rainfall_deviation',
        'rainfall_squared', 'temp_squared', 'pesticide_per_rainfall',
        'year_normalized',
    ]
    assert FEATURE_COLUMNS == expected
