# tests/test_feature_engineer.py
"""Tests for feature_engineer module."""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.feature_engineer import (
    calculate_interaction_features,
    add_year_based_features,
    get_feature_names
)


def test_calculate_interaction_features():
    """Test interaction feature calculation."""
    df = pd.DataFrame({
        'avg_temp': [25.0, 30.0],
        'average_rain_fall_mm_per_year': [1000.0, 1500.0],
        'pesticides_tonnes': [100.0, 200.0]
    })
    result = calculate_interaction_features(df)

    # Check temp_rainfall_interaction
    assert result['temp_rainfall_interaction'].tolist() == [25000.0, 45000.0]

    # Check rainfall_squared
    assert result['rainfall_squared'].tolist() == [1000000.0, 2250000.0]

    # Check pesticide_per_rainfall (with +1 to avoid division by zero)
    expected_ratio = [100.0 / 1001.0, 200.0 / 1501.0]
    np.testing.assert_almost_equal(
        result['pesticide_per_rainfall'].tolist(),
        expected_ratio,
        decimal=5
    )


def test_add_year_based_features():
    """Test year normalization."""
    df = pd.DataFrame({
        'Year': [1990, 2000, 2010]
    })
    result = add_year_based_features(df)

    # Min year (1990) should be 0, max year (2010) should be 1
    assert result['year_normalized'].tolist() == [0.0, 0.5, 1.0]


def test_get_feature_names():
    """Test that feature names are returned correctly."""
    features = get_feature_names()
    assert isinstance(features, list)
    assert len(features) > 0
    assert 'temp_rainfall_interaction' in features


if __name__ == "__main__":
    test_calculate_interaction_features()
    test_add_year_based_features()
    test_get_feature_names()
    print("All tests passed!")
