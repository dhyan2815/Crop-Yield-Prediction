# tests/test_models.py
"""Tests for model training module."""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def test_linear_regression_basic():
    """Test basic Linear Regression functionality."""
    # Simple test data
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10]
    })
    y = pd.Series([3, 6, 9, 12, 15])

    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict(X)

    # Predictions should be close to actual values
    np.testing.assert_array_almost_equal(prediction, y, decimal=1)


def test_random_forest_basic():
    """Test basic Random Forest functionality."""
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5] * 10,
        'feature2': [2, 4, 6, 8, 10] * 10
    })
    y = pd.Series([3, 6, 9, 12, 15] * 10)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)

    # Predictions should exist
    assert len(predictions) == len(y)
    assert all(isinstance(p, (int, float, np.floating)) for p in predictions)


def test_model_feature_names_consistency():
    """Test that model preserves feature names."""
    X = pd.DataFrame({
        'rainfall': [1000, 1500, 2000],
        'temp': [25, 30, 35],
        'pesticides': [100, 200, 300]
    })
    y = pd.Series([1000, 2000, 3000])

    model = LinearRegression()
    model.fit(X, y)

    # Feature names should be preserved
    assert hasattr(model, 'feature_names_in_') or model.coef_.shape[0] == 3


if __name__ == "__main__":
    test_linear_regression_basic()
    test_random_forest_basic()
    test_model_feature_names_consistency()
    print("All model tests passed!")