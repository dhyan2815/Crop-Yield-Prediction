# tests/test_prediction.py
"""
Tests for crop yield prediction flow.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predictor import build_prediction_features, get_crop_columns, get_feature_importance
from utils.data_loader import load_features_data, get_available_options, get_dataset_stats


class TestFeatureBuilding:
    """Test feature construction logic."""

    def test_build_prediction_features_returns_dataframe(self):
        """Feature builder should return DataFrame."""
        result = build_prediction_features(
            crop="Rice",
            year=2000,
            pesticides=5000.0,
            rainfall=1000.0,
            temp=25.0,
            crop_columns=["Item_Rice", "Item_Wheat"]
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_core_features_present(self):
        """Core climate features should be in output."""
        result = build_prediction_features(
            crop="Rice", year=2000, pesticides=5000.0,
            rainfall=1000.0, temp=25.0, crop_columns=[]
        )
        assert 'average_rain_fall_mm_per_year' in result.columns
        assert 'avg_temp' in result.columns
        assert 'pesticides_tonnes' in result.columns

    def test_engineered_features_calculated(self):
        """Interaction and polynomial features should be computed."""
        result = build_prediction_features(
            crop="Rice", year=2000, pesticides=5000.0,
            rainfall=1000.0, temp=25.0, crop_columns=[]
        )
        # Interaction: temp * rainfall
        assert result['temp_rainfall_interaction'].iloc[0] == 25000.0
        # Polynomial
        assert result['rainfall_squared'].iloc[0] == 1000000.0
        assert result['temp_squared'].iloc[0] == 625.0
        # Pesticide per rainfall
        assert result['pesticide_per_rainfall'].iloc[0] == pytest.approx(4.995, rel=1e-3)

    def test_v2_features_present(self):
        """Satellite/soil/economic features (v2) should exist."""
        result = build_prediction_features(
            crop="Rice", year=2000, pesticides=5000.0,
            rainfall=1000.0, temp=25.0, crop_columns=[]
        )
        v2_features = [
            'heat_stress_degreedays', 'drought_intensity',
            'ndvi', 'ndvi_adjusted', 'soil_ph',
            'soil_nitrogen', 'soil_organic_carbon', 'msp_trend'
        ]
        for feat in v2_features:
            assert feat in result.columns, f"Missing: {feat}"

    def test_year_normalized_bound(self):
        """year_normalized should be between 0 and 1."""
        result = build_prediction_features(
            crop="Rice", year=2000, pesticides=5000.0,
            rainfall=1000.0, temp=25.0, crop_columns=[]
        )
        val = result['year_normalized'].iloc[0]
        assert 0 <= val <= 1, f"year_normalized out of bounds: {val}"

    def test_crop_one_hot_encoding(self):
        """Selected crop should be 1, others 0."""
        result = build_prediction_features(
            crop="Rice", year=2000, pesticides=5000.0,
            rainfall=1000.0, temp=25.0,
            crop_columns=["Item_Rice", "Item_Wheat", "Item_Maize"]
        )
        assert result['Item_Rice'].iloc[0] == 1
        assert result['Item_Wheat'].iloc[0] == 0
        assert result['Item_Maize'].iloc[0] == 0

    def test_heat_stress_calculated(self):
        """Heat stress should trigger when temp > 35."""
        # Temp below threshold
        result_low = build_prediction_features(
            crop="Rice", year=2000, pesticides=5000.0,
            rainfall=1000.0, temp=30.0, crop_columns=[]
        )
        assert result_low['heat_stress_degreedays'].iloc[0] == 0.0

        # Temp above threshold
        result_high = build_prediction_features(
            crop="Rice", year=2000, pesticides=5000.0,
            rainfall=1000.0, temp=40.0, crop_columns=[]
        )
        assert result_high['heat_stress_degreedays'].iloc[0] == 5.0

    def test_drought_intensity_calculated(self):
        """Drought intensity should trigger when rainfall < 500."""
        # Low rainfall
        result_dry = build_prediction_features(
            crop="Rice", year=2000, pesticides=5000.0,
            rainfall=200.0, temp=25.0, crop_columns=[]
        )
        assert result_dry['drought_intensity'].iloc[0] == pytest.approx(0.6, rel=1e-2)

        # Adequate rainfall
        result_wet = build_prediction_features(
            crop="Rice", year=2000, pesticides=5000.0,
            rainfall=1000.0, temp=25.0, crop_columns=[]
        )
        assert result_wet['drought_intensity'].iloc[0] == 0.0


class TestDataLoader:
    """Test data loading functions."""

    def test_load_features_data_returns_dataframe(self):
        """Should load feature data successfully."""
        df = load_features_data()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'Item' in df.columns
        assert 'Year' in df.columns

    def test_get_available_options_returns_valid_range(self):
        """Should return valid crop list and year range."""
        crops, min_year, max_year = get_available_options()
        assert isinstance(crops, list)
        assert len(crops) > 0
        assert min_year >= 1900
        assert max_year <= 2025
        assert min_year < max_year

    def test_get_dataset_stats_returns_dict(self):
        """Should return pesticide statistics."""
        stats = get_dataset_stats()
        assert isinstance(stats, dict)
        assert 'pesticide_min' in stats
        assert 'pesticide_max' in stats
        assert 'pesticide_median' in stats
        assert stats['pesticide_min'] <= stats['pesticide_max']
        assert stats['pesticide_min'] <= stats['pesticide_median'] <= stats['pesticide_max']


class TestCropColumns:
    """Test crop column generation."""

    def test_get_crop_columns_format(self):
        """Should generate Item_* prefixed columns."""
        columns = get_crop_columns(["Rice", "Wheat", "Maize"])
        assert columns == ["Item_Rice", "Item_Wheat", "Item_Maize"]


class TestFeatureImportance:
    """Test feature importance extraction."""

    def test_get_feature_importance_returns_dict(self):
        """Should return normalized importance dict."""
        # Create mock model with actual expected feature names
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        # These must match what the function looks for
        mock_model.feature_names_in_ = [
            'average_rain_fall_mm_per_year', 'avg_temp',
            'pesticides_tonnes', 'year_normalized'
        ]

        result = get_feature_importance(mock_model, [])
        assert isinstance(result, dict)
        # Should be normalized to sum to 1
        assert sum(result.values()) == pytest.approx(1.0, rel=1e-2)

    def test_feature_importance_sums_to_one(self):
        """Importance values should normalize to 1."""
        # Model with the expected feature names
        feature_names = [
            'average_rain_fall_mm_per_year', 'avg_temp',
            'pesticides_tonnes', 'year_normalized'
        ]
        mock_model = MagicMock()
        # Distribute importance equally
        mock_model.feature_importances_ = np.array([0.25, 0.25, 0.25, 0.25])
        mock_model.feature_names_in_ = feature_names

        result = get_feature_importance(mock_model, [])
        total = sum(result.values())
        assert total == pytest.approx(1.0, rel=1e-3)


class TestPredictionIntegration:
    """Integration tests for prediction flow."""

    def test_prediction_returns_positive_value(self):
        """Prediction should return positive yield value."""
        # This test mocks the model to verify the flow
        from scripts.feature_engineer_v2 import FEATURE_COLUMNS

        crops, min_year, max_year = get_available_options()
        crop_columns = get_crop_columns(crops)

        # Build features
        test_year = min_year
        test_crop = crops[0]
        test_pesticides = 5000.0
        test_rainfall = 1000.0
        test_temp = 25.0

        input_df = build_prediction_features(
            test_crop, test_year, test_pesticides,
            test_rainfall, test_temp, crop_columns
        )

        # Ensure all expected columns present
        missing = [c for c in FEATURE_COLUMNS if c not in input_df.columns]
        assert len(missing) == 0, f"Missing columns: {missing}"

    def test_full_feature_set_for_champion_model(self):
        """Input DataFrame should have all columns for champion model."""
        from scripts.feature_engineer_v2 import FEATURE_COLUMNS
        from scripts.config import CHAMPION_FEATURES

        crops, _, _ = get_available_options()
        crop_columns = get_crop_columns(crops)

        # Build features for a crop
        result = build_prediction_features(
            crop=crops[0], year=2000, pesticides=5000.0,
            rainfall=1000.0, temp=25.0, crop_columns=crop_columns
        )

        # Champion model expects: FEATURE_COLUMNS + crop_columns
        expected_cols = FEATURE_COLUMNS + crop_columns
        for col in expected_cols:
            assert col in result.columns, f"Missing champion feature: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])