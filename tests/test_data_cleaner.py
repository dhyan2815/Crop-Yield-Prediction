# tests/test_data_cleaner.py
"""Tests for data_cleaner module."""

import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_cleaner import (
    clean_crop_names,
    filter_india_data,
    convert_yield_units
)


def test_clean_crop_names():
    """Test that crop names are cleaned properly."""
    df = pd.DataFrame({
        'Item': ['  Cassava  ', '"Rice Paddy"', 'Maize            ']
    })
    result = clean_crop_names(df)
    assert result['Item'].tolist() == ['Cassava', 'Rice Paddy', 'Maize']


def test_filter_india_data():
    """Test filtering for India data only."""
    df = pd.DataFrame({
        'Area': ['India', 'USA', 'india', 'India  ']
    })
    result = filter_india_data(df)
    # Only 'India', 'india', 'India  ' match (not 'USA')
    assert len(result) == 3
    assert result['Area'].str.lower().str.strip().tolist() == ['india', 'india', 'india']


def test_convert_yield_units():
    """Test conversion from hg/ha to kg/ha."""
    df = pd.DataFrame({
        'hg/ha_yield': [1000, 2000, 3000]
    })
    result = convert_yield_units(df)
    assert result['kg_per_ha_yield'].tolist() == [100.0, 200.0, 300.0]


def test_clean_crop_names_preserves_other_columns():
    """Test that other columns are not affected by cleaning."""
    df = pd.DataFrame({
        'Item': ['  Wheat  ', 'Rice'],
        'Year': [2020, 2021],
        'Yield': [100, 200]
    })
    result = clean_crop_names(df)
    assert result['Year'].tolist() == [2020, 2021]
    assert result['Yield'].tolist() == [100, 200]


if __name__ == "__main__":
    test_clean_crop_names()
    test_filter_india_data()
    test_convert_yield_units()
    test_clean_crop_names_preserves_other_columns()
    print("All tests passed!")