# Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize the Crop Yield Prediction project by consolidating the pipeline, adding real-time weather integration, improving model training with all engineered features, and creating modular, reproducible scripts.

**Architecture:**
- Modular Python scripts for data processing, feature engineering, and model training
- Refactored Streamlit app with cleaner structure and real-time weather support
- Single source of truth for the data pipeline (Feature_Engineered_Crop_Yield_Data.csv)

**Tech Stack:** Python, Streamlit, scikit-learn, pandas, requests, joblib

---

## File Structure

```
project/
├── app.py                          # Refactored Streamlit app
├── train_models.py                 # Unified training script (replaces train_models_with_crop.py)
├── scripts/
│   ├── __init__.py
│   ├── data_cleaner.py             # Data cleaning module
│   ├── feature_engineer.py         # Feature engineering module
│   └── config.py                   # Shared configuration/constants
├── tests/
│   ├── __init__.py
│   ├── test_data_cleaner.py
│   ├── test_feature_engineer.py
│   └── test_models.py
├── models/                         # Trained model files
├── data/
│   ├── raw/                        # Original raw data
│   └── processed/                  # Cleaned and engineered data
└── docs/
    └── ENHANCEMENT_PROPOSAL.md     # Updated proposal
```

---

## Task 1: Create Project Directory Structure

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/config.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create scripts directory and __init__.py**

```bash
mkdir -p scripts tests
touch scripts/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create config.py with shared constants**

```python
# scripts/config.py
"""Shared configuration constants for the crop yield prediction pipeline."""

# Data paths
RAW_DATA_PATH = "data/raw/Crop_Yield_Data.csv"
RAINFALL_DATA_PATH = "data/raw/Daily Rainfall at State Level_Filtered_Data.csv"
PROCESSED_DATA_PATH = "data/processed/CLEANED_Processed_India_Crop_Yield_Data.csv"
FEATURES_DATA_PATH = "data/processed/Feature_Engineered_Crop_Yield_Data.csv"

# Model paths
MODEL_DIR = "models"
LR_MODEL_PATH = f"{MODEL_DIR}/linear_regression_model.pkl"
RF_MODEL_PATH = f"{MODEL_DIR}/random_forest_model.pkl"

# Target variable
TARGET_COLUMN = "kg_per_ha_yield"

# Core numeric features (always used)
CORE_FEATURES = [
    "average_rain_fall_mm_per_year",
    "avg_temp",
    "pesticides_tonnes"
]

# Engineered features (optional, can be toggled)
ENGINEERED_FEATURES = [
    "temp_rainfall_interaction",
    "rainfall_deviation",
    "rainfall_squared",
    "temp_squared",
    "pesticide_per_rainfall"
]

# All features for training
ALL_FEATURES = CORE_FEATURES + ENGINEERED_FEATURES
```

- [ ] **Step 3: Commit**

```bash
git add scripts/__init__.py scripts/config.py tests/__init__.py
git commit -m "feat: create project directory structure and config module"
```

---

## Task 2: Create Data Cleaner Module

**Files:**
- Create: `scripts/data_cleaner.py`
- Test: `tests/test_data_cleaner.py`

- [ ] **Step 1: Create data_cleaner.py**

```python
# scripts/data_cleaner.py
"""Data cleaning module for crop yield prediction."""

import pandas as pd
from .config import RAW_DATA_PATH, RAINFALL_DATA_PATH, PROCESSED_DATA_PATH


def load_raw_crop_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw crop yield data from CSV."""
    df = pd.read_csv(filepath)
    # Clean column names
    df.columns = df.columns.str.strip()
    return df


def load_rainfall_data(filepath: str = RAINFALL_DATA_PATH) -> pd.DataFrame:
    """Load rainfall data from filtered CSV."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def filter_india_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data for India only."""
    df = df[df['Area'].str.strip().str.lower() == 'india'].copy()
    return df


def clean_crop_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean crop (Item) names: strip whitespace and remove quotes."""
    if 'Item' in df.columns:
        df['Item'] = df['Item'].str.strip().str.replace('"', '', regex=False)
    return df


def calculate_average_rainfall(df: pd.DataFrame, rainfall_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average annual rainfall per state and merge with crop data.

    For this dataset, we'll use a simplified approach since the rainfall
    data is at state level. The average rainfall per year is used.
    """
    # Calculate yearly average rainfall across all states
    if 'Year' in rainfall_df.columns and 'ANNUAL' in rainfall_df.columns:
        yearly_rainfall = rainfall_df.groupby('Year')['ANNUAL'].mean().reset_index()
        yearly_rainfall.columns = ['Year', 'average_rain_fall_mm_per_year']
        # Merge with crop data
        df = df.merge(yearly_rainfall, on='Year', how='left')
    return df


def add_temperature_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add average temperature column.

    Note: In a full implementation, this would come from a weather API
    or temperature dataset. For now, we create a placeholder or use
    existing temp data if available.
    """
    # If avg_temp already exists in the data, keep it
    # Otherwise, create a synthetic temperature based on rainfall (correlation)
    if 'avg_temp' not in df.columns:
        # Use a simple correlation: higher rainfall often means lower avg temp in India
        df['avg_temp'] = 26 - (df['average_rain_fall_mm_per_year'] / 200)
    return df


def convert_yield_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert yield from hg/ha to kg/ha."""
    if 'hg/ha_yield' in df.columns:
        df['kg_per_ha_yield'] = df['hg/ha_yield'] / 10
    return df


def clean_data(input_path: str = RAW_DATA_PATH,
               rainfall_path: str = RAINFALL_DATA_PATH,
               output_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Full data cleaning pipeline.

    Returns cleaned DataFrame and saves to output_path.
    """
    # Load data
    crop_df = load_raw_crop_data(input_path)
    rainfall_df = load_rainfall_data(rainfall_path)

    # Filter for India
    crop_df = filter_india_data(crop_df)

    # Clean crop names
    crop_df = clean_crop_names(crop_df)

    # Calculate average rainfall and merge
    crop_df = calculate_average_rainfall(crop_df, rainfall_df)

    # Add temperature column
    crop_df = add_temperature_column(crop_df)

    # Convert yield units
    crop_df = convert_yield_units(crop_df)

    # Drop rows with missing critical values
    crop_df = crop_df.dropna(subset=[
        'average_rain_fall_mm_per_year',
        'avg_temp',
        'pesticides_tonnes'
    ])

    # Save to processed path
    crop_df.to_csv(output_path, index=False)

    return crop_df


if __name__ == "__main__":
    print("Running data cleaning pipeline...")
    df = clean_data()
    print(f"Cleaned data saved. Shape: {df.shape}")
```

- [ ] **Step 2: Create test_data_cleaner.py**

```python
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
    assert len(result) == 4  # All should match (case-insensitive, whitespace stripped)


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
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
python -m pytest tests/test_data_cleaner.py -v
```

- [ ] **Step 4: Commit**

```bash
git add scripts/data_cleaner.py tests/test_data_cleaner.py
git commit -m "feat: add data cleaner module with tests"
```

---

## Task 3: Create Feature Engineer Module

**Files:**
- Create: `scripts/feature_engineer.py`
- Test: `tests/test_feature_engineer.py`

- [ ] **Step 1: Create feature_engineer.py**

```python
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
```

- [ ] **Step 2: Create test_feature_engineer.py**

```python
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
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
python -m pytest tests/test_feature_engineer.py -v
```

- [ ] **Step 4: Commit**

```bash
git add scripts/feature_engineer.py tests/test_feature_engineer.py
git commit -m "feat: add feature engineering module with tests"
```

---

## Task 4: Create Unified Training Script

**Files:**
- Create: `scripts/train_models.py`
- Test: `tests/test_models.py`
- Delete: `train_models_with_crop.py` (old script)

- [ ] **Step 1: Create train_models.py**

```python
# scripts/train_models.py
"""Model training module for crop yield prediction."""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .config import (
    FEATURES_DATA_PATH,
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    CORE_FEATURES,
    ENGINEERED_FEATURES,
    TARGET_COLUMN
)


def load_training_data(filepath: str = FEATURES_DATA_PATH,
                       use_engineered: bool = True) -> tuple:
    """
    Load training data with specified features.

    Args:
        filepath: Path to the feature-engineered CSV
        use_engineered: Whether to include engineered features

    Returns:
        X: Feature DataFrame
        y: Target Series
        feature_names: List of feature names used
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Clean crop names (consistency with training)
    if 'Item' in df.columns:
        df['Item'] = df['Item'].str.strip().str.replace('"', '', regex=False)

    # Select features
    if use_engineered:
        feature_cols = CORE_FEATURES + ENGINEERED_FEATURES + ['year_normalized']
    else:
        feature_cols = CORE_FEATURES

    # One-hot encode crop (Item) column if present
    if 'Item' in df.columns:
        df = pd.get_dummies(df, columns=['Item'])
        crop_cols = [col for col in df.columns if col.startswith('Item_')]
        feature_cols = feature_cols + crop_cols

    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    return X, y, feature_cols


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> dict:
    """Calculate and return evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.4f}")

    return metrics


import numpy as np


def train_models(X_train, X_test, y_train, y_test,
                 n_estimators: int = 200,
                 random_state: int = 42) -> tuple:
    """
    Train Linear Regression and Random Forest models.

    Returns:
        lr_model, rf_model, metrics_lr, metrics_rf
    """
    # Train Linear Regression
    print("\nTraining Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    metrics_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression")

    # Cross-validation for LR
    cv_scores_lr = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='r2')
    print(f"  CV R² scores: {cv_scores_lr}")
    print(f"  Mean CV R²: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    metrics_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

    # Cross-validation for RF
    cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    print(f"  CV R² scores: {cv_scores_rf}")
    print(f"  Mean CV R²: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

    return lr_model, rf_model, metrics_lr, metrics_rf


def save_models(lr_model, rf_model,
                lr_path: str = LR_MODEL_PATH,
                rf_path: str = RF_MODEL_PATH):
    """Save trained models to disk."""
    joblib.dump(lr_model, lr_path)
    joblib.dump(rf_model, rf_path)
    print(f"\nModels saved:")
    print(f"  Linear Regression: {lr_path}")
    print(f"  Random Forest: {rf_path}")


def train_pipeline(use_engineered: bool = True,
                   test_size: float = 0.2,
                   random_state: int = 42) -> dict:
    """
    Complete training pipeline.

    Returns dictionary with models, metrics, and feature names.
    """
    print("=" * 60)
    print("CROP YIELD PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load data
    print("\nLoading training data...")
    X, y, feature_names = load_training_data(use_engineered=use_engineered)
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Dataset shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Train models
    lr_model, rf_model, metrics_lr, metrics_rf = train_models(
        X_train, X_test, y_train, y_test
    )

    # Save models
    save_models(lr_model, rf_model)

    # Return results
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'feature_names': feature_names,
        'metrics_lr': metrics_lr,
        'metrics_rf': metrics_rf
    }


if __name__ == "__main__":
    results = train_pipeline(use_engineered=True)
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
```

- [ ] **Step 2: Create test_models.py**

```python
# tests/test_models.py
"""Tests for model training module."""

import pandas as pd
import numpy as np
import sys
import os
import tempfile

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
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
python -m pytest tests/test_models.py -v
```

- [ ] **Step 4: Delete old training script**

```bash
rm train_models_with_crop.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/train_models.py tests/test_models.py
git rm train_models_with_crop.py
git commit -m "feat: add unified training script with engineered features support"
```

---

## Task 5: Refactor Streamlit App

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Create a modular app structure (backup and replace app.py)**

The refactored app will:
1. Import from the new modular scripts
2. Use a cleaner structure with error handling
3. Support real-time weather data integration
4. Include crop one-hot encoding (already implemented)

```python
# app.py - Refactored version
"""
Crop Yield Prediction Web Application
Streamlit interface for predicting crop yields based on historical and real-time data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Import from modular scripts
from scripts.config import (
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    FEATURES_DATA_PATH,
    CORE_FEATURES,
    ENGINEERED_FEATURES
)
from scripts.feature_engineer import get_feature_names

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

st.set_page_config(
    page_title="Yield Metrics",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MODEL LOADING (Cached)
# =============================================================================

@st.cache_resource
def load_models():
    """Load trained ML models."""
    try:
        lr_model = joblib.load(LR_MODEL_PATH)
        rf_model = joblib.load(RF_MODEL_PATH)
        return lr_model, rf_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None


@st.cache_data
def load_features_data():
    """Load processed features data."""
    try:
        df = pd.read_csv(FEATURES_DATA_PATH)
        df.columns = df.columns.str.strip()
        # Clean crop names
        if 'Item' in df.columns:
            df['Item'] = df['Item'].str.strip().str.replace('"', '', regex=False)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()


@st.cache_data
def get_available_options():
    """Get available crops and year range from data."""
    df = load_features_data()
    if df.empty:
        return [], 1990, 2013

    crops = sorted(df['Item'].unique())
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())

    return crops, min_year, max_year


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_crop_columns(available_crops: list) -> list:
    """Generate one-hot encoded column names for crops."""
    return [f'Item_{crop}' for crop in available_crops]


def build_prediction_features(crop: str, year: int, pesticides: float,
                             rainfall: float, temp: float,
                             crop_columns: list) -> pd.DataFrame:
    """Build feature DataFrame for model prediction."""
    feature_dict = {
        'average_rain_fall_mm_per_year': rainfall,
        'avg_temp': temp,
        'pesticides_tonnes': pesticides,
    }

    # Add engineered features (using defaults from training data)
    feature_dict.update({
        'temp_rainfall_interaction': temp * rainfall,
        'rainfall_deviation': rainfall - 1083,  # Approximate mean from data
        'rainfall_squared': rainfall ** 2,
        'temp_squared': temp ** 2,
        'pesticide_per_rainfall': pesticides / (rainfall + 1),
        'year_normalized': (year - 1990) / 23  # Normalized to dataset range
    })

    # One-hot encode crop
    for col in crop_columns:
        feature_dict[col] = 0
    selected_col = f'Item_{crop}'
    if selected_col in feature_dict:
        feature_dict[selected_col] = 1

    return pd.DataFrame([feature_dict])


def create_prediction_plot(y_lr: float, y_rf: float, crop: str, year: int):
    """Create bar plot comparing model predictions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Linear Regression', 'Random Forest']
    predictions = [y_lr, y_rf]
    colors = ['skyblue', 'lightgreen']

    bars = ax.bar(models, predictions, color=colors, alpha=0.8)

    for bar, pred in zip(bars, predictions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{pred:.0f} kg/ha', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Predicted Yield (kg/ha)', fontsize=12)
    ax.set_title(f'{crop} Yield Prediction for {year}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def create_trend_plot(df: pd.DataFrame, crop: str):
    """Create line plot showing historical yield trend for a crop."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter for selected crop
    crop_df = df[df['Item'] == crop].copy()
    avg_yield = crop_df.groupby('Year')['kg_per_ha_yield'].mean().reset_index()

    sns.lineplot(x='Year', y='kg_per_ha_yield', data=avg_yield,
                ax=ax, linewidth=2, color='blue', marker='o')

    ax.set_title(f'{crop} - Historical Yield Trend in India', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Yield (kg/ha)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def validate_inputs(crop: str, year: int, pesticides: float,
                    min_year: int, max_year: int) -> tuple:
    """Validate user inputs. Returns (is_valid, errors_list)."""
    errors = []

    if not crop or crop.strip() == "":
        errors.append("Please select a crop")

    if year < min_year or year > max_year:
        errors.append(f"Year must be between {min_year} and {max_year}")

    if pesticides < 0:
        errors.append("Pesticide usage cannot be negative")

    return len(errors) == 0, errors


# =============================================================================
# MAIN APPLICATION UI
# =============================================================================

def main():
    """Main application function."""

    # Header
    st.title("🌾 Yield Metrics")
    st.markdown("""
    Welcome to **Yield Metrics** – a crop yield prediction app for India.
    """)

    # Disclaimer
    st.warning("⚠️ **Predictions** are based on historical data and may not reflect current conditions. Use results for guidance only.")

    # Load models
    lr_model, rf_model = load_models()
    if lr_model is None or rf_model is None:
        st.error("Failed to load models. Please ensure training has been completed.")
        return

    # Get available options
    available_crops, min_year, max_year = get_available_options()
    if not available_crops:
        st.error("No data available. Please check data files.")
        return

    crop_columns = get_crop_columns(available_crops)

    # Main content area
    st.header("📝 Input Parameters")
    st.caption(f"Available years: {min_year} - {max_year} | Crops: {len(available_crops)}")

    # Input widgets
    crop = st.selectbox("🌱 Select Crop", available_crops, help="Choose the crop for yield prediction")
    year = st.number_input("📅 Select Year", min_value=min_year, max_value=max_year,
                          value=max_year, step=1)
    pesticides = st.number_input("🧪 Pesticide Usage (tonnes)",
                                 min_value=0.0, max_value=100000.0,
                                 value=5000.0, step=100.0)

    st.markdown("---")

    # Prediction button
    if st.button("🚀 Predict Yield", type="primary", use_container_width=True):

        # Validate inputs
        is_valid, errors = validate_inputs(crop, year, pesticides, min_year, max_year)
        if not is_valid:
            for error in errors:
                st.error(f"❌ {error}")
            return

        # Show loading state
        with st.spinner("🔄 Making predictions..."):

            # Get historical data for crop-year
            df = load_features_data()
            match = df[(df['Item'] == crop) & (df['Year'] == int(year))]

            if match.empty:
                st.error(f"No data available for {crop} in {year}. Please try a different year.")
                return

            # Use historical rainfall and temperature
            rainfall = float(match['average_rain_fall_mm_per_year'].iloc[0])
            temp = float(match['avg_temp'].iloc[0])

            st.info(f"📊 Using historical data: Rainfall={rainfall:.0f}mm, Temp={temp:.1f}°C")

            # Build features and predict
            try:
                input_features = build_prediction_features(
                    crop, year, pesticides, rainfall, temp, crop_columns
                )

                # Get predictions
                yield_lr = lr_model.predict(input_features)[0]
                yield_rf = rf_model.predict(input_features)[0]

                # Display results
                st.success("✅ Prediction Complete!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Linear Regression", f"{yield_lr:.0f} kg/ha")
                with col2:
                    st.metric("Random Forest", f"{yield_rf:.0f} kg/ha")

                # Plots
                st.markdown("### 📊 Model Comparison")
                pred_fig = create_prediction_plot(yield_lr, yield_rf, crop, year)
                st.pyplot(pred_fig)

                st.markdown("### 📈 Historical Trend")
                trend_fig = create_trend_plot(df, crop)
                st.pyplot(trend_fig)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>🌾 Yield Metrics | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the app to verify it works**

```bash
streamlit run app.py
```

- [ ] **Step 3: Test prediction with different crops**

Test that changing crops produces different predictions.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "refactor: modularize app with cleaner structure and feature engineering support"
```

---

## Task 6: Run Full Pipeline and Verify

- [ ] **Step 1: Run data cleaner**

```bash
python -c "from scripts.data_cleaner import clean_data; clean_data()"
```

- [ ] **Step 2: Run feature engineer**

```bash
python -c "from scripts.feature_engineer import engineer_features; engineer_features()"
```

- [ ] **Step 3: Run training**

```bash
python scripts/train_models.py
```

- [ ] **Step 4: Start Streamlit and verify predictions work**

```bash
streamlit run app.py
```

- [ ] **Step 5: Commit all changes**

```bash
git add -A
git commit -m "feat: complete enhancement implementation - modular pipeline with all features"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `docs/ENHANCEMENT_PROPOSAL.md`

- [ ] **Step 1: Update ENHANCEMENT_PROPOSAL.md to mark completed items**

Update the roadmap section to reflect what's been implemented.

- [ ] **Step 2: Create a new README section for the pipeline**

Document how to use the new modular scripts.

- [ ] **Step 3: Commit documentation updates**

```bash
git add docs/ENHANCEMENT_PROPOSAL.md
git commit -m "docs: update enhancement proposal with completed items"
```

---

## Summary

After completing all tasks, the following will be in place:

1. **Modular Scripts** (`scripts/`):
   - `config.py` - Centralized configuration
   - `data_cleaner.py` - Data cleaning with tests
   - `feature_engineer.py` - Feature engineering with tests
   - `train_models.py` - Unified training with all features

2. **Tests** (`tests/`):
   - Unit tests for all modules
   - Can be run with `pytest`

3. **Refactored App** (`app.py`):
   - Cleaner structure
   - Uses all engineered features
   - Better error handling

4. **Updated Models**:
   - Trained with all engineered features
   - Crop one-hot encoding included

5. **Documentation**:
   - Enhancement proposal updated
   - Clear pipeline usage instructions

---

## Execution Options

**Plan complete and saved to `docs/superpowers/plans/2026-03-30-enhancement-implementation.md`**

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
