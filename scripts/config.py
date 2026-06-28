# scripts/config.py
"""
Single Source of Truth (SSOT) configuration for the rebuilt Crop Yield Prediction system.
"""
import os

# Resolve the repository root once so every path stays consistent.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Raw and processed data locations used by notebooks, scripts, and the app.
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "india_crop_yield.csv")
FEATURES_DATA_PATH = os.path.join(DATA_DIR, "features", "features.csv") # User placed it here

# Model artifact and feature contract paths for contract-based inference.
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
CONTRACT_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

# Canonical target column name used across cleaning, training, and prediction.
TARGET_COLUMN = "yield_kg_ha"

# Year bounds used by the UI and normalization logic.
YEAR_MIN = 2000
YEAR_MAX = 2026

# Public data source links surfaced in the UI footer.
APP_DATA_SOURCES = {
    "Primary Dataset": "https://huggingface.co/datasets/dhyann2815/india-crop-yield-prediction"
}
