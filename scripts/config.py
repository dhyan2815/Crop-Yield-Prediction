"""
Single Source of Truth (SSOT) configuration for the Crop Yield Prediction system.

This module centralizes directory locations, file paths, model contract constants,
and UI configuration settings across scripts and the web application.
"""

import os

# Resolve the repository root directory (one level up from scripts/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Key subdirectories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data file paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "india_crop_yield.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "processed", "cleaned.csv")
FEATURES_DATA_PATH = os.path.join(DATA_DIR, "features", "features.csv")

# Model artifact and feature contract paths
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
CONTRACT_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

# Core dataset specifications
TARGET_COLUMN = "yield_kg_ha"

# Temporal boundaries for year normalization and UI sliders
YEAR_MIN = 2000
YEAR_MAX = 2026

# Dataset provenance links displayed in the application UI
APP_DATA_SOURCES = {
    "Primary Dataset": "https://huggingface.co/datasets/dhyann2815/india-crop-yield-prediction"
}
