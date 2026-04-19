# scripts/config.py
"""
Single Source of Truth (SSOT) configuration for the rebuilt Crop Yield Prediction system.
"""
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data paths (pointing to the newly created files)
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "india_crop_yield.csv")
FEATURES_DATA_PATH = os.path.join(DATA_DIR, "features", "features.csv") # User placed it here

# Model & Contract paths
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
CONTRACT_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

# Target variable definition
TARGET_COLUMN = "yield_kg_ha"

# Metadata for UI
TOP_15_CROPS = [
    "Rice", "Maize", "Wheat", "Cotton(lint)", "Sugarcane", "Gram", "Arhar/Tur",
    "Bajra", "Jowar", "Groundnut", "Urad", "Moong(Green Gram)", "Sesamum",
    "Sunflower", "Castor seed"
]

# Year range (based on typical Kaggle India Crop Yield dataset)
YEAR_MIN = 1997
YEAR_MAX = 2020

# Data sourcing references for transparency
APP_DATA_SOURCES = {
    "Primary Dataset": "https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states",
    "Open-Meteo": "https://open-meteo.com/",
    "FAOSTAT": "https://www.fao.org/faostat/en/#data/QC"
}
