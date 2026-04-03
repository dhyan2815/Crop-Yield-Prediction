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
CHAMPION_MODEL_PATH = f"{MODEL_DIR}/champion_model_v2.pkl"

# Target variable
TARGET_COLUMN = "yield_kg_ha"  # standardized to kg/ha

# Core numeric features (always used)
CORE_FEATURES = [
    "average_rain_fall_mm_per_year",
    "avg_temp",
    "pesticides_tonnes"
]

# Engineered features (v1, legacy)
ENGINEERED_FEATURES = [
    "temp_rainfall_interaction",
    "rainfall_deviation",
    "rainfall_squared",
    "temp_squared",
    "pesticide_per_rainfall"
]

# v2 feature additions
V2_FEATURES = [
    "heat_stress_degreedays",
    "drought_intensity",
    "ndvi",
    "ndvi_adjusted",
    "soil_ph",
    "soil_nitrogen",
    "soil_organic_carbon",
    "msp_trend",
    "year_normalized",
]

# All features the champion model expects
CHAMPION_FEATURES = CORE_FEATURES + V2_FEATURES + ENGINEERED_FEATURES
