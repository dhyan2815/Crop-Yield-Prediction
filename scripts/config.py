# scripts/config.py
"""
Shared configuration constants for the Crop Yield Intelligence System (Project 2026).
"""

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "Crop_Yield_Data.csv")
RAINFALL_DATA_PATH = os.path.join(DATA_DIR, "raw", "Daily Rainfall at State Level_Filtered_Data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "CLEANED_Processed_India_Crop_Yield_Data.csv")
FEATURES_DATA_PATH = os.path.join(DATA_DIR, "processed", "Feature_Engineered_Crop_Yield_Data.csv")

# Model paths
LR_MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression_model.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "champion_model_v2.pkl")
V2_FEATURES_PATH = os.path.join(MODEL_DIR, "v2_features.joblib")

# Target variable
TARGET_COLUMN = "yield_kg_ha"  # Standardized to kg/ha

# =============================================================================
# FEATURE SCHEMAS
# =============================================================================

# Core climate and agricultural inputs
CORE_FEATURES = [
    "average_rain_fall_mm_per_year",
    "avg_temp",
    "pesticides_tonnes"
]

# Legacy interaction and polynomial features (v1)
ENGINEERED_FEATURES = [
    "temp_rainfall_interaction",
    "rainfall_deviation",
    "rainfall_squared",
    "temp_squared",
    "pesticide_per_rainfall"
]

# Professional agronomic and satellite features (v2)
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

# Full canonical feature set for the 2026 Champion Model
CHAMPION_FEATURES = CORE_FEATURES + V2_FEATURES + ENGINEERED_FEATURES

# =============================================================================
# DATA SOURCE MAPPINGS (URL transparency per crop)
# =============================================================================

CROP_DATA_SOURCES = {
    "Rice": {
        "UPAg Yield Statistics": "https://api.upag.gov.in/v1/yield",
        "FAOSTAT Rice Data": "https://www.fao.org/faostat/en/#data/QC",
        "Sentinel-2 NDVI": "https://services.sentinel-hub.com/ogc/wms",
        "SoilGrids India": "https://rest.isric.org/soilgrids/v2.0/properties/query",
    },
    "Wheat": {
        "UPAg Yield Statistics": "https://api.upag.gov.in/v1/yield",
        "Agmarknet MSP": "https://api.data.gov.in/resource/9ef273ef-a641-4de2-a243-a04145617300",
        "Open-Meteo Weather": "https://archive-api.open-meteo.com/v1/archive",
    },
    "Maize": {
        "UPAg Yield Statistics": "https://api.upag.gov.in/v1/yield",
        "ICAR Research": "https://icar.org.in/technical-documents",
    },
    "Sugar Cane": {
        "Agmarknet MSP": "https://api.data.gov.in/resource/9ef273ef-a641-4de2-a243-a04145617300",
        "FAOSTAT Sugar Cane Data": "https://www.fao.org/faostat/en/#data/QC",
    },
    "default": {
        "UPAg API Documentation": "https://api.upag.gov.in/docs",
        "FAOSTAT Data Portal": "https://www.fao.org/faostat/en/#data/QC",
        "India Data Portal": "https://data.gov.in",
    },
}
