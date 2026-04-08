import pandas as pd
import numpy as np
from scripts.feature_engineer_v2 import (
    DEFAULT_NDVI, DEFAULT_SOIL_PH, DEFAULT_SOIL_NITROGEN, DEFAULT_SOIL_ORGANIC_CARBON
)

def get_crop_columns(available_crops: list) -> list:
    return [f'Item_{c}' for c in available_crops]

def build_prediction_features(crop, year, pesticides, rainfall, temp, crop_columns):
    """
    Build feature dict for model prediction using the v2 feature pipeline.
    All v2 features are constructed here, using defaults for satellite/soil
    when external APIs are unavailable.
    """
    features = {
        'average_rain_fall_mm_per_year': rainfall,
        'avg_temp': temp,
        'pesticides_tonnes': pesticides,
    }

    features['temp_rainfall_interaction'] = temp * rainfall
    features['rainfall_squared'] = rainfall ** 2
    features['temp_squared'] = temp ** 2
    features['pesticide_per_rainfall'] = pesticides / (rainfall + 1)
    features['rainfall_deviation'] = rainfall - 1083

    year_min, year_max = 1990, 2013
    if year_min != year_max:
        features['year_normalized'] = (year - year_min) / (year_max - year_min)
    else:
        features['year_normalized'] = 1.0

    features['heat_stress_degreedays'] = float(max(0.0, temp - 35.0))
    features['drought_intensity'] = float(max(0.0, 1.0 - (rainfall / 500.0)) if rainfall < 500 else 0.0)

    features['ndvi'] = float(DEFAULT_NDVI)
    ndvi_adj = DEFAULT_NDVI + (rainfall / 2000.0) - (features['heat_stress_degreedays'] / 10.0)
    features['ndvi_adjusted'] = float(np.clip(ndvi_adj, 0, 1))
    features['soil_ph'] = DEFAULT_SOIL_PH
    features['soil_nitrogen'] = DEFAULT_SOIL_NITROGEN
    features['soil_organic_carbon'] = DEFAULT_SOIL_ORGANIC_CARBON

    base_year = 1990
    features['msp_trend'] = 1.0 + (year - base_year) * 0.03

    for col in crop_columns:
        features[col] = 0
    sel = f'Item_{crop}'
    if sel in features:
        features[sel] = 1

    return pd.DataFrame([features])

def get_feature_importance(model, crop_columns):
    """Extract feature importance, grouping crop one-hot encodings into 'Crop Type'."""
    importances = model.feature_importances_

    if hasattr(model, 'feature_names_in_'):
        names = model.feature_names_in_
    else:
        names = [
            'average_rain_fall_mm_per_year', 'avg_temp', 'pesticides_tonnes',
            'heat_stress_degreedays', 'drought_intensity',
            'ndvi', 'ndvi_adjusted',
            'soil_ph', 'soil_nitrogen', 'soil_organic_carbon',
            'msp_trend',
            'temp_rainfall_interaction', 'rainfall_deviation',
            'rainfall_squared', 'temp_squared', 'pesticide_per_rainfall',
            'year_normalized'
        ] + crop_columns

    imp_dict = dict(zip(names, importances))

    crop_imp = sum(imp_dict.get(c, 0) for c in crop_columns)

    display = {
        'Crop Type': crop_imp,
        'Rainfall': imp_dict.get('average_rain_fall_mm_per_year', 0),
        'Temperature': imp_dict.get('avg_temp', 0),
        'Pesticides': imp_dict.get('pesticides_tonnes', 0),
        'Year': imp_dict.get('year_normalized', 0),
    }
    total = sum(display.values())
    if total > 0:
        display = {k: v / total for k, v in display.items()}
    return display
