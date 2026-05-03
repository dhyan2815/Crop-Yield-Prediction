import pandas as pd
import numpy as np
from scripts.config import YEAR_MIN, YEAR_MAX

def predict_yield(model, contract, inputs):
    """
    Robust prediction function that aligns input to the training contract.
    Uses metadata to handle inverse transformations dynamically.
    """
    features = contract.get('features', [])
    transform = contract.get('target_transform', None)
    
    # 1. Initialize all contract features to 0
    row = {col: 0 for col in features}
    
    # 2. Add numeric features (with normalization)
    if 'year_normalized' in row:
        year = inputs.get('year', YEAR_MAX)
        row['year_normalized'] = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
    
    # 3. Handle One-Hot Encoding
    state_col = f"state_{inputs.get('state')}"
    crop_col = f"crop_{inputs.get('crop')}"
    season_col = f"season_{inputs.get('season')}"
    
    if state_col in row: row[state_col] = 1
    if crop_col in row: row[crop_col] = 1
    if season_col in row: row[season_col] = 1
        
    # 4. Create DataFrame and enforce order
    input_df = pd.DataFrame([row])
    input_df = input_df[features] 
    
    # 5. Predict and Handle Inverse Transform
    prediction = model.predict(input_df)[0]
    
    if transform == "log1p":
        prediction = np.expm1(prediction)
    
    # 6. Safety Clipping (Prevent inf/NaN and impossible values)
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=100000.0)
    return max(0, float(prediction))

def get_risk_assessment(yield_val, crop, avg_yield=0):
    """Dynamic assessment based on deviation from crop-specific averages."""
    if avg_yield <= 0:
        return "Unknown", "Insufficient benchmark data for this crop."
    
    performance_ratio = (yield_val / avg_yield) * 100
    
    if performance_ratio < 75:
        return "Critical Low", f"Yield is {100-performance_ratio:.1f}% below average. High risk of supply shortage."
    elif performance_ratio < 110:
        return "Stable", f"Yield is aligned with historical benchmarks ({performance_ratio:.1f}% performance)."
    else:
        return "Optimal", f"Yield is {performance_ratio-100:.1f}% above average. Excellent production potential."
