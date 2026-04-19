import pandas as pd
import numpy as np
from scripts.config import YEAR_MIN, YEAR_MAX

def predict_yield(model, contract, inputs):
    """
    Robust prediction function that aligns input to the training contract.
    Inputs: dict with {state, crop, year, season, ...}
    """
    # 1. Initialize all contract features to 0
    row = {col: 0 for col in contract}
    
    # 2. Add numeric features (with normalization)
    if 'year_normalized' in row:
        year = inputs.get('year', YEAR_MAX)
        row['year_normalized'] = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
    
    # 3. Handle One-Hot Encoding (Setting to 1 for chosen options)
    # The columns are named: state_NAME, crop_NAME, season_NAME
    state_col = f"state_{inputs.get('state')}"
    crop_col = f"crop_{inputs.get('crop')}"
    season_col = f"season_{inputs.get('season')}"
    
    if state_col in row:
        row[state_col] = 1
    if crop_col in row:
        row[crop_col] = 1
    if season_col in row:
        row[season_col] = 1
        
    # 4. Create DataFrame and enforce order (DANGEROUS if skipped)
    input_df = pd.DataFrame([row])
    input_df = input_df[contract] # Strict alignment to contract
    
    # 5. Predict
    prediction = model.predict(input_df)[0]
    return max(0, float(prediction)) # No negative yields

def get_risk_assessment(yield_val, crop):
    """Simple logic to assess if predicted yield is good/bad for that crop."""
    # This is a placeholder for more advanced agronomic logic
    if yield_val < 500:
        return "Critical Low", "Severe risk of crop failure. Consider drought-resistant varieties."
    elif yield_val < 2000:
        return "Moderate", "Average yield expected. Ensure regular irrigation."
    else:
        return "Healthy", "High yield trajectory. Optimal conditions predicted."
