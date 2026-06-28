import pandas as pd
import numpy as np
import warnings
from scripts.config import YEAR_MIN, YEAR_MAX

# Maximum physically plausible yield in log-space:
# log1p(150000) ≈ 11.92. We use 13.0 as a generous ceiling.
# Any raw model output above this means the model was NOT trained with log-transform.
_LOG_SPACE_MAX = 13.0

def predict_yield(model, contract, inputs):
    """
    Robust prediction function that aligns input to the training contract.
    Uses metadata to handle inverse transformations dynamically.

    Safety layers (in order):
      1. Log-space ceiling guard — catches stale models not trained with log1p.
      2. nan_to_num — converts any remaining NaN/inf to bounded floats.
      3. max(0, ...) — ensures physically non-negative yield output.
    """
    # Read the expected feature order and output transform directly from the contract.
    features = contract.get('features', [])
    transform = contract.get('target_transform', None)
    
    # Initialize every expected column to zero before filling the selected scenario values.
    row = {col: 0 for col in features}
    
    # Normalize the year so inference uses the same scale as training.
    if 'year_normalized' in row:
        year = inputs.get('year', YEAR_MAX)
        row['year_normalized'] = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)
    
    # Activate the one-hot encoded state, crop, and season columns that match the selected inputs.
    state_col = f"state_{inputs.get('state')}"
    crop_col = f"crop_{inputs.get('crop')}"
    season_col = f"season_{inputs.get('season')}"
    
    if state_col in row: row[state_col] = 1
    if crop_col in row: row[crop_col] = 1
    if season_col in row: row[season_col] = 1
        
    # Build a single-row DataFrame and enforce the exact contract column order.
    input_df = pd.DataFrame([row])
    input_df = input_df[features] 
    
    # Run the model on the contract-aligned input row.
    prediction = model.predict(input_df)[0]

    # Reverse the log transform only when the contract says the model was trained in log space.
    if transform == "log1p":
        # Guard against stale models that output raw yield values despite a log1p contract.
        if prediction > _LOG_SPACE_MAX:
            warnings.warn(
                f"[predictor] Log-space prediction {prediction:.2f} exceeds safe ceiling "
                f"({_LOG_SPACE_MAX}). This strongly indicates the saved model.pkl was "
                f"NOT trained with log1p. The contract says target_transform='log1p' but "
                f"the model appears to output raw kg/ha. "
                f"ACTION REQUIRED: Re-run scripts/run_pipeline.py to retrain the model.",
                RuntimeWarning, stacklevel=2
            )
            # Treat the raw value as already-decoded yield and cap it to a conservative ceiling.
            prediction = min(float(prediction), 100000.0)
        else:
            prediction = np.expm1(prediction)
    
    # Clamp invalid floating-point outputs before returning the final physical yield.
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=100000.0, neginf=0.0)
    return max(0, float(prediction))

def get_risk_assessment(yield_val, crop, avg_yield=0):
    """Dynamic assessment based on deviation from crop-specific averages."""
    if avg_yield <= 0:
        return "Unknown", "Insufficient benchmark data for this crop."
    
    # Express the forecast as a percentage of the historical crop average.
    performance_ratio = (yield_val / avg_yield) * 100
    
    if performance_ratio < 75:
        return "Critical Low", f"Yield is {100-performance_ratio:.1f}% below average. High risk of supply shortage."
    elif performance_ratio < 110:
        return "Stable", f"Yield is aligned with historical benchmarks ({performance_ratio:.1f}% performance)."
    else:
        return "Optimal", f"Yield is {performance_ratio-100:.1f}% above average. Excellent production potential."
