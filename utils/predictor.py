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
    
    # 5. Predict
    prediction = model.predict(input_df)[0]

    # 6. Inverse Transform — with log-space ceiling guard
    if transform == "log1p":
        # Guard: if prediction is > _LOG_SPACE_MAX, the model was likely trained
        # WITHOUT log-transform (stale model / contract mismatch).
        # np.expm1(values > ~710) overflows float64 to +inf silently.
        if prediction > _LOG_SPACE_MAX:
            warnings.warn(
                f"[predictor] Log-space prediction {prediction:.2f} exceeds safe ceiling "
                f"({_LOG_SPACE_MAX}). This strongly indicates the saved model.pkl was "
                f"NOT trained with log1p. The contract says target_transform='log1p' but "
                f"the model appears to output raw kg/ha. "
                f"ACTION REQUIRED: Re-run scripts/run_pipeline.py to retrain the model.",
                RuntimeWarning, stacklevel=2
            )
            # Best-effort: treat the raw value as the actual yield (no expm1)
            # and cap it at the dataset ceiling from run_pipeline.py (100,000 kg/ha)
            prediction = min(float(prediction), 100000.0)
        else:
            prediction = np.expm1(prediction)
    
    # 7. Safety Clipping (Prevent inf/NaN and impossible values)
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=100000.0, neginf=0.0)
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
