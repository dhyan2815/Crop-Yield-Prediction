"""
Model Prediction & Risk Assessment Engine

Provides contract-aligned inference and dynamic yield risk benchmarking.
Includes 3 defensive safety layers:
  1. Log-space ceiling guard: Catches stale models trained without log1p.
  2. nan_to_num conversion: Handles NaN/infinity floating-point outputs.
  3. Physical non-negative floor: Clamps negative yields to 0.0 kg/ha.
"""

import warnings
from typing import Any
import numpy as np
import pandas as pd

from scripts.config import YEAR_MAX, YEAR_MIN

# Maximum physically plausible log-space yield: log1p(150,000) ~ 11.92.
# 13.0 is a generous ceiling. Outputs above this indicate missing log-transform during training.
_LOG_SPACE_MAX = 13.0


def predict_yield(model: Any, contract: dict[str, Any], inputs: dict[str, Any]) -> float:
    """Align user inputs to feature contract schema and predict crop yield (kg/ha).

    Args:
        model: Trained estimator (e.g. RandomForestRegressor).
        contract: Feature contract dictionary containing expected feature names and target transform.
        inputs: Dictionary containing state, crop, season, and year parameters.

    Returns:
        Predicted yield value in kg/ha (bounded non-negative float).
    """
    features = contract.get("features", [])
    transform = contract.get("target_transform", None)

    # Initialize feature vector with zeros
    row = {col: 0.0 for col in features}

    # Normalize year slider input
    if "year_normalized" in row:
        year = inputs.get("year", YEAR_MAX)
        row["year_normalized"] = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)

    # Set active one-hot encoded flags for selected categories
    state_col = f"state_{inputs.get('state')}"
    crop_col = f"crop_{inputs.get('crop')}"
    season_col = f"season_{inputs.get('season')}"

    if state_col in row:
        row[state_col] = 1.0
    if crop_col in row:
        row[crop_col] = 1.0
    if season_col in row:
        row[season_col] = 1.0

    # Build single-row DataFrame aligned strictly to training column order
    input_df = pd.DataFrame([row])[features]

    # Predict raw model output
    prediction = float(model.predict(input_df)[0])

    # Reverse log1p transform if target transform is log-scale
    if transform == "log1p":
        if prediction > _LOG_SPACE_MAX:
            warnings.warn(
                f"[predictor] Log-space prediction {prediction:.2f} exceeds ceiling ({_LOG_SPACE_MAX}). "
                f"Model appears to output raw kg/ha instead of log1p. Capping to safe max.",
                RuntimeWarning,
                stacklevel=2,
            )
            prediction = min(prediction, 100000.0)
        else:
            prediction = np.expm1(prediction)

    # Sanitize NaN/Inf outputs and clamp to non-negative physical yield
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=100000.0, neginf=0.0)
    return max(0.0, float(prediction))


def get_risk_assessment(yield_val: float, crop: str, avg_yield: float = 0.0) -> tuple[str, str]:
    """Evaluate predicted yield against historical crop averages to generate risk status and summary text.

    Args:
        yield_val: Forecasted yield in kg/ha.
        crop: Selected crop name.
        avg_yield: Historical national average yield for the crop.

    Returns:
        Tuple of (status_label, assessment_message).
    """
    if avg_yield <= 0:
        return "Unknown", "Insufficient historical benchmark data for this crop."

    performance_ratio = (yield_val / avg_yield) * 100.0

    if performance_ratio < 75.0:
        pct_below = 100.0 - performance_ratio
        return "Critical Low", f"Yield is {pct_below:.1f}% below historical average. High risk of crop stress."
    if performance_ratio < 110.0:
        return "Stable", f"Yield is aligned with historical benchmarks ({performance_ratio:.1f}% performance index)."

    pct_above = performance_ratio - 100.0
    return "Optimal", f"Yield is {pct_above:.1f}% above historical average. Excellent production conditions expected."
