"""
Prediction System Diagnostic & Verification Script

This script validates model inference across typical production scenarios and safety edge cases.
It tests log1p transformation handling and checks whether predictions stay within safe bounds.
"""

import json
import warnings
import joblib

from scripts.config import CONTRACT_PATH, MODEL_PATH
from utils.predictor import predict_yield

# Enable runtime warnings so ceiling-guard alerts are explicitly visible during diagnostics.
warnings.filterwarnings("always", category=RuntimeWarning)


def load_test_artifacts():
    """Load model artifact (preferring compressed version) and feature contract."""
    model_file = "models/model_compressed_9.joblib"
    try:
        model = joblib.load(model_file)
    except FileNotFoundError:
        model = joblib.load(MODEL_PATH)

    with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
        contract = json.load(f)

    return model, contract


def run_prediction_tests():
    """Execute prediction test cases and verify output validity."""
    model, contract = load_test_artifacts()

    # Benchmark scenarios representing common crop-state combinations
    test_cases = [
        {"year": 2026, "state": "West Bengal", "crop": "Rice", "season": "Kharif"},
        {"year": 2026, "state": "Punjab", "crop": "Wheat", "season": "Rabi"},
        {"year": 2026, "state": "Uttar Pradesh", "crop": "Sugarcane", "season": "Kharif"},
    ]

    print("\nRunning Model Prediction Tests:")
    print("-" * 60)

    all_passed = True
    for idx, inputs in enumerate(test_cases, start=1):
        prediction = predict_yield(model, contract, inputs)
        scenario = f"{inputs['crop']} in {inputs['state']} ({inputs['season']}, {inputs['year']})"

        print(f"Test {idx}: {scenario}")
        print(f"  -> Actual Prediction Value: {prediction:,.2f} kg/ha")

        # Check if output hit the 100,000 ceiling guard (indicating potential log-scale issue)
        if prediction >= 100000.0:
            print("  -> STATUS: FAILED (Model output hit the 100,000 kg/ha ceiling guard)")
            all_passed = False
        else:
            print("  -> STATUS: PASSED (Valid dynamic prediction)")

    print("-" * 60)
    if all_passed:
        print("Conclusion: The log1p transformation bug is locally resolved if tests passed.")
    else:
        print("Conclusion: One or more predictions hit ceiling guard bounds. Retraining recommended.")


if __name__ == "__main__":
    run_prediction_tests()

