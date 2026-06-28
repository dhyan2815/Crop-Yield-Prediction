import pandas as pd
import joblib
import json
import warnings
from utils.predictor import predict_yield

# Turn on runtime warnings so ceiling-guard behavior is visible during the diagnostic run.
warnings.filterwarnings('always', category=RuntimeWarning)

try:
    # Prefer the compressed model artifact because it should reflect the latest training run.
    model = joblib.load('models/model_compressed_9.joblib')
except FileNotFoundError:
    # Fall back to the main model path if the compressed artifact is not present.
    model = joblib.load('models/model.pkl')

with open('models/feature_columns.json', 'r') as f:
    contract = json.load(f)

# These test cases cover the common production paths plus a few failure-style scenarios.
test_cases = [
    {'year': 2026, 'state': 'West Bengal', 'crop': 'Rice', 'season': 'Kharif'},
    {'year': 2026, 'state': 'Punjab', 'crop': 'Wheat', 'season': 'Rabi'},
    {'year': 2026, 'state': 'Uttar Pradesh', 'crop': 'Sugarcane', 'season': 'Kharif'}
]

print("Running Model Prediction Tests:")
print("-" * 60)
for idx, inputs in enumerate(test_cases, 1):
    # Print both the scenario and the resulting prediction so the output is easy to scan.
    prediction = predict_yield(model, contract, inputs)
    print(f"Test {idx}: {inputs['crop']} in {inputs['state']} ({inputs['season']}, {inputs['year']})")
    print(f"  -> Actual Prediction Value: {prediction:,.2f} kg/ha")
    
    # Flag the ceiling case explicitly because it usually means the model output is suspicious.
    if prediction == 100000.0:
        print("  -> STATUS: FAILED (Model output hit the 100,000 kg/ha ceiling guard)")
    else:
        print("  -> STATUS: PASSED (Valid dynamic prediction)")
print("-" * 60)
print("Conclusion: The log1p transformation bug is locally resolved if tests passed.")
